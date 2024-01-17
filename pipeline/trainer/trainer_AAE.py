from cv2 import norm
from pipeline.datasets.preprocessing import denormalization
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from ..core import PipelineError
from ..logger import LOGGER
from ..storage.state import StateStorageBase
from ..scheduler.base import SchedulerBase
from ..metrics.base import MetricsCalculatorBase

from ..utils import move_to_device, save_model, load_model, save_im,tresh_im


import os

# save_tresh = [4.585E-8,1.651E-8,6.102E-8] # set variable to None for automatic tresholding
# save_tresh = [0.21,0.10,0.21]
# save_tresh = [0.21,0.10,0.21] # 2 channels
save_tresh = [800,500,800]

track_loss = True
debug = False # display grad norm
w_clip = False # Gradient clipping for reconstruction loss
w_value = 0.8

class TrainerAAE:
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            discriminator: nn.Module,
            train_data_loader: Iterable,
            eval_data_loader: Iterable,
            epoch_count: int,
            opt_enc: Optimizer,
            opt_dec: Optimizer,
            opt_gen: Optimizer,
            opt_disc: Optimizer,
            scheduler: SchedulerBase,
            rec_loss: nn.Module,
            disc_loss: nn.Module,
            gen_loss: nn.Module,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            state_storage: StateStorageBase,
            loss_lr_storage: StateStorageBase,
            norm,
            metrics_calculator: MetricsCalculatorBase ) -> None:

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.discriminator = discriminator.to(device)
        self.opt_enc = opt_enc
        self.opt_dec = opt_dec
        self.opt_gen = opt_gen
        self.opt_disc = opt_disc
        
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.norm = norm

        self.rec_loss = rec_loss
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss
        self.metrics_calculator = metrics_calculator

        self.epoch_count = epoch_count
        self.scheduler = scheduler
        self.print_frequency = print_frequency
        
        self.model_save_path = model_save_path
        self.state_storage = state_storage
        self.loss_lr_storage = loss_lr_storage

        self.device = device
        
        if debug:
            self.grad_enc = []
            self.grad_dec = []
            self.grad_disc_gen = []
            self.grad_enc_max = 0
            self.grad_dec_max = 0
            self.grad_disc_gen_max = 0

        if track_loss:
            self.l_gen = []
            self.l_disc = []
            self.l_rec = []
            self.l_lr = []

        

    def train_step(self, input_data: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)

        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        # Reconstruction loss
        z_output = self.encoder(input_data)
        model_output = self.decoder(z_output)
        rec_loss = self.rec_loss(model_output, input_data)
        rec_loss.backward()

        if w_clip:
                nn.utils.clip_grad_value_(self.encoder.parameters(), clip_value=1.0)
                nn.utils.clip_grad_value_(self.decoder.parameters(), clip_value=1.0)

        if debug :
            for p in list(filter(lambda p: p.grad is not None, self.encoder.parameters())):
                    grads = p.grad.detach()
                    norm_grad = grads.data.norm().cpu().numpy()
                    max_norm_grad = torch.max(torch.abs(grads)).cpu().item()
                    self.grad_enc_max = max(self.grad_enc_max,max_norm_grad)
                    self.grad_enc.append(norm_grad)

            for p in list(filter(lambda p: p.grad is not None, self.decoder.parameters())):
                    grads = p.grad.detach()
                    norm_grad = grads.data.norm().cpu().numpy()
                    max_norm_grad = torch.max(torch.abs(grads)).cpu().item()
                    self.grad_dec_max = max(self.grad_dec_max,max_norm_grad)
                    self.grad_dec.append(norm_grad)

        self.opt_enc.step(closure=None)
        self.opt_dec.step(closure=None)

        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        # Discriminator loss
        self.encoder.eval()
        z_output = self.encoder(input_data)
        z_dp = torch.randn(z_output.size(),device=self.device,requires_grad=True) # Normal
        # z_dp = torch.rand(z_output.size(),device=self.device,requires_grad=True) # Uniform

        disc_real = self.discriminator(z_dp)
        disc_fake = self.discriminator(z_output)
        disc_loss = self.disc_loss(disc_real,disc_fake)
        disc_loss.backward()
        self.opt_disc.step()

        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        # Generator loss
        self.encoder.train()
        z_output = self.encoder(input_data)

        disc_fake = self.discriminator(z_output)
        gen_loss = self.gen_loss(disc_fake)
        gen_loss.backward()

        if debug :
            for p in list(filter(lambda p: p.grad is not None, self.encoder.parameters())):
                    grads = p.grad.detach()
                    norm_grad = grads.data.norm().cpu().numpy()
                    max_norm_grad = torch.max(torch.abs(grads)).cpu().item()
                    self.grad_disc_gen_max = max(self.grad_disc_gen_max,max_norm_grad)
                    self.grad_disc_gen.append(norm_grad)

        self.opt_gen.step()

        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        return rec_loss.cpu().data.numpy(), disc_loss.cpu().data.numpy(), gen_loss.cpu().data.numpy()

    def predict_step(self ,input_data: torch.Tensor, label: torch.Tensor ,name: str ,epoch_id, step_id ,denorm):
        patch_size = 32
        stride = 16

        input_np = torch.permute(torch.squeeze(input_data),(1,2,0)) # for residual
        input_np = input_np.cpu().data.numpy()

        input_data = move_to_device(input_data, device=self.device)
        (un,c,h,w) = input_data.shape
        result = torch.zeros(c,h,w,device=self.device)
        count = torch.zeros(h,w,device=self.device)
        
        if h == patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0,h-patch_size,stride))
            if (x_range[-1]+patch_size)<h : x_range.extend(range(h-patch_size,h-patch_size+1))
        
        if w == patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0,w-patch_size,stride))
            if (y_range[-1]+patch_size)<w : y_range.extend(range(w-patch_size,w-patch_size+1))
            
        for x in x_range:
            for y in y_range:
                patch = input_data[:,:,x:x+patch_size,y:y+patch_size]
                clean = self.decoder(self.encoder(patch))
                result[:,x:x+patch_size,y:y+patch_size] += torch.squeeze(clean)
                count[x:x+patch_size,y:y+patch_size] += torch.ones(patch_size,patch_size,device=self.device)

        result = torch.div(result,count)
        loss = self.rec_loss(result,torch.squeeze(input_data))
        result = torch.permute(result,(1,2,0))
        result_np = result.cpu().data.numpy()
        result_denorm = denorm(result_np)

        
        save_im(result_denorm,'./data/sample/reconstruction_{}_epoch_{}.png'.format(name[0],epoch_id),tresh=save_tresh)
        np.save('./data/sample/reconstruction_{}_epoch_{}.npy'.format(name[0],epoch_id),result_denorm)

        
        norm_L1_01 = np.abs(input_np-result_np)
        norm_L1 = norm_L1_01*2*(self.norm[1]-self.norm[0])
        save_im(np.sum(norm_L1,axis=2),'./data/sample/residual_{}_epoch_{}.png'.format(name[0],epoch_id))

        return loss.cpu().data.numpy()

        


    def log_train_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float, disc_loss: float, gen_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{:.6} s] Epoch {}. Train step {}. Reconstruction loss {:.6}. Discriminator loss {:.6}. Generator loss{:.6} Mean loss {:.6}".format(
                epoch_time, epoch_id, step_id, loss, disc_loss, gen_loss, mean_loss))
            if debug:
                print("Average norm enc : {}\nAverage norm dec : {}\nAverage norm disc enc : {}".format(np.mean(self.grad_enc),np.mean(self.grad_dec),np.mean(self.grad_disc_gen)))
                print("max norm enc : {}\nmax norm dec : {}\nmax norm disc enc : {}".format(np.mean(self.grad_enc_max),np.mean(self.grad_dec_max),np.mean(self.grad_disc_gen_max)))
            return True

        return False

    def log_evaluation_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{:.6} s] Epoch {}. Evaluation step {}. Loss {:.6}. Mean loss {:.6}".format(
                epoch_time, epoch_id, step_id, loss, mean_loss))

            return True

        return False

    def log_train_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float):
        LOGGER.info("Training Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}".format(
            epoch_id, epoch_time, mean_loss))
        return True

    def log_evaluation_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float, metrics: dict):
        LOGGER.info("Evaluation Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}. Metrics: {:.6}".format(
            epoch_id, epoch_time, mean_loss, str(metrics)))
        return True

    def run_train_epoch(self, epoch_id: int):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        start_time = time.time()
        mean_loss = 0
        step_count = 0

        for step_id, input_data in enumerate(self.train_data_loader):
            rec_loss, disc_loss, gen_loss = self.train_step(input_data)
            epoch_time = time.time() - start_time

            mean_loss += rec_loss+disc_loss+gen_loss
            step_count += 1

            self.log_train_step(epoch_id, step_id, epoch_time, rec_loss, mean_loss / step_count, disc_loss, gen_loss)

            if track_loss:
                self.l_gen.append(gen_loss)
                self.l_disc.append(disc_loss)
                self.l_rec.append(rec_loss)
                self.l_lr.append(self.opt_enc.param_groups[0]["lr"])

            self.scheduler.step() # batch scheduler

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)

        self.log_train_epoch(epoch_id, epoch_time, mean_loss)

        return epoch_time, mean_loss

    def run_evaluation_epoch(self, epoch_id: int,denorm):
        self.encoder.eval()
        self.decoder.eval()

        self.metrics_calculator.zero_cache()
        mean_loss = 0
        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data,label,name) in enumerate(self.eval_data_loader):
                loss = self.predict_step(input_data,label,name,epoch_id,step_id,denorm)
             
                step_count += 1
                mean_loss += loss
                epoch_time = time.time() - start_time

                self.log_evaluation_step(epoch_id, step_id, epoch_time, loss, mean_loss/step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        metrics = self.metrics_calculator.calculate()
        
        self.log_evaluation_epoch(epoch_id, epoch_time, mean_loss, metrics)

        return epoch_time, mean_loss, metrics

    
    def load_optimizer_state(self):
        if not (self.state_storage.has_key("lr_enc") and self.state_storage.has_key("lr_dec") and self.state_storage.has_key("lr_gen") and self.state_storage.has_key("lr_disc") and self.state_storage.has_key("scheduler")):
            print("One or more optimizer state not found...")
            return

        self.opt_enc.load_state_dict(self.state_storage.get_value("lr_enc"))
        self.opt_dec.load_state_dict(self.state_storage.get_value("lr_dec"))
        self.opt_disc.load_state_dict(self.state_storage.get_value("lr_disc"))
        self.opt_gen.load_state_dict(self.state_storage.get_value("lr_gen"))

        self.scheduler.load(self.state_storage.get_value("scheduler"))
        
    def save_optimizer_state(self):
        
        self.state_storage.set_value("lr_enc", self.opt_enc.state_dict())
        self.state_storage.set_value("lr_dec", self.opt_dec.state_dict())
        self.state_storage.set_value("lr_gen", self.opt_gen.state_dict())
        self.state_storage.set_value("lr_disc", self.opt_disc.state_dict())
        
        scheduler_param = self.scheduler.get_param()
        self.state_storage.set_value("scheduler",scheduler_param)
        

    def save_last_model(self, epoch_id):
        os.makedirs(self.model_save_path, exist_ok=True)
        index = ["encoder","decoder","discriminator"]
        for model in index :
            model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,epoch_id))
            
            if model == "encoder":
                save_model(self.encoder, model_path)
                LOGGER.info("{} was saved in {}".format(model,model_path))
            elif model == "decoder":
                save_model(self.decoder, model_path)
                LOGGER.info("{} was saved in {}".format(model,model_path))
            else:
                save_model(self.discriminator, model_path)
                LOGGER.info("{} was saved in {}".format(model,model_path))

    def load_last_model(self, epoch_id):
        index = ["encoder","decoder","discriminator"]
        for model in index :
            last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,epoch_id))
            if model == "encoder":
                load_model(self.encoder, last_model_path)
            elif model == "decoder":
                load_model(self.decoder, last_model_path)
            else:
                load_model(self.discriminator, last_model_path)

    def load_loss_lr(self):
        if not (self.loss_lr_storage.has_key('gen') and self.loss_lr_storage.has_key('rec') and self.loss_lr_storage.has_key('disc') and self.loss_lr_storage.has_key('lr')):
            print('One or more loss/lr tracker not found ...')
            return

        self.l_gen = self.loss_lr_storage.get_value('gen')
        self.l_rec = self.loss_lr_storage.get_value('rec')
        self.l_disc = self.loss_lr_storage.get_value('disc')
        self.l_lr = self.loss_lr_storage.get_value('lr')
    
    def save_loss_lr(self):
        self.loss_lr_storage.set_value('gen',self.l_gen) 
        self.loss_lr_storage.set_value('rec',self.l_rec) 
        self.loss_lr_storage.set_value('disc',self.l_disc) 
        self.loss_lr_storage.set_value('lr',self.l_lr) 
            
    def run(self):
        
        denorm = denormalization(self.norm[0],self.norm[1])
        os.makedirs('./data/sample', exist_ok=True)
        
        start_epoch_id = 0

        if self.state_storage.has_key("start_epoch_id"):
            last_saved_epoch = self.state_storage.get_value("start_epoch_id")-1
            start_epoch_id = last_saved_epoch+1

            try:
                self.load_last_model(last_saved_epoch)
                LOGGER.info("Last saved weights epoch {}, starting training epoch {}".format(last_saved_epoch,start_epoch_id))
                            
            except:
                LOGGER.exception("Exception occurs during loading a model. Starting to train a model from scratch...")
                
        else:
            LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))

        self.load_optimizer_state()

        if track_loss:
            self.load_loss_lr()

        epoch_id = start_epoch_id
        while self.epoch_count is None or epoch_id < self.epoch_count:
            
            _, mean_train_loss = self.run_train_epoch(epoch_id)

            _, mean_loss, metrics = self.run_evaluation_epoch(epoch_id,denorm)
            
            # self.scheduler.step(mean_train_loss,metrics,epoch_id) # scheduler per epoch

            self.state_storage.set_value("start_epoch_id", epoch_id + 1)
            self.save_optimizer_state()
            self.save_last_model(epoch_id)

            epoch_id += 1

        if track_loss:
            self.save_loss_lr()
















# def load_optimizer_state(self):
    #     if not (self.state_storage.has_key("lr_enc") and self.state_storage.has_key("lr_dec") and self.state_storage.has_key("lr_gen") and self.state_storage.has_key("lr_disc")):
    #         print("One or more optimizer state not found...")
    #         return

    #     lr_enc = self.state_storage.get_value("lr_enc")
    #     for learning_rate, param_group in zip(lr_enc, self.opt_enc.param_groups):
    #         param_group["lr"] = learning_rate

    #     lr_dec = self.state_storage.get_value("lr_dec")
    #     for learning_rate, param_group in zip(lr_dec, self.opt_dec.param_groups):
    #         param_group["lr"] = learning_rate

    #     lr_disc = self.state_storage.get_value("lr_disc")
    #     for learning_rate, param_group in zip(lr_disc, self.opt_disc.param_groups):
    #         param_group["lr"] = learning_rate

    #     lr_gen = self.state_storage.get_value("lr_gen")
    #     for learning_rate, param_group in zip(lr_gen, self.opt_gen.param_groups):
    #         param_group["lr"] = learning_rate


    # def save_optimizer_state(self):

    #     learning_rates = []
    #     for param_group in self.opt_enc.param_groups:
    #         learning_rates.append(float(param_group['lr']))
    #     self.state_storage.set_value("lr_enc", learning_rates)

    #     learning_rates = []
    #     for param_group in self.opt_dec.param_groups:
    #         learning_rates.append(float(param_group['lr']))
    #     self.state_storage.set_value("lr_dec", learning_rates)

    #     learning_rates = []
    #     for param_group in self.opt_disc.param_groups:
    #         learning_rates.append(float(param_group['lr']))
    #     self.state_storage.set_value("lr_disc", learning_rates)

    #     learning_rates = []
    #     for param_group in self.opt_gen.param_groups:
    #         learning_rates.append(float(param_group['lr']))
    #     self.state_storage.set_value("lr_gen", learning_rates)
    