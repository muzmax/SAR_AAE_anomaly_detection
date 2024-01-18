import time
from typing import Iterable

import torch
import torch.nn as nn
from sklearn import metrics

import numpy as np
import pickle

from ..logger import LOGGER
from ..utils import move_to_device, load_model, plot_hist, save_im, save_plot, tresh_im, normalize01, save_curve
from ..datasets.preprocessing import denormalization
from ..metrics.base import MetricsCalculatorBase

import os

save_tresh = None
save_ROC = True

class PredictorAE:
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            data_loader: Iterable,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            norm,
            save_im_path:str,
            ano_score,
            metrics_calculator:MetricsCalculatorBase,
            patch_size : int,
            stride : int) -> None:

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.data_loader = data_loader
        self.print_frequency = print_frequency
        self.device = device
        self.model_save_path = model_save_path
        self.norm = norm
        self.save_im_path = save_im_path
        self.ano_score = ano_score
        self.metrics_calculator = metrics_calculator
        self.patch_size = patch_size
        self.stride = stride
        
    def predict_step(self, input_data: torch.Tensor,label: torch.Tensor,name: str, step_id,denorm):
        
        patch_size = self.patch_size
        stride = self.stride 
        # Save input image to png and npy format   
        input_np = torch.permute(torch.squeeze(input_data),(1,2,0))
        input_np = input_np.cpu().data.numpy()
        input_np_denorm = denorm(input_np)
        save_im(input_np_denorm,'{}/{}_input.png'.format(self.save_im_path,name[0]),tresh=save_tresh)
        np.save('{}/{}_input.npy'.format(self.save_im_path,name[0]),input_np_denorm)

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
                clean = self.decoder(self.encoder(input_data[:,:,x:x+patch_size,y:y+patch_size]))
                result[:,x:x+patch_size,y:y+patch_size] += torch.squeeze(clean)
                count[x:x+patch_size,y:y+patch_size] += torch.ones(patch_size,patch_size,device=self.device)
        
        ############################
        # Data Analysis
        ############################

        # Reconstruction
        result = torch.div(result,count)
        result_cpu = result.cpu()
        result_np = torch.permute(result_cpu,(1,2,0))
        result_np = result_np.data.numpy()
        result_denorm = denorm(result_np)
        save_im(result_denorm,'{}/{}_reconstruction.png'.format(self.save_im_path,name[0]),tresh=save_tresh)
        np.save('{}/{}_reconstruction.npy'.format(self.save_im_path,name[0]),result_denorm)

        # Anomaly score (change detection with covariance)
        maps = self.ano_score(result_cpu,torch.reshape(input_data.cpu(),(c,h,w)))
        for key in maps:
            
            # Save parameters
            if key == 'param' :
                with open('{}/{}_{}'.format(self.save_im_path,name[0],key), "wb") as fout:
                    pickle.dump(maps[key], fout)
            
            # Save ano map
            else :
                if key in ['L1','Euclidean','mix']:
                    ano = maps[key]
                    np.save('{}/{}_{}.npy'.format(self.save_im_path,name[0],key),ano)
                    save_plot(normalize01(np.clip(ano,None,np.mean(ano)+3*np.std(ano))),'{}/{}_{}_thresh.png'.format(self.save_im_path,name[0],key))
        label_np = label.cpu().data.numpy()
        ano = maps['Euclidean']
        self.metrics_calculator.add(ano,label_np)
        if save_ROC:
            if torch.numel(label) == 0:
                print("save_ROC is true but no label is available ...")
            else:
                fpr, tpr, tresh = metrics.roc_curve(label_np.flatten(),ano.flatten())
                auc = metrics.auc(fpr, tpr)
                curve_param = np.concatenate((fpr[:,np.newaxis],tpr[:,np.newaxis]),axis = 1)
                np.save('{}/{}_ROC_param.npy'.format(self.save_im_path,name[0]),curve_param)
                save_curve(fpr,tpr,'{}/{}_ROC.png'.format(self.save_im_path,name[0]),'ROC - auc = {:.3}'.format(auc))
      

    def log_predict_step(self, step_id: int, predict_time: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Predict step {}".format(predict_time, step_id))
            return True

        return False

    def log_predict_completed(self, predict_time: float,metric:dict):
        LOGGER.info("[{:.6} s] Predict is completed. Metric : {} ".format(predict_time,metric))
        return True

    """ Load latest model in folder self.model_save_path """
    def load_last_model(self):
        if os.path.exists(self.model_save_path):
            index = ["encoder","decoder"]
            nb_model = len(index)
            count = 0
            for model in index:
                epochs = filter(lambda file: file.startswith("{}_epoch_".format(model)), os.listdir(self.model_save_path))
                epochs = map(lambda file: int(file[file.find("h_")+2:]), epochs)
                epochs = list(epochs)

                if epochs:
                    count += 1
                    last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,max(epochs)))
                    if model == "encoder":
                        load_model(self.encoder, last_model_path)
                        LOGGER.info("{} found at epoch {}...".format(model,max(epochs)))
                        
                    elif model == "decoder":
                        load_model(self.decoder, last_model_path)
                        LOGGER.info("{} found at epoch {}...".format(model,max(epochs)))

                    if count == nb_model:
                        return

        LOGGER.info(" {} out of {} model(s) not found in {}...".format(nb_model-count,nb_model,self.model_save_path))

    def run(self):
        self.load_last_model()
        self.encoder.eval()
        self.decoder.eval()

        denorm = denormalization(self.norm[0],self.norm[1])
        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data,label,name) in enumerate(self.data_loader):
                self.predict_step(input_data,label,name,step_id,denorm)

                step_count += 1
                predict_time = time.time() - start_time
                self.log_predict_step(step_id, predict_time)

        metric = self.metrics_calculator.calculate()
        predict_time = time.time() - start_time
        self.log_predict_completed(predict_time,metric)
        return predict_time
