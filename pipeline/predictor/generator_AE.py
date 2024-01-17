import time
from typing import Iterable
from PIL import Image  

import torch
import torch.nn as nn

import numpy as np

from ..logger import LOGGER
from ..utils import move_to_device, load_model, save_im, tresh_im, disp_sar
from ..datasets.preprocessing import denormalization

import os

save_tresh = None # set variable to None for automatic tresholding

class GeneratorAE:
    def __init__(
            self,
            decoder: nn.Module,
            discriminator: nn.Module,
            device: str,
            model_save_path: str,
            norm,
            save_im_path:str,
            pred_nb:int) -> None:
        self.decoder = decoder.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.model_save_path = model_save_path
        self.norm = norm
        self.save_im_path = save_im_path
        self.pred_nb = pred_nb

    def predict_step(self,denorm,step_id: int):
        noise = torch.randn(self.discriminator.lin1.in_features,device=self.device)
        result = self.decoder(noise)
        proba = self.discriminator(noise)
        result_np = torch.permute(torch.squeeze(result),(1,2,0))
        result_np = result_np.cpu().data.numpy()
        result_denorm = denorm(result_np)
        save_im(result_denorm,'{}/gen_{}.png'.format(self.save_im_path,step_id),tresh=save_tresh)
        np.save('{}/gen_{}.npy'.format(self.save_im_path,step_id),result_denorm)
        return proba.cpu().data.numpy()

    def log_predict_step(self, step_id: int, predict_time: float, proba: float, mean_proba: float):
        LOGGER.info("[{} s] Generation step {}. Discriminator proba {}. Mean proba {}".format(predict_time, step_id,proba,mean_proba))
        return True
    def log_predict_completed(self, predict_time: float, mean_proba:float):
        LOGGER.info("[{} s] Generation is completed. Mean proba {}".format(predict_time,mean_proba))
        return True

    """ Load latest model in folder self.model_save_path """
    def load_last_model(self):
        if os.path.exists(self.model_save_path):
            index = ["discriminator","decoder"]
            nb_model = len(index)
            count = 0
            for model in index:
                epochs = filter(lambda file: file.startswith("{}_epoch_".format(model)), os.listdir(self.model_save_path))
                epochs = map(lambda file: int(file[file.find("h_")+2:]), epochs)
                epochs = list(epochs)
                if epochs:
                    count += 1
                    last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,max(epochs)))
                    if model == "discriminator":
                        load_model(self.discriminator, last_model_path)
                        LOGGER.info("{} found at epoch {}...".format(model,max(epochs)))
                    elif model == "decoder":
                        load_model(self.decoder, last_model_path)
                        LOGGER.info("{} found at epoch {}...".format(model,max(epochs)))
                    if count == nb_model:
                        return
        LOGGER.info(" {} out of {} model(s) not found in {}...".format(nb_model-count,nb_model,self.model_save_path))

    def run(self):
        self.load_last_model()
        self.decoder.eval()
        self.discriminator.eval()
        denorm = denormalization(self.norm[0],self.norm[1])
        step_count = 0
        start_time = time.time()
        step_count = 0
        mean_proba = 0
        with torch.no_grad():
            for _ in range(self.pred_nb):
                proba= self.predict_step(denorm,step_count)
                predict_time = time.time() - start_time
                mean_proba += proba
                step_count += 1
                self.log_predict_step(step_count, predict_time,proba,mean_proba/step_count)
        mean_proba /= max(step_count,1)
        predict_time = time.time() - start_time
        self.log_predict_completed(predict_time,mean_proba)
        return predict_time
