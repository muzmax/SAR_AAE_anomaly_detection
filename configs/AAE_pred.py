import os
from configs.pred_config_base import PredictConfigBaseAE

from pipeline.datasets.datasets import test_data
from pipeline.datasets.load import load_eval_data
from pipeline.datasets.preprocessing import *

from pipeline.metrics.auroc import MetricsCalculatorAuroc

from torchvision import transforms

from pipeline.models.AAE import ConvEncoder,ConvDecoder
from pipeline.predictor.predictor_AE import PredictorAE

from pipeline.loss.score import detection_scores

class Config_pred_AAE(PredictConfigBaseAE):
    def __init__(self,params):
        norm = params['norm']
        eval = load_eval_data(params['eval_dir'],label=params['label'])
        process_eval = transforms.Compose([normalization(norm[0],norm[1]),ToTensor()])
        eval_dataset = test_data(eval,
                                 process_eval,
                                 label=params['label'],
                                 nb_ch=params['channels'])   
        encoder = ConvEncoder(im_ch=params['channels'],
                              nz=params['z_size'],
                              patch_size=params['pat_size'])
        decoder = ConvDecoder(im_ch=params['channels'],
                              nz=params['z_size'],
                              patch_size=params['pat_size'])
        predictor = PredictorAE
        if params['label']:
            metrics_calculator = MetricsCalculatorAuroc()
        else :
            metrics_calculator = None

        self.ano_score = detection_scores(params['semi_kernel_sz']) 
        self.patch_size = params['pat_size']
        self.stride = params['stride']
        super().__init__(model_save_path = params['weights_dir'],
                         encoder = encoder,
                         decoder = decoder,
                         dataset=eval_dataset,
                         predictor = predictor,
                         norm = norm,
                         save_im_path = params['save_dir'],
                         device=params['device'],
                         batch_size=params['batch_size'],
                         num_workers=0,
                         print_frequency=params['print_freq'],
                         metrics_calculator = metrics_calculator,
                         )


        