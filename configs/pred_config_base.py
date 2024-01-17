import torch
from pipeline.metrics.base import MetricsCalculatorEmpty


class PredictConfigBase:
    def __init__(
            self,
            model_save_path,
            dataset,
            predictor,
            norm,
            save_im_path = './data/results',
            device=None,
            batch_size=1,
            num_workers=0,
            print_frequency=1):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.norm = norm
        self.dataset = dataset
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_frequency = print_frequency
        self.predictor = predictor
        self.device = device
        self.save_im_path = save_im_path


class PredictConfigBaseAE:
    def __init__(
            self,
            model_save_path,
            encoder,
            decoder,
            dataset,
            predictor,
            norm,
            save_im_path = './data/results',
            device=None,
            batch_size=1,
            num_workers=0,
            print_frequency=1,
            metrics_calculator = None):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if metrics_calculator is None:
            metrics_calculator = MetricsCalculatorEmpty()

        self.norm = norm
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_frequency = print_frequency
        self.predictor = predictor
        self.device = device
        self.save_im_path = save_im_path
        self.metrics_calculator = metrics_calculator

class GenConfigBaseAE:
    def __init__(
            self,
            model_save_path,
            decoder,
            discriminator,
            generator,
            norm,
            save_im_path = './data/results',
            device=None,
            num_workers=0,
            pred_nb=10):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.norm = norm
        self.decoder = decoder
        self.discriminator = discriminator
        self.model_save_path = model_save_path
        self.num_workers = num_workers
        self.generator = generator
        self.device = device
        self.save_im_path = save_im_path
        self.pred_nb = pred_nb


        