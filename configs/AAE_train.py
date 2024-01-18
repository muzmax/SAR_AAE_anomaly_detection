import os

from configs.train_config_base import ConfigMultiNN

from pipeline.datasets.load import generate_patches, load_eval_data
from pipeline.datasets.datasets import train,test_data
from pipeline.datasets.preprocessing import *

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from pipeline.models.AAE import ConvEncoder,ConvDecoder,Discriminator

from pipeline.loss.l1_loss import l1_loss
from pipeline.loss.adversarial_loss import disc_loss, gen_loss
from pipeline.loss.log_likelihood import log_likelihood_pol
from pipeline.trainer.trainer_AAE import TrainerAAE

from pipeline.scheduler.ae import Scheduler_AE
from pipeline.storage.state import StateStorageFile

class Config_AAE(ConfigMultiNN):
    def __init__(self,params):
        process = transforms.Compose([normalization(params['norm'][0],params['norm'][1]),ToTensor()])
        patches = generate_patches(params['train_dir'],
                                   pat_size=params['pat_size'],
                                   stride=params['stride'],
                                   bat_size=params['batch_size'],
                                   nb_ch=params['channels'])
        train_patches = train(patches,process)
        eval = load_eval_data(params['eval_dir'])
        eval_dataset = test_data(eval,process,nb_ch=params['channels'])

        state_storage = StateStorageFile(os.path.join(params['weights_dir'], "state"))
        loss_lr_storage = StateStorageFile(os.path.join(params['weights_dir'], "loss_lr"))
        trainer = TrainerAAE

        os.makedirs(params['save_dir'],exist_ok=True)
        
        # Adding variables
        # ==================================================
        self.save_path = params['save_dir']
        self.norm = params['norm']

        self.encoder = ConvEncoder(im_ch=params['channels'],
                                   nz=params['z_size'],
                                   patch_size=params['pat_size'])
        self.INIT_LR_ENC_ = 1e-3
        self.INIT_LR_GEN_ = 1e-3
        self.opt_enc = Adam(self.encoder.parameters(), lr=self.INIT_LR_ENC_)
        self.opt_gen = Adam(self.encoder.parameters(), lr=self.INIT_LR_GEN_)
    
        self.decoder = ConvDecoder(im_ch=params['channels'],
                                   nz=params['z_size'],
                                   patch_size=params['pat_size'])
        self.INIT_LR_DEC_ = 1e-3
        self.opt_dec = Adam(self.decoder.parameters(), lr=self.INIT_LR_DEC_)

        self.discriminator = Discriminator(nz=params['z_size'])
        self.INIT_LR_DISC_ = 1e-3
        self.opt_disc = Adam(self.discriminator.parameters(), lr=self.INIT_LR_DISC_)

        # self.rec_loss = log_likelihood_pol(norm,device)
        # self.rec_loss = torch.nn.L1Loss()
        self.rec_loss = l1_loss(params['norm'][0],params['norm'][1],device=params['device'])
        self.discr_loss = disc_loss()
        self.gene_loss = gen_loss()
        # ==================================================
        # s_enc = torch.optim.lr_scheduler.CyclicLR(self.opt_enc, base_lr=0.001, max_lr=0.01,step_size_up=2558,cycle_momentum=False)
        # s_dec = torch.optim.lr_scheduler.CyclicLR(self.opt_dec, base_lr=0.001, max_lr=0.01,step_size_up=2558,cycle_momentum=False)
        # s_gen = torch.optim.lr_scheduler.CyclicLR(self.opt_gen, base_lr=0.001, max_lr=0.01,step_size_up=2558,cycle_momentum=False)
        # s_disc = torch.optim.lr_scheduler.CyclicLR(self.opt_disc, base_lr=0.001, max_lr=0.01,step_size_up=2558,cycle_momentum=False)
        s_enc = torch.optim.lr_scheduler.CyclicLR(self.opt_enc, base_lr=0.001, max_lr=0.01,step_size_up=params['step_up_scheduler'],cycle_momentum=False)
        s_dec = torch.optim.lr_scheduler.CyclicLR(self.opt_dec, base_lr=0.001, max_lr=0.01,step_size_up=params['step_up_scheduler'],cycle_momentum=False)
        s_gen = torch.optim.lr_scheduler.CyclicLR(self.opt_gen, base_lr=0.001, max_lr=0.01,step_size_up=params['step_up_scheduler'],cycle_momentum=False)
        s_disc = torch.optim.lr_scheduler.CyclicLR(self.opt_disc, base_lr=0.001, max_lr=0.01,step_size_up=params['step_up_scheduler'],cycle_momentum=False)
    
        scheduler = Scheduler_AE(s_enc,s_dec,s_gen,s_disc) 

        # scheduler = None

        super().__init__(model_save_path=params['weights_dir'],
                         train_dataset=train_patches,
                         trainer=trainer,
                         device=params['device'],
                         eval_dataset=eval_dataset,
                         batch_size=params['batch_size'],
                         num_workers=0,
                         epoch_count=params['epochs'],
                         print_frequency=params['print_freq'],
                         state_storage=state_storage,
                         loss_lr_storage=loss_lr_storage,
                         scheduler = scheduler )


