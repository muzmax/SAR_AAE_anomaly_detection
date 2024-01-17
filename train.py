from pipeline.utils import run_train_AAE,load_config
import sys
import numpy as np



def mainAAERX(params):
    
    config = load_config('configs/AAE_RX_train','Config_AAE_RX')
    conf = config(params)
    run_train_AAE(conf)

def mainAAE(params):
    
    config = load_config('configs/AAE_train','Config_AAE')
    conf = config(params)
    run_train_AAE(conf)
    
    
if __name__ == "__main__":
    params = {}
    params['RX'] = False # use the standard AAE or the modified version with RX filter
    params['device'] = 'cuda'
    params['save_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/results/rx_0.04_3'
    params['eval_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/denoised'
    params['train_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/denoised'
    params['weights_dir'] = './pipeline/out/L_rx_0.04'
    params['norm'] = [-2.0,9.5] # TSX
    params['channels'] = 4 # 1 ofr SLC and 4 for full pol
    params['pat_size'] = 32
    params['stride'] = 16
    params['z_size'] = 128 # lattent space size
    params['batch_size'] = 100
    params['epochs'] = 50
    params['label'] = False # If there is labels in the folder
    params['print_freq'] = 100 # Print in the logger for each image
    params['semi_kernel_sz'] = 5 # size of the semi kernel to compute scm

    if params['RX']:
        params['train_RX_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/denoised'
        params['bin_RX'] = True
        params['bin_method'] = 'median' # method to replace the abnormal pixel. 'median', 'delete' and 'mean' are suported
        params['tresh_RX'] = 0.04
        mainAAERX(params)
    else:
        mainAAE(params)


