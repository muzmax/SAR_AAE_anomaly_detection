from pipeline.utils import run_train_AAE,load_config
import numpy as np

def mainAAE(params):
    if params['RX']:
        config = load_config('configs/AAE_RX_train','Config_AAE_RX')
    else:
        config = load_config('configs/AAE_train','Config_AAE')
    conf = config(params)
    run_train_AAE(conf)
    
    
if __name__ == "__main__":
    params = {}
    params['RX'] = True # use the standard AAE or the modified version with RX filter
    params['device'] = 'cuda'
    params['save_dir'] = './data/res_train_test' # save evaluation images
    params['eval_dir'] = './data/eval'
    params['train_dir'] = './data/train'
    params['weights_dir'] = './pipeline/out/L'
    params['norm'] = np.array([[-1,-1,-1,-1],[1,1,1,1]]) # matrix of size (2,c) c being the number of channels
    params['channels'] = 4 # 1 ofr SLC and 4 for full pol
    params['pat_size'] = 32
    params['stride'] = 32
    params['z_size'] = 128 # lattent space size
    params['batch_size'] = 100
    params['epochs'] = 50
    params['label'] = False # If there is labels in the folder
    params['print_freq'] = 100 # Print in the logger for each image
    params['semi_kernel_sz'] = 5 # size of the semi kernel to compute scm
    params['step_up_scheduler'] = 900 # size to go from low to high in the scheduler
    """ if RX is set to true the folder train_RX_dir should contain anomaly maps
        it consist in images that have a dynamic between 0 (normal) and 1 (abnormal)
        anomaly maps should have the exact same name as the original image (to be sorted in the same order)"""
    if params['RX']:
        params['train_RX_dir'] = './train_rx'
        params['bin_RX'] = True
        params['bin_method'] = 'median' # method to replace the abnormal pixel. 'median', 'delete' and 'mean' are suported
        params['tresh_RX'] = 0.04

    mainAAE(params)


