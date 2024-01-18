from pipeline.utils import run_predict_AAE,load_config
import numpy as np

def mainAAE(params): # Set to True to Generate fake patch with decoder 

    config = load_config('configs/AAE_pred','Config_pred_AAE')
    conf = config(params) 
    run_predict_AAE(conf)

if __name__ == "__main__":
    
    params = {}
    params['type'] = 'generator' # 'predictor' to reconstruct images and 'generator' to generate patches only
    params['device'] = 'cuda'
    params['save_dir'] = './data/res_pred_test'
    params['weights_dir'] = './pipeline/out/L_rx_0.04'
    params['norm'] = np.array([[-1,-1,-1,-1],[10.5,10.5,10.5,10.5]]) # matrix of size (2,c) c being the number of channels
    params['channels'] = 4 # 1 for SLC and 4 for full pol
    params['pat_size'] = 32
    params['stride'] = 16
    params['z_size'] = 128 # lattent space size
    params['eval_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/denoised'
    params['batch_size'] = 1 # images of different sizes
    params['print_freq'] = 1 # Print in the logger for each image
    params['semi_kernel_sz'] = 5 # size of the semi kernel to compute scm
    """ If there is labels in the folder to compute ROC and AUC. 
        Label image should be binary in the same folder as the images.
        For an image im.npy the masked image should be im_mask.npy """
    params['label'] = False 
    mainAAE(params)
