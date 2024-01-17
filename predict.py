from pipeline.utils import run_predict_AAE, run_gen_AAE,load_config

def mainAAE(params): # Set to True to Generate fake patch with decoder 

    # Encoding - decoding config
    if params['type'] == 'predictor':
        config = load_config('configs/AAE_pred','Config_pred_AAE')
        conf = config(params) 
        run_predict_AAE(conf)
    # Generator config
    elif params['type'] == 'generator':
        config = load_config('configs/AAE_pred','Config_gen_AAE')
        conf = config(params) 
        run_gen_AAE(conf)

if __name__ == "__main__":
    
    params = {}
    params['type'] = 'predictor' # 'predictor' to reconstruct images and 'generator' to generate patches only
    params['device'] = 'cuda'
    params['save_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/results/rx_0.04_3'
    params['weights_dir'] = './pipeline/out/L_rx_0.04'
    params['norm'] = [-2.0,9.5] # TSX
    params['channels'] = 4 # 1 ofr SLC and 4 for full pol
    params['pat_size'] = 32
    params['stride'] = 32
    params['z_size'] = 128 # lattent space size
    assert params['type'] in ['predictor','generator']
    if params['type'] == 'predictor':
        params['eval_dir'] = '/media/max/TOSHIBA EXT/ONERA_SONDRA/data/under/Gretsi 2k23/eval_patches/denoised'
        params['batch_size'] = 100
        params['print_freq'] = 1 # Print in the logger for each image
        params['semi_kernel_sz'] = 5 # size of the semi kernel to compute scm
        params['label'] = False # If there is labels in the folder to compute ROC and AUC
    elif params['type'] == 'generator':
        params['pred_nb'] = 10 # number of patches to create

    mainAAE(params)
