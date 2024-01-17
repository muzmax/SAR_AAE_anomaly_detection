import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from PIL import Image  

from .logger import setup_logger



# ======================================
#         ADVERSARIAL AUTOENCODER
# ======================================

# Train an adversarial autoencoder with his config
def run_train_AAE(config):
    
    train_data = DataLoader(config.train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers)
    
    eval_data = DataLoader(config.eval_dataset)
    model_save_path = config.model_save_path
    os.makedirs(model_save_path, exist_ok=True)
    logger_path = os.path.join(model_save_path, "log.txt")
    setup_logger(out_file=logger_path)
    
    trainer = config.trainer(encoder=config.encoder,
                                decoder = config.decoder,
                                discriminator = config.discriminator,
                                train_data_loader=train_data,
                                eval_data_loader=eval_data,
                                epoch_count=config.epoch_count,
                                opt_enc =config.opt_enc,
                                opt_dec =config.opt_dec,
                                opt_gen =config.opt_gen,
                                opt_disc =config.opt_disc,
                                scheduler=config.scheduler,
                                rec_loss=config.rec_loss,
                                disc_loss=config.discr_loss,
                                gen_loss=config.gene_loss,
                                print_frequency=config.print_frequency,
                                device=config.device,
                                model_save_path=model_save_path,
                                state_storage=config.state_storage,
                                loss_lr_storage = config.loss_lr_storage,
                                norm = config.norm,
                                metrics_calculator = config.metrics_calculator)
    trainer.run()
    
# Make predictions with an adversarial autoencoder in a config
def run_predict_AAE(config):

    eval_data = DataLoader(
        config.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers)

    model_save_path = config.model_save_path
    assert os.path.exists(model_save_path), "{} does not exist".format(model_save_path)
    os.makedirs(config.save_im_path, exist_ok=True)
    logger_path = os.path.join(model_save_path, "log_predict.txt")
    setup_logger(out_file=logger_path)
    
    predictor = config.predictor(encoder=config.encoder,
                                    decoder=config.decoder,
                                    data_loader=eval_data,
                                    print_frequency=config.print_frequency,
                                    device=config.device,
                                    model_save_path=model_save_path,
                                    norm = config.norm,
                                    save_im_path = config.save_im_path,
                                    ano_score = config.ano_score,
                                    metrics_calculator = config.metrics_calculator,
                                    patch_size = config.patch_size,
                                    stride = config.stride)
    predictor.run()

# Make predictions with an adversarial autoencoder in a config
def run_gen_AAE(config):

    model_save_path = config.model_save_path
    assert os.path.exists(model_save_path), "{} does not exist".format(model_save_path)
    os.makedirs(config.save_im_path, exist_ok=True)
    logger_path = os.path.join(model_save_path, "log_gen.txt")
    setup_logger(out_file=logger_path)

    generator = config.generator(decoder=config.decoder,
                                    discriminator = config.discriminator,
                                    device=config.device,
                                    model_save_path=model_save_path,
                                    norm = config.norm,
                                    save_im_path = config.save_im_path,
                                    pred_nb = config.pred_nb
                                    )
    generator.run()

# =====================================================================================

# Load a configuration
def load_config(module_path, cls_name):
    module_path_fixed = module_path
    if module_path_fixed.endswith(".py"):
        module_path_fixed = module_path_fixed[:-3]
    module_path_fixed = module_path_fixed.replace("/", ".")
    module = importlib.import_module(module_path_fixed)
    assert hasattr(module, cls_name), "{} file should contain {} class".format(module_path, cls_name)

    cls = getattr(module, cls_name)
    return cls

# Save a model
def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module
        
    with open(path, "wb") as fout:
        torch.save(model.state_dict(), fout)
        
# Load a model
def load_model(model, path):
    with open(path, "rb") as fin:
        state_dict = torch.load(fin)
        
    model.load_state_dict(state_dict)

# From cpu to gpu or the opposite
def move_to_device(tensor: list or tuple or torch.Tensor, device: str):
    if isinstance(tensor, list):
        return [move_to_device(elem, device=device) for elem in tensor]
    if isinstance(tensor, tuple):
        return (move_to_device(elem, device=device) for elem in tensor)
    return tensor.to(device)

def normalize01(im,val=None):
    if val == None:
        m = np.amin(im)
        M = np.amax(im)
    else:
        m = val[0]
        M = val[1]
    im_norm = (im-m)/(M-m)
    return im_norm

# Apply a treshold, a defined treshold or mean+3*var
def tresh_im(img,treshold=None,k=3):

    imabs = np.abs(img)
    sh = imabs.shape

    if treshold == None:
        if len(sh) == 2:
            mean = np.mean(imabs)
            std = np.std(imabs)
            treshold = mean+k*std
            imabs = np.clip(imabs,None,treshold)
            imabs = normalize01(imabs)
            # print('treshold : {}'.format(treshold))

        elif len(sh) == 3:
            for i in range(sh[2]):
                im_p = imabs[:,:,i] # take channel i
                mean = np.mean(im_p)
                std = np.std(im_p)
                treshold_p = mean+k*std # compute treshold
                im_p = np.clip(im_p,None,treshold_p) # apply treshold
                im_p = normalize01(im_p) # normalize [0-1]
                imabs[:,:,i] = im_p
                # print('treshold for channel {} : {}'.format(i,treshold_p))

    else:
        if len(sh) == 2:
            imabs = np.clip(imabs,None,treshold)
            imabs = normalize01(imabs)

        elif len(sh) == 3:
            for i in range(sh[2]):
                if len(treshold) == sh[2]:
                    im_p = imabs[:,:,i] # take channel i
                    im_p = np.clip(im_p,None,treshold[i]) # apply treshold
                    im_p = normalize01(im_p,[0,treshold[i]]) # normalize [0-max/treshold]
                    imabs[:,:,i] = im_p
                else:
                    print('Number of tresholds should be the same as number of channels but got {} and {}'.format(len(treshold),sh[2]))    
    return imabs

# Plot an image (no treshold is applied)
def plot_im(img,title = '',bar = False,save=False,save_path='.'):
    if len(img.shape) == 2:
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)
    if bar:
        plt.colorbar()
    plt.axis('off')

    if not save:
        plt.title(title)
    else:
        plt.savefig('{}/{}.png'.format(save_path,title),bbox_inches = 'tight',pad_inches = 0)
    plt.show()

def save_plot(im,fold,bar = False,):
    if len(im.shape) == 2:
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    if bar:
        cb = plt.colorbar()
    plt.axis('off')
    plt.savefig(fold,bbox_inches = 'tight',pad_inches = 0)
    # if bar:
    #     cb.remove()
    plt.clf()

def save_curve(x,y,fold,curve_name=''):
    plt.plot(x,y,label = curve_name)
    plt.legend()
    plt.savefig(fold,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

# Plot an image with a treshold, if tresh is None it's an automatic treshold mean+3*var
def disp_sar(im,tresh=None, ch=None, title = ''):
    im = np.abs(im)
    shape_im = im.shape

    # Take one chanel
    if ch is not None:
        im = im[:,:,ch]

    # Create RGB image with G = 1/2(hv+vh)
    elif len(shape_im) == 3 and shape_im[2] == 4:
        polsar_im = np.zeros((shape_im[0],shape_im[1],3),dtype=np.single)
        polsar_im[:,:,0] = im[:,:,0]
        polsar_im[:,:,1] = (im[:,:,1]+im[:,:,2])/2
        polsar_im[:,:,2] = im[:,:,3]
        im = polsar_im

    # Apply treshold to image
    if tresh == None:
        im_t = tresh_im(im)
    else:
        im_t = tresh_im(im,treshold=tresh)
    
    plot_im(im_t, title = title)

# Save an image
def save_im(im,fold,is_sar=True,tresh=None):

    im = np.abs(im)
    shape_im = im.shape

    if len(shape_im) == 2:
        polsar_im = im
    elif len(shape_im) == 3:
        if shape_im[2] == 4:
            polsar_im = np.zeros((shape_im[0],shape_im[1],3),dtype=np.single)
            polsar_im[:,:,0] = im[:,:,0]
            polsar_im[:,:,1] = (im[:,:,1]+im[:,:,2])/2
            polsar_im[:,:,2] = im[:,:,3]
        elif shape_im[2] == 2:
            polsar_im = np.zeros((shape_im[0],shape_im[1],3),dtype=np.single)
            polsar_im[:,:,0] = im[:,:,0]
            polsar_im[:,:,1] = im[:,:,1]
            polsar_im[:,:,2] = im[:,:,1]
        elif shape_im[2] == 3:
            polsar_im = im
        else:
            print('Number of channel should be 2, 3 or 4 but is {}'.format(shape_im[2]))
            return 
    else :
        name = os.path.basename(fold)
        print('can''t save {} because it is not an image'.format(name))
        return

    if is_sar:
        polsar_im = tresh_im(polsar_im,treshold=tresh)*255
    else :
        polsar_im = normalize01(polsar_im)*255
    if len(shape_im) == 2:
        polsar_im = Image.fromarray(polsar_im.astype(np.uint8))
    elif len(shape_im) == 3:
        polsar_im = Image.fromarray(polsar_im.astype(np.uint8), 'RGB')
    polsar_im.save(fold) 
    
def plot_hist(im):
    m = np.amin(im)
    M = np.amax(im)
    std = np.std(im)
    mean = np.mean(im)

    plt.figure()
    plt.hist(np.ravel(im),bins='auto',density=True)  
    plt.title('min : {:.3}. max : {:.3}. std : {:.3}. mean : {:.3}'.format(m,M,std,mean))
    plt.show()

if __name__ == "__main__":  
    pass
