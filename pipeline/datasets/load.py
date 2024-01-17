import glob
import random
import os
import numpy as np
import gc
import shutil
from scipy import signal
from skimage.morphology import disk,opening

from ..utils import normalize01
from ..datasets.preprocessing import normalization

def data_augmentation(image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)

"""
input : directory name
output : numpy matrix [numPatches, pat_size, pat_size, ch] of patches
"""
def generate_patches(src_dir="./data/train",pat_size=256,step=0,stride=16,bat_size=4,data_aug_times=1,nb_ch=4, num_pic=None):

        # calculate the number of patches
        count = 0
        filepaths = glob.glob(src_dir + '/*.npy')
        print("number of training data = %d" % len(filepaths))
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count * data_aug_times
        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print("total patches = %d , batch size = %d, total batches = %d" % \
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches=int(numPatches)
        multi_ch = (len(img.shape) != 2)
        if not multi_ch: # 1 channel
            inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="float32") 
        else: # More than 1 channel
            inputs = np.zeros((numPatches, pat_size, pat_size, nb_ch), dtype="float32") 

        # generate patches
        count = 0
        for i in range(len(filepaths)): #scan through images
            img = np.load(filepaths[i])
            img = np.abs(img)
            img_s = np.array(img, dtype="float32")
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            if not multi_ch:
                img_s_ = np.reshape(img_s, (np.size(img_s, 0), np.size(img_s, 1), 1))  # extend one dimension
            else :
                if nb_ch == 3:
                    img_s_ = np.zeros((im_h,im_w,nb_ch),dtype="float32") 
                    img_s_[:,:,0] = img_s[:,:,0]
                    img_s_[:,:,1] = 0.5*(img_s[:,:,1]+img_s[:,:,2])
                    img_s_[:,:,2] = img_s[:,:,3]
                elif nb_ch == 2:
                    img_s_ = np.zeros((im_h,im_w,nb_ch),dtype="float32") 
                    img_s_[:,:,0] = img_s[:,:,0]
                    img_s_[:,:,1] = img_s[:,:,3]
                elif nb_ch == 4:
                    img_s_ = img_s
            img_s = img_s_

            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    inputs[count, :, :, :] = img_s[x:x + pat_size, y:y + pat_size, :]
                    count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
        
        return inputs

# Changer la padding dans calcul des patches, mettre les asserts avant
def generate_patches_RX(dir,rx_dir,pat_size=256,step=0,stride=16,bat_size=4,nb_ch=4,rx_tresh = 0.71,binarize = False):
        count = 0
        filepaths = glob.glob(dir + '/*.npy')
        filepaths_RX = glob.glob(rx_dir + '/*.npy')
        assert len(filepaths) == len(filepaths_RX)
        print("number of training data = %d" % len(filepaths))

        # calculate the number of patches
        for i in range(len(filepaths)):
            path_im = filepaths[i]
            path_rx = filepaths_RX[i]
            assert os.path.basename(path_im) == os.path.basename(path_rx)
            img = np.load(path_im)
            rx = np.load(path_rx)

            shape_img = np.asarray(img.shape)
            shape_rx = np.asarray(rx.shape)
            shape_dif = (shape_img[0]-shape_rx[0],shape_img[1]-shape_rx[1])
            assert shape_dif[0] == shape_dif[1] and (shape_dif[0]%2) == 0

            im_h = shape_rx[0]
            im_w = shape_rx[1]
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count

        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print("total patches = %d , batch size = %d, total batches = %d" % \
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches=int(numPatches)
        multi_ch = (len(img.shape) != 2)
        if not multi_ch: # 1 channel
            inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="float32") 
        else: # More than 1 channel
            inputs = np.zeros((numPatches, pat_size, pat_size, nb_ch), dtype="float32") 

        RX_map = np.zeros((numPatches,pat_size,pat_size,1))

        count = 0
        # generate patches
        for i in range(len(filepaths)): #scan through images
            
            path_im = filepaths[i]
            path_rx = filepaths_RX[i]

            img = np.load(path_im)
            img = np.abs(img)

            img_s = np.array(img, dtype="float32")
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            if not multi_ch:
                img_s_ = np.reshape(img_s, (np.size(img_s, 0), np.size(img_s, 1), 1))  # extend one dimension
            else :
                if nb_ch == 3:
                    img_s_ = np.zeros((im_h,im_w,nb_ch),dtype="float32") 
                    img_s_[:,:,0] = img_s[:,:,0]
                    img_s_[:,:,1] = 0.5*(img_s[:,:,1]+img_s[:,:,2])
                    img_s_[:,:,2] = img_s[:,:,3]
                elif nb_ch == 2:
                    img_s_ = np.zeros((im_h,im_w,nb_ch),dtype="float32") 
                    img_s_[:,:,0] = img_s[:,:,0]
                    img_s_[:,:,1] = img_s[:,:,3]
                elif nb_ch == 4:
                    img_s_ = img_s
            img_s = img_s_

            rx = np.load(path_rx)
            if binarize :
                rx = np.where(rx>=rx_tresh,0,1)
                mask = disk(2)
                # mask = disk(1)
                ker = np.array([[0,1,0],
                                [1,1,1],
                                [0,1,0]])
                nz = signal.convolve2d(1-rx,ker,mode='same')
                rx = np.where(nz>=3,rx,1)                    
                rx = opening(rx,mask)
            else :
                rx = np.clip(rx,None,rx_tresh)
                rx = normalize01(rx,(0,rx_tresh))
                rx = 1-rx
            assert np.amin(rx)>=0 and np.amax(rx) <= 1

            shape_img = np.asarray(img_s.shape)
            shape_rx = np.asarray(rx.shape)
            shape_dif = (shape_img[0]-shape_rx[0],shape_img[1]-shape_rx[1])
            
            img_s = img_s[int(shape_dif[0]/2):-int(shape_dif[0]/2),int(shape_dif[1]/2):-int(shape_dif[1]/2),:]
            assert img_s.shape[0] == shape_rx[0]

            for x in range(0 + step, shape_rx[0] - pat_size, stride):
                for y in range(0 + step, shape_rx[1] - pat_size, stride):
                    
                    inputs[count, :, :, :] = img_s[x:x + pat_size, y:y + pat_size, :]
                    RX_map[count, :, :, 0] = rx[x:x + pat_size, y:y + pat_size]
                    count += 1

        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
            RX_map[-to_pad:, :, :, :] = RX_map[:to_pad, :, :, :]

        return inputs,RX_map

        
          
def load_eval_data(dir="./data/eval",label = False):
    
    filepaths = glob.glob(dir + '/*.npy')
    
    if label:
        imagepaths = list(filter(lambda file : 'mask' not in file,filepaths))
        maskpaths = list(filter(lambda file : 'mask' in file,filepaths))
        nb = len(imagepaths)
        assert(nb == len(maskpaths))
        print("Number of eval data with label = %d" %nb)
        return imagepaths,maskpaths

    nb = len(filepaths)
    print("Number of eval data without label = %d" %nb)
        
    return filepaths


    