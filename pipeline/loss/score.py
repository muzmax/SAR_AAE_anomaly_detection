
import torch.nn as nn 
import torch
import numpy as np
import copy

class anomaly_score():
    def __init__(self,l=0.9) -> None:
        assert isinstance(l,(int,float))
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 
        self.l_ = l
    
    def __call__(self, real,fake,f_real,f_fake):
        
        L = self.mse(f_real,f_fake)
        R = self.l1(real,fake)
        score = self.l_ * R + (1-self.l_) * L
        return score

def normalize01(im,val=None):
    if val == None:
        m = np.amin(im)
        M = np.amax(im)
    else:
        m = val[0]
        M = val[1]
    im_norm = (im-m)/(M-m)
    return im_norm

class print_progress():
    def __init__(self,n=None,width=40) -> None:
        self.n = n
        self.w = width
    def set_n(self,n):
        self.n = n
    def progress(self,percent=0):
        left = self.w * percent // 100
        right = self.w - left     
        tags = "#" * left
        spaces = " " * right
        print("\r[{}{}]{:.0f}%".format(tags,spaces,percent),end="",flush=True) 
        if percent == 100:
            print("")
    def __call__(self,i,p_print):
        assert self.n != None
        p = (i/(self.n-1))*100
        if p>=p_print:
            self.progress(p_print)
            p_print+=1
        return p_print

class detection_scores():
    def __init__(self,sk) -> None:
        self.progress = print_progress()
        self.sk = sk # half of kernel size
        self.maps = {}
        
    def compute_scm_smv(self,x): 
        mu = np.mean(x,axis=0)
        x_centered = (x-mu).T
        # x_centered = x.T
        (p, N) = x.shape
        sigma = (x_centered @ x_centered.conj().T) / (N-1)
        return sigma,mu

    def multi_comp(self,S1,S2,i,j):
        
        Q1,D1,_ = np.linalg.svd(S1,hermitian=True)
        Q1_h = Q1.conj().T
        Q2,D2,_ = np.linalg.svd(S2,hermitian=True)
        Q2_h = Q2.conj().T

        D1_log = np.diag(np.log(D1))
        D2_log = np.diag(np.log(D2))
        S1_log = Q1@D1_log@Q1_h
        S2_log = Q2@D2_log@Q2_h

        D1_root = np.diag(np.sqrt(D1))
        D2_root = np.diag(np.sqrt(D2))
        S1_root = Q1@D1_root@Q1_h
        S2_root = Q2@D2_root@Q2_h
            
        self.maps['Euclidean'][i,j] = np.linalg.norm(S1-S2,ord='fro')**2
        self.maps['Log-Euclidean'][i,j] = np.linalg.norm(S1_log-S2_log,ord='fro')**2
        self.maps['Root-Euclidean'][i,j] = np.linalg.norm(S1_root-S2_root,ord='fro')**2

    # im1 : rec, im2 : input
    def create_maps(self,im1:torch.tensor,im2:torch.tensor): 

        assert im1.shape == im2.shape
        im1 = im1.cpu().data.numpy()
        im1 = im1.transpose((1,2,0))
        im2 = im2.cpu().data.numpy()
        im2 = im2.transpose((1,2,0))
        (m,n,c) = im1.shape

        # test of different metric
        self.maps = {}
        self.maps['L1'] = np.sum(np.abs(im1-im2),axis=2)
        self.maps['Euclidean'] = np.zeros((m,n))
        self.maps['Log-Euclidean'] = np.zeros((m,n))
        self.maps['Root-Euclidean'] = np.zeros((m,n))
        self.maps['mean'] = np.zeros((m,n))

        # Save parameters
        self.maps['param'] = {}
        self.maps['param']['cov_rec'] = np.zeros((m,n,c,c))
        self.maps['param']['mean_rec'] = np.zeros((m,n,c))
        self.maps['param']['cov_inp'] = np.zeros((m,n,c,c))
        self.maps['param']['mean_inp'] = np.zeros((m,n,c))
        # ===============================================================
        
        k = self.sk
        self.progress.set_n(m)
        p_print = 0
        for i in range(m):
            p_print = self.progress(i,p_print)
            for j in range(n):

                # Compute scm and smv
                up = max(0,i-k)
                down = min(m,i+k+1)
                left = max(0,j-k)
                right = min(n,j+k+1)
                sub_im1 = im1[up:down,left:right,:]
                sub_im1 = np.reshape(sub_im1,(-1,c))
                cov_ij_1,mean_ij_1 = self.compute_scm_smv(sub_im1)
                sub_im2 = im2[up:down,left:right,:]
                sub_im2 = np.reshape(sub_im2,(-1,c))
                cov_ij_2,mean_ij_2 = self.compute_scm_smv(sub_im2)
                
                # Store parameters
                self.maps['mean'][i,j] = np.linalg.norm(mean_ij_1-mean_ij_2)**2
                self.maps['param']['cov_rec'][i,j,:,:] = cov_ij_1
                self.maps['param']['mean_rec'][i,j,:] = mean_ij_1
                self.maps['param']['cov_inp'][i,j,:,:] = cov_ij_2
                self.maps['param']['mean_inp'][i,j,:] = mean_ij_2
                # Compute and store metrics
                self.multi_comp(cov_ij_1,cov_ij_2,i,j)
        
        L = 0.7
        self.maps['mix'] = L*normalize01(self.maps['mean'])+(1-L)*normalize01(self.maps['Euclidean'])

        return self.maps
                
    def __call__(self,im1:torch.tensor,im2:torch.tensor):

        with torch.no_grad():
            maps = self.create_maps(im1,im2)
            return maps

