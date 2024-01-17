import torch.nn as nn
import torch
from pipeline.utils import move_to_device
class log_likelihood(nn.Module):
    
    def __init__(self,m,M) -> None:
        super().__init__()
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        self.m_ = m
        self.M_ = M
        
    def forward(self,denoised,ref):
        
        batch_size = denoised.shape[0]
        coef1 = torch.mul(torch.sub(denoised,ref),2*(self.M_-self.m_))
        coef2 = torch.exp(torch.mul(torch.sub(ref,denoised),2*(self.M_-self.m_)))
        loss = torch.div(torch.sum(torch.add(coef1,coef2)),batch_size)
        return loss

class log_likelihood_pol(nn.Module):
    
    def __init__(self,norm,device) -> None:
        super().__init__()
        norm = torch.from_numpy(2*(norm[1]-norm[0]))
        norm = move_to_device(norm,device)
        self.norm_ = torch.reshape(norm,(1,norm.shape[0],1,1))
        
    def forward(self,estimation,ref):
        
        coef1 = torch.mul(torch.sub(estimation,ref),self.norm_)
        coef2 = torch.exp(torch.mul(torch.sub(ref,estimation),self.norm_))
        loss = torch.mean(torch.add(coef1,coef2))
        return loss
    
class log_likelihood_mean(nn.Module):
    
    def __init__(self,m,M) -> None:
        super().__init__()
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        self.m_ = m
        self.M_ = M
        
    def forward(self,estimation,ref):
        
        batch_size = estimation.shape[0]
        coef1 = torch.mul(torch.sub(estimation,ref),2*(self.M_-self.m_))
        coef2 = torch.exp(torch.mul(torch.sub(ref,estimation),2*(self.M_-self.m_)))
        loss = torch.mean(torch.add(coef1,coef2))
        return loss
 