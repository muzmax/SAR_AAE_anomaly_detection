import torch.nn as nn
import torch
from pipeline.utils import move_to_device, disp_sar

class l1_loss(torch.nn.Module):
    def __init__(self,m, M,device='cuda') -> None:
        super().__init__()

        denorm = 2*(M-m)
        denorm = torch.from_numpy(denorm)
        denorm = move_to_device(denorm,device)

        self.denorm = torch.reshape(denorm,(1,denorm.shape[0],1,1))
        self.loss = nn.L1Loss()
        
    def forward(self,inp,out):
            
        inp_log = torch.mul(inp,self.denorm)
        out_log = torch.mul(out,self.denorm)
        return self.loss(inp_log,out_log)


class l1_loss_weight(torch.nn.Module):
    def __init__(self,m, M,device='cuda',method='delete', mean_size=5) -> None:
        super().__init__()

        denorm = 2*(M-m)
        denorm = torch.from_numpy(denorm)
        denorm = move_to_device(denorm,device)

        self.device = device
        self.denorm = torch.reshape(denorm,(1,denorm.shape[0],1,1))
        self.loss = nn.L1Loss()
        self.method = method
        self.mean_size = mean_size
    
    def mean_weight(self,inp,weights):
        b,c,m,n = inp.shape

        inp_weight = torch.mul(inp,weights).float()
        median = torch.empty(b,c,device=self.device,dtype=torch.float)
        for batch in range(b):
            ind_batch = (inp_weight[batch,0,:,:] != 0).nonzero(as_tuple=True)
            assert(ind_batch[0].shape[0]!=0)
            median_batch = inp_weight[batch,:,ind_batch[0],ind_batch[1]]
            median[batch,:] = torch.mean(median_batch,1)[0]
            
        median_mat = torch.ones(inp_weight.shape,device=self.device,dtype=torch.float)
        median_mat = torch.mul(median_mat,median.view(b,c,1,1))

        inp_weight = torch.where(inp_weight==0,median_mat,inp_weight)
        return inp_weight

    def median_weight(self,inp,weights):
        b,c,m,n = inp.shape

        inp_weight = torch.mul(inp,weights).float()
        median = torch.empty(b,c,device=self.device,dtype=torch.float)
        for batch in range(b):
            ind_batch = (inp_weight[batch,0,:,:] != 0).nonzero(as_tuple=True)
            median_batch = inp_weight[batch,:,ind_batch[0],ind_batch[1]]
            median[batch,:] = torch.median(median_batch,1)[0]
            
        median_mat = torch.ones(inp_weight.shape,device=self.device,dtype=torch.float)
        median_mat = torch.mul(median_mat,median.view(b,c,1,1))

        inp_weight = torch.where(inp_weight==0,median_mat,inp_weight)
        return inp_weight
                        

    def forward(self,inp,out,weights):

        inp_log = torch.mul(inp,self.denorm)
        out_log = torch.mul(out,self.denorm)

        if self.method == 'delete':
            inp_weight = torch.mul(inp_log,weights)
            out_weight = torch.mul(out_log,weights)

        elif self.method == 'mean' :
            inp_weight = self.mean_weight(inp_log,weights)
            out_weight = out_log
        
        elif self.method == 'median' :
            inp_weight = self.median_weight(inp_log,weights)
            out_weight = out_log

        else :
            raise NameError('Method is not known')
        
        return self.loss(inp_weight,out_weight)