import torch.nn as nn
import torch

class disc_loss(torch.nn.Module):
    def __init__(self, eps = 10**(-10)) -> None:
        super().__init__()
        self.eps = eps
            
    def forward(self,disc_real,disc_fake):
        return -torch.mean(torch.log(disc_real+self.eps) + torch.log(1-disc_fake+self.eps))

class gen_loss(torch.nn.Module):
    def __init__(self, eps = 10**(-10)) -> None:
        super().__init__()
        self.eps = eps
        
    def forward(self,disc_fake):
        return torch.mean(torch.log(1-disc_fake+self.eps))


