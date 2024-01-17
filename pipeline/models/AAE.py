import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Blocks
# ============================================================================
class Encoder_block(nn.Module):
    def __init__(self,in_ch,out_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4,stride = (2,2), padding=(1,1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.lr(self.bn(self.conv(x)))

class Decoder_block(nn.Module):
    def __init__(self,in_ch,out_ch,stride=(2,2),padding=(1,1)) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch,out_ch,4,stride,padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.lr(self.bn(self.upconv(x)))

# Networks
# ============================================================================

# Encoder
class ConvEncoder(nn.Module):
    def __init__(self,im_ch,c1f=32,nz=128,patch_size=64):
        super().__init__()

        fc_in_features = int(((patch_size/16)**2)*int(c1f/2))

        self.e1 = Encoder_block(im_ch,c1f)
        self.e2= Encoder_block(c1f,c1f*2)
        self.e3 = Encoder_block(c1f*2,c1f*4)
        self.e4 = Encoder_block(c1f*4,c1f*4)
        self.conv1 = nn.Conv2d(c1f*4,int(c1f/2),1) # 1d conv
        self.flat = nn.Flatten()
        self.fc = nn.Linear(fc_in_features,nz)

    def forward(self, x):
        out = self.e1(x)
        out = self.e2(out)
        out = self.e3(out)
        out = self.e4(out)
        out = self.conv1(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

# Decoder
class ConvDecoder(nn.Module):
    def __init__(self,im_ch,c1f=32,nz=128,patch_size=64):
        super().__init__()

        fc_in_features = int(((patch_size/16)**2)*int(c1f/2))

        self.patche_size = patch_size
        self.c1f = c1f

        self.fc = nn.Linear(nz,fc_in_features)
        self.conv1 = nn.Conv2d(int(c1f/2),c1f*4,1)
        self.d1 = Decoder_block(c1f*4,c1f*4)
        self.d2= Decoder_block(c1f*4,c1f*2)
        self.d3 = Decoder_block(c1f*2,c1f)
        self.d4 = Decoder_block(c1f,c1f)
        self.conv2 = nn.Conv2d(c1f,im_ch,1)
        self.act = nn.Tanh()
        

    def forward(self, x):

        out = self.fc(x)
        out = out.view(-1,int(self.c1f/2),int(self.patche_size/16),int(self.patche_size/16))
        out = self.conv1(out)
        out = self.d1(out)
        out = self.d2(out)
        out = self.d3(out)
        out = self.d4(out)
        out = self.conv2(out)
        out = self.act(out)

        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self,nz=128):
        super().__init__()

        self.lin1 = nn.Linear(nz, int(nz/2))
        self.lin2 = nn.Linear(int(nz/2), int(nz/4))
        self.lin3 = nn.Linear(int(nz/4), int(nz/8))
        self.lin4 = nn.Linear(int(nz/8), 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.drop(self.lin1(x))
        x = self.relu(x)
        x = self.drop(self.lin2(x))
        x = self.relu(x)
        x = self.drop(self.lin3(x))
        x = self.relu(x)
        
        return self.act(self.lin4(x))

# connaitre la taille des patches en sortie de convolution
# ============================================================================

def conv_size(size,ker=(3,3),pad=(1,1),stride=(1,1),dil=(1,1)):
    h_out = (size[0]+2*pad[0]-dil[0]*(ker[0]-1)-1)/stride[0]+1
    w_out = (size[1]+2*pad[1]-dil[1]*(ker[1]-1)-1)/stride[1]+1
    return((h_out,w_out))

def conv_transpose_size(size,ker=(3,3),pad=(1,1),stride=(1,1),dil=(1,1),out_pad=(0,0)):
    h_out = (size[0]-1)*stride[0]-2*pad[0]+dil[0]*(ker[0]-1)+out_pad[0]
    w_out = (size[1]-1)*stride[1]-2*pad[1]+dil[1]*(ker[1]-1)+out_pad[1]
    return((h_out,w_out))
