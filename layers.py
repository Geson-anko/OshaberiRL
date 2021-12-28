import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import init_weights ,get_padding,get_stft_outch,get_stft_outlen
import copy 


class ResBlock(nn.Module):
    def __init__(self,channels,kernel_size:int=3,num_layers:int=1):
        super().__init__()
        self.num_layers= num_layers
        self.kernel_size = kernel_size

        self.convs1 = self.get_convs(channels,kernel_size,num_layers)
        self.convs1.apply(init_weights)
        self.convs2 = self.get_convs(channels,kernel_size,num_layers)
        self.convs2.apply(init_weights)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            c1,c2 = (
                self.convs1[i],self.convs2[i]
            )
            xt = c1(x)
            xt = F.relu(xt)
            xt = c2(xt)
            xt = F.relu(xt)
            x = xt+x
        return x
    def get_convs(self,channels,kernel_size,num_layers):
        convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels,channels,kernel_size,1,get_padding(kernel_size,1))) \
                for _ in range(num_layers)
        ])
        return convs

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

if __name__ == '__main__':
    from torchsummaryX import summary
    m = ResBlock(32,3,2)
    dum = torch.randn(1,32,100)
    summary(m,dum)
    m.remove_weight_norm()
