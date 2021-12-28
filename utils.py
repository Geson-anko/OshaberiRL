from torch.nn.utils import weight_norm
import torch
from typing import Tuple
import json
import math
import warnings
from datetime import datetime
def get_now(strf:str = '%Y-%m-%d_%H-%M-%S'):
    now = datetime.now().strftime(strf)
    return now
class AttrDict(dict):
    """ This class treats the dict as class attribute."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
def load_config(path_to_config:str) -> AttrDict:
    """ load the config file"""
    with open(path_to_config,"r", encoding="utf-8") as f:
        d = json.load(f)
    return AttrDict(d)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_conv1d_outlen(in_length:int, kernel_size:int, padding:int = 0, stride:int = 1, dilation:int = 1) -> int:
    return int((in_length + 2*padding - dilation * (kernel_size -1) -1)/stride + 1)

def get_conv1dTranspose_outlen(in_length:int, kernel_size:int, padding:int=0, stride:int=1, output_padding:int=0, dilation:int=1) -> int:
    return (in_length -1)*stride - 2*padding + dilation*(kernel_size -1) + output_padding+1

def get_padding_down(Lin:int, Lout:int, kernel_size:int, stride:int=1, dilation:int=1) -> int:
    """return padding length to match target Lout. (for down samplling)"""
    assert Lin >= Lout
    return math.ceil((stride*(Lout - 1) - Lin+ dilation*(kernel_size - 1) + 1)/2)

def get_padding_up(Lin:int, Lout:int, kernel_size:int, stride=1, output_padding:int=0, dilation:int=1) -> int:
    """return padding length to math target Lout. (for up sampling)"""
    assert Lin <= Lout
    pad = ((Lin - 1)*stride + dilation*(kernel_size - 1) + output_padding+ 1 - Lout)/2
    if pad < 0:
        raise ValueError(f"padding size is invalid {pad}. Please check {Lin, Lout, kernel_size,stride,output_padding,dilation}")
    elif pad%1 > 0:
        warnings.warn(f"No exists pilitic padding :{pad}. Please check {Lin, Lout, kernel_size,stride,output_padding,dilation}", UserWarning)
    return int(pad)

def walk_ratent_space(dims:int, steps:int, resolution:int=4,start:float = 0.0, end:float = 1.0,dtype=torch.float, device:torch.device="cpu") -> torch.Tensor:
    """walk ratent space thoroughly"""
    line = (torch.linspace(start,end,steps,device=device,dtype=dtype)*math.pi).unsqueeze(1)
    f = (dims*resolution)**torch.arange(dims,device=device,dtype=dtype).unsqueeze(0)
    actions = torch.cos(line*f).unsqueeze(-1)
    return actions

def get_stft_outlen(breath_len:int,hop_length:int) -> int:
    return (breath_len // hop_length) + 1

def get_stft_outch(n_fft:int) -> int:
    return (n_fft//2) + 1