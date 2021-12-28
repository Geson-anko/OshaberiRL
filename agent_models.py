import torch
import torch.nn as nn
from utils import AttrDict,get_stft_outch,get_stft_outlen,init_weights,get_padding_down
from torch.nn.utils import weight_norm,remove_weight_norm
from torchsummaryX import summary
from layers import ResBlock

class BaseModel(nn.Module):
    """
    This base model is to create Actor and Critic.
    Please make the output layer after this.
    """

    def __init__(self,h:AttrDict) -> None:
        super().__init__()
        self.h=h
        self.in_ch = h.num_mels
        self.in_len = get_stft_outlen(h.breath_len,h.hop_len)
        dks = h.downsample_kernel_sizes
        self.n_downsamples = len(dks)
        self.init_ch = h.init_ch
        dsr = h.downsample_rate
        uscr = int(dsr/2) # up sample channel rate
        rks = h.res_kernel_size
        rnl = h.res_num_layers
        self.base_out_ch = h.base_out_ch

        self.input_size= (1,self.in_ch,self.in_len)
        # layers

        self.conv_pre = weight_norm(nn.Conv1d(self.in_ch,self.init_ch,3,1,1))
        self.conv_pre.apply(init_weights)

        ds_tgt_lens = [int(self.in_len/(dsr**i)) for i in range(dsr+1)]
        ds_pads = [
            get_padding_down(ds_tgt_lens[i],ds_tgt_lens[i+1],dks[i],dsr) for i in range(self.n_downsamples)
            ]
        ds_ches = [
            (self.init_ch*2)*(uscr**i) for i in range(self.n_downsamples+1)
        ]
        self.ds_convs = nn.ModuleList() # down sampling convolutions
        for i in range(self.n_downsamples):
            self.ds_convs.append(
                weight_norm(nn.Conv1d(ds_ches[i],ds_ches[i+1],dks[i],dsr,ds_pads[i]))
            )
        self.ds_convs.apply(init_weights)

        self.res_blocks = nn.ModuleList([
            ResBlock(ds_ches[i+1],rks,rnl) for i in range(self.n_downsamples)
        ])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear_post = nn.Linear(ds_ches[-1],self.base_out_ch)

    def forward(self, src_spect:torch.Tensor, generated_spect:torch.Tensor) -> torch.Tensor:
        x0,x1 = self.conv_pre(src_spect),self.conv_pre(generated_spect)
        x = torch.cat([x0,x1],dim=1).relu()
        for i in range(self.n_downsamples):
            x = self.ds_convs[i](x).relu()
            print("ds conv",i,x.shape)
            x = self.res_blocks[i](x)
            print("res conv",i,x.shape)

        x = self.gap(x).squeeze(-1)
        x = self.linear_post(x)
        return x

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy,dummy)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ds_convs:
            remove_weight_norm(l)
        for l in self.res_blocks:
            l.remove_weight_norm()



if __name__ == "__main__":
    from utils import load_config
    h = load_config("hparams/origin.json")
    m = BaseModel(h)
    m.summary()
    m.remove_weight_norm()