import torch
import torch.nn as nn
from utils import AttrDict,get_stft_outch,get_stft_outlen,init_weights,get_padding_down
from torch.nn.utils import weight_norm,remove_weight_norm
from torchsummaryX import summary
from layers import ResBlock
import math

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

def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis


def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis
    
class SACActor(nn.Module):

    def __init__(self,h:AttrDict):
        super().__init__()
        self.h = h
        self.layer1 = BaseModel(h)
        ich = h.base_out_ch
        och = h.action_space_size
        self.layer_mean = nn.Linear(ich,och)
        self.layer_log_std = nn.Linear(ich,och)

    def forward(self, states:tuple[torch.Tensor]) -> torch.Tensor:
        """
        states : (src_spects:(N,C,L), generated_spects:(N,C,L))
        return : (actions)
        """
        #assert states[0].shape == states[1].shape
        x = self.layer1(states[0],states[1]).relu()
        x = self.layer_mean(x).tanh()
        return x

    def sample(self,states:tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        x = self.layer1(states[0],states[1])
        means,log_stds = self.layer_mean(x).squeeze(0),self.layer_log_std(x)
        return reparameterize(means,log_stds.clamp(-20,2))

class _sac_critic(nn.Module):

    def __init__(self,h:AttrDict):
        super().__init__()

        self.s_ch = h.base_out_ch
        self.a_ch = h.critic_action_hdim
        self.outch = self.s_ch + self.a_ch

        self.s_net = BaseModel(h)
        self.a_net = nn.Sequential(
            nn.Linear(h.action_space_size,self.a_ch),nn.ReLU(),
            nn.Linear(self.a_ch,self.a_ch)
        )
        self.linear_post = nn.Linear(self.outch,1)

    def forward(self, states:tuple[torch.Tensor],action:torch.Tensor) -> torch.Tensor:

        x1 = self.s_net(states[0],states[1])
        x2= self.a_net(action[None,])
        x = torch.cat([x1,x2],dim=-1).relu()
        x = self.linear_post(x)
        return x

class SACCritic(nn.Module):
    def __init__(self,h:AttrDict) -> None:
        super().__init__()

        self.net1 = _sac_critic(h)
        self.net2 = _sac_critic(h)

    def forward(self, states:tuple[torch.Tensor],actions:torch.Tensor) -> torch.Tensor:
        return self.net1(states,actions),self.net2(states,actions)
        




if __name__ == "__main__":
    from utils import load_config
    h = load_config("hparams/origin.json")
    m = BaseModel(h)
    m.summary()
    m.remove_weight_norm()