from utils import AttrDict,get_stft_outlen
import torch
from datasets import RandomSample
from torchaudio.transforms import MelSpectrogram
import librosa
import numpy as np
from pydub import AudioSegment
import torch.nn.functional as F
from scipy.io import wavfile



class OshaberiEnv(object):

    def __init__(self, config:AttrDict,dataset_path:str,on_memory:bool=False, device:torch.device ='cpu',dtype:torch.dtype=torch.float) -> None:
        """configure and initialize environment"""
        self.h = config
        self.dataset_path = dataset_path
        self.on_memory = on_memory
        self.device = device
        self.dtype=dtype


        self.base_voice_file = config.base_voice_file
        self.frame_rate = config.frame_rate
        self.sample_width = config.sample_width
        self.sample_ch = config.sample_ch
        self.n_fft = config.n_fft
        self.hop_len = config.hop_len
        self.win_len = config.win_len
        self.f_min = config.f_min
        self.f_max = config.f_max
        self.num_mels = config.num_mels
        self.breath_len = config.breath_len
        self.action_space_size = config.action_space_size

        self.sample_range = 2**(8*self.sample_width-1)
        self.power_range = config.power_range
        self.pitch_shift_range = config.pitch_shift_range
        self.min_duration = self.n_fft
        self.max_duration = self.breath_len
        self.spect_len = get_stft_outlen(self.breath_len,self.hop_len)


        self.mel_spector = MelSpectrogram(
            self.frame_rate,self.n_fft,self.win_len,self.hop_len,
            self.f_min,self.f_max,n_mels=self.num_mels).to(self.device,self.dtype)

        self.dataset = RandomSample(config,self.dataset_path,self.on_memory)

        self.reset()

    def reset(self) -> tuple[torch.Tensor]:
        """reset env. """
        self.set_initial_state()
        self.load_base_voice()
        self.generated_wave = np.empty((0,),np.float32)
        self.generated_spect_len  = 0
        return (self.source_spect.T, self.generated_pad_spect.T)

    source_spect:torch.Tensor# (timestep, channels)
    generated_pad_spect:torch.Tensor
    def set_initial_state(self) -> None:
        """set source spectrogram and empty generated spectrogram tensor. """
        wave = self.dataset[0][0].squeeze(0).to(self.device,self.dtype)
        mel = torch.log1p(self.mel_spector(wave)).T
        self.source_spect = mel
        self.generated_pad_spect = torch.ones_like(mel) * -1        
        
    base_voice:np.ndarray #(samples,)
    base_voice_silence:np.ndarray
    def load_base_voice(self) -> None:
        """loading base voice file"""
        base_voice:AudioSegment = AudioSegment.from_file(self.base_voice_file)
        base_voice = base_voice.set_channels(self.sample_ch)
        base_voice = base_voice.set_frame_rate(self.frame_rate)
        base_voice = base_voice.set_sample_width(self.sample_width)
        self.base_voice = np.array(base_voice.get_array_of_samples()).reshape(-1)/self.sample_range
        self.base_voice_silence = np.zeros_like(self.base_voice)


    def step(self,action:torch.Tensor) -> tuple[torch.Tensor,float,bool,None]:
        """step to next state and return infomations.
        action: (3,) [pitch, power, duration] -1 ~ 1
        return -> (src_spect, generated_spect), reward, done
        """
        assert action.size(0) == self.action_space_size
        pitch,power,duration = action.cpu().detach().numpy()
        pitch = self.get_n_shift(pitch)
        power = self.get_power(power)
        duration = self.get_duration(duration)
        
        wave = self.generate_wave(pitch,power,duration)
        self.generated_wave = np.concatenate([self.generated_wave,wave])
        wave = torch.from_numpy(self.generated_wave).to(self.device,self.dtype)
        gened_spect = self.mel_spector(wave).log1p().T
        reward = self.get_reward(gened_spect,self.generated_spect_len)

        self.generated_spect_len = gened_spect.size(0)
        gened_spect = torch.cat([gened_spect, self.generated_pad_spect[self.generated_spect_len:] ] ,dim=0)

        next_state = (self.source_spect.T,gened_spect.T)
        done = self.generated_spect_len >= self.source_spect.size(0)

        return next_state,reward,done,None
        
    def get_reward(self, generated_spect:torch.Tensor,previous_length:int) -> float:
        """return is -1 * mse 
        generated_spect: (time step, channels)
        """
        gl = generated_spect.size(0)
        tgt = self.source_spect[previous_length:gl]
        g= generated_spect[previous_length:]
        return -F.mse_loss(g,tgt,reduction="sum")
    
    def get_duration(self,out_duration:float) -> int:
        """convert output duration to real duration"""
        out_duration = int((out_duration + 1) / 2 * self.max_duration)
        out_duration = np.clip(out_duration,self.min_duration,self.max_duration)
        return out_duration

    def get_power(self, out_power:float) -> float:
        return np.clip(out_power,0.0,self.power_range)        

    def get_n_shift(self, out_pitch:float) -> float:
        return out_pitch * self.pitch_shift_range

    def generate_wave(self, n_shift:float, power:float, duration:int) -> np.ndarray:
        """cutting, shifting, and adjust the power"""
        if power == 0.0:
            return self.base_voice_silence[:duration]
        else:
            wave = self.base_voice[:duration]
            wave = librosa.effects.pitch_shift(wave,self.frame_rate,n_shift)
            max_p = np.max(np.abs(wave))
            wave = (wave/max_p) * power
            return wave

    def sample_action(self) -> torch.Tensor:
        """return random sampled action values"""
        action = torch.randn(self.action_space_size,device=self.device,dtype=self.dtype)
        return action

    def get_generated_wave(self) -> np.ndarray:
        return self.generate_wave

    def get_observation_space_size(self) -> tuple:
        s = (self.num_mels,self.spect_len)
        return s,s

    def save_generated_wave(self, out_path:str) -> None:
        wave = np.round(self.generated_wave.copy() * self.sample_range).astype(np.int16)
        wavfile.write(out_path,self.frame_rate,wave)

    def seed(self, seed:int) -> None:
        """ re-set seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)




if __name__ == '__main__':
    from utils import load_config
    config = load_config("hparams/origin.json")
    env = OshaberiEnv(config,"data/kiritan2021-12-07_20-40-44.csv",False)
    ns,r,d,_ = env.step(torch.ones(3))
    env.reset()
