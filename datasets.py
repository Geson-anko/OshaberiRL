from torch.utils import data as DataUtil
import pandas as pd
import numpy as np
from scipy.io import wavfile
import torch
from typing import Tuple

class RandomSample(DataUtil.Dataset):
    """This class provides random sampled batch from data set
    
    return -> (sound1 : 1024, sound2: 1024, answer: 1280) 
    """


    def __init__(self, config, dataset_csv_path:str, on_memory:bool = False, normalize=True, max_length:int=None) -> None:
        """ Load csv and prepare to sound loading."""
        data_csv = pd.read_csv(dataset_csv_path)
        length_order = np.array(data_csv["length"])
        self.overall_len = length_order[-1]
        self.length_order = length_order
        self.paths = data_csv["path"].tolist()

        self.sample_length = config.n_fft + config.hop_len
        if max_length is None:
            self._len = int(self.overall_len/self.sample_length)
        else:
            self._len = max_length

        self.on_memory = on_memory
        self.normalize=normalize
        self.sample_range = 2**(8*config.sample_width-1)

        if on_memory:
            self.sound_arrays = []
            for i in self.paths:
                out = wavfile.read(i)[-1]
                if normalize:
                    out = out / self.sample_range
                self.sound_arrays.append(out)
            
        self.config = config
        

    def __len__(self):
        return self._len
    
    def __getitem__(self, _:int) -> Tuple[torch.Tensor]:
        idx = np.random.randint(0, self.overall_len)
        FileIdx = np.searchsorted(self.length_order, idx)
        start_point = idx - self.length_order[FileIdx]
        sound = self.get_sound_data(start_point, FileIdx)
        return sound,
    
    def get_sound_data(self, start_point ,file_idx:str) -> np.ndarray:
        """loading sound file. if length is over, pad zeros."""
        end_point = start_point + self.sample_length
        if self.on_memory:
            out = self.sound_arrays[file_idx][start_point:end_point]
        else:
            out = wavfile.read(self.paths[file_idx])[1][start_point:end_point]
            if self.normalize:
                out = out / self.sample_range
                out = out.astype(float)
        if len(out.shape) == 1:
            out = out[:, None]

        pad_len = self.sample_length - len(out)
        if pad_len:
            pad = np.zeros((pad_len,out.shape[-1]), dtype=out.dtype)
            out = np.concatenate([out, pad],axis=0)
        out = torch.from_numpy(out).T
        return out

if __name__ == "__main__":
    from utils import load_config
    config = load_config("hparams/origin.json")
    ds = RandomSample(config, "data/kiritan2021-12-07_20-40-44.csv",True,True)
    print(len(ds))
    for i in ds[0]:
        print(i.shape)
            


