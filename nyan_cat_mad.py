import mido
from utils import load_config
import librosa
import numpy as np
from scipy.io import wavfile
from typing import List,Tuple
from pydub import AudioSegment

# configs
config_file = "hparams/nyan_cat.json"

config = load_config(config_file)
note_A = config.a 

def noteToFreq(note):
    a = note_A #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

def get_f0freq(sound_array:np.ndarray,fps:int,search_range:int = 5) -> float:
    freq_ax=np.fft.rfftfreq(len(sound_array),1/fps)
    ffted = np.abs(np.fft.rfft(sound_array))
    freqs = freq_ax[np.argsort(ffted)[-search_range:]]
    return np.min(freqs)

def load_sound(sound_file:str) -> Tuple[np.ndarray,int]:
    """return is raw sound and fps"""
    sound = AudioSegment.from_file(sound_file)
    sound = sound.set_channels(1)
    width = sound.sample_width
    fps = sound.frame_rate
    sound = np.array(sound.get_array_of_samples()) / (2**(8*width -1))
    return sound,fps


def midi2freq_duration(midi_file:str) -> Tuple[List[float]]:
    midi_sound = mido.MidiFile(midi_file)
    freqs, durs = [],[]
    tempo = 60/120 # beat second
    for i,msg in enumerate(midi_sound.tracks[0]):
        if msg.is_meta:
            if msg.type == "set_tempo":
                t = msg.tempo
                tempo = t/10**6
            if msg.type == "end_of_track":
                durs.append(1) # no wait
        else:
            if msg.type != "note_on" and msg.type != "note_off":
                continue
            if msg.type == "note_on":
                if msg.velocity > 0:
                    f = noteToFreq(msg.note)
                    freqs.append(f)
                else:
                    freqs.append(0.0) # no sound
            elif msg.type == "note_off":
                freqs.append(0.0)
            
            t = msg.time
            t = t/10**3* 2*tempo
            durs.append(t)
    return freqs,durs[1:]

def duration2length(duration:float,fps:int) -> int:
    return int(fps*duration)

def get_nshift(origin:float,target:float) -> float:
    tgt =  12*np.log2(target/origin)
    
    return tgt

def get_MAD(sound_file:str,midi_file:str,out_path:str) -> np.ndarray:
    sound,fps = load_sound(sound_file)
    freqs,durs = midi2freq_duration(midi_file)
    f0= get_f0freq(sound,fps,search_range=config.f0_search_range)
    mad = []
    silence = np.zeros_like(sound)
    for i,(f,d) in enumerate(zip(freqs,durs)):
        no_shift = False
        if not f> 0.0:
            source = silence
            no_shift= True
        else:
            source = sound

        length = duration2length(d,fps)
        src = source[:length]
        if len(src) < length:
            pad = np.zeros(length-len(src),src.dtype)
            src = np.concatenate([src,pad])
        
        if no_shift:
            mad.append(src)
            continue
        else:
            shif = get_nshift(f0,f)
            shifted = librosa.effects.pitch_shift(src,fps,shif)
            mad.append(shifted)
    mad=  np.concatenate(mad)*(2**15)
    mad = mad.astype(np.int16)
    wavfile.write(out_path,fps,mad)
    return mad

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    midi_file = "data/nyan_cat_base.mid"
    #print(*midi2freq_duration(midi_file))
    base_voice_file = "data/sahya_g4.wav"
    mad = get_MAD(base_voice_file,midi_file,"data/nyan_mad_base.wav")
    plt.plot(mad)
    plt.show()



