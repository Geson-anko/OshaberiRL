from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import os
import pandas as pd
from utils import load_config,get_now
import glob
from concurrent.futures import ProcessPoolExecutor

class RandomSampleDataset:
    """
    This class produce a base procedure of making the dataset.
    
    """
    def __init__(self, input_dir:str, output_dir:str, config):
        self.input_dir = input_dir
        self.input_dir_name = os.path.split(input_dir)[-1]
        self.output_dir = output_dir
        now = get_now()
        self.output_files_dir = os.path.join(output_dir, self.input_dir_name+now)
        self.info_csv_path = os.path.join(self.output_dir, self.input_dir_name+now+".csv")

        
        self.config = config
        self.set_config(config)

    def set_config(self,config):
        self.frame_rate = config.frame_rate
        self.sample_width = config.sample_width
        self.sample_ch = config.sample_ch

    def run(self):
        """run the defined preprocess"""
    
        if not os.path.exists(self.output_files_dir):
            os.makedirs(self.output_files_dir)
        else:
            FileExistsError("")
        files = self.load_files()
        with ProcessPoolExecutor(os.cpu_count()//2) as p:
            results = p.map(self.process, files) # (file_path, length)
        csv_list = []
        count = 0
        for r in results:
            count += r[1]
            csv_list.append([r[0], count])
        pd.DataFrame(csv_list,columns=("path", "length")).to_csv(self.info_csv_path, index=False)
        
    def load_files(self) -> list:
        """ load preprocessing file pathes"""
        files = glob.glob(os.path.join(self.input_dir, "*"))
        return files

    def process(self, sound_file:str) -> tuple:
        """preprocessing"""
        sound = AudioSegment.from_file(sound_file)
        if sound.channels != self.sample_ch:
            sound.set_channels(self.sample_ch)
        if sound.sample_width != self.sample_width:
            sound.set_sample_width(self.sample_width)
        if sound.frame_rate != self.frame_rate:
            sound.set_frame_rate(self.frame_rate)
        
        array = np.array(sound.get_array_of_samples())
        file_name = os.path.split(sound_file)[-1]
        file_names = file_name.split(".")
        file_names[-1] = ".wav"
        file_name = "".join(file_names)
        file_path = os.path.join(self.output_files_dir, file_name)
        
        wavfile.write(file_path,self.frame_rate, array)
        return file_path, len(array)

        
        


if __name__ == "__main__":
    from parsers import get_preprocess_parser
    parser = get_preprocess_parser()
    args = parser.parse_args()

    #debugging
    config = load_config(args.config_file)
    preprocessor = RandomSampleDataset(args.input_dir, args.output_dir, config)
    f = preprocessor.load_files()[0]
    preprocessor.run()