import json
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