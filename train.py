#%%
from torch.utils.tensorboard import SummaryWriter
from algorithm import SAC
from environment import OshaberiEnv
from trainer import Trainer
from parsers import get_training_parser
from utils import load_config,from_precision

#%%
if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    config_file = args.config_file
    dataset_file = args.dataset
    on_memory = args.on_memory
    device = args.device
    buf_device = args.buf_device
    precision = args.precision
    num_steps = args.num_steps
    eval_interval = args.eval_interval
    dtype = from_precision(precision)
    num_eval_episode = args.num_eval_episode
    log_dir = args.logdir
    logname = args.logname
    log_writer = SummaryWriter()
    
    config = load_config(config_file)
    
    env = OshaberiEnv(config,dataset_file,on_memory,device,dtype)
    env_test = OshaberiEnv(config,dataset_file,on_memory,device,dtype)

    algo = SAC(config,device,dtype,buf_device,dtype,log_writer)
    trainer = Trainer(config,env,env_test, algo,num_steps,eval_interval,num_eval_episode,log_writer)

    try:
        trainer.train()
    except:
        print("Key board interrupt.")
    algo.save_model()