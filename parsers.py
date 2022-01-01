import argparse

def get_preprocess_parser(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/kiritan")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--config_file",type=str, default="hparams/origin.json")
    return parser

def get_training_parser(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if not parser:
        parser =argparse.ArgumentParser()
    parser.add_argument("--config_file",type=str, default="hparams/origin.json")
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--logdir",type=str,default="training_log")
    parser.add_argument("--logname",type=str,default="default")
    parser.add_argument("--on_memory",type=bool,default=True)
    parser.add_argument("--num_gpus",type=int,default=1)
    parser.add_argument("--num_nodes",type=int,default=1)
    parser.add_argument("--precision",type=int,default=16)
    parser.add_argument("--save_name",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default="parameters")
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--max_length",type=int,default=None)
    parser.add_argument("--view_interval",type=int,default=10)
    parser.add_argument("--log_every_n_steps",type=int,default=64)
    parser.add_argument("--gradient_clip_val",type=float,default=0.0)
    return parser