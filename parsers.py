import argparse

def get_preprocess_parser(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/kiritan")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--config_file",type=str, default="hparams/origin.json")
    return parser

def get_env_paraser(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if not parser:
        parser =argparse.ArgumentParser()
    parser.add_argument("--config_file",type=str, default="hparams/origin.json")
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--on_memory",type=bool,default=True)

def get_training_parser(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if not parser:
        parser =argparse.ArgumentParser()
    parser.add_argument("--config_file",type=str, default="hparams/origin.json")
    parser.add_argument("--dataset",type=str,default="data/kiritan2021-12-07_20-40-44.csv")
    parser.add_argument("--device",type=str, default="cuda")
    parser.add_argument("--precision",type=int,default=32)
    parser.add_argument("--buf_device",type=str,default="cuda")
    parser.add_argument("--num_steps",type=int,default=5 * 10 ** 4)
    parser.add_argument("--eval_interval",type=int,default=10**3)
    parser.add_argument("--num_eval_episode",type=int,default=3)
    parser.add_argument("--logdir",type=str,default="training_log")
    parser.add_argument("--logname",type=str,default="default")
    parser.add_argument("--on_memory",type=bool,default=True)
    parser.add_argument("--save_dir",type=str,default="parameters")
    return parser