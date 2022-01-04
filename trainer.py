from torch.utils.tensorboard import SummaryWriter
from utils import AttrDict,get_now
from environment import OshaberiEnv
from algorithm import SAC
import time
import torch
import numpy as np
from datetime import timedelta

class Trainer:
    def __init__(
        self,config:AttrDict, env:OshaberiEnv, env_test:OshaberiEnv,algo:SAC,
        num_steps:int, eval_interval:int, num_eval_episodes:int=3,
        log_writer:SummaryWriter = None,log_interval:int=16
        ) -> None:
        self.h = config
        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(config.seed)
        self.env_test.seed(2**31-config.seed)

        # tensorboard
        if log_writer is None:
            log_writer = SummaryWriter()
        self.log_writer = log_writer

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes
        # Tensorboard に記録するインターバル
        self.log_interval = log_interval
        self.training_config = {
            "num_steps":num_steps,
            "eval_interval":eval_interval,
            "num_eval_episodes":num_eval_episodes
            }
        print(f"Training settings\n{self.training_config}")
        self.h.update(self.training_config)

    def train(self):
        print("Training Start!")
        self.log_writer.add_hparams(self.h,{})
        self.start_time = time.time()
        t = 0

        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            state,t,r,mr = self.algo.step(self.env,state,t,steps)

            if self.algo.is_update(steps):
                self.algo.update()

            if steps % self.eval_interval == 0:
                self.evaluate(steps)

            if steps% self.log_interval:
                self.log_writer.add_scalar("reward",r,steps)
                self.log_writer.add_scalar("MSE reward",mr,steps)

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """
        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, mean_reward = self.env_test.step(action)
                episode_return += reward
            returns.append(episode_return)
        mean_return = np.mean(returns)
        self.log_writer.add_scalar("evaluate mean return",mean_return,steps)
        self.log_writer.add_audio("generated audio",torch.from_numpy(self.env_test.get_generated_wave()),steps,sample_rate=self.h.frame_rate)
        self.algo.save_checkpoint(self.log_writer.get_logdir())

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')
    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time.time() - self.start_time)))