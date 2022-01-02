from abc import ABC,abstractmethod
import torch
from agent_models import SACActor,SACCritic
from utils import AttrDict,get_stft_outlen
import numpy as np
from environment import OshaberiEnv

class ReplayBuffer(object):

    def __init__(self, config:AttrDict,device:torch.device="cpu",dtype:torch.dtype=torch.float) -> None:
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = config.replay_size
        self.state_shape = (config.num_mels,get_stft_outlen(config.breath_len,config.hop_len))
        

        self.device = device
        self.dtype = dtype
        # 保存するデータ

        self.src_states = torch.empty((self.buffer_size,*self.state_shape),dtype=dtype,device=device)
        self.gened_states = self.src_states.clone()
        self.actions = torch.empty((self.buffer_size,config.action_space_size),dtype=dtype,device=device)
        self.rewards = torch.empty((self.buffer_size,1),dtype=torch.double,device=device)
        self.dones = torch.empty((self.buffer_size,1),dtype=torch.float,device=device)
        self.next_src_states = self.src_states.clone()
        self.next_gened_states = self.src_states.clone()

    def append(self, state:tuple[torch.Tensor], action:torch.Tensor, reward:float, done:bool, next_state:tuple[torch.Tensor]) -> None:
        self.src_states[self._p].copy_(state[0])
        self.gened_states[self._p].copy_(state[1])
        self.actions[self._p].copy_(action)
        self.rewards[self._p] = reward
        self.dones[self._p] = done
        self.next_src_states[self._p].copy_(next_state[0])
        self.next_gened_states[self._p].copy_(next_state[1])

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size:int) -> tuple[torch.Tensor]:
        idxes = np.random.randint(0,self._n,batch_size)
        return (
            (self.src_states[idxes],self.gened_states[idxes]),
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            (self.next_src_states[idxes],self.next_gened_states[idxes])
        )


class Algorithm(ABC):
    actor:SACActor
    critic:SACCritic
    critic_target:SACCritic

    device:torch.device
    dtype:torch.dtype

    def explore(self, state:tuple[torch.Tensor]) -> tuple[torch.Tensor, float]:
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action, log_pi.item()

    def exploit(self, state:tuple[torch.Tensor]) -> torch.Tensor:
        """ 決定論的な行動を返す． """
        with torch.no_grad():
            action = self.actor(state)
        return action

    @abstractmethod
    def is_update(self, steps):
        """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
        """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass

    def reset_seed(self,seed:int = 0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class SAC(Algorithm):
    def __init__(self, config:AttrDict,device="cuda",dtype=torch.float,buf_device='cpu',buf_dtype=torch.float):
        super().__init__()

        self.seed = config.seed 
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic

        self.learning_steps = 0
        self.batch_size = config.batch_size
        self.device=device
        self.dtype=dtype
        self.buf_device=buf_device
        self.buf_dtype=buf_dtype
        self.gamma = config.gamma
        self.start_steps= config.start_steps
        self.tau = config.tau
        self.alpha = config.alpha
        self.reward_scale = config.reward_scale

        # set seed
        self.reset_seed(self.seed)

        # replay buffer
        self.buffer = ReplayBuffer(config,buf_device,buf_dtype)
        
        # configure Actor-Critic network
        self.actor = SACActor(config).to(self.device,self.dtype)
        self.critic = SACCritic(config).to(self.device,self.dtype)
        self.critic_target = SACCritic(config).to(self.device,self.dtype)

        # initialize target network and requires_grad=False
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)


    def is_update(self, steps:int) -> bool:
        return steps >= max(self.start_steps,self.batch_size)

    def step(self, env:OshaberiEnv,state:tuple[torch.Tensor], t:int, steps:int):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.sample_action()
        else:
            action,_ = self.explore(state)

        next_state, reward, done ,_ = env.step(action)

        self.buffer.append(state,action,reward,done,next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        states,actions,rewards,dones,next_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states:tuple[torch.Tensor], actions:torch.Tensor,rewards:torch.Tensor, dones:torch.Tensor, next_states:tuple[torch.Tensor]):
        curr_qs1, curr_qs2 = self.critic(states,actions)

        with torch.no_grad():
            next_actions, log_pis= self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states,next_actions)
            next_qs = torch.min(next_qs1,next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(return_graph=False)
        self.optim_critic.step()

    def update_actor(self, states:tuple[torch.Tensor]) -> None:
        actions, log_pis= self.actor.sample(states)
        qs1,qs2 = self.critic(states,actions)
        loss_actor = (self.alpha*log_pis - torch.min(qs1, qs2)).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)