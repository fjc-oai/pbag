# Reference
#   Huggingface Deep RL Course https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
#   vwxyzjn implementation https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
#
# TODO:
#   0. refactor
#   1. wandb
#   2. eval
#   3. save/load
#   4. hparam tuning
# !apt-get install swig cmake ffmpeg > /dev/null 2>&1
# !pip install gymnasium > /dev/null 2>&1
# !pip install gymnasium[box2d] > /dev/null 2>&1
# !apt-get install -y xvfb ffmpeg > /dev/null 2>&1
# !pip install pyvirtualdisplay > /dev/null 2>&1
# !pip install gymnasium[atari]
# !pip install gymnasium[accept-rom-license]

import gymnasium as gym
import os
import pyvirtualdisplay
import base64
import io
from IPython import display as ipythondisplay
import imageio
import glob
from IPython.display import HTML
from dataclasses import dataclass
import numpy as np
import itertools
from tqdm import tqdm
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import torch

assert torch.cuda.is_available()
d = torch.device("cuda")


@dataclass
class Config:
    n_episode = 100
    record_every_n = 50
    n_step = 10000
    train_start_at_step_n = 100
    train_every_step_n = 8
    sync_every_step_n = 64
    batch_size = 32
    lr = 0.1
    discount = 0.95
    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    replay_buffer_size = 1000
    eval_n_episode = 10
    eval_every_n = 50


def make_env(cfg):
    env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        "",
        episode_trigger=lambda x: (x + 1) % cfg.record_every_n == 0,
        name_prefix="arita",
    )
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


class QNetwork(torch.nn.Module):
    def __init__(self, n_action):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_action),
        )
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if hasattr(param, "weight"):
                torch.nn.init.xavier_uniform(param.weight)
            if hasattr(param, "bias"):
                param.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x / 255.0)


class ReplayBuffer:
    @dataclass
    class In:
        obs: torch.Tensor
        action: int
        next_obs: torch.Tensor
        reward: float
        done: float

    @dataclass
    class BatchedOut:
        obs: torch.Tensor
        action: torch.Tensor
        next_obs: torch.Tensor
        reward: torch.Tensor
        done: torch.Tensor

    def __init__(self, size):
        self.size = size
        self.queue = []

    def add(self, obs, action, new_obs, reward, termination, truncation, info):
        data = ReplayBuffer.In(
            obs, action, new_obs, reward, 1.0 if (termination or truncation) else 0.0
        )
        self.queue.append(data)
        if len(self.queue) > self.size:
            self.queue.pop(0)

    def sample(self, batch_size):
        idxs = torch.randperm(len(self.queue))[:batch_size]
        samples = [self.queue[idx] for idx in idxs]
        obs = torch.stack([torch.tensor(sample.obs) for sample in samples], dim=0)
        action = torch.tensor([sample.action for sample in samples])
        next_obs = torch.stack(
            [torch.tensor(sample.next_obs) for sample in samples], dim=0
        )
        reward = torch.tensor([sample.reward for sample in samples])
        done = torch.tensor([sample.done for sample in samples])
        return ReplayBuffer.BatchedOut(obs, action, next_obs, reward, done)


def train(env, m, target_m, opt, rb, cfg):
    eval_res = {}
    for episode in tqdm(range(cfg.n_episode)):
        obs, info = env.reset()
        for step in range(cfg.n_step):
            eps = cfg.min_eps + (cfg.max_eps - cfg.min_eps) * np.exp(
                -cfg.decay_rate * episode
            )
            rand = np.random.uniform(0.0, 1.0)
            if rand >= eps:
                x = torch.Tensor(obs).unsqueeze(0).to(d)
                logits = m(x)
                action = torch.argmax(logits).cpu().item()
            else:
                action = env.action_space.sample()
            next_obs, reward, termination, truncation, info = env.step(action)

            if termination or truncation:
                break

            rb.add(obs, action, next_obs, reward, termination, truncation, info)

            if step > cfg.train_start_at_step_n:
                if step % cfg.train_every_step_n == 0:
                    data = rb.sample(cfg.batch_size)
                    with torch.no_grad():
                        max_q, _ = target_m(data.next_obs.to(d)).max(dim=1)
                        y = data.reward.to(d) + cfg.discount * max_q * (
                            1.0 - data.done.to(d)
                        )
                    pred = m(data.obs.to(d)).gather(
                        dim=1, index=data.action.view(-1, 1).to(d)
                    ).squeeze(1)
                    loss = torch.nn.functional.mse_loss(pred, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if step % cfg.sync_every_step_n == 0:
                    target_m.load_state_dict(m.state_dict())
        print(f'At {episode} {loss=}')
        if (episode + 1) % cfg.eval_every_n == 0:
            rewards = []
            for eval_episode in range(cfg.eval_n_episode):
                reward_acc = 0
                obs, info = env.reset()
                while True:
                    x = torch.Tensor(obs).unsqueeze(0).to(d)
                    logits = m(x)
                    action = torch.argmax(logits).cpu().item()
                    next_obs, reward, termination, truncation, info = env.step(action)
                    reward_acc += reward
                    if termination or truncation:
                        rewards.append(reward_acc)
                        break
            eval_res[episode] = dict(
                mean= np.mean(rewards),
                std= np.std(rewards)
            )
    return eval_res



def main():
    cfg = Config()
    env = make_env(cfg)
    m = QNetwork(env.action_space.n).to(d)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    target_m = QNetwork(env.action_space.n).to(d)
    target_m.load_state_dict(m.state_dict())
    rb = ReplayBuffer(cfg.replay_buffer_size)
    eval_res = train(env, m, target_m, opt, rb, cfg)
    print(eval_res)
    env.close()


if __name__ == "__main__":
    main()
