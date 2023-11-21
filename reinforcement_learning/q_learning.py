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


class VideoRecorder:
    def __init__(self, filename, fps=30, record=True):
        self.record = record
        if record:
            self.filename = filename
            self.writer = imageio.get_writer(filename, fps=fps)

    def record_frame(self, env):
        if not self.record:
            return
        frame = env.render()
        self.writer.append_data(frame)

    def close(self, *args, **kwargs):
        if not self.record:
            return
        self.writer.close(*args, **kwargs)

    def play(self):
        if not self.record:
            return
        # Used for colab display
        self.close()
        mp4list = glob.glob(self.filename)
        assert len(mp4list) > 0
        mp4 = mp4list[0]
        video = io.open(mp4, "r+b").read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(
            HTML(
                data="""<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>""".format(
                    encoded.decode("ascii")
                )
            )
        )


class Bucketizer:
    def __init__(self, space, n_buckets):
        self.space = space
        self.lows = space.low
        self.highs = space.high
        self.shape = space.shape[0]
        assert len(n_buckets) == self.shape
        self.n_buckets = n_buckets
        self.bucket_sizes = [
            (self.highs[i] - self.lows[i]) / self.n_buckets[i]
            for i in range(self.shape)
        ]

    def __call__(self, obs):
        assert len(obs) == self.shape
        return tuple(
            math.floor((obs[i] - self.lows[i]) / self.bucket_sizes[i])
            for i in range(self.shape)
        )

    def all_buckets(self):
        values = [list(range(n_bucket + 1)) for n_bucket in self.n_buckets]
        return tuple(itertools.product(*values))


@dataclass
class Config:
    n_episode = 1000000
    record_every_n = 10000
    n_eval_episode = 100
    eval_every_n = 10000
    lr = 0.1
    discount = 0.95
    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    n_buckets = [50, 50]


def policy(env, q_table, bkt_state, eps=0.0):
    rand = np.random.uniform(0.0, 1.0)
    if rand >= eps:
        return np.argmax(q_table[bkt_state])
    else:
        return env.action_space.sample()


def eval(env, q_table, cfg, bkt):
    rewards = []
    for episod in range(cfg.n_eval_episode):
        state, info = env.reset()
        state = bkt(state)
        tot_reward = 0.0
        while True:
            action = policy(env, q_table, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = bkt(new_state)
            tot_reward += reward
            if terminated or truncated:
                break
            state = new_state
        rewards.append(tot_reward)
    return np.mean(rewards), np.std(rewards)


def train(env, q_table, cfg, bkt):
    recorders = []
    first_hit = None
    eval_res = {}
    for episode in tqdm(range(cfg.n_episode)):
        if (episode + 1) % cfg.eval_every_n == 0:
            m, s = eval(env, q_table, cfg, bkt)
            eval_res[episode] = (m, s)
        eps = cfg.min_eps + (cfg.max_eps - cfg.min_eps) * np.exp(
            -cfg.decay_rate * episode
        )
        state, info = env.reset()
        state = bkt(state)
        recorder = VideoRecorder(
            filename=f"{episode+1}.mp4",
            record=((episode + 1) % cfg.record_every_n == 0),
        )
        recorder.record_frame(env)
        while True:
            action = policy(env, q_table, state, eps)
            new_state, reward, terminated, truncated, info = env.step(action)
            recorder.record_frame(env)
            if not first_hit and new_state[0] >= env.unwrapped.goal_position:
                first_hit = episode
            new_state = bkt(new_state)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            q_table[state][action] = (
                (1 - cfg.lr) * q_table[state][action]
                # Instead of new_bkt_state, which might be the result of a exploration, the best new state should be used?
                + cfg.lr * (reward + cfg.discount * np.max(q_table[new_state]))
            )
            if terminated or truncated:
                break
            state = new_state
        recorder.close()
        if recorder.record:
            recorders.append(recorder)
    return q_table, recorders, first_hit, eval_res


def main():
    if not os.environ.get("DISPLAY"):
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    cfg = Config()
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    bkt = Bucketizer(env.observation_space, cfg.n_buckets)
    q_table = {}
    for bucket in bkt.all_buckets():
        q_table[bucket] = np.zeros(env.action_space.n)

    q_table, recorders, first_hit, eval_res = train(env, q_table, cfg, bkt)
    print(f"First hit at {first_hit}")
    for k, v in eval_res.items():
        print(f"At episode {k+1} mean: {v[0]} std: {v[1]}")


if __name__ == "__main__":
    main()
