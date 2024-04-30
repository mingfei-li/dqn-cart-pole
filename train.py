from config import Config
from collections import deque
from gymnasium.experimental.wrappers import RecordVideoV0
from mlp_model import MLPModel
from logger import Logger
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm

class Agent():
    def __init__(self, env, config: Config):
        self.env = env
        self.config = config
        self.eps = config.max_eps
        self.eps_step = (config.max_eps - config.min_eps) / config.n_eps
        self.lr = config.max_lr
        self.lr_step = (config.max_lr - config.min_lr) / config.n_lr

        self.policy_model = MLPModel(
            in_features=self.env.observation_space.shape[0],
            out_features=self.env.action_space.n,
        )
        self.target_model = MLPModel(
            in_features=self.env.observation_space.shape[0],
            out_features=self.env.action_space.n,
        )
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.logger = Logger()
        self.t = 0

    def train(self):
        for i in tqdm(range(self.config.num_episodes), desc="Episode"):
            total_reward = 0
            obs, _ = env.reset()
            done = False
            while not done:
                if random.random() < self.eps or self.t < self.config.learning_start:
                    action = env.action_space.sample()
                else:
                    self.policy_model.eval()
                    states = torch.unsqueeze(torch.tensor(obs), dim=0)
                    with torch.no_grad():
                        q = self.policy_model(states)[0]
                    action = torch.argmax(q, dim=0).item()
                    self.logger.add_scalar("q_a", q[action].item())

                new_obs, reward, done, *_ = env.step(action)
                self.replay_buffer.append([obs, action, reward, new_obs, done])

                if self.t >= self.config.learning_start:
                    self.train_step()

                self.eps = max(self.eps - self.eps_step, self.config.min_eps)
                self.lr = max(self.lr - self.lr_step, self.config.min_lr)
                self.logger.add_scalar("eps", self.eps)
                self.logger.add_scalar("lr", self.lr)

                total_reward += reward
                obs = new_obs
                self.t += 1

            self.logger.add_scalar("total_reward", total_reward)
            self.logger.flush(self.t)

    def train_step(self):
        states, actions, rewards, next_states, dones = self.sample()

        self.target_model.eval()
        with torch.no_grad():
            tq = self.target_model(next_states)
        tq_max, _ = torch.max(tq, dim=1) 
        tq_max *= 1 - dones.int()
        targets = rewards + self.config.gamma * tq_max

        self.policy_model.train()
        q = self.policy_model(states)
        q_a = torch.gather(q, 1, actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = nn.MSELoss()(q_a, targets)
        optimizer = torch.optim.Adam(
            params=self.policy_model.parameters(),
            lr=self.lr,
        )
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.policy_model.parameters(),
        #     max_norm=self.config.grad_clip,
        # )
        optimizer.step()

        if self.t % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        grad_norm = 0
        for p in self.policy_model.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.logger.add_scalar("grad_norm", grad_norm)
        self.logger.add_scalar("loss", loss.item())
        
    def sample(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        experiences = random.sample(self.replay_buffer, self.config.batch_size)
        for exp in experiences:
            state, action, reward, next_state, done = exp
            states.append(torch.tensor(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.tensor(next_state))
            dones.append(done)

        return [
            torch.stack(states, dim=0),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states, dim=0),
            torch.tensor(dones),
        ]

if __name__ == "__main__":
    env = gym.make('CartPole-v0', render_mode="rgb_array")
    env = RecordVideoV0(
        env,
        video_folder="results/videos",
        step_trigger=lambda t: t % 10000 == 0,
    )
    agent = Agent(env, Config())
    agent.train()