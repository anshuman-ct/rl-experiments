import torch
import torch.nn.functional as F
import random
import math

import config
from model.dqn import DQN
from utils.replay_buffer import ReplayMemory

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN().to(config.DEVICE)
        self.target_net = DQN().to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.memory = ReplayMemory(config.MEMORY_SIZE)

        self.steps_done = 0

    def select_action(self, state):
        eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
              math.exp(-1. * self.steps_done / config.EPS_DECAY)

        self.steps_done += 1

        if random.random() < eps:
            return random.randint(0, 3)

        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).to(config.DEVICE)
            return self.policy_net(state).argmax().item()

    def optimize(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        batch = self.memory.sample(config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).to(config.DEVICE)
        actions = torch.tensor(actions).unsqueeze(1).to(config.DEVICE)
        rewards = torch.tensor(rewards).to(config.DEVICE)
        next_states = torch.tensor(next_states).to(config.DEVICE)
        dones = torch.tensor(dones).float().to(config.DEVICE)

        q = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + config.GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())