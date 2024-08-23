import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import os
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, state_dim, alpha, hidden_dims_1=256, hidden_dims_2=256):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(RUNS_DIR, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims_1),
            nn.ReLU(),
            nn.Linear(hidden_dims_1, hidden_dims_2),
            nn.ReLU(),
            nn.Linear(hidden_dims_2, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        return Categorical(dist)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, alpha, hidden_dims_1=256, hidden_dims_2=256):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(RUNS_DIR, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dims_1),
            nn.ReLU(),
            nn.Linear(hidden_dims_1, hidden_dims_2),
            nn.ReLU(),
            nn.Linear(hidden_dims_2, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))