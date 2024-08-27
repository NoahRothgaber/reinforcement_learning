import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

import random
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones)
    
    def get_all(self):
        """Retrieve all data from the buffer."""
        if len(self.buffer) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        states, actions, rewards, dones = zip(*self.buffer)
        
         # Ensure that tensors are moved to CPU before converting to NumPy arrays
        states = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in states]
        actions = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in actions]
        rewards = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in rewards]
        dones = [d.cpu().numpy() if isinstance(d, torch.Tensor) else d for d in dones]
        
       
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones)

    def size(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer.clear()

        
class A3CActorCritic(nn.module):
    def __init__(self, input_dims, n_actions, gamma, replay_buffer_size, hidden_layer1=128):
        super(A3CActorCritic, self).__init__()

        self.gamma = gamma

        self.policy_two = nn.Linear(*input_dims, hidden_layer1)
        self.value_two = nn.Linear(*input_dims, hidden_layer1)
        self.policy_one = nn.Linear(hidden_layer1, n_actions)
        self.value_one = nn.Linear(hidden_layer1, 1)

        self.memory = ReplayBuffer(replay_buffer_size)

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.memory.reset()


    def forward(self, state):
        policy_two = F.relu(self.policy_two(state))
        value_two = F.relu(self.value_two(state))

        policy_one = self.policy_one(policy_two)
        value_one = self.value_one(value_two)
        return policy_one, value_one
    
    def calc_R(self, done):
        states = torch.tensor(self.states, dtype=torch.Float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.Tensor(batch_return, dtype=torch.float)

        return batch_return
    
    def calc_loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        policy, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = torch.softmax(policy, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        policy, value = self.forward(state)
        probs = torch.softmax(policy, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy[0]
        
        return action