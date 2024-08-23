# Proximal Policy Optimization
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from collections import deque
from datetime import datetime, timedelta
import argparse
import itertools
import os
import torch.optim as optim
from typing import Deque, Dict, List, Tuple
from torch.nn.utils import clip_grad_norm_
import flappy_bird_gymnasium

# source for this implementation
# https://www.youtube.com/watch?v=hlv79rcHws0 User: Machine Learning with Phil

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arrange(0, n_states, self.batch_size)
        indices = np.arrange(n_states, dtype=np.int64)
        np.random.shuffle(indices) #random batch
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
        np.array(self.actions),\
        np.array(self.log_probs),\
        np.array(self.values),\
        np.array(self.rewards),\
        np.array(self.dones),\
        batches 
    
    def store_memory(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []        

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, state_dim, alpha, hidden_dims_1=256,
                 hidden_dims_2=256):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(RUNS_DIR, 'actor_torch_ppo')
        self.actor == nn.Sequential(
            nn.Linear(*state_dim, hidden_dims_1),
            nn.ReLU(),
            nn.Linear(hidden_dims_1, hidden_dims_2),
            nn.ReLu(),
            nn.Linear(hidden_dims_2, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        dist = self.actor(state)
        dist = nn.Categorical(dist)
        return dist
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, alpha, 
                 hidden_dims_1=256, hidden_dims_2=256):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(RUNS_DIR, 'critc_torch_ppo')
        self.critic = nn.Sequential(
             nn.Linear(*state_dim, hidden_dims_1),
             nn.ReLU(),
             nn.Linear(hidden_dims_1, hidden_dims_2),
             nn.ReLU(),
             nn.Linear(hidden_dims_2, 1)
        )

        self.optimizer = optim.Adam(self.parameters, lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPO_Agent:
    def __init__(self, train, endless, continue_training, render, use_gpu, hyperparameter_set):
        
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_gamma = hyperparameters['discount_factor_gamma']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.gae_lambda = hyperparameters['gae_lambda']
        self.policy_clip = hyperparameters['policy_clip']
        #n_step 
        self.n_step = hyperparameters['n_step']
        self.n_epochs = hyperparameters['n_epochs']
        # noisy layer
        self.alpha = hyperparameters['alpha']  # Default alpha value for prioritized sampling
        self.env_make_params = hyperparameters.get('env_make_params', {})    
        # Store additional parameters
        self.train = train
        self.endless = endless
        self.continue_training = continue_training
        self.render = render
        self.use_gpu = use_gpu
        self.RUNS_DIR = "runs"
        os.makedirs(self.RUNS_DIR, exist_ok=True)
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        self.actor = ActorNetwork(self.num_actions, self.num_states, self.alpha)
        self.critic = CriticNetwork(self.num_states, self.alpha)
        self.memory = PPOMemory(self.mini_batch_size)
    
    def remember(self, state, action, log_probs, values, reward, done):
        self.memory.store_memory(state, action, log_probs, values, reward, done)

    def save(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)
        torch.save(self.actor.state_dict(), f"{self.MODEL_FILE}_actor")
        torch.save(self.critic.state_dict(), f"{self.MODEL_FILE}_critic")
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--continue_training', help='Continue training mode', action='store_true')
    parser.add_argument('--render', help='Rendering mode', action='store_true')
    parser.add_argument('--use_gpu', help='Device mode', action='store_true')
    parser.add_argument('--endless', help='Endless mode', action='store_true')
    args = parser.parse_args()

    proximal_policy = PPO_Agent(args.train, args.endless, args.continue_training, args.render, args.use_gpu, hyperparameter_set=args.hyperparameters)
    proximal_policy.run(args.train, args.render)