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
from prioritized_replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from noisy_linear_network import NoisyLinear
import torch.optim as optim
from typing import Deque, Dict, List, Tuple
from torch.nn.utils import clip_grad_norm_
import flappy_bird_gymnasium
# Currently Utilizing Double DQN, Dueling DQN, PERB (Prioritized Experience Replay Buffer), noise to be added. 

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, support, atom_size, hidden_dim=256):
        super(DQN, self).__init__()
        self.support = support
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.advantage_hidden_layer = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage_layer = NoisyLinear(hidden_dim, action_dim * atom_size)
        self.value_hidden_layer = NoisyLinear(hidden_dim, hidden_dim)
        self.value_layer = NoisyLinear(hidden_dim, atom_size)

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)        
        return q
    
    def dist(self, x):
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.action_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean (dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min = 1e-3)
        return dist

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# Deep Q-Learning Agent
class Agent():
    def __init__(self, train, endless, continue_training, render, use_gpu, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        
        #categorical DQN params
        self.v_min = hyperparameters['v_min']
        self.v_max = hyperparameters['v_max']
        self.atom_size = hyperparameters['atom_size']
        #n_step dqn
        self.n_step = hyperparameters['n_step']
        # noisy layer
        self.alpha = hyperparameters['alpha']  # Default alpha value for prioritized sampling
        self.beta = hyperparameters['beta']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.loss_fn = nn.MSELoss()
        self.rewards_per_episode = []
        # Directory for saving run info
        self.RUNS_DIR = "runs"
        os.makedirs(self.RUNS_DIR, exist_ok=True)
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        # For printing date and time
        self.DATE_FORMAT = "%m-%d %H:%M:%S"
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        self.rewards_per_episode = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # networks
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)
        self.policy_dqn = DQN(self.num_states, self.num_actions, atom_size=self.atom_size, support=self.support).to(self.device)
        self.target_dqn = DQN(self.num_states, self.num_actions, atom_size=self.atom_size, support=self.support).to(self.device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()
        #memory for one step
        self.memory = PrioritizedReplayBuffer(self.num_states, self.replay_memory_size, self.mini_batch_size, alpha=self.alpha, gamma=self.discount_factor_g)

        #  memory for N-step Learning
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.memory_n = ReplayBuffer(obs_dim=self.num_states, size=self.replay_memory_size, batch_size=self.mini_batch_size, n_step=self.n_step, gamma=self.discount_factor_g)

        self.optimizer = optim.Adam(self.policy_dqn.parameters())
        self.transition = list()

        self.implementation = "Full Rainbow DQN"
        self.graph_title = f'{self.implementation} using {self.env_id}'

        # Store additional parameters
        self.train = train
        self.endless = endless
        self.continue_training = continue_training
        self.render = render
        self.use_gpu = use_gpu

        if self.train:
            start_time = datetime.now()
            self.last_graph_update_time = start_time

    def select_action(self, state):
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.policy_dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.train:
            self.transition = [state, selected_action]
        
        return selected_action

    def update_model(self):
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
        samples["weights"].reshape(-1,1)
        ).to(self.device)
        indices = samples["indices"]
        
        elementwise_loss = self._compute_dqn_loss(samples, self.discount_factor_g)

        # N-step Learning loss
        if self.use_n_step:
            gamma = self.discount_factor_g ** self.n_step
            samples_n = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma)
            elementwise_loss += elementwise_loss_n

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    
    def run(self, is_training=True, render=True):
        best_average_reward = None
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        if is_training:
            epsilon = self.epsilon_init
            self.epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            self.policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            self.policy_dqn.eval()

        for episode in itertools.count():
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated and episode_reward < self.stop_on_reward:
                self.policy_dqn.reset_noise()   
                if is_training and random.random() < epsilon:
                    action = self.env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=self.device)
                else:
                    with torch.no_grad():
                        action = self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = self.env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if is_training:
                    transition = (state.cpu().numpy(), action.item(), reward.item(), new_state.cpu().numpy(), terminated)
                    
                    # Store the transition in the n-step buffer first
                    if self.use_n_step:
                        one_step_transition = self.memory_n.store(*transition)
                    else:
                        one_step_transition = transition
                    
                    # If the n-step transition is ready, store it in the main memory
                    if one_step_transition:
                        self.memory.store(*one_step_transition)
    
                    step_count += 1

                state = new_state

            self.rewards_per_episode.append(episode_reward)

            # if is_training:
            #     average_reward = np.mean(self.rewards_per_episode[-100:])
            #     if average_reward >= (self.stop_on_reward * 0.98):
            #         print(f'Average meets/exceeds best reward for environment {(self.stop_on_reward * 0.98)}... saving model...')

            #     if episode_reward > best_reward:
            #         log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
            #         print(log_message)
            #         with open(self.LOG_FILE, 'a') as file:
            #             file.write(log_message + '\n')
            #         # torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
            #         best_reward = episode_reward

            #     current_time = datetime.now()
            #     if current_time - last_graph_update_time > timedelta(seconds=10):
            #         self.save_graph(rewards_per_episode, epsilon_history)
            #         last_graph_update_time = current_time
            if is_training:
                current_time = datetime.now()
                if current_time - self.last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(self.rewards_per_episode, epsilon_history=self.epsilon_history)
                    self.last_graph_update_time = current_time
                
                average_reward = np.mean(self.rewards_per_episode[-100:])
                if average_reward >= (self.stop_on_reward * 0.98):
                    self.save()
                    log_message = f'Average meets/exceeds best reward for environment **{(self.stop_on_reward * 0.98)}**... saving model...'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')
                    self.save_graph(self.rewards_per_episode, epsilon_history=self.epsilon_history)        
                    exit()
                if (episode + 1) % 100 == 0:
                    if best_average_reward == None:
                        best_average_reward = average_reward
    
                    time_now = datetime.now()
                    log_message = f"{time_now.strftime(self.DATE_FORMAT)}: Average Reward over last 100 episodes: {average_reward:0.1f} at episode: {episode + 1}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    if average_reward >= best_average_reward:
                        best_average_reward = average_reward  # Update the best average reward
                        # Save model
                        self.save()
                        log_message = f"{time_now.strftime(self.DATE_FORMAT)}: New Best Average Reward: {best_average_reward:0.1f} at episode: {episode + 1}, saving model..."
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')

                if best_reward == None:
                    best_reward = episode_reward


                if episode_reward > best_reward and episode > 0:
                    log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: New Best Reward: {episode_reward:0.1f} ({abs((episode_reward-best_reward)/best_reward)*100:+.1f}%) at episode {episode}"
                    print(log_message)
                    best_reward = episode_reward

                if len(self.memory) > self.mini_batch_size:
                    mini_batch = self.memory.sample_batch(self.beta)
                    self.optimize(mini_batch, self.policy_dqn, self.target_dqn, self.memory)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    self.epsilon_history.append(epsilon)
                    if step_count > self.network_sync_rate:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        step_count = 0    
            else:
                log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: This Episode Reward: {episode_reward:0.1f}"
                print(log_message)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.policy_dqn(next_state).argmax(1)
            next_dist = self.target_dqn.dist(next_state)
            next_dist = next_dist[range(self.mini_batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.mini_batch_size - 1) * self.atom_size, self.mini_batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.mini_batch_size, self.atom_size)
                .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.policy_dqn.dist(state)
        log_p = torch.log(dist[range(self.mini_batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        plt.title(self.env_id)
        fig, ax1 = plt.subplots()

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        mean_total = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_total)):
            mean_total[x] = np.mean(rewards_per_episode[0:(x+1)])

        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Reward Last 100 Episodes', color='tab:blue')
        ax1.plot(mean_rewards, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a third y-axis
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.set_ylabel('Cumulative Mean Reward', color='tab:green')
        ax3.plot(mean_total, color='tab:green', linestyle='--')
        ax3.tick_params(axis='y', labelcolor='tab:green')

        # Make y axis 1 and 3 the same scale
        ax1.set_ylim([min(min(mean_rewards), min(mean_total)), max(max(mean_rewards), max(mean_total))])
        ax3.set_ylim(ax1.get_ylim())

        # Add a legend
        # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.title(self.graph_title)
        # Save the figure
        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

     # There is no functional difference between . pt and . pth when saving PyTorch models
    def save(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)
        torch.save(self.policy_dqn.state_dict(), f"{self.MODEL_FILE}")
    
    def optimize(self, mini_batch, policy_dqn, target_dqn, memory):
            states = torch.tensor(mini_batch['obs'], dtype=torch.float, device=self.device)
            actions = torch.tensor(mini_batch['acts'], dtype=torch.int64, device=self.device)
            rewards = torch.tensor(mini_batch['rews'], dtype=torch.float, device=self.device)
            new_states = torch.tensor(mini_batch['next_obs'], dtype=torch.float, device=self.device)
            terminations = torch.tensor(mini_batch['done'], dtype=torch.float, device=self.device)
            weights = torch.tensor(mini_batch['weights'], dtype=torch.float, device=self.device)

            with torch.no_grad():
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

            current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

            # Compute loss for the entire mini-batch
            loss = self.loss_fn(current_q, target_q)

            # Apply importance sampling weight
            weighted_loss = weights * loss  # Element-wise multiplication with weights
            mean_weighted_loss = weighted_loss.mean()  # Mean across the batch
            self.optimizer.zero_grad()
            mean_weighted_loss.backward()  # Backpropagate the mean weighted loss
            self.optimizer.step()

            # Update priorities for the sampled transitions
            priorities = (weighted_loss.detach().cpu().numpy() + 1e-5)  # Avoid zero priority
            mini_batch_indices = mini_batch['indices']
            memory.update_priorities(mini_batch_indices, priorities)

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

    rainbow_dqn = Agent(args.train, args.endless, args.continue_training, args.render, args.use_gpu, hyperparameter_set=args.hyperparameters)
    rainbow_dqn.run(args.train, args.render)
