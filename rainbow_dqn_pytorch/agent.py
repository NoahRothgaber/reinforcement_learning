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
from prioritized_replay_buffer import PrioritizedReplayBuffer

# Currently Utilizing Double DQN, Dueling DQN, PERB (Prioritized Experience Replay Buffer), noise to be added next. 

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        #Forward method implementation
        # Pass input through the common feature layer
        feature = self.feature_layer(x)
        
        # Pass the feature through the value stream
        value = self.value_layer(feature)
        # Pass the feature through the advantage stream
        advantage = self.advantage_layer(feature)

        # Combine value and advantage to compute Q-values
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q

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
        self.alpha = hyperparameters['alpha']  # Default alpha value for prioritized sampling
        self.beta = hyperparameters['beta']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.rewards_per_episode = []
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        # For printing date and time
        self.DATE_FORMAT = "%m-%d %H:%M:%S"
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        self.rewards_per_episode = []
        self.policy_dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes).to(device)

        # Directory for saving run info
        self.RUNS_DIR = "runs"
        os.makedirs(self.RUNS_DIR, exist_ok=True)

        self.implementation = "DDQN w/ PERB and Dueling DQN"
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
            memory = PrioritizedReplayBuffer(self.num_states, self.replay_memory_size, self.mini_batch_size, self.alpha)
            target_dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
            self.epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            self.policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            self.policy_dqn.eval()

        for episode in itertools.count():
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = self.env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = self.policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = self.env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.store(state.cpu().numpy(), action.item(), reward.item(), new_state.cpu().numpy(), terminated)
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

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample_batch(self.beta)
                    self.optimize(mini_batch, self.policy_dqn, target_dqn, memory)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    self.epsilon_history.append(epsilon)
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        step_count = 0    
            else:
                log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: This Episode Reward: {episode_reward:0.1f}"
                print(log_message)

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

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon Decay', color='tab:red')
        ax2.plot(epsilon_history, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

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
        torch.save(self.policy_dqn.state_dict(), f"{self.RUNS_DIR}/{self.hyperparameter_set}_actor.pth")
    
    def optimize(self, mini_batch, policy_dqn, target_dqn, memory):
        states = torch.tensor(mini_batch['obs'], dtype=torch.float, device=device)
        actions = torch.tensor(mini_batch['acts'], dtype=torch.int64, device=device)
        rewards = torch.tensor(mini_batch['rews'], dtype=torch.float, device=device)
        new_states = torch.tensor(mini_batch['next_obs'], dtype=torch.float, device=device)
        terminations = torch.tensor(mini_batch['done'], dtype=torch.float, device=device)
        weights = torch.tensor(mini_batch['weights'], dtype=torch.float, device=device)

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
