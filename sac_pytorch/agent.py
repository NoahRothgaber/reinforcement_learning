import gymnasium as gym

import numpy as np
from collections import deque

import matplotlib
import matplotlib.pyplot as plt

import random
import itertools

import torch
from torch import nn
import torch.nn.functional as F

import argparse
import yaml

from datetime import datetime, timedelta
import os

from sac import SAC_Actor, SAC_Critic, SAC_Entropy

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

# SAC Agent
class Agent():

    def __init__(self, is_training, endless, continue_training, render, use_gpu, hyperparameter_set):
        with open(os.path.join(os.getcwd(), 'hyperparameters.yml'), 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id                 = hyperparameters['env_id']
        self.input_model_name       = hyperparameters['input_model_name']
        self.output_model_name      = hyperparameters['output_model_name']
        self.state_dim              = hyperparameters['state_dim']
        self.action_dim             = hyperparameters['action_dim']
        self.action_low             = hyperparameters['action_low']
        self.action_high            = hyperparameters['action_high']
        self.replay_memory_size     = hyperparameters['replay_memory_size']         # size of replay memory
        self.batch_size             = hyperparameters['batch_size']            # size of the training data set sampled from the replay memory
        self.discount               = hyperparameters['discount']
        self.tau                    = hyperparameters['tau']
        self.learning_rate          = hyperparameters['learning_rate']
        self.policy_noise           = hyperparameters['policy_noise']
        self.noise_clip             = hyperparameters['noise_clip']
        self.policy_freq            = hyperparameters['policy_freq']
        self.model_save_freq        = hyperparameters['model_save_freq']
        self.max_reward             = hyperparameters['max_reward']
        self.max_timestep           = hyperparameters['max_timestep']
        self.max_episodes           = hyperparameters['max_episodes']
        self.alpha                  = hyperparameters['alpha']
        self.entropy_coefficient    = hyperparameters['entropy_coefficient'] 
        self.minimum_entropy        = hyperparameters['minimum_entropy']
        self.entropy_decay          = hyperparameters['entropy_decay']
        self.env_make_params        = hyperparameters.get('env_make_params',{})     # Get optional environment-specific parameters, default to empty dict

        if self.input_model_name == None:
            self.input_model_name = hyperparameter_set
        if self.output_model_name == None:
            self.output_model_name = hyperparameter_set

        # Path to Run info, create if does not exist
        self.RUNS_DIR = "runs"
        self.INPUT_FILENAME = self.input_model_name
        self.OUTPUT_FILENAME = self.output_model_name
        os.makedirs(self.RUNS_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(self.RUNS_DIR, f'{self.INPUT_FILENAME}.log')
        self.GRAPH_FILE = os.path.join(self.RUNS_DIR, f'{self.OUTPUT_FILENAME}.png')
        self.DATE_FORMAT = "%m-%d %H:%M:%S"

        # Set device based on device arg
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # set endless mode if endless arg is true, otherwise set max episodes based on parameters 
        if endless or not is_training:
            self.max_episodes = itertools.count()
        else:
            self.max_episodes = range(self.max_episodes)

        # Create instance of the environment.
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions & observation space size
        self.num_actions = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # Target Entropy 
        self.target_entropy = -np.prod(self.entropy_coefficient) 

        # List to keep track of rewards collected per episode.
        self.rewards_per_episode = []
        self.total_it = 0
        self.target_entropies = []  # List to track target entropy values

        # Create actor and critic networks
        self.actor = SAC_Actor(self.num_states, self.num_actions, use_gpu, self.action_low, self.action_high).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = SAC_Critic(self.num_states, self.num_actions, use_gpu).to(self.device)
        self.critic_target = SAC_Critic(self.num_states, self.num_actions, use_gpu).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Create Entropy network
        self.entropy = SAC_Entropy(self.num_states, self.num_actions, use_gpu).to(self.device)
        self.entropy_optimizer = torch.optim.Adam(self.entropy.parameters(), lr=self.learning_rate)

        # Initialize alpha for entropy term (automatic tuning)
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(self.replay_memory_size)
        
        if is_training or continue_training:
            # Initialize log file
            start_time = datetime.now()
            self.last_graph_update_time = start_time

            log_message = f"{start_time.strftime(self.DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
            
            if continue_training:
                self.load()

        # if we are not training, generate the actor and critic policies based on the saved model
        else:
            self.load()
            self.actor.eval()
            self.critic.eval()
            start_time = datetime.now()
            log_message = f"{start_time.strftime(self.DATE_FORMAT)}: Run starting..."
            print(log_message)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # at testing REMOVE STOCHASTICITY
        action, log_prob = self.actor.sample(state)  # Use the sample method to get action and log probability
        return action.detach().cpu().numpy().flatten(), log_prob


    def train(self):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update the critic networks
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_action)
            next_v = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.discount * next_v

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        action, log_prob = self.actor.sample(states)
        q1, q2 = self.critic(states, action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the entropy coefficient network (alpha)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Increment iteration counter
        self.total_it += 1

    def run(self, is_training=True, continue_training=False):

        best_reward = None # Used to track best reward
        best_average_reward = None

        for episode in self.max_episodes:

            state, _ = self.env.reset()  # Initialize environment. Reset returns (state,info).
            terminated = False      # True when agent reaches goal or fails
            truncated = False       # True when max_timestep is reached
            episode_reward = 0.0    # Used to accumulate rewards per episode
            step_count = 0          # Used for syncing policy => target network

            if not is_training or continue_training:
                self.load()

            while(not terminated and not truncated and not step_count == self.max_timestep):
               # Action selection
                action, _ = self.select_action(state)  # Sample action from the actor

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminated = step_count == self.max_timestep - 1 or terminated

                if is_training or continue_training:
                    self.replay_buffer.add(state, action, reward, next_state, terminated)

                    if self.replay_buffer.size() > self.batch_size and step_count % self.policy_freq == 0:
                        # Train the agent
                        self.train()

                state = next_state
                episode_reward += reward
                step_count += 1
                
            # Keep track of the rewards collected per episode and save model
            self.rewards_per_episode.append(episode_reward)

            if is_training or continue_training:
                current_time = datetime.now()
                if current_time - self.last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(self.rewards_per_episode)
                    self.last_graph_update_time = current_time

                if (episode + 1) % 100 == 0:
                    average_reward = np.mean(self.rewards_per_episode[-100:])
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
            else:
                log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: This Episode Reward: {episode_reward:0.1f}"
                print(log_message)
            # decay entropy target
            self.target_entropies.append(self.target_entropy)

            if self.target_entropy <= -self.minimum_entropy:
                self.target_entropy = self.target_entropy * self.entropy_decay
                #print(f'target_entropy: {self.target_entropy}')

    # There is no functional difference between . pt and . pth when saving PyTorch models
    def save(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)
        torch.save(self.actor.state_dict(), f"{self.RUNS_DIR}/{self.OUTPUT_FILENAME}_actor.pth")
        torch.save(self.critic.state_dict(), f"{self.RUNS_DIR}/{self.OUTPUT_FILENAME}_critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load(f"{self.RUNS_DIR}/{self.INPUT_FILENAME}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{self.RUNS_DIR}/{self.INPUT_FILENAME}_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())
    

    def save_graph(self, rewards_per_episode):
        # Save plots
        fig, ax1 = plt.subplots()

        plt.title(f'{self.env_id}')

        # Plot average rewards per last 100 episodes , and the cumulative mean over all episodes (Y-axis) vs episodes (X-axis)
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
        ax2.set_ylabel('Cumulative Mean Reward', color='tab:green')
        ax2.plot(mean_total, color='tab:green', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Plot the target entropy on the same graph
        ax3 = ax1.twinx()  # Create a third y-axis
        ax3.set_ylabel('Target Entropy', color='tab:red')
        ax3.plot(self.target_entropies, color='tab:red', linestyle='-.', alpha=0.7)
        ax3.tick_params(axis='y', labelcolor='tab:red')
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis to the right
        
        # Set the same scale for all y-axes
        ax1.set_ylim([min(min(mean_rewards), min(mean_total)), max(max(mean_rewards), max(mean_total))])
        ax2.set_ylim(ax1.get_ylim())
        ax3.set_ylim(min(self.target_entropies), max(self.target_entropies))  # Adjust scale for entropy

        # Adjust layout to create more space for the text box
        fig.tight_layout(rect=[0, 0.26, 1, 1])  # Adjust rect to leave more space at the bottom

        # Add text box below the graph
        text = (
            f'env_id: {self.env_id}\n'
            f'input_model_name: {self.input_model_name}    '
            f'output_model_name: {self.output_model_name}\n'
            f'replay_memory_size: {self.replay_memory_size}    '
            f'mini_batch_size: {self.batch_size}    '
            f'discount: {self.discount}\n'
            f'tau: {self.tau}    '
            f'learning_rate: {self.learning_rate}    '
            f'policy_noise: {self.policy_noise}\n'
            f'noise_clip: {self.noise_clip}    '
            f'policy_freq: {self.policy_freq}    '
            f'entropy_coefficient: {self.entropy_coefficient}\n'
            f'minimum_entropy: {self.minimum_entropy}    '
            f'entropy_decay: {self.entropy_decay}\n'
        )

        fig.text(0.5, 0.02, text, ha='center', fontsize=10, bbox=dict(facecolor='lightgrey', alpha=0.5))

        # Save the figure
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)



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

    SAC = Agent(args.train, args.endless, args.continue_training, args.render, args.use_gpu, hyperparameter_set=args.hyperparameters)
    SAC.run(args.train, args.continue_training)