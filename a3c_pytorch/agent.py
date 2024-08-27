import gymnasium as gym

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools

import torch
from torch import nn
import torch.nn.functional as F

import argparse
import yaml

from datetime import datetime, timedelta
import os
import torch.multiprocessing as mp

from a3c import A3CActorCritic

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class SharedOptimizer(torch.optim.Adam):
    def __init__(self, params, learning_rate, betas, epsilon=1e-8, weight_decay=0):
        super(SharedOptimizer, self).__init__(params,lr=learning_rate,betas=betas,weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'].share_memory_()
                
class WorkerAgent(mp.Process):
    def __init__(self, name, env_id, gamma, tau, learning_rate, max_reward, max_timestep,
                  replay_buffer_size, hidden_dims, global_actor_critic, global_episode_index, optimizer):

        # Initizalize parameters
        self.name = name
        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.max_reward = max_reward
        self.max_timestep = max_timestep
        self.replay_buffer_size = replay_buffer_size
        self.hidden_dims = hidden_dims
        self.global_actor_critic = global_actor_critic
        self.global_episode_index = global_episode_index
        self.optimizer = optimizer

        
        

        # Create instance of the environment.
        self.env = gym.make(self.env_id, render_mode=None, **self.env_make_params)

        # Number of possible actions & observation space size
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Initialize the Local Actor Critic Network
        self.local_actor_critic = A3CActorCritic(self.num_states, self.num_actions, self.gamma, self.replay_buffer_size, self.hidden_dims)
        self.global_actor_critic = global_actor_critic


    def run(self):

        for episode in itertools.count():
            # init stuff
            state, _ = self.env.reset()  # Initialize environment. Reset returns (state,info)
            
            terminated = False      # True when agent reaches goal or fails
            truncated = False       # True when max_timestep is reached

            episode_reward = 0.0    # Used to accumulate rewards per episode

            self.replay_buffer.reset()  # clear all the data in the replay buffer at the start of each episode

            # reset some data
            self.values = []
            self.log_probs = []
            self.entropy_term = 0

            while(not terminated and not truncated and not self.step_count == self.max_timestep):
                    action = self.local_actor_critic.choose_action(state)
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action)

                    episode_reward+= reward

                    terminated = self.step_count == self.max_timestep - 1 or terminated or truncated
                    
                    self.local_actor_critic.memory.add(state, action, reward, terminated)

                    self.step_count += 1
                    state = next_state
            # Train after each episode        
            self.train(terminated)     

            with self.episode_idx.get_lock():
                self.global_episode_index  += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % episode_reward)       

    def train(self, terminated):
        # Calculate the Actor Critic loss
        loss = self.local_actor_critic.calc_loss(terminated) 
        self.optimizer.zero_grad()
        loss.backward()

        # takes gradients from local agent and throws them over to global agent
        for local_param, global_param in zip(
            self.local_actor_critic.parameters(),
            self.global_actor_critic.parameters()):
                global_param._grad = local_param.grad
        
        self.optimizer.step()
        # Update the weights from the Global Network to the Worker Network
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

        # Clears previous episode memory
        self.local_actor_critic.memory.reset()



# A3C Agent
class GlobalAgent():
    def __init__(self, is_training, endless, continue_training, render, use_gpu, hyperparameter_set):
        with open(os.path.join(os.getcwd(), 'hyperparameters.yml'), 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.is_training = is_training
        self.continue_training = continue_training

        self.env_id                 = hyperparameters['env_id']
        self.input_model_name       = hyperparameters['input_model_name']
        self.output_model_name      = hyperparameters['output_model_name']
        self.replay_buffer_size     = hyperparameters['replay_memory_size']         # size of replay memory
        self.batch_size             = hyperparameters['mini_batch_size']            # size of the training data set sampled from the replay memory
        self.gamma                  = hyperparameters['gamma']
        self.tau                    = hyperparameters['tau']
        self.learning_rate          = hyperparameters['learning_rate']
        self.model_save_freq        = hyperparameters['model_save_freq']
        self.max_reward             = hyperparameters['max_reward']
        self.max_timestep           = hyperparameters['max_timestep']
        self.max_episodes           = hyperparameters['max_episodes']
        self.hidden_dims           = hyperparameters['hidden_dims']
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

        self.device = 'cpu'

        # set endless mode if endless arg is true, otherwise set max episodes based on parameters 
        if endless or not self.is_training:
            self.max_episodes = itertools.count()
        else:
            self.max_episodes = range(self.max_episodes)

        if self.continue_training:
            self.is_training = True


        # List to keep track of rewards collected per episode.
        self.rewards_per_episode = []

        self.step_count = 0

        self.log_probs = []
        self.values = []

        # GLOBAL AGENT INIT
        self.global_actor_critic = A3CActorCritic(self.num_states, self.num_actions, self.gamma, self.replay_buffer_size, self.hidden_dims)
        self.workers = []

        # Create instance of the environment.
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions & observation space size
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)
        self.env.close() # close the environment as we're just getting num_actions and states from it
        
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
        
    def run(self, is_training=True, continue_training=False):
        # best_reward = float(-np.inf)   # Used to track best reward
        best_reward = None

        for episode in self.max_episodes:

            state, _ = self.env.reset()  # Initialize environment. Reset returns (state,info).
            terminated = False      # True when agent reaches goal or fails
            truncated = False       # True when max_timestep is reached
            episode_reward = 0.0    # Used to accumulate rewards per episode
            self.step_count = 0          # Used for syncing policy => target network

            self.replay_buffer.reset()  # clear all the data in the replay buffer at the start of each episode

            # reset some data
            self.values = []
            self.log_probs = []
            self.entropy_term = 0


            if not is_training or continue_training:
                self.load()

            while(not terminated and not truncated and not self.step_count == self.max_timestep):
               
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Move state to the correct device

                value = self.critic(state)
                policy_dist = self.actor(state)

                value = value.cpu().detach().numpy()[0,0]
                dist = policy_dist.cpu().detach().numpy() 

                action = np.random.choice(self.num_actions, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])

                entropy = -np.sum(np.mean(dist) * np.log(dist))

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminated = self.step_count == self.max_timestep - 1 or terminated
                self.step_count += 1

                if is_training or continue_training:
                    self.replay_buffer.add(state, action, reward, next_state, terminated)
                    self.log_probs.append(log_prob)
                    self.values.append(value)
                    self.entropy_term += entropy

                    if terminated: # train if the episode has ended
                        # Train the agent
                        self.train()

                state = next_state
                episode_reward += reward
               

            # Keep track of the rewards collected per episode and save model
            self.rewards_per_episode.append(episode_reward)

            if is_training or continue_training:
                current_time = datetime.now()
                if current_time - self.last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(self.rewards_per_episode)
                    self.last_graph_update_time = current_time

                if (episode + 1) % 100 == 0:
                    #Save model
                    self.save()
                    time_now = datetime.now()
                    average_reward = np.mean(self.rewards_per_episode[-100:])
                    log_message = f"{time_now.strftime(self.DATE_FORMAT)}: Saving Model at Episode: {episode + 1} Average Reward: {average_reward:0.1f}"
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

    # There is no functional difference between . pt and . pth when saving PyTorch models
    def save(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)
        torch.save(self.actor.state_dict(), f"{self.RUNS_DIR}/{self.OUTPUT_FILENAME}_actor.pth")
        torch.save(self.critic.state_dict(), f"{self.RUNS_DIR}/{self.OUTPUT_FILENAME}_critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load(f"{self.RUNS_DIR}/{self.INPUT_FILENAME}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{self.RUNS_DIR}/{self.INPUT_FILENAME}_critic.pth"))
    
        
    
    def save_graph(self, rewards_per_episode):
        # Save plots
        fig, ax1 = plt.subplots()

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

        # Make y axis 1 and 2 the same scale
        ax1.set_ylim([min(min(mean_rewards), min(mean_total)), max(max(mean_rewards), max(mean_total))])
        ax2.set_ylim(ax1.get_ylim())

        # Save the figure
        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def initialize_workers(self):
        for i in range(mp.cpu_count()):
            worker_name = f'Worker {i}'
            temp_worker = WorkerAgent(worker_name, self.env_id, self.gamma, self.tau, self.learning_rate, 
                                      self.max_episodes, self.max_timestep, self.replay_buffer_size)
            self.workers.append(temp_worker)
        

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




    # create global agent
    globalAgent = GlobalAgent(args.train, args.endless, args.continue_training, args.render, args.use_gpu, hyperparameter_set=args.hyperparameters)

    global_actor_critic.run(args.train, args.continue_training)


    global_ep = mp.Value('i', 0) # global episode tracker

    # create workers 

    workers = [Agent(global_actor_critic,
                    name=i,
                    global_ep_idx=global_ep) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers] 