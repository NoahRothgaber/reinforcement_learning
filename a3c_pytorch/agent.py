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

import flappy_bird_gymnasium

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class SharedOptimizer(torch.optim.Adam):
    def __init__(self, params, learning_rate, betas, epsilon=1e-8, weight_decay=0):
        super(SharedOptimizer, self).__init__(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Change 'step' to be a tensor rather than an integer
                state['step'] = torch.tensor(0, dtype=torch.int64).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # Share the memory for exp_avg and exp_avg_sq to ensure compatibility in multiprocessing
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

                
class WorkerAgent(mp.Process):

    def __init__(self, name, env_id, env_make_params, gamma, tau, learning_rate, max_reward, max_timestep, max_episodes,
                  replay_buffer_size, hidden_dims, global_actor_critic, global_episode_index, rewards_per_episode, optimizer):
        super(WorkerAgent, self).__init__()

        # Initizalize parameters
        self.name = name
        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.max_reward = max_reward
        self.max_timestep = max_timestep
        self.max_episodes = max_episodes
        self.replay_buffer_size = replay_buffer_size
        self.hidden_dims = hidden_dims
        self.global_actor_critic = global_actor_critic
        self.global_episode_index = global_episode_index
        self.rewards_per_episode = rewards_per_episode
        self.optimizer = optimizer
        self.env_make_params = env_make_params


    
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

            self.local_actor_critic.memory.reset()  # clear all the data in the replay buffer at the start of each episode

            # reset some data
            self.values = []
            self.log_probs = []
            self.entropy_term = 0
            self.step_count = 0

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

            with self.global_episode_index.get_lock():
                self.global_episode_index.value += 1

            with self.rewards_per_episode.get_lock():
                self.rewards_per_episode[self.global_episode_index.value] = episode_reward

            print(self.name, 'episode ', self.global_episode_index.value, 'reward %.1f' % episode_reward)

             # Check if global episode count exceeds max_episodes
            if self.global_episode_index.value >= self.max_episodes:
                print(f"{self.name} stopping: global episode count reached max_episodes {self.max_episodes}.")
                break  

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
        
        # Global variable that can be modified across threads
        self.global_episode_index = mp.Value('i', 0) # global episode tracker
        self.rewards_per_episode = mp.Array('d', self.max_episodes, lock=True)  # 'd' is for double precision float

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
        # if endless or not self.is_training:
        #     self.max_episodes = itertools.count()
        # else:
        #     self.max_episodes = range(self.max_episodes)

        if self.continue_training:
            self.is_training = True


    

        self.step_count = 0

        self.log_probs = []
        self.values = []

        # GLOBAL AGENT INIT
    
        self.workers = []



        # Create instance of the environment.
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions & observation space size
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)


        self.global_actor_critic = A3CActorCritic(self.num_states, self.num_actions, self.gamma, self.replay_buffer_size, self.hidden_dims)
        self.global_actor_critic.share_memory()
        self.optimizer = SharedOptimizer(self.global_actor_critic.parameters(), learning_rate=self.learning_rate, betas=(0.92, 0.999))
        
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

            self.run()

        # if we are not training, generate the actor and critic policies based on the saved model
        else:
            self.load()
            self.actor.eval()
            self.critic.eval()
            start_time = datetime.now()
            log_message = f"{start_time.strftime(self.DATE_FORMAT)}: Run starting..."
            print(log_message)
            # then need to create environment and loop through testing episode 
            # make a function for this
        
    def run(self):

        self.initialize_workers()
        self.save_graph()
        print("Max Episodes Reached!")
        



    # There is no functional difference between . pt and . pth when saving PyTorch models
    def save(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)
        torch.save(self.global_actor_critic.state_dict(), f"{self.RUNS_DIR}/{self.OUTPUT_FILENAME}_global_actor_critic.pth")

    def load(self):
        try:
            self.global_actor_critic.load_state_dict(
                torch.load(f"{self.RUNS_DIR}/{self.INPUT_FILENAME}_global_actor_critic.pth")
            )
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model load failed. Switching to training mode.")
            
            # Create the "runs" directory if it doesn't exist
            if not os.path.exists(self.RUNS_DIR):
                os.makedirs(self.RUNS_DIR)
                print(f"Created directory: {self.RUNS_DIR}")

            # Switch to training mode
            self.is_training = True
            print(f"Starting Training...")
    
        
    def save_graph(self):
        # Convert shared array to list
        rewards = list(self.rewards_per_episode)
        
        fig, ax = plt.subplots()

        # Plot the rewards
        ax.plot(rewards, label='Reward per Episode')
        
        # Optionally, calculate and plot the moving average
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax.plot(range(len(moving_avg)), moving_avg, label='100-episode Moving Avg', color='orange')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()

        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def initialize_workers(self):
        for i in range(mp.cpu_count()):
            worker_name = f'Worker {i}'
            temp_worker = WorkerAgent(name=worker_name, env_id=self.env_id, gamma=self.gamma, tau=self.tau, learning_rate=self.learning_rate, 
                                      max_timestep=self.max_timestep, max_episodes=self.max_episodes, replay_buffer_size=self.replay_buffer_size, 
                                      global_actor_critic=self.global_actor_critic, global_episode_index=self.global_episode_index, rewards_per_episode=self.rewards_per_episode, optimizer=self.optimizer,
                                      max_reward=self.max_reward, hidden_dims=self.hidden_dims, env_make_params= self.env_make_params)
            self.workers.append(temp_worker)
        [w.start() for w in self.workers]
        [w.join() for w in self.workers] 
        

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

