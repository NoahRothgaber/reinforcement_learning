# Proximal Policy Optimization

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import torch
from torch import nn, optim
import gymnasium as gym
from networks import ActorNetwork, CriticNetwork

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
plt.switch_backend('Agg')


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
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        states_array = np.array(self.states, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.int64)
        log_probs_array = np.array(self.log_probs, dtype=np.float32)
        values_array = np.array(self.values, dtype=np.float32)
        rewards_array = np.array(self.rewards, dtype=np.float32)
        dones_array = np.array(self.dones, dtype=np.bool_)

        return states_array, actions_array, log_probs_array, values_array, rewards_array, dones_array, batches

    def store_memory(self, state, action, log_prob, value, reward, done):
        # Ensure state is moved to CPU before converting to a NumPy array
        self.states.append(state.cpu().numpy())
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
        self.gae_lambda = hyperparameters['gae_lambda']
        self.policy_clip = hyperparameters['policy_clip']
        self.n_epochs = hyperparameters['n_epochs']
        self.alpha = hyperparameters['alpha']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.rewards_per_episode = []
        self.train = train
        self.endless = endless
        self.continue_training = continue_training
        self.render = render
        self.use_gpu = use_gpu

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
        # Ensure all tensors are moved to CPU before storing in memory
        self.memory.store_memory(state.cpu(), action, log_probs, values, reward, done)

    def save(self):
        torch.save(self.actor.state_dict(), f"{self.MODEL_FILE}_actor")
        torch.save(self.critic.state_dict(), f"{self.MODEL_FILE}_critic")

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)  # Fix: remove list wrapping

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, values_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            state_tensor = torch.tensor(state_arr, dtype=torch.float32).to(self.actor.device)
            action_tensor = torch.tensor(action_arr, dtype=torch.long).to(self.actor.device)
            old_probs_tensor = torch.tensor(old_probs_arr, dtype=torch.float32).to(self.actor.device)
            value_tensor = torch.tensor(values_arr, dtype=torch.float32).to(self.actor.device)
            reward_tensor = torch.tensor(reward_arr, dtype=torch.float32).to(self.actor.device)
            done_tensor = torch.tensor(done_arr, dtype=torch.bool).to(self.actor.device)

            values = value_tensor.cpu().numpy()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                advantage_at_t = 0
                for k in range(t, len(reward_arr) - 1):
                    advantage_at_t += discount * (reward_arr[k] + self.discount_factor_gamma * values[k + 1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.discount_factor_gamma * self.gae_lambda
                advantage[t] = advantage_at_t

            advantage = torch.tensor(advantage, dtype=torch.float32).to(self.actor.device)

            for batch in batches:
                states = state_tensor[batch]
                actions = action_tensor[batch]
                old_probs = old_probs_tensor[batch]
                advantages = advantage[batch]
                returns = advantages + value_tensor[batch]

                dist = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_graph(self, rewards_per_episode):
        plt.figure(figsize=(10, 5))

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        cumulative_mean_reward = np.zeros(len(rewards_per_episode))
        for x in range(len(cumulative_mean_reward)):
            cumulative_mean_reward[x] = np.mean(rewards_per_episode[0:(x + 1)])

        plt.plot(mean_rewards, label='Mean Reward (Last 100 Episodes)', color='tab:blue')
        plt.plot(cumulative_mean_reward, label='Cumulative Mean Reward', color='tab:green', linestyle='--')

        plt.title(f"Training Progress for {self.env_id} using Proximal Policy Optimization (PPO)")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE)
        plt.close()

    def run(self, is_training=True, render=False):
        n_games = 10000000
        N = 20
        learn_iters = 0
        n_steps = 0
        best_score = -float('inf')
        score_history = []

        if is_training:
            start_time = datetime.now()
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        for episode in range(n_games):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.actor.device)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                action, log_prob, value = self.choose_action(state.cpu().numpy())

                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float32, device=self.actor.device)

                if is_training:
                    # Ensure values are moved to CPU and then converted to numpy arrays if needed
                    self.remember(state, action, log_prob, value, torch.tensor(reward, dtype=torch.float32), terminated)
                    n_steps += 1
                    if n_steps % N == 0:
                        self.learn()
                        learn_iters += 1

                state = new_state

            self.rewards_per_episode.append(episode_reward)
            score_history.append(episode_reward)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.save()
                log_message = f"New best average score: {best_score:.2f} at episode {episode}"
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

            if is_training and episode % 25 == 0:
                self.save_graph(self.rewards_per_episode)

        self.save_graph(self.rewards_per_episode)
        if avg_score >= self.stop_on_reward * .95:   
            print(f"Training completed. Average Reward {avg_score} exceeded reward cap of {self.stop_on_reward * .95}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--continue_training', help='Continue training mode', action='store_true')
    parser.add_argument('--render', help='Rendering mode', action='store_true')
    parser.add_argument('--use_gpu', help='Device mode', action='store_true')
    parser.add_argument('--endless', help='Endless mode', action='store_true')
    args = parser.parse_args()

    proximal_policy = PPO_Agent(
        args.train,
        args.endless,
        args.continue_training,
        args.render,
        args.use_gpu,
        hyperparameter_set=args.hyperparameters
    )
    proximal_policy.run(args.train, args.render)
