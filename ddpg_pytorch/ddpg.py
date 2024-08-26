import torch
from torch import nn
import torch.nn.functional as F

# Actor Network
class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, action_low=None, action_high=None):
        super(DDPG_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Ensure action_low and action_high are specified
        if action_low is None or action_high is None:
            raise ValueError("action_low and action_high must be specified")
        
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(self.device)

        self.max_action = (self.action_high - self.action_low) / 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output scaled to the action range
        return self.max_action * torch.tanh(self.fc3(x)) + (self.action_high + self.action_low) / 2

# Critic Network (Single Critic Network for DDPG)
class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=400, hidden_dim_2=300):
        super(DDPG_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, x, u):
        xu = torch.cat([x, u], 1).to(self.device)
        q = F.relu(self.fc1(xu))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

if __name__ == '__main__':
    state_dim = 12                                          # define number of input variables (the state)
    action_dim = 2                                          # define the number of possible outputs (the action)
    action_high = [1]
    action_low = [-1]                                       # define the maximum action value (continuous action space)
    
    use_gpu = False  # Set to True if using GPU
    actor = DDPG_Actor(state_dim, action_dim, use_gpu, action_low, action_high)  # Create Actor network
    critic = DDPG_Critic(state_dim, action_dim, use_gpu)                          # Create Critic network
    
    state = torch.randn(1, state_dim)                       # create some random input
    action = torch.randn(1, action_dim)                     # create some random action
    
    actor_output = actor(state)                             # send some random input into the actor network
    critic_output = critic(state, action)                   # send some random state-action pair into the critic network
    
    print("Actor Output:", actor_output)                    # print the output of the actor network
    print("Critic Output:", critic_output)                  # print the output of the critic network
