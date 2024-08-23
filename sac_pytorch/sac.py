import torch
from torch import nn
import torch.nn.functional as F

#scales the cost function to encourage exploration
# smooth


# Actor Network
class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, use_gpu):
        super(SAC_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mean = nn.Linear(300, action_dim)
        self.fc_log_std = nn.Linear(300, action_dim)
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(self.device)

    def forward(self, x): # Produces mean and log standard deviation of the action distribution
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
 
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, state): # Samples an action space from the stochastic policy using the reparameterization trick
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        #print(f'normal: {normal}')
        z = normal.rsample()
        action = torch.tanh(z) # Selected action

        #action clamp tanh here maybe

        log_prob = normal.log_prob(z) # Calculates the log probability of the acton 
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob # log_prob is used to calculate the entropy term


# Critic Network (Twin Critic Networks)
class SAC_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=400, hidden_dim_2=300):
        super(SAC_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc5 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc6 = nn.Linear(hidden_dim_2, 1)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, x, u):
        xu = torch.cat([x, u], 1).to(self.device)
        q1 = F.relu(self.fc1(xu))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(xu))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1).to(self.device)
        q1 = F.relu(self.fc1(xu))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
    

# Value Network
class SAC_Entropy(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu):
        super(SAC_Entropy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


    
if __name__ == '__main__':
    state_dim = 12                                          # define number of input variables (the state)
    action_dim = 2                                          # define the number of possible outputs (the action)
    action_high = [1]
    action_low = [-1]                                      # define the maximum action value (continuous action space)
    net = SAC_Actor(state_dim, action_dim, action_high, action_low)      # define the network with the state dimensions (12) and action dimensions (2)
    state = torch.randn(1, state_dim)                       # create some random input
    output = net(state)                                     # send some random input into the network
    print(output)                                           # print the output