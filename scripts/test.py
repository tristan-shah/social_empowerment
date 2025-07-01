import jax
from jax import numpy as jnp

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from soc_emp import Dynamics

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output mean and log std of action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Clamp log std to prevent numerical issues
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        return dist

    def sample_action(self, state):
        dist = self.forward(state)
        action = dist.rsample()  # Reparameterized sample for backprop
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.tanh(), log_prob
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    T = 1000

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    xt = dyn.init_state()
    xt = xt.at[0].set(0.0)

    X = jnp.zeros((T+1, dyn.state_dim))
    U = jnp.zeros((T, dyn.control_dim))
    X = X.at[0].set(xt)

    policy = Policy(2, 1)

    for t in range(T):

        with torch.no_grad():
            ut = policy.sample_action(torch.tensor(xt, dtype = torch.float))[0]
        
        ut = jnp.asarray(ut.to(torch.float64))

        xt = dyn.step(xt, ut)
        print(t, xt, ut)

        X = X.at[t + 1].set(xt)
        U = U.at[t].set(ut)

    dyn.render(X, path = 'pendulum.mp4')