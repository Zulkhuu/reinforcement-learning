import sys
sys.path.append('../')

from agents.ddpg import DDPG

import numpy as np
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, params):

        self.agents = []
        for i in range(params['n_agents']):
            self.agents.append(DDPG(params))

        for agent in self.agents:
            print(agent)

    def act(self, states, add_noise=True):
        """Returns actions for each agent's observation as per current policy."""
        actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.act(states[idx], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        actions = np.reshape(actions, (1, 4))
        states = np.reshape(states, (1, 48))
        next_states = np.reshape(next_states, (1, 48))

        for idx, agent in enumerate(self.agents):
            agent.step(states, actions, rewards[idx], next_states, dones[idx])
