# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import DDPGAgent
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, params):

        self.agents = []

        for i in range(params['number_of_agents']):
            self.agents.append(DDPGAgent(params))


        for agent in self.agents:
            print agent

    def step(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions
