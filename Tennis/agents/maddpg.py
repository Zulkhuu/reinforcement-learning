import sys
sys.path.append('../')

from agents.ddpg import DDPG

from utils.ReplayBuffer import ReplayBuffer
from collections import namedtuple, deque
import numpy as np
import random
import copy
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, params):

        self.n_agents = params['n_agents']
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.buffer_size = int(params['buffer_size'])    # replay buffer size
        self.batch_size = params['batch_size']           # minibatch size
        self.gamma = params['gamma']                     # discount factor
        self.tau = params['tau']
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(DDPG(params))

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, params['random_seed'])

    def act(self, obs_all_agents, noise=0.0001):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.agents, obs_all_agents)]
        #actions = []
        #for idx in range(2):
        #    actions.append(self.agents[idx].act(obs_all_agents[idx],noise))

        return actions


    def target_act(self, obs_all_agents, noise=0.0001):
        """Get target network actions from all the agent in the MADDPG object"""

        #target_actions = [agent.target_act(obs, noise) for ddpg_agent,obs in zip(self.agents, obs_all_agents)]
        target_actions = []
        for idx in range(2):
            target_actions.append(self.agents[idx].target_act(obs_all_agents[:,idx,:],noise))

        return target_actions

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experience = self.memory.sample()
            self.learn(experience)

    def learn(self, experience):
        state, action, reward, next_state, done = experience

        state_full = torch.cat((state[:,0,:],state[:,1,:]),dim=1)
        next_state_full = torch.cat((next_state[:,0,:],next_state[:,1,:]),dim=1)

        for agent_id, agent in enumerate(self.agents):
            #update critic network
            agent.critic_optimizer.zero_grad()

            #critic loss = batch mean of (y- Q(s,a) from target network)^2
            #y = reward of this timestep + discount * Q(st+1,at+1) from target network
            target_actions = self.target_act(next_state)
            target_actions = torch.cat(target_actions,dim=1)

            target_critic_input = torch.cat((next_state_full,target_actions),dim=1).to(device)
            #print(target_critic_input.size())
            with torch.no_grad():
                q_next = agent.target_critic(target_critic_input)

            y= reward[:,agent_id].reshape(-1,1) +self.gamma*q_next * (1 - done[:,agent_id].reshape(-1,1))
            #print(y.shape)
            #print("Before:{}".format(action))
            action = action.view(-1,4)
            #print("After:{}".format(action))
            critic_input = torch.cat((state_full,action),dim=1).to(device)
            q = agent.critic(critic_input)

            huber_loss = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()
            critic_loss = huber_loss(q,y.detach())
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()


            #update actor network using policy gradient
            agent.actor_optimizer.zero_grad()
            # make input to agent
            # detach the other agents to save computation
            # saves some time for computing derivative

            #obsesrvation = state[:,agent_id,:]
            #q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
            #           else self.maddpg_agent[i].actor(ob).detach()
            #           for i, ob in enumerate(obs) ]

            q_input = []
            for idx in range(2):
                if idx == agent_id:
                    q_input.append(self.agents[idx].actor(state[:,idx,:]))
                else:
                    q_input.append(self.agents[idx].actor(state[:,idx,:]).detach())

            q_input = torch.cat(q_input,dim=1)
            q_input2 = torch.cat((state_full,q_input),dim=1)

            actor_loss = -agent.critic(q_input2).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()

            self.update_targets()

    def update_targets(self):
        """soft update target networks"""
        for agent in self.agents:
            agent.soft_update(agent.target_actor, agent.actor, self.tau)
            agent.soft_update(agent.target_critic, agent.critic, self.tau)
