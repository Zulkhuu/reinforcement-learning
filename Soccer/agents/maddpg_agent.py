import sys
sys.path.append('../')

from collections import namedtuple, deque
from utils.model import Actor, Critic
from torch.optim import Adam
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():

    def __init__(self, params):
        """Initialize an DDPG object"""

        self.batch_size = params['batch_size']
        self.buffer_size = params['buffer_size']
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.num_agents = params['n_agents']
        self.random_seed = params['random_seed']

        self.t_step = 0
        #self.agents = [DDPG(params), DDPG(params)]
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(DDPG(params))

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.random_seed)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        #states = np.reshape(states, (1,self.state_size*self.num_agents))
        """Returns actions for given state as per current policy."""
        actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.act(states[idx], add_noise)
            actions.append(action)
        actions = np.reshape(actions, (1, self.action_size*self.num_agents))
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        states = np.reshape(states, (1,self.state_size*self.num_agents))
        next_states = np.reshape(next_states, (1, self.state_size*self.num_agents))
        rewards = np.reshape(rewards, (1, self.num_agents))

        self.t_step += 1
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        for agent in self.agents:
            agent.noise.scale = max(agent.noise.scale*agent.noise_decay, agent.noise_end)

        # Learn, if enough samples are available in memory and at interval settings
        if len(self.memory) > self.batch_size:
            if self.t_step % 1 == 0:
                for idx, agent in enumerate(self.agents):
                    experiences = self.memory.sample()
                    agent.learn(experiences, agent.gamma, idx)

    def load(self, filename):
        for idx, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('models/{}_actor{}.pth'.format(filename, idx)))
            agent.critic_local.load_state_dict(torch.load('models/{}_critic{}.pth'.format(filename, idx)))

    def save(self, filename):
        for idx, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'models/{}_actor{}.pth'.format(filename, idx))
            torch.save(agent.critic_local.state_dict(), 'models/{}_critic{}.pth'.format(filename, idx))

class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, params):
        """Initialize an DDPG object"""

        self.state_size = int(params['state_size'])
        self.action_size = int(params['action_size'])
        self.num_agents = int(params['n_agents'])
        self.random_seed = params['random_seed']

        self.weight_decay = params['weight_decay']
        self.lr_critic = params['lr_critic']
        self.lr_actor = params['lr_actor']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.noise_duration = params['noise_duration']
        self.noise_decay = params['noise_decay']
        self.noise_end = params['noise_end']

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.random_seed).to(device)
        self.critic_target = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(self.action_size, params['noise_start'])


    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        if agent_id == 0:
            actions_next = self.actor_target(next_states[:,:self.state_size])
            actions_next = torch.cat((actions_next, actions[:,self.action_size:]), dim=1)
            rewards = rewards[:,:1]
        else:
            actions_next = self.actor_target(next_states[:,self.state_size:])
            actions_next = torch.cat((actions[:,:self.action_size], actions_next), dim=1)
            rewards = rewards[:,1:]


        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss

        if agent_id == 0:
            actions_pred = self.actor_local(states[:,:self.state_size])
            actions_pred = torch.cat((actions_pred, actions[:,self.action_size:]), dim=1)
        else:
            actions_pred = self.actor_local(states[:,self.state_size:])
            actions_pred = torch.cat((actions[:,:self.action_size], actions_pred), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, scale=1.0, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.scale = scale
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.scale*self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object"""
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
