#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')

from unityagents import UnityEnvironment
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils.config
import pprint
import torch

env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))


#from agents.maddpg import MADDPG
from agents.maddpg_agent import MADDPG

# Load parameters from file
params = utils.config.HYPERPARAMS['Tennis1']
hparams = params['agent']

# Save results to csv file
result_filename = 'tennis_tuning.csv'
hyperscores = []

# Number of episodes for evaluating
N_EPISODES = 800

def train_maddpg(lr_actor=1e-3, lr_critic=1e-3, tau=0.06, noise_start=5, noise_duration=400, n_run=0):

    # Set tunable parameters
    hparams['lr_actor'] = lr_actor
    hparams['lr_critic'] = lr_critic
    hparams['tau'] = tau
    hparams['eb_start'] = noise_start
    hparams['eb_duration'] = noise_duration
    # Run for 200 episodes only
    params['n_episodes'] = N_EPISODES

    # Filename string
    filename = "{:s}_lra{:.0E}_lrc{:.0E}_tau{:.0E}_nstart{:.1f}_nt{:d}_run{:d}"
    filename = filename.format(hparams['agent_name'], hparams['lr_actor'], hparams['lr_critic'],
                               hparams['tau'], hparams['eb_start'], hparams['eb_duration'], n_run)

    # Create agent instance
    print("Created agent with following hyperparameter values:")
    print("filename:{}".format(filename))
    agent = MADDPG(params['agent'])
    pprint.pprint(params['agent'])

    # Reset and set environment to training mode
    env_info = env.reset(train_mode=True)[brain_name]

    # Maximum number of training episodes
    n_episodes = params['n_episodes']

    # List containing scores from each episode
    scores = []

    # Store last 100 scores
    scores_window = deque(maxlen=params['scores_window_size'])
    # Flag to indicate environment is solved
    solved = False

    # Train loop
    for i_episode in range(1, n_episodes+1):
        # Reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # Reset score and agent's noise
        agent_scores = np.zeros(num_agents)
        agent.reset()

        # Loop each episode
        while True:
            # Select and take action
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            # Get next state, reward and done
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # Store experience and learn
            agent.step(states, actions, rewards, next_states, dones)
            # State transition
            states = next_states
            # Update total score
            agent_scores += rewards
            # Exit loop if episode finished
            if np.any(dones):
                break

        # Save most recent score
        scores_window.append(np.max(agent_scores))
        scores.append([np.max(agent_scores), np.mean(scores_window)])

        # Print learning progress
        print('\rEpisode {}\tMax Score: {:.6f}\tAverage Score: {:.6f}'.format(i_episode, np.max(agent_scores), np.mean(scores_window)), end="")
        if i_episode % params['scores_window_size'] == 0:
            print('\rEpisode {}\tMax Score: {:.6f}\tAverage Score: {:.6f}'.format(i_episode, np.max(agent_scores), np.mean(scores_window)))
        if np.mean(scores_window)>=params['solve_score']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))

            model_filename = "{:s}_lra{:.0E}_lrc{:.0E}_tau{:.0E}_nstart{:.1f}_nt{:d}_solved{:d}"
            model_filename = model_filename.format(hparams['agent_name'], hparams['lr_actor'], hparams['lr_critic'],
                hparams['tau'], hparams['eb_start'], hparams['eb_duration'], i_episode-100)
            for idx, agent_ in enumerate(agent.agents):
                torch.save(agent_.actor_local.state_dict(), '../models/{}_actor{}.pth'.format(model_filename, idx))
                torch.save(agent_.critic_local.state_dict(), '../models/{}_critic{}.pth'.format(model_filename, idx))
            df = pd.DataFrame(scores,columns=['scores','average_scores'])
            df.to_csv('../scores/{:s}.csv'.format(model_filename))
            break

    # Save score
    df = pd.DataFrame(scores,columns=['scores','average_scores'])
    df.to_csv('../scores/{:s}.csv'.format(filename))

    return i_episode-100

# How many times to try same parameter configurations
'''
for idx in range(10):
    train_maddpg()
'''
n_try = 3

# Learning rate
lrs = [5e-3, 2e-3, 1e-3, 1e-4, 5e-5]
for idx in range(n_try):
    for lr in lrs:
        train_maddpg(lr_actor=lr, lr_critic=lr, n_run=idx)

# Soft update decay
taus = [0.1, 0.06, 0.05, 0.01, 0.005, 0.001]
for idx in range(n_try):
    for tau in taus:
        train_maddpg(tau=tau, n_run=idx)

# Exploration boost duration
n_ts = [500, 400, 350, 300, 250, 200, 150]
for idx in range(n_try):
    for n_t in n_ts:
        train_maddpg(noise_duration=n_t, n_run=idx)

# Exploration boost start
n_starts = [10, 7, 5, 3, 2, 1]
for idx in range(n_try):
    for n_start in n_starts:
        train_maddpg(noise_start=n_start, n_run=idx)

# Close the environment
env.close()
