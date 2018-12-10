#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')

from unityagents import UnityEnvironment
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
import pprint
import torch
import time

from agents.ddpg_agent import DDPG

env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Load parameters from file
hparams = config.HYPERPARAMS['DDPG']
params = config.TRAINPARAMS['Reacher']

# Save results to csv file
result_filename = 'reacher_tuning.csv'
hyperscores = []

# Number of episodes for evaluating
N_EPISODES = 200

def train_ddpg(lr_actor=1e-4, lr_critic=1e-4, batch_size=128, fc1_units=256, fc2_units=128):

    # Set tunable parameters
    hparams['lr_actor'] = lr_actor
    hparams['lr_critic'] = lr_critic
    hparams['batch_size'] = batch_size
    hparams['fc1_units'] = int(fc1_units)
    hparams['fc2_units'] = int(fc2_units)
    # Run for 200 episodes only
    params['n_episodes'] = N_EPISODES

    # Filename string
    filename = "{:s}_lra{:.0E}_lrc{:.0E}_batch{:d}_fc:{:d}:{:d}"
    filename = filename.format(hparams['name'], hparams['lr_actor'], hparams['lr_critic'],
                               hparams['batch_size'], hparams['fc1_units'], hparams['fc2_units'])

    # Create agent instance
    print("Created agent with following hyperparameter values:")
    print("filename:{}".format(filename))
    agent = DDPG(hparams)
    pprint.pprint(hparams)

    # Maximum number of training episodes
    n_episodes = params['n_episodes']
    # List containing scores from each episode
    scores = []
    # Store last 100 scores
    scores_window = deque(maxlen=params['scores_window_size'])

    #states = env_info.vector_observations                  # get the current state (for each agent)
    #scores = np.zeros(num_agents)                          # initialize the score (for each agent)

    # Train loop
    for i_episode in range(1, n_episodes+1):
        # Reset environment
        env_info = env.reset(train_mode=True)[brain_name]

        # Observe current state
        state = env_info.vector_observations[0]

        # Reset score and done flag
        score = 0
        done = False

        # Loop each episode
        while not done:

            # Select action with e-greedy policy
            action = agent.act(state)

            # Take action
            env_info = env.step(action)[brain_name]

            # Observe the next state
            next_state = env_info.vector_observations[0]

            # Get the reward
            reward = env_info.rewards[0]

            # Check if episode is finished
            done = env_info.local_done[0]

            # Store experience
            agent.step(state, action, reward, next_state, done)

            # State transition
            state = next_state

            # Update total score
            score += reward

        # Save most recent score
        scores_window.append(score)
        scores.append([score, np.mean(scores_window)])

        # Print learning progress
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % params['scores_window_size'] == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=params['solve_score']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'models/{:s}_actor.pth'.format(filename))
            torch.save(agent.critic_local.state_dict(), 'models/{:s}_critic.pth'.format(filename))
            break


    # Save score
    df = pd.DataFrame(scores,columns=['scores','average_scores'])
    df.to_csv('../scores/{:s}.csv'.format(filename))

    time.sleep(1)
    return i_episode-100

# Hidden layer size
#train_ddpg(1e-4, 1e-4, 128, 400, 300)
#train_ddpg(1e-4, 1e-4, 128, 256, 128)
#train_ddpg(1e-4, 1e-4, 128, 128, 64)

# Learning rates
#train_ddpg(1e-4, 5e-4, 128, 256, 128)
#train_ddpg(5e-4, 1e-4, 128, 256, 128)
#train_ddpg(1e-4, 5e-5, 128, 256, 128)
#train_ddpg(5e-5, 1e-4, 128, 256, 128)

# Batch size
train_ddpg(1e-4, 1e-4, 256, 256, 128)
#train_ddpg(1e-4, 1e-4, 128, 256, 128)
train_ddpg(1e-4, 1e-4, 64, 256, 128)

# Close the environment
env.close()
