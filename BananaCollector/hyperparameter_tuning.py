#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')

from unityagents import UnityEnvironment
#from agents import *
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
import pprint
import torch
import optuna
import time

from agents.ddqn_agent import DDQN

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Load parameters from file
hparams = config.HYPERPARAMS['DDQN']
params = config.TRAINPARAMS['BananaCollector']

hyperscores = []

def train_dqn(learning_rate, buffer_size, update_every): #batch_size, fc_units):

    # Set tunable parameters
    hparams['learning_rate'] = learning_rate
    hparams['update_every'] = int(update_every)
    hparams['buffer_size'] = int(buffer_size)
    #hparams['batch_size'] = int(batch_size)
    #hparams['fc1_units'] = int(fc_units)
    #hparams['fc2_units'] = int(fc_units)

    # Create agent instance
    print("Created agent with following hyperparameter values:")
    agent = DDQN(hparams)
    pprint.pprint(hparams)

    # ### 3. Train DQN agent!

    # Maximum number of training episodes
    n_episodes = params['n_episodes']
    # Initialize epsilon
    epsilon = params['epsilon_start']
    # List containing scores from each episode
    scores = []
    # Store last 100 scores
    scores_window = deque(maxlen=params['scores_window_size'])

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
            action = agent.act(state, epsilon)

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

        # Decay epsilon
        epsilon = max(params['epsilon_final'], params['epsilon_decay']*epsilon)

        # Print learning progress
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % params['scores_window_size'] == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=params['solve_score']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # Filename string
            filename = "{:s}_lr{:.1E}_buffer{:d}_update{:d}_solved{:d}"
            filename = filename.format(hparams['name'], hparams['learning_rate'], hparams['buffer_size'], hparams['update_every'], i_episode-100)
            torch.save(agent.qnetwork_local.state_dict(), 'models/{:s}.pth'.format(filename))
            break

    # Save score
    df = pd.DataFrame(scores,columns=['scores','average_scores'])
    df.to_csv('scores/{:s}.csv'.format(filename))
    # Plot scores
    plt.figure(figsize=(10,5))
    #plt.axhline(13, color='red', lw=1, alpha=0.3)
    plt.plot( df.index, 'scores', data=df, color='lime', lw=1, label="score", alpha=0.4)
    plt.plot( df.index, 'average_scores', data=df, color='green', lw=2, label="average score")
    # Set labels and legends
    plt.xlabel('Episode')
    plt.xlim(0, len(df.index))
    plt.xticks(50*np.arange(int(len(df.index)/50+1)))
    plt.ylabel('Score')
    plt.yticks(3*np.arange(8))
    plt.title('DDQN agent')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    # Save figure
    plt.savefig('graphs/{:s}.png'.format(filename), bbox_inches='tight')

    hyperscores.append([learning_rate, buffer_size, update_every, i_episode-100])
    temp_df = pd.DataFrame(hyperscores,columns=['learning_rate', 'buffer_size', 'update_every', 'i_episode'])#, 'batch_size', 'fc_units', 'i_episode'])
    temp_df.to_csv('scores/lr_buffersize_update_tuning.csv')

    time.sleep(1)
    return i_episode-100

def objective(trial):
    #Optuna objective function
    learning_rate = trial.suggest_categorical('learning_rate', [5e-5, 1e-4, 5e-4, 1e-3])
    buffer_size = trial.suggest_categorical('buffer_size', [50000, 100000, 200000])
    update_every = trial.suggest_categorical('update_every', [2, 4, 8])
    #batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    #fc_units = trial.suggest_categorical('fc_units', [32, 64])

    return train_dqn(learning_rate, buffer_size, update_every)#, batch_size, fc_units)

# Create a new Optuna study object.
study = optuna.create_study()
# Invoke optimization of the objective function.
study.optimize(objective , n_trials=1, n_jobs=1)

#Print and Save result to .csv file
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
df = study.trials_dataframe()
df.to_csv('scores/lr_buffersize_update_tuning_optuna.csv')

# Close the environment
env.close()
