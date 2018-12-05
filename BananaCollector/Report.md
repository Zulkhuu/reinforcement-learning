# Project: Banana collector

## Contents
- [Introduction](#Introduction)
  * [Problem definition](#Problem-definition)
- [Background](#Background)
  * [DQN algorithm](#DQN-algorithm)
  * [DDQN algorithm](#DDQN-algorithm)
- [Implementation](#Implementation)
- [Hyperparameter tuning](#Hyperparameter-tuning)
- [Result](#Result)
- [Future work](#Future-work)
- [References](#References)

# Introduction

For this project, we will train an DQN agent to navigate and collect bananas in a [Unity's Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) world.  

## Problem definition

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and scoring average score of +13 over 100 consecutive episodes is considered solving the environment.

# Background

DQN(Deep Q-Networks) uses Neural Network that approximates Q-values based on environment's current state as input. This makes it not only possible to approximate large state space environments, which is not feasible with traditional table based Reinforcement Learning methods, but also it brings superhuman performance in some environments.

## DQN algorithm
Just replacing tables with neural network doesn't bring superhuman performance. According to original [DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), it takes couple of techniques and tricks in order to reach human level performance. Two of the most impactful techniques are: Experience replay and Fixed Q-target.

By using these techniques DQN algorithm's sampling and learning becomes separated as shown in below images:

<!--![alt text](https://cdn-images-1.medium.com/max/1600/1*P33Kshj2iTt-C2FSYxE6wg.png "https://cdn-images-1.medium.com/max/1600/1*P33Kshj2iTt-C2FSYxE6wg.png") -->

<p align="center">
    <img src="assets/dqn.png" height="400px">
</p>

## DDQN algorithm
DDQN(Double DQN) algorithm improves original DQN' algorithm's performance by eradicating its overestimation bias problem. In the paper "Deep Reinforcement Learning with Double Q-learning", it is shown that overestimation bias is resulted from using same Q-Network parameters for both selecting and evaluating next action as shown below.

<p align="center">
    <img src="assets/dqn_update.png" height="150px">
</p>

Double Q Learning solves this issue by using DQN's target Q-Network's parameters when evaluating target value.

<p align="center">
    <img src="assets/ddqn_update.png" height="100px">
</p>

# Implementation

Baseline code for this project is a PyTorch based implementation of vanilla DQN agent for solving OpenAI Gym's Lunar Lander problem. Original source code can be found from below Udacity's Deep Reinforcement Learning nanodegree course's GitHub [page](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution).



# Hyperparameter tuning

For tuning some of the most important hyperparameters, A hyperparameter optimization framework [Optuna](https://optuna.org/) was used. Since Optuna framework comes as a native Python package, it can be integrated effortlessly with current implementation.

Optuna' interface requires only the following:
- Objective function to minimize
- Tunable hyperparameters with its search space

First try will be to run hyperparameter tuning with following setup:
- Objective function to minimize: Number of episodes that it takes to solve the environment
- Hyperparameters:
  - Learning rate: in range [5e-5 to 1e-4]
  - Batch size [32, 64, 128, 256]
  - Q-network size [32, 64]

# Result



# Future work

Implement other algorithms that improve DQN agent's performance.
- Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581)
- Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952)
- Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295)
- Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298)
- Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044)
- Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988)

# References

- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [Deep Reinforcement Learning UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [Udacity Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
