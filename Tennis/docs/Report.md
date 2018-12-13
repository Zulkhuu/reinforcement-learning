[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Project: Tennis

## Contents
- [Introduction](#Introduction)
  * [Problem definition](#Problem-definition)
- [Background](#Background)
  * [MADDPG algorithm](#MADDPG-algorithm)
- [Implementation](#Implementation)
- [Hyperparameter tuning](#Hyperparameter-tuning)
- [Result](#Result)
- [Future work](#Future-work)
- [References](#References)

# Introduction


For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

<p align="center">
  <img src="images/trained_agent.gif" height="300px">
</p>

## Problem definition


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Background

Most straightforward approach to learning in multi-agent environment is to train independently learning agents. However both value based and policy gradient methods perform bad in such situations. One issue is that agent’s policy changes during training, resulting in a non-stationary environment and preventing the naïve application of experience replay according to the [Multi Agent DDPG paper](https://arxiv.org/pdf/1706.02275.pdf).

## MADDPG algorithm

Multi Agent Deep Deterministic Policy Gradient(MADDPG) algorithm addresses those challenges faced by multi-agent learning by adapting DDPG algorithm to use in multi-agent setting.
Novelty of MADDPG algorithm lies in its centralized training and decentralized execution paradigm. As shown in below image, critic of each agent observes the environment fully, while actor acts based on only its own observation.

<p align="center">
  <img src="images/maddpg_diagram.gif" height="300px">
</p>

Another difference from using independent agents, is that MADDPG uses shared replay buffer between agents which holds full environment states, and each agents actions and rewards. Below image illustrates full MADDPG algorithm:

<p align="center">
  <img src="images/maddpg_algorithm.png" height="400px">
</p>

# Implementation

Implementation uses Reacher project's DDPG Implementation as a baseline and modifies it to align with MADDPG algorithm.

Full implementation of MADDPG agent can be found in [maddpg_agent.py]() file.

# Hyperparameter tuning

# Result

# Future work

# References

- [Continuous control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
- [Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Reinforcement Learning UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [Advanced Deep Learning & Reinforcement Learning](https://www.youtube.com/watch?v=iOh7QUZGyiU&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
- [Udacity Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
