# Deep Reinforcement Learning

This repository contains implementations of Deep Reinforcement Learning algorithms.

## Projects using Unity ML-Agents

* [The Taxi Problem]| _Coming soon!_
* [Navigation](https://github.com/Zulkhuu/reinforcement-learning/navigation): DQN agent to collect yellow bananas while avoiding blue bananas. Solved in 381 episodes
* [Continuous Control]| _Coming soon!_
* [Collaboration and Competition]| _Coming soon!_

### Resources

* [Cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning.

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with **Tile Coding** | _Coming soon!_
- `Cartpole-v0` with **Hill Climbing** | _Coming soon!_
- `Cartpole-v0` with **REINFORCE** | _Coming soon!_
- `MountainCarContinuous-v0` with **Cross-Entropy Method** | _Coming soon!_
- `MountainCar-v0` with **Uniform-Grid Discretization** | _Coming soon!_
- `Pendulum-v0` with **Deep Deterministic Policy Gradients (DDPG)** | _Coming soon!_

### Box2d
- `BipedalWalker-v2` with **Deep Deterministic Policy Gradients (DDPG)** | _Coming soon!_
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with **Deep Q-Networks (DQN)** | _Coming soon!_

### Toy Text
- `FrozenLake-v0` with **Dynamic Programming** | _Coming soon!_
- `Blackjack-v0` with **Monte Carlo Methods** | _Coming soon!_
- `CliffWalking-v0` with **Temporal-Difference Methods** | _Coming soon!_

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
