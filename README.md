# Deep Reinforcement Learning

<!-- Insert cool combined gif here-->

This repository contains modular implementations of Deep Reinforcement Learning algorithms.

## Projects based on Unity Environment

* [Banana Collector](https://github.com/Zulkhuu/reinforcement-learning/tree/master/BananaCollector): DQN agent collects yellow bananas while avoiding blue bananas.
* [Reacher](https://github.com/Zulkhuu/reinforcement-learning/tree/master/Reacher): DDPG agent controls double-jointed arm to reach ball.
* [Tennis](https://github.com/Zulkhuu/reinforcement-learning/tree/master/Tennis): MADDPG agents control rackets to play Tennis.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Install [conda](https://conda.io/docs/user-guide/install/)

2. Create and activate a new environment with Python 3.6.

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

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

4. Clone the repository and install several dependencies.
```bash
git clone https://github.com/Zulkhuu/reinforcement-learning.git
cd python
pip install .
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

## Future work

- Collaboration and Competition - Train a pair of agents to play soccer

## Resources

### Online courses
* [UC Berkeley's Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [UC Berkeley's Deep Reinforcement Learning course](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [David Silver's course on reinforcement learning](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
* [DeepMind's Advanced Deep Learning & Reinforcement Learning course](https://www.youtube.com/watch?v=iOh7QUZGyiU&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)
* [Udacity Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/)

### Textbooks
* [Ian Goodfellow: The Deep Learning textbook](http://www.deeplearningbook.org/)
* [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
* [Miguel Morales: Grokking Deep Reinforcement Learning](https://www.manning.com/books/grokking-deep-reinforcement-learning)

### Papers
* [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044)
* [Neural Episodic Control](https://arxiv.org/pdf/1703.01988)
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
* [Continuous control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
