# Deep Reinforcement Learning

<!-- Insert cool combined gif here-->

This repository contains implementations of Deep Reinforcement Learning algorithms.

## Projects

* [Navigation](https://github.com/Zulkhuu/reinforcement-learning/tree/master/BananaCollector): DQN agent to collect yellow bananas while avoiding blue bananas.
* [Continuous Control]| _Coming soon!_
* [Collaboration and Competition]| _Coming soon!_

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

- Continuous control - Train an robotic arm to reach target locations using actor critic method
- Collaboration and Competition - Train a pair of agents to play tennis

## Resources

### Online courses
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [Deep Reinforcement Learning UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [David Silver's course on reinforcement learning](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
* [Udacity Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/)

### Textbooks
* [The Deep Learning textbook](http://www.deeplearningbook.org/)
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
