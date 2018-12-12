import sys
sys.path.append('../')

from utils.model import Network
from utils.OUNoise import OUNoise
from torch.optim import Adam
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG:
    def __init__(self, params):

        in_actor = int(params['state_size'])
        hidden_in_actor = int(params['fc1_units'])
        hidden_out_actor = int(params['fc2_units'])
        out_actor = int(params['action_size'])
        in_critic = int(params['n_agents'])*(in_actor + out_actor)
        hidden_in_critic = int(params['fc1_units'])
        hidden_out_critic = int(params['fc2_units'])
        lr_actor = params['lr_actor']
        lr_critic = params['lr_critic']
        '''
        print("in_actor:{}".format(in_actor))
        print("hidden_in_actor:{}".format(hidden_in_actor))
        print("hidden_out_actor:{}".format(hidden_out_actor))
        print("out_actor:{}".format(out_actor))
        print("in_critic:{}".format(in_critic))
        print("hidden_in_critic:{}".format(hidden_in_critic))
        print("hidden_out_critic:{}".format(hidden_out_critic))
        '''

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, params['random_seed'])

        # initialize targets same as original networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)


    def act(self, obs, noise=0.5):
        obs = torch.from_numpy(obs).float().to(device)
        action = self.actor(obs).cpu() + noise*self.noise.sample()
        return action.data.numpy()

    def target_act(self, obs, noise=0.5):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*torch.Tensor(self.noise.sample()).cuda()
        return action

    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
