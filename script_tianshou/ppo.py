import sys
import jackal_envs
import gym
from jackal_envs.jackal_sim_wrapper import *

import numpy
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from offpolicy import offpolicy_trainer

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import pickle
import argparse
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--config', dest = 'config_path', type = str, default = '../configs/ppo.json', help = 'path to the configuration file')
parser.add_argument('--save', dest = 'save_path', type = str, default = '../results/', help = 'path to the saving folder')

args = parser.parse_args()
config_path = args.config_path
save_path = args.save_path

# Load the config files
if not config_path.startswith('../'):
    config_path = '../configs/' + config_path

with open(config_path, 'rb') as f:
    config = json.load(f)

env_config = config['env_config']
wrapper_config = config['wrapper_config']
training_config = config['training_config']

# Config logging
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(save_path)
with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)

# initialize the env --> num_env can only be one right now
env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_navigation-v0', **env_config), wrapper_config['wrapper_args'])
train_envs = DummyVectorEnv([lambda: env for _ in range(1)])

# config random seed
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
train_envs.seed(config['seed'])

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(training_config['layer_num'], state_shape, action_shape, config['device']).to(config['device'])
actor = Actor(net, action_shape).to(config['device'])
critic = Critic(net).to(config['device'])

# orthogonal initialization
for m in list(actor.modules()) + list(critic.modules()):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)

optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                            lr=training_config['learning_rate'])
dist = torch.distributions.Categorical

policy = PPOPolicy(
        actor, critic, optim, dist,
        training_config['gamma'],
        eps_clip=training_config["eps_clip"],
        vf_coef=training_config["vf_coef"],
        ent_coef=training_config["ent_coef"],
        action_range=None,
        gae_lambda=training_config["gae_lambda"],
        reward_normalization=training_config["rew_norm"],
        dual_clip=None,
        value_clip=training_config["value_clip"])

buf = ReplayBuffer(training_config['buffer_size'])
train_collector = Collector(policy, train_envs, buf)

train_fn =lambda e: [torch.save(policy.state_dict(), os.path.join(save_path, 'policy_%d.pth' %(e)))]

result = onpolicy_trainer(
        policy, train_collector, training_config['epoch'],
        training_config['step_per_epoch'], training_config['collect_per_step'],
        training_config['repaet_per_step'], training_config['batch_size'],
        train_fn=train_fn, writer=writer)

env.close()
