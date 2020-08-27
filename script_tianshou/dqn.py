import sys
import jackal_envs
import gym
from jackal_sim_wrapper import *

import numpy
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from offpolicy import offpolicy_trainer

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import pickle
import argparse
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--config', dest = 'config_path', type = str, default = '../configs/default.json', help = 'path to the configuration file')
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
training_config = config['trainig_config']

# Config logging
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)

# initialize the env --> num_env can only be one right now
env = wrapper_dict[config['wrapper']](gym.make('jackal_navigation-v0', **env_config), config['wrapper_args'])
train_envs = DummyVectorEnv([lambda: env for _ in range(1)])

# config random seed
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
train_envs.seed(config['seed'])

net = Net(training_config['layer_num'], env.state_shape, env.action_shape, config['device']).to(args.device)
optim = torch.optim.Adam(net.parameters(), lr=args.lr)
policy = DQNPolicy(
        net, optim, training_config['gamma'], training_config['n_step'],
        target_update_freq=training_config['target_update_freq'])

if trainig_config['prioritized_replay']:
    buf = PrioritizedReplayBuffer(
            training_config['buffer_size'],
            alpha=training_config['alpha'], beta=training_config['beta'])
else:
    buf = ReplayBuffer(training_config['buffer_size'])

train_collector = Collector(policy, train_envs, buf)
train_collector.collect(n_step=training_config['batch_size'])

result = offpolicy_trainer(
        policy, train_collector, training_config['epoch'],
        training_config['step_per_epoch'], training_config['collect_per_step'],
        training_config['batch_size'], train_fn=train_fn, writer=writer)

env.close()
