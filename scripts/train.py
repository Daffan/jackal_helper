import jackal_envs
import gym
from jackal_sim_wrapper import *

import numpy
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import BaseCallback
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

if not config_path.startswith('../'):
    config_path = '../configs/' + config_path

with open(config_path, 'rb') as f:
    config = json.load(f)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)


if config['wrapper']:
    env = wrapper_dict[config['wrapper']](gym.make('jackal_navigation-v0',
                                        gui = config['gui'],
                                        VLP16 = config['VLP16'],
                                        world_name = config['world_name'],
                                        init_position = config['init_position'],
                                        goal_position = config['goal_position'],
                                        max_step = config['max_step'],
                                        time_step = config['time_step'],
                                        param_delta = config['param_delta'],
                                        param_init = config['param_init'],
                                        param_list = config['param_list']
                                        ), config['wrapper_args'])
else:
    env = gym.make('jackal_navigation-v0',
                    gui = config['gui'],
                    VLP16 = config['VLP16'],
                    world_name = config['world_name'],
                    init_position = config['init_position'],
                    goal_position = config['goal_position'],
                    max_step = config['max_step'],
                    time_step = config['time_step'],
                    param_delta = config['param_delta'],
                    param_init = config['param_init'],
                    param_list = config['param_list']
                    )

class SaveEveryStepIntervalsCallback(BaseCallback):

    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveEveryStepIntervalsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, 'model_%d' %(self.n_calls)))

        return True
callback = SaveEveryStepIntervalsCallback(5000, save_path)

if config['algorithm'] == 'ACKTR':
    env = make_vec_env(lambda: env, n_envs=1)
    model = ACKTR(config['policy_network'], env,
                    lr_schedule = config['lr_schedule'],
                    learning_rate = config['learning_rate'],
                    gamma = config['gamma'], policy_kwargs = config['policy_kwargs'],
                    verbose=1, tensorboard_log = save_path)

elif config['algorithm'] == 'PPO2':
    env = make_vec_env(lambda: env, n_envs=1)
    model = PPO2(config['policy_network'], env,
                    learning_rate = config['learning_rate'],
                    gamma = config['gamma'], policy_kwargs = config['policy_kwargs'],
                    verbose=1, tensorboard_log = save_path)

elif config['algorithm'] == 'DQN':
    model = DQN(config['policy_network'], env,
                    learning_rate = config['learning_rate'],
                    buffer_size = config['buffer_size'],
                    target_network_update_freq = 128,
                    gamma = config['gamma'], # policy_kwargs = config['policy_kwargs'],
                    verbose=1, tensorboard_log = save_path)


model.learn(config['total_steps'], callback = callback)
model.save(os.path.join(save_path, 'model'))

env.close()
