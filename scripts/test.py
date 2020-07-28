import jackal_envs
import gym
from jackal_sim_wrapper import *

import numpy
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

import argparse
from datetime import datetime
import time
import os
import json

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--model', dest = 'model', type = str, default = '../results/ACKTR_random_start_goal_2020_07_12_16_08', help = 'path to the saved model and configuration')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--seed', dest='seed', type = int, default = 43)
parser.add_argument('--avg', dest='avg', type = int, default = 20)


args = parser.parse_args()
model_path = args.model
record = args.record
gui = 'true' if args.gui else 'false'
seed = args.seed
avg = args.avg

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")

if not model_path.startswith('../'):
    config_path = '../results/' + model_path + '/config.json'
else:
    config_path = model_path + '/config.json'

if not model_path.startswith('../'):
    model_path = '../results/' + model_path + '/model.zip'
else:
    model_path = model_path + '/model.zip'

with open(config_path, 'rb') as f:
    config = json.load(f)

if record:
    config['world_name'] = config['world_name'].split('.')[0] + '_camera' + '.world'

if config['wrapper']:
    config['wrapper_args']['seed'] = seed
    env = wrapper_dict[config['wrapper']](gym.make('jackal_navigation-v0',
                                        gui = gui,
                                        VLP16 = config['VLP16'],
                                        world_name = config['world_name'],
                                        init_position = config['init_position'],
                                        goal_position = config['goal_position'],
                                        max_step = config['max_step'],
                                        time_step = config['time_step'],
                                        init_max_vel_x = config['init_max_vel_x'],
                                        max_vel_x_delta = config['max_vel_x_delta']
                                        ), config['wrapper_args'])
else:
    env = gym.make('jackal_navigation-v0',
                    gui = gui,
                    VLP16 = config['VLP16'],
                    world_name = config['world_name'],
                    init_position = config['init_position'],
                    goal_position = config['goal_position'],
                    max_step = config['max_step'],
                    time_step = config['time_step'],
                    init_max_vel_x = config['init_max_vel_x'],
                    max_vel_x_delta = config['max_vel_x_delta']
                    )

if config['algorithm'] == 'ACKTR':
    model = ACKTR.load(model_path)
elif config['algorithm'] == 'PPO2':
    model = PPO2.load(model_path)
elif config['algorithm'] == 'DQN':
    model = DQN.load(model_path)

rs = []
succeed = 0
for i in range(avg):
    print("Running: %d/%d" %(i+1, avg), end="\r")
    r = 0
    obs = env.reset()
    done = False
    actions = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        r += reward
    if r != -config['max_step']:
        succeed += 1
    rs.append(r)
print("max_vel: %.2f \t succeed: %d/%d \t episode reward: %.2f" %(config['init_max_vel_x'], succeed, avg, sum(rs)/float((len(rs)))))

env.close()

from matplotlib import pyplot as plt
vm = config['init_max_vel_x']
vms = [vm]
for a in actions:
    if a == 0:
        pass
    elif a == 1:
        vm += config['max_vel_x_delta']
        vm = min(2, vm)
    elif a == 2:
        vm -= config['max_vel_x_delta']
        vm = max(0.1, vm)
    vms.append(vm)
plt.plot(vms)
plt.show()

print(vms)

######## About recording ###########
# Add the camera model to the world you used to train
# Check onetwofive_meter_split_camera.world
# The frames will be save to folder /tmp/camera_save_tutorial
# run
# ffmpeg -r 30 -pattern_type glob -i '/tmp/camera_save_tutorial/default_camera_link_my_camera*.jpg' -c:v libx264 my_camera.mp4
# to generate the video
