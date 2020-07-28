import jackal_envs
import gym
from jackal_sim_wrapper import *

import numpy

import argparse
import os
import json

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--seed', dest='seed', type = int, default = 43)
parser.add_argument('--avg', dest='avg', type = int, default = 20)

args = parser.parse_args()
record = args.record
gui = 'true' if args.gui else 'false'
seed = args.seed
avg = args.avg

config = {
  "section": "ACKTR_random_start_goal",
  "world_name": "85cm_split.world",
  "VLP16": "false",
  "gui": "false",
  "init_position": [-1, -1, 0],
  "goal_position": [0, 2, 0],
  "wrapper": "",
  "wrapper_args": {
    "start_range": [[-1.5, -0.5], [-1.5, 1.5]],
    "goal_range": [[0.5, 1.5], [-1.5, 1.5]],
    "seed": 43
  },
  "max_step": 50,
  "time_step": 1,
  "init_max_vel_x": 0.1,
  "max_vel_x_delta": 0.2,
  "total_steps": 50000,
  "policy_network": "MlpPolicy",
  "algorithm": "ACKTR",
  "lr_schedule": "constant",
  "learning_rate": 0.1,
  "gamma": 0.95
}

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

rs = []
succeed = 0
for i in range(avg):
    print("Running: %d/%d" %(i+1, avg), end="\r")
    r = 0
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(0)
        r += reward
    if r != -config['max_step']:
        succeed += 1
    rs.append(r)
print("max_vel: %.2f \t succeed: %d/%d \t episode reward: %.2f" %(config['init_max_vel_x'], succeed, avg, sum(rs)/float((len(rs)))))

env.close()

######## About recording ###########
# Add the camera model to the world you used to train
# Check onetwofive_meter_split_camera.world
# The frames will be save to folder /tmp/camera_save_tutorial
# run
# ffmpeg -r 30 -pattern_type glob -i '/tmp/camera_save_tutorial/default_camera_link_my_camera*.jpg' -c:v libx264 my_camera.mp4
# to generate the video
