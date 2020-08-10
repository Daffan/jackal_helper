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
  "world_name": "sequential_applr_testbed.world",
  "VLP16": "false",
  "gui": "false",
  "camera": "false",
  "wrapper": "reward_shaping",
  "wrapper_args": {
    "start_range": [[-1.5, -0.5], [-1.5, 1.5]],
    "goal_range": [[0.5, 1.5], [-1.5, 1.5]],
    "seed": 43,

    "reduction": 10,
    "polar_goal": "true",
    "centered_bin": "",
    "reward_shaping": "false",

    "goal_distance_reward": "true",
    "stuck_punishment": 0.5
  },
  "max_step": 300,
  "time_step": 1,
  "init_position": [-8, 0, 0],
  "goal_position": [54, 0, 0],
  "param_delta": [0.2, 0.3, 1, 2, 0.2, 0.2],
  "param_init": [1.5-0.2, 3.14-0.3, 6-1, 20-2, 0.75-0.2, 1-0.2],
  "param_list": ["max_vel_x", "max_vel_theta", "vx_samples", "vtheta_samples", "path_distance_bias", "goal_distance_bias"],
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
                                        param_delta = config['param_delta'],
                                        param_init = config['param_init'],
                                        param_list = config['param_list']
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
                    param_delta = config['param_delta'],
                    param_init = config['param_init'],
                    param_list = config['param_list']
                    )

rs = []
cs = []
succeed = 0
for i in range(avg):
    print("Running: %d/%d" %(i+1, avg), end="\r")
    r = 0
    obs = env.reset()
    done = False
    count = 0
    while not done:
        count += 1
        obs, reward, done, info = env.step(63)
        for init, pn in zip(env.param_init, env.param_list):
            env.navi_stack.set_navi_param(pn, init)
        r += reward
        print(reward, count)
    if count != config['max_step'] and reward != -1000:
        succeed += 1
        rs.append(r)
        cs.append(count)
print("succeed: %d/%d \t episode reward: %.2f \t steps: %d" %(succeed, avg, sum(rs)/float((len(rs))), sum(cs)/float((len(cs)))))

env.close()

######## About recording ###########
# Add the camera model to the world you used to train
# Check onetwofive_meter_split_camera.world
# The frames will be save to folder /tmp/camera_save_tutorial
# run
# ffmpeg -r 30 -pattern_type glob -i '/tmp/camera_save_tutorial/default_camera_link_my_camera*.jpg' -c:v libx264 my_camera.mp4
# to generate the video
