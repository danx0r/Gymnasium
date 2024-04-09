import time
import math
import sys
import random
import argparse
import pickle
import gymnasium as gym
import numpy as np
import torch
from torch import nn

#
# See README.txt
#

NUM_ACTUATORS = 1

class Controller:
    joints = {
        'hip_lx': 0,
        # 'hip_lz': 1,
        # 'hip_ly': 2,
        # 'hip_rx': 3,
        # 'hip_rz': 4,
        # 'hip_ry': 5,
        # 'knee_l': 6,
        # 'knee_r': 7,
        # 'ankle_l': 8,
        # 'anlke_r': 9,
        # 'foot_l': 10,
        # 'foot_r': 11,
    }

    def __init__(self):
        self.act = [0] * NUM_ACTUATORS
    
    def update(self, obs):
        #
        # MAGIC goes here -- update actions based on observations
        #
        return self.act

    def goto(self, joint, target):
        self.act[self.joints[joint]] = target

def runn(env, steps, adjust=None):
    # speed = 0.032
    # hip_range = 0.4
    # hip_offset = 0.1
    # hip_l_phase = math.pi / 2
    # hip_r_phase = -math.pi / 2
    # knee_range = 0.56
    # knee_offset = 0.1
    # knee_l_phase = -math.pi / 2
    # knee_r_phase = math.pi / 2
    # foot_range = 0.25
    # foot_offset = .14
    # foot_l_phase = math.pi / 2
    # foot_r_phase = -math.pi / 2

    controller = Controller()
    controller.goto('hip_lx', -1)
    observation, info = env.reset()

    for ii in range(steps):
        print ("STEP:", ii, end="")
        action = controller.update(observation)
        if ii == 250:
            controller.goto('hip_lx', 1)
        observation, reward, terminated, truncated, info = env.step(action)
        if VERBOSE & 1:
            print ("ACTION:", action, "OBSERVATION:", observation[23])
        else:
            print()
        if terminated or truncated:
            break
        
        # hip_l = math.sin(ii * speed + hip_l_phase) * hip_range + hip_offset
        # controller.goto('hip_l', hip_l)
        # hip_r = math.sin(ii * speed + hip_r_phase) * hip_range + hip_offset
        # controller.goto('hip_r', hip_r)
        # knee_l = math.sin(ii * speed + knee_l_phase) * knee_range + knee_offset
        # controller.goto('knee_l', knee_l)
        # knee_r = math.sin(ii * speed + knee_r_phase) * knee_range + knee_offset
        # controller.goto('knee_r', knee_r)
        # foot_l = math.sin(ii * speed + foot_l_phase) * foot_range + foot_offset
        # controller.goto('foot_l', foot_l)
        # foot_r = math.sin(ii * speed + foot_r_phase) * foot_range + foot_offset
        # controller.goto('foot_r', foot_r)

    return ii

def train(env, steps, epochs, adjust=None):
    error = runn(env, steps, adjust)
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--adjust", type=float, default=1.)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    SHOW = args.show
    gym.register("Walker3d-v5", entry_point='gymnasium.envs.mujoco.walker3d_v5:Walker3dEnv')
    if args.train:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=True)
        env._max_episode_steps=args.steps
        err = train(env, args.steps, args.epochs, args.adjust)
        print ("ERROR:", err)
    else:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=False)
        env._max_episode_steps=args.steps
        observation, info = env.reset()
        time.sleep(2)
        runn(env, args.steps, args.adjust)
        env.close()
