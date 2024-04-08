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

NUM_SERVOS = 10

class Servo:
    def __init__(self, P, D, name):
        self.P = P
        self.D = D
        self.target = 0.0
        self.name = name

    def goto(self, target):
        self.target = target

    def update(self, pos, vel):
        perr = pos-self.target
        torque = perr * self.P + vel * self.D
        return torque        

class Controller:
    PGAIN = 10
    DGAIN = 1
    PDvals = [
        (-PGAIN, -DGAIN),   #hip L1
        (-PGAIN, -DGAIN),   #hip L2
        (-PGAIN, -DGAIN),   #hip R1
        (-PGAIN, -DGAIN),   #hip R2
        (-PGAIN, -DGAIN),  #knee_L
        (-PGAIN, -DGAIN),  #knee_R
        (-PGAIN, -DGAIN),   #ankle L
        (-PGAIN, -DGAIN),  #ankle_R
        (-PGAIN, -DGAIN),  #foot L
        (-PGAIN, -DGAIN),  #foot_R
        ]

    joints = {
        'hip_l1': 0,
        'hip_l2': 1,
        'hip_r1': 2,
        'hip_r2': 3,
        'knee_l': 4,
        'knee_r': 5,
        'ankle_l': 6,
        'anlke_r': 7,
        'foot_l': 8,
        'foot_r': 9,
    }

    def __init__(self):
        self.servos = [None] * NUM_SERVOS
        self.act = [0] * NUM_SERVOS
        for i in range(NUM_SERVOS):
            self.servos[i] = Servo(self.PDvals[i][0], self.PDvals[i][1], list(self.joints.keys())[i])
    
    def update(self, obs):
        for i in range(NUM_SERVOS):
            pos = 0
            vel = 0
            self.act[i] = self.servos[i].update(pos, vel)
        return self.act

    def goto(self, joint, target):
        self.servos[self.joints[joint]].goto(target)

    def adjust_gain(self, adj):
        if VERBOSE & 2:
            print ("adjust gain:", adj)
        for s in self.servos:
            s.P *= adj
            s.D *= adj
 
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
    # controller.goto('foot_l', .03)
    # controller.goto('foot_r', .02)
    # controller.goto('hip_l', .25)
    # controller.goto('hip_r', .25)
    observation, info = env.reset()

    for ii in range(steps):
        print ("STEP:", ii)
        action = controller.update(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if VERBOSE & 1:
            print (ii, "OBSERVATION:", observation[:5], "\nACTION:", action)
            print ()
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
