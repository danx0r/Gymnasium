import time
import random
import argparse
import pickle
import gymnasium as gym
import numpy as np
import torch
from torch import nn

class Servo:
    def __init__(self, P, D):
        self.P = P
        self.D = D
        self.target = 0.0

    def goto(self, target):
        self.target = target

    def update(self, pos, vel):
        perr = pos-self.target
        torque = perr * P + vel * D
        return torque        

#
# hip_r, knee_r, foot_r, hip_l, knee_l, foot_l
# (mujoco calls them thigh, leg, foot)
# right is tan, left is purple
# hip: -1 extend (back bend) 1 contract (knee to nose)
# knee: -1 contract (heel to butt; kneeling), 1 extend (locked knee)
# foot: -1 extend (high heel/en pointe), 1 contract (walk on your heels)
#
# observation[0] = torso height (z of center)
# observation[1] = torso angle
# observation[2:8] = angle of hip_r, knee_r, foot_r, hip_l, knee_l, foot
# observation[8] = torso x vel
# observation[9] = torso z vel
# observation[10] = torso ang vel
# observation[11:16] = ang vel of hip_r, knee_r, foot_r, hip_l, knee_l, foot
#
def controller(obs):
    # act = [(random.random() * 2 - 1) * .5 for x in range(6)]
    act = [1.0 for x in range(6)]
    return act

def runn(env, steps):
    observation, info = env.reset()
    for ii in range(steps):
        action = controller(observation)
        if ii >= 50:
            action[2] = -1.0
        observation, reward, terminated, truncated, info = env.step(action)
        # print ("DEBUG", terminated, truncated)
        if VERBOSE & 1:
            print (ii, "OBSERVATION:", observation[:5], "\nACTION:", action)
            print ()
        if terminated or truncated:
            break
    return ii

def train(env, steps, epochs):
    error = runn(env, steps)
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--learnrate", type=float, default=0.0001)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    SHOW = args.show
    if args.train:
        env = gym.make("Walker2d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=True)
        err = train(env, args.steps, args.epochs)
        print ("ERROR:", err)
    else:
        env = gym.make("Walker2d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=False)
        observation, info = env.reset()
        time.sleep(2)
        runn(env, args.steps)
        env.close()
