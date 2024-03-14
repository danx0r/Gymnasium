import time
import random
import argparse
import pickle
import gymnasium as gym
import numpy as np
import torch
from torch import nn

#
# hip_r, knee_r, foot_r, hip_l, knee_l, foot_l
# (mujoco calls them thigh, leg, foot)
# right is tan, left is purple
# hip: -1 extend (back bend) 1 contract (knee to nose)
# knee: -1 contract (heel to butt; kneeling), 1 extend (locked knee)
# foot: -1 extend (high heel/en pointe), 1 contract (walk on your heels)
#
def controller(obs):
    act = [0] * 6
    return act

def runn(env, steps):
    observation, info = env.reset()
    for ii in range(steps):
        action = controller(observation)
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
        runn(env, args.steps)
        env.close()
