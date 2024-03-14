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
        if VERBOSE & 1:
            print (ii, "OBSERVATION:", observation[:5], "\nACTION:", action)
            print ()
        # time.sleep(.05)

# def train(steps):
#     env = gym.make("Walker2d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=True)
#     # env._max_episode_steps=1000
#     observation, info = env.reset()

#     time.sleep(.2)

#     for ii in range(250):
#         action = controller(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
#         rewards += reward
#         if VERBOSE & 2:
#             print (ii, "OBSERVATION:", observation[:5], "\nACTION:", action)
#             print ()
#         if terminated or truncated:
#             steps = ii-start
#             start = ii
#             observation, info = env.reset()
#             resets += 1
#             rewards = 0
#             if VERBOSE & 4:
#                 print ("*************************RESET*************************")
#                 print (ii, f"resets: {resets} rewards: {rewards} steps: {steps} loss: {squash(steps*0.1)}")
#         # time.sleep(.05)
#     env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--learnrate", type=float, default=0.0001)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    SHOW = args.show
    if args.train:
        pass
    else:
        env = gym.make("Walker2d-v5", render_mode="human" if SHOW else None, terminate_when_unhealthy=False)
        runn(env, args.steps)
        env.close()
