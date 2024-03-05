import time
import random
import argparse
import pickle
import gymnasium as gym
import numpy as np
import torch
from torch import nn

VERBOSE = True

from classes import *

#
# hip_r, knee_r, foot_r, hip_l, knee_l, foot_l
# (mujoco calls them thigh, leg, foot)
# right is tan, left is purple
# hip: -1 extend (back bend) 1 contract (knee to nose)
# knee: -1 contract (heel to butt; kneeling), 1 extend (locked knee)
# foot: -1 extend (high heel/en pointe), 1 contract (walk on your heels)
#
def controller(obs):
    act = g_model(torch.tensor(obs).float())
    return np.array(act.detach())


def run():
    env = gym.make("Walker2d-v4", render_mode="human", terminate_when_unhealthy=False)
    # env._max_episode_steps=1000
    observation, info = env.reset()

    time.sleep(2)

    for _ in range(100000):
        action = controller(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if VERBOSE:
            print (_, "OBSERVATION:", observation[:5], "\nACTION:", action)
            print ()
        # time.sleep(.05)
    env.close()

def train():
    env = gym.make("Walker2d-v4", render_mode="human", terminate_when_unhealthy=True)
    # env._max_episode_steps=1000
    observation, info = env.reset()

    time.sleep(2)

    resets = 0
    rewards = 0
    for _ in range(250):
        action = controller(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        if VERBOSE:
            print (_, "OBSERVATION:", observation[:5], "\nACTION:", action)
            print ()
        if terminated or truncated:
            observation, info = env.reset()
            resets += 1
            rewards = 0
            if VERBOSE:
                print ("*************************RESET*************************")
                print (_, f"resets: {resets} rewards: {rewards}:")
        # time.sleep(.05)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--saveevery", type=int, default=100)
    parser.add_argument("--learnrate", type=float, default=0.0001)
    parser.add_argument("--model", default="testmodel.pkl")
    parser.add_argument("--device") #cuda or cpu
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("DEVICE:", device)

    if not args.train:
        f = open(args.model, 'rb')
        g_model = pickle.load(f)
        f.close()
        run()
    else:
        g_model = NeuralNetwork()
        t = torch.tensor([0.0] * 17)
        pred = g_model(t)
        print (pred)
        train()