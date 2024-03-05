import time
import random
import gymnasium as gym
import numpy as np
import torch

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
def controller(obs, step):
    act = g_model(torch.tensor(obs).float())
    return np.array(act.detach())


def main():
    env = gym.make("Walker2d-v4", render_mode="human", terminate_when_unhealthy=True)
    # env._max_episode_steps=1000
    observation, info = env.reset()

    time.sleep(2)

    resets = 0
    rewards = 0
    for _ in range(250):
        action = controller(observation, _)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        if VERBOSE:
            print (_, terminated, truncated)
        if terminated or truncated:
            if VERBOSE:
                print ("*************************RESET*************************")
                print (_, observation, "rewards:", rewards)
            observation, info = env.reset()
            resets += 1
            rewards = 0
        # time.sleep(.05)
    env.close()

    print ("Golf score:", resets)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("DEVICE:", device)
    # main()
    from torch import nn
    g_model = NeuralNetwork()
    t = torch.tensor([0.0] * 17)
    pred = g_model(t)
    print (pred)
    main()