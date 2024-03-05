import time
import random
import gymnasium as gym
import torch

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
    print (f"torso height: {obs[0]}")
    if step < 350:
        act = [1, -1, 1, 1, -1, 1]
    elif step < 700:
        act = [-1, 1, 0, -1, 1, 0]
    else:
        act = [1, -1, 0, 1, -1, 0]
    return act


def main():
    env = gym.make("Walker2d-v4", render_mode="human", terminate_when_unhealthy=False)
    # env._max_episode_steps=1000
    observation, info = env.reset()

    time.sleep(2.5)

    resets = 0
    rewards = 0
    for _ in range(1000):
        action = controller(observation, _)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        print (_, terminated, truncated)
        if terminated or truncated:
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
    model = NeuralNetwork()
    t = torch.tensor([0.0, 0, 0, 0, 0, 0])
    pred = model(t)
    print (pred)