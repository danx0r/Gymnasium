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
        torque = perr * self.P + vel * self.D
        # print ("TORQUE:", torque)
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
class Controller:
    PDvals = [
        (-5.0, -0.3),   #hip_r
        (-3.0, -0.2),   #knee_r
        (-1.0, -0.1),   #foot_r
        (-5.0, -0.3),   #hip_l
        (-3.0, -0.2),   #knee_l
        (-1.0, -0.1),   #foot_l
        ]
    
    joints = {
        'hip_r': 0,
        'knee_r': 1,
        'foot_r': 2,
        'hip_l': 3,
        'knee_l': 41,
        'foot_l': 5,
    }

    def __init__(self):
        self.servos = [None] * 6
        self.act = [0] * 6
        for i in range(6):
            self.servos[i] = Servo(self.PDvals[i][0], self.PDvals[i][1])
    
    def update(self, obs):
        for i in range(6):
            pos = obs[2 + i]
            vel = obs[11 + i]
            self.act[i] = self.servos[i].update(pos, vel)
        return self.act

    def goto(self, joint, target):
        self.servos[self.joints[joint]].goto(target)
 
def runn(env, steps):
    controller = Controller()
    controller.goto('foot_l', .15)
    controller.goto('foot_r', .15)
    observation, info = env.reset()
    for ii in range(steps):
        action = controller.update(observation)
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
