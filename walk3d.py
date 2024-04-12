import time
import math
# import sys
# import random
import argparse
# import pickle
import gymnasium as gym
from gymnasium.utils.save_video import save_video
# import numpy as np
# import torch
# from torch import nn

#
# See README.txt
#

NUM_ACTUATORS = 14

class Controller:
    joints = {
        'hip_rx': 0,
        'hip_rz': 1,
        'hip_ry': 2,
        'knee_r': 3,
        'ankle_r': 4,
        'foot_r': 5,
        'hip_lx': 6,
        'hip_lz': 7,
        'hip_ly': 8,
        'knee_l': 9,
        'ankle_l': 10,
        'foot_l': 11,
        'shoulder_l': 12,
        'shoulder_r': 13,
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
    speed = 0.038
    hip_l_range = 0.33
    hip_r_range = 0.33
    hip_r_offset = -0.4
    hip_r_phase = -math.pi / 2
    hip_l_offset = 0.4
    hip_l_phase = -math.pi / 2
    knee_l_range = 0.5
    knee_r_range = 0.5
    knee_r_offset = 0.7
    knee_r_phase = 0
    knee_l_offset = -0.7
    knee_l_phase = 0
    # foot_range = 0.25
    # foot_offset = .14
    # foot_l_phase = math.pi / 2
    # foot_r_phase = -math.pi / 2
    restpose = [-0.45, -0.5, -0.916, -0.001, 0.0, 0.22, 0.45, -0.5, 0.916, 0.001, 0.0, -0.22, 1.35, -1.35]

    controller = Controller()
    for i, j in enumerate(controller.joints.keys()):
        controller.goto(j, restpose[i])
    observation, info = env.reset()
    bugg = 0
    frames = []
    for ii in range(steps):
        action = controller.update(observation)
        # if ii == 100 * (bugg+3):
        #     j = list(controller.joints.keys())[bugg]
        #     print (f"MODIFY joint: {bugg} {j}")
        #     controller.goto(j, 2)
        #     bugg += 1
        if ii == 50:
            env.env.env.data.body("root").xfrc_applied[0]=500
        if ii == 66:
            env.env.env.data.body("root").xfrc_applied[0]=0
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        if VERBOSE & 1:
            print ("STEP:", ii, "ACTION:", action, "\nOBSERVATION:\n", f"{observation[-53]:7.3f} {observation[-52]:7.3f} {observation[-51]:7.3f} {observation[-49]:7.3f} {observation[-47]:7.3f} {observation[-46]:7.3f}"
                   f" {observation[-45]:7.3f} {observation[-44]:7.3f} {observation[-43]:7.3f} {observation[-41]:7.3f} {observation[-40]:7.3f} {observation[-39]:7.3f}")
        # print ("DEBUG joint position:", env.env.env.data.joint("joint_left_arm_2_x8_1_dof_x8").qpos[0])

        if ii > 80:
            hip_l = math.sin(ii * speed + hip_l_phase) * hip_l_range + hip_l_offset
            controller.goto('hip_ly', hip_l)
            hip_r = math.sin(ii * speed + hip_r_phase) * hip_r_range + hip_r_offset
            controller.goto('hip_ry', hip_r)
            knee_l = math.sin(ii * speed + knee_l_phase) * knee_l_range + knee_l_offset
            controller.goto('knee_l', knee_l)
            knee_r = math.sin(ii * speed + knee_r_phase) * knee_r_range + knee_r_offset
            controller.goto('knee_r', knee_r)
            # foot_l = math.sin(ii * speed + foot_l_phase) * foot_range + foot_offset
            # controller.goto('foot_l', foot_l)
            # foot_r = math.sin(ii * speed + foot_r_phase) * foot_range + foot_offset
            # controller.goto('foot_r', foot_r)

        if RECORDING:
            frames.append(env.render())

    if RECORDING:
        print (f"SAVING {len(frames)} frames of video")
        save_video(frames, "videos", fps=env.metadata["render_fps"])
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
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    SHOW = args.show
    RECORDING = args.record
    gym.register("Walker3d-v5", entry_point='gymnasium.envs.mujoco.walker3d_v5:Walker3dEnv')
    if args.train:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else "rgb_array", terminate_when_unhealthy=True)
        env._max_episode_steps=args.steps
        err = train(env, args.steps, args.epochs, args.adjust)
        print ("ERROR:", err)
    else:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else "rgb_array", terminate_when_unhealthy=False)
        env._max_episode_steps=args.steps
        observation, info = env.reset()
        time.sleep(2)
        runn(env, args.steps, args.adjust)
        # env.close()
