import time
import math
# import sys
import random
import argparse
# import pickle
import gymnasium as gym
from gymnasium.utils.save_video import save_video
# import numpy as np
from numpy import random as np_random
# import torch
# from torch import nn

#
# See README.txt
#

NUM_ACTUATORS = 14
TARGET_VEL = 0.67

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

def runn(env, steps, params=None, quit_when_unhealthy=True):
    if params:
        if len(params) == 6:
            HIPRANGE_ADJ, HIPOFFSET_ADJ, SPEED_ADJ, DAMP, TORSO_LIN_ADJ, DAMP2 = params
        else:
            HIPRANGE_ADJ, HIPOFFSET_ADJ, SPEED_ADJ, DAMP = params
    speed = 0.04
    hip_l_range = 0.34
    hip_r_range = 0.34
    hip_l_offset = 0.56
    hip_r_offset = -0.56
    hip_l_phase = -math.pi / 2
    hip_r_phase = -math.pi / 2
    knee_l_range = 0.45
    knee_r_range = 0.45
    knee_l_offset = -0.62
    knee_r_offset = 0.62
    knee_l_phase = 0
    knee_r_phase = 0
    # foot_range = 0.25
    # foot_offset = .14
    # foot_l_phase = math.pi / 2
    # foot_r_phase = -math.pi / 2
    restpose = [-0.42, -0.5, -0.916, -0.001, 0.0, 0.22, 0.52, -0.5, 0.916, 0.001, 0.0, -0.22, 1.35, -1.35]

    controller = Controller()
    for i, j in enumerate(controller.joints.keys()):
        controller.goto(j, restpose[i])
    observation, info = env.reset(seed=SEED)
    bugg = 0
    frames = []
    for ii in range(steps):
        action = controller.update(observation)
        # if ii == 100 * (bugg+3):
        #     j = list(controller.joints.keys())[bugg]
        #     print (f"MODIFY joint: {bugg} {j}")
        #     controller.goto(j, 2)
        #     bugg += 1
        # if ii == 50:
        #     env.env.env.data.body("root").xfrc_applied[0]=510
        # if ii == 66:
        #     env.env.env.data.body("root").xfrc_applied[0]=0
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        if VERBOSE & 1:
            print ("STEP:", ii, "ACTION:", action, "\nOBSERVATION:\n", f"{observation[-53]:7.3f} {observation[-52]:7.3f} {observation[-51]:7.3f} {observation[-49]:7.3f} {observation[-47]:7.3f} {observation[-46]:7.3f}"
                   f" {observation[-45]:7.3f} {observation[-44]:7.3f} {observation[-43]:7.3f} {observation[-41]:7.3f} {observation[-40]:7.3f} {observation[-39]:7.3f}")

        tzpos = env.env.env.data.body("root").xpos[2]

        if quit_when_unhealthy and tzpos < -.75:
            break

        trop = 0#env.env.env.data.joint("rooty").qpos[0]
        trov = 0#env.env.env.data.joint("rooty").qvel[0]
        tpov = env.env.env.data.joint("rootx").qvel[0] + random.gauss(0, .01)
        tpac = env.env.env.data.joint("rootx").qacc[0]
        if VERBOSE and 2:
            print (f'{ii} trop {trop} trov {trov} tpov {tpov} tpac {tpac} tzpos {tzpos}')
        if ii == 80:
            controller.goto("hip_rx", -45)
            controller.goto("hip_lx", 45)
        if ii > 80:
            if params is not None:
                PD = trop + trov * DAMP + (tpov-TARGET_VEL) * TORSO_LIN_ADJ + tpac * TORSO_LIN_ADJ * DAMP2
                speed_use = speed + SPEED_ADJ * PD
                hip_l_range_use =  hip_l_range + HIPRANGE_ADJ * PD
                hip_r_range_use =  hip_r_range + HIPRANGE_ADJ * PD
                hip_l_offset_use =  hip_l_offset + HIPOFFSET_ADJ * PD
                hip_r_offset_use =  hip_r_offset - HIPOFFSET_ADJ * PD
            else:
                speed_use = speed
                hip_l_range_use = hip_l_range
                hip_r_range_use = hip_r_range
                hip_l_offset_use = hip_l_offset
                hip_r_offset_use = hip_r_offset
            hip_l = math.sin(ii * speed_use + hip_l_phase) * hip_l_range_use + hip_l_offset_use
            controller.goto('hip_ly', hip_l)
            hip_r = math.sin(ii * speed_use + hip_r_phase) * hip_r_range_use + hip_r_offset_use
            controller.goto('hip_ry', hip_r)
            knee_l = math.sin(ii * speed_use + knee_l_phase) * knee_l_range + knee_l_offset
            controller.goto('knee_l', knee_l)
            knee_r = math.sin(ii * speed_use + knee_r_phase) * knee_r_range + knee_r_offset
            controller.goto('knee_r', knee_r)

        if RECORDING:
            frames.append(env.render())

    if RECORDING:
        print (f"SAVING {len(frames)} frames of video")
        save_video(frames, "videos", fps=env.metadata["render_fps"])
    return ii

def train(env, steps, epochs, params, temp=0.2):
    global SEED
    total = 0
    best = 0
    best_params = None
    for i in range(epochs):
        save_params = list(params)
        for j in range(4, 6):
            params[j] += random.gauss(0, temp)
        print ("  RANDOMIZED params:", params)
        score = 999999
        random.seed(time.time()*1000)
        for k in range(3):          #take worst of 3
            params[4] += random.gauss(0, .00001)                 # Inject some entropy
            time_score = runn(env, steps, params)
            score1 = env.env.env.data.joint("rootx").qpos[0]
            print ("    SCORE of 3:", score1)
            if score1 < best:
                score = score1
                break
            if score1 < score:
                score = score1 
        if score > best:
            print ("  NEW BEST:", score)
            best = score
            best_params = list(params)
            save_params = params   # now search from new best
            temp *= 10
        else:
            temp *= 0.9
        print (f"  TRAINED epoch: {i} score {score} steps {time_score} temp: {temp} best {best} best params:{best_params}")
        total += score
        params = save_params

    average = total / (i+1)
    print (f"TRAINING DONE average loss: {average}")
    return best, best_params

if __name__ == "__main__":
    SEED = 123
    np_random.seed(SEED)
    random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--params", type = float, nargs=6)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no_quit", action="store_true")
    parser.add_argument("--temp", type = float, default=0.1)
    args = parser.parse_args()

    if args.params:
        params = args.params
    else:
        # params = [0.7154389233615354, -0.13465742540463885, 0.1647139466597792, 0.015743438166364233, 0.036960511142429694, 0.01629851169066194]
        params = [0, 0, 0, 0, 0, 0]

    VERBOSE = args.verbose
    SHOW = args.show
    RECORDING = args.record
    gym.register("Walker3d-v5", entry_point='gymnasium.envs.mujoco.walker3d_v5:Walker3dEnv')
    if args.train:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else "rgb_array", terminate_when_unhealthy=True)
        env._max_episode_steps=args.steps

        if args.params:
            params = args.params
        score, params = train(env, args.steps, args.epochs, params, args.temp)
        print ("TOP SCORE:", score, "TOP PARAMETERS:", params)
    else:
        env = gym.make("Walker3d-v5", render_mode="human" if SHOW else "rgb_array", terminate_when_unhealthy=False)
        env._max_episode_steps=args.steps
        observation, info = env.reset(seed=SEED)
        time.sleep(2)
        runn(env, args.steps, params=params, quit_when_unhealthy=not args.no_quit)
        # env.close()
