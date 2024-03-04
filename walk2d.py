import time
import random
import gymnasium as gym

def controller(observation):
    GAIN = 1.0
    P = 5 * GAIN
    D = 8 * GAIN
    A = 16 * GAIN
    V = 10 * GAIN
    pos, vel, ang, angv  = observation
    ctl = pos * P + vel * D + ang * A + angv * V
    action = 1 if ctl + 0.5 > random.random() else 0
    return action


env = gym.make("Walker2d-v4", render_mode="human")
env._max_episode_steps=1000
observation, info = env.reset()

resets = 0
rewards = 0
for _ in range(1000):
    action = controller(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    #print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated or abs(observation[0]) > 0.15:
        print ("*************************RESET*************************")
        print (_, observation, "rewards:", rewards)
        observation, info = env.reset()
        resets += 1
        rewards = 0
    #time.sleep(.005)
env.close()

print ("Golf score:", resets)
