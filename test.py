import time
import random
import gymnasium as gym

P = 5
D = 8
A = 10
V = 10

env = gym.make("CartPole-v1", render_mode="human")
env._max_episode_steps=1000
observation, info = env.reset()

resets = 0
rewards = 0
for _ in range(1000):
    pos, vel, ang, angv  = observation
    ctl = pos * P + vel * D + ang * A + angv * V
    #print (ctl)
    action = 1 if ctl + 0.5 > random.random() else 0
    observation, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    #print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated or abs(observation[0]) > 0.1:
        print ("*************************RESET*************************")
        print (_, observation, "rewards:", rewards)
        observation, info = env.reset()
        resets += 1
        rewards = 0
    #time.sleep(.005)
env.close()

print ("Golf score:", resets)
