import time
import random
import gymnasium as gym

P = 0
D = 0
A = 10
V = 10

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

resets = 0
for _ in range(100):
    pos, vel, ang, angv  = observation
    ctl = pos * P + vel * D + ang * A + angv * V
    #print (ctl)
    action = 1 if ctl + 0.5 > random.random() else 0
    observation, reward, terminated, truncated, info = env.step(action)
    print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        observation, info = env.reset()
        resets += 1
    time.sleep(.005)
env.close()

print ("Golf score:", resets)
