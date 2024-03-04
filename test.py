import time
import random
import gymnasium as gym

P = 9
D = 0
A = 10
V = 5

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(500):
    pos, vel, ang, angv  = observation
    ctl = pos * P + vel * D + ang * A + angv * V
    action = 1 if ctl + 0.5 > random.random() else 0
    observation, reward, terminated, truncated, info = env.step(action)
    print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        observation, info = env.reset()
    time.sleep(.015)
env.close()
