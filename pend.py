import time
import random
import gymnasium as gym

def controller(observation):
    co, si, av  = observation
    print ("CONTROLLER obs:", co, si, av)
    action = 1.0
    return [action]


env = gym.make("Pendulum-v1", render_mode="human")
env._max_episode_steps=200
while True:
    observation, info = env.reset()
    co, si, av = observation
    print (co, si)
    if abs(si) < .1 and co > .9:
        break
time.sleep(1)

resets = 0
rewards = 0
for _ in range(200):
    print (_)
    action = controller(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    #print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        print (_, observation, "rewards:", rewards)
        observation, info = env.reset()
        resets += 1
        rewards = 0
    #time.sleep(.005)
env.close()

print ("Golf score:", resets)
