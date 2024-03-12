import sys, time
import random
import gymnasium as gym

def controller(observation):
    GAIN = float(sys.argv[1])
    co, si, av  = observation
    action = si * -100 + av * -1
    action *= GAIN
    print ("CONTROLLER obs:", co, si, av, "ACT:", action)
    return [action]


env = gym.make("Pendulum-v1", render_mode="human")
env._max_episode_steps=500
while True:
    observation, info = env.reset()
    co, si, av = observation
    if abs(si) < .2 and co > .9:
        break
time.sleep(1)

resets = 0
rewards = 0
for ii in range(300):
    print (ii, end=" ")
    action = controller(observation)
    # print (action)
    if ii == 200:
        action = [100]
    observation, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    #print (ii, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        print (ii, observation, "rewards:", rewards)
        observation, info = env.reset()
        resets += 1
        rewards = 0
    # time.sleep(.02)
env.close()

print ("Golf score:", resets)
