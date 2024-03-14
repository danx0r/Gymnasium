import sys, time
import random
import gymnasium as gym

def controller(observation):
    GAIN = float(sys.argv[1])
    co, si, av  = observation
    action = si * -100 + av * -1
    action *= GAIN
    # print ("CONTROLLER obs:", co, si, av, "ACT:", action)
    return [action]


# env = gym.make("Pendulum-v1", render_mode="human")
env = gym.make("Pendulum-v1")
env._max_episode_steps=200
observation, info = env.reset()

STEPS = 100
error = 0
for ii in range(STEPS):
    action = controller(observation)
    # print (action)
    observation, reward, terminated, truncated, info = env.step(action)
    error += abs(observation[1])
    # print (ii, observation, error)
    if terminated or truncated:
        print ("*************************RESET*************************")
        print (ii, observation)
        observation, info = env.reset()
    # time.sleep(.02)
error = error ** .5 / STEPS
print ("ERROR:", error)
env.close()
