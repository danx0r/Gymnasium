import time
import gymnasium as gym

P = -1
D = 0

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    pos = observation[0]
    vel = observation[1]
    ctl = pos * P + vel * D
    action = 1 if ctl > 0 else 0
    observation, reward, terminated, truncated, info = env.step(action)
    print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        observation, info = env.reset()
    time.sleep(.05)
env.close()
