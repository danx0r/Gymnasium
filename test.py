import time
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    time.sleep(.1)
    action = 0 if observation[0] > 0 else 1  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print (_, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        print ("*************************RESET*************************")
        observation, info = env.reset()
env.close()
