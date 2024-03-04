import time
import random
import gymnasium as gym

#
# hip_r, knee_r, foot_r, hip_l, knee_l, foot_l
# (mujoco calls them thigh, leg, foot)
# right is tan, left is purple
# hip: -1 extend (back bend) 1 contract (knee to nose)
# knee: -1 contract (heel to butt; kneeling), 1 extend (locked knee)
# foot: -1 extend (high heel/en pointe), 1 contract (walk on your heels)
#
def controller(obs, step):
    print (f"torso height: {obs[0]}")
    if step < 350:
        act = [0, 1, 0, 0, 1, 0]
    elif step < 700:
        act = [-1, 1, 0, -1, 1, 0]
    else:
        act = [1, -1, 0, 1, -1, 0]
    return act


env = gym.make("Walker2d-v4", render_mode="human", terminate_when_unhealthy=False)
# env._max_episode_steps=1000
observation, info = env.reset()

resets = 0
rewards = 0
for _ in range(1000):
    action = controller(observation, _)
    observation, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    print (_, terminated, truncated)
    if terminated or truncated:
        print ("*************************RESET*************************")
        print (_, observation, "rewards:", rewards)
        observation, info = env.reset()
        resets += 1
        rewards = 0
    # time.sleep(.05)
env.close()

print ("Golf score:", resets)
