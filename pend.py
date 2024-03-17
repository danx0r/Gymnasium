import sys, time, argparse
import random
import gymnasium as gym

VERBOSE = 0

def controller(observation, P, D):
    co, si, av  = observation
    action = si * P + av * D
    if VERBOSE & 2:
        print ("CONTROLLER obs:", co, si, av, "ACT:", action)
    return [action]


STEPS = 100

def go(P, D, env):
    error = 0
    observation, info = env.reset()
    if VERBOSE & 4:
        print (f"P={P} D={D}")
    for ii in range(STEPS):
        action = controller(observation, P, D)
        # print (action)
        observation, reward, terminated, truncated, info = env.step(action)
        error += abs(observation[1])
        # print (ii, observation, error)
        if terminated or truncated:
            print ("*************************RESET*************************")
            print (ii, observation)
            observation, info = env.reset()
        # time.sleep(.02)
    error = error / STEPS
    # print ("ERROR:", error)
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--P", type=float)
    parser.add_argument("--D", type=float)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    VERBOSE = args.verbose

    if args.show:
        env = gym.make("Pendulum-v1", render_mode="human")
    else:
        env = gym.make("Pendulum-v1")
    env._max_episode_steps=200

    if not args.train:
        P = args.P
        D = args.D
        err = go(P, D, env)
        print (f"error={err}")
    else:
        best = 999999
        for i in range(args.epochs):
            if VERBOSE & 1:
                print (f"epoch={i}", end=" ")
            P = random.random() * 500 - 250
            D = random.random() * 20 - 10

            error = go(P, D, env)
            if VERBOSE & 1:
                print (f"P={P} D={D} error={error}")
            if error < best:
                best = error
                best_P = P
                best_D = D
                print (f"new best P={best_P} D={best_D} error={error}")

        for i in range(args.epochs):
            if VERBOSE & 1:
                print (f"epoch={i}", end=" ")
            P = random.random() * 5.0 - 2.5 + best_P
            D = random.random() * .2 - .1 + best_D

            error = go(P, D, env)
            if VERBOSE & 1:
                print (f"P={P} D={D} error={error}")
            if error < best:
                best = error
                best_P = P
                best_D = D
                print (f"new best P={best_P} D={best_D} error={error}")

        print (f"best P={best_P} D={best_D} error={best}")
    env.close()
