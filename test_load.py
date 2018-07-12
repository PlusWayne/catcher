import gym
import sys
sys.path.append('C:/Users/xuwei1/Documents/baselines')

from baselines import deepq
import time

def main():
    env = gym.make("Catcher-v0")
    act = deepq.load("model/catcher.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.5)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
