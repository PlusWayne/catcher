import gym
import sys
sys.path.append('C:/Users/xuwei1/Documents/baselines')

from baselines import deepq
import time

def main():
    env = gym.make("Catcher-v0")
    act = deepq.load("model/catcher.pkl")
    episode = 0
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            # time.sleep(0.5)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        if episode_rew == -1:
        	print('Game over at Episode ', episode)
        	break
        else:
        	episode += 1
        	print("Episode : {} reward : {}".format(episode,episode_rew))

if __name__ == '__main__':
    main()
