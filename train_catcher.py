import gym
import sys
sys.path.append('C:/Users/xuwei1/Documents/baselines')
from baselines import deepq
import time

def main():
    env = gym.make('Catcher-v0')
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.5,
        exploration_final_eps=0,
        print_freq=10,
        batch_size=32,
        
    )
    print("Saving model to catcher_model.pkl")
    act.save("model/catcher.pkl")

def test():
    env = gym.make('Catcher-v0')
    state = env.reset()
    print(state.shape)
    for _ in range(10):
    	env.render()
    	time.sleep(0.5)
    	state, reward, done, _= env.step(env.action_space.sample()) # take a random action
    	print((state.shape,reward,done))
    	if done:
    		break

if __name__ == '__main__':
    main()