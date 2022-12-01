from mimetypes import init
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from kaggle_environments import evaluate, make

import random
from random import choice
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5):
        self.env = make('connectx', debug=True)
        self.pair = [None, 'negamax']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(
            config.columns * config.rows)

    def step(self, action):
        return self.trainer.step(action)

    def set_state(self,init_state):
        self.env.reset()
        self.env.state[0]['observation'] = deepcopy(init_state)

    def reset(self):  
        return self.trainer.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

class MonteCarlo:
    def __init__(self,env,gamma,episodes):
        '''
        env是当前需要进行模拟采样的状态
        gamma用于计算Reward
        episodes是每个(s,a)的采样次数
        '''
        self.state = deepcopy(env)
        self.all_rewards = [0,0,0,0,0,0,0]
        self.gamma = gamma
        self.episodes = episodes

    def rollout(self,env):
        # 对当前状态进行rollout采样
        mean_reward = 0
        ini_state = env.env.state[0]['observation']
        for i in range(self.episodes):
            r = 0
            gamma = 1
            env.set_state(ini_state)
            state = deepcopy(ini_state)
            
            #env.env = deepcopy(ini_state)
            done = False
            
            while not done:
                space_list = [c for c in range(7) if state['board'][c] == 0]
                action = choice(space_list)

                next_state, reward, done, info = env.step(action)

                if done:
                    if reward == 1: 
                        r += 20*gamma
                    elif reward == 0: 
                        r -= 20*gamma
                    else: 
                        r += 1*gamma

                    mean_reward += r
                else:
                    r -= 0.05*gamma

                gamma *= self.gamma
                state = next_state

        return mean_reward/self.episodes

def play(num,env,gamma,episodes):
    result = {'win':0,'loss':0,'draw':0}
    for i in range(num):
        print('[GAME{}]:'.format(i))
        done = False
        state = env.reset()
        while not done:
            ini_state = env.env.state[0]['observation']
            space_list = [c for c in range(7) if ini_state['board'][c] == 0]
            R = []
            for action in space_list:  # rollout模拟采样

                env.set_state(ini_state)

                next_state, reward, done, info = env.step(action)
                mc = MonteCarlo(env, gamma, episodes)
                reward = reward + gamma*mc.rollout(env)
                R.append(reward)

            Action = int(np.argmax(R))  # 根据采样结果选择动作
            print('R = ', R)
            print('Action = ', Action)

            env.set_state(ini_state)
            next_state, reward, done, info = env.step(Action)
            print(env.render(mode="ansi"))

            if done:
                if reward == 1:  # Won
                    result['win'] += 1
                    print('you win!')
                elif reward == 0:  # Lost
                    result['loss'] += 1
                    print('you loss!')
                else:
                    result['draw'] += 1
                    print('draw!')
    print('My Agent vs Negamax Agent:', '{0} episodes- won: {1} | loss: {2} | draw: {3} | winning rate: {4}%'.format(
        num,
        result['win'],
        result['loss'],
        result['draw'],
        (result['win'] / num)*100
    ))
                
    


env = ConnectX()
gamma = 1				    # discount factor γ
episodes = 10                # 模拟采样轮数

my_agent = '''
def my_agent(observation, configuration):
    import numpy as np
            '''

play(10,env,gamma,episodes)
