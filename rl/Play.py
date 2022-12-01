#Play with your agent
from submission import my_agent
import os
import sys
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make, utils
#from debug import ConnectX,QTable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5):
        self.env = make('connectx', debug=True)
        self.pair = [None, my_agent]
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(
            config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(action)

    def reset(self):  # 有switch_prob的几率更换先手顺序
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)


class QTable:
    def __init__(self, action_space):
        self.table = dict()
        self.action_space = action_space

    def add_item(self, state_key):
        self.table[state_key] = list(np.zeros(self.action_space.n))

    def __call__(self, state):
        board = state['board'][:]  # Get a copy
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        state_key = hex(int(''.join(state_key), 3))[2:]
        if state_key not in self.table.keys():
            self.add_item(state_key)

        return self.table[state_key]

env = ConnectX()

#print(env.render(mode="ansi"))

state = env.reset()
done = False

while not done:
    sys.stdout.flush()
    print(env.render(mode="ansi"))
    action = int(input('Input your action:(0-6)'))

    next_state, reward, done, info = env.step(action)

    if done:
        if reward == 1: # Won
            print('you win!')
        elif reward == 0: # Lost
            print('you loss!')
        else: 
            print('draw!')

