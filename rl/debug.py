import os
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from tqdm import tqdm
from kaggle_environments import evaluate, make, utils
import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

def mysave(list,name):
    q = np.array(list)
    np.save(name + '.npy',q)

def myload(name):
    p = np.load(name + '.npy',allow_pickle=True)
    return p.tolist()

env = ConnectX()

alpha = 0.1				    # 学习率
gamma = 0.6				    # discount factor γ
epsilon = 0.9				# ε-greedy策略的ε
min_epsilon = 0.1			# 最小ε

episodes = 10000			# 采样轮数

alpha_decay_step = 1000
alpha_decay_rate = 0.9		# α衰减率
epsilon_decay_rate = 0.99	# ε衰减率

q_table = QTable(env.action_space)

#加载Q表，接着上一次的训练
q_table.table = myload('qtable')
#加载相关训练结果
all_epochs = myload('all_epochs')
all_total_rewards = myload('all_total_rewards')
all_avg_rewards = myload('all_avg_rewards')
all_qtable_rows = myload('all_qtable_rows')
all_epsilons = myload('all_epsilons')
'''
all_epochs = []
all_total_rewards = []
all_avg_rewards = []  # Last 100 steps
all_qtable_rows = []
all_epsilons = []
'''


for i in tqdm(range(episodes)):
    state = env.reset()     # 清空棋盘
    epochs, total_rewards = 0, 0
    epsilon = max(min_epsilon, epsilon*epsilon_decay_rate)   # ε每轮衰减
    done = False

    while not done:    # 开始一轮采样
        space_list = [c for c in range(
            env.action_space.n) if state['board'][c] == 0]

        if random.uniform(0, 1) <= epsilon:  # ε-greedy-->选择随机策略
            action = choice(space_list)
        else:  # ε-greedy-->选择贪心策略
            row = np.array(q_table(state)[:])
            row[[c for c in range(env.action_space.n)
                 if state['board'][c] != 0]] = -1
            action = int(np.argmax(row))

        next_state, reward, done, info = env.step(action)

        if done:
            if reward == 1:  # Won
                reward = 20
            elif reward == 0:  # Lost
                reward = -20
            else:
                reward = 1  # draw
        else:
            reward = -0.05

        old_value = q_table(state)[action]
        next_max = np.max(q_table(next_state))

        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table(state)[action] = new_value

        state = next_state
        epochs += 1
        total_rewards += reward

    all_epochs.append(epochs)
    all_total_rewards.append(total_rewards)
    avg_rewards = np.mean(all_total_rewards[max(0, i-100):(i+1)])
    all_avg_rewards.append(avg_rewards)
    all_qtable_rows.append(len(q_table.table))
    all_epsilons.append(epsilon)

    if (i+1) % alpha_decay_step == 0:
        alpha *= alpha_decay_rate

#保存Q表
mysave(q_table.table,'qtable')
#保存相关训练结果
mysave(all_epochs,'all_epochs')
mysave(all_total_rewards,'all_total_rewards')
mysave(all_avg_rewards,'all_avg_rewards')
mysave(all_qtable_rows,'all_qtable_rows')
mysave(all_epsilons,'all_epsilons')

my_agent = '''def my_agent(observation, configuration):
    from random import choice

    q_table = ''' \
    + str(dict_q_table).replace(' ', '') \
    + '''

    board = observation.board[:]
    board.append(observation.mark)
    state_key = list(map(str, board))
    state_key = hex(int(''.join(state_key), 3))[2:]

    if state_key not in q_table.keys():
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    action = q_table[state_key]

    if observation.board[action] != 0:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    return action
    '''

with open('submission.py', 'w') as f:
    f.write(my_agent)
