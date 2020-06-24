import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from random import random, choice, sample
from operator import itemgetter



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4,48)
        self.fc2 = nn.Linear(48,48)
        self.fc3 = nn.Linear(48,2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Agent:

    def __init__(self):
        self.lr = 0.001
        self.epsilon = 1
        self.gamma = 0.95
        self.net = Net()
        self.optim = optim.RMSprop(self.net.parameters(), lr=self.lr)
        self.replaybuffer = list()
        self.maxbuf = 10000
        self.env = gym.make('CartPole-v1')
        self.actions = [0,1]

    def rollout(self, draw=False):
        # TODO: reward to go should bootstrap from Q value and not use actual future reward for comparability with paper
        states = list()
        actions = list()
        rewards = list()
        flags = list()
        state = self.env.reset()
        done = False
        states.append(state)
        while not done:
            if draw:
                self.env.render()
            # generate step
            coin = random()
            if coin < self.epsilon:
                action = choice(self.actions)
            else:
                with torch.no_grad():
                    Qs = self.net(torch.FloatTensor(state))
                    idx = torch.argmax(Qs)
                action = self.actions[idx.detach().numpy()]

            actions.append(action)
            state, rew, done, _ = self.env.step(action)
            states.append(state)
            rewards.append(rew)
            flags.append(done)

        # compute y for every timestep
        reward = sum(rewards)
        memory = list()

        for st, act, rew, next, done in zip(states[:-1], actions, rewards, states[1:], flags):
            memory.append((st, act, rew, next, done))

        return memory, reward

    def loss(self, sample):
        l = 0
        for s,a,r,n,d in sample:
            Qs = self.net(torch.FloatTensor(s))
            a = torch.LongTensor([a])
            Q = torch.index_select(Qs, index=a, dim=0)

            if d:
                y = r
            else:
                with torch.no_grad():
                    Qs_next = self.net(torch.FloatTensor(n))
                    Qmax = torch.max(Qs_next)
                y = r + self.gamma * Qmax

            l += (y - Q)**2
        return l

    def train(self, n_ep, minbatchsize=100):
        rewards = list()
        eps = list()
        self.epsilon = 1
        for i in range(n_ep):
            eps.append(self.epsilon)
            memory, reward = self.rollout()
            self.optim.zero_grad()
            self.replaybuffer += memory
            rewards.append(reward)
            nsample = min(minbatchsize, len(self.replaybuffer))
            minibatch = sample(self.replaybuffer,nsample)
            loss = self.loss(minibatch)
            loss.backward()
            self.optim.step()

            # truncate buffer
            if len(self.replaybuffer) > self.maxbuf:
                overflow = len(self.replaybuffer) - self.maxbuf
                self.replaybuffer = self.replaybuffer[overflow:]

            # update epsilon
            self.epsilon = np.exp(-(1/2000)*i)

        return rewards, eps

    def evaluate(self, n_ep):
        self.epsilon = 0
        rewards = list()
        for _ in range(n_ep):
            _, reward = self.rollout()
            rewards.append(reward)
        return rewards

agent = Agent()
rewards, epsilons = agent.train(1000)
eval_rewards = agent.evaluate(100)

plt.figure()
plt.subplot('311')
plt.plot(range(len(rewards)), rewards)
plt.subplot('312')
plt.plot(range(len(epsilons)), epsilons)
plt.subplot('313')
plt.plot(range(len(eval_rewards)), eval_rewards)
plt.show()

for _ in range(10):
    agent.rollout(draw=True)
agent.env.close()
