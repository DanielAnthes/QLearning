from operator import itemgetter
from random import random, choice, sample
import torch
import numpy as np
import gym
import torch.optim as optim

from network import Net

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

class Agent:
    '''
    Defines a RL Agent that learns to solve the LunarLander task in OpenAI's Gym.
    This Agent implements Deep Q-Learning with a replay buffer.
    '''

    def __init__(self):
        self.lr = 0.001  # learning rate
        self.epsilon = 1  # probability with which the agent makes a random action choice instead of following its policy. Encourages exploration
        self.gamma = 0.95  # discount
        self.net = Net().to(DEVICE)
        self.optim = optim.RMSprop(self.net.parameters(), lr=self.lr)
        self.replaybuffer = list()
        self.maxbuf = 10000  # max size for replay buffer
        self.env = gym.make('LunarLander-v2')
        self.actions = [0,1,2,3]

    def rollout(self, draw=False):
        '''
        play one episode of the game
        '''
        states = list()
        actions = list()
        rewards = list()
        flags = list()
        state = self.env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
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
                    Qs = self.net(state)
                    idx = torch.argmax(Qs)
                action = self.actions[idx.to("cpu").detach().numpy()]


            state, rew, done, _ = self.env.step(action)
            state = torch.FloatTensor(state).to(DEVICE)
            action = torch.LongTensor([action]).to(DEVICE)
            actions.append(action)
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
            Qs = self.net(s)
            Q = torch.index_select(Qs, index=a, dim=0)

            if d:
                y = r
            else:
                with torch.no_grad():
                    Qs_next = self.net(n)
                    Qmax = torch.max(Qs_next)
                y = r + self.gamma * Qmax

            l += (y - Q)**2
        return l

    def train(self, n_ep, minbatchsize=100):
        '''
        train the agent for a number of rollouts
        The agent is trained with a decaying epsilon parameter. Epsilon is always initialized as 1 - all actions the agent takes are random.
        Epsilon decreases over rollouts until the agent's behaviour is fully determined by its's policy.
        This is done to encourage exploration at the start of training and implicitly encodes uncertainty about the policy.

        INPUT:
            n_ep            - number of games to play
            minibatchsize   - batch size
        '''
        rewards = list()
        eps = list()
        self.epsilon = 1
        for i in range(n_ep):
            if i % 50 == 0:
                print(f"Iteration: {i}")
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
            self.epsilon = np.exp(-(1/4000)*i)

        return rewards, eps

    def evaluate(self, n_ep):
        '''
        play a number of games without updating the network for evaluating performance.
        '''
        self.epsilon = 0
        rewards = list()
        for _ in range(n_ep):
            _, reward = self.rollout()
            rewards.append(reward)
        return rewards


