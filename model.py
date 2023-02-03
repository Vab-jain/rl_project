import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
 Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        # Define the replay memory
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.h_layer1 = nn.Linear(n_observations, 128)
        self.h_layer2 = nn.Linear(128, 128)
        self.h_layer3 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128,n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.h_layer1(x))
        x = F.relu(self.h_layer2(x))
        x = F.relu(self.h_layer3(x))
        return self.out_layer(x)
