'''
    February 2020
    Implementation of the hallway domain.
    
    Actions: stay, forward, turn right, turn left, turn around
    Orientations: up, right, down, left
    State layout: 0-3, 4-7, 8-11, 16-19, 20-23, 28-31, 32-35, 40-43, 44-47, 49-52, 53-56
                            12-15        24-27         36-39          48
'''

import gym
from gym import spaces
import numpy as np
from random import choice, uniform

class Hallway(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.reward_range = (0, 1)
    self.action_space = spaces.Discrete(5)
    self.observation_space = spaces.Discrete(20)
    self.state_space = spaces.Discrete(57)
    
    self.MAX_STEPS = 100 # max. number of steps per episodes
    self.num_steps = 0 # steps taken so far
    
    self.PROB_ACTION_SUCCESS = 0.8 # probability of success for all actions but "stay" 
    self.PROB_SEE_WALL_TRUE = 0.9 # probability to see a wall if a wall is there
    self.PROB_SEE_WALL_FALSE = 0.05 # probability to see a wall if a wall is not there
    
    # types of states
    self.TERMINAL_STATES = [48]
    self.NON_TERMINAL_STATES = np.arange(0, self.state_space.n)
    self.NON_TERMINAL_STATES = [i for i in self.NON_TERMINAL_STATES if i not in self.TERMINAL_STATES]
    
    # Walls: in front, to right, behind, to left
    self.WALLS = np.zeros((self.state_space.n, 4))
    for s in [4, 16, 28, 40, 49, 6, 18, 30, 42, 51]:
        self.WALLS[s] = [1, 0, 1, 0]
    for s in [5, 17, 29, 41, 50, 7, 19, 31, 43, 52]:
        self.WALLS[s] = [0, 1, 0, 1]
    for s in [12, 24, 36, 1, 56]:
        self.WALLS[s] = [0, 1, 1, 1]
    for s in [13, 25, 37, 2, 53]:
        self.WALLS[s] = [1, 1, 1, 0]
    for s in [14, 26, 38, 3, 54]:
        self.WALLS[s] = [1, 1, 0, 1]
    for s in [15, 27, 39, 0, 55]:
        self.WALLS[s] = [1, 0, 1, 1]
    for s in [8, 20, 32, 44]:
        self.WALLS[s] = [1, 0, 0, 0]
    for s in [9, 21, 33, 45]:
        self.WALLS[s] = [0, 0, 0, 1]
    for s in [10, 22, 34, 46]:
        self.WALLS[s] = [0, 0, 1, 0]
    for s in [11, 23, 35, 47]:
        self.WALLS[s] = [0, 1, 0, 0]
  
    # true next states for moving forward
    self.FORWARD_STATE = [0, 5, 2, 3, 
                          4, 9, 6, 3, 
                          8, 17, 10, 7, 
                          8, 13, 14, 15,
                          16, 21, 18, 11, 
                          20, 29, 22, 19, 
                          20, 25, 26, 27, 
                          28, 33, 30, 23,
                          32, 41, 34, 31, 
                          32, 37, 38, 39, 
                          40, 45, 42, 35, 
                          44, 50, 46, 43, 
                          48, 
                          49, 54, 51, 47,
                          53, 54, 55, 52]
    
    # state transition matrix
    self.P = np.zeros((self.action_space.n,
                          self.state_space.n,
                          self.state_space.n))
    
    # start state
    self.state = choice(self.NON_TERMINAL_STATES)
    
    # any action taken in terminal state has no effect
    for i in self.TERMINAL_STATES:
        self.P[:, i, i] = 1
    
    # transition probabilities
    for action in range(self.action_space.n):
        for s in self.NON_TERMINAL_STATES:
            s_prime = self.act(s, action)
            if action != 0:
                self.P[action, s, int(s_prime)] = self.PROB_ACTION_SUCCESS
                for action_other in [i for i in range(self.action_space.n) if not i == action]:
                    s_prime = self.act(s, action_other)
                    self.P[action_other, s, int(s_prime)] = (1 - self.PROB_ACTION_SUCCESS)/4
            else:
                self.P[action, s, int(s_prime)] = 1
    
    # rewards are given for arriving in a certain state
    self.R = np.full((self.state_space.n), 0)
    
    for i in self.TERMINAL_STATES:
        self.R[i] = 1
        
  def get_observation(self, s_prime):
    '''
    Returns observation for arriving in state s_prime.
    '''
    
    s_prime = int(s_prime)
    
    # special observations for facing south in 3 grid positions
    if s_prime == 14:
        return 16
    if s_prime == 26:
        return 17
    if s_prime == 38:
        return 18
    
    # arrived in goal state
    if s_prime == 48: 
        return 19
    
    obs = np.zeros(4)
    for wall_pos in range(4):
        prob = uniform(0, 1)
        if self.WALLS[s_prime][wall_pos] == 0 and prob < self.PROB_SEE_WALL_FALSE:
            obs[wall_pos] = 1
        elif self.WALLS[s_prime][wall_pos] == 1 and prob < self.PROB_SEE_WALL_TRUE:
            obs[wall_pos] = 1
    return int(obs[0] * 1 + obs[1] * 2 + obs[2] * 4 + obs[3] * 8)
        
  def act(self, s, action):
    '''
    Deterministic next state for taking an action in state s.
    '''
    
    # actions have no effect in terminal state
    if s in self.TERMINAL_STATES:
        return s
    
    if action == 0:
        s_prime = s
    elif action == 1: # move forward
        s_prime = self.FORWARD_STATE[s]
    elif action == 2: # turn right
        if s > 48:
            s -= 1
        s_prime = s + 1
        if np.floor(s / 4) != np.floor(s_prime/4):
            s_prime = np.floor(s/4) * 4
        if s_prime >= 48:
            s_prime += 1
    elif action == 3: # turn left
        if s > 48:
            s -= 1
        s_prime = s - 1
        if np.floor(s / 4) != np.floor(s_prime/4):
            s_prime = np.floor(s/4) * 4 + 3
        if s_prime >= 48:
            s_prime += 1
    elif action == 4: # turn around
        if s > 48:
            s -= 1
        if np.floor(s/4) <= 1: # orientations are up or right
            s_prime = s + 2
        else:
            s_prime = s - 2
        if s_prime >= 48:
            s_prime += 1

    return int(s_prime)
    
  def step(self, action):
    
    # action is not "stay"
    if action > 0:    
        prob = uniform(0, 1)
        if prob < self.PROB_ACTION_SUCCESS:
            self.state = self.act(self.state, action)
        else:
            self.state = self.act(self.state, choice([i for i in range(self.action_space.n) if not i == action]))      
    self.num_steps += 1
    done = (self.state in self.TERMINAL_STATES) or (self.num_steps == self.MAX_STEPS - 1)
    return self.get_observation(self.state), self.R[self.state], done , ""
      
  def reset(self):
    self.state = choice(self.NON_TERMINAL_STATES)
    self.num_steps = 0
    return self.get_observation(self.state)

  def render(self, mode='human', close=False):
    '''
        An arrow marks the current agent position.
    '''
    # TODO
    