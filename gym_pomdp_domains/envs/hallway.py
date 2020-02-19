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
from PIL import Image, ImageDraw
import pygame as pg
import time

class Hallway(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, prob_action_success = 0.8, prob_see_wall_true = 0.9,
               prob_see_wall_false = 0.05):
    '''
        prob_action_success: probability of success for all actions but "stay" 
        prob_see_wall_true: probability to see a wall if a wall is there
        prob_see_wall_false: probability to see a wall if a wall is not there
    '''
    self.reward_range = (0, 1)
    self.action_space = spaces.Discrete(5)
    self.observation_space = spaces.Discrete(20)
    self.state_space = spaces.Discrete(57)
    
    self.MAX_STEPS = 100 # max. number of steps per episodes
    self.num_steps = 0 # steps taken so far
    
    self.PROB_ACTION_SUCCESS = prob_action_success
    self.PROB_SEE_WALL_TRUE = prob_see_wall_true
    self.PROB_SEE_WALL_FALSE = prob_see_wall_false
    
    # optimal actions per state (5 signals the goal state where nothing is optimal)
    self.OPTIMAL_ACTIONS = [2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4,
                            1, 3, 4, 2,
                            2, 1, 3, 4, 2, 1, 3, 4,
                            1, 3, 4, 2, 
                            2, 1, 3, 4, 2, 1, 3, 4,
                            1, 3, 4, 2,
                            2, 1, 3, 4,
                            4, 2, 1, 3,
                            5,
                            3, 4, 2, 1, 3, 4, 2, 1]
    
    # types of states
    self.TERMINAL_STATES = [48]
    self.NON_TERMINAL_STATES = np.arange(0, self.state_space.n)
    self.NON_TERMINAL_STATES = [i for i in self.NON_TERMINAL_STATES if i not in self.TERMINAL_STATES]
    
    # start state
    self.state = choice(self.NON_TERMINAL_STATES)
    
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
  
    # true next states for moving forward (i.e. if action succeeds)
    self.FORWARD_STATE = [0, 5, 2, 3, 
                          4, 9, 6, 3, 
                          8, 17, 14, 7, 
                          8, 13, 14, 15,
                          16, 21, 18, 11, 
                          20, 29, 26, 19, 
                          20, 25, 26, 27, 
                          28, 33, 30, 23,
                          32, 41, 38, 31, 
                          32, 37, 38, 39, 
                          40, 45, 42, 35, 
                          44, 50, 48, 43, 
                          48, 
                          49, 54, 51, 47,
                          53, 54, 55, 52]
    
    # state transition matrix
    self.P = np.zeros((self.action_space.n,
                          self.state_space.n,
                          self.state_space.n))
    
    # observation matrix
    self.O = np.zeros((self.action_space.n,
                          self.state_space.n,
                          self.observation_space.n))
    
    # any action taken in terminal state has no effect
    for i in self.TERMINAL_STATES:
        self.P[:, i, i] = 1
        self.O[:, i] = self.get_observation_probabilities(i)
    
    # transition and observation probabilities
    for action in range(self.action_space.n):
        for s in self.NON_TERMINAL_STATES:
            s_prime = self.act(s, action)
            
            if action != 0:
                # action succeeds
                self.P[action, s, int(s_prime)] = self.PROB_ACTION_SUCCESS
                self.O[action, s] = self.PROB_ACTION_SUCCESS * self.get_observation_probabilities(s_prime)
                
                # action does not succeed
                for action_other in [i for i in range(self.action_space.n) if not i == action]:
                    s_prime_other = self.act(s, action_other)
                    self.P[action_other, s, int(s_prime_other)] += (1 - self.PROB_ACTION_SUCCESS)/4
                    self.O[action, s] += self.get_observation_probabilities(s_prime_other) * (1 - self.PROB_ACTION_SUCCESS)/4
            else:
                self.P[action, s, int(s_prime)] = 1
                self.O[action, s] = self.get_observation_probabilities(s_prime)
    
    # rewards are given for arriving in a certain state
    self.R = np.full((self.state_space.n), 0)
    
    for i in self.TERMINAL_STATES:
        self.R[i] = 1
        
  def get_observation(self, s_prime):
    '''
    Returns observation for arriving in state s_prime.
    '''
        
    s_prime = int(s_prime)
    
    # special observations for facing south in 3 grid positions in second row
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
        elif self.WALLS[s_prime][wall_pos] == 1 and prob <= self.PROB_SEE_WALL_TRUE:
            obs[wall_pos] = 1
            
    return self.encode_observation(obs)

  def get_observation_probabilities(self, s):
    '''
    Computes observation probabilities for arriving in state s.
    '''
    probs = np.zeros(self.observation_space.n)
    true_obs = self.WALLS[s]
    
    wall_probs = [[1 - self.PROB_SEE_WALL_FALSE, self.PROB_SEE_WALL_FALSE],
                  [1 - self.PROB_SEE_WALL_TRUE, self.PROB_SEE_WALL_TRUE]]
    
    if s == 14:
        probs[16] = 1
    elif s == 26:
        probs[17] = 1
    elif s == 38:
        probs[18] = 1
    elif s == 48:  # arrived in goal state
        probs[19] = 1
    else:
        for obs in [[i, j, k, l] for i in range(2) for j in range(2) for k in range(2) for l in range(2)]:
            p = 1
            for w in range(4):
                p *= wall_probs[int(true_obs[w])][int(obs[w])]
            probs[self.encode_observation(obs)] = p
     
    return probs
    
  def encode_observation(self, obs):
    '''
    Encodes observation of walls to an integer.
    '''
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
        if prob <= self.PROB_ACTION_SUCCESS:
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

  def render(self, mode='human', close = True):
    '''
        A blue circle marks the current agent position.
        A white circle marks the current agent orientation.
        The goal state is colored in bright green.
    '''
    
    if mode == "human":
        square_size = 30
        height = 11 * square_size + 10
        width = 2 * square_size + 10
        
        offsets = [[0.4, 0.1], [0.7, 0.4], [0.4, 0.7], [0.1, 0.4]]
        dot_width = 0.2
        
        mapping = [0, 1, 2, 13, 3, 4, 15, 5, 6, 17, 7, 8, 19, 9, 10]
        
        color_fill = "#FFFFBF"
        color_goal = "#33CC00"
        color_agent = "blue"
        color_orientation = "white"
        
        if self.state <= 48:
            index = int(np.floor(self.state/4))
            rem = int(self.state % 4)
        else:
            index = int(np.floor((self.state + 3)/4))
            rem = int((self.state + 3) % 4)
            
        mapped_row = np.floor(mapping[index] / 11)
        
        image = Image.new("RGB", size=(height, width), color="#FFFFFF")
    
        draw = ImageDraw.Draw(image)
       
        # first row of squares
        for i in range(11):
            shape = [(i * square_size, 0), ((i+1) * square_size, square_size)] 
            draw.rectangle(shape, fill = color_fill, outline ="black") 
        
        # second row
        for i in range(2, 9, 2):
            # goal state
            if i == 8:
                color = color_goal
            else:
                color = color_fill
            shape = [(i * square_size, square_size), ((i+1) * square_size, 2 * square_size)] 
            draw.rectangle(shape, fill = color, outline ="black") 
        
        # draw location of agent
        draw.ellipse(((mapping[index] - mapped_row * 11) * square_size, 
                      mapped_row * square_size, 
                      (mapping[index] + 1 - mapped_row * 11) * square_size, 
                      square_size * (mapped_row + 1)), 
                     fill = color_agent, outline = color_agent)
        
        if not self.state == 48:
            # draw orientation of agent
            draw.ellipse(((mapping[index] - mapped_row * 11 + offsets[rem][0]) * square_size, 
                          (offsets[rem][1] + mapped_row) * square_size, 
                          (mapping[index] + offsets[rem][0] + dot_width - mapped_row * 11) * square_size, 
                          (offsets[rem][1] + dot_width + mapped_row) * square_size), 
                         fill = color_orientation, outline = 'black')
            
        del draw
        
        pg.init()
        screen = pg.display.set_mode((height, width))
        screen_rect = screen.get_rect()
        image = pg.image.fromstring(image.tobytes(), image.size, image.mode).convert()
        screen.blit(image, image.get_rect(center=screen_rect.center))
        pg.display.update()
        
        # close window
        if close:
            time.sleep(2)
            pg.quit()
    
    else:
        print("Current state:", self.state)