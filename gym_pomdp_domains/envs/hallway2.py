'''
    February 2020
    Implementation of the hallway2 domain.
    
    Actions: stay, forward, turn right, turn left, turn around
    Orientations: up, right, down, left
'''
import gym
from gym import spaces
import numpy as np
from random import choice, uniform
from PIL import Image, ImageDraw

class Hallway2(gym.Env):
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
    self.observation_space = spaces.Discrete(17)
    self.state_space = spaces.Discrete(89)
    
    self.MAX_STEPS = 100 # max. number of steps per episodes
    self.num_steps = 0 # steps taken so far
    
    self.PROB_ACTION_SUCCESS = prob_action_success
    self.PROB_SEE_WALL_TRUE = prob_see_wall_true
    self.PROB_SEE_WALL_FALSE = prob_see_wall_false
    
    # types of states
    self.TERMINAL_STATES = [68]
    self.NON_TERMINAL_STATES = np.arange(0, self.state_space.n)
    self.NON_TERMINAL_STATES = [i for i in self.NON_TERMINAL_STATES if i not in self.TERMINAL_STATES]
    
    # start state
    self.state = choice(self.NON_TERMINAL_STATES)
    
    # optimal actions per state (5 signals the goal state where nothing is optimal)
    self.OPTIMAL_ACTIONS = [2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 4, 2, 1, 3,
                            2, 1, 3, 4, 1, 23, 1, 23, 1, 23, 1, 23, 4, 2, 1, 3, 3, 4, 2, 1,
                            4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3,
                            2, 1, 3, 4, 4, 2, 1, 3, 4, 2, 1, 3, 2, 1, 3, 4, 5,
                            2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 2, 1, 3, 4, 1, 3, 4, 2]
    
    # Walls: in front, to right, behind, to left
    self.WALLS = np.zeros((self.state_space.n, 4))
    for s in [8, 25, 35, 57, 67, 79]:
        self.WALLS[s] = [1, 0, 0, 0]
    for s in [9, 26, 32, 58, 64, 80]:
        self.WALLS[s] = [0, 0, 0, 1]
    for s in [10, 27, 33, 59, 65, 77]:
        self.WALLS[s] = [0, 0, 1, 0]
    for s in [11, 24, 34, 56, 66, 78]:
        self.WALLS[s] = [0, 1, 0, 0]
    for s in [0, 17, 72, 87]:
        self.WALLS[s] = [1, 0, 0, 1]
    for s in [1, 18, 69, 88]:
        self.WALLS[s] = [0, 0, 1, 1]
    for s in [2, 19, 70, 85]:
        self.WALLS[s] = [0, 1, 1, 0]
    for s in [3, 16, 71, 86]:
        self.WALLS[s] = [1, 1, 0, 0]
    for s in [4, 6, 12, 14, 41, 43, 45, 47, 49, 51, 61, 63, 73, 75, 81, 83]:
        self.WALLS[s] = [1, 0, 1, 0]
    for s in [5, 7, 13, 15, 40, 42, 44, 46, 48, 50, 60, 62, 74, 76, 82, 84]:
        self.WALLS[s] = [0, 1, 0, 1]
    for s in [20, 38, 52]:
        self.WALLS[s] = [1, 0, 1, 1]
    for s in [21, 39, 53]:
        self.WALLS[s] = [0, 1, 1, 1]
    for s in [22, 36, 54]:
        self.WALLS[s] = [1, 1, 1, 0]
    for s in [23, 37, 55]:
        self.WALLS[s] = [1, 1, 0, 1]
  
    # true next states for moving forward
    self.FORWARD_STATE = [0, 5, 26, 3,
                          4, 9, 6, 3,
                          8, 13, 30, 7,
                          12, 17, 14, 11,
                          16, 17, 34, 15,
                          20, 25, 22, 23,
                          0, 25, 42, 23,
                          8, 29, 46, 31,
                          16, 37, 50, 35,
                          36, 37, 38, 35,
                          24, 41, 58, 43,
                          28, 45, 62, 47, 
                          32, 49, 66, 51,
                          52, 57, 54, 55,
                          40, 57, 71, 55,
                          44, 61, 79, 63, 
                          48, 68, 87, 67,
                          68,
                          56, 74, 71, 72,
                          73, 78, 75, 72,
                          60, 82, 79, 76,
                          81, 86, 83, 80,
                          64, 86, 87, 84]
    
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
    Returns observation for being in state s_prime.
    '''
    s_prime = int(s_prime)
    
    # goal state
    if s_prime == 68: 
        return 16
    
    obs = np.zeros(4)
    for wall_pos in range(4):
        prob = uniform(0, 1)
        if self.WALLS[s_prime][wall_pos] == 0 and prob < self.PROB_SEE_WALL_FALSE:
            obs[wall_pos] = 1
        elif self.WALLS[s_prime][wall_pos] == 1 and prob < self.PROB_SEE_WALL_TRUE:
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
    
    if s == 68: # goal state
        probs[16] = 1
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
        if s > 68:
            s -= 1
        s_prime = s + 1
        if np.floor(s / 4) != np.floor(s_prime/4):
            s_prime = np.floor(s/4) * 4
        if s_prime >= 68:
            s_prime += 1
    elif action == 3: # turn left
        if s > 68:
            s -= 1
        s_prime = s - 1
        if np.floor(s / 4) != np.floor(s_prime/4):
            s_prime = np.floor(s/4) * 4 + 3
        if s_prime >= 68:
            s_prime += 1
    elif action == 4: # turn around
        if s > 68:
            s -= 1
        if s % 4 <= 1: # orientations are up or right
            s_prime = s + 2
        else:
            s_prime = s - 2
        if s_prime >= 68:
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

  def render(self, mode='human'):
    '''
        A blue circle marks the current agent position.
        A white circle marks the current agent orientation.
        The goal state is colored in bright green.
    '''
    if mode == "human":
        square_size = 30
        height = 7 * square_size + 10
        width = 5 * square_size + 10
        
        # for drawing the agent orientation
        offsets = [[0.4, 0.1], [0.7, 0.4], [0.4, 0.7], [0.1, 0.4]]
        dot_width = 0.2
        
        # map grid locations to squares in plot
        mapping = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13,
                   15, 17, 19,
                   21, 22, 24, 26, 27,
                   29, 30, 31, 32, 33]
        
        color_fill = "#FFFFBF" # color for non-goal squares
        agent_color = "blue"
        color_goal = "#33CC00"
        orientation_color = "white"
        
        if self.state <= 68:
            index = int(np.floor(self.state/4))
            rem = int(self.state % 4)
        else:
            index = int(np.floor((self.state + 3)/4))
            rem = int((self.state + 3) % 4)
        
        mapped_row = np.floor(mapping[index] / 7)
       
        image = Image.new("RGB", size=(height, width), color="#FFFFFF")
    
        draw = ImageDraw.Draw(image)
       
        # first and last row of squares
        for j in [0, 4]:
            for i in range(1, 6):
                shape = [(i * square_size, j * square_size), ((i+1) * square_size, (j+1) * square_size)] 
                draw.rectangle(shape, fill = color_fill, outline ="black") 
        
        # third row
        for i in [1, 3, 5]:
            shape = [(i * square_size, 2 * square_size), ((i+1) * square_size, 3 * square_size)] 
            draw.rectangle(shape, fill = color_fill, outline ="black") 
            
        # 2nd and 4th row
        for j in [1, 3]:
            color = color_fill
            for i in [0, 1, 3, 5, 6]:
                if i == 6 and j == 3:
                    color = color_goal
                shape = [(i * square_size, j * square_size), ((i+1) * square_size, (j+1) * square_size)] 
                draw.rectangle(shape, fill = color, outline = "black") 
                
        
        # draw location of agent
        draw.ellipse(((mapping[index] - mapped_row * 7) * square_size, 
                      mapped_row * square_size, 
                      (mapping[index] + 1 - mapped_row * 7) * square_size, 
                      square_size * (mapped_row + 1)), 
                     fill = agent_color, outline = agent_color)
        
        if not self.state == 68:
            # draw orientation of agent
            draw.ellipse(((mapping[index] - mapped_row * 7 + offsets[rem][0]) * square_size, 
                          (offsets[rem][1] + mapped_row) * square_size, 
                          (mapping[index] + offsets[rem][0] + dot_width - mapped_row * 7) * square_size, 
                          (offsets[rem][1] + dot_width + mapped_row) * square_size), 
                         fill = orientation_color, outline = 'black')
       
        del draw
        image.show()
    
    else:
        print("Current state:", self.state)