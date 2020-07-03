'''
Super-Class for all Hallway and Hallway2 domains.
'''
import gym
from gym import spaces
from random import choice, uniform

class Hall(gym.Env):
    
    def __init__(self, prob_action_success = 0.8, prob_see_wall_true = 0.9,
               prob_see_wall_false = 0.05, max_steps = 100):
        
        self.MAX_STEPS = max_steps # max. number of steps per episodes
        self.num_steps = 0 # steps taken so far
        
        self.PROB_ACTION_SUCCESS = prob_action_success
        self.PROB_SEE_WALL_TRUE = prob_see_wall_true
        self.PROB_SEE_WALL_FALSE = prob_see_wall_false
        
    def encode_observation(self, obs):
        '''
        Encodes observation of walls to an integer.
        '''
        return int(obs[0] * 1 + obs[1] * 2 + obs[2] * 4 + obs[3] * 8)
    
    def step(self, action):
        prob = uniform(0, 1)
        
        # action succeeds
        if prob <= self.PROB_ACTION_SUCCESS:
            self.state = int(self.NEXT_STATES_DET[self.state, action])
            
        # action does not succeed
        else:
            self.state = int(self.NEXT_STATES_DET[self.state, choice([i for i in range(self.action_space.n) if not i == action])])   
        self.num_steps += 1
        done = (self.state in self.TERMINAL_STATES) or (self.num_steps == self.MAX_STEPS - 1)
        return self.get_observation(self.state), self.R[self.state], done , ""
      
    def reset(self):
        self.state = choice(self.NON_TERMINAL_STATES)
        self.num_steps = 0
        return self.get_observation(self.state)