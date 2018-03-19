import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.01, gamma=1.0, episode_constant=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.alpha = alpha 
        self.gamma = gamma
        
        self.episode_constant = episode_constant
        
        self.i_episode = 0
        

    def _epsilon_greedy_state_action_probs(self, action_values):       
        epsilon = 1.0 / (self.i_episode / self.episode_constant + 1)
        
        num_actions = len(action_values)      

        avg_action_prob = float(epsilon) / num_actions
        action_probs = avg_action_prob * np.ones(num_actions)

        max_action_idx = np.argmax(action_values)            
        greedy_action_prob = 1 - epsilon + avg_action_prob                
        action_probs[max_action_idx] = greedy_action_prob

        return action_probs

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # random solution:
        # action = np.random.choice(self.nA)
        
        action_probs = self._epsilon_greedy_state_action_probs(self.Q[state])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)         
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # sample code as placeholder
        # self.Q[state][action] += 1

        next_state_action_values = self.Q[next_state]
        next_state_action_probs = self._epsilon_greedy_state_action_probs(next_state_action_values)                                    
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
            reward + self.gamma * np.dot(next_state_action_probs, next_state_action_values) - self.Q[state][action]
        )
        