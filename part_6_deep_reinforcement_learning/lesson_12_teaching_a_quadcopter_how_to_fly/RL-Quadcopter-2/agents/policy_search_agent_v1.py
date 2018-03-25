import numpy as np

class PolicySearchAgentV1():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))  # start producing actions in a decent range
        ) 

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.best_episode = None
        self.best_total_reward = None
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        
        # bound action
        action = [np.max([i, self.action_low]) for i in action]
        action = [np.min([i, self.action_high]) for i in action]
            
        # use max action
#         action = [self.action_high for i in action]            
        
        return action

    def step(self, reward, done, episode):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self._learn(episode)
            
    def _learn(self, episode):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_episode = episode
            self.best_total_reward = self.total_reward
            self.best_w = self.w
#             self.noise_scale = max(0.5 * self.noise_scale, 0.01)
            self.noise_scale = 0.1
        else:
            self.w = self.best_w
#             self.noise_scale = min(2.0 * self.noise_scale, 3.2)
#             self.noise_scale = min(100.0 * self.noise_scale, 300.0)
            self.noise_scale = 100.0
        
        # equal noise in all directions            
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  
        