import numpy as np

from agents.ddpg_agent.actor import Actor
from agents.ddpg_agent.critic import Critic
from agents.ddpg_agent.replay_buffer import ReplayBuffer
from agents.ddpg_agent.ou_noise import OUNoise


class DDPGAgent():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2

        # not good
        # self.exploration_theta = 0.05
        # self.exploration_sigma = 0.1

        # no improvement, and drop at last
        # self.exploration_theta = 0.01
        # self.exploration_sigma = 0.01

        # no obvious improvement  
        # self.exploration_theta = 0.5
        # self.exploration_sigma = 1

        # closer to target, but terminate fast
        # self.exploration_theta = 50
        # self.exploration_sigma = 50

        # bad
        # self.exploration_theta = 0.15
        # self.exploration_sigma = 50

        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000

        # self.batch_size = 64
        # self.batch_size = 128  # bad for move
        self.batch_size = 512  # has improvement
        # self.batch_size = 4096  # bad

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        # self.gamma = 0.8  # no obvious improvement

        # function as learning rate
        # self.tau = 0.01  # for soft update of target parameters
        # self.tau = 0.1  # big improvement with basic implementation
        # self.tau = 0.3  # similar as 0.1
        # self.tau = 0.1  # flucate with L2 regularization
        self.tau = 0.3  # stable with L2 regularization, suboptimal for moving
        # self.tau = 0.4  # 
        # self.tau = 0.6  # one high spike for moving        

        # Score tracker and learning parameters
        self.total_reward = 0
        self.count = 0
        self.score = 0
        self.best_total_reward = -np.inf
        self.best_count = None
        self.best_score = -np.inf
        self.best_episode = None

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state

        self.total_reward = 0.0
        self.count = 0
        self.score = 0

        return state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def step(self, action, reward, next_state, done, episode):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        # metrics
        self.total_reward += reward
        self.count += 1
        if done:
            self.score = self.total_reward / float(self.count) if self.count else 0.0
            if self.total_reward > self.best_total_reward:
                self.best_total_reward = self.total_reward
                self.best_count = self.count
                self.best_score = self.score
                self.best_episode = episode

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target.model.predict_on_batch(next_states)
        next_Q_targets = self.critic_target.model.predict_on_batch([next_states, next_actions])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * next_Q_targets * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

