from agents.actor import Actor
from agents.critic import Critic
from agents.experience import ReplayBuffer
from agents.noise import OUNoise
import numpy as np

class DDPG():
    
    def __init__(self, task):        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())    
        
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())        
        
        self.exploration_mu = 0.1
        self.exploration_sigma = 0.1
        self.exploration_theta = 0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        self.buffer_size = 100000000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size)

        self.gamma = 0.99
        self.tau = 0.001
        
    def act(self, states):
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())

    def learn(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size, self.action_size, self.state_size)

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        self.update_target_weights(self.critic_local.model, self.critic_target.model)
        self.update_target_weights(self.actor_local.model, self.actor_target.model)    
    
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.buffer.add(self.last_state, action, reward, next_state, done)
        self.learn()
        self.last_state = next_state

    def update_target_weights(self, local_model, target_model):
        target_model.set_weights(self.tau * np.array(local_model.get_weights()) + 
                                 (1 - self.tau) * np.array(target_model.get_weights()))