import numpy as np

class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        self.size = size
        self.mu = mu        
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(106)

    def reset(self):
        self.state = np.ones(self.size) * self.mu        

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx        
        return self.state