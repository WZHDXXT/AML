import numpy as np

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        # the times each bandit has been choosen
        self.counts = np.zeros(self.bandit.K)
        # regret at one time
        self.regret = 0
        # record the action(which bandit has been choosen)
        self.actions = []
        # regrets for all time
        self.regrets = []
    
    def update_regret(self, k):
        # r(a) += Q* - Q(k)
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step():
        # which bandit to choose
        pass
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # initialize the estimated expected reward value
        self.estimates = np.array([init_prob] * self.bandit.K)
    
    def run_one_step(self):
        if np.random.random()<self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. /(self.counts[k]+1) * (r - self.estimates[k])
        return k

class BernoulliBandit:
    def __init__(self, K):
        # the probability in K arms
        self.probs = np.random.uniform(size=K)
        # the arms number with the biggest probability
        self.best_idx = np.argmax(self.probs)
        # the biggest probability
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    
    def step(self, k):
        # whether getting reward or not after choosing the arm
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K=K)
epsilon = EpsilonGreedy(bandit_10_arm, 0.01, 1.0)
epsilon.run(10)
print(epsilon.regret)
