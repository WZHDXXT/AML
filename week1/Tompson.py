import numpy as np
from bandits import Solver, BernoulliBandit

class Tompson(Solver):
    def __init__(self, bandit):
        super(Tompson, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)
    
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k) 
        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

np.random.seed(1)
bandit_10_arm = BernoulliBandit(10)
thompson_sampling_solver = Tompson(bandit_10_arm)
thompson_sampling_solver.run(100)