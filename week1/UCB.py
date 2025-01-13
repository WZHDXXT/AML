import numpy as np
from bandits import Solver, BernoulliBandit

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.coef = coef
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        # p = 1/t, coef used to control the uncertainty
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count)/(2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0/(self.counts[k]) * ((r - self.estimates[k]))
        return k

np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
bandit_10_arm = BernoulliBandit(10)
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(10)
print(UCB_solver.regret)