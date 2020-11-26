import numpy as np


class Environment(object):
    def __init__(self, time_horizon, k, d):
        self.time_horizon = time_horizon
        self.k = k
        self.d = d

        self.unif_min = -1
        self.unif_max = 1

    def init(self):
        self.expected_rewards = np.zeros(self.k)
        self.theta = np.random.uniform(self.unif_min, self.unif_max, self.d)
        print('self.env.theta:', self.theta)
        # self.arms = []
        for i in range(self.k):
            arm_feature = np.random.uniform(self.unif_min, self.unif_max, self.d)
            self.expected_rewards[i] = np.inner(arm_feature, self.theta)
            print('arm_feature:', arm_feature)
        print('self.expected_rewards:', self.expected_rewards)