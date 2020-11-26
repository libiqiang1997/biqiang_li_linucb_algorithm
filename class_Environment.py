import numpy as np
from class_Arm import Arm


class Environment(object):
    def __init__(self, k, d, sigma_noise):
        self.k = k
        self.d = d
        self.sigma_noise = sigma_noise

        self.unif_min = -1
        self.unif_max = 1

    def init(self):
        self.expected_rewards = np.zeros(self.k)
        self.theta = np.random.uniform(self.unif_min, self.unif_max, self.d)
        # print('self.env.theta:', self.theta)
        self.arms = []
        for i in range(self.k):
            arm_id = i
            arm_feature = np.random.uniform(self.unif_min, self.unif_max, self.d)
            expected_reward = np.inner(arm_feature, self.theta)
            arm = Arm(arm_id, arm_feature, expected_reward)
            self.arms.append(arm)
            self.expected_rewards[i] = expected_reward
            # print('arm_feature:', arm_feature)
        # print('arms:', self.arms)
        # for arm in self.arms:
        #     print('arm_id:', arm.arm_id)
        #     print('arm_feature:', arm.arm_feature)
        #     print('expected_reward:', arm.expected_reward)
        # print('expected_rewards:', self.expected_rewards)

    def play(self, choice):
        round_reward = self.arms[choice].pull(self.sigma_noise)
        return round_reward

    def get_optimal_expected_reward(self):
        optimal_expected_reward = np.max(self.expected_rewards)
        return optimal_expected_reward

    def get_selected_expected_reward(self, choice):
        selected_expected_reward = self.expected_rewards[choice]
        return selected_expected_reward