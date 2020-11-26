import numpy as np


class LinUCB(object):
    def __init__(self, name, k, d, lambda_, arm_norm_bound, theta_norm_bound, delta, sigma_noise, color):
        self.name = name
        self.k = k
        self.d = d
        self.lambda_ = lambda_
        self.arm_norm_bound = arm_norm_bound
        self.theta_norm_bound = theta_norm_bound
        self.delta = delta
        self.sigma_noise = sigma_noise
        self.color = color

    def init(self):
        self.matrix = self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)
        self.t = 1
        # print('self.matrix:', self.matrix)
        # print('self.b:', self.b)

    def select_arm(self, arms):
        ucbs = np.zeros(self.k)
        inv_matrix = np.linalg.inv(self.matrix)
        estimated_theta = np.inner(inv_matrix, self.b)

        # beta sqrt
        # print('matrix_norm_temp_one:', matrix_norm_temp_one)
        # print('arm.arm_feature:', arm.arm_feature)
        # print('matrix_norm_temp_two:', matrix_norm_temp_two)
        molecule_term = 1 + self.t * self.arm_norm_bound ** 2 / self.lambda_
        exponent_term = molecule_term / self.delta
        log_term = np.log(exponent_term)
        square_term = self.d * log_term
        sqrt_term = np.sqrt(square_term)
        beta_first_term = self.sigma_noise * sqrt_term
        beta_second_term = np.sqrt(self.lambda_) * self.theta_norm_bound
        beta_sqrt = beta_first_term + beta_second_term
        for (i, arm) in enumerate(arms):
            # estimated reward
            estimated_reward = np.inner(arm.arm_feature, estimated_theta)

            # alpha
            # norm term
            matrix_norm_temp_one = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_temp_two = np.inner(matrix_norm_temp_one, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_temp_two)

            # alpha
            alpha = beta_sqrt * matrix_norm

            # ucb
            ucb = estimated_reward + alpha
            ucbs[i] = ucb
        # print('ucbs:', ucbs)
        mixer = np.random.random(self.k)
        ucb_indices = list(np.lexsort((mixer, ucbs)))
        ucb_indices_reverse = ucb_indices[::-1]
        choice = ucb_indices_reverse[0]
        # print('choice:', choice)
        return choice

    def update(self, chosen_arm, round_reward):
        arm_feature = chosen_arm.arm_feature
        matrix_addition = np.outer(arm_feature, arm_feature)
        self.matrix += matrix_addition
        b_addition = round_reward * arm_feature
        self.b += b_addition
        self.t += 1