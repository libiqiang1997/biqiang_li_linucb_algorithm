import multiprocessing as mp
import numpy as np


class Simulator(object):
    def __init__(self, env, policies, time_horizon):
        self.env = env
        self.policies = policies
        self.time_horizon = time_horizon

    def multiprocessing_run_each_thread(self, thread_id, thread_num_mc, thread_avg_regret_dict):
        optimal_expected_rewards = np.zeros(self.time_horizon)
        selected_expected_rewards = np.zeros(self.time_horizon)
        cum_regret_dict = {}
        avg_regret_dict = {}
        # optimal_expected_reward_dict = {}
        # selected_expected_reward_dict = {}
        for policy in self.policies:
            cum_regret_dict[policy.name] = np.zeros((thread_num_mc, self.time_horizon))
            avg_regret_dict[policy.name] = np.zeros(self.time_horizon)
            # optimal_expected_reward_dict[policy.name] = np.zeros(self.time_horizon)
            # selected_expected_reward_dict[policy.name] = np.zeros(self.time_horizon)
        for n_experiment in range(thread_num_mc):
            self.env.init()
            for policy in self.policies:
            # for arm in self.env.arms:
            #     print('arm_id:', arm.arm_id)
            #     print('arm_feature:', arm.arm_feature)
            #     print('expected_reward:', arm.expected_reward)
                policy.init()
                for t in range(1, self.time_horizon + 1):
                    # choose arm
                    choice = policy.select_arm(self.env.arms)
                    # calculate regret
                    optimal_expected_reward = self.env.get_optimal_expected_reward()
                    optimal_expected_rewards[t-1] = optimal_expected_reward
                    selected_expected_reward = self.env.get_selected_expected_reward(choice)
                    selected_expected_rewards[t-1] = selected_expected_reward
                    # update model
                    chosen_arm = self.env.arms[choice]
                    round_reward = self.env.play(choice)
                    policy.update(chosen_arm, round_reward)
                # expected_regrets = optimal_expected_reward_dict[policy.name] - selected_expected_reward_dict[policy.name]
                expected_regrets = optimal_expected_rewards - selected_expected_rewards
                cum_regret_dict[policy.name][n_experiment, :] = np.cumsum(expected_regrets)
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.mean(cum_regret_dict[policy.name], 0)
        thread_avg_regret_dict[thread_id] = avg_regret_dict

    def multiprocessing_run(self, num_thread, num_mc):
        manager = mp.Manager()
        thread_avg_regret_dict = manager.dict()
        threads = []
        thread_border = []
        thread_step = num_mc // num_thread
        for i in range(num_thread):
            thread_border.append(i * thread_step)
        thread_border.append(num_mc)
        for i in range(num_thread):
            thread_id = i
            thread_num_mc = thread_border[i+1] - thread_border[i]
            threads.append(mp.Process(target=self.multiprocessing_run_each_thread,
                                      args=(thread_id, thread_num_mc, thread_avg_regret_dict)))
            threads[i].start()
        for i in range(num_thread):
            threads[i].join()

        avg_regret_dict = {}
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.zeros(self.time_horizon)
        for policy in self.policies:
            for i in range(num_thread):
                avg_regret_dict[policy.name] += thread_avg_regret_dict[i][policy.name]
            avg_regret_dict[policy.name] /= num_thread

        return avg_regret_dict