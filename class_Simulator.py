import multiprocessing as mp
import numpy as np


class Simulator(object):
    def __init__(self, env, policies):
        self.env = env
        self.policies = policies

    def multiprocessing_run_each_thread(self, thread_id, thread_num_mc, thread_avg_regret_dict):
        avg_regret_dict = {}
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.zeros(self.env.time_horizon)
        for n_experiment in range(thread_num_mc):
            self.env.init()
        thread_avg_regret_dict[thread_id] = avg_regret_dict

    def multiprocessing_run(self, num_thread, num_mc):
        manager = mp.Manager()
        thread_avg_regret_dict = manager.dict()
        threads = []
        thread_bound = []
        thread_step = num_mc // num_thread
        for i in range(num_thread):
            thread_bound.append(i * thread_step)
        thread_bound.append(num_mc)
        for i in range(num_thread):
            thread_id = i
            thread_num_mc = thread_bound[i+1] - thread_bound[i]
            threads.append(mp.Process(target=self.multiprocessing_run_each_thread,
                                      args=(thread_id, thread_num_mc, thread_avg_regret_dict)))
            threads[i].start()
        for i in range(num_thread):
            threads[i].join()

        avg_regret_dict = {}
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.zeros(self.env.time_horizon)
        for policy in self.policies:
            for i in range(num_thread):
                avg_regret_dict[policy.name] += thread_avg_regret_dict[i][policy.name]
            avg_regret_dict[policy.name] /= num_thread

        return avg_regret_dict