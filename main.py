from datetime import datetime
import multiprocessing as mp
from class_Environment import Environment
from class_Simulator import Simulator
from util import plot_regret, modify_regret_figure
from policy_class.class_LinUCB import LinUCB
import os

cur_dir_strs = os.getcwd().split('\\')
cur_dir_name = cur_dir_strs[len(cur_dir_strs) - 1]
jupyter_notebook = False
if cur_dir_name == 'jupyter_notebook':
    jupyter_notebook = True
    os.chdir('..')

# debug = True
debug = False
if debug:
    num_thread = 1
    num_mc = 1
    time_horizon = 10
    k = 2
    d = 2
else:
    num_thread = 10
    num_mc = 100
    time_horizon = 5000
    k = 10
    d = 5

lambda_ = 1
arm_norm_bound = 1
theta_norm_bound = 1
delta = 0.01
sigma_noise = 0.1


def parameter_multiprocessing_run(thread_id, sigma_noise):
    figure_name = ('sigma_noise' + str(sigma_noise)).replace('.', 'dot')
    bandit_env = Environment(k, d, sigma_noise)
    policies = [LinUCB('LinUCB', k, d, lambda_, arm_norm_bound, theta_norm_bound, delta, sigma_noise, 'black')]
    simulator = Simulator(bandit_env, policies, time_horizon)
    regret_dict = simulator.multiprocessing_run(num_thread, num_mc)
    plot_regret(figure_name, policies, regret_dict, jupyter_notebook)


def modify_figure():
    # modify figure
    figure_name = ('sigma_noise' + str(sigma_noise)).replace('.', 'dot')
    policies = [LinUCB('LinUCB', k, d, lambda_, arm_norm_bound, theta_norm_bound, delta, sigma_noise, 'black')]
    modify_regret_figure(figure_name, policies, jupyter_notebook)


def run_linucb():
    # run LinUCB
    threads = []
    if debug:
        thread_id = 0
        threads.append(mp.Process(target=parameter_multiprocessing_run,
                                  args=(thread_id, sigma_noise)))
        threads[0].start()
        threads[0].join()
    else:
        sigma_noises = [0.1, 1]
        for i in range(len(sigma_noises)):
            thread_id = i
            threads.append(mp.Process(target=parameter_multiprocessing_run,
                                      args=(thread_id, sigma_noises[i])))
            threads[i].start()
        for thread in threads:
            thread.join()


if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    # run_linucb()
    modify_figure()

    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)