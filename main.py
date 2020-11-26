from datetime import datetime
import multiprocessing as mp
from class_Environment import Environment
from class_Simulator import Simulator
from util import plot_regret
from policy_class.class_LinUCB import LinUCB

debug = True
# debug = False
if debug:
    num_thread = 1
    num_mc = 1
    time_horizon = 10
k = 2
d = 2


def parameter_multiprocessing_run():
    bandit_env = Environment(time_horizon, k, d)
    policies = [LinUCB('LinUCB')]
    simulator = Simulator(bandit_env, policies)
    regret_dict = simulator.multiprocessing_run(num_thread, num_mc)
    plot_regret(regret_dict)


if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    threads = []
    threads.append(mp.Process(target=parameter_multiprocessing_run))
    threads[0].start()
    threads[0].join()

    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)