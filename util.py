import matplotlib.pyplot as plt
import os
from scipy.io import savemat, loadmat


plt.rcParams['legend.fontsize'] = 27
plt.rcParams['axes.labelsize'] = 27
plt.rcParams['xtick.labelsize'] = 27
plt.rcParams['ytick.labelsize'] = 27

# plt.rcParams['figure.subplot.left'] = 0.21
plt.rcParams['figure.subplot.left'] = 0.175
plt.rcParams['figure.subplot.bottom'] = 0.2
plt.rcParams['figure.subplot.right'] = 0.99
plt.rcParams['figure.subplot.top'] = 0.9

plt.rcParams['lines.linewidth'] = 2.5


current_path = os.getcwd()
png_path = current_path + '/' + 'output/png'
if not os.path.exists(png_path):
    os.makedirs(png_path)
eps_path = current_path + '/' + 'output/eps'
if not os.path.exists(eps_path):
    os.makedirs(eps_path)
mat_path = current_path + '/' + 'output/mat'
if not os.path.exists(mat_path):
    os.makedirs(mat_path)


def modify_regret_figure(figure_name, policies, jupyter_notebook):
    data_dict = loadmat(mat_path + '/' + figure_name)
    regret_dict = {}
    for policy in policies:
        regret_dict[policy.name] = data_dict[policy.name][0]
    plot_figure(policies, regret_dict)
    if not jupyter_notebook:
        save_figure(figure_name)


def save_date(figure_name, policies, regret_dict):
    data_dict = {}
    for policy in policies:
        data_dict[policy.name] = regret_dict[policy.name]
    savemat(mat_path + '/' + figure_name + '.mat', data_dict)


def save_figure(figure_name):
    plt.savefig(png_path + '/' + figure_name)
    plt.savefig(eps_path + '/' + figure_name + '.eps', format='eps')


def plot_figure(policies, regret_dict):
    # print(plt.rcParams)
    fig = plt.figure()
    plt.xlabel(r'Round $t$')
    plt.ylabel(r'Cumulative Regret $R_t$')
    for policy in policies:
        policy_regrets = regret_dict[policy.name]
        policy_color = policy.color
        policy_name = policy.name
        plt.plot(range(1, len(policy_regrets) + 1), policy_regrets, label=policy_name, color=policy_color)
        plt.legend()
    # plt.xlim((-100, 5200))
    # plt.ylim((-2, 30))


def plot_regret(figure_name, policies, regret_dict, jupyter_notebook):
    plot_figure(policies, regret_dict)
    if not jupyter_notebook:
        save_figure(figure_name)
        save_date(figure_name, policies, regret_dict)







