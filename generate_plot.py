# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os
import matplotlib
import tueplots
from tueplots import bundles, axes, fonts, cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["font.family"] = "Times Roman"
plot_figures_in_one_line = True
if not plot_figures_in_one_line:
    matplotlib.rcParams['figure.figsize'] = 5.7, 4
else:
    matplotlib.rcParams['figure.figsize'] = 10, 2
# matplotlib.rcParams.update(cycler.cycler(color=['#7229D9', '#668CD9', '#F2B035', '#F2600C', '#D90707'], marker=markers.o_sized[1:6]))
pl = palettes.muted[:5]
# pl[2], pl[3] = pl[3], pl[2]
# matplotlib.rcParams.update(cycler.cycler(color=palettes.muted[:5], marker=['o', 'v', '^', 'D', 'p']))
matplotlib.rcParams.update(cycler.cycler(color=['#ad7021', '#e7cf94', '#eb9172', '#98d7cd', '#23867e'], marker=['o', 'v', '^', 'D', 'p']))
matplotlib.rcParams.update({'font.size': 13})

path = "./saved_outputs/slurm-%d.out"
keys = ['RMSE_U', 'RMSE_I', 'rmse', 'mse', 'mae', 'ndcg']
def load_results(filename):
    f = open(filename, 'r')
    results = []
    for line in f.readlines():
        if ('RMSE_U' in line) and ('mse' in line) and ('testset' not in line):
            result = {}
            for l in line.strip()[1:-1].split(','):
                (k, v) = l.strip().split(':')
                k, v = k.strip()[1:-1], float(v.strip())
                assert k in keys
                result[k] = v
            results.append(result)
    return results

# # # draw horizontal boxplot for method comparisons on semi-synthetic data.
# method_names = ['mf_naive', 'mf_ips_pos', 'mf_ips_pop', 'mf_ips_cor']
# out_file = "./images/sim_pos/"
# if not os.path.exists(out_file):
#     os.makedirs(out_file)

# # slurm_ids = [659989, 659993, 659996, 660763]
# slurm_ids = [660629, 660006, 660008, 660762] # ALS
# results_models = [load_results(path % x) for x in slurm_ids]

# for k in ['mse']:
#     values_models = []
#     avgs, stds = [], []
#     for i in range(len(method_names)):
#         results = results_models[i]
#         results_for_k = [x[k] for x in results]
#         values_models.append(results_for_k)
#         avgs.append(np.mean(results_for_k))
#         stds.append(np.std(results_for_k))
#     plt.bar(method_names, avgs)
#     plt.errorbar(method_names, avgs, yerr=stds, fmt='none', capsize=5, color='gray')
#     # plt.xticks(range(1, len(method_names)+1), method_names)
#     plt.ylabel(k.upper())
#     plt.savefig(out_file + "bar_plot_%s_ALS.pdf" % k, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     plt.close()


# # draw the lineplots on synthetic data for multifactorial bias with mul_alpha varing from 0.1 to 0.9
out_file = "./images/sim_mul/"
path = './saved_outputs/not_sure/slurm-%d_%d.out'
if not os.path.exists(out_file):
    os.makedirs(out_file)

def ci_bootstrapping(data):
    num_bootstrap_samples = 1000
    bootstrap_samples = np.random.choice(data, (num_bootstrap_samples, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    # for 95% CI
    lower_percentile = 2.5  # 2.5th percentile for lower bound
    upper_percentile = 97.5  # 97.5th percentile for upper bound
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    return lower_bound, upper_bound



# def load_last_line(filename):
#     f = open(filename, 'r')
#     results = []
#     line = f.readlines()[-1]
#     assert "['RMSE_U', 'RMSE_I', 'rmse', 'mse', 'mae', 'ndcg', 'valid_loss']" in line
#     result = {}
#     keys = ['RMSE_U', 'RMSE_I', 'rmse', 'mse', 'mae', 'ndcg', 'valid_loss']
#     avaliable_kid = [3, 4]
#     avaliable_vs = line.strip().split(' ')[-14:]
#     avaliable_vs = [x[1:-1] if "(" in x else x for x in avaliable_vs]
#     for vid in avaliable_kid:
#         result[keys[vid]] = (avaliable_vs[vid * 2], avaliable_vs[vid * 2 + 1])
#     return result
using_std_or_ci = 'ns_ci' # 'std' for standard deviation; 'ci' for confident interval, 'ns_ci' for non-symmetric CI
print("Using", using_std_or_ci)
keys = ['RMSE_U', 'RMSE_I', 'rmse', 'mse', 'mae', 'ndcg', 'valid_loss']
def load_results(filename):
    f = open(filename, 'r')
    results = []
    for line in f.readlines():
        if ('RMSE_U' in line) and ('mse' in line) and ('testset' not in line):
            result = {}
            for l in line.strip()[1:-1].split(','):
                (k, v) = l.strip().split(':')
                k, v = k.strip()[1:-1], float(v.strip())
                # print(k, v)
                assert k in keys
                result[k] = v
            results.append(result)
    results = results[:10]
    results_final = {}
    for k in keys:
        v = [x[k] for x in results]
        if using_std_or_ci == 'std':
            std = np.std(v)
        elif using_std_or_ci == 'ci':
            ci = st.t.interval(confidence=0.95, df=len(v)-1, loc=np.mean(v), scale=st.sem(v))
            std = ci[1] - np.mean(v)
        else:
            ns_ci = ci_bootstrapping(v)
            std = ns_ci
        results_final[k] = (np.mean(v), std)
    return results_final



# slurm_id, sub_ids = 710690, range(21, 66) # old one
slurm_id, sub_ids = 728613, range(51, 96) # updated
results_models = [[] for _ in range(5)] 
models = ['none', 'GT', 'pos', 'pop', 'mul']
for x in sub_ids:
    # result_models = [load_last_line(path % (slurm_id, x)) for x in range(21, 66)]
    results_models[(x-1) % 5].append(load_results(path % (slurm_id, x)))
# # read results for alpha1, only positivity bias
# slurm_ids = [694666, 694667, 694669, 694672, 694673] # old
# path = './saved_outputs/sim/slurm-%d.out'
slurm_ids = ['729451_%d' % x for x in range(106, 111)]
path = './saved_outputs/not_sure/slurm-%s.out'
results_models_a1 = [[] for _ in range(5)]
for sid, slurm_id in enumerate(slurm_ids):
    results_models_a1[sid % 5].append(load_results(path % slurm_id))
# # read results for alpha0, only popularity bias
# slurm_ids = ['695370_%d' % x for x in range(6, 11)]
slurm_ids = ['729451_%d' % x for x in range(101, 106)]
path = './saved_outputs/not_sure/slurm-%s.out'
results_models_a0 = [[] for _ in range(5)]
for sid, slurm_id in enumerate(slurm_ids):
    results_models_a0[sid % 5].append(load_results(path % slurm_id))
# print(results_models)
# for a in range(9):
#     print("alpha:", 0.1 * (a+1))
#     for mid, model in enumerate(models):
#         print(" -- Method-", model, "result:", results_models[mid][a])
#     print()
if not plot_figures_in_one_line:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2)

# # plot the lines with error bands to show the variance; each line shows the results on different mul_alpha for each model
model_names = ['MF', r'MF-IPS$^{GT}$', r'MF-IPS$^{Pos}$', r'MF-IPS$^{Pop}$', r'MF-IPS$^{Mul}$']
metric = 'mse'
def draw(metric = 'mse', ax=None):
    alphas = [round((a+1) * 0.1, 1) for a in range(9)]
    for mid, model in enumerate(models):
        results = [results_models_a0[mid][0][metric]] + [results_models[mid][a][metric] for a in range(len(alphas))] + [results_models_a1[mid][0][metric]]
        results_v, results_std = np.array([float(x[0]) for x in results]), np.array([float(x[1]) if len(x[1])==1 else [float(x[1][0]), float(x[1][1])] for x in results])
        print(results_v, results_std)
        if 'GT' in model_names[mid]:
            ax.plot([0.0]+alphas+[1.0], results_v, label=model_names[mid], markerfacecolor='none', linestyle='dashed')
        else:
            ax.plot([0.0]+alphas+[1.0], results_v, label=model_names[mid], markerfacecolor='none')
        if using_std_or_ci == 'ns_ci':
            ax.fill_between([0.0]+alphas+[1.0], results_std[:, 0], results_std[:, 1], alpha=0.3)
        else:    
            ax.fill_between([0.0]+alphas+[1.0], results_v - results_std, results_v + results_std, alpha=0.3)
    ax.set_ylabel(metric.upper())
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 2, 4, 1]
    if metric == 'mse':
        # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        if not plot_figures_in_one_line:
            ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center', bbox_to_anchor=(0.45, 1.2), ncol=3, frameon=False)
        else:
            ax.legend("", frameon=False)
            ax.set_xlabel(r'$\gamma$')
    else:
        if plot_figures_in_one_line:
            ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False, fontsize='12.5')
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False, fontsize='9.5')
        else:
            ax.legend("", frameon=False)
        ax.set_xlabel(r'$\gamma$')

    if metric == 'mse':
        ax.set_ylim((0.68, 1.35))
        ax.set_yticks([0.8, 1.0, 1.2])
        # ax = plt.gca()
        # ax.set_xticklabels([])
    elif metric == 'mae':
        ax.set_ylim((0.58, 0.88))
        ax.set_yticks([0.6, 0.7, 0.8])
draw('mse', ax=ax1)
draw('mae', ax=ax2)
for ax in [ax1, ax2]:
    ax.tick_params(top=False,bottom=True,left=True,right=True, direction='in')
    ax.set_xlim((0,1))
    ax.set_xticks([round(x*0.1, 1) for x in range(0, 11)])

if plot_figures_in_one_line:
    plt.subplots_adjust(wspace=0.18)   
else: 
    plt.subplots_adjust(hspace=0.1)

# plt.savefig(out_file + "sim_mul_plot_%s.pdf" % metric, bbox_inches='tight', pad_inches=0)
# print("Fig saved to", out_file + "sim_mul_plot_%s.pdf" % metric)
plt.savefig(out_file + "sim_mul_plot.pdf", bbox_inches='tight', pad_inches=0)
print("Fig saved to", out_file + "sim_mul_plot.pdf")
plt.show()
plt.close()