import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from dataset import Superset
import numpy as np
from paths import *

save_flag = False

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

res = 0.7
alg = 'mean'


def get_stat(dset_paths):
    superset = Superset(dset_paths)
    eta_0, eta_0_err = superset.get_eta_0()
    f_peak, f_peak_err = superset.get_f_peak(alg, res)
    return eta_0, eta_0_err, f_peak, f_peak_err

eta_0, eta_0_err, f_peak, f_peak_err = get_stat(exp_dset_paths)
ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Experiment')

eta_0, eta_0_err, f_peak, f_peak_err = get_stat(zero_dset_paths)
ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Simulation, $D_r^c = 0$')

# eta_0, eta_0_err, f_peak, f_peak_err = get_stat(D0_1_dset_paths)
# ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Simulation, $D_r^c = 0.1$')

# eta_0, eta_0_err, f_peak, f_peak_err = get_stat(D1_dset_paths)
# ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Simulation, $D_r^c = 1$')

# eta_0, eta_0_err, f_peak, f_peak_err = get_stat(D10_dset_paths)
# ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Simulation, $D_r^c = 10$')

eta_0, eta_0_err, f_peak, f_peak_err = get_stat(inf_dset_paths)
ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Simulation, $D_r^c = \infty$')

ax.axhline(1.0, label='Complete accumulation')

ax.legend(loc='upper right', fontsize=26)
ax.set_xscale('log')
ax.set_xlabel(r'$\eta_0$', fontsize=35)
ax.set_ylabel(r'$\eta / \eta_0$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(0.199, 1.101)
ax.set_xlim(0.006, 0.6)

if save_flag:
    plt.savefig('dependence_eta.pdf', bbox_inches='tight')
else:
    plt.show()
