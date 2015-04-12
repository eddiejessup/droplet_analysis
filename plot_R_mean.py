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


def get_stat(dset_paths):
    superset = Superset(dset_paths)
    R_mean, R_mean_err = superset.get_mean()
    vf, vf_err = superset.get_vf()
    return vf, vf_err, R_mean, R_mean_err

vf, vf_err, R_mean, R_mean_err = get_stat(exp_dset_paths)
ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Experiment')

# vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_14_Drc_0_dset_paths)
# ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = 0$')

# vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_14_Drc_0_1_dset_paths)
# ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = 0.1$')

# vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_14_Drc_1_dset_paths)
# ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = 1$')

# vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_14_Drc_10_dset_paths)
# ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = 10$')

vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_14_Drc_inf_dset_paths)
ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = \infty$')

vf, vf_err, R_mean, R_mean_err = get_stat(Dr_0_1_Drc_inf_dset_paths)
ax.errorbar(vf, R_mean, xerr=vf_err, yerr=R_mean_err, ls='none', label=r'Simulation, $D_r^c = \infty$')

ax.axhline(3.0 / 4.0, c=ejm_rcparams.set2[4], label='Uniform')
ax.axhline(1.0, c=ejm_rcparams.set2[5], label='Complete accumulation')

ax.legend(loc='lower left', fontsize=26)
ax.set_xscale('log')
ax.set_xlabel(r'$\phi$', fontsize=35)
ax.set_ylabel(r'$\bar{r} / R$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
# ax.set_xlim(0.0003, 0.08)
ax.set_ylim(0.7, 1.02)

if save_flag:
    plt.savefig('R_mean.pdf', bbox_inches='tight')
else:
    plt.show()
