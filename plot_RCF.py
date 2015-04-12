import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
import dataset
import numpy as np

save_flag = True

use_latex = save_flag
use_pgf = True

n_samples = 1e4
dr = 0.2
dr_exp = 0.4
R_close = 10.0
dr_peak = 0.7

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

sim_dset_name_0 = '/Users/ejm/Desktop/droplet/data_analysis/n_2430_Rd_28.8_Drc_0_test'
sim_dset_name_inf = '/Users/ejm/Desktop/droplet/data_analysis/n_2430_Rd_28.8_Drc_inf_test'
exp_dset_name = '/Users/ejm/Desktop/droplet/data/exp/smooth/runs/D011'
exp_dset_name_2 = '/Users/ejm/Desktop/droplet/data/exp/smooth/runs/D33'

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

# dset = dataset.get_dset(exp_dset_name, filter_z_flag=True)
# R_peak, R_peak_err = dset.get_R_peak('mean', dr=dr_peak)
# g, ge, R = dset.get_rcf(dr_exp, n_samples, R_min=R_peak)
# close = R < R_close
# g = g[close]
# ge = ge[close]
# R = R[close]
# ax.errorbar(R, g, yerr=ge, label='Experiment')

dset = dataset.get_dset(exp_dset_name_2, filter_z_flag=True)
R_peak, R_peak_err = dset.get_R_peak('mean', dr=dr_peak)
g, ge, R = dset.get_rcf(dr_exp, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label='Experiment')

dset = dataset.get_dset(sim_dset_name_0)
R_peak, R_peak_err = dset.get_R_peak('mean', dr=dr_peak)
g, ge, R = dset.get_rcf(dr, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label='Simulation, $D_r^c = 0$')

dset = dataset.get_dset(sim_dset_name_inf)
R_peak, R_peak_err = dset.get_R_peak('mean', dr=dr_peak)
g, ge, R = dset.get_rcf(dr, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label='Simulation, $D_r^c = \infty$')

ax.axhline(1.0, c=ejm_rcparams.almost_black, ls='--')

ax.legend(loc='lower right', fontsize=26)

ax.set_xlabel(r'$r_{ij} / \si{\um}$', fontsize=35)
ax.set_ylabel(r'$g(r_{ij})$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(0.0, None)
ax.set_xlim(0.0, R_close)

if save_flag:
    plt.savefig('RCF.pdf', bbox_inches='tight')
else:
    plt.show()
