import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from dataset import Superset
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

superset = Superset(exp_dset_paths, theta_max=None)
R = superset.get_R()
eta, eta_err = superset.get_eta(alg, res)
f_peak, f_peak_err = superset.get_f_peak(alg, res)
gamma_fit, gamma_fit_err, k_fit, k_fit_err = superset.fit_to_model(alg, res)
f_peak_model = superset.get_f_peak_model(gamma_fit, k_fit)

ax.errorbar(R, f_peak, yerr=f_peak_err,
            ls='none',
            label=r'Experiment', zorder=0)
ax.scatter(R, f_peak_model,
           zorder=1,
           label=r'Experiment fit, $\gamma = ' + r'{:.2g}'.format(gamma_fit) + r'$, $k = ' + r'{:.2g}'.format(k_fit) + r'$')
ax.axhline(1.0, ls='--', lw=4)

ax.legend(loc='upper right')
# ax.set_xscale('log')
ax.set_xlabel(r'$R$', fontsize=35)
ax.set_ylabel(r'$\eta / \eta_0$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(5.0, 45.0)
ax.set_ylim(0.499, 1.10001)

if save_flag:
    plt.savefig('fit_R.pdf', bbox_inches='tight')
else:
    plt.show()
