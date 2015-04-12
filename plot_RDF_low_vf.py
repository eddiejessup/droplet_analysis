import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset
import glob
from matplotlib.gridspec import GridSpec
from ciabatta import ejm_rcparams

use_latex = True
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)


def get_dsets(dset_dir, dset_names, exp):
    dsets = []
    for name in dset_names:
        run_fnames = glob.glob('{}/{}/dyn/*.npz'.format(dset_dir, name))
        if not exp:
            run_fnames = run_fnames[-10:]
        dset = Dataset(run_fnames)
        dsets.append(dset)
    return dsets


def plot_dset(ax, dset, res, label):
    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(res)
    rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err, label=label)


def plot_dsets(ax, dsets, use_latex):
    for dset in dsets:
        vp, vp_err = dset.get_vp()
        R = dset.R
        if use_latex:
            label = (r'$\SI{' + '{:.3g}'.format(R) + r'}{\um}$, ' +
                     r'$\SI{' + '{:.2g}'.format(vp) + r'}{\percent}$')
        else:
            label = (r'$' + '{:.3g}'.format(R) + r'\mu m$, $' +
                     '{:.2g}'.format(vp) + r'\%$')
        plot_dset(ax, dset, res, label)

res = 0.7

exp_dset_names = [
    'D116',
    'D23',
    'D014',
]

sim_dset_names = [
    'n_13_Rd_9.2_Drc_inf',
    'n_52_Rd_15.8_Drc_inf',
    'n_992_Rd_42.9_Drc_inf',
]

data_dir = '/Users/ejm/Desktop/droplet/data'
exp_dset_dir = '{}/exp/smooth/runs'.format(data_dir)
sim_dset_dir = '{}/sim/2015-03-16'.format(data_dir)
exp_dsets = get_dsets(exp_dset_dir, exp_dset_names, True)
sim_dsets = get_dsets(sim_dset_dir, sim_dset_names, False)

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_exp = fig.add_subplot(gridspec[0])
ax_sim = fig.add_subplot(gridspec[1], sharex=ax_exp, sharey=ax_exp)

ejm_rcparams.prettify_axes(ax_exp, ax_sim)
plot_dsets(ax_exp, exp_dsets, use_latex)
plot_dsets(ax_sim, sim_dsets, use_latex)
ax_exp.legend(loc='upper left', fontsize=26)

ax_exp.set_ylim(0.0, 5.0)
ax_exp.set_xlim(0.0, 1.19)
ax_exp.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=5.0)
ax_exp.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0, labelpad=20.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
fig.text(0.31, 0.95, 'Experiment', ha='center', va='center', fontsize=30)
fig.text(0.71, 0.95, 'Simulation', ha='center', va='center', fontsize=30)
# ax_exp.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_exp.tick_params(axis='both', labelsize=24, pad=10.0)
ax_sim.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_sim.get_yticklabels(), visible=False)
gridspec.update(wspace=0.0)

# plt.show()
plt.savefig('RDF_low_vf.pdf', bbox_inches='tight')
