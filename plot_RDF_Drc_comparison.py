import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset
import glob
from matplotlib.gridspec import GridSpec
from ciabatta import ejm_rcparams

use_latex = True
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)


def get_dsets(dset_dir, dset_names):
    dsets = []
    for name in dset_names:
        run_fnames = glob.glob('{}/{}/dyn/*.npz'.format(dset_dir, name))
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
        if use_latex:
            label = r'$\phi = \SI{' + '{:.2g}'.format(vp) + r'}{\percent}$'
        else:
            label = r'$\phi = {:.2g}'.format(vp) + r'\%$'
        plot_dset(ax, dset, res, label)

res = 0.7

Drc_zero_dset_names = [
    'n_50_Rd_16_Drc_0_test',
    'n_100_Rd_16_Drc_0_test',
    'n_200_Rd_16_Drc_0_test',
    'n_400_Rd_16_Drc_0_test',
    'n_800_Rd_16_Drc_0_test',
    # 'n_1600_Rd_16_Drc_0_test',
]

Drc_inf_dset_names = [
    'n_50_Rd_16_Drc_inf_test',
    'n_100_Rd_16_Drc_inf_test',
    'n_200_Rd_16_Drc_inf_test',
    'n_400_Rd_16_Drc_inf_test',
    'n_800_Rd_16_Drc_inf_test',
    # 'n_1600_Rd_16_Drc_inf_test',
]

# sim_dset_dir = '/Users/ejm/Desktop/droplet/data/sim/2015-03-16'
sim_dset_dir = '/Users/ejm/Desktop/droplet/data_analysis/'
Drc_zero_dsets = get_dsets(sim_dset_dir, Drc_zero_dset_names)
Drc_inf_dsets = get_dsets(sim_dset_dir, Drc_inf_dset_names)

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_Drc_zero = fig.add_subplot(gridspec[0])
ax_Drc_inf = fig.add_subplot(gridspec[1], sharex=ax_Drc_zero,
                             sharey=ax_Drc_zero)

ejm_rcparams.prettify_axes(ax_Drc_zero, ax_Drc_inf)
plot_dsets(ax_Drc_zero, Drc_zero_dsets, use_latex)
plot_dsets(ax_Drc_inf, Drc_inf_dsets, use_latex)
ax_Drc_zero.legend(loc='upper left', fontsize=26)

ax_Drc_zero.set_ylim(0.0, 4.5)
ax_Drc_zero.set_xlim(0.0, 1.19)
ax_Drc_zero.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=5.0)
ax_Drc_zero.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0, labelpad=20.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
fig.text(0.31, 0.95, '$D_r^c = 0', ha='center', va='center', fontsize=30)
fig.text(0.71, 0.95, '$D_r^c = \infty', ha='center', va='center', fontsize=30)
# ax_Drc_zero.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_Drc_zero.tick_params(axis='both', labelsize=24, pad=10.0)
ax_Drc_inf.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_Drc_inf.get_yticklabels(), visible=False)
gridspec.update(wspace=0.0)

# plt.show()
plt.savefig('RDF_Drc_comparison.pdf', bbox_inches='tight')
