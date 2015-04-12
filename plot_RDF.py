import matplotlib.pyplot as plt
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
        dset = Dataset(run_fnames)
        dsets.append(dset)
    return dsets


def plot_dsets(ax, dsets, use_latex):
    for dset in dsets:
        rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(res)
        vp, vp_err = dset.get_vp()
        R = dset.R
        if use_latex:
            label = (r'$\SI{' + '{:.3g}'.format(R) + r'}{\um}$, ' +
                     r'$\SI{' + '{:.2g}'.format(vp) + r'}{\percent}$')
        else:
            label = (r'$' + '{:.3g}'.format(R) + r'\mu m$, $' +
                     '{:.2g}'.format(vp) + r'\%$')
        import numpy as np
        rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
        ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                    label=label)
    ax.legend(loc='upper left', fontsize=24)

res = 0.7

exp_dset_names = ['D23',
                  'D210',
                  'D12',
                  'D32',
                  'D11',
                  'D31',
                  ]

sim_dset_names = ['n_50_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a',
                  'n_100_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a',
                  'n_200_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a',
                  'n_400_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a',
                  'n_800_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a',
                  'n_1600_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a']

data_dir = '/Users/ejm/Projects/Droplet/Data'
exp_dset_dir = '{}/Experiment/smooth/Runs'.format(data_dir)
sim_dset_dir = '{}/Simulation/2014-06-09/Runs'.format(data_dir)
exp_dsets = get_dsets(exp_dset_dir, exp_dset_names)
sim_dsets = get_dsets(sim_dset_dir, sim_dset_names)

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_exp = fig.add_subplot(gridspec[0])
ax_sim = fig.add_subplot(gridspec[1], sharex=ax_exp, sharey=ax_exp)

ejm_rcparams.prettify_axes(ax_exp, ax_sim)

plot_dsets(ax_exp, exp_dsets, use_latex)
plot_dsets(ax_sim, sim_dsets, use_latex)

ax_exp.set_ylim(0.0, 4.5)
ax_exp.set_xlim(0.0, 1.19)
ax_exp.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=12.0)
ax_exp.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
ax_exp.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_exp.tick_params(axis='both', labelsize=24, pad=10.0)
ax_sim.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_sim.get_yticklabels(), visible=False)
# fig.set_tight_layout(True)
gridspec.update(wspace=0.0)

# plt.show()
plt.savefig('Fig 2 RDF.pdf', bbox_inches='tight')
