import matplotlib.pyplot as plt
from dataset import Dataset
import glob
import ejm_rcparams

use_latex = True
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

res = 0.7
data_dir = '/Users/ejm/Projects/Droplet/Data'


for exp in [True, False]:

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
            ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                        label=label, color=color)
        ax.text(0.07, 4.3, label, fontsize=32,
                horizontalalignment='left',
                verticalalignment='center')

    if exp:
        dset_names = ['D116',
                      'D217',
                      'D39',
                      'D23',
                      'D212',
                      'D31',
                      'D224',
                      'D34',
                      'D36']
        dset_dir = '{}/Experiment/smooth/Runs'.format(data_dir)
        fname = 'Fig S4a Experiment grid'
        color = ejm_rcparams.set2[0]
    else:
        dset_names = ['n_8_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_8.7',
                      'n_33_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_8.7',
                      'n_176_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_8.7',
                      'n_33_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_14.1',
                      'n_141_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_14.1',
                      'n_749_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_14.1',
                      'n_204_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_25.9',
                      'n_873_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_25.9',
                      'n_4644_v_13.5_D_0.25_Dr_0.14_Drc_inf_R_0.383_l_1_a_Rd_25.9']
        dset_dir = '{}/Simulation/2014-06-09/Runs'.format(data_dir)
        fname = 'Fig S4b Simulation grid'
        color = ejm_rcparams.set2[1]

    dsets = get_dsets(dset_dir, dset_names)

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(16, 12))
    axs = axs.flatten()
    ejm_rcparams.prettify_axes(*axs)

    for i, ax in enumerate(axs):
        plot_dsets(ax, [dsets[i]], use_latex)

    ax_0 = axs[0]

    ax_0.set_ylabel(r'$\rho(r) / \rho_0$',
                    fontsize=40, labelpad=12.0, alpha=0.0)
    ax_b = axs[-1]
    ax_b.set_xlabel(r'$r / R$', fontsize=40, alpha=0.0)
    fig.text(0.52, 0.02, r'$r / R$', ha='center', va='center', fontsize=40)
    fig.text(0.03, 0.54, r'$\rho(r) / \rho_0$', rotation=90, ha='center',
             va='center', fontsize=40)
    for ax in axs:
        ax.tick_params(axis='both', labelsize=24, pad=10.0)
    fig.set_tight_layout(True)

    ax_0.set_ylim(0.0, 4.99)
    ax_0.set_xlim(0.0, 1.19)

    # plt.show()
    plt.savefig('{}.pdf'.format(fname), bbox_inches='tight')
