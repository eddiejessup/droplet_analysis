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
        if use_latex:
            label = r'$\phi = \SI{' + '{:.2g}'.format(vp) + r'}{\percent}$'
        else:
            label = r'$\phi = {:.2g}'.format(vp) + r'\%$'
        rhos_norm += 1e-5
        ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                    label=label)

res = 0.7

low_dset_names = ['n_50_v_13.5_D_0.25_Dr_0.14_Drc_0_R_0.383_l_1_a',
                  'n_100_v_13.5_D_0.25_Dr_0.14_Drc_0_R_0.383_l_1_a',
                  'n_200_v_13.5_D_0.25_Dr_0.14_Drc_0_R_0.383_l_1_a',
                  'n_400_v_13.5_D_0.25_Dr_0.14_Drc_0_R_0.383_l_1_a',
                  'n_800_v_13.5_D_0.25_Dr_0.14_Drc_0_R_0.383_l_1_a']

high_dset_names = ['n_50_v_13.5_D_0.25_Dr_0.14_Drc_1_R_0.383_l_1_a',
                   'n_100_v_13.5_D_0.25_Dr_0.14_Drc_1_R_0.383_l_1_a',
                   'n_200_v_13.5_D_0.25_Dr_0.14_Drc_1_R_0.383_l_1_a',
                   'n_400_v_13.5_D_0.25_Dr_0.14_Drc_1_R_0.383_l_1_a',
                   'n_800_v_13.5_D_0.25_Dr_0.14_Drc_1_R_0.383_l_1_a']

low_dset_names = low_dset_names
high_dset_names = high_dset_names

data_dir = '/Users/ejm/Projects/Droplet/Data'
dset_dir = '{}/Simulation/2014-06-09/Runs'.format(data_dir)
low_dsets = get_dsets(dset_dir, low_dset_names)
high_dsets = get_dsets(dset_dir, high_dset_names)

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_low = fig.add_subplot(gridspec[0])
ax_high = fig.add_subplot(gridspec[1], sharex=ax_low, sharey=ax_low)

ejm_rcparams.prettify_axes(ax_low, ax_high)
plot_dsets(ax_low, low_dsets, use_latex)
plot_dsets(ax_high, high_dsets, use_latex)
ax_low.legend(loc='upper left', fontsize=26)

ax_low.set_yscale('log')
ax_high.set_yscale('log')
ax_low.set_ylim(1e-2, 10.0)
ax_low.set_xlim(0.0, 1.19)
ax_low.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=5.0)
ax_low.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
# ax_low.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_low.tick_params(axis='both', labelsize=24, pad=10.0)
ax_high.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_high.get_yticklabels(), visible=False)
gridspec.update(wspace=0.0)

# plt.show()
plt.savefig('Fig S6 Log.pdf', bbox_inches='tight')
