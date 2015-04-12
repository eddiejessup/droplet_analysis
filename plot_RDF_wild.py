import matplotlib.pyplot as plt
from dataset import Dataset
import glob
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


res = 0.7

smooth_dset_names = ['D36']
wild_dset_names = ['D31wt']

data_dir = '/Users/ejm/Projects/Droplet/Data'
smooth_dset_dir = '{}/Experiment/smooth/Runs'.format(data_dir)
wild_dset_dir = '{}/Experiment/wild_type/Runs'.format(data_dir)
smooth_dsets = get_dsets(smooth_dset_dir, smooth_dset_names)
wild_dsets = get_dsets(wild_dset_dir, wild_dset_names)
dsets = smooth_dsets + wild_dsets
smooths = [True, False]

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)

ejm_rcparams.prettify_axes(ax)

for dset, smooth in zip(dsets, smooths):
    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(res)
    vp, vp_err = dset.get_vp()
    R = dset.R
    if use_latex:
        label = (r'$\SI{' + '{:.3g}'.format(R) + r'}{\um}$, ' +
                 r'$\SI{' + '{:.2g}'.format(vp) + r'}{\percent}$')
    else:
        label = (r'$' + '{:.3g}'.format(R) + r'\mu m$, $' +
                 '{:.2g}'.format(vp) + r'\%$')
    if smooth:
        label += r', Smooth swimming'
    else:
        label += r', Wild type'

    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                label=label)
ax.legend(loc='upper left', fontsize=26)

ax.set_ylim(0.0, 2.2)
ax.set_xlim(0.0, 1.19)
ax.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35)
ax.set_xlabel(r'$r / R$', fontsize=35, labelpad=10.0)
# ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=24, pad=10.0)

# plt.show()
plt.savefig('Fig 4 Wild.pdf', bbox_inches='tight')
