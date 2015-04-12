import matplotlib.pyplot as plt
from dataset import Dataset
import glob
from ciabatta import ejm_rcparams
import numpy as np

use_latex = True
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)


res = 0.7

dset_name = 'D31'
data_dir = '/Users/ejm/Projects/Droplet/Data'
dset_dir = '{}/Experiment/wholedrop/Runs'.format(data_dir)
run_fnames = glob.glob('{}/{}/dyn/*.npz'.format(dset_dir, dset_name))

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)

ejm_rcparams.prettify_axes(ax)

theta_factors = np.arange(1, 7)

for theta_factor in theta_factors:
    if theta_factor == 1:
        force_hemisphere = False
        theta_max = np.pi / 2.0
    else:
        force_hemisphere = True
        theta_max = np.pi / theta_factor

    dset = Dataset(run_fnames, theta_max=theta_max,
                   force_hemisphere=force_hemisphere)

    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(res)
    vp, vp_err = dset.get_vp()
    R = dset.R
    label = r'$\theta_\mathrm{max} = \pi'
    if theta_factor > 1:
        label += r' / {}'.format(theta_factor)
    label += r'$'

    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                label=label, lw=2)
ax.legend(loc='upper left', fontsize=26, ncol=2)

ax.set_ylim(0.0, 2.0)
ax.set_xlim(0.0, 1.05)
ax.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35)
ax.set_xlabel(r'$r / R$', fontsize=35, labelpad=10.0)
# ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=24, pad=10.0)

ax.tick_params(axis='both', which='both')

# plt.show()
plt.savefig('Fig S3 Theta.pdf', bbox_inches='tight')
