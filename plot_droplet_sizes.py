import matplotlib.pyplot as plt
import ejm_rcparams
import numpy as np

use_latex = True
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

dat_fname = ('/Users/ejm/Projects/Droplet/'
             'Experimental Data/Droplet Properties/Drop Size.csv')

R, n = np.loadtxt(dat_fname, delimiter=',', unpack=True)
res = np.diff(R)[0]

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)

ejm_rcparams.prettify_axes(ax)

ax.bar(R - res, n, width=res,
       color=ejm_rcparams.set2[0], ec=ejm_rcparams.almost_black)

ax.set_ylim(0.0, 32)
# ax.set_xlim(0.0, 1.19)
ax.set_ylabel(r'$N$', fontsize=35)
if use_latex:
    xlabel = r'$R / \si{\um}$'
else:
    xlabel = r'$R / \mu$m'
ax.set_xlabel(xlabel, fontsize=35, labelpad=10.0)
ax.tick_params(axis='both', labelsize=24)

# plt.show()
plt.savefig('Fig S1 Droplet sizes.pdf', bbox_inches='tight')
