from dataset import Superset
import numpy as np
import paths

res = 0.7


superset = Superset(paths.exp_dset_paths)
print(superset.fit_to_model('mean', res))
superset = Superset(paths.exp_dset_paths)
print(superset.fit_to_model('median', res))
