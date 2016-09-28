from dataset import Superset, pickle_load
import numpy as np
import paths

res = 0.7


# superset = Superset(paths.exp_dset_paths)
# print(superset.fit_to_model('mean', res))
# superset = Superset(paths.exp_dset_paths)
# print(superset.fit_to_model('median', res))

# superset = Superset(paths.exp_reproduction_Dr_0_05_Drc_10_dset_paths)
superset = pickle_load(paths.exp_reproduction_Dr_0_05_Drc_10_pickle_path)
print(superset.fit_to_model('mean', res))
