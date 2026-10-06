"""Stub for `from pytorch_lightning import seed_everything`.

Reproduces pytorch_lightning.seed_everything(seed, workers=False) exactly -- the
same generators, seeded in the same order -- so a run of the unmodified
_ref_LRURec source draws the identical random stream it would with the real
package installed.  Only used when pytorch_lightning is not installed.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed=None, workers=False):
    seed = 0 if seed is None else int(seed)
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PL_SEED_WORKERS'] = f'{int(workers)}'
    return seed
