from typing import List
import numpy as np

def global_loss(loss: List, n_sites: int, target_is_normalized: bool = False) -> float:
    if target_is_normalized:
        hst = 1 - 1/2**(n_sites) * np.abs(loss)**2
    else:
        hst = 1 - 1/2**(2*n_sites) * np.abs(loss)**2
    return hst