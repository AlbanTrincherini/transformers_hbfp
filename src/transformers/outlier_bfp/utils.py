import torch

from ..bfp.bfp_util import get_bfp_args
from .ops import split_outliers, convert


def outlier_bfp_matmul(x, w, **bfp_args):
    if x.dim() == 2:
        outlier_bfp = convert(x, **bfp_args)
        return outlier_bfp.matmul(w)
    elif x.dim() == 3:
        device = None if x.get_device() == -1 else x.get_device()
        res_dimensions = (x.size(0), x.size(1), w.size(1))
        result = torch.empty(res_dimensions, device = device)
        for i in range(x.size(0)):
            result[i] = outlier_bfp_matmul(x[i], w, **bfp_args)
        return result
    else:
        raise NotImplementedError(f'Matrix dimensions not supported : x.dim() = ${x.dim()}')
