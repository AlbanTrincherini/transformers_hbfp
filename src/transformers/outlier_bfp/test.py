import torch

from ..bfp.bfp_ops import F_matmul_bfp
from ..bfp.bfp_util import get_bfp_args
from .ops import split_outliers
from .utils import outlier_bfp_matmul


def accuracy_test():
    bfp_args = get_bfp_args()
    bfp_matmul = F_matmul_bfp(**bfp_args)

    for i in range(5):
        x = torch.rand((7, 5)) * (6 + i/10.0)  # Multiply by 6.4 to have some outliers
        w = torch.rand((5, 7))
        base = x @ w
        bfp = bfp_matmul(x, w)
        outlier_bfp = outlier_bfp_matmul(x, w, **bfp_args)

        print(matmul_accuracy(base, bfp))
        print(matmul_accuracy(base, outlier_bfp))


def matmul_accuracy(base, res):
    diff = base - res
    relative_error = torch.abs(diff / base)
    return torch.sum(relative_error) * 100


def base_test(x, w, **bfp_args):
    bfp_args = get_bfp_args()
    x = torch.rand((7, 5)) * 6.4  # Multiply by 6.4 to have some outliers
    w = torch.rand((5, 7))
    print("============Outlier separation============")
    normal, outliers, outlier_indices = split_outliers(x)
    print(x)
    print(normal)
    print(outliers)

    print("============Random multiplication============")

    xw = outlier_bfp_matmul(x, w, **bfp_args)
    print("Result ")
    print(xw)
    print("Expected ")
    print(x @ w)
    print("Difference")
    print(xw - (x @ w))


if __name__ == "__main__":
    accuracy_test()
