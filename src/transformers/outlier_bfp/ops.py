import torch

from ..bfp.bfp_ops import float_to_bfp_blocked


class OutlierBFP:
    def __init__(self, bfp, outliers, outlier_indices, original_shape, **bfp_args):
        self.bfp = bfp
        self.outliers = outliers
        self.outlier_indices = outlier_indices
        self.original_shape = original_shape
        self.bfp_args = bfp_args

    def matmul(self, t):
        # We first convert t
        column_fp, column_bfp = extract_rows(t, self.outlier_indices)
        column_bfp = float_to_bfp_blocked(column_bfp, **self.bfp_args)

        # Multiply both parts and add them back together
        return torch.matmul(self.bfp, column_bfp) + torch.matmul(self.outliers, column_fp)


def convert(t, **bfp_args):
    """
    Converts a tensor to its outlier BFP form.
    Takes the tensor to convert and the bfp_args used for conversion to bfp as arguments
    Returns an OutlierBFP object
    """
    normal, outliers, outlier_indices = split_outliers(t)
    bfp = float_to_bfp_blocked(normal, **bfp_args)
    return OutlierBFP(bfp, outliers, outlier_indices, t.size, **bfp_args)


def split_outliers(t):
    # Find max along columns
    abs_max = torch.max(torch.abs(t), dim=0).values

    outlier_indices = torch.flatten(torch.argwhere(abs_max > 6))

    # Separate outliers from matrix
    outliers, normal = split_on_indices(t, outlier_indices, 1)

    return normal, outliers, outlier_indices


def extract_rows(t, indices):
    return split_on_indices(t, indices, 0)


def split_on_indices(t, indices, dim):
    rest_indices = other_indices(indices, t.size(dim))

    extracted = torch.index_select(t, dim, indices)
    rest = torch.index_select(t, dim, rest_indices)

    return extracted, rest


def other_indices(indices, total_size):
    device = None if indices.get_device() == -1 else indices.get_device()
    bool_tensor = torch.zeros(total_size, dtype=torch.bool, device = device)
    bool_tensor[indices] = True
    return torch.flatten(torch.argwhere(~bool_tensor))
