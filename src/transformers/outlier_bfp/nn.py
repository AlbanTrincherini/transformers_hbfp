import torch
import torch.nn.functional as F
from .utils import *


class OutlierBFPLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **bfp_args):
        self.bfp_args = bfp_args
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        #print(f'New layer! in : {in_features}, out : {out_features}')

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfp':
            l = outlier_bfp_matmul(input, torch.t(self.weight), **self.bfp_args)
            if self.bias is not None:
                return l + self.bias
            else:
                return l

        else:
            raise NotImplementedError('NumFormat not implemented')
