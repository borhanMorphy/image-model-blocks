import torch.nn as nn

get_same_padding = lambda kernel_size: (kernel_size - 1) // 2

def same_padding_conv(kernel_size:int, infeatures:int, outfeatures:int, bias:bool=False) -> nn.Module:
    # conv block with same spatial input/output size
    stride = 1
    padding = get_same_padding(kernel_size)
    # TODO checkout bias
    return nn.Conv2d(infeatures, outfeatures, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias)