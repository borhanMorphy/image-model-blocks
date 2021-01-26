import torch
import torch.nn as nn
from typing import Tuple

from ..utils import get_same_padding,same_padding_conv

class InceptionV1(nn.Module):

    def __init__(self, infeatures:int, f_1x1:int, f_3x3:Tuple[int,int],
            f_5x5:Tuple[int,int], f_mp3x3:int):
        super().__init__()

        self.branch_1x1 = nn.Sequential(
            same_padding_conv(1, infeatures, f_1x1),
            nn.ReLU()
        )
        self.branch_3x3 = nn.Sequential(
            same_padding_conv(1, infeatures, f_3x3[0]),
            nn.ReLU(),
            same_padding_conv(3, f_3x3[0], f_3x3[1]),
            nn.ReLU()
        )
        self.branch_5x5 = nn.Sequential(
            same_padding_conv(1, infeatures, f_5x5[0]),
            nn.ReLU(),
            same_padding_conv(5, f_5x5[0], f_5x5[1]),
            nn.ReLU()
        )
        self.branch_mp3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1,
                padding=get_same_padding(3), ceil_mode=True),
            same_padding_conv(1, infeatures, f_mp3x3),
            nn.ReLU()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """[summary] # TODO

        Args:
            x (torch.Tensor): (B x C x H X W)

        Returns:
            torch.Tensor: (B x [f_1x1 + f_3x3[1] + f_5x5[1] + f_mp3x3] x H x W)
        """
        # (B x C x H x W) => (B x f_1x1 x H x W)
        f1 = self.branch_1x1(x)

        # (B x C x H x W) => (B x f_3x3[1] x H x W)
        f2 = self.branch_3x3(x)

        # (B x C x H x W) => (B x f_5x5[1] x H x W)
        f3 = self.branch_5x5(x)

        # (B x C x H x W) => (B x f_mp3x3 x H x W)
        f4 = self.branch_mp3x3(x)

        # depthwise concatination
        # B x [f_1x1 + f_3x3[1] + f_5x5[1] + f_mp3x3] x H x W => (B x (f_1x1 + f_3x3[1] + f_5x5[1] + f_mp3x3) x H x W)
        return torch.cat([f1,f2,f3,f4], dim=1)