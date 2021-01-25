import torch
import torch.nn as nn
from ..utils import inception_conv,calculate_same_padding

class Inception3A(nn.Module):
    def __init__(self, infeatures:int):
        super().__init__()

        self.branch_1x1 = nn.Sequential(
            inception_conv(1, infeatures, 64),
            nn.ReLU()
        )
        self.branch_3x3 = nn.Sequential(
            inception_conv(1, infeatures, 96),
            nn.ReLU(),
            inception_conv(3, 96, 128),
            nn.ReLU()
        )
        self.branch_5x5 = nn.Sequential(
            inception_conv(1, infeatures, 16),
            nn.ReLU(),
            inception_conv(3, 16, 32),
            nn.ReLU()
        )
        self.branch_mp3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1,
                padding=calculate_same_padding(3), ceil_mode=True),
            inception_conv(1, infeatures, 32),
            nn.ReLU()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """[summary] # TODO

        Args:
            x (torch.Tensor): (B x C x H X W)

        Returns:
            torch.Tensor: (B x 256 x H x W)
        """
        # (B x C x H x W) => (B x 64 x H x W)
        f1 = self.branch_1x1(x)

        # (B x C x H x W) => (B x 128 x H x W)
        f2 = self.branch_3x3(x)

        # (B x C x H x W) => (B x 32 x H x W)
        f3 = self.branch_5x5(x)

        # (B x C x H x W) => (B x 32 x H x W)
        f4 = self.branch_mp3x3(x)

        # depthwise concatination
        # B x [64 + 128 + 32 + 32] x H x W => (B x 256 x H x W)
        return torch.cat([f1,f2,f3,f4], dim=1)