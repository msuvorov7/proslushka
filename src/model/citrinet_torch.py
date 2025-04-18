import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_SIZES = {
    'K1': [5, 3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, 9, 41],
    'K2': [5, 5, 7, 7, 9, 9, 11, 7, 7, 9, 9, 11, 11, 13, 13, 13, 15, 15, 17, 17, 19, 19, 41],
    'K3': [5, 9, 9, 11, 13, 15, 15, 9, 11, 13, 15, 15, 17, 19, 19, 21, 21, 23, 25, 27, 27, 29, 41],
    'K4': [5, 11, 13, 15, 17, 19, 21, 13, 15, 17, 19, 21, 23, 25, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}


class CitriNet(nn.Module):
    def __init__(
        self, K: int,
        C: int,
        R_repeat,
        Gamma: int,
        in_channels: int,
        out_channels: int,
    ):
        """
        CitriNet model from https://arxiv.org/pdf/2104.01721
        :param K: kernel layout configuration (KERNEL_SIZES)
        :param C: num channels (for example 256, 384, 512, 768, 1024)
        :param R_repeat: repeat R_repeat times JasperBlocks
        :param Gamma: scaling factor for SE block
        :param in_channels: input dim of mel spectrogram (n_mels)
        :param out_channels: output dim (vocab size with blank token)
        """
        super().__init__()
        
        self.K = KERNEL_SIZES[f'K{K}']
        self.C = C
        self.R_repeat = R_repeat
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.C1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=self.K[0],
                padding=self.K[0] // 2,
                stride=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(in_channels, self.C, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.C, eps=0.001, momentum=0.1, affine=True),
            SqueezeExcite(self.C, Gamma),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
        )

        self.megablock1 = nn.Sequential(*[
            JasperBlock(
                in_channels=self.C,
                repeats=self.R_repeat,
                k=self.K[i + 1],
                stride=2 if i == 0 else 1,
                gamma=Gamma,
            )
            for i in range(6) # 6 block in first megablock from paper
        ])

        self.megablock2 = nn.Sequential(*[
            JasperBlock(
                in_channels=self.C,
                repeats=self.R_repeat,
                k=self.K[i + 7],
                stride=2 if i == 0 else 1,
                gamma=Gamma,
            )
            for i in range(7) # 7 block in second megablock
        ])
        
        self.megablock3 = nn.Sequential(*[
            JasperBlock(
                in_channels=self.C,
                repeats=self.R_repeat,
                k=self.K[i + 14],
                stride=2 if i == 0 else 1,
                gamma=Gamma,
            )
            for i in range(8) # 8 block in second megablock
        ])

        self.C2 = nn.Sequential(
            nn.Conv1d(
                self.C,
                self.C,
                kernel_size=self.K[-1],
                padding=self.K[-1] // 2,
                groups=self.C,
                bias=False
            ),
            nn.Conv1d(self.C, 640, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(640, eps=0.001, momentum=0.1, affine=True),
            SqueezeExcite(640, Gamma),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(
                640,
                out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x):
        x = self.C1(x)
        
        x = self.megablock1(x)
        x = self.megablock2(x)
        x = self.megablock3(x)

        x = self.C2(x)
        x = self.C3(x)

        return x
    

class JasperBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        repeats: int,
        k: int,
        stride: int,
        gamma: int,
    ):
        super().__init__()
        self.repeats = repeats

        blocks = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                k=k,
                padding=k // 2,
            ) for i in range(self.repeats)
        ]
        
        blocks += [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=k,
                groups=in_channels,
                padding=k // 2,
                stride=stride,
                bias=False,
            ),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.1, affine=True),
            SqueezeExcite(
                in_channels=in_channels,
                reduction_ratio=gamma,
            ),
        ]
        
        self.blocks = nn.Sequential(*blocks)

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=(1,),
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(in_channels, eps=0.001, momentum=0.1, affine=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(
            self.relu(
                self.blocks(x) + self.residual(x)
            )
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, padding):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=k,
                stride=[1],
                padding=(padding,),
                dilation=[1],
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=(1,),
                stride=(1,),
                bias=False
            ),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
        )

    def forward(self, x):
        return self.layers(x)
    

class SqueezeExcite(nn.Module):
    def __init__(
            self,
            in_channels: int,
            reduction_ratio: int,
            context_window: int = -1,
            interpolation_mode: str = 'nearest',
    ):
        super().__init__()

        self.context_window = context_window
        self.interpolation_mode = interpolation_mode

        self.fc = nn.Sequential(
            nn.Linear(
                in_channels,
                in_channels // reduction_ratio,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_channels // reduction_ratio,
                in_channels,
                bias=False
            ),
        )
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, timestamps = x.size()

        y = self.gap(x)
        y = y.transpose(1, -1)
        y = self.fc(y)
        y = y.transpose(1, -1)

        if self.context_window > 0:
            y = F.interpolate(y, size=timestamps, mode=self.interpolation_mode)

        y = self.sigmoid(y)
        return x * y
