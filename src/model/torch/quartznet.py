import torch
import torch.nn as nn
import torch.nn.functional as F


class QuartzNet(nn.Module):
    def __init__(
        self,
        R_repeat: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        include_se_block: bool = False,
    ):
        """
        QuartzNet model from https://arxiv.org/pdf/1910.10261
        :param R_repeat: repeat R_repeat times JasperBlocks: 0 - QuartzNet5x5, 1 - QuartzNet10x5, 2 - QuartzNet15x5
        :param in_channels: input dim of mel spectrogram (n_mels)
        :param out_channels: output dim (vocab size plus blank token)
        :param dropout: dropout probability
        :param include_se_block: add SqueezeExcite block to ending JasperBlock
        """
        super().__init__()
        self.R_repeat = R_repeat
        self.in_channels = in_channels
        self.out_channels = out_channels

        block_channels: list = [256, 256, 256, 512, 512, 512]
        block_k: list = [33, 39, 51, 63, 75]

        self.C1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=33,
                padding=16,
                stride=2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(in_channels, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(256, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.B = nn.ModuleList([])

        for i in range(5):
            num_in = block_channels[i]
            num_out = block_channels[i + 1]
            pad = block_k[i] // 2
            k = block_k[i]

            # print(i, num_in, num_out, pad, k)

            b_bloks = [JasperBlock(num_in, num_out, k, pad, include_se_block=include_se_block), ]
            for rep in range(R_repeat):
                b_bloks.append(JasperBlock(num_out, num_out, k, pad, include_se_block=include_se_block))

            self.B.append(nn.Sequential(*b_bloks))

        self.C2 = nn.Sequential(
            nn.Conv1d(
                512, 512, kernel_size=87, padding=86, dilation=2, groups=512, bias=False
            ),
            nn.Conv1d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.C4 = nn.Conv1d(1024, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.C1(x)

        for block in self.B:
            x = block(x)

        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)

        return x


class JasperBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        padding: int,
        dropout: float = 0.0,
        include_se_block: bool = False,
    ):
        """
        QuartzNet B-block
        :param in_channels:
        :param out_channels:
        :param k: kernel size
        :param padding: padding
        :param dropout: dropout probability
        :param include_se_block: add SqueezeExcite block to ending
        """
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels,  out_channels, k, padding, is_last=False),
            ConvBlock(out_channels, out_channels, k, padding, is_last=False),
            ConvBlock(out_channels, out_channels, k, padding, is_last=False),
            ConvBlock(out_channels, out_channels, k, padding, is_last=False),
            ConvBlock(out_channels, out_channels, k, padding, is_last=True, include_se_block=include_se_block),
        )

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=(1,), stride=[1], bias=False
            ),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.residual(x) + self.blocks(x)))


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        padding: int,
        dropout: float = 0.0,
        is_last: bool = False,
        include_se_block: bool = False,
    ):
        super().__init__()

        layers = [
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
                in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False
            ),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]

        if is_last:
            if include_se_block:
                self.layers = nn.Sequential(
                    *(layers[:-2] + [SqueezeExcite(out_channels, 8), ])
                )
            else:
                self.layers = nn.Sequential(*layers[:-2])
        else:
            self.layers = nn.Sequential(*layers)

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
