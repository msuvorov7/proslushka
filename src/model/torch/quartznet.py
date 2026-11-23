import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(x, lengths):
    """Обнуляет всё, что выходит за пределы реальной длины аудио"""
    max_len = x.size(2)
    mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) < lengths.unsqueeze(1)
    return x.masked_fill(~mask.unsqueeze(1), 0.0)


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

            b_bloks = [JasperBlock(num_in, num_out, k, pad, dropout=dropout, include_se_block=include_se_block), ]
            for rep in range(R_repeat):
                b_bloks.append(JasperBlock(num_out, num_out, k, pad, dropout=dropout, include_se_block=include_se_block))

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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.C1(x)
        lengths = ((lengths + 2 * 16 - 33) // 2) + 1
        x = apply_mask(x, lengths)

        for block in self.B:
            x = block(x)
            x = apply_mask(x, lengths)

        x = self.C2(x)
        x = apply_mask(x, lengths)
        x = self.C3(x)
        x = apply_mask(x, lengths)
        x = self.C4(x)
        x = apply_mask(x, lengths)

        # приводим к стандарту (B, T, Vocab)
        x = x.transpose(1, 2)
        return x, lengths


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
    def __init__(self, in_channels: int, reduction_ratio: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // reduction_ratio,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels // reduction_ratio,
                in_channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)
        y = self.fc(y)
        return x * self.sigmoid(y)
