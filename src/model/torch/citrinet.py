import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_SIZES = {
    'K1': [5, 3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, 9, 41],
    'K2': [5, 5, 7, 7, 9, 9, 11, 7, 7, 9, 9, 11, 11, 13, 13, 13, 15, 15, 17, 17, 19, 19, 41],
    'K3': [5, 9, 9, 11, 13, 15, 15, 9, 11, 13, 15, 15, 17, 19, 19, 21, 21, 23, 25, 27, 27, 29, 41],
    'K4': [5, 11, 13, 15, 17, 19, 21, 13, 15, 17, 19, 21, 23, 25, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}

def apply_mask(x, lengths):
    """Обнуляет всё, что выходит за пределы реальной длины аудио"""
    max_len = x.size(2)
    mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) < lengths.unsqueeze(1)
    return x * mask.unsqueeze(1).float()


class MaskedSequential(nn.Sequential):
    """
    Ведет себя как nn.Sequential, 
    но умеет передавать lengths между Jasper-блоками.
    """
    def forward(self, x, lengths):
        for module in self:
            x, lengths = module(x, lengths)
        return x, lengths


class CitriNet(nn.Module):
    def __init__(
        self, K: int,
        C: int,
        R_repeat: int,
        Gamma: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
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
            nn.Dropout1d(p=dropout),
        )

        self.megablock1 = MaskedSequential(*[
            JasperBlock(
                in_channels=self.C,
                repeats=self.R_repeat,
                k=self.K[i + 1],
                stride=2 if i == 0 else 1,
                gamma=Gamma,
            )
            for i in range(6) # 6 block in first megablock from paper
        ])

        self.megablock2 = MaskedSequential(*[
            JasperBlock(
                in_channels=self.C,
                repeats=self.R_repeat,
                k=self.K[i + 7],
                stride=2 if i == 0 else 1,
                gamma=Gamma,
            )
            for i in range(7) # 7 block in second megablock
        ])

        self.megablock3 = MaskedSequential(*[
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
            nn.Dropout1d(p=dropout),
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(
                640,
                out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x, lengths):
        x = self.C1(x)
        x = apply_mask(x, lengths)
        
        x, lengths = self.megablock1(x, lengths)
        x, lengths = self.megablock2(x, lengths)
        x, lengths = self.megablock3(x, lengths)

        x = self.C2(x)
        x = apply_mask(x, lengths)
        
        x = self.C3(x)
        x = apply_mask(x, lengths)
        return x


class JasperBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        repeats: int,
        k: int,
        stride: int,
        gamma: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.repeats = repeats
        self.stride = stride

        blocks = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                k=k,
                padding=k // 2,
                dropout=dropout,
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
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        out = self.blocks(x) + self.residual(x)
        out = self.relu(out)
        out = self.dropout(out)

        if self.stride > 1:
            lengths = (lengths + 1) // self.stride

        out = apply_mask(out, lengths)

        return out, lengths


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, padding, dropout):
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
            nn.Dropout1d(p=dropout),
        )

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
