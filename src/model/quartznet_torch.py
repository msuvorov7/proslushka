import torch.nn as nn


class QuartzNet(nn.Module):
    def __init__(
        self,
        R_repeat: int,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        """
        QuartzNet model from https://arxiv.org/pdf/1910.10261
        :param R_repeat: repeat R_repeat times JasperBlocks: 0 - QuartzNet5x5, 1 - QuartzNet10x5, 2 - QuartzNet15x5
        :param in_channels: input dim of mel spectrogram (n_mels)
        :param out_channels: output dim (vocab size plus blank token)
        :param dropout: dropout probability
        """
        super(QuartzNet, self).__init__()

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

            print(i, num_in, num_out, pad, k)

            self.B.append(JasperBlock(num_in, num_out, k, pad))

            for rep in range(R_repeat):
                self.B.append(JasperBlock(num_out, num_out, k, pad))

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
    ):
        """
        QuartzNet B-block
        :param in_channels:
        :param out_channels:
        :param k: kernel size
        :param padding: padding
        :param dropout: dropout probability
        """
        super(JasperBlock, self).__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
            ConvBlock(out_channels, out_channels, k, padding),
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
        dropout: float = 0.0
    ):
        super(ConvBlock, self).__init__()

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
                in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False
            ),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)
