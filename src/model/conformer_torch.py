import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionSubsamplingLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
              ),
            nn.ReLU(),
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs.unsqueeze(1))
        batch_size, channels, seq_len, emb_dim = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, channels * emb_dim)
        return output


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        expansion_factor: int,
        residual_factor: float,
        dropout: float,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.expansion_factor = expansion_factor
        self.residual_factor = residual_factor
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-05, elementwise_affine=True),
            nn.Linear(in_dim, in_dim * expansion_factor, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_dim * expansion_factor, in_dim, bias=True),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.residual_factor * self.model(inputs)


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        encoder_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.encoder_dim = encoder_dim

        self.register_buffer("m", self.get_alibi_slope(self.num_heads))
        self.query = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.key = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.value = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.projection = nn.Linear(encoder_dim, encoder_dim, bias=True)

    @staticmethod
    def get_relative_positions(seq_len: int) -> torch.tensor:
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return x - y

    @staticmethod
    def get_alibi_slope(num_heads):
        x = (2 ** 8) ** (1 / num_heads)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = inputs.shape
        device = inputs.device

        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        # q/k/v = [batch_size, num_heads, seq_len, d_head]
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # bias = [1, num_heads, seq_len, seq_len]
        bias = (self.m * self.get_relative_positions(seq_len)).unsqueeze(0).to(device)

        attn = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=bias
        ).transpose(1, 2).reshape(batch_size, seq_len, -1)

        out = self.projection(attn)

        return out


class MultiHeadAttentionModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            ALiBiMultiHeadAttention(
                num_heads=num_heads,
                encoder_dim=encoder_dim,
            ),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.model(inputs)


class Transpose(nn.Module):
    def __init__(
        self,
        dim_0: int,
        dim_1: int,
    ):
        super().__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.transpose(self.dim_0, self.dim_1)


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        self.model = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Transpose(1, 2),
            nn.Conv1d(encoder_dim, encoder_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.GLU(dim=1),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=31, stride=1, padding='same', groups=encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout, inplace=True),
            Transpose(1, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.model(inputs)
    

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_mels: int,
        encoder_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.conv_subsampler = ConvolutionSubsamplingLayer(
            in_dim,
            encoder_dim,
            kernel_size=3,
            stride=2,
            padding=1,
          )
        subsampling_dim = self.evaluate_conv_out_dim(n_mels, 3, 2, 1)
        subsampling_dim = self.evaluate_conv_out_dim(subsampling_dim, 3, 2, 1)

        self.projection = ProjectionLayer(
            subsampling_dim * encoder_dim,
            encoder_dim,
            dropout,
        )

    @staticmethod
    def evaluate_conv_out_dim(in_dim: int, kernel_size: int, stride: int, padding: int) -> int:
        return int((in_dim + 2 * padding - kernel_size) / stride) + 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv_subsampler(inputs)
        output = self.projection(output)
        return output


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads

        self.model = nn.Sequential(
            FeedForwardModule(
                in_dim=encoder_dim,
                expansion_factor=4,
                residual_factor=0.5,
                dropout=0.1,
            ),
            MultiHeadAttentionModule(
                encoder_dim=encoder_dim,
                num_heads=num_heads,
                dropout=0.1,
            ),
            ConvolutionModule(
                encoder_dim=encoder_dim,
                dropout=0.1,
            ),
            FeedForwardModule(
                in_dim=encoder_dim,
                expansion_factor=4,
                residual_factor=0.5,
                dropout=0.1,
            ),
            nn.LayerNorm(encoder_dim),
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class Conformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_mels: int,
        encoder_dim: int,
        num_blocks: int,
        num_heads: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(in_dim, n_mels, encoder_dim, dropout)
        self.conf_blocks = nn.ModuleList([
            ConformerBlock(encoder_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.decoder = nn.Linear(encoder_dim, out_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # print(inputs.shape)
        output = self.encoder(inputs.transpose(1, 2))
        # print(output.shape)

        for block in self.conf_blocks:
            output = block(output)
            # print(output.shape)

        return self.decoder(output).transpose(1, 2)
