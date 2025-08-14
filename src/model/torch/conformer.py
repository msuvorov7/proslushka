import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention import SDPBackend


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
            # nn.Dropout(dropout, inplace=True),
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
            # nn.Dropout(dropout, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.residual_factor * self.model(inputs)


# class ALiBiMultiHeadAttention(nn.Module):
#     def __init__(
#         self,
#         num_heads: int,
#         encoder_dim: int,
#     ) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         self.encoder_dim = encoder_dim

#         self.register_buffer("m", self.get_alibi_slope(self.num_heads))
#         self.query = nn.Linear(encoder_dim, encoder_dim, bias=True)
#         self.key = nn.Linear(encoder_dim, encoder_dim, bias=True)
#         self.value = nn.Linear(encoder_dim, encoder_dim, bias=True)
#         self.projection = nn.Linear(encoder_dim, encoder_dim, bias=True)

#     @staticmethod
#     def get_relative_positions(seq_len: int) -> torch.tensor:
#         x = torch.arange(seq_len)[None, :]
#         y = torch.arange(seq_len)[:, None]
#         return x - y

#     @staticmethod
#     def get_alibi_slope(num_heads):
#         x = (2 ** 8) ** (1 / num_heads)
#         return (
#             torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
#             .unsqueeze(-1)
#             .unsqueeze(-1)
#         )

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, emb_dim = inputs.shape
#         device = inputs.device

#         query = self.query(inputs)
#         key = self.key(inputs)
#         value = self.value(inputs)

#         # q/k/v = [batch_size, num_heads, seq_len, d_head]
#         query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         key = key.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
#         value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

#         # bias = [1, num_heads, seq_len, seq_len]
#         bias = (self.m.to(device) * self.get_relative_positions(seq_len).to(device)).unsqueeze(0)

#         with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
#             attn = F.scaled_dot_product_attention(
#                 query, key, value, dropout_p=0.0, is_causal=False, attn_mask=bias
#             ).transpose(1, 2).reshape(batch_size, seq_len, -1)

#         out = self.projection(attn)

#         return out
    

class PositionalEncoder(nn.Module):
    INF_VAL: float = 10000.0

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        xscale: float = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout)

        # self.extend_pe('cpu', torch.float32)

    def create_pe(self, positions: torch.Tensor, dtype):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(positions.size(0), self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(self.INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, device, dtype):
        length = self.max_len
        # device = next(self.parameters()).device
        # dtype = next(self.parameters()).dtype
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, inputs: torch.Tensor, cache_len: int = 0) -> torch.Tensor:
        if self.xscale:
            inputs = inputs * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = inputs.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        return self.dropout(inputs), pos_emb


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
    ):
        super().__init__()
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        nn.init.zeros_(self.pos_bias_u)
        nn.init.zeros_(self.pos_bias_v)

    def forward_qkv(self, query, key, value):
        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # q/k/v = [batch_size, num_heads, seq_len, d_k]
        return q, k, v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, pos_emb):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # [batch_size, seq_len, num_heads, d_k]

        n_batch_pos = pos_emb.size(0)
        n_batch = value.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # [batch, head, seq_len, d_k]

        # [batch, head, seq_len, d_k]
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # [batch, head, seq_len, d_k]
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
        matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=0
            )
        out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # [batch, time1, d_model]
        out = self.linear_out(out)

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

        self.norm_attn = nn.LayerNorm(encoder_dim)
        self.attn = RelPositionMultiHeadAttention(
            n_head=num_heads,
            n_feat=encoder_dim,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputs: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        residual = inputs
        inputs = self.norm_attn(inputs)
        inputs = self.attn(inputs, inputs, inputs, pos_emb)
        return residual + self.dropout(inputs)


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

        self.pos_encoder = PositionalEncoder(encoder_dim, xscale=math.sqrt(encoder_dim))
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_encoder.extend_pe(device, dtype)

    @staticmethod
    def evaluate_conv_out_dim(in_dim: int, kernel_size: int, stride: int, padding: int) -> int:
        return int((in_dim + 2 * padding - kernel_size) / stride) + 1

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.conv_subsampler(inputs)
        output = self.projection(output)
        output, pos_emb = self.pos_encoder(output)
        return output, pos_emb


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads

        self.feed_forward_1 = FeedForwardModule(
            in_dim=encoder_dim,
            expansion_factor=4,
            residual_factor=0.5,
            dropout=0.1,
        )
        self.attention = MultiHeadAttentionModule(
            num_heads=num_heads,
            encoder_dim=encoder_dim,
            dropout=0.1,
        )
        self.conv = ConvolutionModule(
            encoder_dim=encoder_dim,
            dropout=0.1,
        )
        self.feed_forward_2 = FeedForwardModule(
            in_dim=encoder_dim,
            expansion_factor=4,
            residual_factor=0.5,
            dropout=0.1,
        )
        self.norm_out = nn.LayerNorm(encoder_dim)


    def forward(self, inputs: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        inputs = self.feed_forward_1(inputs)
        inputs = self.attention(inputs, pos_emb=pos_emb)
        inputs = self.conv(inputs)
        inputs = self.feed_forward_2(inputs)
        inputs = self.norm_out(inputs)
        return inputs


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
        output, pos_emb = self.encoder(inputs.transpose(1, 2))
        # print(output.shape)

        for block in self.conf_blocks:
            output = block(output, pos_emb)
            # print(output.shape)

        return self.decoder(output).transpose(1, 2)
