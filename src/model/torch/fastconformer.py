import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention import SDPBackend


def apply_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x: (B, T, D). Безопасное in-place зануление через masked_fill"""
    max_len = x.size(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x.masked_fill(~mask.unsqueeze(-1), 0.0)


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

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
        )

    def _calc_out_length(self, lengths: torch.Tensor) -> torch.Tensor:
        """Пересчитываем длины после свертки"""
        return ((lengths + 2 * self.padding - self.kernel_size) // self.stride) + 1

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(inputs.unsqueeze(1))

        # Пересчитываем длины трижды
        lengths = self._calc_out_length(lengths)
        lengths = self._calc_out_length(lengths)
        lengths = self._calc_out_length(lengths)
        
        batch_size, channels, seq_len, emb_dim = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, channels * emb_dim)
        return output, lengths
    

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
            nn.Dropout(dropout),
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

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = inputs + self.residual_factor * self.model(inputs)
        return apply_mask(out, lengths)


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

        # (1, 2*max_len - 1, D); строка r <-> относительное смещение (max_len - 1 - r)
        pe = self._generate_pe()
        self.register_buffer('pe', pe, persistent=False)

    def _generate_pe(self):
        # Позиции от (max_len - 1) до -(max_len - 1), по убыванию
        positions = torch.arange(self.max_len - 1, -self.max_len, -1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(self.INF_VAL) / self.d_model)
        )
        pe = torch.zeros(positions.size(0), self.d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.xscale:
            inputs = inputs * self.xscale

        input_len = inputs.size(1)
        center = self.max_len - 1
        # Центрируем окно: смещения от -(T-1) до +(T-1)
        pos_emb = self.pe[:, center - (input_len - 1) : center + input_len, :]  # (1, 2T-1, D)
        return self.dropout(inputs), pos_emb
    

class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=True)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    def forward_qkv(self, query, key, value):
        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # q/k/v = [batch_size, num_heads, seq_len, d_k]
        return q, k, v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Безопасный сдвиг через gather (O(T^2))"""
        t = x.size(-2)
        i = torch.arange(t, device=x.device).unsqueeze(1)
        j = torch.arange(t, device=x.device).unsqueeze(0)
        idx = i - j + t - 1
        idx_expanded = idx.expand(x.size(0), x.size(1), t, t)
        return x.gather(-1, idx_expanded)

    def forward(self, query, key, value, pos_emb, padding_mask):
        batch_size = query.size(0)

        q, k, v = self.forward_qkv(query, key, value)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scale_factor = 1 / math.sqrt(self.d_k)
        matrix_bd *= scale_factor

        if padding_mask is not None:
            # padding_mask: (B, T) где True - это паддинг
            mask = padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, T)
            matrix_bd = matrix_bd.masked_fill(mask, float('-inf'))

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=0
            )
            
        out = out.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)
        return self.linear_out(out)


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

    def forward(self, inputs: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        residual = inputs
        inputs = self.norm_attn(inputs)
        inputs = self.attn(inputs, inputs, inputs, pos_emb, padding_mask)
        out = residual + self.dropout(inputs)
        return apply_mask(out, lengths)


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        self.norm = nn.LayerNorm(encoder_dim)
        self.pointwise1 = nn.Conv1d(encoder_dim, encoder_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=9, stride=1, padding='same', groups=encoder_dim)
        self.bn = nn.BatchNorm1d(encoder_dim)
        self.silu = nn.SiLU()
        self.pointwise2 = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs).transpose(1, 2) # (B, C, T)
        x = self.pointwise1(x)
        x = self.glu(x)
        
        # Защита от паддинга перед широкой сверткой
        max_len = x.size(2)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        
        out = inputs + x.transpose(1, 2)
        return apply_mask(out, lengths)


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
        
        # Считаем размерность после двух сверток (stride=2)
        subsampling_dim = self.conv_subsampler._calc_out_length(torch.tensor(n_mels))
        subsampling_dim = self.conv_subsampler._calc_out_length(subsampling_dim)
        subsampling_dim = self.conv_subsampler._calc_out_length(subsampling_dim).item()

        self.projection = ProjectionLayer(subsampling_dim * encoder_dim, encoder_dim, dropout)
        self.pos_encoder = PositionalEncoder(encoder_dim, xscale=math.sqrt(encoder_dim))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        output, lengths = self.conv_subsampler(inputs, lengths)
        output = self.projection(output)
        output = apply_mask(output, lengths) 
        output, pos_emb = self.pos_encoder(output)
        return output, pos_emb, lengths
    

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


    def forward(self, inputs: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        inputs = self.feed_forward_1(inputs, lengths)
        inputs = self.attention(inputs, pos_emb, padding_mask, lengths)
        inputs = self.conv(inputs, lengths)
        inputs = self.feed_forward_2(inputs, lengths)
        inputs = self.norm_out(inputs)
        return apply_mask(inputs, lengths)
    

class FastConformer(nn.Module):
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

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        # Вход: (B, n_mels, T)
        output, pos_emb, out_lengths = self.encoder(inputs, lengths)

        # Создаем инвертированную маску для SDPA (True = это паддинг)
        max_len = output.size(1)
        padding_mask = torch.arange(max_len, device=output.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        for block in self.conf_blocks:
            output = block(output, pos_emb, padding_mask, out_lengths)

        logits = self.decoder(output)
        logits = apply_mask(logits, out_lengths)
        
        return logits, out_lengths
