import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(x, lengths):
    """Обнуляет всё, что выходит за пределы реальной длины аудио"""
    max_len = x.size(2)
    mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) < lengths.unsqueeze(1)
    return x * mask.unsqueeze(1).float()


def conv_out_length(
    lengths: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    return (
        torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor")
        + 1
    )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=channels,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_mels: int = 80,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.stride = 2
        self.padding = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.d_model,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )
        self.dw_block_1 = DepthwiseSeparableConv2d(
            d_model, self.kernel_size, self.stride,
        )
        self.dw_block_2 = DepthwiseSeparableConv2d(
            d_model, self.kernel_size, self.stride,
        )

        freq_out = n_mels
        for _ in range(3):
            freq_out = self.evaluate_conv_out_dim(freq_out, self.kernel_size, self.stride, self.padding)
        self.out_proj = nn.Linear(self.d_model * freq_out, self.d_model)

    @staticmethod
    def evaluate_conv_out_dim(in_dim: int, kernel_size: int, stride: int, padding: int) -> int:
        return int((in_dim + 2 * padding - kernel_size) / stride) + 1

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        x = features.unsqueeze(1)  # (B, 1, T, n_mels)

        x = self.conv_block(x)
        lengths = conv_out_length(lengths, self.kernel_size, self.stride, self.padding)
        x = apply_mask(x, lengths)

        x = self.dw_block_1(x)
        lengths = conv_out_length(lengths, self.kernel_size, self.stride, self.padding)
        x = apply_mask(x, lengths)

        x = self.dw_block_2(x)  # (B, d_model, T // 8, n_mels // 8)
        lengths = conv_out_length(lengths, self.kernel_size, self.stride, self.padding)
        x = apply_mask(x, lengths)

        batch_size, n_channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time, n_channels * freq)
        x = self.out_proj(x)  # (B, T // 8, d_model)
        return x, lengths


class RelPositionalEncoding(nn.Module):
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

        # Генерируем PE один раз при инициализации на CPU.
        # register_buffer с persistent=False гарантирует, что:
        # 1. Он автоматически уедет на GPU вместе с моделью.
        # 2. Он НЕ будет сохраняться в state_dict (не будет раздувать чекпоинты).
        pe = self._generate_pe()
        self.register_buffer('pe', pe, persistent=False)

    def _generate_pe(self) -> torch.Tensor:
        # Индексы от (max_len - 1) до -(max_len - 1)
        positions = torch.arange(
            self.max_len - 1, -self.max_len, -1, dtype=torch.float32
        ).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(self.INF_VAL) / self.d_model)
        )
        
        pe = torch.zeros(positions.size(0), self.d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        return pe.unsqueeze(0) # (1, 2*max_len-1, d_model)

    def forward(self, inputs: torch.Tensor, cache_len: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if self.xscale is not None:
            inputs = inputs * self.xscale

        # Вырезаем нужный кусок позиционного кодирования под текущую длину батча
        input_len = inputs.size(1) + cache_len
        
        center_pos = self.max_len - 1
        start_pos = center_pos - input_len + 1
        end_pos = center_pos + input_len
        
        pos_emb = self.pe[:, start_pos:end_pos]

        return self.dropout(inputs), pos_emb


class BiasNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model)."""
        rms = (x - self.bias).pow(2).mean(dim=-1, keepdim=True).clamp(min=self.eps).sqrt()
        return x / rms * self.log_scale.exp()



def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, 2T-1) -> (B, H, T, T)"""
    b, h, t, t2_minus_1 = x.size()
    # Добавляем колонку нулей
    zero_pad = torch.zeros((b, h, t, 1), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1) # (B, H, T, 2T)
    # Решейпим и сдвигаем
    x_padded = x_padded.view(b, h, t2_minus_1 + 1, t) # (B, H, 2T, T)
    # Берем нужный кусок
    return x_padded[:, :, 1:t+1, :] # (B, H, T, T)


class SwooshL(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 4.0) - 0.08 * x - 0.035


class SwooshR(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1.0) - 0.08 * x - 0.313


class AttentionWeights(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_head_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attn_head_dim = attn_head_dim
        attn_dim = n_heads * attn_head_dim

        self.norm = BiasNorm(d_model)
        self.linear_q = nn.Linear(d_model, attn_dim)
        self.linear_k = nn.Linear(d_model, attn_dim)
        self.linear_pos = nn.Linear(d_model, attn_dim, bias=False)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, attn_head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, attn_head_dim))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, lengths: torch.Tensor):
        b, t, _ = x.shape
        x = self.norm(x)

        q = self.linear_q(x).view(b, t, self.n_heads, self.attn_head_dim).transpose(1, 2)
        k = self.linear_k(x).view(b, t, self.n_heads, self.attn_head_dim).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.n_heads, self.attn_head_dim).transpose(1, 2)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = rel_shift(torch.matmul(q_with_bias_v, p.transpose(-2, -1)))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.attn_head_dim)

        # Конвертируем lengths в маску для Attention (-inf для паддинга)
        arange = torch.arange(t, device=x.device).unsqueeze(0)
        padding_mask = arange >= lengths.unsqueeze(1) # (B, T)
        
        mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        scores = scores.masked_fill(mask, float("-inf"))

        return self.dropout(torch.softmax(scores, dim=-1))


class NonLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm = BiasNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden_dim * 3)  # values, gate_a, gate_b
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model), attn_weights: (B, H, T, T)."""
        b, t, _ = x.shape
        x_norm = self.norm(x)
        values, gate_a, gate_b = self.in_proj(x_norm).chunk(3, dim=-1)
        gated = torch.tanh(gate_a) * torch.sigmoid(gate_b) * values  # (B, T, hidden_dim)

        gated = gated.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        out = torch.matmul(attn_weights, gated)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        
        out = self.dropout(self.out_proj(out))
        # Защита от паддинга
        return apply_mask(out, lengths)


class SharedWeightMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = BiasNorm(d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model), attn_weights: (B, H, T, T)."""
        b, t, _ = x.shape
        x_norm = self.norm(x)
        v = self.linear_v(x_norm).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        
        out = self.dropout(self.linear_out(out))
        # Защита от паддинга
        return apply_mask(out, lengths)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.module = nn.Sequential(
            BiasNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            SwooshR(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        # Защищаем от мусора в паддингах
        return apply_mask(out, lengths)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric 'same' padding"
        self.norm = BiasNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model,
        )
        self.mid_norm = BiasNorm(d_model)
        self.swoosh_l = SwooshL()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = self.norm(x)
        
        # Создаем маску для сверток (где True - это полезный сигнал, False - паддинг)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) < lengths.unsqueeze(1)
        mask_1d = mask.unsqueeze(1).float() # (B, 1, T)

        x = x.transpose(1, 2)  # (B, d_model, T)

        # Маскируем перед первой сверткой
        x = x * mask_1d

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Маскируем перед depthwise сверткой (убиваем bias от pw_conv)
        x = x * mask_1d

        x = self.depthwise_conv(x)
        
        # BiasNorm ожидает (B, T, D), поэтому транспонируем туда-обратно
        x = self.mid_norm(x.transpose(1, 2)).transpose(1, 2)
        
        x = self.swoosh_l(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        out = x.transpose(1, 2)  # (B, T, d_model)
        # Финальная маска на выходе
        return apply_mask(out, lengths)


class BypassModule(nn.Module):
    def __init__(self, d_model: int, min_scale: float = 0.4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.min_scale = min_scale

    def forward(self, x: torch.Tensor, block_out: torch.Tensor) -> torch.Tensor:
        scale = self.scale.clamp(min=self.min_scale, max=1.0)
        return x + scale * (block_out - x)


class ZipformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_head_dim: int = 24,
        nla_hidden_dim: int = 192,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        bypass_min_scale: float = 0.4,
        initial_scale: float = 0.05,
    ):
        super().__init__()
        self.attn_weights = AttentionWeights(d_model, n_heads, attn_head_dim, dropout)
        self.nla = NonLinearAttention(d_model, n_heads, nla_hidden_dim, dropout)
        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.self_attn1 = SharedWeightMHSA(d_model, n_heads, dropout)
        self.conv1 = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.self_attn2 = SharedWeightMHSA(d_model, n_heads, dropout)
        self.conv2 = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.final_norm = BiasNorm(d_model)
        self.bypass = BypassModule(d_model, bypass_min_scale)

        # Тот самый гениальный хак для стабилизации градиентов
        for module, name in (
            (self.nla, "out_proj"), (self.self_attn1, "linear_out"),
            (self.self_attn2, "linear_out"), (self.ff1, "linear2"),
            (self.ff2, "linear2"), (self.conv1, "pointwise_conv2"),
            (self.conv2, "pointwise_conv2"),
        ):
            layer = getattr(module, name)
            with torch.no_grad():
                layer.weight.mul_(initial_scale)
                if layer.bias is not None:
                    layer.bias.mul_(initial_scale)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, lengths: torch.Tensor):
        residual = x
        
        # 1. Считаем матрицу внимания ОДИН РАЗ
        attn_weights = self.attn_weights(x, pos_emb, lengths)

        # 2. Прогоняем через сэндвич, передавая lengths в каждый блок
        x = x + self.nla(x, attn_weights, lengths)
        x = x + self.ff1(x, lengths)
        x = x + self.self_attn1(x, attn_weights, lengths)
        x = x + self.conv1(x, lengths)
        x = x + self.ff2(x, lengths)
        x = x + self.self_attn2(x, attn_weights, lengths)
        x = x + self.conv2(x, lengths)
        
        x = self.final_norm(x)

        # 3. Умный Bypass вместо обычного residual
        return self.bypass(residual, x)


class ZipformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256, # Для 30-40 млн параметров
        n_layers: int = 12, # 12-16 слоев
        n_heads: int = 4,
        attn_head_dim: int = 24,
        nla_hidden_dim: int = 192,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        bypass_min_scale: float = 0.4,
        initial_scale: float = 0.05,
    ):
        super().__init__()
        # Сжатие времени в 8 раз
        self.subsampling = ConvSubsampling(d_model, n_mels)
        self.pos_encoding = RelPositionalEncoding(d_model)
        
        self.blocks = nn.ModuleList([
            ZipformerBlock(
                d_model, n_heads, attn_head_dim, nla_hidden_dim,
                conv_kernel_size, ff_expansion_factor, dropout, 
                bypass_min_scale, initial_scale
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = BiasNorm(d_model)

    def forward(self, melspecs: torch.Tensor, lengths: torch.Tensor):
        """
        melspecs: (B, n_mels, T_frames) - приходит из DataLoader
        lengths: (B,)
        """
        # ConvSubsampling ожидает (B, T_frames, n_mels)
        x = melspecs.transpose(1, 2)
        
        # Сабсэмплинг (x станет (B, T//8, d_model))
        x, lengths = self.subsampling(x, lengths)
        
        # Позиционное кодирование
        x, pos_emb = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, pos_emb, lengths)
            
        x = self.final_norm(x)
        return x, lengths


class ZipformerCTC(nn.Module):
    def __init__(self, vocab_size: int, **encoder_kwargs):
        super().__init__()
        self.encoder = ZipformerEncoder(**encoder_kwargs)
        # Классическая CTC голова - просто линейная проекция в размер словаря
        self.ctc_head = nn.Linear(encoder_kwargs.get('d_model', 256), vocab_size)

    def forward(self, melspecs: torch.Tensor, lengths: torch.Tensor):
        # 1. Получаем акустические фичи
        x, out_lengths = self.encoder(melspecs, lengths)
        
        # 2. Проецируем в токены
        logits = self.ctc_head(x) # (B, T//8, vocab_size)
        
        # 3. Защита от паддингов
        logits = apply_mask(logits, out_lengths)
        
        # Возвращаем в формате, который ожидает ASRLightning
        # logits: (B, T, Vocab), out_lengths: (B,)
        return logits, out_lengths