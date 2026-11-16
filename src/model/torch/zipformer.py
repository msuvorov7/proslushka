import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(x, lengths):
    """Обнуляет всё, что выходит за пределы реальной длины аудио"""
    max_len = x.size(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x * mask.unsqueeze(-1).float()


def conv_out_length(
    lengths: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    return (
        torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor")
        + 1
    )


class ConvNeXt(nn.Module):
    """
    ConvNeXt-блок (https://arxiv.org/pdf/2206.14747.pdf) в интерпретации Zipformer.
    Ставится поверх subsampling-сверток. Размер не меняет: (B, C, T, F) -> (B, C, T, F).
    """

    def __init__(self, channels: int, hidden_ratio: int = 3, kernel_size: int = 7):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        hidden_channels = channels * hidden_ratio

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
        )
        self.pointwise_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=hidden_channels, kernel_size=1
        )
        self.activation = SwooshL()
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=channels, kernel_size=1
        )
        # В оригинале (scaling.py) здесь дополнительно:
        #   - Balancer после pointwise_conv1,
        #   - pointwise_conv2 как ScaledConv2d(initial_scale=0.01),
        #   - out_balancer + out_whiten после residual-сложения.
        # Для воспроизведения динамики обучения k2 их стоит вернуть.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return bypass + x


class Conv2dSubsampling(nn.Module):
    """
    Subsampling из Zipformer (icefall, zipformer2.py).

    (B, T, n_mels) -> (B, (T - 7) // 2, d_model).
    Фактор: /2 по времени, /4 по частоте.
    Требует T >= 7 и n_mels >= 7.
    """

    def __init__(self, d_model: int, n_mels: int = 80, dropout: float = 0.1):
        super().__init__()
        assert n_mels >= 7

        layer1_channels = 8
        layer2_channels = 32
        layer3_channels = 128

        # Паддинги (time, freq). По времени p=0 у всех сверток — длина
        # монотонно уменьшается, что упрощает пересчет lengths.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),
            ),
            SwooshR(),  # в оригинале перед этим ScaleGrad(0.2) + Balancer (важно для amp/fp16)
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            SwooshR(),  # + Balancer
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2),  # (time, freq)
                padding=0,
            ),
            SwooshR(),  # + Balancer
        )
        self.convnext = ConvNeXt(layer3_channels, kernel_size=7)

        # Ширина по частоте после трех сверток (n_mels=80 -> 80 -> 39 -> 19)
        freq_out = n_mels
        freq_out = self.evaluate_conv_out_dim(freq_out, kernel_size=3, stride=1, padding=1)
        freq_out = self.evaluate_conv_out_dim(freq_out, kernel_size=3, stride=2, padding=0)
        freq_out = self.evaluate_conv_out_dim(freq_out, kernel_size=3, stride=2, padding=0)
        self.out_width = freq_out

        self.out_proj = nn.Linear(self.out_width * layer3_channels, d_model)
        self.out_norm = BiasNorm(d_model)
        # В оригинале после out_proj: Whiten(...), затем out_norm, затем Dropout3
        # (общая dropout-маска). Здесь обычный Dropout.
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def evaluate_conv_out_dim(in_dim: int, kernel_size: int, stride: int, padding: int) -> int:
        return (in_dim + 2 * padding - kernel_size) // stride + 1

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        x = features.unsqueeze(1)  # (B, 1, T, n_mels)

        x = self.conv(x)      # (B, 128, T', F'), где T' = (T - 7) // 2, F' = out_width
        x = self.convnext(x)  # residual, размер не меняет

        batch_size, n_channels, time, freq = x.shape
        x = x.transpose(1, 2).reshape(batch_size, time, n_channels * freq)
        x = self.out_proj(x)  # (B, T', d_model)
        x = self.out_norm(x)
        x = self.dropout(x)

        # Длины после сверток: (k=3, s=1, p=0) -> (k=3, s=2, p=0) -> (k=3, s=1, p=0).
        # ConvNeXt (k=7, p=3) длину не меняет. Итого эквивалентно (lengths - 7) // 2.
        lengths = conv_out_length(lengths, kernel_size=3, stride=1, padding=0)
        lengths = conv_out_length(lengths, kernel_size=3, stride=2, padding=0)
        lengths = conv_out_length(lengths, kernel_size=3, stride=1, padding=0)

        # Финальная маска — не из оригинала, а контракт вашего пайплайна
        # ("вход в ZipformerBlock замаскирован по времени").
        return apply_mask(x, lengths), lengths


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


class SwooshL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 4.0) - 0.08 * x - 0.035


class SwooshR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class CompactRelPositionalEncoding(nn.Module):
    """
    Компактное относительное позиционное кодирование из Zipformer2.
    Смещение сжимается логарифмом + atan, затем раскладывается по частотам:
    близкие смещения кодируются различимо, далёкие — грубо.
    Строка k <-> смещение (k - (T - 1)) относительно центра.
    """

    def __init__(
        self,
        pos_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        length_factor: float = 1.0,
    ):
        super().__init__()
        assert pos_dim % 2 == 0
        assert length_factor >= 1.0
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.length_factor = length_factor
        self.dropout = nn.Dropout(dropout)  # в оригинале Dropout2

        pe = self._generate_pe(max_len)  # (2 * max_len - 1, pos_dim)
        self.register_buffer("pe", pe, persistent=False)

    def _generate_pe(self, t: int) -> torch.Tensor:
        x = torch.arange(-(t - 1), t, dtype=torch.float32).unsqueeze(1)  # (2T-1, 1)
        freqs = 1.0 + torch.arange(self.pos_dim // 2, dtype=torch.float32)

        compression_length = self.pos_dim ** 0.5
        x_compressed = compression_length * x.sign() * (
            (x.abs() + compression_length).log() - math.log(compression_length)
        )

        length_scale = self.length_factor * self.pos_dim / (2.0 * math.pi)
        x_atan = (x_compressed / length_scale).atan()

        pe = torch.zeros(2 * t - 1, self.pos_dim)
        pe[:, 0::2] = (x_atan * freqs).cos()
        pe[:, 1::2] = (x_atan * freqs).sin()
        pe[:, -1] = 1.0  # канал-константа: эффективный bias (linear_pos без bias)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — используется только длина. Возвращает (1, 2T-1, pos_dim)."""
        t = x.size(1)
        assert t <= self.max_len, (
            f"последовательность {t} длиннее max_len={self.max_len}; "
            "в оригинале здесь lazy-расширение кэша (extend_pe)"
        )
        center = self.max_len - 1
        pos_emb = self.pe[center - (t - 1) : center + t]  # (2T-1, pos_dim)
        return self.dropout(pos_emb.unsqueeze(0))


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, T, 2T-1) -> (B, H, T, T).
    out[i, j] = x[i, (T-1) + (j - i)] — согласовано с CompactRelPositionalEncoding.
    ВНИМАНИЕ: направление зеркально v1-варианту, где было (T-1) - (j - i).
    """
    t = x.size(-2)
    i = torch.arange(t, device=x.device).unsqueeze(1)  # query
    j = torch.arange(t, device=x.device).unsqueeze(0)  # key
    idx = (t - 1) + (j - i)  # (T, T), значения в [0, 2T-2]
    return x.gather(-1, idx.expand(x.size(0), x.size(1), t, t))
    # Zero-copy вариант из оригинала (для contiguous x):
    # return x.as_strided(
    #     (b, h, t, t),
    #     (x.stride(0), x.stride(1), x.stride(2) - x.stride(3), x.stride(3)),
    #     storage_offset=x.stride(3) * (t - 1),
    # )


class RelPositionMultiheadAttentionWeights(nn.Module):
    """
    Считает общие attention weights с относительным позиционным кодированием.
    Их потребляют SelfAttention и NonlinAttention.

    В zipformer2: один общий in_proj (q, k, p), БЕЗ pos_bias_u/v и БЕЗ нормы,
    масштаб 1/sqrt(d) перенесен в initial_scale.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        # В оригинале — расписание (0.5 -> 0.0 за 4000 шагов).
        self.pos_emb_skip_rate = pos_emb_skip_rate

        in_proj_dim = (2 * query_head_dim + pos_head_dim) * num_heads
        # initial_scale = qhd**-0.25: q и k в произведении дают qhd**-0.5.
        # ВНИМАНИЕ: при этой схеме НЕ делите скоры на sqrt(head_dim) в forward.
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim ** -0.25
        )
        # В оригинале ключи проходят через whiten_keys + balance_keys
        # (защита от дрейфа bias ключей) — опущено.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, T, embed_dim), pos_emb: (1, 2T-1, pos_dim) -> (B, H, T, T)."""
        b, t, _ = x.shape
        h = self.num_heads
        qhd, phd = self.query_head_dim, self.pos_head_dim

        x = self.in_proj(x)
        q, k, p = x.split([qhd * h, qhd * h, phd * h], dim=-1)

        q = q.view(b, t, h, qhd).transpose(1, 2)                    # (B, H, T, qhd)
        k = k.view(b, t, h, qhd).transpose(1, 2).transpose(-2, -1)  # (B, H, qhd, T)
        p = p.view(b, t, h, phd).transpose(1, 2)                    # (B, H, T, phd)

        attn_scores = torch.matmul(q, k)  # (B, H, T, T)

        if not self.training or random.random() >= self.pos_emb_skip_rate:
            pos_emb = self.linear_pos(pos_emb)  # (1, 2T-1, H*phd)
            l2 = pos_emb.size(1)
            pos_emb = pos_emb.view(1, l2, h, phd).permute(0, 2, 3, 1)  # (1, H, phd, 2T-1)
            pos_scores = torch.matmul(p, pos_emb)  # (B, H, T, 2T-1)
            attn_scores = attn_scores + rel_shift(pos_scores)

        if lengths is not None:
            mask = torch.arange(t, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            # -1000, а не -inf: полностью замаскированная строка не дает NaN
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1000)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        return attn_weights


class NonlinAttention(nn.Module):
    """
    NonlinAttention из Zipformer2: как ConvolutionModule, но вместо depthwise
    свертки — умножение на (заимствованные) attention weights. Второй
    нелинейности после attention нет — вместо нее линейный гейт y.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.tanh = nn.Tanh()
        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05
        )
        # В оригинале: balancer перед tanh (сscheduled min_positive/max_positive),
        # whiten1 после `x = x * s`, whiten2 после out_proj — опущены.

    def forward(
        self, x: torch.Tensor, attn_weights: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, T, channels), attn_weights: (B, H, T, T)."""
        b, t, _ = x.shape

        x = self.in_proj(x)  # (B, T, 3 * hidden_channels)
        s, x, y = x.chunk(3, dim=-1)

        s = self.tanh(s)
        x = x * s  # гейт до attention

        num_heads = attn_weights.size(1)
        head_dim = self.hidden_channels // num_heads
        x = x.view(b, t, num_heads, head_dim).transpose(1, 2)
        # (B, H, T, head_dim)

        x = torch.matmul(attn_weights, x)

        x = x.transpose(1, 2).contiguous().view(b, t, self.hidden_channels)
        x = x * y

        x = self.out_proj(x)
        return apply_mask(x, lengths)


class SelfAttention(nn.Module):
    """
    Value-only attention из Zipformer2: работает с уже посчитанными attention
    weights (их считает RelPositionMultiheadAttentionWeights).

    x: (B, T, embed_dim), attn_weights: (B, H, T, T), суммы по последней оси = 1.
    """

    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim

        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )
        # В оригинале после out_proj — Whiten(...), опущен.

    def forward(
        self, x: torch.Tensor, attn_weights: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        b, t, _ = x.shape

        x = self.in_proj(x)  # (B, T, H * value_head_dim)
        x = x.view(b, t, self.num_heads, self.value_head_dim).transpose(1, 2)
        # (B, H, T, value_head_dim)

        x = torch.matmul(attn_weights, x)

        x = x.transpose(1, 2).contiguous().view(
            b, t, self.num_heads * self.value_head_dim
        )
        x = self.out_proj(x)

        # Маска — контракт вашего пайплайна: без нее in_proj(0) = bias дал бы
        # мусор в паддинге residual-потока.
        return apply_mask(x, lengths)


class FeedforwardModule(nn.Module):
    """
    Feedforward-модуль из Zipformer2 (icefall).

    Вход/выход: (B, T, embed_dim). Норм внутри НЕТ — нормализация
    живет в соседних модулях (BiasNorm в attention/conv) и на уровне энкодера.
    """

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        # В оригинале out_proj — это ActivationDropoutAndLinear:
        #   SwooshL -> Dropout3(shared_dim=0) -> ScaledLinear(initial_scale=0.1),
        # слитые в один autograd.Function ради экономии памяти.
        # Эмулируем initial_scale маленькой инициализацией:
        self.out_proj = ScaledLinear(feedforward_dim, embed_dim, initial_scale=0.1)
        # В оригинале здесь также: hidden_balancer (после in_proj)
        # и out_whiten (после out_proj) — оба опущены.

        self.activation = SwooshL()
        self.dropout = dropout

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.activation(x)

        # Dropout3 c shared_dim=0: маска формы (1, T, D), общая для всего батча.
        if self.training and self.dropout > 0.0:
            mask = (
                torch.rand((1, x.size(1), x.size(2)), device=x.device, dtype=x.dtype)
                > self.dropout
            )
            x = x * mask / (1.0 - self.dropout)

        x = self.out_proj(x)

        # Маска — не из оригинала, а контракт вашего пайплайна
        # (незамаскированный bias in_proj давал бы мусор в паддинге residual-потока).
        return apply_mask(x, lengths)


class ConvolutionModule(nn.Module):
    """
    ConvolutionModule из Zipformer2. Норм и дропаута внутри нет,
    активация — SwooshR внутри out_proj.
    """

    def __init__(self, channels: int, kernel_size: int, causal: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size должен быть нечетным для 'same' паддинга"
        self.causal = causal

        self.in_proj = nn.Linear(channels, 2 * channels)
        self.sigmoid = nn.Sigmoid()

        # Каузальный вариант (ChunkCausalDepthwiseConv1d) опущен.
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        # В оригинале: ActivationDropoutAndLinear(..., activation="SwooshR",
        # dropout_p=0.0, initial_scale=0.05) + balancer1/balancer2/whiten — опущены.
        self.activation = SwooshR()
        self.out_proj = ScaledLinear(channels, channels, bias=True, initial_scale=0.05)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T, channels)."""
        x = self.in_proj(x)  # (B, T, 2C)

        x, s = x.chunk(2, dim=-1)  # эквивалент nn.GLU, но в два шага
        s = self.sigmoid(s)
        x = x * s

        x = x.transpose(1, 2)  # (B, C, T)
        # Единственная маска в модуле — перед depthwise conv (как в оригинале)
        valid = (
            torch.arange(x.size(2), device=x.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        x = x.masked_fill(~valid.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)  # (B, T, C)

        x = self.activation(x)  # SwooshR
        x = self.out_proj(x)
        return apply_mask(x, lengths)  # финальная маска — контракт вашего пайплайна


class BypassModule(nn.Module):
    """
    Обучаемый bypass из Zipformer2:
        y = src_orig + (src - src_orig) * bypass_scale
    bypass_scale — per-channel, init 0.5. Во время обучения зажимается в
    [scale_min, scale_max]. В оригинале scale_min — расписание (0.9 -> 0.2
    за 20000 шагов): рано модуль обязан учиться (вклад ~residual), позже
    модели разрешается его ослаблять. Здесь фиксированное значение,
    расписание можно навесить снаружи через set_scale_min().
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: float = 0.0,
        straight_through_rate: float = 0.0,
        scale_min: float = 0.2,
        scale_max: float = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = skip_rate
        self.straight_through_rate = straight_through_rate
        self.scale_min = scale_min
        self.scale_max = scale_max

    def set_scale_min(self, value: float) -> None:
        # Точка входа для внешнего расписания (в оригинале ScheduledFloat).
        self.scale_min = value

    def _get_bypass_scale(self, batch_size: int) -> torch.Tensor:
        if not self.training:
            return self.bypass_scale
        # limit_param_value в оригинале: clamp, градиент течет только в диапазоне.
        ans = self.bypass_scale.clamp(min=self.scale_min, max=self.scale_max)
        if self.skip_rate != 0.0:
            mask = torch.rand((batch_size, 1), device=ans.device) > self.skip_rate
            ans = ans * mask  # (B, C): skip для случайных последовательностей
        if self.straight_through_rate != 0.0:
            mask = torch.rand((batch_size, 1), device=ans.device) < self.straight_through_rate
            ans = torch.maximum(ans, mask.to(ans.dtype))
        return ans

    def forward(self, src_orig: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        bypass_scale = self._get_bypass_scale(src.size(0))
        return src_orig + (src - src_orig) * bypass_scale


class Zipformer2EncoderLayer(nn.Module):
    """
    Блок Zipformer2: FF1 -> NLA -> SA1 -> Conv1 -> FF2 -> bypass_mid
    -> SA2 -> Conv2 -> FF3 -> BiasNorm -> bypass.
    Общие attention weights считаются один раз из входа слоя.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # В оригинале bypass'ы создаются с skip_rate (расписание bypass_skip_rate)
        # и skip-вероятностями у динамических модулей — опущено.
        self.bypass = BypassModule(embed_dim)
        self.bypass_mid = BypassModule(embed_dim)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
            # в оригинале pos_emb_skip_rate с расписанием — опущен
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)
        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)

        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)
        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        self.norm = BiasNorm(embed_dim)
        # В оригинале после norm/bypass еще balancer1, balancer2, whiten — опущены.

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """src: (B, T, embed_dim), pos_emb: (1, 2T-1, pos_dim)."""
        src_orig = src

        # Общие attention weights — из ВХОДА слоя, один раз
        attn_weights = self.self_attn_weights(src, pos_emb, lengths)

        # FF1 (dim 3/4) — обычный residual
        src = src + self.feed_forward1(src, lengths)

        # NLA: только голова 0 общих весов
        selected_attn_weights = attn_weights[:, 0:1]
        src = src + self.nonlin_attention(src, selected_attn_weights, lengths)

        # SA1 — все головы
        src = src + self.self_attn1(src, attn_weights, lengths)

        # Conv1
        src = src + self.conv_module1(src, lengths)

        # FF2 (dim 1.0)
        src = src + self.feed_forward2(src, lengths)

        # bypass_mid: стягивает текущее состояние с ВХОДОМ слоя
        src = self.bypass_mid(src_orig, src)

        # SA2 — те же общие веса
        src = src + self.self_attn2(src, attn_weights, lengths)

        # Conv2
        src = src + self.conv_module2(src, lengths)

        # FF3 (dim 5/4)
        src = src + self.feed_forward3(src, lengths)

        # Финальная норма и bypass к входу слоя
        src = self.norm(src)
        src = self.bypass(src_orig, src)

        return src


class ZipformerEncoder(nn.Module):
    """
    Flat-вариант Zipformer2: один стек блоков без down/up-sampling.
    (B, n_mels, T) -> (B, (T-7)//2, d_model): /2 по времени, /4 по частоте.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        n_layers: int = 12,
        num_heads: int = 4,
        query_head_dim: int = 24,
        pos_head_dim: int = 4,
        value_head_dim: int = 24,
        pos_dim: int = 128,
        feedforward_dim: int = 512,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        pe_max_len: int = 5000,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(d_model, n_mels, dropout)
        self.pos_encoding = CompactRelPositionalEncoding(
            pos_dim=pos_dim, dropout=dropout, max_len=pe_max_len
        )

        self.blocks = nn.ModuleList([
            Zipformer2EncoderLayer(
                embed_dim=d_model,
                pos_dim=pos_dim,
                num_heads=num_heads,
                query_head_dim=query_head_dim,
                pos_head_dim=pos_head_dim,
                value_head_dim=value_head_dim,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                cnn_module_kernel=conv_kernel_size,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = BiasNorm(d_model)

    def forward(self, melspecs: torch.Tensor, lengths: torch.Tensor):
        """
        melspecs: (B, n_mels, T) из DataLoader, lengths: (B,) в кадрах fbank.
        """
        x = melspecs.transpose(1, 2)                      # (B, T, n_mels)
        x, lengths = self.subsampling(x, lengths)         # (B, (T-7)//2, d_model)
        pos_emb = self.pos_encoding(x)                    # (1, 2T'-1, pos_dim)

        for block in self.blocks:
            x = block(x, pos_emb, lengths)

        x = self.final_norm(x)
        return x, lengths


class ZipformerCTC(nn.Module):
    def __init__(self, vocab_size: int, **encoder_kwargs):
        super().__init__()
        self.encoder = ZipformerEncoder(**encoder_kwargs)
        self.ctc_head = nn.Linear(encoder_kwargs["d_model"], vocab_size)

    def forward(self, melspecs: torch.Tensor, lengths: torch.Tensor):
        # 1. Получаем акустические фичи
        x, out_lengths = self.encoder(melspecs, lengths)

        # 2. Проецируем в токены
        logits = self.ctc_head(x)               # (B, T', V)
        logits = apply_mask(logits, out_lengths)
        # ВАЖНО: возвращаем out_lengths наружу
        return logits, out_lengths
