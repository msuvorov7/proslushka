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
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=(1, 1), groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=(1, 1), groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
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


class LocalAttRelPositionalEncoding(PositionalEncoder):
    def __init__(self, att_context_size, **kwargs):
        super(LocalAttRelPositionalEncoding, self).__init__(**kwargs)
        self.left_context = att_context_size[0]
        self.right_context = att_context_size[1]

    def extend_pe(self, device, dtype):
        positions = torch.arange(
            self.left_context, -self.right_context - 1, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, inputs: torch.Tensor, cache_len: int = 0):
        if self.xscale:
            inputs = inputs * self.xscale

        end_pos = self.left_context + self.right_context + 1
        pos_emb = self.pe[:, :end_pos]
        return self.dropout(inputs), pos_emb
    

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
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=9, stride=1, padding='same', groups=encoder_dim),
            # Transpose(1, 2),
            # nn.LayerNorm(encoder_dim),
            # Transpose(1, 2),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout, inplace=True),
            Transpose(1, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.model(inputs)
    

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


class RelPositionMultiHeadAttentionLongformer(RelPositionMultiHeadAttention):
    INF_VAL: float = 10000.0
    def __init__(
        self,
        n_head,
        n_feat,
        att_context_size: list,
        dropout: float = 0.1,
    ):
        """Construct an RelPositionMultiHeadAttentionLongformer object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
        )
        self.att_context_size = att_context_size
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, pos_emb, cache=None):
        q, k, v = self.forward_qkv(query, key, value)
        n_batch, _, T, _ = q.size()
        pad_mask = torch.zeros(n_batch, T, dtype=torch.bool)

        w = max(self.att_context_size[0], self.att_context_size[1])
        pad_len = (2 * w - T % (2 * w)) % (2 * w)  # pad time to 2w
        q = F.pad(q, (0, 0, 0, pad_len))  # (batch, head, time, size)
        k = F.pad(k, (0, 0, 0, pad_len))  # (batch, head, time, size)
        v = F.pad(v, (0, 0, 0, pad_len))  # (batch, head, time, size)
        mask = F.pad(pad_mask, (0, pad_len), value=1.0)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # (batch, head, time, size)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)  # (batch, head, time, size)

        diagonal_matrix_ac = self.sliding_chunks_matmul_qk(
            q_with_bias_u, k, w, padding_value=0.0
        )  # (batch, head, time, 2w + 1)

        # add relative positional embedding

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
        # (batch, head, 2w, size)
        diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # (batch, head, time, 2w + 1)

        start_pos = w - self.att_context_size[0]
        end_pos = w + self.att_context_size[1]

        diagonal_matrix_ac[:, :, :, : self.att_context_size[0]] += diagonal_matrix_bd[
            :, :, :, : self.att_context_size[0]
        ]
        diagonal_matrix_ac[:, :, :, -(self.att_context_size[1] + 1) :] += diagonal_matrix_bd[
            :, :, :, self.att_context_size[0] :
        ]
        scores = diagonal_matrix_ac / self.s_d_k
        # (batch, head, time, 2w + 1)

        # mask invalid positions
        scores[:, :, :, :start_pos] = -self.INF_VAL
        scores[:, :, :, end_pos + 1 :] = -self.INF_VAL

        # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
        # from (bsz x seq_len) to (bsz x num_heads x seqlen x hidden_size)
        mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
        # cast to float/half then replace 1's with -inf
        float_mask = mask.type_as(scores).masked_fill(mask, -self.INF_VAL)
        ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
        # diagonal mask with zeros everywhere and -inf inplace of padding
        d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
        # (batch, head, time, 2w + 1)

        scores += d_mask

        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        p_attn = self.dropout(attn)
        # (batch, head, time, 2w + 1)

        # compute local attn only
        out = self.sliding_chunks_matmul_pv(p_attn, v, w)

        out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]

        ret = self.linear_out(out)

        return ret

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Convert into overlapping chunks.

        Args:
            x (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): # (batch x head, chunk_count, 2w, size)
        """

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def _skew(self, x: torch.Tensor, direction: list[int], padding_value: float) -> torch.Tensor:
        """Convert diagonals into columns (or columns into diagonals depending on `direction`

        Args:
            x (torch.Tensor): (batch x head, chunk_count, 2w, 2w)
            direction (List[int]): padding directions
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunk_count, 2w, 2w + 1)

        """
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self, x: torch.Tensor, padding_value: float) -> torch.Tensor:
        """Shift every row 1 step to right converting columns into diagonals

        Args:
            x (torch.Tensor): (batch x head, chunks_count + 1, w, 2w + 1)
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunks_count + 1, w, 3w)
        """
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    def _get_invalid_locations_mask(self, w: int, device: str):

        diagonals_list = []
        for j in range(-w, 1):
            diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)

        mask = torch.stack(diagonals_list, dim=-1)
        mask = mask[None, None, :, :]

        ending_mask = mask.flip(dims=(2, 3)).bool().to(device)
        return mask.bool().to(device), ending_mask

    def mask_invalid_locations(
        self,
        input_tensor: torch.Tensor,
        w: int,
    ):
        """
        Mask locations invalid for the sliding window attention

        Args:
            input_tensor (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size
        """
        beginning_mask, ending_mask = self._get_invalid_locations_mask(w, input_tensor.device)
        seq_len = input_tensor.size(2)
        beginning_input = input_tensor[:, :, :w, : w + 1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))

        ending_input = input_tensor[:, :, -w:, -(w + 1) :]
        ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        """Matrix multiplication of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w
        with an overlap of size w

        Args:
            q (torch.Tensor): (batch, head, time, size)
            k (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size
            padding_value (float): Value to pad with

        Returns:
            output (torch.Tensor): (batch, head, time, 2w + 1)
        """
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)  # (batch x head, chunk_count, 2w, size)
        chunk_k = self._chunk_overlap(k, w)  # (batch x head, chunk_count, 2w, size)

        # matrix multipication
        # bcxd: bsz*num_heads x chunks x 2w x head_dim
        # bcyd: bsz*num_heads x chunks x 2w x head_dim
        # bcxy: bsz*num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
        # (batch x head, chunk_count, 2w, 2w)

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        # (batch x head, chunk_count, 2w, 2w + 1)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        # (batch x head, chunk_count + 1, w, 2w + 1)

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate bsz and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        # (batch, head, time, 2w + 1)

        self.mask_invalid_locations(diagonal_attn, w)

        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as sliding_chunks_matmul_qk but for prob and value tensors.

        Args:
            prob (torch.Tensor): (batch, head, time, size)
            v (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): (batch, time, head, size)
        """
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        # (batch x head, chunks_count + 1, w, 2w + 1)

        # group bsz and num_heads dimensions into one
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        # (batch x head, time, size)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
        # (batch x head, time + 2w, size)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
        # (batch x head, chunks_count + 1, 3w, size)

        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        # (batch x head, chunks_count + 1, w, 3w)

        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        # (batch x head, chunks_count + 1, w, size)

        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)
    

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
        self.attn = RelPositionMultiHeadAttentionLongformer(
            n_head=num_heads,
            n_feat=encoder_dim,
            att_context_size=[128, 128],
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputs: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        residual = inputs
        inputs = self.norm_attn(inputs)
        inputs = self.attn(inputs, inputs, inputs, pos_emb)
        return residual + self.dropout(inputs)
    

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
            256,
            kernel_size=3,
            stride=2,
          )
        subsampling_dim = self.evaluate_conv_out_dim(n_mels, 3, 2, 1)
        subsampling_dim = self.evaluate_conv_out_dim(subsampling_dim, 3, 2, 1)
        subsampling_dim = self.evaluate_conv_out_dim(subsampling_dim, 3, 2, 1)

        self.projection = ProjectionLayer(
            subsampling_dim * 256,
            encoder_dim,
            dropout,
        )

        self.pos_encoder = LocalAttRelPositionalEncoding(
            att_context_size=[128, 128],
            d_model=encoder_dim,
            xscale=math.sqrt(encoder_dim),
        )
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, pos_emb = self.encoder(inputs.transpose(1, 2))

        for block in self.conf_blocks:
            output = block(output, pos_emb)

        return self.decoder(output).transpose(1, 2)