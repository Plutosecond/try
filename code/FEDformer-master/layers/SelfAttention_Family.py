import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


class MomentumAttention(nn.Module):
    """
    改进的动量注意力机制：支持多特征融合
    """

    def __init__(self,
                 mask_flag=True,
                 factor=5,
                 scale=None,
                 attention_dropout=0.1,
                 output_attention=False,
                 momentum_alpha=1.0,
                 momentum_k=1,
                 d_model=None,
                 fusion_method='learned',
                 num_heads=None):
        """
        Args:
            mask_flag: 是否使用因果掩码
            attention_dropout: dropout率
            output_attention: 是否输出注意力权重
            momentum_alpha: 趋势对齐惩罚系数（越大趋势约束越强）
            momentum_k: 动量回看步长 (x_t - x_{t-k})
            d_model: 输入特征维度（用于可学习融合）
            fusion_method: 特征融合方法 ['learned', 'mean', 'multihead']
        """
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.momentum_alpha = momentum_alpha
        self.momentum_k = momentum_k
        self.fusion_method = fusion_method

        self.num_heads = num_heads

        self.alpha_trend = nn.Parameter(torch.ones(num_heads))
        # self.alpha_season = nn.Parameter(torch.zeros(num_heads))


        # 特征融合层
        if fusion_method == 'learned' and d_model is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )
        else:
            self.feature_proj = None

    def compute_soft_momentum_bias(self, raw_series):
        """
        计算软动量偏置

        Args:
            raw_series: [B, L] 或 [B, L, D] 原始序列

        Returns:
            soft_bias: [B, 1, L, L] 动量偏置矩阵
        """
        # ===========================
        # 1. 多特征融合
        # ===========================
        if raw_series.dim() == 2:
            # 单特征输入 [B, L]
            fused = raw_series
            B, L = fused.shape

        elif raw_series.dim() == 3:
            # 多特征输入 [B, L, D]
            B, L, D = raw_series.shape

            if self.fusion_method == 'learned' and self.feature_proj is not None:
                # 可学习权重融合
                fused = self.feature_proj(raw_series).squeeze(-1)  # [B, L]

            elif self.fusion_method == 'multihead':
                # 多头动量（每个特征独立计算后平均）
                return self._compute_multihead_momentum(raw_series)

            else:
                # 默认：均值池化
                fused = raw_series.mean(dim=-1)  # [B, L]
        else:
            raise ValueError(f"raw_series should be 2D or 3D, got {raw_series.dim()}D")

        # ===========================
        # 2. 计算动量 M_t = x_t - x_{t-k}
        # ===========================
        k = min(self.momentum_k, L - 1)
        if k < 1:
            k = 1

        # 差分计算
        momentum = fused[:, k:] - fused[:, :-k]  # [B, L-k]
        momentum = nn.functional.pad(momentum, (k, 0), value=0)  # [B, L]

        # ===========================
        # 3. 计算趋势差异矩阵 |M_i - M_j|
        # ===========================
        S_i = momentum.unsqueeze(2)  # [B, L, 1]
        S_j = momentum.unsqueeze(1)  # [B, 1, L]
        diff = torch.abs(S_i - S_j)  # [B, L, L]

        # ===========================
        # 4. 生成软偏置（惩罚趋势不一致）
        # ===========================
        soft_bias = -self.momentum_alpha * diff  # [B, L, L]
        soft_bias = soft_bias.unsqueeze(1)  # [B, 1, L, L] 添加head维度

        return soft_bias

    def _compute_multihead_momentum(self, raw_series):
        """
        多头动量计算（每个特征独立）

        Args:
            raw_series: [B, L, D]

        Returns:
            soft_bias: [B, 1, L, L]
        """
        B, L, D = raw_series.shape
        k = min(self.momentum_k, L - 1)

        # 每个特征独立计算动量
        momentum_list = []
        for d in range(D):
            feature_d = raw_series[:, :, d]  # [B, L]
            mom_d = feature_d[:, k:] - feature_d[:, :-k]
            mom_d = nn.functional.pad(mom_d, (k, 0), value=0)
            momentum_list.append(mom_d)

        momentum = torch.stack(momentum_list, dim=1)  # [B, D, L]

        # 计算每个特征的趋势差异
        S_i = momentum.unsqueeze(3)  # [B, D, L, 1]
        S_j = momentum.unsqueeze(2)  # [B, D, 1, L]
        diff = torch.abs(S_i - S_j)  # [B, D, L, L]

        # 聚合所有特征（平均）
        soft_bias = -self.momentum_alpha * diff.mean(dim=1, keepdim=True)  # [B, 1, L, L]

        return soft_bias

    def forward(self, queries, keys, values, attn_mask, raw_series):
        """
        前向传播

        Args:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            values: [B, S, H, D]
            attn_mask: 注意力掩码（可选）
            raw_series: [B, L] 或 [B, L, D] 原始输入序列

        Returns:
            output: [B, L, H, D]
            attention: [B, H, L, S] (if output_attention=True)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # ================================
        # 1. 计算标准注意力分数
        # ================================
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        # ================================
        # 2. 添加动量偏置（核心创新）
        # ================================
        soft_bias = self.compute_soft_momentum_bias(raw_series)  # [B, 1, L, L]

        # 广播加到所有注意力头
        if S == L:  # 只在自注意力时添加
            scores = scores + soft_bias*self.alpha_trend.view(1,-1,1,1)  # [B, H, L, L]

        # ================================
        # 3. decoder自注意力应用因果掩码
        # ================================
        if self.mask_flag:
            if attn_mask is None:
                # 自动生成三角因果掩码
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # ================================
        # 4. Softmax + Dropout
        # ================================
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]

        # ================================
        # 5. 加权聚合 Value
        # ================================
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, L, H, D]

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask,raw_series):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask,raw_series=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            raw_series
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
