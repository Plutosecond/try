import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.SelfAttention_Family import FullAttention


class series_decomp_ema_momentum(nn.Module):
    """
    使用指数移动平均(EMA)计算历史动量作为趋势
    核心思想:
    1. 计算历史变化率(速度)
    2. 对历史速度进行EMA平滑得到动量
    3. 动量累积 = 趋势
    """

    def __init__(self, d_model, momentum_window=10, ema_alphas=[0.9, 0.95, 0.99],
                 learnable_alpha=True):
        """
        Args:
            d_model: 特征维度
            momentum_window: 用于计算动量的历史窗口长度
            ema_alphas: EMA系数列表(支持多尺度)
            learnable_alpha: EMA系数是否可学习
        """
        super(series_decomp_ema_momentum, self).__init__()
        self.d_model = d_model
        self.momentum_window = momentum_window
        self.num_alphas = len(ema_alphas)

        # EMA系数
        if learnable_alpha:
            # 使用logit表示,通过sigmoid映射到(0,1)
            self.ema_alpha_logits = nn.Parameter(
                torch.logit(torch.tensor(ema_alphas, dtype=torch.float32))
            )
        else:
            self.register_buffer('ema_alphas', torch.tensor(ema_alphas))

        self.learnable_alpha = learnable_alpha

        # 融合多尺度EMA动量
        self.momentum_fusion = nn.Sequential(
            nn.Linear(self.num_alphas, self.num_alphas),
            nn.GELU(),
            nn.Linear(self.num_alphas, 1),
            nn.Sigmoid()
        )

        # 动量到趋势的转换网络
        self.momentum_to_trend = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        # 趋势平滑层
        self.trend_smooth = nn.Conv1d(
            d_model, d_model,
            kernel_size=5, padding=2,
            groups=d_model,
            bias=False
        )

    def compute_velocity(self, x):
        """
        计算历史速度(一阶差分)

        Args:
            x: [B, L, D]
        Returns:
            velocity: [B, L, D]
        """
        # 计算相邻时间步的差分
        velocity = torch.diff(x, dim=1)
        # padding保持长度
        velocity = F.pad(velocity, (0, 0, 1, 0), mode='replicate')
        return velocity

    def compute_ema_momentum(self, velocity, alpha):

        B, L, D = velocity.shape

        momentum_list = []
        momentum_prev = velocity[:, 0:1, :]  # [B, 1, D]
        momentum_list.append(momentum_prev)

        for t in range(1, L):
            momentum_curr = alpha * momentum_prev + (1 - alpha) * velocity[:, t:t + 1, :]
            momentum_list.append(momentum_curr)
            momentum_prev = momentum_curr

        momentum = torch.cat(momentum_list, dim=1)  # [B, L, D]
        return momentum

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - 输入序列
        Returns:
            seasonal: [B, L, D] - 季节项(残差)
            trend: [B, L, D] - 趋势项(基于历史动量)
        """
        B, L, D = x.shape

        # 1. 计算历史速度
        velocity = self.compute_velocity(x)

        # 2. 使用多尺度EMA计算动量
        if self.learnable_alpha:
            ema_alphas = torch.sigmoid(self.ema_alpha_logits)
        else:
            ema_alphas = self.ema_alphas

        momentum_list = []
        for alpha in ema_alphas:
            momentum = self.compute_ema_momentum(velocity, alpha)
            momentum_list.append(momentum.unsqueeze(-1))

        momentum_stack = torch.cat(momentum_list, dim=-1)  # [B, L, D, num_alphas]

        # 3. 融合多尺度动量
        # 方法A: 简单加权平均
        weights = F.softmax(ema_alphas, dim=0)
        momentum_fused = torch.sum(
            momentum_stack * weights.view(1, 1, 1, -1),
            dim=-1
        )  # [B, L, D]

        # 方法B: 基于内容的自适应融合(可选)
        # fusion_weights = self.momentum_fusion(x)  # [B, L, 1]
        # momentum_fused = torch.sum(momentum_stack * fusion_weights.unsqueeze(2), dim=-1)

        # 4. 动量转换为趋势
        trend_raw = self.momentum_to_trend(momentum_fused)

        # 5. 累积动量得到趋势(积分)
        trend_cumsum = torch.cumsum(trend_raw, dim=1)

        # 6. 对齐初始值
        initial_offset = x[:, 0:1, :] - trend_cumsum[:, 0:1, :]
        trend = trend_cumsum + initial_offset

        # 7. 平滑趋势
        trend = self.trend_smooth(trend.transpose(1, 2)).transpose(1, 2)

        # 8. 残差作为季节项
        seasonal = x - trend

        return seasonal, trend

class LearnableMomentumBias(nn.Module):
    """
    可学习动量偏置模块
    """

    def __init__(self, d_model, momentum_hidden_dim=None, dropout=0.1):
        super(LearnableMomentumBias, self).__init__()
        self.d_model = d_model
        momentum_hidden_dim = momentum_hidden_dim or d_model // 2

        # 全局可学习动量系数
        self.global_momentum_weight = nn.Parameter(torch.tensor(0.5))

        # 特征维度的可学习权重
        self.feature_momentum_weight = nn.Parameter(torch.ones(d_model))

        # 基于内容的自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(d_model, momentum_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(momentum_hidden_dim, 1),
            nn.Sigmoid()
        )

        # 动量投影层
        self.momentum_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.global_momentum_weight, 0.5)
        nn.init.ones_(self.feature_momentum_weight)

    def compute_momentum(self, trend_current, trend_previous=None):
        if trend_previous is not None:
            momentum = trend_current - trend_previous
        else:
            momentum = torch.diff(trend_current, dim=1)
            momentum = F.pad(momentum, (0, 0, 1, 0), mode='replicate')
        return momentum

    def forward(self, trend_current, trend_previous=None,
                use_projection=True, use_gate=True):
        # 1. 计算动量
        momentum = self.compute_momentum(trend_current, trend_previous)

        # 2. 动量投影
        if use_projection:
            momentum_projected = self.momentum_projection(momentum)
        else:
            momentum_projected = momentum

        # 3. 计算权重
        global_weight = torch.sigmoid(self.global_momentum_weight)
        feature_weight = torch.sigmoid(self.feature_momentum_weight)
        adaptive_weight = self.adaptive_weight_net(trend_current)

        # 4. 应用权重
        combined_weight = global_weight * adaptive_weight
        momentum_weighted = momentum_projected * combined_weight * feature_weight

        # 5. 门控
        if use_gate and trend_previous is not None:
            gate_input = torch.cat([trend_current, momentum_projected], dim=-1)
            gate_value = self.gate(gate_input)
            momentum_weighted = momentum_weighted * gate_value

        # 6. 残差连接
        momentum_weighted = self.dropout(momentum_weighted)
        trend_enhanced = trend_current + momentum_weighted

        # 7. 返回信息
        momentum_info = {
            'global_weight': global_weight.item(),
            'adaptive_weight_mean': adaptive_weight.mean().item(),
        }

        return trend_enhanced, momentum_info

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            # self.decomp1 = series_decomp_multi(moving_avg)
            # self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp1 = series_decomp_ema_momentum(d_model)
            self.decomp2 = series_decomp_ema_momentum(d_model)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu",use_momentum=True, momentum_config=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            # self.decomp1 = series_decomp_multi(moving_avg)
            # self.decomp2 = series_decomp_multi(moving_avg)
            # self.decomp3 = series_decomp_multi(moving_avg)
            self.decomp1 = series_decomp_ema_momentum(d_model)
            self.decomp2 = series_decomp_ema_momentum(d_model)
            self.decomp3 = series_decomp_ema_momentum(d_model)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.use_momentum = use_momentum
        if use_momentum:
            if momentum_config is None:
                momentum_config = {
                    'momentum_hidden_dim': d_model // 2,
                    'dropout': dropout
                }

            self.momentum1 = LearnableMomentumBias(d_model, **momentum_config)
            self.momentum2 = LearnableMomentumBias(d_model, **momentum_config)
            self.momentum3 = LearnableMomentumBias(d_model, **momentum_config)

    def forward(self, x, cross, x_mask=None, cross_mask=None,trend=None):
        # ========== 第一阶段 ==========
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)

        # ===== 动量增强1 ===== (新增)
        if self.use_momentum:
            trend1, _ = self.momentum1(trend1, trend,
                                       use_projection=True, use_gate=True)
        # ===================

        # ========== 第二阶段 ==========
        x = x + self.dropout(self.cross_attention(x, cross, cross,
                                                  attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)

        # ===== 动量增强2 ===== (新增)
        if self.use_momentum:
            trend2, _ = self.momentum2(trend2, trend1,
                                       use_projection=True, use_gate=True)
        # ===================

        # ========== 第三阶段 ==========
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x, trend3 = self.decomp3(x)

        # ===== 动量增强3 ===== (新增)
        if self.use_momentum:
            trend3, _ = self.momentum3(trend3, trend2,
                                       use_projection=True, use_gate=True)
        # ===================

        # ========== 聚合trend ==========
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
