"""
CoDATS.py —— Contrastive Domain Adaptation for Time Series (CoDATS) 销售预测模型

作为 HierDA 的对比基线，实现 CoDATS 框架：
  - 时序卷积网络（TCN）特征提取器：通过多层因果扩张卷积捕捉时序依赖
  - 域分类器 + 梯度反转层（GRL）：对抗训练对齐源域与目标域分布
  - 弱监督对比损失（Contrastive Loss）：拉近同域样本、推远跨域样本
  - 销售预测头（Label Predictor）

与其他基线的核心区别：
  - DANN：MLP 特征提取 + GRL，无时序感知
  - DeepCoral：MLP 特征提取 + 二阶统计对齐，无对抗
  - CoDATS：TCN 时序特征提取 + GRL 对抗 + 对比损失，专为时序域适应设计

接口与 HierDA / DANN / DeepCoral 完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Wilson et al., "Multi-Source Deep Domain Adaptation with Weak Supervision
  for Time-Series Sensor Data", KDD 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ══════════════════════════════════════════════════════════════════════════════
# 0. 梯度反转层（GRL）
# ══════════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 因果扩张卷积块（Causal Dilated Conv Block）
#    CoDATS 的核心时序感知模块，每块包含：
#      - 因果填充保证不泄露未来信息
#      - 扩张卷积以指数级扩大感受野
#      - 残差连接缓解梯度消失
# ══════════════════════════════════════════════════════════════════════════════

class CausalDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # 因果填充：仅填充左侧，确保 t 时刻只看到 t 及之前的信息
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm    = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

        # 残差投影（当输入输出通道数不同时）
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # 裁剪右侧多余的填充，保证因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu(self.norm(out))
        out = self.dropout(out)
        return out + self.residual_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TCN 时序特征提取器
#    多层因果扩张卷积，扩张率按 2^i 递增以指数级扩大感受野
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
# ══════════════════════════════════════════════════════════════════════════════

class TCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim,
                 kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()

        layers = []
        in_ch  = input_dim
        for i in range(num_layers):
            dilation = 2 ** i           # 扩张率：1, 2, 4, ...
            out_ch   = hidden_dim if i < num_layers - 1 else feat_dim
            layers.append(CausalDilatedConvBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))
            in_ch = out_ch

        self.tcn  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Conv1d 期望 (batch, channels, seq_len)
        x = x.permute(0, 2, 1)              # (batch, input_dim, seq_len)
        out = self.tcn(x)                   # (batch, feat_dim, seq_len)
        out = out.permute(0, 2, 1)          # (batch, seq_len, feat_dim)
        out = self.norm(out)
        return out.mean(dim=1)              # (batch, feat_dim) 时序均值池化


# ══════════════════════════════════════════════════════════════════════════════
# 3. 域分类器（Domain Classifier w/ GRL）
# ══════════════════════════════════════════════════════════════════════════════

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim, grl_alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(self.grl(x))


# ══════════════════════════════════════════════════════════════════════════════
# 4. 弱监督对比损失（Contrastive Loss）
#    CoDATS 的关键创新：不依赖目标域标签，仅通过域标签做弱监督
#    正样本对：同域特征（源-源 或 目标-目标）
#    负样本对：跨域特征（源-目标）
# ══════════════════════════════════════════════════════════════════════════════

def contrastive_loss(source_feat, target_feat, temperature=0.1):
    """
    基于 NT-Xent 的弱监督对比损失。

    正样本：同域内特征对（源域内部 / 目标域内部）
    负样本：跨域特征对（源域 vs 目标域）

    目标：拉近域内特征，推远跨域特征（使域不变特征空间更紧凑）

    参数：
        source_feat:  (batch_s, feat_dim)
        target_feat:  (batch_t, feat_dim)
        temperature:  softmax 温度系数
    返回：
        scalar tensor
    """
    batch_s = source_feat.shape[0]
    batch_t = target_feat.shape[0]
    min_b   = min(batch_s, batch_t)

    # batch < 2 时对比损失无意义，直接返回 0
    if min_b < 2:
        return torch.tensor(0.0, device=source_feat.device, requires_grad=True)

    src = F.normalize(source_feat[:min_b], dim=-1)   # (min_b, feat_dim)
    tgt = F.normalize(target_feat[:min_b], dim=-1)   # (min_b, feat_dim)

    # 拼接所有特征：前 min_b 为源域，后 min_b 为目标域
    all_feat = torch.cat([src, tgt], dim=0)           # (2*min_b, feat_dim)

    # 相似度矩阵
    sim_matrix = torch.mm(all_feat, all_feat.T) / temperature  # (2*min_b, 2*min_b)

    # 屏蔽对角线（自身相似度）
    mask_diag = torch.eye(2 * min_b, device=src.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask_diag, float('-inf'))

    # 正样本索引：每个样本的正样本为同域内的其他样本
    # source i 的正样本 = source 0..min_b-1（除自身）
    # target i 的正样本 = target min_b..2*min_b-1（除自身）
    labels = torch.cat([
        torch.arange(min_b),            # source i → source block
        torch.arange(min_b),            # target i → target block（偏移在 loss 计算中体现）
    ]).to(src.device)

    # 构造正样本 mask：同域内（非自身）视为正样本
    pos_mask = torch.zeros(2 * min_b, 2 * min_b, device=src.device)
    pos_mask[:min_b, :min_b] = 1.0     # 源域内互为正样本
    pos_mask[min_b:, min_b:] = 1.0     # 目标域内互为正样本
    pos_mask.fill_diagonal_(0.0)        # 排除自身

    # 对比损失：log-sum-exp over 负样本，正样本取均值
    log_prob    = F.log_softmax(sim_matrix, dim=-1)   # (2*min_b, 2*min_b)
    pos_log_sum = (log_prob * pos_mask).sum(dim=-1)   # (2*min_b,)
    pos_count   = pos_mask.sum(dim=-1).clamp(min=1)
    loss        = -(pos_log_sum / pos_count).mean()
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 5. 销售预测头（与 HierDA / DANN / DeepCoral 保持一致）
# ══════════════════════════════════════════════════════════════════════════════

class LabelPredictor(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Linear(feat_dim + 1, 64),   # feat_dim + history_mean
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.output_scale = nn.Linear(1, 1)
        nn.init.constant_(self.output_scale.weight, 1.0)
        nn.init.constant_(self.output_scale.bias,   0.0)

    def forward(self, feat, history_mean):
        """
        feat:         (batch, feat_dim)
        history_mean: (batch, 1)
        返回:         (batch, 1)
        """
        x        = torch.cat([feat, history_mean], dim=-1)
        residual = self.residual_net(x)
        return self.output_scale(history_mean + residual)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CoDATS 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    CoDATS 完整实现（HierDA 对比基线）。

    三个损失项：
      Loss P  — 销售预测损失（主损失，由训练循环计算）
      Loss C  — GRL 域分类对抗损失（extra_loss 的一部分）
      Loss Con— 弱监督对比损失（extra_loss 的一部分）

    forward 接口与 HierDA / DANN / DeepCoral 完全兼容：
      model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
        batch_x_src  → x_enc    (源域序列)
        batch_x      → x_target (目标域序列)
    """

    def __init__(self, configs):
        super().__init__()

        input_dim = configs.enc_in
        device    = 'cuda' if configs.use_gpu else 'cpu'

        self.device     = device
        self.pred_len   = configs.pred_len
        self.label_len  = configs.label_len
        self.input_dim  = input_dim
        self.extra_loss = None

        hidden_dim   = getattr(configs, 'd_model',        128)
        feat_dim     = getattr(configs, 'feat_dim',       128)
        dropout      = getattr(configs, 'dropout',        0.1)
        grl_alpha    = getattr(configs, 'grl_alpha',      1.0)
        kernel_size  = getattr(configs, 'tcn_kernel',     3)
        num_layers   = getattr(configs, 'tcn_layers',     3)
        temperature  = getattr(configs, 'temperature',    0.1)

        # 损失权重（可通过 configs 调节）
        self.w_adv  = getattr(configs, 'w_adv',  0.1)   # GRL 对抗损失权重
        self.w_con  = getattr(configs, 'w_con',  0.1)   # 对比损失权重
        self.temperature = temperature

        # TCN 特征提取器（源域和目标域共用权重）
        self.feature_extractor = TCNFeatureExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # 域分类器（含 GRL）
        self.domain_classifier = DomainClassifier(feat_dim, grl_alpha).to(device)

        # 销售预测头
        self.label_predictor = LabelPredictor(feat_dim, dropout).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        """
        参数（与 HierDA 保持一致）：
          x_enc    — 源域序列  (batch, seq_len, input_dim)
          x_target — 目标域序列 (batch, seq_len, input_dim)
                     若为 None，退化为无域适应

        返回：
          pred — (batch, pred_len, input_dim)
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── TCN 特征提取（源域 & 目标域共享提取器） ──────────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch, feat_dim)

        # ── GRL 域分类对抗损失（Loss C） ─────────────────────────────────────
        domain_feat  = torch.cat([source_feat, target_feat], dim=0)
        domain_label = torch.cat([
            torch.zeros(source_feat.shape[0]),
            torch.ones(target_feat.shape[0]),
        ]).to(source_data.device)
        domain_pred = self.domain_classifier(domain_feat).squeeze(-1)
        loss_adv = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        # ── 弱监督对比损失（Loss Con） ────────────────────────────────────────
        loss_con = contrastive_loss(source_feat, target_feat, self.temperature)

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = self.w_adv * loss_adv + self.w_con * loss_con

        # ── 预测：基于目标域特征 ──────────────────────────────────────────────
        history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)   # (batch, 1)
        pred_value   = self.label_predictor(target_feat, history_mean)   # (batch, 1)
        pred_value   = pred_value.expand(-1, self.pred_len)              # (batch, pred_len)

        # 补零对齐其余特征维度，保持输出形状与 HierDA 一致
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device,
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)   # (batch, pred_len, input_dim)
        return pred