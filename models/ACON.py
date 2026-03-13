"""
ACON.py —— Adversarial CO-learning Networks (ACON) 销售预测模型

作为 HierDA 的对比基线，实现 ACON 框架（NeurIPS 2024）：

核心观察：
  - 频域特征在域内具有更强的判别性（Discriminability）
  - 时域特征在跨域迁移中具有更强的可迁移性（Transferability）
  因此不能对时域与频域特征一视同仁，需要差异化处理。

三大核心模块：
  ① 多周期频域特征学习（Multi-Period Frequency Feature Learning）：
      对时序做多尺度 FFT（不同截断长度），捕捉销售数据中
      月度、季度等多种周期性模式，增强频域特征的域内判别性。

  ② 时频互学习（Temporal-Frequency Mutual Learning）：
      用频域特征监督时域特征学习（提升时域判别性），
      用时域特征监督频域特征学习（提升频域可迁移性），
      通过相互蒸馏实现协同增强。

  ③ 时频相关子空间对抗对齐（Adversarial Alignment in
     Temporal-Frequency Correlation Subspace）：
      不在原始特征空间做对抗，而是在时域特征与频域特征的
      外积相关矩阵展开后的子空间中做 GRL 对抗对齐，
      捕捉时频联合分布的域差异，进一步提升迁移性。

接口与所有对比基线完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Liu et al., "Boosting Transferability and Discriminability for
  Time Series Domain Adaptation", NeurIPS 2024.
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
# 1. 时域特征提取器（Temporal Encoder）
#    多层 1D 卷积 + 残差连接，输出时序均值池化表示
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim,
                 kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch  = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else feat_dim
            layers += [
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.net(x.permute(0, 2, 1))   # (batch, feat_dim, seq_len)
        return h.mean(dim=-1)              # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 多周期频域特征学习（Multi-Period Frequency Feature Learning）
#
#    核心思想（ACON Module ①）：
#      销售时序中同时存在月度、季度等多种周期性，
#      单一 FFT 长度只能捕捉特定频率范围。
#      通过多尺度截断 FFT 提取不同粒度的频谱特征，
#      再融合为统一的频域表示，增强域内判别性。
#
#    多尺度策略：
#      对原始时序在全长、3/4长、1/2长分别做 FFT，
#      每个尺度用独立 MLP 提取频谱特征后加权融合。
# ══════════════════════════════════════════════════════════════════════════════

class MultiPeriodFreqEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, feat_dim,
                 periods=(1, 2, 4), dropout=0.1):
        """
        参数：
            seq_len: 输入序列长度
            periods: 多尺度截断比例的分母列表，
                     1→全长，2→1/2长，4→1/4长
            feat_dim: 输出特征维度
        """
        super().__init__()
        self.seq_len = seq_len
        self.periods = periods

        # 每个尺度的 FFT 输出维度（取前 seq_len//p//2+1 个频率分量）
        self.freq_dims = [seq_len // p // 2 + 1 for p in periods]

        # 每个尺度独立的频谱特征提取 MLP
        self.period_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fd * input_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for fd in self.freq_dims
        ])

        # 多尺度注意力加权融合
        self.attn_weight = nn.Linear(feat_dim, 1)
        self.out_proj    = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, feat_dim)
        """
        period_feats = []
        for i, p in enumerate(self.periods):
            # 截断到 seq_len // p 长度后做 FFT
            trunc_len = max(2, self.seq_len // p)
            x_trunc   = x[:, :trunc_len, :]              # (batch, trunc_len, D)

            # FFT 取幅度谱（实信号利用共轭对称性只取一半）
            xf     = torch.fft.rfft(x_trunc, dim=1)      # (batch, freq_dim, D)
            xf_amp = torch.abs(xf)                        # (batch, freq_dim, D)

            # 归一化
            xf_amp = (xf_amp - xf_amp.mean(dim=1, keepdim=True)) / (
                xf_amp.std(dim=1, keepdim=True) + 1e-8
            )

            # 展平后经 MLP 提取特征
            batch = xf_amp.shape[0]
            xf_flat = xf_amp.reshape(batch, -1)          # (batch, freq_dim*D)

            # 对齐到期望维度（防止因 trunc_len 差异导致维度不匹配）
            expected = self.freq_dims[i] * x.shape[-1]
            if xf_flat.shape[1] < expected:
                pad = torch.zeros(batch, expected - xf_flat.shape[1],
                                  device=x.device)
                xf_flat = torch.cat([xf_flat, pad], dim=1)
            else:
                xf_flat = xf_flat[:, :expected]

            feat = self.period_nets[i](xf_flat)           # (batch, feat_dim)
            period_feats.append(feat)

        # 注意力加权融合各尺度特征
        stacked = torch.stack(period_feats, dim=1)        # (batch, n_periods, feat_dim)
        weights = F.softmax(self.attn_weight(stacked), dim=1)  # (batch, n_periods, 1)
        fused   = (stacked * weights).sum(dim=1)          # (batch, feat_dim)
        return self.out_proj(fused)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 时频互学习损失（Temporal-Frequency Mutual Learning Loss）
#
#    核心思想（ACON Module ②）：
#      用频域特征作为时域特征的监督信号（提升时域判别性）：
#        L_t = MSE(proj_t(t_feat), f_feat.detach())
#      用时域特征作为频域特征的监督信号（提升频域迁移性）：
#        L_f = MSE(proj_f(f_feat), t_feat.detach())
#
#      通过 .detach() 确保两个方向的蒸馏梯度不互相干扰，
#      各自优化对方的投影映射。
# ══════════════════════════════════════════════════════════════════════════════

class MutualLearningLoss(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 时域 → 频域方向的投影
        self.proj_t2f = nn.Linear(feat_dim, feat_dim)
        # 频域 → 时域方向的投影
        self.proj_f2t = nn.Linear(feat_dim, feat_dim)

    def forward(self, t_feat, f_feat):
        """
        t_feat: (batch, feat_dim) 时域特征
        f_feat: (batch, feat_dim) 频域特征
        返回: scalar tensor
        """
        # 频域监督时域（time → freq 投影后拟合频域目标）
        loss_t = F.mse_loss(self.proj_t2f(t_feat), f_feat.detach())
        # 时域监督频域（freq → time 投影后拟合时域目标）
        loss_f = F.mse_loss(self.proj_f2t(f_feat), t_feat.detach())
        return (loss_t + loss_f) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. 时频相关子空间对抗对齐
#    （Adversarial Alignment in Temporal-Frequency Correlation Subspace）
#
#    核心思想（ACON Module ③）：
#      在原始特征空间做对抗往往只对齐一阶统计量；
#      ACON 改为在时域特征与频域特征的外积（向量化后投影降维）
#      所构成的相关子空间中做对抗，能同时捕捉时频联合分布的域差异。
#
#      相关向量 = flatten(t_feat ⊗ f_feat) → 经投影降维到 corr_dim
#      然后在 corr_dim 维空间做标准 GRL 域分类对抗。
# ══════════════════════════════════════════════════════════════════════════════

class TFCorrelationDomainClassifier(nn.Module):
    def __init__(self, feat_dim, corr_dim=64, grl_alpha=1.0):
        """
        参数：
            feat_dim: 时域/频域特征维度
            corr_dim: 相关子空间投影维度（避免 feat_dim^2 的维度爆炸）
            grl_alpha: GRL 反转系数
        """
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        # 外积展开后维度为 feat_dim^2，先投影到 corr_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * feat_dim, corr_dim),
            nn.ReLU(),
        )
        # 域分类头
        self.classifier = nn.Sequential(
            nn.Linear(corr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, t_feat, f_feat):
        """
        t_feat: (batch, feat_dim)
        f_feat: (batch, feat_dim)
        返回: (batch, 1) 域分类 logit
        """
        # 外积相关矩阵：t_feat^T · f_feat → (batch, feat_dim, feat_dim)
        corr = torch.bmm(
            t_feat.unsqueeze(-1),    # (batch, feat_dim, 1)
            f_feat.unsqueeze(1),     # (batch, 1, feat_dim)
        )                            # (batch, feat_dim, feat_dim)

        # 展平并投影到相关子空间
        corr_flat = corr.reshape(corr.shape[0], -1)    # (batch, feat_dim^2)
        corr_proj = self.proj(corr_flat)               # (batch, corr_dim)

        # GRL + 域分类
        return self.classifier(self.grl(corr_proj))    # (batch, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 销售预测头（与所有对比基线保持一致）
# ══════════════════════════════════════════════════════════════════════════════

class LabelPredictor(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        # 时域特征 + 频域特征拼接后预测
        self.residual_net = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, 64),   # t_feat + f_feat + history_mean
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.output_scale = nn.Linear(1, 1)
        nn.init.constant_(self.output_scale.weight, 1.0)
        nn.init.constant_(self.output_scale.bias,   0.0)

    def forward(self, t_feat, f_feat, history_mean):
        """
        t_feat:       (batch, feat_dim)  时域特征
        f_feat:       (batch, feat_dim)  频域特征
        history_mean: (batch, 1)
        返回:         (batch, 1)
        """
        x        = torch.cat([t_feat, f_feat, history_mean], dim=-1)
        residual = self.residual_net(x)
        return self.output_scale(history_mean + residual)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ACON 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    ACON 完整实现（HierDA 对比基线，NeurIPS 2024）。

    四个损失项：
      Loss P    — 销售预测损失（主损失，由训练循环计算）
      Loss Mut  — 时频互学习损失（extra_loss 的一部分）
      Loss Adv  — 时频相关子空间对抗损失（extra_loss 的一部分）
      Loss Ent  — 目标域预测熵正则（extra_loss 的一部分，对应原文 entropy_trade_off）

    forward 接口与所有对比基线完全兼容：
      model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
        batch_x_src  → x_enc    (源域序列)
        batch_x      → x_target (目标域序列)
    """

    def __init__(self, configs):
        super().__init__()

        input_dim = configs.enc_in
        seq_len   = configs.seq_len
        device    = 'cuda' if configs.use_gpu else 'cpu'

        self.device     = device
        self.pred_len   = configs.pred_len
        self.label_len  = configs.label_len
        self.input_dim  = input_dim
        self.extra_loss = None

        hidden_dim  = getattr(configs, 'd_model',      128)
        feat_dim    = getattr(configs, 'feat_dim',     64)    # 时域/频域各 feat_dim
        dropout     = getattr(configs, 'dropout',      0.1)
        kernel_size = getattr(configs, 'cnn_kernel',   3)
        num_layers  = getattr(configs, 'cnn_layers',   3)
        grl_alpha   = getattr(configs, 'grl_alpha',    1.0)
        corr_dim    = getattr(configs, 'corr_dim',     64)
        periods     = getattr(configs, 'freq_periods', (1, 2, 4))

        # 损失权重（对应原文超参数）
        self.w_mut  = getattr(configs, 'w_mut',  1.0)    # 时频互学习
        self.w_adv  = getattr(configs, 'w_adv',  1.0)    # 相关子空间对抗
        self.w_ent  = getattr(configs, 'w_ent',  0.01)   # 熵正则

        # ── 时域特征提取器（源域 & 目标域共享权重） ───────────────────────────
        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, feat_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # ── 多周期频域特征学习（源域 & 目标域共享权重） ───────────────────────
        self.freq_encoder = MultiPeriodFreqEncoder(
            seq_len, input_dim, feat_dim,
            periods=periods,
            dropout=dropout,
        ).to(device)

        # ── 时频互学习模块 ────────────────────────────────────────────────────
        self.mutual_loss = MutualLearningLoss(feat_dim).to(device)

        # ── 时频相关子空间域分类器（含 GRL） ─────────────────────────────────
        self.tf_domain_classifier = TFCorrelationDomainClassifier(
            feat_dim, corr_dim=corr_dim, grl_alpha=grl_alpha
        ).to(device)

        # ── 销售预测头 ────────────────────────────────────────────────────────
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

        # ── Module ① + 时域提取：时域 & 频域特征（源域 & 目标域） ─────────────
        src_t = self.temporal_encoder(source_data)   # (batch_s, feat_dim)
        src_f = self.freq_encoder(source_data)       # (batch_s, feat_dim)
        tgt_t = self.temporal_encoder(target_data)   # (batch_t, feat_dim)
        tgt_f = self.freq_encoder(target_data)       # (batch_t, feat_dim)

        # ── Module ②：时频互学习损失（源域上计算，提升时域判别性与频域迁移性） ──
        loss_mut_src = self.mutual_loss(src_t, src_f)
        loss_mut_tgt = self.mutual_loss(tgt_t, tgt_f)
        loss_mut = (loss_mut_src + loss_mut_tgt) / 2.0

        # ── Module ③：时频相关子空间对抗对齐 ────────────────────────────────
        # 拼接源域与目标域，计算对抗损失
        all_t = torch.cat([src_t, tgt_t], dim=0)    # (batch_s+batch_t, feat_dim)
        all_f = torch.cat([src_f, tgt_f], dim=0)
        domain_logit = self.tf_domain_classifier(all_t, all_f).squeeze(-1)
        domain_label = torch.cat([
            torch.zeros(src_t.shape[0]),
            torch.ones(tgt_t.shape[0]),
        ]).to(source_data.device)
        loss_adv = F.binary_cross_entropy_with_logits(domain_logit, domain_label)

        # ── 目标域预测熵正则（鼓励目标域预测置信，减少不确定性） ───────────────
        tgt_history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)
        pred_tgt_raw = self.label_predictor(tgt_t, tgt_f, tgt_history_mean)
        # 回归任务中熵正则用预测值方差代替（批次内方差越小 → 预测越集中 → 熵越低）
        pred_var = pred_tgt_raw.var(dim=0).mean()
        loss_ent = pred_var

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = (
            self.w_mut * loss_mut
            + self.w_adv * loss_adv
            + self.w_ent * loss_ent
        )

        # ── 预测：基于目标域时域 + 频域特征 ──────────────────────────────────
        pred_value = pred_tgt_raw.expand(-1, self.pred_len)   # (batch_t, pred_len)

        # 补零对齐其余特征维度，保持输出形状与 HierDA 一致
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device,
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)
        return pred