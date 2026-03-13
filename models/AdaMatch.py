"""
AdaMatch.py —— AdaMatch: A Unified Approach to Semi-Supervised Domain Adaptation

作为 HierDA 的对比基线，实现 AdaMatch 框架（适配时序回归任务）：
  - 时序 CNN 特征提取器：捕捉局部时序模式
  - 随机时序增强（Random Temporal Augmentation）：
      对同一样本生成弱增强版本与强增强版本，
      弱增强用于生成伪标签，强增强用于一致性训练
  - 分布对齐伪标签（Distribution-Aligned Pseudo-Labels）：
      先用弱增强样本产生预测作为软伪标签，
      再对源域预测分布与目标域预测分布做对齐修正，
      减少域间预测偏移对伪标签质量的影响
  - 相对置信度阈值（Relative Confidence Thresholding）：
      基于批次内相对置信度而非固定阈值过滤低质量伪标签，
      适配域间样本分布不均衡的场景
  - 无监督一致性损失（Unsupervised Consistency Loss）：
      约束强增强样本的预测与分布对齐后的伪标签一致
  - 销售预测头（Label Predictor）

与其他基线的核心区别：
  ┌──────────────┬────────┬───────────┬─────────┬──────────┬──────────┬────────┬──────────────────────────┐
  │              │ DANN   │ DeepCoral │ CoDATS  │ AdvSKM   │RAINCOAT  │ CotMix │ AdaMatch                 │
  ├──────────────┼────────┼───────────┼─────────┼──────────┼──────────┼────────┼──────────────────────────┤
  │ 特征提取     │ MLP    │ MLP       │ TCN     │ LSTM     │CNN时频   │ CNN    │ CNN                      │
  │ 域对齐核心   │ GRL    │ CORAL     │GRL+对比 │ SKM+GRL  │原型+对比 │时序混合│ 分布对齐伪标签+一致性    │
  │ 伪标签       │ ✗      │ ✗         │ ✗       │ ✗        │ ✗        │ ✗      │ ✓ 分布对齐修正           │
  │ 数据增强     │ ✗      │ ✗         │ ✗       │ ✗        │ ✗        │ Mixup  │ ✓ 弱/强双增强            │
  │ 置信度过滤   │ ✗      │ ✗         │ ✗       │ ✗        │ ✗        │ ✗      │ ✓ 相对阈值               │
  │ 对抗训练     │ ✓      │ ✗         │ ✓       │ ✓        │ ✗        │ ✗      │ ✗                        │
  └──────────────┴────────┴───────────┴─────────┴──────────┴──────────┴────────┴──────────────────────────┘

接口与所有对比基线完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Berthelot et al., "AdaMatch: A Unified Approach to Semi-Supervised
  Domain Adaptation", ICLR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. 随机时序增强（Random Temporal Augmentation）
#
#    为同一序列生成弱增强版本与强增强版本：
#      弱增强：轻微的加性高斯噪声，保留整体分布不变，用于生成高质量伪标签
#      强增强：时序窗口遮蔽 + 幅度缩放 + 噪声叠加，
#              模拟更大的域内扰动，用于一致性训练的输入
# ══════════════════════════════════════════════════════════════════════════════

class RandomTemporalAugmentation(nn.Module):
    def __init__(self,
                 weak_noise_std=0.01,
                 strong_noise_std=0.05,
                 mask_ratio=0.2,
                 scale_range=(0.8, 1.2)):
        """
        参数：
            weak_noise_std:   弱增强高斯噪声标准差
            strong_noise_std: 强增强高斯噪声标准差
            mask_ratio:       强增强时序遮蔽比例（随机置零的时间步比例）
            scale_range:      强增强幅度缩放范围 [low, high]
        """
        super().__init__()
        self.weak_noise_std   = weak_noise_std
        self.strong_noise_std = strong_noise_std
        self.mask_ratio       = mask_ratio
        self.scale_low        = scale_range[0]
        self.scale_high       = scale_range[1]

    def weak_augment(self, x):
        """
        弱增强：仅加轻微高斯噪声，不改变时序结构。
        x: (batch, seq_len, input_dim)
        """
        if not self.training:
            return x
        noise = torch.randn_like(x) * self.weak_noise_std
        return x + noise

    def strong_augment(self, x):
        """
        强增强：时序遮蔽 + 幅度缩放 + 较强噪声。
        x: (batch, seq_len, input_dim)
        """
        if not self.training:
            return x
        batch, seq_len, input_dim = x.shape

        # ① 时序窗口遮蔽：随机选取连续时间步置零
        x_aug = x.clone()
        n_mask = max(1, int(seq_len * self.mask_ratio))
        for b in range(batch):
            start = torch.randint(0, max(1, seq_len - n_mask), (1,)).item()
            x_aug[b, start:start + n_mask, :] = 0.0

        # ② 幅度随机缩放：每个样本独立采样缩放因子
        scale = (
            torch.rand(batch, 1, 1, device=x.device)
            * (self.scale_high - self.scale_low)
            + self.scale_low
        )
        x_aug = x_aug * scale

        # ③ 叠加强噪声
        noise = torch.randn_like(x_aug) * self.strong_noise_std
        return x_aug + noise


# ══════════════════════════════════════════════════════════════════════════════
# 2. 时序 CNN 特征提取器
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalCNNExtractor(nn.Module):
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
        x   = x.permute(0, 2, 1)      # (batch, input_dim, seq_len)
        out = self.net(x)             # (batch, feat_dim, seq_len)
        return out.mean(dim=-1)       # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 分布对齐伪标签生成（Distribution-Aligned Pseudo-Labels）
#
#    AdaMatch 的核心创新（适配回归任务）：
#      原始 AdaMatch 针对分类任务对预测概率分布做对齐；
#      本实现适配回归任务，改为对预测值的均值和方差做对齐：
#
#      μ_src, σ_src — 源域弱增强样本预测值的均值和标准差
#      μ_tgt, σ_tgt — 目标域弱增强样本预测值的均值和标准差
#
#      对齐修正公式：
#        pseudo_label = (pred_tgt_weak - μ_tgt) / (σ_tgt + ε) * σ_src + μ_src
#
#      含义：将目标域预测值归一化到源域的预测分布尺度，
#            消除域间系统性预测偏移（如源域整体销量高于目标域）。
# ══════════════════════════════════════════════════════════════════════════════

def distribution_aligned_pseudo_labels(pred_src_weak, pred_tgt_weak, eps=1e-8):
    """
    生成分布对齐后的目标域伪标签。

    参数：
        pred_src_weak: (batch_s, 1) 源域弱增强样本预测值
        pred_tgt_weak: (batch_t, 1) 目标域弱增强样本预测值
        eps:           数值稳定项
    返回：
        pseudo_labels: (batch_t, 1) 分布对齐后的伪标签
        confidence:    (batch_t,)   置信度（预测值的稳定性估计）
    """
    # 源域预测分布统计量
    mu_src  = pred_src_weak.mean()
    std_src = pred_src_weak.std().clamp(min=eps)

    # 目标域预测分布统计量
    mu_tgt  = pred_tgt_weak.mean()
    std_tgt = pred_tgt_weak.std().clamp(min=eps)

    # 分布对齐：将目标域预测归一化到源域分布尺度
    pred_tgt_norm   = (pred_tgt_weak - mu_tgt) / std_tgt   # 标准化
    pseudo_labels   = pred_tgt_norm * std_src + mu_src      # 反归一化到源域尺度

    # 置信度估计：偏离批次均值越远，置信度越低
    # 用到批次内相对稳定性：越接近均值的预测越可靠
    deviation    = torch.abs(pred_tgt_weak - mu_tgt) / std_tgt   # 标准化偏差
    confidence   = torch.exp(-deviation.squeeze(-1))              # (batch_t,) ∈ (0,1]

    return pseudo_labels.detach(), confidence.detach()


# ══════════════════════════════════════════════════════════════════════════════
# 4. 相对置信度阈值过滤（Relative Confidence Thresholding）
#
#    AdaMatch 的第二个关键创新：
#      不使用固定阈值，而是基于批次内置信度分布的相对阈值：
#        threshold = mean(confidence) * τ
#      其中 τ ∈ (0, 1] 为超参数（默认 0.8）
#
#      优势：自适应批次内样本质量，避免训练初期因模型不成熟
#            导致所有伪标签被过滤（固定阈值的常见问题）。
# ══════════════════════════════════════════════════════════════════════════════

def relative_confidence_mask(confidence, tau=0.8):
    """
    生成相对置信度掩码。

    参数：
        confidence: (batch,) 置信度分数
        tau:        相对阈值系数
    返回：
        mask: (batch,) bool 掩码，True 表示该样本伪标签可信
    """
    threshold = confidence.mean() * tau
    return confidence >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# 5. 销售预测头（与所有对比基线保持一致）
# ══════════════════════════════════════════════════════════════════════════════

class LabelPredictor(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Linear(feat_dim + 1, 64),
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
        x        = torch.cat([feat, history_mean], dim=-1)
        residual = self.residual_net(x)
        return self.output_scale(history_mean + residual)


# ══════════════════════════════════════════════════════════════════════════════
# 6. AdaMatch 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    AdaMatch 完整实现（HierDA 对比基线）。

    两个损失项（均挂载到 extra_loss）：
      Loss U — 无监督一致性损失：
               强增强样本预测 vs 分布对齐伪标签，
               仅对置信度超过相对阈值的样本计算
      （主损失 Loss P 由训练循环在源域预测上计算）

    forward 接口与所有对比基线完全兼容：
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

        hidden_dim        = getattr(configs, 'd_model',           128)
        feat_dim          = getattr(configs, 'feat_dim',          128)
        dropout           = getattr(configs, 'dropout',           0.1)
        kernel_size       = getattr(configs, 'cnn_kernel',        3)
        num_layers        = getattr(configs, 'cnn_layers',        3)
        weak_noise_std    = getattr(configs, 'weak_noise_std',    0.01)
        strong_noise_std  = getattr(configs, 'strong_noise_std',  0.05)
        mask_ratio        = getattr(configs, 'aug_mask_ratio',    0.2)
        scale_range       = getattr(configs, 'aug_scale_range',   (0.8, 1.2))
        self.tau          = getattr(configs, 'conf_tau',          0.8)

        # 无监督一致性损失权重（μ_u，AdaMatch 原文符号）
        self.w_unsup = getattr(configs, 'w_unsup', 0.1)

        # 时序 CNN 特征提取器（源域、目标域共用权重）
        self.feature_extractor = TemporalCNNExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # 随机时序增强模块
        self.augmenter = RandomTemporalAugmentation(
            weak_noise_std=weak_noise_std,
            strong_noise_std=strong_noise_std,
            mask_ratio=mask_ratio,
            scale_range=scale_range,
        )

        # 销售预测头
        self.label_predictor = LabelPredictor(feat_dim, dropout).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        """
        参数（与 HierDA 保持一致）：
          x_enc    — 源域序列  (batch, seq_len, input_dim)
          x_target — 目标域序列 (batch, seq_len, input_dim)
                     若为 None，退化为无域适应

        返回：
          pred — (batch, pred_len, input_dim)，基于目标域特征的预测
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── Step 1：弱增强 → 生成伪标签 ──────────────────────────────────────
        src_weak = self.augmenter.weak_augment(source_data)   # (batch_s, seq_len, D)
        tgt_weak = self.augmenter.weak_augment(target_data)   # (batch_t, seq_len, D)

        src_weak_feat = self.feature_extractor(src_weak)      # (batch_s, feat_dim)
        tgt_weak_feat = self.feature_extractor(tgt_weak)      # (batch_t, feat_dim)

        src_history_mean = source_data[:, :, 0].mean(dim=1, keepdim=True)
        tgt_history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)

        pred_src_weak = self.label_predictor(src_weak_feat, src_history_mean)  # (batch_s, 1)
        pred_tgt_weak = self.label_predictor(tgt_weak_feat, tgt_history_mean)  # (batch_t, 1)

        # ── Step 2：分布对齐伪标签 ────────────────────────────────────────────
        pseudo_labels, confidence = distribution_aligned_pseudo_labels(
            pred_src_weak, pred_tgt_weak
        )
        # 相对置信度掩码：过滤低质量伪标签
        mask = relative_confidence_mask(confidence, tau=self.tau)   # (batch_t,) bool

        # ── Step 3：强增强 → 一致性训练 ───────────────────────────────────────
        tgt_strong      = self.augmenter.strong_augment(target_data)    # (batch_t, seq_len, D)
        tgt_strong_feat = self.feature_extractor(tgt_strong)            # (batch_t, feat_dim)
        pred_tgt_strong = self.label_predictor(
            tgt_strong_feat, tgt_history_mean
        )                                                               # (batch_t, 1)

        # ── Step 4：无监督一致性损失（Loss U） ───────────────────────────────
        # 仅对置信度足够高的样本计算损失
        if mask.sum() > 0:
            loss_u = F.mse_loss(
                pred_tgt_strong[mask],
                pseudo_labels[mask],
                reduction='mean'
            )
        else:
            # 全批次置信度过低时跳过（避免空张量的梯度问题）
            loss_u = torch.tensor(0.0, device=source_data.device, requires_grad=True)

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = self.w_unsup * loss_u

        # ── Step 5：目标域预测（用于主损失和最终推理） ────────────────────────
        # 推理时直接用原始目标域特征（不加增强）
        target_feat  = self.feature_extractor(target_data)             # (batch_t, feat_dim)
        history_mean = tgt_history_mean                                # (batch_t, 1)
        pred_value   = self.label_predictor(target_feat, history_mean) # (batch_t, 1)
        pred_value   = pred_value.expand(-1, self.pred_len)            # (batch_t, pred_len)

        # 补零对齐其余特征维度，保持输出形状与 HierDA 一致
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device,
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)  # (batch, pred_len, D)
        return pred