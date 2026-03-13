"""
DeepCoral.py —— Deep CORAL (Correlation Alignment) 销售预测模型

作为 HierDA 的对比基线，实现 Deep CORAL 框架：
  - 共享特征提取器（Feature Extractor，源域和目标域共用权重）
  - CORAL Loss：对齐源域与目标域特征的二阶统计量（协方差矩阵）
  - 销售预测头（Label Predictor，Loss P）

与 DANN 的核心区别：
  - DANN 通过 GRL + 域分类器做对抗对齐（一阶分布对齐）
  - Deep CORAL 直接最小化源域与目标域特征协方差矩阵之差（二阶统计对齐）
  - 无需对抗训练，训练更稳定

接口与 HierDA / DANN 完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 0. CORAL Loss：最小化源域与目标域特征协方差矩阵之差
# ══════════════════════════════════════════════════════════════════════════════

def coral_loss(source_feat, target_feat):
    """
    计算源域与目标域特征的 CORAL Loss。

    CORAL Loss = (1 / 4d^2) * ||C_s - C_t||_F^2
      C_s, C_t — 源域/目标域特征协方差矩阵，shape: (feat_dim, feat_dim)
      ||·||_F   — Frobenius 范数

    参数：
        source_feat: (batch_s, feat_dim)
        target_feat: (batch_t, feat_dim)
    返回：
        scalar tensor
    """
    d = source_feat.shape[1]

    # 去均值
    src = source_feat - source_feat.mean(dim=0, keepdim=True)
    tgt = target_feat - target_feat.mean(dim=0, keepdim=True)

    # 协方差矩阵（无偏估计）
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]

    cov_src = (src.T @ src) / max(n_s - 1, 1)   # (d, d)
    cov_tgt = (tgt.T @ tgt) / max(n_t - 1, 1)   # (d, d)

    # Frobenius 范数平方，归一化
    loss = torch.norm(cov_src - cov_tgt, p='fro') ** 2 / (4 * d * d)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 1. 共享特征提取器
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
#    结构：两层线性 + LayerNorm + ReLU，时序维度做均值池化
#    （与 DANN 保持一致，确保对比实验中特征提取能力相同）
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out = self.net(x)        # (batch, seq_len, feat_dim)
        return out.mean(dim=1)   # (batch, feat_dim)  时序均值池化


# ══════════════════════════════════════════════════════════════════════════════
# 2. 销售预测头（与 HierDA / DANN 保持一致）
#    history_mean 作为先验基准，预测残差后相加
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
        x = torch.cat([feat, history_mean], dim=-1)   # (batch, feat_dim+1)
        residual = self.residual_net(x)                # (batch, 1)
        return self.output_scale(history_mean + residual)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Deep CORAL 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    Deep CORAL 完整实现（HierDA 对比基线）。

    与 HierDA / DANN 的核心区别：
      ┌──────────────┬───────────────────────────────┬──────────────────────────┐
      │              │ HierDA                        │ DANN          │ DeepCORAL│
      ├──────────────┼───────────────────────────────┼───────────────┼──────────┤
      │ 域对齐方式   │ Transport Map（二阶）+ GRL    │ GRL 对抗      │ CORAL    │
      │              │ 多粒度时频特征                 │              │ （二阶） │
      │ 特征提取     │ SetFeat4 多粒度时频            │ 共享 MLP      │ 共享 MLP │
      │ 数据补全     │ VAE                           │ ✗             │ ✗        │
      │ 对抗训练     │ 有（GRL）                     │ 有（GRL）     │ 无       │
      └──────────────┴───────────────────────────────┴───────────────┴──────────┘

    forward 接口与 HierDA / DANN / exp_long_term_forecasting.py 完全兼容：
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
        self.extra_loss = None   # 训练循环检测此字段并累加到主损失

        hidden_dim   = getattr(configs, 'd_model',      128)
        feat_dim     = getattr(configs, 'feat_dim',     128)
        dropout      = getattr(configs, 'dropout',      0.1)
        # coral_weight：CORAL Loss 相对于预测 Loss 的权重系数
        self.coral_weight = getattr(configs, 'coral_weight', 0.1)

        # 共享特征提取器（源域和目标域共用同一组权重）
        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, feat_dim, dropout
        ).to(device)

        # 销售预测头
        self.label_predictor = LabelPredictor(feat_dim, dropout).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        """
        参数（与 HierDA 保持一致）：
          x_enc    — 源域序列  (batch, seq_len, input_dim)
          x_target — 目标域序列 (batch, seq_len, input_dim)
                     若为 None，退化为无域适应（仅用 x_enc）

        返回：
          pred — (batch, pred_len, input_dim)
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── 共享特征提取 ──────────────────────────────────────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch, feat_dim)

        # ── CORAL Loss：对齐二阶统计量 ────────────────────────────────────────
        loss_coral = coral_loss(source_feat, target_feat)
        self.extra_loss = self.coral_weight * loss_coral

        # ── 预测：基于目标域特征 ──────────────────────────────────────────────
        history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)   # (batch, 1)
        pred_value   = self.label_predictor(target_feat, history_mean)   # (batch, 1)
        pred_value   = pred_value.expand(-1, self.pred_len)              # (batch, pred_len)

        # 补零对齐其余特征维度，保持输出形状与 HierDA 一致
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)   # (batch, pred_len, input_dim)
        return pred