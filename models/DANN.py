"""
DANN.py —— Domain-Adversarial Neural Network (DANN) 销售预测模型

作为 HierDA 的对比基线，实现经典 DANN 框架：
  - 共享特征提取器（Feature Extractor）
  - 销售预测头（Label Predictor，Loss P）
  - 域分类器 + 梯度反转层（Domain Classifier w/ GRL，Loss C）

接口与 HierDA 完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
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
# 1. 共享特征提取器
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
#    结构：两层线性 + LayerNorm + ReLU，时序维度做均值池化
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
        out = self.net(x)           # (batch, seq_len, feat_dim)
        return out.mean(dim=1)      # (batch, feat_dim)  时序均值池化


# ══════════════════════════════════════════════════════════════════════════════
# 2. 销售预测头（Label Predictor）
#    history_mean 作为先验基准，预测残差后相加，与 HierDA 保持一致
# ══════════════════════════════════════════════════════════════════════════════

class LabelPredictor(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        # feat_dim + 1（history_mean）→ 残差预测
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
        """
        feat:         (batch, feat_dim)
        history_mean: (batch, 1)
        返回:         (batch, 1)
        """
        x = torch.cat([feat, history_mean], dim=-1)   # (batch, feat_dim+1)
        residual = self.residual_net(x)                # (batch, 1)
        return self.output_scale(history_mean + residual)


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
# 4. DANN 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    DANN 完整实现（HierDA 对比基线）。

    与 HierDA 的核心区别：
      - 无 VAE 数据补全
      - 无多粒度时频特征提取（SetFeat4）
      - 无层级 Transport Map 对齐（Loss F）
      - 只有经典 DANN：共享特征提取器 + GRL 域分类器

    forward 接口与 HierDA / exp_long_term_forecasting.py 完全兼容：
      model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
        batch_x_src  → x_enc    (源域序列)
        batch_x      → x_target (目标域序列)
    """

    def __init__(self, configs):
        super().__init__()

        seq_len   = configs.seq_len
        input_dim = configs.enc_in
        device    = 'cuda' if configs.use_gpu else 'cpu'

        self.device    = device
        self.pred_len  = configs.pred_len
        self.label_len = configs.label_len
        self.input_dim = input_dim
        self.extra_loss = None   # 与 HierDA 保持一致，训练循环会检测此字段

        # 超参数（可通过 configs 传入，保留默认值兼容旧配置）
        hidden_dim = getattr(configs, 'd_model',   128)
        feat_dim   = getattr(configs, 'feat_dim',  128)
        dropout    = getattr(configs, 'dropout',   0.1)
        grl_alpha  = getattr(configs, 'grl_alpha', 1.0)

        # 共享特征提取器（源域和目标域共用同一组权重）
        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, feat_dim, dropout
        ).to(device)

        # 销售预测头
        self.label_predictor = LabelPredictor(feat_dim, dropout).to(device)

        # 域分类器（含 GRL）
        self.domain_classifier = DomainClassifier(feat_dim, grl_alpha).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        """
        参数（与 HierDA 保持一致）：
          x_enc    — 源域序列  (batch, seq_len, input_dim)
          x_target — 目标域序列 (batch, seq_len, input_dim)
                     若为 None，退化为无域适应（仅用 x_enc 预测）

        返回：
          pred — (batch, pred_len, input_dim)，与训练循环期望的输出格式一致
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── 特征提取（源域 & 目标域共享提取器） ──────────────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch, feat_dim)

        # ── 域分类器对抗损失（Loss C） ────────────────────────────────────────
        domain_feat  = torch.cat([source_feat, target_feat], dim=0)
        domain_label = torch.cat([
            torch.zeros(source_feat.shape[0]),
            torch.ones(target_feat.shape[0])
        ]).to(source_data.device)
        domain_pred = self.domain_classifier(domain_feat).squeeze(-1)
        loss_c = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        # 辅助损失挂载到 extra_loss（训练循环自动累加到 Loss P）
        self.extra_loss = loss_c * 0.1

        # ── 预测：基于目标域特征 ──────────────────────────────────────────────
        # 历史均值作为先验基准（与 HierDA 保持一致）
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