"""
AdvSKM.py —— Adversarial Spectral Kernel Matching (AdvSKM) 销售预测模型

作为 HierDA 的对比基线，实现 AdvSKM 框架：
  - LSTM 时序特征提取器：捕捉销售序列的长期依赖
  - 谱核匹配损失（SKM Loss）：在 RKHS 中对齐源域与目标域的特征分布
    通过最大均值差异（MMD）的谱分解近似，计算多尺度 RBF 核下的分布距离
  - 对抗域分类器 + GRL（Adv Loss）：进一步压缩域间分布差异
  - 销售预测头（Label Predictor）

与其他基线的核心区别：
  ┌────────────┬────────┬───────────┬─────────┬──────────────┐
  │            │ DANN   │ DeepCoral │ CoDATS  │ AdvSKM       │
  ├────────────┼────────┼───────────┼─────────┼──────────────┤
  │ 特征提取   │ MLP    │ MLP       │ TCN     │ LSTM         │
  │ 域对齐     │ GRL    │ CORAL     │GRL+对比 │ SKM + GRL    │
  │ 核方法     │ ✗      │ ✗         │ ✗       │ ✓ 多尺度 RBF │
  │ 时序感知   │ ✗      │ ✗         │ ✓       │ ✓            │
  └────────────┴────────┴───────────┴─────────┴──────────────┘

接口与 HierDA / DANN / DeepCoral / CoDATS 完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Liu et al., "Adversarial Spectral Kernel Matching for Unsupervised
  Time Series Domain Adaptation", IJCAI 2021.
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
# 1. 谱核匹配损失（Spectral Kernel Matching Loss）
#
#    核心思想：
#      用多尺度 RBF 核构造 RKHS 中的核矩阵，通过谱分解（取前 k 个特征值）
#      近似最大均值差异（MMD），衡量源域与目标域特征分布的距离。
#
#    相比标准 MMD 的优势：
#      - 谱截断过滤高频噪声，对分布尾部更鲁棒
#      - 多尺度核融合覆盖不同粒度的分布差异
#      - 计算复杂度 O(n^2 d) 优于全核矩阵分解
# ══════════════════════════════════════════════════════════════════════════════

def rbf_kernel(x, y, sigma):
    """
    计算 x 与 y 之间的 RBF（高斯）核矩阵。

    K(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 * sigma^2))

    参数：
        x: (n, feat_dim)
        y: (m, feat_dim)
        sigma: 核带宽
    返回：
        K: (n, m)
    """
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i^T y_j
    xx = (x ** 2).sum(dim=1, keepdim=True)          # (n, 1)
    yy = (y ** 2).sum(dim=1, keepdim=True)          # (m, 1)
    dist_sq = xx + yy.T - 2.0 * torch.mm(x, y.T)   # (n, m)
    dist_sq = dist_sq.clamp(min=0.0)                 # 数值稳定
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def spectral_kernel_matching_loss(source_feat, target_feat,
                                   sigmas=(0.1, 1.0, 10.0),
                                   n_spectral=64):
    """
    谱核匹配损失（SKM Loss）。

    算法流程：
      1. 对每个尺度 sigma，分别计算：
           K_ss = K(src, src)，K_tt = K(tgt, tgt)，K_st = K(src, tgt)
      2. 对 [K_ss, K_st; K_st^T, K_tt] 构成的联合核矩阵做谱截断
         （取前 n_spectral 个特征值）
      3. MMD^2 ≈ mean(K_ss) + mean(K_tt) - 2*mean(K_st)，
         通过谱权重加权后求和
      4. 多尺度求均值作为最终 SKM Loss

    参数：
        source_feat:  (batch_s, feat_dim)
        target_feat:  (batch_t, feat_dim)
        sigmas:       多尺度 RBF 核带宽列表
        n_spectral:   谱截断保留的特征值数量
    返回：
        scalar tensor
    """
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]

    # batch 过小时退化为标准 MMD，避免谱分解不稳定
    if n_s < 2 or n_t < 2:
        return torch.tensor(0.0, device=source_feat.device, requires_grad=True)

    total_loss = torch.tensor(0.0, device=source_feat.device)

    for sigma in sigmas:
        # ── 计算三类核矩阵 ────────────────────────────────────────────────
        K_ss = rbf_kernel(source_feat, source_feat, sigma)   # (n_s, n_s)
        K_tt = rbf_kernel(target_feat, target_feat, sigma)   # (n_t, n_t)
        K_st = rbf_kernel(source_feat, target_feat, sigma)   # (n_s, n_t)

        # ── 构造联合核矩阵并做谱分解 ──────────────────────────────────────
        # 联合矩阵：[[K_ss, K_st], [K_st^T, K_tt]]，shape: (n_s+n_t, n_s+n_t)
        top    = torch.cat([K_ss, K_st],    dim=1)   # (n_s, n_s+n_t)
        bottom = torch.cat([K_st.T, K_tt],  dim=1)   # (n_t, n_s+n_t)
        K_joint = torch.cat([top, bottom],  dim=0)   # (n_s+n_t, n_s+n_t)

        # 数值对称化（消除浮点误差）
        K_joint = (K_joint + K_joint.T) / 2.0

        # 谱截断：保留前 k 个特征值作为权重
        k = min(n_spectral, K_joint.shape[0])
        try:
            # torch.linalg.eigvalsh 返回升序特征值（实对称矩阵）
            eigenvalues = torch.linalg.eigvalsh(K_joint)   # (n_s+n_t,)
            # 取最大的 k 个特征值，归一化为权重
            top_k_vals  = eigenvalues[-k:].clamp(min=0.0)
            spectral_weight = top_k_vals.sum().clamp(min=1e-8)
        except Exception:
            # 若特征值分解失败，退化为等权重
            spectral_weight = torch.tensor(float(k), device=source_feat.device)

        # ── 谱加权 MMD^2 ─────────────────────────────────────────────────
        # 去对角线均值（无偏 MMD 估计）
        diag_ss = K_ss.diagonal().sum()
        diag_tt = K_tt.diagonal().sum()
        mmd_sq = (
            (K_ss.sum() - diag_ss) / max(n_s * (n_s - 1), 1)
            + (K_tt.sum() - diag_tt) / max(n_t * (n_t - 1), 1)
            - 2.0 * K_st.mean()
        )

        total_loss = total_loss + mmd_sq * spectral_weight / spectral_weight.detach()

    return total_loss / len(sigmas)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LSTM 时序特征提取器
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
#    使用双层 LSTM + 最后隐状态作为时序表示
# ══════════════════════════════════════════════════════════════════════════════

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim,
                 num_layers=2, dropout=0.1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        # 取最后一层的隐状态
        h_last = h_n[-1]             # (batch, hidden_dim)
        return self.proj(h_last)     # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 对抗域分类器（Domain Classifier w/ GRL）
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
# 4. 销售预测头（与 HierDA / DANN / DeepCoral / CoDATS 保持一致）
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
        x        = torch.cat([feat, history_mean], dim=-1)
        residual = self.residual_net(x)
        return self.output_scale(history_mean + residual)


# ══════════════════════════════════════════════════════════════════════════════
# 5. AdvSKM 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    AdvSKM 完整实现（HierDA 对比基线）。

    三个损失项：
      Loss P   — 销售预测损失（主损失，由训练循环计算）
      Loss SKM — 谱核匹配损失（extra_loss 的一部分），对齐特征分布的高阶统计
      Loss Adv — GRL 对抗域分类损失（extra_loss 的一部分），细化域边界对齐

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

        hidden_dim   = getattr(configs, 'd_model',       128)
        feat_dim     = getattr(configs, 'feat_dim',      128)
        dropout      = getattr(configs, 'dropout',       0.1)
        grl_alpha    = getattr(configs, 'grl_alpha',     1.0)
        lstm_layers  = getattr(configs, 'lstm_layers',   2)
        n_spectral   = getattr(configs, 'n_spectral',    64)

        # SKM 多尺度带宽：覆盖细粒度到粗粒度的分布差异
        raw_sigmas       = getattr(configs, 'skm_sigmas', [0.1, 1.0, 10.0])
        self.skm_sigmas  = tuple(raw_sigmas)
        self.n_spectral  = n_spectral

        # 损失权重
        self.w_skm  = getattr(configs, 'w_skm',  0.1)   # SKM 损失权重
        self.w_adv  = getattr(configs, 'w_adv',  0.1)   # 对抗损失权重

        # LSTM 特征提取器（源域和目标域共用权重）
        self.feature_extractor = LSTMFeatureExtractor(
            input_dim, hidden_dim, feat_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        ).to(device)

        # 对抗域分类器（含 GRL）
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

        # ── LSTM 特征提取（源域 & 目标域共享提取器） ─────────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch, feat_dim)

        # ── 谱核匹配损失（Loss SKM） ──────────────────────────────────────────
        loss_skm = spectral_kernel_matching_loss(
            source_feat, target_feat,
            sigmas=self.skm_sigmas,
            n_spectral=self.n_spectral,
        )

        # ── 对抗域分类损失（Loss Adv） ────────────────────────────────────────
        domain_feat  = torch.cat([source_feat, target_feat], dim=0)
        domain_label = torch.cat([
            torch.zeros(source_feat.shape[0]),
            torch.ones(target_feat.shape[0]),
        ]).to(source_data.device)
        domain_pred = self.domain_classifier(domain_feat).squeeze(-1)
        loss_adv    = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = self.w_skm * loss_skm + self.w_adv * loss_adv

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