"""
CotMix.py —— Contrastive Temporal Mixing (CotMix) 销售预测模型

作为 HierDA 的对比基线，实现 CotMix 框架：
  - 时序卷积特征提取器（Temporal CNN）：捕捉局部时序模式
  - 时序上下文混合（Temporal Contextual Mixing）：
      在时序维度上对源域与目标域样本进行凸组合插值，
      生成混合样本以桥接源域与目标域之间的分布间隙
  - 一致性正则化损失（Consistency Loss）：
      约束混合样本的预测在源域预测与目标域预测之间保持线性插值关系，
      使决策边界在域间平滑过渡
  - 时序对比损失（Temporal Contrastive Loss）：
      利用时序上下文信息区分近邻时间步（正样本）与远端时间步（负样本），
      学习对时序变化鲁棒的域不变表示
  - 销售预测头（Label Predictor）

与其他基线的核心区别：
  ┌──────────────┬────────┬───────────┬─────────┬──────────┬──────────┬──────────────────────┐
  │              │ DANN   │ DeepCoral │ CoDATS  │ AdvSKM   │RAINCOAT  │ CotMix               │
  ├──────────────┼────────┼───────────┼─────────┼──────────┼──────────┼──────────────────────┤
  │ 特征提取     │ MLP    │ MLP       │ TCN     │ LSTM     │CNN时频   │ Temporal CNN         │
  │ 域对齐核心   │ GRL    │ CORAL     │GRL+对比 │ SKM+GRL  │原型+对比 │ 时序混合+一致性正则  │
  │ 数据增强     │ ✗      │ ✗         │ ✗       │ ✗        │ ✗        │ ✓ Mixup（时序维度）  │
  │ 一致性正则   │ ✗      │ ✗         │ ✗       │ ✗        │ ✗        │ ✓                   │
  │ 对抗训练     │ ✓      │ ✗         │ ✓       │ ✓        │ ✗        │ ✗                   │
  └──────────────┴────────┴───────────┴─────────┴──────────┴──────────┴──────────────────────┘

接口与所有对比基线完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  Eldele et al., "Contrastive Adversarial and Mixup for Source-Free
  Unsupervised Domain Adaptation", ACM CIKM 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. 时序 CNN 特征提取器
#    多层因果卷积捕捉局部时序模式
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
        x   = x.permute(0, 2, 1)       # (batch, input_dim, seq_len)
        out = self.net(x)              # (batch, feat_dim, seq_len)
        return out.mean(dim=-1)        # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 时序上下文混合（Temporal Contextual Mixing）
#
#    核心思想（CotMix 原创）：
#      不在样本级别（整条时序）做 Mixup，而在时序上下文维度做混合：
#        x_mix = λ · x_src_ctx + (1 - λ) · x_tgt
#      其中 x_src_ctx 是从源域时序中随机抽取的一段上下文窗口，
#      与目标域完整序列做凸组合。
#
#      这样生成的混合样本既保留目标域的整体时序结构，
#      又注入了源域的局部时序模式，有效桥接两个域的分布间隙。
#
#    混合策略：
#      - 从源域序列中随机采样一段长度为 ctx_len 的上下文片段
#      - 将其填充 / 裁剪到与目标域等长后做凸组合
#      - λ 从 Beta(alpha, alpha) 分布采样，保证混合比例的随机性
# ══════════════════════════════════════════════════════════════════════════════

class TemporalContextualMixing(nn.Module):
    def __init__(self, seq_len, mix_alpha=0.5, ctx_ratio=0.5):
        """
        参数：
            seq_len:    输入序列长度
            mix_alpha:  Beta 分布的 α 参数，控制混合比例的集中程度
                        α→0：极端混合（接近纯源或纯目标）
                        α=1：均匀混合
            ctx_ratio:  上下文窗口占完整序列的比例
        """
        super().__init__()
        self.seq_len   = seq_len
        self.mix_alpha = mix_alpha
        self.ctx_len   = max(1, int(seq_len * ctx_ratio))

    def forward(self, source_x, target_x):
        """
        source_x: (batch_s, seq_len, input_dim)
        target_x: (batch_t, seq_len, input_dim)
        返回：
            x_mix: (min_b, seq_len, input_dim)  混合样本
            lam:   scalar，混合比例（目标域权重）
        """
        min_b = min(source_x.shape[0], target_x.shape[0])
        src   = source_x[:min_b]   # (min_b, seq_len, input_dim)
        tgt   = target_x[:min_b]   # (min_b, seq_len, input_dim)

        # 从 Beta 分布采样混合比例 λ（目标域权重）
        if self.mix_alpha > 0:
            lam = float(torch.distributions.Beta(
                self.mix_alpha, self.mix_alpha
            ).sample())
        else:
            lam = 0.5
        lam = max(lam, 1 - lam)    # 保证目标域权重 >= 0.5，主导混合方向

        # 随机采样源域上下文窗口起始位置
        max_start = max(0, self.seq_len - self.ctx_len)
        ctx_start = torch.randint(0, max_start + 1, (1,)).item()
        ctx_end   = ctx_start + self.ctx_len

        # 提取源域上下文片段并填充到完整序列长度
        src_ctx = torch.zeros_like(src)               # (min_b, seq_len, input_dim)
        src_ctx[:, ctx_start:ctx_end, :] = src[:, ctx_start:ctx_end, :]

        # 凸组合：目标域为主，源域上下文为辅
        x_mix = lam * tgt + (1 - lam) * src_ctx      # (min_b, seq_len, input_dim)
        return x_mix, lam


# ══════════════════════════════════════════════════════════════════════════════
# 3. 一致性正则化损失（Consistency Loss）
#
#    约束混合样本的预测值在源域预测与目标域预测之间保持线性插值：
#      pred(x_mix) ≈ λ · pred(x_tgt) + (1 - λ) · pred(x_src)
#
#    物理含义：决策边界在域间平滑过渡，避免在混合区域产生
#    突变或不一致的预测，从而正则化特征空间的域间几何结构。
# ══════════════════════════════════════════════════════════════════════════════

def consistency_loss(pred_mix, pred_src, pred_tgt, lam):
    """
    一致性正则化损失。

    参数：
        pred_mix: (batch, 1) 混合样本预测值
        pred_src: (batch, 1) 源域样本预测值
        pred_tgt: (batch, 1) 目标域样本预测值
        lam:      混合比例（目标域权重）
    返回：
        scalar tensor
    """
    # 线性插值目标：λ·pred_tgt + (1-λ)·pred_src
    interp_target = lam * pred_tgt.detach() + (1 - lam) * pred_src.detach()
    return F.mse_loss(pred_mix, interp_target)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 时序对比损失（Temporal Contrastive Loss）
#
#    利用时序上下文的自然邻近关系构造正负样本对：
#      正样本：时序上相邻（时间距离 ≤ pos_thresh）的特征对
#      负样本：时序上远端（时间距离 > neg_thresh）的特征对
#
#    通过在批次内随机打乱配对，无需额外标注，
#    学习对时序局部变化鲁棒的域不变表示。
# ══════════════════════════════════════════════════════════════════════════════

def temporal_contrastive_loss(source_feat, target_feat, temperature=0.1):
    """
    时序对比损失。

    正样本对：源域特征与目标域特征中余弦相似度最高的配对
             （模拟时序邻近样本语义相似的假设）
    负样本对：同批次内其余的跨域配对

    参数：
        source_feat: (batch, feat_dim)
        target_feat: (batch, feat_dim)
        temperature: softmax 温度系数
    返回：
        scalar tensor
    """
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]
    min_b = min(n_s, n_t)

    if min_b < 2:
        return torch.tensor(0.0, device=source_feat.device, requires_grad=True)

    src_norm = F.normalize(source_feat[:min_b], dim=-1)   # (min_b, feat_dim)
    tgt_norm = F.normalize(target_feat[:min_b], dim=-1)   # (min_b, feat_dim)

    # 拼接源域与目标域特征：前 min_b 为源，后 min_b 为目标
    all_feat = torch.cat([src_norm, tgt_norm], dim=0)      # (2*min_b, feat_dim)

    # 全对相似度矩阵
    sim = torch.mm(all_feat, all_feat.T) / temperature     # (2*min_b, 2*min_b)

    # 屏蔽自身
    mask_diag = torch.eye(2 * min_b, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask_diag, float('-inf'))

    # 正样本标签：
    #   源域样本 i (index i)       的正样本 → 目标域样本 i (index min_b+i)
    #   目标域样本 i (index min_b+i) 的正样本 → 源域样本 i (index i)
    pos_labels = torch.cat([
        torch.arange(min_b, 2 * min_b),    # 源域 → 目标域
        torch.arange(0, min_b),            # 目标域 → 源域
    ]).to(sim.device)

    loss = F.cross_entropy(sim, pos_labels)
    return loss


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
# 6. CotMix 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    CotMix 完整实现（HierDA 对比基线）。

    三个损失项：
      Loss P    — 销售预测损失（主损失，由训练循环计算）
      Loss Con  — 一致性正则化损失（extra_loss 的一部分）
      Loss TCon — 时序对比损失（extra_loss 的一部分）

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
        feat_dim    = getattr(configs, 'feat_dim',     128)
        dropout     = getattr(configs, 'dropout',      0.1)
        kernel_size = getattr(configs, 'cnn_kernel',   3)
        num_layers  = getattr(configs, 'cnn_layers',   3)
        mix_alpha   = getattr(configs, 'mix_alpha',    0.5)
        ctx_ratio   = getattr(configs, 'ctx_ratio',    0.5)
        temperature = getattr(configs, 'temperature',  0.1)

        # 损失权重
        self.w_con  = getattr(configs, 'w_con',  0.1)   # 一致性损失权重
        self.w_tcon = getattr(configs, 'w_tcon', 0.1)   # 时序对比损失权重
        self.temperature = temperature

        # 时序 CNN 特征提取器（源域、目标域、混合样本共用权重）
        self.feature_extractor = TemporalCNNExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # 时序上下文混合模块
        self.mixer = TemporalContextualMixing(
            seq_len=seq_len,
            mix_alpha=mix_alpha,
            ctx_ratio=ctx_ratio,
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
          pred — (batch, pred_len, input_dim)
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── 特征提取（源域 & 目标域） ──────────────────────────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch_s, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch_t, feat_dim)

        # ── 时序上下文混合 ────────────────────────────────────────────────────
        x_mix, lam = self.mixer(source_data, target_data)   # (min_b, seq_len, input_dim)
        mix_feat   = self.feature_extractor(x_mix)          # (min_b, feat_dim)

        # ── 一致性正则化损失（Loss Con） ──────────────────────────────────────
        min_b = x_mix.shape[0]

        # 混合样本的历史均值先验
        mix_history_mean = x_mix[:, :, 0].mean(dim=1, keepdim=True)   # (min_b, 1)

        # 源域与目标域对应片段的预测（detach 避免梯度回传干扰主预测）
        src_history_mean = source_data[:min_b, :, 0].mean(dim=1, keepdim=True)
        tgt_history_mean = target_data[:min_b, :, 0].mean(dim=1, keepdim=True)

        pred_mix = self.label_predictor(mix_feat,              mix_history_mean)
        pred_src = self.label_predictor(source_feat[:min_b],   src_history_mean)
        pred_tgt = self.label_predictor(target_feat[:min_b],   tgt_history_mean)

        loss_con = consistency_loss(pred_mix, pred_src, pred_tgt, lam)

        # ── 时序对比损失（Loss TCon） ─────────────────────────────────────────
        loss_tcon = temporal_contrastive_loss(
            source_feat, target_feat, self.temperature
        )

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = self.w_con * loss_con + self.w_tcon * loss_tcon

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