"""
RAINCOAT.py —— Domain Adaptation for Time Series via Frequency and Prototype Alignment

作为 HierDA 的对比基线，实现 RAINCOAT 框架：
  - 时域特征提取器（CNN）：捕捉局部时序模式
  - 频域特征提取器（FFT + CNN）：捕捉周期性与趋势信息
  - 时频联合表示：拼接时域与频域特征
  - 原型对齐损失（Prototype Alignment Loss）：
      在特征空间中构造源域类原型，约束目标域特征向最近原型聚拢
  - 跨域对比损失（Cross-Domain Contrastive Loss）：
      拉近源域与目标域中语义相似的样本，推远语义不同的样本
  - 销售预测头（Label Predictor）

与其他基线的核心区别：
  ┌────────────┬────────┬───────────┬─────────┬──────────┬──────────────────┐
  │            │ DANN   │ DeepCoral │ CoDATS  │ AdvSKM   │ RAINCOAT         │
  ├────────────┼────────┼───────────┼─────────┼──────────┼──────────────────┤
  │ 特征提取   │ MLP    │ MLP       │ TCN     │ LSTM     │ CNN(时域+频域)   │
  │ 域对齐     │ GRL    │ CORAL     │GRL+对比 │ SKM+GRL  │ 原型对齐+跨域对比│
  │ 频域建模   │ ✗      │ ✗         │ ✗       │ ✗        │ ✓ FFT           │
  │ 原型学习   │ ✗      │ ✗         │ ✗       │ ✗        │ ✓               │
  └────────────┴────────┴───────────┴─────────┴──────────┴──────────────────┘

接口与所有对比基线完全兼容：
  forward(x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None)
    x_enc    — 源域序列  (batch, seq_len, input_dim)
    x_target — 目标域序列 (batch, seq_len, input_dim)

参考文献：
  He et al., "DOMAIN ADAPTATION FOR TIME SERIES UNDER FEATURE AND LABEL SHIFTS",
  ICML 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. 时域 CNN 特征提取器
#    多层 1D 卷积捕捉局部时序模式，与频域提取器结构对称
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
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)          # (batch, input_dim, seq_len)
        out = self.net(x)               # (batch, feat_dim, seq_len)
        return out.mean(dim=-1)         # (batch, feat_dim) 时序均值池化


# ══════════════════════════════════════════════════════════════════════════════
# 2. 频域 CNN 特征提取器
#    先做 FFT 取幅度谱，再用 1D 卷积在频域上提取模式
#    FFT 幅度谱天然具有平移不变性，能更稳定地捕捉周期信号
#    输入：(batch, seq_len, input_dim)
#    输出：(batch, feat_dim)
# ══════════════════════════════════════════════════════════════════════════════

class FrequencyCNNExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim,
                 kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch  = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else feat_dim
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # FFT 幅度谱：沿时序维度做 FFT，取模得实值幅度
        xf = torch.abs(torch.fft.fft(x, dim=1))   # (batch, seq_len, input_dim)
        # 标准化：消除销售数据量级差异
        xf = (xf - xf.mean(dim=1, keepdim=True)) / (
            xf.std(dim=1, keepdim=True) + 1e-8
        )
        xf = xf.permute(0, 2, 1)                  # (batch, input_dim, seq_len)
        out = self.net(xf)                         # (batch, feat_dim, seq_len)
        return out.mean(dim=-1)                    # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 时频联合特征提取器
#    拼接时域与频域特征后经投影层压缩到统一 feat_dim
# ══════════════════════════════════════════════════════════════════════════════

class TimeFreqExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim,
                 kernel_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        self.temporal_extractor = TemporalCNNExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size, num_layers, dropout
        )
        self.freq_extractor = FrequencyCNNExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size, num_layers, dropout
        )
        # 融合投影：时域 feat_dim + 频域 feat_dim → feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x):
        t_feat = self.temporal_extractor(x)    # (batch, feat_dim)
        f_feat = self.freq_extractor(x)        # (batch, feat_dim)
        return self.fusion(
            torch.cat([t_feat, f_feat], dim=-1)
        )                                      # (batch, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 原型对齐损失（Prototype Alignment Loss）
#
#    核心思想（RAINCOAT 原创）：
#      在源域特征空间中构造 K 个原型（聚类中心），
#      约束目标域特征向最近原型聚拢，从而隐式地将目标域
#      特征拉向源域的语义簇，实现无监督的语义对齐。
#
#    计算流程：
#      1. 将源域特征按照量化（argmin 距离）分配到最近原型
#      2. 更新原型为所属特征的均值（soft EMA 更新）
#      3. 目标域特征到最近原型的平均距离作为对齐损失
# ══════════════════════════════════════════════════════════════════════════════

class PrototypeAlignmentLoss(nn.Module):
    def __init__(self, feat_dim, num_prototypes=8, momentum=0.9):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.momentum       = momentum
        # 原型用 buffer 存储（不参与梯度更新，通过 EMA 更新）
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(num_prototypes, feat_dim), dim=-1)
        )

    @torch.no_grad()
    def _update_prototypes(self, source_feat):
        """
        用源域特征的 EMA 更新原型。

        1. 计算每个源域特征到各原型的距离，分配到最近原型
        2. 对每个原型，求所属特征均值，做 EMA 动量更新
        """
        src_norm = F.normalize(source_feat, dim=-1)         # (n_s, feat_dim)
        # 距离矩阵：(n_s, num_prototypes)
        dists = torch.cdist(src_norm, self.prototypes)
        assign = dists.argmin(dim=-1)                        # (n_s,)

        for k in range(self.num_prototypes):
            mask = (assign == k)
            if mask.sum() == 0:
                continue
            cluster_mean = F.normalize(
                src_norm[mask].mean(dim=0), dim=-1
            )                                                # (feat_dim,)
            self.prototypes[k] = (
                self.momentum * self.prototypes[k]
                + (1 - self.momentum) * cluster_mean
            )
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    def forward(self, source_feat, target_feat):
        """
        source_feat: (batch_s, feat_dim)
        target_feat: (batch_t, feat_dim)
        返回: scalar tensor
        """
        # EMA 更新原型（仅用源域特征）
        self._update_prototypes(source_feat)

        # 目标域特征到最近原型的余弦距离
        tgt_norm = F.normalize(target_feat, dim=-1)          # (n_t, feat_dim)
        # 余弦相似度 → 距离 = 1 - similarity
        cos_sim  = torch.mm(tgt_norm, self.prototypes.T)     # (n_t, num_prototypes)
        # 每个目标域样本取最近原型的距离
        min_dist = 1.0 - cos_sim.max(dim=-1).values          # (n_t,)
        return min_dist.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5. 跨域对比损失（Cross-Domain Contrastive Loss）
#
#    RAINCOAT 的第二个关键对齐机制：
#      正样本对：源域与目标域中特征最相近的样本对（跨域语义匹配）
#      负样本对：源域与目标域中特征最远的样本对
#    通过贪心最近邻匹配构造正样本，无需标签。
# ══════════════════════════════════════════════════════════════════════════════

def cross_domain_contrastive_loss(source_feat, target_feat, temperature=0.1):
    """
    跨域对比损失。

    算法：
      1. 对源域每个样本，在目标域中找最近邻作为正样本
      2. 其余目标域样本作为负样本
      3. 计算 NT-Xent 对比损失

    参数：
        source_feat: (batch_s, feat_dim)
        target_feat: (batch_t, feat_dim)
        temperature: softmax 温度
    返回：
        scalar tensor
    """
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]
    if n_s < 1 or n_t < 2:
        return torch.tensor(0.0, device=source_feat.device, requires_grad=True)

    src_norm = F.normalize(source_feat, dim=-1)   # (n_s, feat_dim)
    tgt_norm = F.normalize(target_feat, dim=-1)   # (n_t, feat_dim)

    # 跨域余弦相似度矩阵：(n_s, n_t)
    sim_matrix = torch.mm(src_norm, tgt_norm.T) / temperature

    # 正样本：目标域中与当前源域样本最相近的样本（贪心最近邻）
    pos_idx = sim_matrix.argmax(dim=-1)            # (n_s,)

    # NT-Xent：以所有目标域样本为候选，正样本为最近邻
    loss = F.cross_entropy(sim_matrix, pos_idx)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 6. 销售预测头（与所有对比基线保持一致）
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
# 7. RAINCOAT 整体模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    RAINCOAT 完整实现（HierDA 对比基线）。

    三个损失项：
      Loss P    — 销售预测损失（主损失，由训练循环计算）
      Loss Proto— 原型对齐损失（extra_loss 的一部分）
      Loss Con  — 跨域对比损失（extra_loss 的一部分）

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

        hidden_dim      = getattr(configs, 'd_model',         128)
        feat_dim        = getattr(configs, 'feat_dim',        128)
        dropout         = getattr(configs, 'dropout',         0.1)
        kernel_size     = getattr(configs, 'cnn_kernel',      3)
        num_layers      = getattr(configs, 'cnn_layers',      3)
        num_prototypes  = getattr(configs, 'num_prototypes',  8)
        proto_momentum  = getattr(configs, 'proto_momentum',  0.9)
        temperature     = getattr(configs, 'temperature',     0.1)

        # 损失权重
        self.w_proto = getattr(configs, 'w_proto', 0.1)   # 原型对齐损失权重
        self.w_con   = getattr(configs, 'w_con',   0.1)   # 跨域对比损失权重
        self.temperature = temperature

        # 时频联合特征提取器（源域和目标域共用权重）
        self.feature_extractor = TimeFreqExtractor(
            input_dim, hidden_dim, feat_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # 原型对齐模块
        self.prototype_align = PrototypeAlignmentLoss(
            feat_dim,
            num_prototypes=num_prototypes,
            momentum=proto_momentum,
        ).to(device)

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

        # ── 时频联合特征提取（源域 & 目标域共享提取器） ───────────────────────
        source_feat = self.feature_extractor(source_data)   # (batch, feat_dim)
        target_feat = self.feature_extractor(target_data)   # (batch, feat_dim)

        # ── 原型对齐损失（Loss Proto） ────────────────────────────────────────
        loss_proto = self.prototype_align(source_feat, target_feat)

        # ── 跨域对比损失（Loss Con） ──────────────────────────────────────────
        loss_con = cross_domain_contrastive_loss(
            source_feat, target_feat, self.temperature
        )

        # 辅助损失挂载（训练循环自动累加到主损失）
        self.extra_loss = self.w_proto * loss_proto + self.w_con * loss_con

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