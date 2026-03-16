import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 0. 梯度反转层（GRL） —— Domain Classifier 对抗训练必需
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
# 1. SetFeat 组件（来自 setfeat_network.py，适配时序输入）
# ══════════════════════════════════════════════════════════════════════════════

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


class LinearAttention(nn.Module):
    """线性注意力，输入输出均为 (batch, seq_len, dim)"""
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.linear_q = nn.Linear(input_size, input_size)
        self.linear_k = nn.Linear(input_size, input_size)
        self.linear_v = nn.Linear(input_size, input_size)
        self.linear_final = nn.Linear(input_size, input_size)

    def forward(self, x):
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        scale = x.size(-1) ** 0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        return self.linear_final(torch.matmul(attn, v))

def mmd_loss(src, tgt):
    """RBF 核的 MMD，直接衡量两个分布的距离"""
    def rbf_kernel(x, y, sigma=1.0):
        diff = x.unsqueeze(1) - y.unsqueeze(0)        # (B, B, D)
        dist = (diff ** 2).sum(-1)                     # (B, B)
        return torch.exp(-dist / (2 * sigma ** 2))

    k_ss = rbf_kernel(src, src).mean()
    k_tt = rbf_kernel(tgt, tgt).mean()
    k_st = rbf_kernel(src, tgt).mean()
    return k_ss + k_tt - 2 * k_st


class SetFeat4(nn.Module):
    """
    三层 LinearBlock + LinearAttention，严格对应框架图(c)的 block1/block2/block3。

    forward 返回：
        blocks: [a1, a2, a3]，每个 (batch, seq_len, n_filters[i])，供逐层对齐
        repr_:  (batch, n_filters[-1])，block3 均值池化，作为最终表示
    """
    def __init__(self, input_dim, n_filters, n_heads):
        super().__init__()
        self.layer1 = LinearBlock(input_dim,    n_filters[0])
        self.layer2 = LinearBlock(n_filters[0], n_filters[1])
        self.layer3 = LinearBlock(n_filters[1], n_filters[2])
        self.atten1 = LinearAttention(n_filters[0], n_heads[0])
        self.atten2 = LinearAttention(n_filters[1], n_heads[1])
        self.atten3 = LinearAttention(n_filters[2], n_heads[2])

    def forward(self, x):
        x1 = self.layer1(x);  a1 = self.atten1(x1)
        x2 = self.layer2(x1); a2 = self.atten2(x2)
        x3 = self.layer3(x2); a3 = self.atten3(x3)
        return [a1, a2, a3], a3.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. (a) Sales Data Completion — VAE
# ══════════════════════════════════════════════════════════════════════════════

class VAEForSalesCompletion(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.flatten_dim = seq_len * input_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        # 修复1：去掉 Sigmoid，销售数据不在 [0,1] 范围内，Sigmoid 会压缩数值导致重构误差爆炸
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.flatten_dim)
        )

    def reparameterize(self, mu, logvar):
        # 修复2：clamp logvar 防止 exp 溢出
        logvar = logvar.clamp(min=-4, max=4)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def normalize(self, x):
        """batch 内归一化，消除销售数据量级（万级）对损失的影响"""
        self._x_mean = x.mean(dim=(1, 2), keepdim=True)
        self._x_std  = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (x - self._x_mean) / self._x_std

    def denormalize(self, x_norm):
        return x_norm * self._x_std + self._x_mean

    def forward(self, x, mask):
        # 修复3：先归一化再送入 encoder，避免万级数值导致梯度爆炸
        x_norm = self.normalize(x)
        x_flat = (x_norm * (1 - mask)).reshape(x.shape[0], self.flatten_dim)
        h = self.encoder(x_flat)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon_norm = self.decoder(z).reshape(x.shape)
        # 还原到原始量级
        x_recon = self.denormalize(x_recon_norm)
        x_completed = x * (1 - mask) + x_recon * mask
        return x_completed, mu, logvar

    def loss_fn(self, x_recon, x, mu, logvar):
        # 修复4：用 mean 代替 sum，避免因样本数/特征数累加导致数值爆炸
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        logvar = logvar.clamp(min=-4, max=4)
        # 修复5：KL 散度加权系数 0.001，防止 KL 项主导损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss


# ══════════════════════════════════════════════════════════════════════════════
# 3. (b) 时域 + 频域特征提取（单域，Target/Source 各实例化一个）
# ══════════════════════════════════════════════════════════════════════════════

class DomainFeatureExtractor(nn.Module):
    """
    框架图右侧 Target/Source Feature Extraction 模块。

    多粒度策略（对应框架图(b)/(c) block1/2/3 的不同时间窗口）：
        level 0 : 完整序列          → 细粒度
        level 1 : 后 1/2 段         → 中粒度
        level 2 : 后 1/3 段         → 粗粒度
    每个粒度用独立的 SetFeat4 分别处理时域和频域。
    """
    def __init__(self, input_dim, n_filters, n_heads, granularity_levels=3):
        super().__init__()
        self.granularity_levels = granularity_levels
        self.n_filters = n_filters

        self.time_extractors = nn.ModuleList([
            SetFeat4(input_dim, n_filters, n_heads)
            for _ in range(granularity_levels)
        ])
        self.freq_extractors = nn.ModuleList([
            SetFeat4(input_dim, n_filters, n_heads)
            for _ in range(granularity_levels)
        ])

        # 时域 block_i + 频域 block_i 融合投影，每粒度每层独立
        self.fusion_projs = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(n_filters[layer] * 2, n_filters[layer])
                for layer in range(3)
            ])
            for _ in range(granularity_levels)
        ])

    @staticmethod
    def get_freq(x):
        xf = torch.abs(fft.fft(x, dim=1))
        return (xf - xf.mean(1, keepdim=True)) / (xf.std(1, keepdim=True) + 1e-8)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        return:
            all_blocks: List[level] of List[layer] of (batch, slice_len, n_filters[layer])
            all_reprs:  List[level] of (batch, n_filters[-1]*2)
        """
        seq_len = x.shape[1]
        starts  = [0, seq_len // 2, seq_len - seq_len // 3]

        all_blocks, all_reprs = [], []
        for level in range(self.granularity_levels):
            x_slice = x[:, starts[level]:, :]
            x_freq  = self.get_freq(x_slice)

            t_blocks, t_repr = self.time_extractors[level](x_slice)
            f_blocks, f_repr = self.freq_extractors[level](x_freq)

            fused_blocks = [
                self.fusion_projs[level][layer](
                    torch.cat([t_blocks[layer], f_blocks[layer]], dim=-1)
                )
                for layer in range(3)
            ]
            all_blocks.append(fused_blocks)
            all_reprs.append(torch.cat([t_repr, f_repr], dim=-1))

        return all_blocks, all_reprs


# ══════════════════════════════════════════════════════════════════════════════
# 4. (c) 逐粒度逐层 Transport Map 对齐 — Loss F
# ══════════════════════════════════════════════════════════════════════════════

class TransportMap(nn.Module):
    """
    单层 transport map（框架图(c)虚线框）：
    将源域 block 线性映射后与目标域 block 做余弦对比对齐。
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.map = nn.Linear(feat_dim, feat_dim)

    def forward(self, src_block, tgt_block, temperature=0.1):
        s = F.normalize(self.map(src_block.mean(1)), dim=-1)
        t = F.normalize(tgt_block.mean(1),           dim=-1)
        min_b = min(s.shape[0], t.shape[0])
        # 修复：batch < 2 时对比损失无意义且不稳定，直接返回 0
        if min_b < 2:
            return torch.tensor(0.0, device=s.device, requires_grad=True)
        s, t   = s[:min_b], t[:min_b]
        sim    = torch.mm(s, t.T) / temperature
        labels = torch.arange(min_b, device=s.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class MultiLevelAlignmentLoss(nn.Module):
    """granularity_levels × 3 个 TransportMap，覆盖所有粒度和层级"""
    def __init__(self, granularity_levels, n_filters):
        super().__init__()
        self.transport_maps = nn.ModuleList([
            nn.ModuleList([TransportMap(n_filters[layer]) for layer in range(3)])
            for _ in range(granularity_levels)
        ])
        self.granularity_levels = granularity_levels

    def forward(self, src_all_blocks, tgt_all_blocks):
        total = sum(
            self.transport_maps[lv][ly](src_all_blocks[lv][ly], tgt_all_blocks[lv][ly])
            for lv in range(self.granularity_levels)
            for ly in range(3)
        )
        return total / (self.granularity_levels * 3)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 多粒度表示融合
# ══════════════════════════════════════════════════════════════════════════════

class RepresentationFusion(nn.Module):
    """注意力加权融合各粒度表示 → 单一全局表示"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, reprs):
        stacked = torch.stack(reprs, dim=1)                    # (B, L, in_dim)
        weights = F.softmax(self.attn(stacked), dim=1)         # (B, L, 1)
        fused   = (stacked * weights).sum(dim=1)               # (B, in_dim)
        return self.proj(fused)                                # (B, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sales Predictor（Loss P）+ Domain Classifier w/ GRL（Loss C）
# ══════════════════════════════════════════════════════════════════════════════

class SalesPredictor(nn.Module):
    def __init__(self, feat_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self, feat_dim, grl_alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(self.grl(x))


# ══════════════════════════════════════════════════════════════════════════════
# 7. 整体 HierDA 模型
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    HierDA 完整实现，严格对齐算法框架图：

      Input
       → (a) VAE 数据补全
       → (b) 时域 + 频域多粒度特征提取（Target/Source 各自独立 DomainFeatureExtractor）
       → (c) 逐粒度逐层 TransportMap 对齐（Loss F）
       → 多粒度 RepresentationFusion → Target/Source Representation
       → SalesPredictor（Loss P）+ DomainClassifier w/ GRL（Loss C）
    """
    def __init__(self, configs, granularity_levels=3):
        super().__init__()
        seq_len   = configs.seq_len
        input_dim = configs.enc_in
        device    = 'cuda' if configs.use_gpu else 'cpu'

        self.device             = device
        self.granularity_levels = granularity_levels
        self.pred_len           = configs.pred_len
        self.label_len          = configs.label_len
        self.input_dim          = input_dim
        self.extra_loss         = None

        n_filters = getattr(configs, 'setfeat_filters', [64, 64, 64])
        n_heads   = getattr(configs, 'setfeat_heads',   [4,  4,  4 ])
        grl_alpha = getattr(configs, 'grl_alpha',        1.0)
        feat_dim  = getattr(configs, 'feat_dim',         128)

        repr_dim = n_filters[-1] * 2   # time block3 + freq block3

        # (a) VAE
        self.vae = VAEForSalesCompletion(seq_len, input_dim).to(device)

        # (b) 独立的 Target / Source 特征提取器
        self.target_extractor = DomainFeatureExtractor(
            input_dim, n_filters, n_heads, granularity_levels).to(device)
        self.source_extractor = DomainFeatureExtractor(
            input_dim, n_filters, n_heads, granularity_levels).to(device)

        # (c) 多粒度逐层对齐
        self.align_loss_module = MultiLevelAlignmentLoss(
            granularity_levels, n_filters).to(device)

        # 表示融合
        self.target_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)
        self.source_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)

        # 预测 & 分类
        self.sales_predictor   = SalesPredictor(feat_dim).to(device)
        self.domain_classifier = DomainClassifier(feat_dim, grl_alpha).to(device)

        self.output_scale = nn.Linear(1, 1).to(device)
        nn.init.constant_(self.output_scale.weight, 1.0)
        nn.init.constant_(self.output_scale.bias, 0.0)

        self.residual_predictor = nn.Sequential(
            nn.Linear(feat_dim + 1, 32),  # feat_dim + 历史均值
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        """
        与 exp_long_term_forecasting.py 训练循环完全兼容：
          x_enc     — 源域序列  (batch, seq_len, input_dim)
          x_target  — 目标域序列 (batch, seq_len, input_dim)，由 data_loader 提供
        """
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        # ── (a) VAE 补全 ──────────────────────────────────────────────────────
        src_mask = torch.zeros_like(source_data)
        tgt_mask = torch.zeros_like(target_data)
        src_comp, s_mu, s_lv = self.vae(source_data, src_mask)
        tgt_comp, t_mu, t_lv = self.vae(target_data, tgt_mask)
        loss_vae = (self.vae.loss_fn(src_comp, source_data, s_mu, s_lv) +
                    self.vae.loss_fn(tgt_comp, target_data, t_mu, t_lv)) / 2

        # ── (b) 时频多粒度特征提取 ────────────────────────────────────────────
        src_blocks, src_reprs = self.source_extractor(src_comp)
        tgt_blocks, tgt_reprs = self.target_extractor(tgt_comp)

        # ── (c) 逐粒度逐层对齐（Loss F）──────────────────────────────────────
        loss_f = self.align_loss_module(src_blocks, tgt_blocks)

        # ── Representation 融合 ───────────────────────────────────────────────
        target_repr = self.target_fusion(tgt_reprs)   # (batch, feat_dim)
        source_repr = self.source_fusion(src_reprs)   # (batch, feat_dim)

        # ── Domain Classifier w/ GRL（Loss C）────────────────────────────────
        domain_feat  = torch.cat([source_repr, target_repr], dim=0)
        domain_label = torch.cat([
            torch.zeros(source_repr.shape[0]),
            torch.ones(target_repr.shape[0])
        ]).to(source_data.device)
        domain_pred = self.domain_classifier(domain_feat).squeeze(-1)
        loss_c = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        loss_mmd = mmd_loss(source_repr, target_repr)

        # ── 辅助损失挂载（训练循环中累加到 Loss P）───────────────────────────
        # self.extra_loss = loss_f * 0.01 + loss_c * 0.01 + loss_vae * 0.001
        # self.extra_loss = loss_f * 0.001 + loss_c * 0.001 + loss_vae * 0.0001
        # self.extra_loss = loss_f * 0.1 + loss_c * 0.1 + loss_vae * 0.001
        self.extra_loss = loss_f * 0.1 + loss_c * 0.1 + loss_vae * 0.001 + loss_mmd * 0.5

        # 历史销量均值作为先验基准（只取第0列即销量列）
        # history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)  # (batch, 1)
        # 在 HierDA.py 的 forward 里修改这一行
        history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True).detach()  # 加 detach

        # 将历史均值拼接到特征向量，辅助预测
        feat_with_prior = torch.cat([target_repr, history_mean], dim=-1)  # (batch, feat_dim+1)

        # 预测残差（相对于历史均值的偏差）
        residual = self.residual_predictor(feat_with_prior)  # (batch, 1)

        # 最终预测 = 历史均值 + 残差，再经过 output_scale 校正
        pred_value = self.output_scale(history_mean + residual)  # (batch, 1)
        pred_value = pred_value.expand(-1, self.pred_len)  # (batch, pred_len)

        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)
        return pred


# ══════════════════════════════════════════════════════════════════════════════
# ★ 消融专用组件（新增，原有代码不受影响）
# ══════════════════════════════════════════════════════════════════════════════

class AblationDomainFeatureExtractor(nn.Module):
    """
    在 DomainFeatureExtractor 基础上新增两个消融维度：
      granularity_levels : 1/2/3，控制使用几个粒度（Group2）
      freq_mode          : 'both'/'time_only'/'freq_only'，控制时频分支（Group3）

    网络结构固定按 granularity_levels 实例化，freq_mode 通过前向置零实现，
    不改变参数量，保证与完整模型的公平对比。
    """
    def __init__(self, input_dim, n_filters, n_heads,
                 granularity_levels=3, freq_mode='both'):
        super().__init__()
        self.granularity_levels = granularity_levels
        self.freq_mode          = freq_mode
        self.n_filters          = n_filters

        self.time_extractors = nn.ModuleList([
            SetFeat4(input_dim, n_filters, n_heads)
            for _ in range(granularity_levels)
        ])
        self.freq_extractors = nn.ModuleList([
            SetFeat4(input_dim, n_filters, n_heads)
            for _ in range(granularity_levels)
        ])
        self.fusion_projs = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(n_filters[layer] * 2, n_filters[layer])
                for layer in range(3)
            ])
            for _ in range(granularity_levels)
        ])

    @staticmethod
    def get_freq(x):
        xf = torch.abs(fft.fft(x, dim=1))
        return (xf - xf.mean(1, keepdim=True)) / (xf.std(1, keepdim=True) + 1e-8)

    def forward(self, x):
        seq_len = x.shape[1]
        # 三个粒度起点；granularity_levels < 3 时只取前几个
        all_starts = [0, seq_len // 2, seq_len - seq_len // 3]
        starts = all_starts[:self.granularity_levels]

        all_blocks, all_reprs = [], []
        for level, start in enumerate(starts):
            x_slice = x[:, start:, :]
            x_freq  = self.get_freq(x_slice)

            t_blocks, t_repr = self.time_extractors[level](x_slice)
            f_blocks, f_repr = self.freq_extractors[level](x_freq)

            # ── Group3 消融：置零对应分支，不改变网络结构 ──────────────────
            # 使用 torch.zeros_like 产生同形状零张量，梯度不流过被置零分支
            if self.freq_mode == 'time_only':
                f_blocks = [torch.zeros_like(fb) for fb in f_blocks]
                f_repr   = torch.zeros_like(f_repr)
            elif self.freq_mode == 'freq_only':
                t_blocks = [torch.zeros_like(tb) for tb in t_blocks]
                t_repr   = torch.zeros_like(t_repr)

            fused_blocks = [
                self.fusion_projs[level][layer](
                    torch.cat([t_blocks[layer], f_blocks[layer]], dim=-1)
                )
                for layer in range(3)
            ]
            all_blocks.append(fused_blocks)
            all_reprs.append(torch.cat([t_repr, f_repr], dim=-1))

        return all_blocks, all_reprs


class AblationModel(nn.Module):
    """
    消融实验专用模型，与原始 Model 类完全独立，不修改后者任何代码。

    通过 configs 读取以下消融参数（均有安全默认值，不传即等于完整模型）：

    ablation_mode（str，默认 'full'）
    ── Group1 VAE ──────────────────────────────────────────────────────────
      'full'          完整模型基准
      'wo_vae'        跳过VAE，直接用原始输入；loss_vae=0
      'vae_noloss'    VAE补全路径保留，但loss_vae不加入extra_loss
      'vae_detach'    VAE输出detach后送入特征提取，loss_vae=0

    ── Group4 对齐损失 ──────────────────────────────────────────────────────
      'wo_transport'  去掉loss_f，保留loss_c + loss_vae
      'wo_grl'        去掉loss_c，保留loss_f + loss_vae
      'wo_da'         去掉loss_f和loss_c，仅保留loss_vae
      'no_source'     源域完全不参与，三项对齐损失全部跳过

    ── Group6 预测头 ────────────────────────────────────────────────────────
      'direct_pred'   去掉历史均值先验，直接 SalesPredictor(target_repr)
      'prior_only'    仅输出 OutputScale(history_mean)，无残差
      'no_scale'      有残差+先验，去掉 OutputScale 线性校正

    granularity_levels（int，默认 3）── Group2 多粒度数量
    freq_mode（str，默认 'both'）── Group3 时频双流
      'both' / 'time_only' / 'freq_only'

    lambda_f / lambda_c / lambda_vae（float）── 损失系数，默认与原模型一致
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
        self.extra_loss = None

        # ── 消融控制参数（全部从 configs 读，有安全默认值） ──────────────────
        self.ablation_mode      = getattr(configs, 'ablation_mode',      'full')
        self.granularity_levels = getattr(configs, 'granularity_levels', 3)
        self.freq_mode          = getattr(configs, 'freq_mode',          'both')
        self.lf                 = getattr(configs, 'lambda_f',           0.001)
        self.lc                 = getattr(configs, 'lambda_c',           0.001)
        self.lv                 = getattr(configs, 'lambda_vae',         0.0001)

        n_filters = getattr(configs, 'setfeat_filters', [64, 64, 64])
        n_heads   = getattr(configs, 'setfeat_heads',   [4,  4,  4])
        grl_alpha = getattr(configs, 'grl_alpha',        1.0)
        feat_dim  = getattr(configs, 'feat_dim',         128)
        repr_dim  = n_filters[-1] * 2

        # ── (a) VAE ──────────────────────────────────────────────────────────
        self.vae = VAEForSalesCompletion(seq_len, input_dim).to(device)

        # ── (b) 特征提取器：使用支持消融的 AblationDomainFeatureExtractor ────
        self.target_extractor = AblationDomainFeatureExtractor(
            input_dim, n_filters, n_heads,
            granularity_levels=self.granularity_levels,
            freq_mode=self.freq_mode
        ).to(device)
        self.source_extractor = AblationDomainFeatureExtractor(
            input_dim, n_filters, n_heads,
            granularity_levels=self.granularity_levels,
            freq_mode=self.freq_mode
        ).to(device)

        # ── (c) Transport Map 对齐 ────────────────────────────────────────────
        self.align_loss_module = MultiLevelAlignmentLoss(
            self.granularity_levels, n_filters).to(device)

        # ── 表示融合 ──────────────────────────────────────────────────────────
        self.target_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)
        self.source_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)

        # ── 预测头 & 域分类器 ──────────────────────────────────────────────────
        self.sales_predictor   = SalesPredictor(feat_dim).to(device)
        self.domain_classifier = DomainClassifier(feat_dim, grl_alpha).to(device)

        self.output_scale = nn.Linear(1, 1).to(device)
        nn.init.constant_(self.output_scale.weight, 1.0)
        nn.init.constant_(self.output_scale.bias,   0.0)

        self.residual_predictor = nn.Sequential(
            nn.Linear(feat_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]
        ablation    = self.ablation_mode

        # ══════════════════════════════════════════════════════════════════════
        # (a) VAE 补全 ── Group1 消融
        # ══════════════════════════════════════════════════════════════════════
        if ablation == 'wo_vae':
            # 完全跳过 VAE
            src_comp = source_data
            tgt_comp = target_data
            loss_vae = torch.tensor(0.0, device=source_data.device)

        else:
            src_mask = torch.zeros_like(source_data)
            tgt_mask = torch.zeros_like(target_data)
            src_comp, s_mu, s_lv = self.vae(source_data, src_mask)
            tgt_comp, t_mu, t_lv = self.vae(target_data, tgt_mask)

            if ablation == 'no_source':
                # 源域 VAE loss 不计
                loss_vae = self.vae.loss_fn(tgt_comp, target_data, t_mu, t_lv)
            else:
                loss_vae = (
                    self.vae.loss_fn(src_comp, source_data, s_mu, s_lv) +
                    self.vae.loss_fn(tgt_comp, target_data, t_mu, t_lv)
                ) / 2

            if ablation == 'vae_detach':
                # 阻断梯度回传至 VAE，loss_vae 同样不计
                src_comp = src_comp.detach()
                tgt_comp = tgt_comp.detach()
                loss_vae = torch.tensor(0.0, device=source_data.device)

        # ══════════════════════════════════════════════════════════════════════
        # (b) 特征提取 ── Group2/Group3 消融由 AblationDomainFeatureExtractor 内部处理
        # ══════════════════════════════════════════════════════════════════════
        tgt_blocks, tgt_reprs = self.target_extractor(tgt_comp)

        if ablation == 'no_source':
            # 源域不参与：用目标域 blocks/reprs 占位，保持接口一致
            src_blocks = tgt_blocks
            src_reprs  = tgt_reprs
        else:
            src_blocks, src_reprs = self.source_extractor(src_comp)

        # ══════════════════════════════════════════════════════════════════════
        # (c) Transport Map 对齐 ── Group4 消融
        # ══════════════════════════════════════════════════════════════════════
        if ablation in ('wo_transport', 'wo_da', 'no_source'):
            loss_f = torch.tensor(0.0, device=source_data.device)
        else:
            loss_f = self.align_loss_module(src_blocks, tgt_blocks)

        # ── 表示融合 ──────────────────────────────────────────────────────────
        target_repr = self.target_fusion(tgt_reprs)
        source_repr = self.source_fusion(src_reprs)

        # ── Domain Classifier / GRL ── Group4 消融 ───────────────────────────
        if ablation in ('wo_grl', 'wo_da', 'no_source'):
            loss_c = torch.tensor(0.0, device=source_data.device)
        else:
            domain_feat  = torch.cat([source_repr, target_repr], dim=0)
            domain_label = torch.cat([
                torch.zeros(source_repr.shape[0]),
                torch.ones(target_repr.shape[0])
            ]).to(source_data.device)
            domain_pred = self.domain_classifier(domain_feat).squeeze(-1)
            loss_c = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        # ══════════════════════════════════════════════════════════════════════
        # extra_loss 组合
        # ══════════════════════════════════════════════════════════════════════
        if ablation == 'vae_noloss':
            self.extra_loss = loss_f * self.lf + loss_c * self.lc
        elif ablation in ('wo_vae', 'vae_detach'):
            # loss_vae 已在上方置 0，仍按统一公式合并，语义清晰
            self.extra_loss = loss_f * self.lf + loss_c * self.lc + loss_vae * self.lv
        else:
            self.extra_loss = loss_f * self.lf + loss_c * self.lc + loss_vae * self.lv

        # ══════════════════════════════════════════════════════════════════════
        # 预测头 ── Group6 消融
        # ══════════════════════════════════════════════════════════════════════
        history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)  # (B, 1)

        if ablation == 'direct_pred':
            # 去掉历史均值先验，直接预测
            pred_value = self.sales_predictor(target_repr)              # (B, 1)

        elif ablation == 'prior_only':
            # 仅输出历史均值，无残差
            pred_value = self.output_scale(history_mean)                # (B, 1)

        elif ablation == 'no_scale':
            # 有残差+先验，去掉 OutputScale
            feat_with_prior = torch.cat([target_repr, history_mean], dim=-1)
            residual        = self.residual_predictor(feat_with_prior)
            pred_value      = history_mean + residual                   # (B, 1)

        else:
            # full 及所有其他消融变体：完整残差预测头
            feat_with_prior = torch.cat([target_repr, history_mean], dim=-1)
            residual        = self.residual_predictor(feat_with_prior)
            pred_value      = self.output_scale(history_mean + residual)  # (B, 1)

        pred_value = pred_value.expand(-1, self.pred_len)
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device
        )
        return torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)
