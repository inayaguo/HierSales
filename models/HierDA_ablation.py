import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


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
# 1. SetFeat 组件
# ══════════════════════════════════════════════════════════════════════════════

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


class LinearAttention(nn.Module):
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


class SetFeat4(nn.Module):
    """
    三层 LinearBlock + LinearAttention。
    forward 返回：
        blocks: [a1, a2, a3]，每个 (batch, seq_len, n_filters[i])
        repr_:  (batch, n_filters[-1])，block3 均值池化
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
# 2. (a) VAE 数据补全
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
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.flatten_dim)
        )

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(min=-4, max=4)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def normalize(self, x):
        self._x_mean = x.mean(dim=(1, 2), keepdim=True)
        self._x_std  = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (x - self._x_mean) / self._x_std

    def denormalize(self, x_norm):
        return x_norm * self._x_std + self._x_mean

    def forward(self, x, mask):
        x_norm = self.normalize(x)
        x_flat = (x_norm * (1 - mask)).reshape(x.shape[0], self.flatten_dim)
        h = self.encoder(x_flat)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon_norm = self.decoder(z).reshape(x.shape)
        x_recon = self.denormalize(x_recon_norm)
        x_completed = x * (1 - mask) + x_recon * mask
        return x_completed, mu, logvar

    def loss_fn(self, x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        logvar = logvar.clamp(min=-4, max=4)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss


# ══════════════════════════════════════════════════════════════════════════════
# 3. (b) 时域 + 频域多粒度特征提取
#
# 消融 Group2（多粒度）和 Group3（时频）均在此类中通过参数控制：
#   granularity_levels : 1 / 2 / 3  对应单/双/三粒度
#   freq_mode          : 'both' / 'time_only' / 'freq_only'
# ══════════════════════════════════════════════════════════════════════════════

class DomainFeatureExtractor(nn.Module):
    """
    freq_mode:
        'both'      — 时域 + 频域双流（完整，默认）
        'time_only' — 仅时域，频域分支输出零张量（不参与梯度）
        'freq_only' — 仅频域，时域分支输出零张量（不参与梯度）
    granularity_levels:
        3           — 完整三粒度（默认）
        2           — 双粒度（取前两个切片起点）
        1           — 单粒度（仅完整序列）
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
        # 全部三个起点；granularity_levels < 3 时只取前几个
        all_starts = [0, seq_len // 2, seq_len - seq_len // 3]
        starts = all_starts[:self.granularity_levels]

        all_blocks, all_reprs = [], []
        for level, start in enumerate(starts):
            x_slice = x[:, start:, :]
            x_freq  = self.get_freq(x_slice)

            t_blocks, t_repr = self.time_extractors[level](x_slice)
            f_blocks, f_repr = self.freq_extractors[level](x_freq)

            # ── Group3 消融：屏蔽时域或频域分支 ─────────────────────────────
            if self.freq_mode == 'time_only':
                # 频域分支用零替换，保持张量形状不变（不影响 fusion_projs 的输入维度）
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. (c) Transport Map 对齐 — Loss F
# ══════════════════════════════════════════════════════════════════════════════

class TransportMap(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.map = nn.Linear(feat_dim, feat_dim)

    def forward(self, src_block, tgt_block, temperature=0.1):
        s = F.normalize(self.map(src_block.mean(1)), dim=-1)
        t = F.normalize(tgt_block.mean(1),           dim=-1)
        min_b = min(s.shape[0], t.shape[0])
        if min_b < 2:
            return torch.tensor(0.0, device=s.device, requires_grad=True)
        s, t   = s[:min_b], t[:min_b]
        sim    = torch.mm(s, t.T) / temperature
        labels = torch.arange(min_b, device=s.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class MultiLevelAlignmentLoss(nn.Module):
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, reprs):
        stacked = torch.stack(reprs, dim=1)
        weights = F.softmax(self.attn(stacked), dim=1)
        fused   = (stacked * weights).sum(dim=1)
        return self.proj(fused)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sales Predictor + Domain Classifier
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
#
# ── 消融参数说明 ──────────────────────────────────────────────────────────────
#
# ablation_mode（通过 configs 传入，默认 'full'）：
#
#   【Group1 VAE 消融】
#   'full'            — 完整模型（默认）
#   'wo_vae'          — 跳过 VAE，直接使用原始输入；loss_vae 不计入 extra_loss
#   'vae_noloss'      — VAE 补全路径保留，但 loss_vae 不加入 extra_loss
#   'vae_detach'      — VAE 补全后 detach，不向主网络回传 VAE 梯度；loss_vae 不计入
#
#   【Group4 对齐损失消融】
#   'wo_transport'    — 去掉 loss_f（Transport Map），保留 loss_c + loss_vae
#   'wo_grl'          — 去掉 loss_c（GRL 域分类器），保留 loss_f + loss_vae
#   'wo_da'           — 去掉 loss_f 和 loss_c，仅保留 loss_vae（无域适应）
#   'no_source'       — 源域完全不参与：loss_f / loss_c / loss_vae(src) 均跳过，
#                       退化为纯单域模型（source_repr 仍需构造以对齐接口）
#
#   【Group6 预测头消融】
#   'direct_pred'     — 去掉历史均值先验，直接用 SalesPredictor(target_repr) 预测
#   'prior_only'      — 输出固定为 output_scale(history_mean)，残差预测器不参与
#   'no_scale'        — 去掉 output_scale，直接输出 history_mean + residual
#
# granularity_levels（Group2 多粒度消融，通过 configs 传入，默认 3）：
#   3 — 三粒度（完整）
#   2 — 双粒度（细 + 中）
#   1 — 单粒度（仅完整序列）
#
# freq_mode（Group3 时频消融，通过 configs 传入，默认 'both'）：
#   'both'      — 时域 + 频域（完整）
#   'time_only' — 仅时域
#   'freq_only' — 仅频域
#
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    def __init__(self, configs, granularity_levels=3):
        super().__init__()
        seq_len   = configs.seq_len
        input_dim = configs.enc_in
        device    = 'cuda' if configs.use_gpu else 'cpu'

        self.device             = device
        self.pred_len           = configs.pred_len
        self.label_len          = configs.label_len
        self.input_dim          = input_dim
        self.extra_loss         = None

        # ── 消融控制参数（从 configs 读取，均有安全默认值） ──────────────────
        self.ablation_mode      = getattr(configs, 'ablation_mode',      'full')
        self.granularity_levels = getattr(configs, 'granularity_levels', granularity_levels)
        self.freq_mode          = getattr(configs, 'freq_mode',          'both')

        n_filters = getattr(configs, 'setfeat_filters', [64, 64, 64])
        n_heads   = getattr(configs, 'setfeat_heads',   [4,  4,  4])
        grl_alpha = getattr(configs, 'grl_alpha',        1.0)
        feat_dim  = getattr(configs, 'feat_dim',         128)

        # loss 系数（可由 configs 覆盖，默认与原始一致）
        self.lf  = getattr(configs, 'lambda_f',   0.001)
        self.lc  = getattr(configs, 'lambda_c',   0.001)
        self.lv  = getattr(configs, 'lambda_vae', 0.0001)

        repr_dim = n_filters[-1] * 2

        # ── 子模块实例化 ──────────────────────────────────────────────────────
        # (a) VAE
        self.vae = VAEForSalesCompletion(seq_len, input_dim).to(device)

        # (b) Target / Source 特征提取器（注入 granularity_levels 和 freq_mode）
        self.target_extractor = DomainFeatureExtractor(
            input_dim, n_filters, n_heads,
            granularity_levels=self.granularity_levels,
            freq_mode=self.freq_mode
        ).to(device)
        self.source_extractor = DomainFeatureExtractor(
            input_dim, n_filters, n_heads,
            granularity_levels=self.granularity_levels,
            freq_mode=self.freq_mode
        ).to(device)

        # (c) Transport Map 对齐
        self.align_loss_module = MultiLevelAlignmentLoss(
            self.granularity_levels, n_filters).to(device)

        # 表示融合
        self.target_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)
        self.source_fusion = RepresentationFusion(repr_dim, feat_dim).to(device)

        # 预测头 & 域分类器
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

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_target=None):
        source_data = x_enc
        target_data = x_target if x_target is not None else x_dec[:, :self.label_len, :]

        ablation = self.ablation_mode

        # ══════════════════════════════════════════════════════════════════════
        # (a) VAE 补全  ── Group1 消融在此分支
        # ══════════════════════════════════════════════════════════════════════
        if ablation == 'wo_vae':
            # 完全跳过 VAE，直接用原始数据；loss_vae 不计
            src_comp  = source_data
            tgt_comp  = target_data
            loss_vae  = torch.tensor(0.0, device=source_data.device)

        else:
            # 其余所有模式均走 VAE 路径
            src_mask = torch.zeros_like(source_data)
            tgt_mask = torch.zeros_like(target_data)
            src_comp, s_mu, s_lv = self.vae(source_data, src_mask)
            tgt_comp, t_mu, t_lv = self.vae(target_data, tgt_mask)

            if ablation == 'no_source':
                # 源域 VAE loss 不计（源域数据仍走前向，但不贡献 loss）
                loss_vae = self.vae.loss_fn(tgt_comp, target_data, t_mu, t_lv)
            else:
                loss_vae = (
                    self.vae.loss_fn(src_comp, source_data, s_mu, s_lv) +
                    self.vae.loss_fn(tgt_comp, target_data, t_mu, t_lv)
                ) / 2

            if ablation == 'vae_detach':
                # VAE 补全结果 detach，阻断梯度回传到 VAE；loss_vae 同样不加
                src_comp = src_comp.detach()
                tgt_comp = tgt_comp.detach()
                loss_vae = torch.tensor(0.0, device=source_data.device)

        # ══════════════════════════════════════════════════════════════════════
        # (b) 时频多粒度特征提取
        # Group2（多粒度）、Group3（时频）的消融由 DomainFeatureExtractor 内部控制
        # ══════════════════════════════════════════════════════════════════════
        tgt_blocks, tgt_reprs = self.target_extractor(tgt_comp)

        if ablation == 'no_source':
            # 源域不参与对齐：用目标域 blocks/reprs 替代（接口对齐，不计 loss）
            src_blocks = tgt_blocks
            src_reprs  = tgt_reprs
        else:
            src_blocks, src_reprs = self.source_extractor(src_comp)

        # ══════════════════════════════════════════════════════════════════════
        # (c) Transport Map 对齐  ── Group4 消融在此
        # ══════════════════════════════════════════════════════════════════════
        if ablation in ('wo_transport', 'wo_da', 'no_source'):
            loss_f = torch.tensor(0.0, device=source_data.device)
        else:
            loss_f = self.align_loss_module(src_blocks, tgt_blocks)

        # ── 表示融合 ──────────────────────────────────────────────────────────
        target_repr = self.target_fusion(tgt_reprs)
        source_repr = self.source_fusion(src_reprs)

        # ── Domain Classifier / GRL  ── Group4 消融 ──────────────────────────
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
        # extra_loss 组合  ── 根据消融模式决定各项是否纳入
        # ══════════════════════════════════════════════════════════════════════
        if ablation == 'vae_noloss':
            # VAE 路径存在但 loss_vae 不加
            self.extra_loss = loss_f * self.lf + loss_c * self.lc
        elif ablation in ('wo_vae', 'vae_detach'):
            # VAE 已跳过或 detach，loss_vae=0 已在上方设置
            self.extra_loss = loss_f * self.lf + loss_c * self.lc
        else:
            # full / wo_transport / wo_grl / wo_da / no_source / Group6 变体
            self.extra_loss = loss_f * self.lf + loss_c * self.lc + loss_vae * self.lv

        # ══════════════════════════════════════════════════════════════════════
        # 预测头  ── Group6 消融
        # ══════════════════════════════════════════════════════════════════════
        history_mean = target_data[:, :, 0].mean(dim=1, keepdim=True)  # (B, 1)

        if ablation == 'direct_pred':
            # 去掉历史均值先验，直接从 target_repr 预测
            pred_value = self.sales_predictor(target_repr)   # (B, 1)

        elif ablation == 'prior_only':
            # 只用历史均值，不接残差预测器
            pred_value = self.output_scale(history_mean)      # (B, 1)

        elif ablation == 'no_scale':
            # 有先验+残差，但去掉 output_scale 线性校正
            feat_with_prior = torch.cat([target_repr, history_mean], dim=-1)
            residual   = self.residual_predictor(feat_with_prior)
            pred_value = history_mean + residual               # (B, 1)

        else:
            # full / 所有其他消融模式：完整残差预测头
            feat_with_prior = torch.cat([target_repr, history_mean], dim=-1)
            residual   = self.residual_predictor(feat_with_prior)
            pred_value = self.output_scale(history_mean + residual)  # (B, 1)

        pred_value = pred_value.expand(-1, self.pred_len)     # (B, pred_len)
        padding = torch.zeros(
            pred_value.shape[0], self.pred_len, self.input_dim - 1,
            device=pred_value.device
        )
        pred = torch.cat([pred_value.unsqueeze(-1), padding], dim=-1)
        return pred