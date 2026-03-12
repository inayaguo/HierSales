import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ===================== 1. 数据补全模块（VAE + 显式规则） =====================
class VAEForSalesCompletion(nn.Module):
    """基于VAE的销售数据补全（对应框架图(a)部分）"""
    def __init__(self, seq_len, input_dim, hidden_dim=64, latent_dim=32):
        super(VAEForSalesCompletion, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flatten_dim = seq_len * input_dim  # 展平后的总维度
        # 编码器：输入维度改为展平后的总维度
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )
        # 解码器：输出维度也对应展平后的总维度
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flatten_dim),
            nn.Sigmoid()  # 归一化到0-1，适配销售数据分布
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask):
        """
        x: 原始销售数据 (batch, seq_len, input_dim)
        mask: 缺失值掩码（1=缺失，0=存在），基于显式规则生成
        """
        # 仅用非缺失数据编码
        x_masked = x * (1 - mask)
        # 展平：(batch, seq_len*input_dim)
        x_flat = x_masked.reshape(x.shape[0], self.flatten_dim)
        h = self.encoder(x_flat)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon_flat = self.decoder(z)
        # 恢复原始形状：(batch, seq_len, input_dim)
        x_recon = x_recon_flat.reshape(x.shape)
        # 补全缺失值：缺失位置用重构值填充，非缺失位置保留原值
        x_completed = x * (1 - mask) + x_recon * mask
        return x_completed, mu, logvar

    def loss_fn(self, x_recon, x, mu, logvar):
        """VAE损失：重构损失 + KL散度"""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

# ===================== 2. 多粒度销售域构建 + 特征提取【修复频域维度问题】 =====================
class MultiGranularityFeatureExtractor(nn.Module):
    """多粒度特征提取（时域+频域，对应框架图(b)和特征提取模块）"""
    def __init__(self, input_dim, time_feat_dim=64, freq_feat_dim=64, granularity_levels=3):
        super(MultiGranularityFeatureExtractor, self).__init__()
        self.granularity_levels = granularity_levels  # 多粒度层级（对应框架图的block1/2/3）
        self.time_feat_dim = time_feat_dim
        self.freq_feat_dim = freq_feat_dim

        # 时域特征提取（不同粒度）- 保持不变
        self.time_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, time_feat_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(granularity_levels)
        ])

        # 频域特征提取（FFT + 不同粒度）- 关键修复：改用Conv1d，和时域逻辑统一
        self.freq_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, freq_feat_dim, kernel_size=3, padding=1),  # 替换Linear为Conv1d
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(granularity_levels)
        ])

    def get_frequency_domain(self, x):
        """FFT变换获取频域特征（对应框架图的FFT模块）"""
        # 对时间维度做FFT
        x_fft = fft.fft(x, dim=1)
        # 取幅度谱（实部），归一化
        x_freq = torch.abs(x_fft)
        x_freq = (x_freq - x_freq.mean(dim=1, keepdim=True)) / (x_freq.std(dim=1, keepdim=True) + 1e-8)
        return x_freq

    def forward(self, x, is_source=True):
        """
        x: 补全后的销售数据 (batch, seq_len, input_dim)
        is_source: 是否为源域数据（区分source/target）
        return: 多粒度的时域+频域融合特征 (granularity_levels, batch, feat_dim)
        """
        # 维度调整：(batch, input_dim, seq_len) 适配Conv1d
        x = x.permute(0, 2, 1)
        # 1. 时域特征
        time_feats = []
        for level in range(self.granularity_levels):
            # 不同粒度：通过步长/池化模拟（框架图的block1/2/3）
            x_time = x[:, :, ::(level+1)] if level > 0 else x  # 降采样实现多粒度
            time_feat = self.time_encoders[level](x_time).squeeze(-1)  # (batch, time_feat_dim)
            time_feats.append(time_feat)

        # 2. 频域特征 - 逻辑不变，仅网络结构改为Conv1d
        x_freq = self.get_frequency_domain(x.permute(0,2,1)).permute(0,2,1)  # 转回 (batch, input_dim, seq_len)
        freq_feats = []
        for level in range(self.granularity_levels):
            x_f = x_freq[:, :, ::(level+1)] if level > 0 else x_freq
            freq_feat = self.freq_encoders[level](x_f).squeeze(-1)  # (batch, freq_feat_dim)
            freq_feats.append(freq_feat)

        # 3. 时域+频域特征融合
        fusion_feats = [torch.cat([t, f], dim=-1) for t, f in zip(time_feats, freq_feats)]
        return fusion_feats  # 列表：[level1_feat, level2_feat, level3_feat]

# ===================== 3. 多粒度特征对齐 Loss F =====================
def multi_level_alignment_loss(source_feats, target_feats, temperature=0.1):
    """
    多粒度特征对齐损失（对应框架图Loss F）
    source_feats/target_feats: 多粒度特征列表 [level1, level2, level3]
    """
    loss_f = 0.0
    for s_feat, t_feat in zip(source_feats, target_feats):
        # 归一化特征（余弦相似度）
        s_feat = F.normalize(s_feat, dim=-1)
        t_feat = F.normalize(t_feat, dim=-1)
        # 计算互信息/对比损失（对齐源域和目标域特征）
        sim_matrix = torch.mm(s_feat, t_feat.T) / temperature
        batch_size = s_feat.shape[0]
        labels = torch.arange(batch_size).to(s_feat.device)
        # 双向对比损失
        loss_s2t = F.cross_entropy(sim_matrix, labels)
        loss_t2s = F.cross_entropy(sim_matrix.T, labels)
        loss_f += (loss_s2t + loss_t2s) / 2
    return loss_f / len(source_feats)  # 平均多粒度损失

# ===================== 4. 销售预测器（Loss P） + 域分类器（Loss C） =====================
class SalesPredictor(nn.Module):
    """销售预测头（对应框架图Sales Predictor）"""
    def __init__(self, feat_dim, output_dim=1):
        super(SalesPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, feat):
        return self.predictor(feat)

class DomainClassifier(nn.Module):
    """域分类器（对抗训练，区分源/目标域，对应框架图Domain Classifier）"""
    def __init__(self, feat_dim):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        return self.classifier(feat)

# ===================== 5. 整体HierDA模型（串联所有模块） =====================
class Model(nn.Module):
    """整体HierDA模型（对齐算法框架全流程）"""
    def __init__(self, configs, granularity_levels=3):
        super(Model, self).__init__()
        seq_len = configs.seq_len
        input_dim = configs.enc_in
        device = 'cuda' if configs.use_gpu else 'cpu'

        self.device = device
        self.granularity_levels = granularity_levels
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.extra_loss = None # 初始化辅助损失

        feat_dim = 128  # time_feat_dim(64) + freq_feat_dim(64)

        # 模块1：数据补全VAE
        self.vae_completion = VAEForSalesCompletion(
            seq_len=seq_len,
            input_dim=input_dim
        ).to(device)
        # 模块2：多粒度特征提取
        self.feat_extractor = MultiGranularityFeatureExtractor(
            input_dim=input_dim,
            granularity_levels=granularity_levels
        ).to(device)
        # 模块3：销售预测器
        self.sales_predictor = SalesPredictor(feat_dim=feat_dim).to(device)
        # 模块4：域分类器
        self.domain_classifier = DomainClassifier(feat_dim=feat_dim).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        标准encoder-decoder接口，兼容训练循环调用方式。
        x_enc:      (batch, seq_len, input_dim)  — 编码器输入（源域）
        x_mark_enc: (batch, seq_len, time_dim)   — 编码器时间特征（暂不使用）
        x_dec:      (batch, label_len+pred_len, input_dim) — 解码器输入（目标域）
        x_mark_dec: (batch, label_len+pred_len, time_dim)  — 解码器时间特征（暂不使用）
        """
        # 将解码器输入中的预测部分作为目标域数据
        # x_dec前label_len步为已知，后pred_len步为待预测（已被置零）
        source_data = x_enc  # (batch, seq_len, input_dim)
        target_data = x_dec[:, :self.label_len, :]  # (batch, label_len, input_dim) 用已知的label部分

        # 自动生成mask（全0，表示无缺失；如有真实缺失逻辑可替换）
        source_mask = torch.zeros_like(source_data)
        target_mask = torch.zeros_like(target_data)

        # Step1: 数据补全（VAE）
        source_completed, s_mu, s_logvar = self.vae_completion(source_data, source_mask)
        target_completed, t_mu, t_logvar = self.vae_completion(target_data, target_mask)

        # Step2: 多粒度特征提取
        source_feats = self.feat_extractor(source_completed, is_source=True)
        target_feats = self.feat_extractor(target_completed, is_source=False)

        # Step3: 域对齐损失 + VAE损失（存入extra_loss供训练循环使用）
        loss_f = multi_level_alignment_loss(source_feats, target_feats)

        source_final_feat = source_feats[-1]
        target_final_feat = target_feats[-1]
        domain_feat = torch.cat([source_final_feat, target_final_feat], dim=0)
        domain_label = torch.cat([
            torch.zeros(source_final_feat.shape[0]),
            torch.ones(target_final_feat.shape[0])
        ]).to(source_data.device)
        domain_pred = self.domain_classifier(domain_feat).squeeze()
        loss_c = F.binary_cross_entropy_with_logits(domain_pred, domain_label)

        loss_vae_s = self.vae_completion.loss_fn(source_completed, source_data, s_mu, s_logvar)
        loss_vae_t = self.vae_completion.loss_fn(target_completed, target_data, t_mu, t_logvar)
        loss_vae = (loss_vae_s + loss_vae_t) / 2

        # 将辅助损失挂载到模型上，训练循环中会累加
        self.extra_loss = loss_f * 0.2 + loss_c * 0.2 + loss_vae * 0.1

        # Step4: 预测输出 —— 将target特征映射回 (batch, pred_len, input_dim)
        pred = self.sales_predictor(target_final_feat)  # (batch, 1)
        # 扩展为标准输出形状 (batch, pred_len, input_dim)
        pred = pred.unsqueeze(1).expand(-1, self.pred_len, self.input_dim)

        return pred

# # ===================== 6. 测试代码（验证框架流程） =====================
# if __name__ == "__main__":
#     # 设备选择（适配之前的CUDA问题）
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # 模拟输入：batch=8, seq_len=10, input_dim=5（销售数据维度）
#     batch_size = 8
#     seq_len = 10
#     input_dim = 5
#     # 源域/目标域数据 + 缺失掩码（1=缺失，0=存在）
#     source_data = torch.randn(batch_size, seq_len, input_dim).to(device)
#     source_mask = torch.randint(0, 2, (batch_size, seq_len, input_dim)).to(device)  # 随机缺失
#     target_data = torch.randn(batch_size, seq_len, input_dim).to(device)
#     target_mask = torch.randint(0, 2, (batch_size, seq_len, input_dim)).to(device)
#     target_label = torch.randn(batch_size).to(device)  # 销售预测标签
#
#     # 初始化模型（传入seq_len和input_dim）
#     model = HierDA(seq_len=seq_len, input_dim=input_dim, granularity_levels=3, device=device)
#     # 前向传播
#     outputs = model(source_data, source_mask, target_data, target_mask, target_label)
#
#     # 打印损失值，验证流程正常
#     print("Total Loss:", outputs['total_loss'].item())
#     print("Loss F (特征对齐):", outputs['loss_f'].item())
#     print("Loss P (销售预测):", outputs['loss_p'].item())
#     print("Loss C (域分类):", outputs['loss_c'].item())
#     print("Sales Prediction Shape:", outputs['sales_pred'].shape)