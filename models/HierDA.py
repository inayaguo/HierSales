import torch
import torch.nn as nn
import torch.nn.functional as F

from setfeat_network import SetFeat4


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReverseFunction.apply(x, alpha)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        # x: [B, L, C] -> [B, L*C]
        b, l, c = x.shape
        x_flat = x.reshape(b, l * c)
        stats = self.fc_enc(x_flat)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_flat = self.fc_dec(z)
        recon = recon_flat.reshape(b, l, c)

        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, recon_loss, kld


class SetFeatExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        n_filters = [hidden_dim, hidden_dim, hidden_dim]
        n_heads = [num_heads, num_heads, num_heads]
        self.backbone = SetFeat4(input_dim, n_filters, n_heads)

    def forward(self, x):
        # x: [B, L, C] -> [B, L_feat, H] -> pooled [B, H]
        feat_seq = self.backbone(x)
        feat = feat_seq.mean(dim=1)
        return feat


class Model(nn.Module):
    """
    层级域适应销量预测模型：
    - VAE 用于对目标域输入进行数据补全；
    - SetFeatExtractor 作为源域/目标域特征抽取器；
    - Label Predictor 为简单 FNN；
    - Domain Classifier 为简单 FNN，通过梯度反转实现对抗式域适应。
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        input_dim = configs.enc_in
        hidden_dim = configs.d_model
        num_heads = max(1, configs.n_heads // 2)

        # VAE 数据补全（按序列整体展开）
        self.vae = VAE(input_dim=input_dim * configs.seq_len, latent_dim=hidden_dim // 4)

        # 源域 & 目标域特征抽取器（基于 SetFeat4）
        self.source_extractor = SetFeatExtractor(input_dim, hidden_dim, num_heads)
        self.target_extractor = SetFeatExtractor(input_dim, hidden_dim, num_heads)

        feat_dim = hidden_dim * 2

        # Label Predictor（销量预测）
        self.label_predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pred_len * self.c_out),
        )

        # Domain Classifier（区分“源/目标”，使用梯度反转）
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        # 域对抗与 VAE 损失的权重
        self.da_alpha = 1.0
        self.lambda_da = 0.1
        self.lambda_vae = 0.1
        self.kld_weight = 0.001

        self.extra_loss = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        与现有 Exp_Long_Term_Forecast 接口保持一致：
        x_enc: [B, seq_len, enc_in]
        返回: [B, pred_len, c_out]
        """
        if self.task_name not in ["long_term_forecast", "short_term_forecast"]:
            raise ValueError(f"HierDA Model 目前只支持 long_term_forecast 任务, 当前为: {self.task_name}")

        b, l, c = x_enc.shape

        # VAE 数据补全：将同一批次的原始输入视为“目标域”，VAE 重构视为“源域”
        x_flat = x_enc.reshape(b, l * c)
        # 将 VAE 的输入维度适配为 [B, L*C]
        recon_flat, recon_loss, kld = self.vae(x_flat.view(b, 1, -1))
        x_source = recon_flat.view(b, l, c)
        x_target = x_enc

        # 源域/目标域特征抽取
        feat_source = self.source_extractor(x_source)
        feat_target = self.target_extractor(x_target)
        feat_joint = torch.cat([feat_source, feat_target], dim=-1)

        # Label Predictor：进行销量预测
        preds_flat = self.label_predictor(feat_joint)
        preds = preds_flat.view(b, self.pred_len, self.c_out)

        # 仅在训练阶段计算域对抗与 VAE 额外损失
        if self.training:
            # 域标签：0 为“源域”(VAE 重构)，1 为“目标域”(原始输入)
            feat_da = torch.cat([feat_source, feat_target], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(b, dtype=torch.long, device=x_enc.device),
                    torch.ones(b, dtype=torch.long, device=x_enc.device),
                ],
                dim=0,
            )

            feat_da_rev = grad_reverse(feat_da, self.da_alpha)
            domain_logits = self.domain_classifier(feat_da_rev)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)

            vae_loss = recon_loss + self.kld_weight * kld
            self.extra_loss = self.lambda_da * domain_loss + self.lambda_vae * vae_loss
        else:
            self.extra_loss = None

        return preds