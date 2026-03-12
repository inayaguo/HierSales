import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class RoutingLayer(nn.Module):
    """
    Routing Layer that combines MoE gating weights with encoder inputs to direct them to specific sub-networks.
    """
    def __init__(self, num_experts, d_model):
        super(RoutingLayer, self).__init__()
        self.num_experts = num_experts
        self.linear_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, moe_gates):
        """
        Args:
            x: Input tensor, shape [B, L, D]
            moe_gates: Gating weights, shape [B, L, num_experts]

        Returns:
            Routed output tensor, shape [B, L, D]
        """
        # Ensure gates are normalized
        moe_gates = self.softmax(moe_gates)

        # Process input through each expert and weight by gating values
        outputs = []
        for i in range(self.num_experts):
            expert_output = self.linear_layers[i](x)  # [B, L, D]
            weighted_output = expert_output * moe_gates[..., i].unsqueeze(-1)  # [B, L, D]
            outputs.append(weighted_output)

        # Combine outputs from all experts
        routed_output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)  # [B, L, D]

        return routed_output


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module with gating network based on input similarity.
    """
    def __init__(self, input_dim, num_experts, expert_hidden_dim):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)  # 归一化权重
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x_flat = x.reshape(batch_size * seq_len, input_dim)  # [B*L, D]
        gates = self.gate(x_flat)  # [B*L, num_experts]
        outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*L, num_experts, D]
        weighted_output = torch.sum(gates.unsqueeze(-1) * outputs, dim=1)  # [B*L, D]
        weighted_output = weighted_output.reshape(batch_size, seq_len, input_dim)  # [B, L, D]
        gates = gates.reshape(batch_size, seq_len, self.num_experts)  # [B, L, num_experts]
        return weighted_output, gates

class Model(nn.Module):
    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomposition
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.routing_layer = RoutingLayer(num_experts=2, d_model=configs.d_model)

        # MoE structure
        self.moe = MoE(input_dim=configs.d_model, num_experts=2, expert_hidden_dim=64)

        # Encoder and Decoder initialization
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def save_embedding_weights(self):
        """新增方法：保存当前迭代的模型参数权重（训练阶段调用）"""
        # 保存编码器embedding参数
        self.enc_embedding.save_embedding_params("enc_dataembedding_params.csv")
        # 保存解码器embedding参数
        self.dec_embedding.save_embedding_params("dec_dataembedding_params.csv")

    def print_feature_contribution(self, configs):
        """新增方法：打印当前迭代的特征贡献度（训练阶段调用）"""
        # 提取特征贡献度并打印（用于销量预测特征分析）
        enc_core_weights = self.enc_embedding.get_core_weights()
        print("\n📊 编码器DataEmbedding特征贡献度（输入特征维度：{}）:".format(configs.enc_in))
        for idx, contrib in enumerate(enc_core_weights["feature_contribution"]):
            print(f"  特征{idx + 1}: {contrib:.6f}")

        dec_core_weights = self.dec_embedding.get_core_weights()
        print("\n📊 解码器DataEmbedding特征贡献度（输入特征维度：{}）:".format(configs.dec_in))
        for idx, contrib in enumerate(dec_core_weights["feature_contribution"]):
            print(f"  特征{idx + 1}: {contrib:.6f}")

        return enc_core_weights, dec_core_weights

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Decomposition
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # MoE Routing
        moe_output, moe_gates = self.moe(enc_out)  # [B, L, D] 和 [B, L, num_experts]

        # Apply RoutingLayer
        routed_enc_out = self.routing_layer(enc_out, moe_gates)  # [B, L, D]

        # Encoder
        enc_out, attns = self.encoder(routed_enc_out, attn_mask=None)

        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        dec_out = seasonal_part + trend_part

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
