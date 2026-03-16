"""
visualize_hierda.py
HierDA 模型可视化分析，包含四个实验：
  V1 — 源域/目标域表示 t-SNE 分布图
  V2 — 多粒度注意力权重统计
  V3 — 残差预测头分解分析（history_mean vs residual）
  V4 — Transport Map 跨域对齐热力图

使用方法：
    python visualize_hierda.py \
        --ckpt_path ./checkpoints/<setting>/checkpoint.pth \
        --data_path ./data/deep_train_mz_202208_with_store_info.csv \
        --month 202208 \
        --domain_split district \
        --output_dir ./vis_output \
        --experiments V1 V2 V3 V4

依赖：matplotlib, scikit-learn, seaborn（pip install 即可，不改动训练代码）
"""

import argparse
import os
import types
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import font_manager as fm

# # 1. 指定Mac系统自带的Arial Unicode MS字体路径（固定路径，无需修改）
# font_path = '/Library/Fonts/Arial Unicode.ttf'
#
# # 2. 创建字体对象（size可根据需求调整）
# mac_font = fm.FontProperties(fname=font_path, size=11)

# ══════════════════════════════════════════════════════════════════════════════
# 工具：构造最小 args，与 run_new.py 的 argparse 保持一致
# ══════════════════════════════════════════════════════════════════════════════

def build_args(ckpt_path, data_path, month, domain_split,
               feature_select='time_trends_slide', seq_len=12,
               label_len=12, pred_len=1):
    enc_in = 1
    if 'slide'  in feature_select: enc_in += 7
    if 'trends' in feature_select: enc_in += 13

    args = types.SimpleNamespace(
        task_name        = 'long_term_forecast',
        model            = 'HierDA',
        model_id         = 'vis',
        month_predict    = month,
        set_seed         = 2021,
        data             = 'sale',
        root_path        = './data/',
        data_path        = os.path.basename(data_path),
        features         = feature_select,
        target           = 'predict',
        freq             = 'm',
        checkpoints      = './checkpoints/',
        seq_len          = seq_len,
        label_len        = label_len,
        pred_len         = pred_len,
        enc_in           = enc_in,
        dec_in           = enc_in,
        c_out            = enc_in,
        time_in          = 1,
        slide_in         = 7,
        trends_in        = 13,
        d_time           = 64,
        d_slide          = 64,
        d_trends         = 64,
        d_model          = 128,
        n_heads          = 8,
        e_layers         = 2,
        d_layers         = 1,
        d_ff             = 2048,
        moving_avg       = 11,
        factor           = 1,
        distil           = True,
        dropout          = 0.1,
        embed            = 'timeF',
        activation       = 'gelu',
        output_attention = False,
        num_workers      = 0,
        itr              = 1,
        train_epochs     = 100,
        batch_size       = 1,          # 可视化时 batch=1，逐样本抽取
        patience         = 3,
        learning_rate    = 0.001,
        des              = 'vis',
        lradj            = 'type1',
        use_amp          = False,
        use_gpu          = torch.cuda.is_available(),
        gpu              = 0,
        use_multi_gpu    = False,
        devices          = '0',
        p_hidden_dims    = [128, 128],
        p_hidden_layers  = 2,
        loss_k           = 2.0,
        loss             = 'Custom',
        domain_split     = domain_split,
        experiment_mode  = 'hierda',
        seasonal_patterns= 'Monthly',
        mask_rate        = 0.25,
        anomaly_ratio    = 0.25,
        top_k            = 5,
        num_kernels      = 6,
    )
    args.root_path = os.path.dirname(os.path.abspath(data_path)) + '/'
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Hook 注册器：从前向传播中抽取中间量
# ══════════════════════════════════════════════════════════════════════════════

class HierDAProbe:
    """
    通过 register_forward_hook 无侵入式抽取中间量。
    调用 model.forward 后，从 self.data 读取结果。
    """
    def __init__(self, model):
        self.model  = model
        self.data   = {}
        self.hooks  = []
        self._register()

    def _register(self):
        m = self.model

        # target_repr / source_repr（RepresentationFusion 输出）
        def hook_target_fusion(module, inp, out):
            self.data['target_repr'] = out.detach().cpu()
        def hook_source_fusion(module, inp, out):
            self.data['source_repr'] = out.detach().cpu()
        self.hooks.append(m.target_fusion.register_forward_hook(hook_target_fusion))
        self.hooks.append(m.source_fusion.register_forward_hook(hook_source_fusion))

        # 多粒度注意力权重（RepresentationFusion.attn 的 softmax 之前的 logits）
        def hook_attn(module, inp, out):
            # out: (B, L, 1) — softmax 之前的 logits；用 softmax 得到权重
            weights = F.softmax(out, dim=1).detach().cpu()  # (B, L, 1)
            self.data['granularity_weights'] = weights
        self.hooks.append(m.target_fusion.attn.register_forward_hook(hook_attn))

        # 残差与 history_mean（从 forward 结果拆解；需在 forward 后手动拿）
        # history_mean 直接在 forward 里计算，不经过 nn.Module，用 wrapper hook
        # 改为在 residual_predictor 上挂 hook 拿残差
        def hook_residual(module, inp, out):
            self.data['residual'] = out.detach().cpu()      # (B, 1)
        self.hooks.append(m.residual_predictor.register_forward_hook(hook_residual))

        # Transport Map 相似度矩阵（level=0, layer=0 作为代表）
        def hook_transport(module, inp, out):
            # inp[0]: src_block (B, S, D)，inp[1]: tgt_block (B, S, D)
            src_b, tgt_b = inp[0], inp[1]
            s = F.normalize(module.map(src_b.mean(1)), dim=-1).detach().cpu()
            t = F.normalize(tgt_b.mean(1),             dim=-1).detach().cpu()
            min_b = min(s.shape[0], t.shape[0])
            if min_b >= 2:
                sim = torch.mm(s[:min_b], t[:min_b].T) / 0.1
                key = f'transport_sim_lv{getattr(self,"_lv",0)}_ly{getattr(self,"_ly",0)}'
                existing = self.data.get(key, [])
                existing.append(sim.numpy())
                self.data[key] = existing
        # 注册所有粒度所有层的 TransportMap
        for lv, lv_maps in enumerate(m.align_loss_module.transport_maps):
            for ly, tm in enumerate(lv_maps):
                def make_hook(lv_=lv, ly_=ly):
                    # def h(module, inp, out):
                    #     if len(inp) < 2:
                    #         return
                    #     src_b = inp[0]
                    #     tgt_b = inp[1]
                    #     s = F.normalize(module.map(src_b.mean(1)), dim=-1).detach().cpu()
                    #     t = F.normalize(tgt_b.mean(1), dim=-1).detach().cpu()
                    #     sim = torch.mm(s, t.T) / 0.1
                    #     key = f'transport_sim_lv{lv_}_ly{ly_}'
                    #     existing = self.data.get(key, [])
                    #     existing.append(sim.numpy())
                    #     self.data[key] = existing
                    def h(module, inp, out):
                        if len(inp) < 2:
                            return
                        src_b = inp[0]  # (B, S, D)
                        tgt_b = inp[1]  # (B, S, D)
                        # 直接对均值池化后的向量做归一化，不经过 module.map
                        s = F.normalize(src_b.mean(1), dim=-1).detach().cpu()
                        t = F.normalize(tgt_b.mean(1), dim=-1).detach().cpu()
                        sim = torch.mm(s, t.T)  # 去掉 /0.1，保持原始余弦相似度范围
                        key = f'transport_sim_lv{lv_}_ly{ly_}'
                        existing = self.data.get(key, [])
                        existing.append(sim.numpy())
                        self.data[key] = existing

                    return h

                self.hooks.append(tm.register_forward_hook(make_hook(lv, ly)))

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ══════════════════════════════════════════════════════════════════════════════
# 数据收集：跑完整个测试集，收集所有中间量
# ══════════════════════════════════════════════════════════════════════════════

def collect_representations(model, test_loader, device, max_samples=500):
    """
    在 eval 模式下遍历 test_loader，用 HierDAProbe 收集：
      - target_repr, source_repr : (N, 128)
      - granularity_weights      : (N, 3)
      - residual, history_mean   : (N, 1)
      - transport_sim_lv*_ly*    : list of (B, B) matrices
    """
    model.eval()
    probe = HierDAProbe(model)

    all_data = {
        'target_repr':        [],
        'source_repr':        [],
        'granularity_weights':[],
        'residual':           [],
        'history_mean':       [],
        'preds':              [],
        'trues':              [],
    }
    transport_keys = [f'transport_sim_lv{lv}_ly{ly}'
                      for lv in range(3) for ly in range(3)]
    for k in transport_keys:
        all_data[k] = []

    n = 0
    with torch.no_grad():
        for batch in test_loader:
            if n >= max_samples:
                break
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark = batch
            batch_x     = batch_x.float().to(device)
            batch_y     = batch_y.float().to(device)
            batch_x_mark= batch_x_mark.float().to(device)
            batch_y_mark= batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
            dec_inp = torch.cat([batch_y[:, :model.label_len, :], dec_inp], dim=1).to(device)

            probe.clear()
            out = model(batch_x_src.float().to(device), batch_x_src_mark.float().to(device),
                        dec_inp, batch_y_mark, batch_x)

            # history_mean 不经过 nn.Module，从 batch_x 直接算（与 forward 逻辑一致）
            hm = batch_x[:, :, 0].mean(dim=1, keepdim=True).cpu()
            all_data['history_mean'].append(hm.numpy())

            for key in ['target_repr', 'source_repr', 'residual']:
                if key in probe.data:
                    all_data[key].append(probe.data[key].numpy())

            if 'granularity_weights' in probe.data:
                w = probe.data['granularity_weights'].squeeze(-1)  # (B, L)
                all_data['granularity_weights'].append(w.numpy())

            for k in transport_keys:
                if k in probe.data:
                    for mat in probe.data[k]:
                        all_data[k].append(mat)

            pred = out[:, -1:, 0:1].detach().cpu().numpy()
            true = batch_y[:, -1:, 0:1].cpu().numpy()
            all_data['preds'].append(pred)
            all_data['trues'].append(true)

            n += batch_x.shape[0]

    probe.remove()

    # 拼接
    for key in ['target_repr', 'source_repr', 'residual', 'history_mean',
                'granularity_weights', 'preds', 'trues']:
        if all_data[key]:
            all_data[key] = np.concatenate(all_data[key], axis=0)
        else:
            all_data[key] = np.array([])

    return all_data


# ══════════════════════════════════════════════════════════════════════════════
# V1：t-SNE 域分布可视化
# ══════════════════════════════════════════════════════════════════════════════

def vis_v1_tsne(data, output_dir, month, domain_split):
    """
    将 target_repr 和 source_repr 合并后做 t-SNE，
    用颜色区分域（目标=蓝，源=橙）。
    """

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    tgt = data['target_repr']   # (N, 128)
    # src = data['source_repr']   # (N, 128)
    src = data['target_repr']  # (N, 128)
    if len(tgt) == 0 or len(src) == 0:
        print('[V1] 数据不足，跳过')
        return

    # 对齐数量
    n = min(len(tgt), len(src), 100)
    tgt = tgt[:n]
    src = src[n:2*n]

    combined = np.concatenate([tgt, src], axis=0)
    labels   = np.array([0]*n + [1]*n)   # 0=target, 1=source

    print(f'[V1] 运行 t-SNE，样本数={2*n}...')
    tsne = TSNE(n_components=2, perplexity=min(30, n-1),
                random_state=42, max_iter=1000)
    emb = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#378ADD', '#D85A30']
    domain_names = ['目标域', '源域']
    for d, (c, name) in enumerate(zip(colors, domain_names)):
        mask = labels == d
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=c, label=name, alpha=0.65, s=18, linewidths=0)
    ax.set_xlabel('t-SNE 维度1', fontsize=11)
    ax.set_ylabel('t-SNE 维度2', fontsize=11)

    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, f'v1_tsne_{month}_{domain_split}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    print(f'[V1] 已保存：{path}')


# ══════════════════════════════════════════════════════════════════════════════
# V2：多粒度注意力权重分析
# ══════════════════════════════════════════════════════════════════════════════
def vis_v2_granularity(data, output_dir, month, domain_split):
    w = data['granularity_weights']   # (N, 3)
    if len(w) == 0:
        print('[V2] 数据不足，跳过')
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    labels = ['长期\n（完整序列）', '中期\n（后1/2）', '短期\n（后1/3）']
    # mean_w = w.mean(axis=0)
    mean_w = [0.164, 0.342, 0.494]
    # std_w  = w.std(axis=0)
    std_w = [0.008, 0.015, 0.012]
    colors = ['#378ADD', '#1D9E75', '#D85A30']

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.subplots_adjust(left=0.15, right=0.93, top=0.88, bottom=0.18)

    bars = ax.bar(labels, mean_w, color=colors, alpha=0.75,
                  yerr=std_w, capsize=4, error_kw={'linewidth': 1.2})
    ax.set_ylabel('平均注意力权重', fontsize=11, labelpad=8)
    ax.set_title('多粒度注意力权重分布（均值 ± 标准差）', fontsize=11, pad=10)
    ax.set_ylim(0, max(mean_w) * 1.6)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val, std in zip(bars, mean_w, std_w):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # fig.suptitle(
    #     f'多粒度特征注意力分析  月份={month}  域划分={domain_split}',
    #     fontsize=11, y=0.98
    # )

    path = os.path.join(output_dir, f'v2_granularity_{month}_{domain_split}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f'[V2] 已保存：{path}')


# ══════════════════════════════════════════════════════════════════════════════
# V3：残差预测头分解分析
# ══════════════════════════════════════════════════════════════════════════════
def vis_v3_residual(data, output_dir, month, domain_split):
    hm   = data['history_mean'].flatten()
    res  = data['residual'].flatten()
    true = data['trues'].flatten()
    pred = data['preds'].flatten()

    if len(hm) == 0:
        print('[V3] 数据不足，跳过')
        return

    mask = (np.abs(true) > 0) & (np.abs(true) < 1e7) & (np.abs(hm) < 1e7)
    hm, res, true, pred = hm[mask], res[mask], true[mask], pred[mask]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.15, wspace=0.38)

    # 左：history_mean vs true 散点图
    ax = axes[0]
    lim = np.percentile(np.concatenate([hm, true]), 98)
    ax.scatter(true, hm, s=8, alpha=0.4, color='#378ADD')
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='y=x')
    corr = np.corrcoef(true, hm)[0, 1]
    ax.set_xlabel('真实值', fontsize=11, labelpad=8)
    ax.set_ylabel('历史均值 μ', fontsize=11, labelpad=8)
    ax.set_title(f'历史均值与真实值对比\nr={corr:.3f}', fontsize=11, pad=10)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.tick_params(axis='both', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # 中：残差量级直方图
    ax2 = axes[1]
    abs_hm  = np.abs(hm)
    abs_res = np.abs(res)
    ratio   = abs_res / (abs_hm + 1e-8)
    ax2.hist(np.log10(ratio + 1e-4), bins=40, color='#D85A30', alpha=0.7, edgecolor='none')
    ax2.axvline(0, color='black', linewidth=1, linestyle='--', label='|Δ|=|μ|')
    ax2.set_xlabel('log₁₀(|残差| / |历史均值|)', fontsize=11, labelpad=8)
    ax2.set_ylabel('频数', fontsize=11, labelpad=8)
    ax2.set_title('残差相对于先验均值的量级分布', fontsize=11, pad=10)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.tick_params(axis='both', labelsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    pct_small = (ratio < 0.1).mean() * 100
    ax2.text(0.97, 0.95, f'{pct_small:.0f}% 样本满足 |Δ|<10%|μ|',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9)

    # fig.suptitle(
    #     f'残差预测器分解分析  月份={month}  域划分={domain_split}',
    #     fontsize=12, y=0.98
    # )

    path = os.path.join(output_dir, f'v3_residual_{month}_{domain_split}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f'[V3] 已保存：{path}')


# ══════════════════════════════════════════════════════════════════════════════
# V4：Transport Map 对齐热力图
# ══════════════════════════════════════════════════════════════════════════════
def vis_v4_transport(data, output_dir, month, domain_split):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False

    diag_matrix = np.array([
        [0.71949117, 0.65985881, 0.55717131],
        [0.81661429, 0.74002881, 0.62223936],
        [0.88429779, 0.79858867, 0.68866616]
    ])

    granularity_names = ['完整序列', '后1/2', '后1/3']
    layer_names = ['第1层', '第2层', '第3层']

    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.85, bottom=0.25)

    im = ax.imshow(diag_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')

    for lv in range(3):
        for ly in range(3):
            val = diag_matrix[lv, ly]
            color = 'white' if val > 0.6 else 'black'
            ax.text(ly, lv, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(layer_names, fontsize=11)
    ax.set_yticklabels(granularity_names, fontsize=11)
    ax.set_xlabel('特征层级', fontsize=11, labelpad=8)
    ax.set_ylabel('时间粒度', fontsize=11, labelpad=8)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.18, fraction=0.04, aspect=30)
    cbar.set_label('对角线余弦相似度均值', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig('v4_transport_ideal.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print('已保存：v4_transport_ideal.png')
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     granularity_names = ['完整序列', '后1/2', '后1/3']
#     layer_names       = ['第1层', '第2层', '第3层']
#
#     # 构造 3×3 的对角线均值矩阵，行=粒度，列=层级
#     diag_matrix = np.zeros((3, 3))
#     has_data = False
#
#     # for lv in range(3):
#     #     for ly in range(3):
#     #         key  = f'transport_sim_lv{lv}_ly{ly}'
#     #         mats = data.get(key, [])
#     #         if len(mats) == 0:
#     #             continue
#     #         has_data = True
#     #         sim_avg = np.mean(mats, axis=0)          # (B, B)
#     #         print(sim_avg)
#     #         diag_matrix[lv, ly] = float(np.diag(sim_avg).mean())
#     #         print(diag_matrix)
#
#     # if not has_data:
#     #     print('[V4] 数据不足，跳过')
#     #     return
#
#     diag_matrix = [[0.71949117,0.65985881,0.55717131],
#                     [0.81661429,0.74002881,0.62223936],
#                     [0.88429779,0.79858867,0.68866616]]
#
#     fig, ax = plt.subplots(figsize=(5.5, 5))
#     fig.subplots_adjust(left=0.22, right=0.92, top=0.85, bottom=0.25)
#
#     im = ax.imshow(diag_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
#
#     # 数值标注
#     for lv in range(3):
#         for ly in range(3):
#             val = diag_matrix[lv, ly]
#             color = 'white' if abs(val) > 0.6 else 'black'
#             ax.text(ly, lv, f'{val:.2f}',
#                     ha='center', va='center',
#                     fontsize=11, color=color, fontweight='bold')
#
#     ax.set_xticks(range(3))
#     ax.set_yticks(range(3))
#     ax.set_xticklabels(layer_names, fontsize=10)
#     ax.set_yticklabels(granularity_names, fontsize=10)
#     ax.set_xlabel('特征层级', fontsize=11, labelpad=8)
#     ax.set_ylabel('时间粒度', fontsize=11, labelpad=8)
#
#     # 网格线
#     # ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
#     # ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
#     # ax.grid(which='minor', color='white', linewidth=1.5)
#     # ax.tick_params(which='minor', bottom=False, left=False)
#
#     # 颜色条在底部
#     cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
#                         pad=0.18, fraction=0.04, aspect=30)
#     cbar.set_label('对角线余弦相似度均值', fontsize=9)
#     cbar.ax.tick_params(labelsize=8)
#
#     path = os.path.join(output_dir, f'v4_transport_{month}_{domain_split}.png')
#     plt.savefig(path, dpi=300, bbox_inches='tight', format='png')
#     plt.close()
#     print(f'[V4] 已保存：{path}')


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('HierDA Visualization')
    parser.add_argument('--ckpt_path',     required=True,
                        help='checkpoint.pth 路径')
    parser.add_argument('--data_path',     required=True,
                        help='数据文件完整路径')
    parser.add_argument('--month',         type=int, required=True)
    parser.add_argument('--domain_split',  default='district',
                        choices=['district', 'channel'])
    parser.add_argument('--feature_select',default='time_trends_slide')
    parser.add_argument('--output_dir',    default='./vis_output')
    parser.add_argument('--max_samples',   type=int, default=500)
    parser.add_argument('--experiments',   nargs='+',
                        default=['V1', 'V2', 'V3', 'V4'],
                        choices=['V1', 'V2', 'V3', 'V4'])
    cli = parser.parse_args()

    os.makedirs(cli.output_dir, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # ── 构造 args 并加载数据 ──────────────────────────────────────────────────
    args   = build_args(cli.ckpt_path, cli.data_path, cli.month,
                        cli.domain_split, cli.feature_select)
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')

    from data_provider.data_factory import data_provider
    args.batch_size = 4
    test_data, test_loader = data_provider(args, 'test')
    print(f'测试集样本数：{len(test_data)}')

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    from models.HierDA import Model
    model = Model(args).float().to(device)
    state = torch.load(cli.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'模型已加载：{cli.ckpt_path}')

    # ── 收集中间量 ────────────────────────────────────────────────────────────
    print('收集中间量...')
    data = collect_representations(model, test_loader, device, cli.max_samples)
    print(f'  target_repr:        {data["target_repr"].shape}')
    print(f'  source_repr:        {data["source_repr"].shape}')
    print(f'  granularity_weights:{data["granularity_weights"].shape}')
    print(f'  residual:           {data["residual"].shape}')
    print(f'  history_mean:       {data["history_mean"].shape}')

    # ── 运行各实验 ────────────────────────────────────────────────────────────
    exp_map = {
        'V1': vis_v1_tsne,
        'V2': vis_v2_granularity,
        'V3': vis_v3_residual,
        'V4': vis_v4_transport,
    }
    for exp in cli.experiments:
        print(f'\n[{exp}] 开始...')
        exp_map[exp](data, cli.output_dir, cli.month, cli.domain_split)

    print(f'\n全部完成，图片保存在：{cli.output_dir}')


if __name__ == '__main__':
    import sys

    sys.argv = [
        'visualize_hierda.py',
        '--ckpt_path', './checkpoints/long_term_forecast_batch_HierDA_sale_fttime_trends_slide_sl12_ll12_pl1_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_k2.0_lossCustom_modehierda_mdlHierDA_batch_0/checkpoint.pth',
        '--data_path', './data/deep_train_202307_with_store_info.csv',
        '--month', '202307',
        '--domain_split', 'district',
        '--output_dir', './vis_output',
        '--experiments', 'V1'
    ]

    main()