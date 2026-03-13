"""
run_ablation.py — HierDA 消融实验批量入口

在 run_new.py 的 batch_experiment 基础上，新增三个维度的控制参数：
    ablation_mode       : Group1 VAE / Group4 对齐损失 / Group6 预测头
    granularity_levels  : Group2 多粒度
    freq_mode           : Group3 时频

所有变体通过同一套训练/评估流程，结果写入同一 CSV，方便横向对比。
"""

import random
import argparse
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join("output_hiersales", "result")
RESULT_CSV = os.path.join(RESULT_DIR, "ablation_results.csv")
ERROR_LOG  = os.path.join(RESULT_DIR, "ablation_error.log")

RESULT_COLUMNS = [
    "run_time",
    "month",
    "domain_split",
    "ablation_group",    # 'G1_vae' / 'G2_granularity' / 'G3_freq' / 'G4_align' / 'G6_pred'
    "ablation_variant",  # 具体变体名，如 'full' / 'wo_vae' / 'time_only' 等
    "ablation_mode",
    "granularity_levels",
    "freq_mode",
    "feature_select",
    "d_model",
    "e_layers",
    "batch_size",
    "learning_rate",
    "loss_type",
    "loss_k",
    "lambda_f",
    "lambda_c",
    "lambda_vae",
    "seq_len",
    "label_len",
    "pred_len",
    "seed",
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "store_acc",
    "phase",
]

# ══════════════════════════════════════════════════════════════════════════════
# 消融变体定义
# 每个 group 是一个 list，每个元素是一个 dict，描述一组消融参数。
# 'full' 变体必须出现在每个 group 中作为组内基准（只跑一次即可）。
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_GROUPS = {

    # ── Group1: VAE 数据补全 ───────────────────────────────────────────────
    "G1_vae": [
        {"ablation_mode": "full",       "granularity_levels": 3, "freq_mode": "both"},  # 完整模型（基准）
        {"ablation_mode": "wo_vae",     "granularity_levels": 3, "freq_mode": "both"},  # 跳过 VAE
        {"ablation_mode": "vae_noloss", "granularity_levels": 3, "freq_mode": "both"},  # VAE 路径存在但 loss 不计
        {"ablation_mode": "vae_detach", "granularity_levels": 3, "freq_mode": "both"},  # VAE detach，阻断梯度
    ],

    # ── Group2: 多粒度数量 ────────────────────────────────────────────────
    "G2_granularity": [
        {"ablation_mode": "full", "granularity_levels": 3, "freq_mode": "both"},   # 三粒度（完整基准）
        {"ablation_mode": "full", "granularity_levels": 2, "freq_mode": "both"},   # 双粒度（细 + 中）
        {"ablation_mode": "full", "granularity_levels": 1, "freq_mode": "both"},   # 单粒度（仅完整序列）
    ],

    # ── Group3: 时域 vs 频域 ──────────────────────────────────────────────
    "G3_freq": [
        {"ablation_mode": "full", "granularity_levels": 3, "freq_mode": "both"},       # 时 + 频（完整基准）
        {"ablation_mode": "full", "granularity_levels": 3, "freq_mode": "time_only"},  # 仅时域
        {"ablation_mode": "full", "granularity_levels": 3, "freq_mode": "freq_only"},  # 仅频域
    ],

    # ── Group4: 对齐损失 ──────────────────────────────────────────────────
    "G4_align": [
        {"ablation_mode": "full",         "granularity_levels": 3, "freq_mode": "both"},  # Transport + GRL（完整基准）
        {"ablation_mode": "wo_transport", "granularity_levels": 3, "freq_mode": "both"},  # 去掉 Transport Map
        {"ablation_mode": "wo_grl",       "granularity_levels": 3, "freq_mode": "both"},  # 去掉 GRL 域分类器
        {"ablation_mode": "wo_da",        "granularity_levels": 3, "freq_mode": "both"},  # 去掉全部域适应损失
        {"ablation_mode": "no_source",    "granularity_levels": 3, "freq_mode": "both"},  # 源域数据完全不参与
    ],

    # ── Group6: 预测头设计 ────────────────────────────────────────────────
    "G6_pred": [
        {"ablation_mode": "full",        "granularity_levels": 3, "freq_mode": "both"},  # 残差+先验+scale（完整基准）
        {"ablation_mode": "direct_pred", "granularity_levels": 3, "freq_mode": "both"},  # 直接预测，无先验
        {"ablation_mode": "prior_only",  "granularity_levels": 3, "freq_mode": "both"},  # 仅输出历史均值
        {"ablation_mode": "no_scale",    "granularity_levels": 3, "freq_mode": "both"},  # 有残差+先验，无 output_scale
    ],
}

# 'full' 变体在各 group 均出现，为避免重复训练同一配置，
# 统一只在 G1_vae 中真正训练 full，其余 group 复用结果（直接读 CSV 对比）。
# 若希望每组独立跑 full 作为 sanity check，设为 False。
DEDUPLICATE_FULL = True


# ══════════════════════════════════════════════════════════════════════════════
# 结果写入
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_csv():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULT_CSV, index=False)


def write_result(row: dict):
    _ensure_csv()
    row.setdefault("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS).to_csv(
        RESULT_CSV, mode="a", header=False, index=False
    )


def write_error(msg: str):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 单次实验
# ══════════════════════════════════════════════════════════════════════════════

def run_one_ablation(
    *,
    # 实验标识
    ablation_group: str,
    ablation_variant: str,
    # 消融参数
    ablation_mode: str       = "full",
    granularity_levels: int  = 3,
    freq_mode: str           = "both",
    lambda_f: float          = 0.001,
    lambda_c: float          = 0.001,
    lambda_vae: float        = 0.0001,
    # 数据 & 超参
    feature_select: str,
    dimension_model: int,
    encoder_layers: int,
    batch_size: int,
    learning_rate: float,
    is_training: int,
    data_input: str,
    seed: int,
    month: int,
    domain_split: str        = "district",
    loss_k: float            = 2.0,
    loss_type: str           = "Custom",
    seq_len: int             = 12,
    label_len: int           = 12,
    pred_len: int            = 1,
    train_epochs: int        = 100,
    patience: int            = 3,
):
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    parser = argparse.ArgumentParser(add_help=False)

    # 基础
    parser.add_argument("--task_name",          default="long_term_forecast")
    parser.add_argument("--is_training",        type=int,   default=is_training)
    parser.add_argument("--model_id",           default="ablation")
    parser.add_argument("--model",              default="HierDA")
    parser.add_argument("--month_predict",      type=int,   default=month)
    parser.add_argument("--set_seed",           type=int,   default=seed)

    # 消融参数（透传给 Model.__init__ 和 DomainFeatureExtractor）
    parser.add_argument("--ablation_mode",      default=ablation_mode)
    parser.add_argument("--granularity_levels", type=int,   default=granularity_levels)
    parser.add_argument("--freq_mode",          default=freq_mode)
    parser.add_argument("--lambda_f",           type=float, default=lambda_f)
    parser.add_argument("--lambda_c",           type=float, default=lambda_c)
    parser.add_argument("--lambda_vae",         type=float, default=lambda_vae)

    # 损失 & 域划分 & 实验模式
    parser.add_argument("--loss_k",             type=float, default=loss_k)
    parser.add_argument("--loss",               default=loss_type)
    parser.add_argument("--domain_split",       default=domain_split)
    parser.add_argument("--experiment_mode",    default="hierda")

    # 数据
    parser.add_argument("--data",               default="sale")
    parser.add_argument("--root_path",          default="./data/")
    parser.add_argument("--data_path",          default=data_input)
    parser.add_argument("--features",           default=feature_select)
    parser.add_argument("--target",             default="predict")
    parser.add_argument("--freq",               default="m")
    parser.add_argument("--checkpoints",        default="./checkpoints/")

    # 序列长度
    parser.add_argument("--seq_len",            type=int, default=seq_len)
    parser.add_argument("--label_len",          type=int, default=label_len)
    parser.add_argument("--pred_len",           type=int, default=pred_len)
    parser.add_argument("--seasonal_patterns",  default="Monthly")
    parser.add_argument("--mask_rate",          type=float, default=0.25)
    parser.add_argument("--anomaly_ratio",      type=float, default=0.25)

    # 特征维度
    enc_in = 1
    if "slide"  in feature_select: enc_in += 7
    if "trends" in feature_select: enc_in += 13
    for arg in ["enc_in", "dec_in", "c_out"]:
        parser.add_argument(f"--{arg}", type=int, default=enc_in)
    for arg, val in [("time_in", 1), ("slide_in", 7), ("trends_in", 13),
                     ("d_time", 64), ("d_slide", 64), ("d_trends", 64)]:
        parser.add_argument(f"--{arg}", type=int, default=val)

    # 模型结构
    parser.add_argument("--top_k",             type=int,   default=5)
    parser.add_argument("--num_kernels",        type=int,   default=6)
    parser.add_argument("--d_model",            type=int,   default=dimension_model)
    parser.add_argument("--n_heads",            type=int,   default=8)
    parser.add_argument("--e_layers",           type=int,   default=encoder_layers)
    parser.add_argument("--d_layers",           type=int,   default=1)
    parser.add_argument("--d_ff",               type=int,   default=2048)
    parser.add_argument("--moving_avg",         type=int,   default=11)
    parser.add_argument("--factor",             type=int,   default=1)
    parser.add_argument("--distil",             action="store_false", default=True)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--embed",              default="timeF")
    parser.add_argument("--activation",         default="gelu")
    parser.add_argument("--output_attention",   action="store_true", default=False)

    # 优化
    parser.add_argument("--num_workers",        type=int,   default=0)
    parser.add_argument("--itr",                type=int,   default=1)
    parser.add_argument("--train_epochs",       type=int,   default=train_epochs)
    parser.add_argument("--batch_size",         type=int,   default=batch_size)
    parser.add_argument("--patience",           type=int,   default=patience)
    parser.add_argument("--learning_rate",      type=float, default=learning_rate)
    parser.add_argument("--des",                default="abl")
    parser.add_argument("--lradj",              default="type1")
    parser.add_argument("--use_amp",            action="store_true", default=False)

    # GPU
    parser.add_argument("--use_gpu",            type=bool,  default=True)
    parser.add_argument("--gpu",                type=int,   default=0)
    parser.add_argument("--use_multi_gpu",      action="store_true", default=False)
    parser.add_argument("--devices",            default="0")
    parser.add_argument("--p_hidden_dims",      type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers",    type=int, default=2)

    args = parser.parse_args([])
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    # checkpoint 路径包含所有消融维度，确保不同变体互不覆盖
    setting = (
        f"ablation_HierDA_sale"
        f"_sl{seq_len}_ll{label_len}_pl{pred_len}"
        f"_dm{dimension_model}_el{encoder_layers}"
        f"_k{loss_k}_loss{loss_type}"
        f"_abl{ablation_mode}_g{granularity_levels}_f{freq_mode}"
        f"_m{month}_split{domain_split}_0"
    )

    exp = Exp_Long_Term_Forecast(args)

    base_info = dict(
        month               = month,
        domain_split        = domain_split,
        ablation_group      = ablation_group,
        ablation_variant    = ablation_variant,
        ablation_mode       = ablation_mode,
        granularity_levels  = granularity_levels,
        freq_mode           = freq_mode,
        feature_select      = feature_select,
        d_model             = dimension_model,
        e_layers            = encoder_layers,
        batch_size          = batch_size,
        learning_rate       = learning_rate,
        loss_type           = loss_type,
        loss_k              = loss_k,
        lambda_f            = lambda_f,
        lambda_c            = lambda_c,
        lambda_vae          = lambda_vae,
        seq_len             = seq_len,
        label_len           = label_len,
        pred_len            = pred_len,
        seed                = seed,
    )

    if is_training:
        exp.train(setting)
        torch.cuda.empty_cache()
        metrics = exp.test(setting, test=0, return_metrics=True)
        phase = "train+test"
    else:
        metrics = exp.test(setting, test=1, return_metrics=True)
        phase = "test"

    torch.cuda.empty_cache()

    if metrics:
        write_result({**base_info, **metrics, "phase": phase})
        print(f"  ✅ 写入完成 → {RESULT_CSV}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 消融批量主函数
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_experiment(config: dict):
    """
    遍历所有消融组 × 月份 × 域划分，每个变体独立训练并写入结果。

    DEDUPLICATE_FULL=True 时，'full' 变体只在第一个包含它的 group（G1_vae）
    里真正运行，其余 group 的 'full' 行直接跳过（结果已存在 CSV 中）。
    """
    _ensure_csv()

    months        = config.get("months",        [])
    domain_splits = config.get("domain_splits", ["district"])
    groups        = config.get("groups",        list(ABLATION_GROUPS.keys()))

    # 统计总任务数
    total = 0
    full_trained = set()   # 记录已训练的 full 变体（month, domain_split 组合）
    for grp in groups:
        for v in ABLATION_GROUPS[grp]:
            for m in months:
                for ds in domain_splits:
                    total += 1

    done = 0
    print(f"\n{'='*70}")
    print(f"  HierDA 消融实验启动：共 {total} 组（含 full 复用去重）")
    print(f"  组别：{groups}")
    print(f"  结果文件：{RESULT_CSV}")
    print(f"{'='*70}\n")

    for grp in groups:
        variants = ABLATION_GROUPS[grp]
        for variant_cfg in variants:
            ablation_mode      = variant_cfg["ablation_mode"]
            granularity_levels = variant_cfg["granularity_levels"]
            freq_mode          = variant_cfg["freq_mode"]

            # 生成易读的变体名
            variant_name = ablation_mode
            if ablation_mode == "full":
                variant_name = f"full_g{granularity_levels}_{freq_mode}"

            for month in months:
                data_input = config["data_template"].format(month=month)
                for domain_split in domain_splits:
                    done += 1

                    # ── full 变体去重 ────────────────────────────────────────
                    full_key = (month, domain_split)
                    if (DEDUPLICATE_FULL
                            and ablation_mode == "full"
                            and grp != groups[0]       # 非第一个 group
                            and full_key in full_trained):
                        print(f"  [{done}/{total}] {grp} | {variant_name} | "
                              f"month={month} split={domain_split} → 跳过（full 已训练）")
                        continue

                    tag = (f"[{done}/{total}] {grp} | {variant_name} | "
                           f"month={month} | split={domain_split}")
                    print(f"\n{'─'*65}")
                    print(f"  {tag}")
                    print(f"{'─'*65}")

                    try:
                        run_one_ablation(
                            ablation_group      = grp,
                            ablation_variant    = variant_name,
                            ablation_mode       = ablation_mode,
                            granularity_levels  = granularity_levels,
                            freq_mode           = freq_mode,
                            lambda_f            = config.get("lambda_f",   0.001),
                            lambda_c            = config.get("lambda_c",   0.001),
                            lambda_vae          = config.get("lambda_vae", 0.0001),
                            feature_select      = config.get("feature_select",  "time_trends_slide"),
                            dimension_model     = config.get("dimension_model", 128),
                            encoder_layers      = config.get("encoder_layers",  2),
                            batch_size          = config.get("batch_size",      16),
                            learning_rate       = config.get("learning_rate",   0.001),
                            is_training         = config.get("is_training",     1),
                            data_input          = data_input,
                            seed                = config.get("seed",            2021),
                            month               = month,
                            domain_split        = domain_split,
                            loss_k              = config.get("loss_k",          2.0),
                            loss_type           = config.get("loss_type",       "Custom"),
                            seq_len             = config.get("seq_len",         12),
                            label_len           = config.get("label_len",       12),
                            pred_len            = config.get("pred_len",        1),
                            train_epochs        = config.get("train_epochs",    100),
                            patience            = config.get("patience",        3),
                        )

                        if ablation_mode == "full":
                            full_trained.add(full_key)

                    except Exception as e:
                        err_msg = f"{tag} | error={e}\n{traceback.format_exc()}"
                        print(f"  ❌ 失败：{err_msg}")
                        write_error(err_msg)
                        write_result(dict(
                            month=month, domain_split=domain_split,
                            ablation_group=grp, ablation_variant=variant_name,
                            ablation_mode=ablation_mode,
                            granularity_levels=granularity_levels,
                            freq_mode=freq_mode,
                            feature_select=config.get("feature_select"),
                            d_model=config.get("dimension_model"),
                            e_layers=config.get("encoder_layers"),
                            batch_size=config.get("batch_size"),
                            learning_rate=config.get("learning_rate"),
                            loss_type=config.get("loss_type"),
                            loss_k=config.get("loss_k"),
                            lambda_f=config.get("lambda_f", 0.001),
                            lambda_c=config.get("lambda_c", 0.001),
                            lambda_vae=config.get("lambda_vae", 0.0001),
                            seq_len=config.get("seq_len", 12),
                            label_len=config.get("label_len", 12),
                            pred_len=config.get("pred_len", 1),
                            seed=config.get("seed"),
                            mae="ERROR", mse="ERROR", rmse="ERROR",
                            mape="ERROR", mspe="ERROR", store_acc="ERROR",
                            phase="failed",
                        ))
                        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  消融实验全部完成！结果文件：{RESULT_CSV}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ABLATION_CONFIG = dict(

        # ── 要跑的消融组（按需选择，默认全跑） ───────────────────────────────
        groups = ["G1_vae", "G2_granularity", "G3_freq", "G4_align", "G6_pred"],

        # ── 月份列表 ──────────────────────────────────────────────────────────
        months = [202208, 202209, 202210, 202211, 202212, 202301, 202302, 202303],

        # ── 域划分 ────────────────────────────────────────────────────────────
        domain_splits = ["district", "channel"],

        # ── 数据文件模板 ──────────────────────────────────────────────────────
        data_template = "deep_train_mz_{month}_with_store_info.csv",

        # ── 损失系数（全局默认，各变体共享） ──────────────────────────────────
        lambda_f   = 0.001,
        lambda_c   = 0.001,
        lambda_vae = 0.0001,

        # ── 通用超参数 ────────────────────────────────────────────────────────
        feature_select  = "time_trends_slide",
        dimension_model = 128,
        encoder_layers  = 2,
        batch_size      = 16,
        learning_rate   = 0.001,
        loss_k          = 2.0,
        loss_type       = "Custom",
        seed            = 2021,
        seq_len         = 12,
        label_len       = 12,
        pred_len        = 1,
        train_epochs    = 100,
        patience        = 3,
        is_training     = 1,
    )

    run_ablation_experiment(ABLATION_CONFIG)
