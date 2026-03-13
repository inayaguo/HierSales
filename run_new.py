import random
import argparse
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ══════════════════════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join("output_hiersales", "result")
# 金佰利
# RESULT_CSV = os.path.join(RESULT_DIR, "batch_experiment_results_all.csv")
# 美赞
RESULT_CSV = os.path.join(RESULT_DIR, "batch_experiment_mz_results_all.csv")
ERROR_LOG  = os.path.join(RESULT_DIR, "experiment_error.log")

RESULT_COLUMNS = [
    "run_time",
    "model",
    "month",
    "domain_split",
    "experiment_mode",
    "feature_select",
    "d_model",
    "e_layers",
    "batch_size",
    "learning_rate",
    "loss_type",
    "loss_k",
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

# 所有支持的对比基线
ALL_BASELINES = [
    "HierDA",
    "DANN",
    "DeepCoral",
    "CoDATS",
    "AdvSKM",
    "RAINCOAT",
    "CotMix",
    "AdaMatch",
    "ACON",
]

# ══════════════════════════════════════════════════════════════════════════════
# 结果写入工具
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_result_csv():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULT_CSV, index=False)


def write_result(row: dict):
    """追加写入一行结果（单进程安全）"""
    _ensure_result_csv()
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

def run_one(
    *,
    feature_select: str,
    dimension_model: int,
    encoder_layers: int,
    batch_size: int,
    learning_rate: float,
    is_training: int,
    data_input: str,
    model: str,
    seed: int,
    month: int,
    loss_k: float        = 2.0,
    loss_type: str       = "Custom",
    domain_split: str    = "district",
    experiment_mode: str = "hierda",
    seq_len: int         = 12,
    label_len: int       = 12,
    pred_len: int        = 1,
    train_epochs: int    = 100,
    patience: int        = 3,
):
    """
    封装单次实验，返回指标 dict（失败时抛出异常由外层捕获）。

    model            : 模型名，见 ALL_BASELINES
    experiment_mode  : 'hierda' | 'target_only' | 'source_only'
    """
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    parser = argparse.ArgumentParser(add_help=False)

    # ── 基础 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--task_name",         default="long_term_forecast")
    parser.add_argument("--is_training",       type=int,   default=is_training)
    parser.add_argument("--model_id",          default="batch")
    parser.add_argument("--model",             default=model)
    parser.add_argument("--month_predict",     type=int,   default=month)
    parser.add_argument("--set_seed",          type=int,   default=seed)

    # ── 损失 / 域划分 / 实验模式 ──────────────────────────────────────────────
    parser.add_argument("--loss_k",            type=float, default=loss_k)
    parser.add_argument("--loss",              default=loss_type)
    parser.add_argument("--domain_split",      default=domain_split)
    parser.add_argument("--experiment_mode",   default=experiment_mode)

    # ── 数据 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data",              default="sale")
    parser.add_argument("--root_path",         default="./data/")
    parser.add_argument("--data_path",         default=data_input)
    parser.add_argument("--features",          default=feature_select)
    parser.add_argument("--target",            default="predict")
    parser.add_argument("--freq",              default="m")
    parser.add_argument("--checkpoints",       default="./checkpoints/")

    # ── 序列长度 ──────────────────────────────────────────────────────────────
    parser.add_argument("--seq_len",           type=int, default=seq_len)
    parser.add_argument("--label_len",         type=int, default=label_len)
    parser.add_argument("--pred_len",          type=int, default=pred_len)
    parser.add_argument("--seasonal_patterns", default="Monthly")
    parser.add_argument("--mask_rate",         type=float, default=0.25)
    parser.add_argument("--anomaly_ratio",     type=float, default=0.25)

    # ── 特征维度 ──────────────────────────────────────────────────────────────
    enc_in = 1
    if "slide"  in feature_select: enc_in += 7
    if "trends" in feature_select: enc_in += 13
    for arg in ["enc_in", "dec_in", "c_out"]:
        parser.add_argument(f"--{arg}", type=int, default=enc_in)
    parser.add_argument("--time_in",   type=int, default=1)
    parser.add_argument("--slide_in",  type=int, default=7)
    parser.add_argument("--trends_in", type=int, default=13)
    parser.add_argument("--d_time",    type=int, default=64)
    parser.add_argument("--d_slide",   type=int, default=64)
    parser.add_argument("--d_trends",  type=int, default=64)

    # ── 模型结构 ──────────────────────────────────────────────────────────────
    parser.add_argument("--top_k",           type=int,   default=5)
    parser.add_argument("--num_kernels",     type=int,   default=6)
    parser.add_argument("--d_model",         type=int,   default=dimension_model)
    parser.add_argument("--n_heads",         type=int,   default=8)
    parser.add_argument("--e_layers",        type=int,   default=encoder_layers)
    parser.add_argument("--d_layers",        type=int,   default=1)
    parser.add_argument("--d_ff",            type=int,   default=2048)
    parser.add_argument("--moving_avg",      type=int,   default=11)
    parser.add_argument("--factor",          type=int,   default=1)
    parser.add_argument("--distil",          action="store_false", default=True)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--embed",           default="timeF")
    parser.add_argument("--activation",      default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)

    # ── 优化 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--itr",             type=int,   default=1)
    parser.add_argument("--train_epochs",    type=int,   default=train_epochs)
    parser.add_argument("--batch_size",      type=int,   default=batch_size)
    parser.add_argument("--patience",        type=int,   default=patience)
    parser.add_argument("--learning_rate",   type=float, default=learning_rate)
    parser.add_argument("--des",             default="batch")
    parser.add_argument("--lradj",           default="type1")
    parser.add_argument("--use_amp",         action="store_true", default=False)

    # ── GPU ───────────────────────────────────────────────────────────────────
    parser.add_argument("--use_gpu",         type=bool,  default=True)
    parser.add_argument("--gpu",             type=int,   default=0)
    parser.add_argument("--use_multi_gpu",   action="store_true", default=False)
    parser.add_argument("--devices",         default="0")
    parser.add_argument("--p_hidden_dims",   type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)

    args = parser.parse_args([])
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    # model + mode 共同写入 setting，确保不同模型/模式的 checkpoint 互不覆盖
    setting = (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
        f"_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}"
        f"_k{args.loss_k}_loss{args.loss}"
        f"_mode{experiment_mode}_mdl{model}_{args.des}_0"
    )

    exp = Exp_Long_Term_Forecast(args)

    base_info = dict(
        model           = model,
        month           = month,
        domain_split    = domain_split,
        experiment_mode = experiment_mode,
        feature_select  = feature_select,
        d_model         = dimension_model,
        e_layers        = encoder_layers,
        batch_size      = batch_size,
        learning_rate   = learning_rate,
        loss_type       = loss_type,
        loss_k          = loss_k,
        seq_len         = seq_len,
        label_len       = label_len,
        pred_len        = pred_len,
        seed            = seed,
    )

    if is_training:
        print(f">>>>>> 训练 [model={model} | mode={experiment_mode}]")
        exp.train(setting)
        torch.cuda.empty_cache()
        metrics = exp.test(setting, test=0, return_metrics=True)
        phase = "train+test"
    else:
        print(f">>>>>> 测试 [model={model} | mode={experiment_mode}]")
        metrics = exp.test(setting, test=1, return_metrics=True)
        phase = "test"

    torch.cuda.empty_cache()

    if metrics:
        write_result({**base_info, **metrics, "phase": phase})
        print(f"✅ [model={model}|mode={experiment_mode}] 结果已写入：{RESULT_CSV}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 批量实验主函数
# ══════════════════════════════════════════════════════════════════════════════

def batch_experiment(config: dict):
    """
    全笛卡尔积批量实验：月份 × 域划分 × 实验模式 × 模型

    关键逻辑
    ────────────────────────────────────────────────────────────────────────
    • target_only / source_only 模式：数据划分与具体模型架构无关，
      自动只用 models[0]（通常是 HierDA）跑一次，避免重复。
    • hierda 模式：遍历 models 列表中的所有模型，做全量对比实验。
    ────────────────────────────────────────────────────────────────────────
    """
    _ensure_result_csv()

    months           = config.get("months",           [])
    domain_splits    = config.get("domain_splits",    ["district"])
    experiment_modes = config.get("experiment_modes", ["hierda"])
    models           = config.get("models",           ["HierDA"])

    # 预估总组数（target_only/source_only 各只跑 1 个模型）
    n_hierda = sum(1 for m in experiment_modes if m == "hierda")
    n_other  = len(experiment_modes) - n_hierda
    total = len(months) * len(domain_splits) * (
        n_hierda * len(models) + n_other * min(1, len(models))
    )
    done = 0

    print(f"\n{'='*70}")
    print(f"  统一对比实验启动：预计 {total} 组")
    print(f"  月份({len(months)}) × 域划分({len(domain_splits)}) "
          f"× 模式({len(experiment_modes)}) × 模型(hierda:{len(models)}, 其他:1)")
    print(f"  模型列表    : {models}")
    print(f"  实验模式    : {experiment_modes}")
    print(f"  域划分方式  : {domain_splits}")
    print(f"  结果文件    : {RESULT_CSV}")
    print(f"{'='*70}\n")

    for month in months:
        data_input = config["data_template"].format(month=month)

        for domain_split in domain_splits:
            for exp_mode in experiment_modes:

                # target_only / source_only：只取第一个模型作代表
                model_list = models if exp_mode == "hierda" else models[:1]

                for model in model_list:
                    done += 1
                    tag = (f"[{done}/{total}] month={month} | split={domain_split} "
                           f"| mode={exp_mode} | model={model}")
                    print(f"\n{'─'*65}")
                    print(f"  {tag}")
                    print(f"{'─'*65}")

                    try:
                        run_one(
                            feature_select  = config.get("feature_select",  "time_trends_slide"),
                            dimension_model = config.get("dimension_model", 128),
                            encoder_layers  = config.get("encoder_layers",  2),
                            batch_size      = config.get("batch_size",      16),
                            learning_rate   = config.get("learning_rate",   0.001),
                            is_training     = config.get("is_training",     1),
                            data_input      = data_input,
                            model           = model,
                            seed            = config.get("seed",            2021),
                            month           = month,
                            loss_k          = config.get("loss_k",          2.0),
                            loss_type       = config.get("loss_type",       "Custom"),
                            domain_split    = domain_split,
                            experiment_mode = exp_mode,
                            seq_len         = config.get("seq_len",         12),
                            label_len       = config.get("label_len",       12),
                            pred_len        = config.get("pred_len",        1),
                            train_epochs    = config.get("train_epochs",    100),
                            patience        = config.get("patience",        3),
                        )
                    except Exception as e:
                        err_msg = f"{tag} | error={e}\n{traceback.format_exc()}"
                        print(f"❌ 实验失败：{err_msg}")
                        write_error(err_msg)
                        write_result(dict(
                            model           = model,
                            month           = month,
                            domain_split    = domain_split,
                            experiment_mode = exp_mode,
                            feature_select  = config.get("feature_select"),
                            d_model         = config.get("dimension_model"),
                            e_layers        = config.get("encoder_layers"),
                            batch_size      = config.get("batch_size"),
                            learning_rate   = config.get("learning_rate"),
                            loss_type       = config.get("loss_type"),
                            loss_k          = config.get("loss_k"),
                            seq_len         = config.get("seq_len", 12),
                            label_len       = config.get("label_len", 12),
                            pred_len        = config.get("pred_len", 1),
                            seed            = config.get("seed"),
                            mae="ERROR", mse="ERROR", rmse="ERROR",
                            mape="ERROR", mspe="ERROR", store_acc="ERROR",
                            phase="failed",
                        ))
                        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  批量实验全部完成！结果文件：{RESULT_CSV}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    EXP_CONFIG = dict(

        # ── 月份列表 ──────────────────────────────────────────────────────────
        months = [202208, 202209, 202210, 202211, 202212,
                  202301, 202302, 202303],
        # months=[202305, 202306, 202307, 202308, 202309, 202310, 202311, 202312],

        # ── 域划分方式 ────────────────────────────────────────────────────────
        domain_splits = ["district", "channel"],

        # ── 实验模式（三重对照） ───────────────────────────────────────────────
        # target_only : 仅目标域训练，测目标域（性能上界，自动只跑 HierDA）
        # source_only : 仅源域训练，测目标域（域差异基准，自动只跑 HierDA）
        # hierda      : 源域+目标域，有迁移（全量 9 个模型逐一对比）
        # experiment_modes = ["target_only", "source_only", "hierda"],
        experiment_modes=["hierda"],

        # ── 参与 hierda 模式对比的模型（target/source_only 自动只取第一个） ───
        models = [
            # "HierDA",       # 本文模型（多粒度时频 + Transport Map + GRL）
            "DANN",         # 经典对抗域适应（JMLR 2016）
            "DeepCoral",    # 二阶统计对齐（ECCV 2016）
            "CoDATS",       # 时序感知对抗（KDD 2020）
            "AdvSKM",       # 谱核匹配（IJCAI 2021）
            "RAINCOAT",     # 时频原型对齐（ICML 2023）
            "CotMIX",       # 时序上下文混合（CIKM 2023）
            "AdaMatch",     # 分布对齐伪标签（ICLR 2022）
            "ACON",         # 时频相关子空间对抗（NeurIPS 2024）
        ],

        # ── 数据文件模板 ──────────────────────────────────────────────────────
        data_template = "deep_train_mz_{month}_with_store_info.csv",

        # ── 通用超参数（所有模型共享） ────────────────────────────────────────
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

        # 1=训练+测试，0=仅测试（加载已有 checkpoint）
        is_training     = 1,
    )

    batch_experiment(EXP_CONFIG)
