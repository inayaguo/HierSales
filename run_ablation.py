"""
run_ablation.py
消融实验批量入口，完全独立于 run_new.py，不修改其任何代码。

关键设计：
  - 模型名固定为 'HierDAabl'，exp_basic.py 里需注册该键（见下方说明）
  - 所有消融参数通过 argparse 传入 args，再由 AblationModel.__init__ 读取
  - 结果写入独立 CSV（ablation_results.csv），不污染对比实验结果文件
"""

import random
import argparse
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

RESULT_DIR  = os.path.join("output_hiersales", "ab_result")
RESULT_CSV  = os.path.join(RESULT_DIR, "ablation_results_mz.csv")
ERROR_LOG   = os.path.join(RESULT_DIR, "ablation_error_kim.log")

RESULT_COLUMNS = [
    "run_time", "ablation_group", "ablation_variant",
    "month", "domain_split",
    "ablation_mode", "granularity_levels", "freq_mode",
    "lambda_f", "lambda_c", "lambda_vae",
    "feature_select", "d_model", "e_layers", "batch_size", "learning_rate",
    "loss_type", "loss_k", "seq_len", "label_len", "pred_len", "seed",
    "mae", "mse", "rmse", "mape", "mspe", "store_acc", "phase",
]

ABLATION_GROUPS = {
    "G1_vae": [
        dict(ablation_mode="full",       granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="wo_vae",     granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="vae_noloss", granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="vae_detach", granularity_levels=3, freq_mode="both"),
    ],
    "G2_granularity": [
        # dict(ablation_mode="full", granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="full", granularity_levels=2, freq_mode="both"),
        dict(ablation_mode="full", granularity_levels=1, freq_mode="both"),
    ],
    "G3_freq": [
        # dict(ablation_mode="full", granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="full", granularity_levels=3, freq_mode="time_only"),
        dict(ablation_mode="full", granularity_levels=3, freq_mode="freq_only"),
    ],
    "G4_align": [
        # dict(ablation_mode="full",         granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="wo_transport", granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="wo_grl",       granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="wo_da",        granularity_levels=3, freq_mode="both"),
        # dict(ablation_mode="no_source",    granularity_levels=3, freq_mode="both"),
    ],
    "G6_pred": [
        # dict(ablation_mode="full",        granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="direct_pred", granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="prior_only",  granularity_levels=3, freq_mode="both"),
        dict(ablation_mode="no_scale",    granularity_levels=3, freq_mode="both"),
    ],
}

DEDUPLICATE_FULL = True


def _ensure_csv():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULT_CSV, index=False)


def write_result(row: dict):
    _ensure_csv()
    row.setdefault("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS).to_csv(
        RESULT_CSV, mode="a", header=False, index=False)


def write_error(msg: str):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def run_one_ablation(
    *, ablation_group, ablation_variant,
    ablation_mode="full", granularity_levels=3, freq_mode="both",
    lambda_f=0.001, lambda_c=0.001, lambda_vae=0.0001,
    feature_select, dimension_model, encoder_layers, batch_size,
    learning_rate, is_training, data_input, seed, month,
    domain_split="district", loss_k=2.0, loss_type="Custom",
    seq_len=12, label_len=12, pred_len=1, train_epochs=100, patience=3,
):
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task_name",          default="long_term_forecast")
    parser.add_argument("--is_training",        type=int,   default=is_training)
    parser.add_argument("--model_id",           default="ablation")
    parser.add_argument("--model",              default="HierDAabl")
    parser.add_argument("--month_predict",      type=int,   default=month)
    parser.add_argument("--set_seed",           type=int,   default=seed)
    # 消融参数
    parser.add_argument("--ablation_mode",      default=ablation_mode)
    parser.add_argument("--granularity_levels", type=int,   default=granularity_levels)
    parser.add_argument("--freq_mode",          default=freq_mode)
    parser.add_argument("--lambda_f",           type=float, default=lambda_f)
    parser.add_argument("--lambda_c",           type=float, default=lambda_c)
    parser.add_argument("--lambda_vae",         type=float, default=lambda_vae)
    # 损失 / 域划分
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
    for arg, val in [("time_in",1),("slide_in",7),("trends_in",13),
                     ("d_time",64),("d_slide",64),("d_trends",64)]:
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

    setting = (
        f"ablation_HierDAabl_sale"
        f"_sl{seq_len}_ll{label_len}_pl{pred_len}"
        f"_dm{dimension_model}_el{encoder_layers}"
        f"_k{loss_k}_{loss_type}"
        f"_abl{ablation_mode}_g{granularity_levels}_f{freq_mode}"
        f"_m{month}_split{domain_split}_seed{seed}"
    )

    print(f"  [消融参数确认] ablation_mode={args.ablation_mode} | "
          f"granularity_levels={args.granularity_levels} | "
          f"freq_mode={args.freq_mode}")
    print(f"  [checkpoint]  {setting}")

    exp = Exp_Long_Term_Forecast(args)

    base_info = dict(
        ablation_group=ablation_group, ablation_variant=ablation_variant,
        month=month, domain_split=domain_split,
        ablation_mode=ablation_mode, granularity_levels=granularity_levels,
        freq_mode=freq_mode, lambda_f=lambda_f, lambda_c=lambda_c,
        lambda_vae=lambda_vae, feature_select=feature_select,
        d_model=dimension_model, e_layers=encoder_layers,
        batch_size=batch_size, learning_rate=learning_rate,
        loss_type=loss_type, loss_k=loss_k,
        seq_len=seq_len, label_len=label_len, pred_len=pred_len, seed=seed,
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
        print(f"  ✅ 写入：{RESULT_CSV}")
    return metrics


def run_ablation_experiment(config: dict):
    _ensure_csv()
    months        = config.get("months",        [])
    domain_splits = config.get("domain_splits", ["district"])
    groups        = config.get("groups",        list(ABLATION_GROUPS.keys()))

    full_trained: set = set()
    first_full_group  = next(
        (g for g in groups
         if any(v["ablation_mode"] == "full" for v in ABLATION_GROUPS[g])),
        None
    )

    total = sum(len(ABLATION_GROUPS[g]) * len(months) * len(domain_splits)
                for g in groups)
    done = 0

    print(f"\n{'='*70}")
    print(f"  HierDA 消融实验启动：预计 {total} 组（含 full 去重前）")
    print(f"  组别：{groups}  结果文件：{RESULT_CSV}")
    print(f"{'='*70}\n")

    for grp in groups:
        for v_cfg in ABLATION_GROUPS[grp]:
            ablation_mode      = v_cfg["ablation_mode"]
            granularity_levels = v_cfg["granularity_levels"]
            freq_mode          = v_cfg["freq_mode"]
            variant_name       = (f"full_g{granularity_levels}_{freq_mode}"
                                  if ablation_mode == "full" else ablation_mode)

            for month in months:
                data_input = config["data_template"].format(month=month)
                for domain_split in domain_splits:
                    done += 1
                    tag = (f"[{done}] {grp}/{variant_name} | "
                           f"month={month} | split={domain_split}")

                    full_key = (month, domain_split)
                    if (DEDUPLICATE_FULL and ablation_mode == "full"
                            and grp != first_full_group
                            and full_key in full_trained):
                        print(f"  {tag} → 跳过（full 已训练）")
                        continue

                    print(f"\n{'─'*65}\n  {tag}\n{'─'*65}")

                    try:
                        run_one_ablation(
                            ablation_group=grp, ablation_variant=variant_name,
                            ablation_mode=ablation_mode,
                            granularity_levels=granularity_levels,
                            freq_mode=freq_mode,
                            lambda_f=config.get("lambda_f",   0.001),
                            lambda_c=config.get("lambda_c",   0.001),
                            lambda_vae=config.get("lambda_vae", 0.0001),
                            feature_select=config.get("feature_select","time_trends_slide"),
                            dimension_model=config.get("dimension_model", 128),
                            encoder_layers=config.get("encoder_layers",  2),
                            batch_size=config.get("batch_size",      16),
                            learning_rate=config.get("learning_rate",   0.001),
                            is_training=config.get("is_training",     1),
                            data_input=data_input,
                            seed=config.get("seed", 2021),
                            month=month, domain_split=domain_split,
                            loss_k=config.get("loss_k",    2.0),
                            loss_type=config.get("loss_type", "Custom"),
                            seq_len=config.get("seq_len",   12),
                            label_len=config.get("label_len", 12),
                            pred_len=config.get("pred_len",  1),
                            train_epochs=config.get("train_epochs", 100),
                            patience=config.get("patience",     3),
                        )
                        if ablation_mode == "full":
                            full_trained.add(full_key)

                    except Exception as e:
                        err_msg = f"{tag} | {e}\n{traceback.format_exc()}"
                        print(f"  ❌ 失败：{err_msg}")
                        write_error(err_msg)
                        err_row = dict(
                            ablation_group=grp, ablation_variant=variant_name,
                            month=month, domain_split=domain_split,
                            ablation_mode=ablation_mode,
                            granularity_levels=granularity_levels,
                            freq_mode=freq_mode,
                            lambda_f=config.get("lambda_f",0.001),
                            lambda_c=config.get("lambda_c",0.001),
                            lambda_vae=config.get("lambda_vae",0.0001),
                            feature_select=config.get("feature_select"),
                            d_model=config.get("dimension_model"),
                            e_layers=config.get("encoder_layers"),
                            batch_size=config.get("batch_size"),
                            learning_rate=config.get("learning_rate"),
                            loss_type=config.get("loss_type"),
                            loss_k=config.get("loss_k"),
                            seq_len=config.get("seq_len",12),
                            label_len=config.get("label_len",12),
                            pred_len=config.get("pred_len",1),
                            seed=config.get("seed"),
                            mae="ERROR", mse="ERROR", rmse="ERROR",
                            mape="ERROR", mspe="ERROR", store_acc="ERROR",
                            phase="failed",
                        )
                        write_result(err_row)
                        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  消融实验完成！结果文件：{RESULT_CSV}\n{'='*70}\n")


if __name__ == "__main__":
    ABLATION_CONFIG = dict(
        groups        = ["G3_freq"],
        # "G2_granularity", "G4_align", "G6_pred"
        # "G1_vae",
        months        = [202208, 202209, 202210, 202211, 202212,
    202301, 202302, 202303],
    #     months=[202305, 202306, 202307, 202308, 202309, 202310, 202311, 202312],
        domain_splits = ["district", "channel"],
        data_template = "deep_train_mz_{month}_with_store_info.csv",
        lambda_f      = 0.001,
        lambda_c      = 0.001,
        lambda_vae    = 0.0001,
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
        train_epochs    = 10,
        patience        = 3,
        is_training     = 1,
    )
    run_ablation_experiment(ABLATION_CONFIG)
