# import argparse
# import torch
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# import random
# import numpy as np
# import pandas as pd
#
# torch.manual_seed(seed=1)
# torch.cuda.manual_seed_all(seed=1)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
#
# def start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, is_training, data_input, model, seed, i, loss_k=2, loss_type='MSE', domain_split='district'):
#     """
#     新增参数：
#     loss_k: 自定义损失函数的惩罚系数k
#     loss_type: 损失函数类型，可选['MSE', 'Custom']
#     """
#     fix_seed = seed
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)
#
#     parser = argparse.ArgumentParser(description='TimesNet')
#     parser.add_argument('--set_seed', type=int, required=False, default=seed)
#
#     # basic config
#     parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
#                         help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
#     parser.add_argument('--is_training', type=int, required=False, default=is_training, help='status')
#     parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
#     parser.add_argument('--model', type=str, required=False, default=model,
#                         help='model name, options: [Autoformer, Transformer, TimesNet, FEDformer]')
#     parser.add_argument('--month_predict', type=int, required=False, default=i)
#     # 新增：损失函数相关参数
#     parser.add_argument('--loss_k', type=float, required=False, default=loss_k, help='惩罚系数k for custom loss')
#     parser.add_argument('--loss', type=str, required=False, default=loss_type, help='loss function: MSE/Custom')
#     # 新增：目标域源域划分策略
#     parser.add_argument('--domain_split', type=str, required=False, default=domain_split,
#                         help='域划分策略: district=按小区划分, channel=按渠道划分')
#
#     # data loader
#     parser.add_argument('--data', type=str, required=False, default='sale', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default=data_input, help='data file')
#
#     # time,slide,trends随意组合
#     parser.add_argument('--features', type=str, default=feature_select,
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='predict', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='m',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#
#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=12, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
#     parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#
#     # inputation task
#     parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
#
#     # anomaly detection task
#     parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
#
#     # model define
#     parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
#     parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
#
#     # time:1,slide:7,trends:13
#     feature_dimension = 1
#     if 'slide' in feature_select:
#         feature_dimension += 7
#     if 'trends' in feature_select:
#         feature_dimension += 13
#     parser.add_argument('--enc_in', type=int, default=feature_dimension, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=feature_dimension, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=feature_dimension, help='output size')
#     parser.add_argument('--time_in', type=int, default=1, help='output size')
#     parser.add_argument('--slide_in', type=int, default=7, help='output size')
#     parser.add_argument('--trends_in', type=int, default=13, help='output size')
#     parser.add_argument('--d_time', type=int, default=64, help='output size')
#     parser.add_argument('--d_slide', type=int, default=64, help='output size')
#     parser.add_argument('--d_trends', type=int, default=64, help='output size')
#
#     parser.add_argument('--d_model', type=int, default=dimension_model, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=encoder_layers, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--moving_avg', type=int, default=11, help='window size of moving average')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false',
#                         help='whether to use distilling in encoder, using this argument means not using distilling',
#                         default=True)
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
#
#     # optimization
#     parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=learning_rate, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#
#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
#
#     # de-stationary projector params
#     parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
#                         help='hidden layer dimensions of projector (List)')
#     parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
#
#     args = parser.parse_args()
#     args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#
#     if args.use_gpu and args.use_multi_gpu:
#         args.devices = args.devices.replace(' ', '')
#         device_ids = args.devices.split(',')
#         args.device_ids = [int(id_) for id_ in device_ids]
#         args.gpu = args.device_ids[0]
#
#     print('Args in experiment:')
#     print(f'Loss k value: {args.loss_k}, Loss type: {args.loss}')  # 新增：打印k值和损失函数类型
#     print(f'GPU: {args.gpu}')
#
#     if args.task_name == 'long_term_forecast':
#         Exp = Exp_Long_Term_Forecast
#     else:
#         Exp = Exp_Long_Term_Forecast
#
#     if args.is_training:
#         for ii in range(args.itr):
#             # 新增：setting中加入k值标识，区分不同实验
#             setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_k{}_loss{}_{}_{}'.format(
#                 args.task_name,
#                 args.model_id,
#                 args.model,
#                 args.data,
#                 args.features,
#                 args.seq_len,
#                 args.label_len,
#                 args.pred_len,
#                 args.d_model,
#                 args.n_heads,
#                 args.e_layers,
#                 args.d_layers,
#                 args.d_ff,
#                 args.factor,
#                 args.embed,
#                 args.distil,
#                 args.loss_k,
#                 args.loss,
#                 args.des, ii)
#
#             exp = Exp(args)  # set experiments
#             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             exp.train(setting)
#
#             torch.cuda.empty_cache()
#     else:
#         ii = 0
#         setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_k{}_loss{}_{}_{}'.format(
#             args.task_name,
#             args.model_id,
#             args.model,
#             args.data,
#             args.features,
#             args.seq_len,
#             args.label_len,
#             args.pred_len,
#             args.d_model,
#             args.n_heads,
#             args.e_layers,
#             args.d_layers,
#             args.d_ff,
#             args.factor,
#             args.embed,
#             args.distil,
#             args.loss_k,
#             args.loss,
#             args.des, ii)
#
#         exp = Exp(args)  # set experiments
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         exp.test(setting, test=1)
#         torch.cuda.empty_cache()
#
#
# def hyper_param_experiment():
#     """
#     超参数试验函数：遍历不同的k值、模型参数，对比效果
#     """
#     # 1. 定义要试验的超参数范围
#     loss_k_list = [2.0]  # 惩罚系数k的试验范围: [1.0, 1.5, 2.0, 2.5, 3.0]
#     dimension_model_list = [128]  # 模型维度: [128, 256]
#     batch_size_list = [16]  # 批次大小: [16, 32]
#     learning_rate_list = [0.001]  # 学习率: [0.001, 0.005]
#     loss_type = 'Custom'  # 使用自定义损失函数（若要对比MSE，可设为['MSE', 'Custom']）
#
#     # 2. 试验的月份范围
#     month_ranges = [
#         range(202208, 202213),
#         range(202301, 202304)
#     ]
#
#     # 3. 遍历所有超参数组合
#     for loss_k in loss_k_list:
#         for dimension_model in dimension_model_list:
#             for batch_size in batch_size_list:
#                 for learning_rate in learning_rate_list:
#                     # 遍历每个月份
#                     for month_range in month_ranges:
#                         for month_data in month_range:
#                             print(
#                                 f'\n========== 开始试验：k={loss_k}, d_model={dimension_model}, batch={batch_size}, lr={learning_rate}, month={month_data} ==========')
#                             input_data = f'/kaggle/input/0105-saleformer/Saleformer/data/deep_train_mz_{month_data}.csv'
#                             try:
#                                 start(
#                                     feature_select='time_trends_slide',
#                                     dimension_model=dimension_model,
#                                     encoder_layers=2,
#                                     batch_size=batch_size,
#                                     learning_rate=learning_rate,
#                                     data_input=input_data,
#                                     model='HierDA',
#                                     seed=2021,
#                                     i=month_data,
#                                     loss_k=loss_k,
#                                     loss_type=loss_type
#                                 )
#                             except Exception as e:
#                                 print(f'试验失败：k={loss_k}, month={month_data}, 错误：{str(e)}')
#                                 # 记录失败的试验
#                                 with open('result/experiment_error.log', 'a') as f:
#                                     f.write(
#                                         f'k={loss_k}, d_model={dimension_model}, batch={batch_size}, lr={learning_rate}, month={month_data}, error={str(e)}\n')
#                             finally:
#                                 torch.cuda.empty_cache()  # 清理GPU缓存
#
#
# def select_train_test(df):
#     """数据清洗函数"""
#     all_groups = []
#     grouped = df.groupby(['name', 'start'])
#
#     for name_start, group in grouped:
#         if len(group) < 13:
#             continue
#
#         mean_value = group.iloc[:12]['predict'].mean()
#         thirteenth_value = group.iloc[12]['predict']
#
#         if mean_value != 0:
#             difference_rate = abs(thirteenth_value - mean_value) / mean_value
#             if difference_rate <= 0.30:
#                 all_groups.append(group)
#
#     result_df = pd.concat(all_groups, ignore_index=True)
#     return result_df
#
#
# if __name__ == '__main__':
#     # 方式1：单参数运行（原逻辑）
#     # input_data = 'deep_train_202305.csv'
#     # start(feature_select='time_trends_slide', dimension_model=128, encoder_layers=2, batch_size=16,
#     #       learning_rate=0.005, data_input=input_data, model='FEDformer', seed=2021, i=202305, loss_k=2.0, loss_type='Custom')
#
#     # 方式2：运行超参数试验（重点：遍历不同k值和模型参数）
#     # hyper_param_experiment()
#     start(
#         feature_select='time_trends_slide',
#         dimension_model=128,
#         encoder_layers=2,
#         batch_size=4,
#         learning_rate=0.00005,
#         data_input='/Users/inaya/Desktop/HierSales/data/deep_train_202305_with_store_info.csv',
#         model='HierDA',
#         seed=2021,
#         i=202305,
#         loss_k=2.0,
#         loss_type='Custom',
#         is_training=0,
#         domain_split='channel'
#     )
"""
run_batch.py —— 批量实验入口

支持：
  - 多月份 × 多域划分方式 × 多超参数 笛卡尔积实验
  - 每次测试结果自动写入 output_hiersales/result/batch_experiment_results.csv
  - 超参数、月份、划分方式、指标全量记录
  - 失败实验自动记录到 output_hiersales/result/experiment_error.log
"""

import random
import argparse
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ────────────────────────────────────────────────────────────────────
# 结果写入工具
# ────────────────────────────────────────────────────────────────────

RESULT_DIR = os.path.join("output_hiersales", "result")
RESULT_CSV  = os.path.join(RESULT_DIR, "batch_experiment_results.csv")
ERROR_LOG   = os.path.join(RESULT_DIR, "experiment_error.log")

RESULT_COLUMNS = [
    "run_time",
    "model",
    "month",
    "domain_split",
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
    "phase",        # train / test
]


def _ensure_result_csv():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not os.path.exists(RESULT_CSV):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULT_CSV, index=False)


def write_result(row: dict):
    """将单次实验结果追加写入 CSV（线程安全程度：单进程够用）"""
    _ensure_result_csv()
    row.setdefault("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_new = pd.DataFrame([row])
    df_new = df_new.reindex(columns=RESULT_COLUMNS)   # 保证列顺序一致
    df_new.to_csv(RESULT_CSV, mode="a", header=False, index=False)


def write_error(msg: str):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# ────────────────────────────────────────────────────────────────────
# 单次实验
# ────────────────────────────────────────────────────────────────────

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
    loss_k: float = 2.0,
    loss_type: str = "Custom",
    domain_split: str = "district",
    seq_len: int = 12,
    label_len: int = 12,
    pred_len: int = 1,
    train_epochs: int = 100,
    patience: int = 3,
):
    """封装单次实验，返回指标 dict 或 None（失败时）"""
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(add_help=False)

    # ── 基础配置 ──
    parser.add_argument("--task_name",      default="long_term_forecast")
    parser.add_argument("--is_training",    type=int,   default=is_training)
    parser.add_argument("--model_id",       default="batch")
    parser.add_argument("--model",          default=model)
    parser.add_argument("--month_predict",  type=int,   default=month)
    parser.add_argument("--set_seed",       type=int,   default=seed)

    # ── 损失 & 域划分 ──
    parser.add_argument("--loss_k",         type=float, default=loss_k)
    parser.add_argument("--loss",           default=loss_type)
    parser.add_argument("--domain_split",   default=domain_split)

    # ── 数据 ──
    parser.add_argument("--data",           default="sale")
    parser.add_argument("--root_path",      default="./data/")
    parser.add_argument("--data_path",      default=data_input)
    parser.add_argument("--features",       default=feature_select)
    parser.add_argument("--target",         default="predict")
    parser.add_argument("--freq",           default="m")
    parser.add_argument("--checkpoints",    default="./checkpoints/")

    # ── 序列长度 ──
    parser.add_argument("--seq_len",        type=int, default=seq_len)
    parser.add_argument("--label_len",      type=int, default=label_len)
    parser.add_argument("--pred_len",       type=int, default=pred_len)
    parser.add_argument("--seasonal_patterns", default="Monthly")
    parser.add_argument("--mask_rate",      type=float, default=0.25)
    parser.add_argument("--anomaly_ratio",  type=float, default=0.25)

    # ── 特征维度 ──
    feat_dim = 1
    if "slide"  in feature_select: feat_dim += 7
    if "trends" in feature_select: feat_dim += 13
    for arg in ["enc_in", "dec_in", "c_out"]:
        parser.add_argument(f"--{arg}", type=int, default=feat_dim)
    parser.add_argument("--time_in",   type=int, default=1)
    parser.add_argument("--slide_in",  type=int, default=7)
    parser.add_argument("--trends_in", type=int, default=13)
    parser.add_argument("--d_time",    type=int, default=64)
    parser.add_argument("--d_slide",   type=int, default=64)
    parser.add_argument("--d_trends",  type=int, default=64)

    # ── 模型结构 ──
    parser.add_argument("--top_k",         type=int,   default=5)
    parser.add_argument("--num_kernels",   type=int,   default=6)
    parser.add_argument("--d_model",       type=int,   default=dimension_model)
    parser.add_argument("--n_heads",       type=int,   default=8)
    parser.add_argument("--e_layers",      type=int,   default=encoder_layers)
    parser.add_argument("--d_layers",      type=int,   default=1)
    parser.add_argument("--d_ff",          type=int,   default=2048)
    parser.add_argument("--moving_avg",    type=int,   default=11)
    parser.add_argument("--factor",        type=int,   default=1)
    parser.add_argument("--distil",        action="store_false", default=True)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--embed",         default="timeF")
    parser.add_argument("--activation",    default="gelu")
    parser.add_argument("--output_attention", action="store_true", default=False)

    # ── 优化 ──
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument("--itr",           type=int,   default=1)
    parser.add_argument("--train_epochs",  type=int,   default=train_epochs)
    parser.add_argument("--batch_size",    type=int,   default=batch_size)
    parser.add_argument("--patience",      type=int,   default=patience)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--des",           default="batch")
    parser.add_argument("--lradj",         default="type1")
    parser.add_argument("--use_amp",       action="store_true", default=False)

    # ── GPU ──
    parser.add_argument("--use_gpu",       type=bool,  default=True)
    parser.add_argument("--gpu",           type=int,   default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices",       default="0")
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)

    args = parser.parse_args([])
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    # setting 字符串（与原始代码保持一致）
    setting = (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
        f"_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}"
        f"_k{args.loss_k}_loss{args.loss}_{args.des}_0"
    )

    exp = Exp_Long_Term_Forecast(args)

    # ── 超参数基础信息（供结果写入） ──
    base_info = dict(
        model        = model,
        month        = month,
        domain_split = domain_split,
        feature_select = feature_select,
        d_model      = dimension_model,
        e_layers     = encoder_layers,
        batch_size   = batch_size,
        learning_rate= learning_rate,
        loss_type    = loss_type,
        loss_k       = loss_k,
        seq_len      = seq_len,
        label_len    = label_len,
        pred_len     = pred_len,
        seed         = seed,
    )

    if is_training:
        print(f">>>>>> 训练：{setting}")
        exp.train(setting)
        torch.cuda.empty_cache()
        # 训练完后立即跑测试集并记录
        metrics = exp.test(setting, test=0, return_metrics=True)
        phase = "train+test"
    else:
        print(f">>>>>> 测试：{setting}")
        metrics = exp.test(setting, test=1, return_metrics=True)
        phase = "test"

    torch.cuda.empty_cache()

    if metrics:
        row = {**base_info, **metrics, "phase": phase}
        write_result(row)
        print(f"✅ 结果已写入：{RESULT_CSV}")
    return metrics


# ────────────────────────────────────────────────────────────────────
# 批量实验主函数
# ────────────────────────────────────────────────────────────────────

def batch_experiment(config: dict):
    """
    config 示例：
    {
        "months": [202301, 202302, 202303],
        "domain_splits": ["district", "channel"],
        "feature_select": "time_trends_slide",
        "model": "HierDA",
        "dimension_model": 128,
        "encoder_layers": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "loss_k": 2.0,
        "loss_type": "Custom",
        "seed": 2021,
        "is_training": 1,
        "data_root": "./data/",
        "data_template": "deep_train_{month}.csv",   # {month} 会被替换
        "train_epochs": 100,
        "patience": 3,
    }
    """
    _ensure_result_csv()

    months       = config.get("months", [])
    domain_splits = config.get("domain_splits", ["district"])
    total = len(months) * len(domain_splits)
    done  = 0

    print(f"\n{'='*60}")
    print(f"  批量实验启动：共 {total} 组（{len(months)} 月 × {len(domain_splits)} 划分方式）")
    print(f"  结果将写入：{RESULT_CSV}")
    print(f"{'='*60}\n")

    for month in months:
        data_path = config["data_template"].format(month=month)
        # 若 data_template 是绝对路径或含目录，则直接用；否则拼 data_root
        if not os.path.isabs(data_path) and not data_path.startswith("./"):
            data_input = data_path          # 由 root_path + data_path 拼接（DataLoader 内部处理）
        else:
            data_input = data_path          # 绝对路径直接传

        for domain_split in domain_splits:
            done += 1
            tag = f"[{done}/{total}] month={month}, split={domain_split}"
            print(f"\n{'─'*50}")
            print(f"  {tag}")
            print(f"{'─'*50}")

            try:
                run_one(
                    feature_select = config.get("feature_select", "time_trends_slide"),
                    dimension_model= config.get("dimension_model", 128),
                    encoder_layers = config.get("encoder_layers", 2),
                    batch_size     = config.get("batch_size", 16),
                    learning_rate  = config.get("learning_rate", 0.001),
                    is_training    = config.get("is_training", 1),
                    data_input     = data_input,
                    model          = config.get("model", "HierDA"),
                    seed           = config.get("seed", 2021),
                    month          = month,
                    loss_k         = config.get("loss_k", 2.0),
                    loss_type      = config.get("loss_type", "Custom"),
                    domain_split   = domain_split,
                    seq_len        = config.get("seq_len", 12),
                    label_len      = config.get("label_len", 12),
                    pred_len       = config.get("pred_len", 1),
                    train_epochs   = config.get("train_epochs", 100),
                    patience       = config.get("patience", 3),
                )
            except Exception as e:
                err_msg = (
                    f"{tag} | error={str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                print(f"❌ 实验失败：{err_msg}")
                write_error(err_msg)
                # 写一行失败记录到 CSV，方便后续追踪
                write_result(dict(
                    model        = config.get("model", "HierDA"),
                    month        = month,
                    domain_split = domain_split,
                    feature_select = config.get("feature_select"),
                    d_model      = config.get("dimension_model"),
                    e_layers     = config.get("encoder_layers"),
                    batch_size   = config.get("batch_size"),
                    learning_rate= config.get("learning_rate"),
                    loss_type    = config.get("loss_type"),
                    loss_k       = config.get("loss_k"),
                    seq_len      = config.get("seq_len", 12),
                    label_len    = config.get("label_len", 12),
                    pred_len     = config.get("pred_len", 1),
                    seed         = config.get("seed"),
                    mae="ERROR", mse="ERROR", rmse="ERROR",
                    mape="ERROR", mspe="ERROR", store_acc="ERROR",
                    phase="failed",
                ))
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  批量实验完成！结果文件：{RESULT_CSV}")
    print(f"{'='*60}\n")


# ────────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 在此修改实验配置 ──────────────────────────────────────────────
    EXP_CONFIG = dict(
        # 要跑的月份列表
        # months=[202305, 202306, 202307, 202308, 202309, 202309, 202310, 202311, 202312],
        months=[202208, 202209, 202210, 202211, 202212, 202301, 202302, 202303],

        # 要对比的域划分方式
        domain_splits=["district", "channel"],

        # 数据文件模板（{month} 占位符自动替换）
        # 绝对路径示例："/kaggle/input/xxx/deep_train_{month}.csv"
        # 相对路径示例（相对 root_path）：
        data_template="deep_train_mz_{month}_with_store_info.csv",

        # 通用超参数
        feature_select  = "time_trends_slide",
        model           = "HierDA",
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

        # 1=训练+测试，0=仅测试
        is_training     = 1,
    )
    # ─────────────────────────────────────────────────────────────────

    batch_experiment(EXP_CONFIG)