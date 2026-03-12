import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import pandas as pd

torch.manual_seed(seed=1)
torch.cuda.manual_seed_all(seed=1)
print(torch.cuda.is_available())
print(torch.cuda.device_count())


def start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, i,
          loss_k=2, loss_type='MSE'):
    """
    新增参数：
    loss_k: 自定义损失函数的惩罚系数k
    loss_type: 损失函数类型，可选['MSE', 'Custom']
    """
    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--set_seed', type=int, required=False, default=seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default=model,
                        help='model name, options: [Autoformer, Transformer, TimesNet, FEDformer]')
    parser.add_argument('--month_predict', type=int, required=False, default=i)
    # 新增：损失函数相关参数
    parser.add_argument('--loss_k', type=float, required=False, default=loss_k, help='惩罚系数k for custom loss')
    parser.add_argument('--loss', type=str, required=False, default=loss_type, help='loss function: MSE/Custom')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='sale', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=data_input, help='data file')

    # time,slide,trends随意组合
    parser.add_argument('--features', type=str, default=feature_select,
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='predict', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='m',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    # time:1,slide:7,trends:13
    feature_dimension = 1
    if 'slide' in feature_select:
        feature_dimension += 7
    if 'trends' in feature_select:
        feature_dimension += 13
    parser.add_argument('--enc_in', type=int, default=feature_dimension, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=feature_dimension, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=feature_dimension, help='output size')
    parser.add_argument('--time_in', type=int, default=1, help='output size')
    parser.add_argument('--slide_in', type=int, default=7, help='output size')
    parser.add_argument('--trends_in', type=int, default=13, help='output size')
    parser.add_argument('--d_time', type=int, default=64, help='output size')
    parser.add_argument('--d_slide', type=int, default=64, help='output size')
    parser.add_argument('--d_trends', type=int, default=64, help='output size')

    parser.add_argument('--d_model', type=int, default=dimension_model, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=encoder_layers, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=11, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(f'Loss k value: {args.loss_k}, Loss type: {args.loss}')  # 新增：打印k值和损失函数类型
    print(f'GPU: {args.gpu}')

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # 新增：setting中加入k值标识，区分不同实验
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_k{}_loss{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.loss_k,
                args.loss,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_k{}_loss{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.loss_k,
            args.loss,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


def hyper_param_experiment():
    """
    超参数试验函数：遍历不同的k值、模型参数，对比效果
    """
    # 1. 定义要试验的超参数范围
    loss_k_list = [2.0]  # 惩罚系数k的试验范围: [1.0, 1.5, 2.0, 2.5, 3.0]
    dimension_model_list = [128]  # 模型维度: [128, 256]
    batch_size_list = [16]  # 批次大小: [16, 32]
    learning_rate_list = [0.001]  # 学习率: [0.001, 0.005]
    loss_type = 'Custom'  # 使用自定义损失函数（若要对比MSE，可设为['MSE', 'Custom']）

    # 2. 试验的月份范围
    month_ranges = [
        range(202208, 202213),
        range(202301, 202304)
    ]

    # 3. 遍历所有超参数组合
    for loss_k in loss_k_list:
        for dimension_model in dimension_model_list:
            for batch_size in batch_size_list:
                for learning_rate in learning_rate_list:
                    # 遍历每个月份
                    for month_range in month_ranges:
                        for month_data in month_range:
                            print(
                                f'\n========== 开始试验：k={loss_k}, d_model={dimension_model}, batch={batch_size}, lr={learning_rate}, month={month_data} ==========')
                            input_data = f'/kaggle/input/0105-saleformer/Saleformer/data/deep_train_mz_{month_data}.csv'
                            try:
                                start(
                                    feature_select='time_trends_slide',
                                    dimension_model=dimension_model,
                                    encoder_layers=2,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    data_input=input_data,
                                    model='HierDA',
                                    seed=2021,
                                    i=month_data,
                                    loss_k=loss_k,
                                    loss_type=loss_type
                                )
                            except Exception as e:
                                print(f'试验失败：k={loss_k}, month={month_data}, 错误：{str(e)}')
                                # 记录失败的试验
                                with open('result/experiment_error.log', 'a') as f:
                                    f.write(
                                        f'k={loss_k}, d_model={dimension_model}, batch={batch_size}, lr={learning_rate}, month={month_data}, error={str(e)}\n')
                            finally:
                                torch.cuda.empty_cache()  # 清理GPU缓存


def select_train_test(df):
    """数据清洗函数"""
    all_groups = []
    grouped = df.groupby(['name', 'start'])

    for name_start, group in grouped:
        if len(group) < 13:
            continue

        mean_value = group.iloc[:12]['predict'].mean()
        thirteenth_value = group.iloc[12]['predict']

        if mean_value != 0:
            difference_rate = abs(thirteenth_value - mean_value) / mean_value
            if difference_rate <= 0.30:
                all_groups.append(group)

    result_df = pd.concat(all_groups, ignore_index=True)
    return result_df


if __name__ == '__main__':
    # 方式1：单参数运行（原逻辑）
    # input_data = 'deep_train_202305.csv'
    # start(feature_select='time_trends_slide', dimension_model=128, encoder_layers=2, batch_size=16,
    #       learning_rate=0.005, data_input=input_data, model='FEDformer', seed=2021, i=202305, loss_k=2.0, loss_type='Custom')

    # 方式2：运行超参数试验（重点：遍历不同k值和模型参数）
    # hyper_param_experiment()
    start(
        feature_select='time_trends_slide',
        dimension_model=128,
        encoder_layers=2,
        batch_size=16,
        learning_rate=0.001,
        data_input='deep_train_202305.csv',
        model='HierDA',
        seed=2021,
        i=202305,
        loss_k=2.0,
        loss_type='Custom'
    )