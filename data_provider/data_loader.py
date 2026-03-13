import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def select_train_test(df):
    all_groups = []

    grouped = df.groupby(['name', 'start'])

    for name_start, group in grouped:
        if len(group) < 13:  # 如果组内数据不足 13 行，则跳过
            continue

        # 计算前 12 行的均值
        mean_value = group.iloc[:12]['predict'].mean()  # 假设 'value' 是需要计算的列
        # 取第 13 行的值
        thirteenth_value = group.iloc[12]['predict']

        # 计算差异率
        if mean_value != 0:  # 避免除以零的情况
            difference_rate = abs(thirteenth_value - mean_value) / mean_value

            # 如果差异率小于等于 10%，则保留该组
            if difference_rate <= 0.30:
                all_groups.append(group)

    result_df = pd.concat(all_groups, ignore_index=True)

    return result_df


# 销量预测_HierSales
class Sale_Prediction(Dataset):
    def __init__(self, root_path, args, flag='train', size=None,
                 features='S', data_path='sale.csv',
                 target='OT', scale=False, timeenc=0, freq='h', train_only=False):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        print('before', df_raw.shape)
        df_raw = select_train_test(df_raw)
        print('after', df_raw.shape)
        df_raw = df_raw.fillna(100)

        # ===================== 读取实验模式 =====================
        # experiment_mode 支持三种值：
        #   'hierda'      — 原有逻辑：源域 + 目标域，有域适应（默认）
        #   'target_only' — 仅目标域训练集训练，预测目标域测试集
        #   'source_only' — 仅源域数据训练，直接预测目标域测试集
        experiment_mode = getattr(self.args, 'experiment_mode', 'hierda')

        # ===================== 按参数选择域划分策略 =====================
        domain_split = getattr(self.args, 'domain_split', 'district')  # 默认按小区划分

        # 美赞
        if domain_split == 'district':
            community_col = '区域'

            community_store_counts = df_raw.groupby(community_col)['name'].nunique().sort_values()
            threshold_20pct = community_store_counts.quantile(0.6)

            target_communities = community_store_counts[
                community_store_counts <= threshold_20pct
                ].index.tolist()
            source_communities = community_store_counts[
                community_store_counts > threshold_20pct
                ].index.tolist()

            print(f'小区门店数分布：\n{community_store_counts}')
            print(f'后20%门店数阈值：{threshold_20pct}')
            print(f'目标域小区数：{len(target_communities)}，'
                  f'涉及门店：{community_store_counts[target_communities].sum()}家')
            print(f'源域小区数：{len(source_communities)}，'
                  f'涉及门店：{community_store_counts[source_communities].sum()}家')

            df_target = df_raw[df_raw[community_col].isin(target_communities)].copy()
            df_source = df_raw[df_raw[community_col].isin(source_communities)].copy()

        elif domain_split == 'channel':
            if '品牌渠道' in df_raw.columns:
                channel_store_counts = df_raw.groupby('品牌渠道')['name'].nunique().sort_values()
                threshold_20pct = channel_store_counts.quantile(0.6)

                target_channels = channel_store_counts[
                    channel_store_counts <= threshold_20pct
                    ].index.tolist()
                source_channels = channel_store_counts[
                    channel_store_counts > threshold_20pct
                    ].index.tolist()

                if len(target_channels) == 0:
                    target_channels = [channel_store_counts.index[0]]
                    source_channels = channel_store_counts.index[1:].tolist()
                    print(f'警告：所有渠道门店数相同，兜底取门店数最少的渠道：{target_channels}')

                print(f'渠道门店数分布：\n{channel_store_counts}')
                print(f'后20%门店数阈值：{threshold_20pct}')
                print(f'目标域渠道数：{len(target_channels)}，'
                      f'涉及门店：{channel_store_counts[target_channels].sum()}家')
                print(f'源域渠道数：{len(source_channels)}，'
                      f'涉及门店：{channel_store_counts[source_channels].sum()}家')

                df_target = df_raw[df_raw['品牌渠道'].isin(target_channels)].copy()
                df_source = df_raw[df_raw['品牌渠道'].isin(source_channels)].copy()
            else:
                print('警告：数据集中未找到"渠道"列，所有数据作为目标域，源域为空')
                df_target = df_raw.copy()
                df_source = pd.DataFrame(columns=df_raw.columns)

        else:
            raise ValueError(f'未知的域划分策略：{domain_split}，请选择 district 或 channel')

        # 金佰利
        # if domain_split == 'district':
        #     community_col = '小区'
        #
        #     community_store_counts = df_raw.groupby(community_col)['name'].nunique().sort_values()
        #     threshold_20pct = community_store_counts.quantile(0.6)
        #
        #     target_communities = community_store_counts[
        #         community_store_counts <= threshold_20pct
        #         ].index.tolist()
        #     source_communities = community_store_counts[
        #         community_store_counts > threshold_20pct
        #         ].index.tolist()
        #
        #     print(f'小区门店数分布：\n{community_store_counts}')
        #     print(f'后20%门店数阈值：{threshold_20pct}')
        #     print(f'目标域小区数：{len(target_communities)}，'
        #           f'涉及门店：{community_store_counts[target_communities].sum()}家')
        #     print(f'源域小区数：{len(source_communities)}，'
        #           f'涉及门店：{community_store_counts[source_communities].sum()}家')
        #
        #     df_target = df_raw[df_raw[community_col].isin(target_communities)].copy()
        #     df_source = df_raw[df_raw[community_col].isin(source_communities)].copy()
        #
        # elif domain_split == 'channel':
        #     if '渠道' in df_raw.columns:
        #         channel_store_counts = df_raw.groupby('渠道')['name'].nunique().sort_values()
        #         threshold_20pct = channel_store_counts.quantile(0.6)
        #
        #         target_channels = channel_store_counts[
        #             channel_store_counts <= threshold_20pct
        #             ].index.tolist()
        #         source_channels = channel_store_counts[
        #             channel_store_counts > threshold_20pct
        #             ].index.tolist()
        #
        #         if len(target_channels) == 0:
        #             target_channels = [channel_store_counts.index[0]]
        #             source_channels = channel_store_counts.index[1:].tolist()
        #             print(f'警告：所有渠道门店数相同，兜底取门店数最少的渠道：{target_channels}')
        #
        #         print(f'渠道门店数分布：\n{channel_store_counts}')
        #         print(f'后20%门店数阈值：{threshold_20pct}')
        #         print(f'目标域渠道数：{len(target_channels)}，'
        #               f'涉及门店：{channel_store_counts[target_channels].sum()}家')
        #         print(f'源域渠道数：{len(source_channels)}，'
        #               f'涉及门店：{channel_store_counts[source_channels].sum()}家')
        #
        #         df_target = df_raw[df_raw['渠道'].isin(target_channels)].copy()
        #         df_source = df_raw[df_raw['渠道'].isin(source_channels)].copy()
        #     else:
        #         print('警告：数据集中未找到"渠道"列，所有数据作为目标域，源域为空')
        #         df_target = df_raw.copy()
        #         df_source = pd.DataFrame(columns=df_raw.columns)
        #
        # else:
        #     raise ValueError(f'未知的域划分策略：{domain_split}，请选择 district 或 channel')
        # ================================================================

        # 特征列定义
        group_columns = ['start', 'name']
        stamp_columns = ['month']
        final_columns = ['predict']
        slide_columns = ['predict_3', 'predict_2', 'predict_1', 'predict_15',
                         'predict_14', 'predict_13', 'predict_12']
        trends_columns = ['mean', 'mean_past', 'standard', 'standard_past',
                          'predict_3_2', 'predict_2_1', 'predict_3_2_past',
                          'predict_2_1_past', 'trend_mean', 'trend_mean_past',
                          'change_1', 'change_2', 'change_3']
        if 'slide' in self.features:
            final_columns += slide_columns
        if 'trends' in self.features:
            final_columns += trends_columns

        # ===================== 目标域时间划分（三种模式共用） =====================
        df_target = df_target[final_columns + group_columns + stamp_columns]
        month_value = sorted(df_target['start'].unique())
        train_months = month_value[:-1]
        test_month = month_value[-1]
        df_target_train = df_target[df_target['start'].isin(train_months)]
        df_target_test = df_target[df_target['start'] == test_month]

        stamp_length = 2 if self.timeenc == 0 else 1

        # ===================== 根据 experiment_mode 决定训练/测试数据来源 =====================

        if experiment_mode == 'target_only':
            # ── Target-Only：仅用目标域训练集训练，预测目标域测试集 ──────────────
            # 训练阶段：目标域训练集
            # 测试/val阶段：目标域测试集
            # source_data 退化为与 data 相同（模型仍需接收源域输入，直接复用目标域数据）
            print(f'[experiment_mode=target_only] 仅使用目标域数据，无域适应')

            if self.set_type == 0:  # train
                df_current = df_target_train
            else:  # val / test
                df_current = df_target_test

            self.data, self.data_stamp = self._build_samples(
                df_current, final_columns, stamp_columns, stamp_length
            )
            # source 复用 target（模型接口不变，但实际无跨域信息）
            self.source_data = self.data.copy()
            self.source_stamp = self.data_stamp.copy()

        elif experiment_mode == 'source_only':
            # ── Source-Only：仅用源域训练集训练，预测目标域测试集 ────────────────
            # 训练阶段：源域（train_months 范围）
            # 测试/val阶段：目标域测试集（评估跨域泛化能力）
            # source_data 在训练时 = 源域，测试时复用 target 数据（只需前向一次）
            print(f'[experiment_mode=source_only] 仅使用源域数据训练，直接预测目标域测试集')

            if len(df_source) > 0:
                df_source = df_source[final_columns + group_columns + stamp_columns]
                df_source_train = df_source[df_source['start'].isin(train_months)]
            else:
                df_source_train = pd.DataFrame(columns=df_target_train.columns)

            if self.set_type == 0:  # train：用源域训练集
                if len(df_source_train) > 0:
                    df_current = df_source_train
                else:
                    print('警告：源域训练集为空，退化为目标域训练集')
                    df_current = df_target_train
            else:  # val / test：评估在目标域测试集上的效果
                df_current = df_target_test

            self.data, self.data_stamp = self._build_samples(
                df_current, final_columns, stamp_columns, stamp_length
            )
            # source 复用 target（接口对齐，source_only 下无需额外源域输入）
            self.source_data = self.data.copy()
            self.source_stamp = self.data_stamp.copy()

        else:
            # ── HierDA（原有逻辑）：源域 + 目标域，有域适应 ──────────────────────
            if self.set_type == 0:
                df_target_current = df_target_train
            else:
                df_target_current = df_target_test

            if len(df_source) > 0:
                df_source = df_source[final_columns + group_columns + stamp_columns]
                df_source_aligned = df_source[df_source['start'].isin(train_months)]
            else:
                df_source_aligned = pd.DataFrame(columns=df_target_current.columns)

            self.data, self.data_stamp = self._build_samples(
                df_target_current, final_columns, stamp_columns, stamp_length
            )

            if len(df_source_aligned) > 0:
                self.source_data, self.source_stamp = self._build_samples(
                    df_source_aligned, final_columns, stamp_columns, stamp_length
                )
            else:
                self.source_data = self.data.copy()
                self.source_stamp = self.data_stamp.copy()

        print(f'[{experiment_mode}] 目标域样本数：{len(self.data)}，源域样本数：{len(self.source_data)}')

    def _build_samples(self, df, final_columns, stamp_columns, stamp_length):
        """将分组后的DataFrame构建为numpy数组样本"""
        grouped = df.groupby(['start', 'name'])
        group_len = len(grouped)
        data_selected = np.empty((group_len, self.seq_len + self.pred_len, len(final_columns)), dtype=np.float32)
        data_stamp = np.empty((group_len, self.seq_len + self.pred_len, stamp_length), dtype=np.float32)

        valid_count = 0
        for index, group_pack in enumerate(grouped):
            group = group_pack[1]
            group_features = group[final_columns]
            if group_features.shape[0] != 13:
                continue
            data_selected[valid_count] = group_features.values
            group_stamp = group[stamp_columns].copy()
            group_stamp['date'] = group_stamp['month']
            if self.timeenc == 0:
                group_stamp['month_feat'] = group_stamp['date'].apply(
                    lambda row: row if not hasattr(row, 'month') else row.month)
                group_stamp['year_feat'] = group_stamp['date'].apply(
                    lambda row: row if not hasattr(row, 'year') else row.year)
                group_stamp = group_stamp[['month_feat', 'year_feat']].values
                data_stamp[valid_count] = group_stamp
            else:
                data_stamp[valid_count] = group_stamp['date'].values.reshape(-1, 1)
            valid_count += 1

        return data_selected[:valid_count], data_stamp[:valid_count]

    def __getitem__(self, index):
        # 目标域样本
        sample = self.data[index]
        sample_stamp = self.data_stamp[index]
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = sample[0:s_end]
        seq_y = sample[r_begin:r_end]
        seq_x_mark = sample_stamp[0:s_end]
        seq_y_mark = sample_stamp[r_begin:r_end]

        # 源域样本：随机采样一条（避免与目标域强对齐索引）
        src_index = index % len(self.source_data)
        src_sample = self.source_data[src_index]
        src_stamp = self.source_stamp[src_index]
        seq_x_src = src_sample[0:s_end]
        seq_x_src_mark = src_stamp[0:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_src, seq_x_src_mark

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)