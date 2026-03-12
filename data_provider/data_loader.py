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

# 销量预测—Saleformer
# class Sale_Prediction(Dataset):
#     def __init__(self, root_path, args, flag='train', size=None,
#                  features='S', data_path='sale.csv',
#                  target='OT', scale=False, timeenc=0, freq='h', train_only=False):
#         self.args = args
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         global df_data
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         print('before', df_raw.shape)
#
#         '''数据清洗'''
#         df_raw = select_train_test(df_raw)
#         print('after', df_raw.shape)
#
#         df_raw = df_raw.fillna(100)
#
#         # 选择需要的特征
#         group_columns = ['start', 'name']
#         stamp_columns = ['month']
#         final_columns = ['predict']  # 基本特征字段
#         slide_columns = ['predict_3', 'predict_2', 'predict_1', 'predict_15', 'predict_14', 'predict_13', 'predict_12']
#         trends_columns = ['mean', 'mean_past', 'standard',
#                           'standard_past', 'predict_3_2', 'predict_2_1', 'predict_3_2_past',
#                           'predict_2_1_past', 'trend_mean', 'trend_mean_past', 'change_1',
#                           'change_2', 'change_3']
#         # 'mean_1', 'id', 'name', 'mean'
#         # 选取不同的输入特征
#         if 'slide' in self.features:
#             final_columns += slide_columns
#         if 'trends' in self.features:
#             final_columns += trends_columns
#
#         def extract_number(s):
#             return int(s.split('month')[-1])
#
#
#         df_raw = df_raw[final_columns + group_columns+stamp_columns]
#         # df_raw['predict_mean'] = df_raw.groupby(['start', 'name'])['predict'].transform('mean')
#         # df_raw = df_raw[df_raw['predict_mean'] > 35000]
#
#         # 划分训练集和测试集
#         month_value = sorted(df_raw['start'].unique())
#         train_months, test_months = month_value[:-1], month_value[-1]
#         df_train = df_raw[df_raw['start'].isin(train_months)]
#         df_test = df_raw[df_raw['start'] == test_months]
#
#         # 划分训练集和测试集
#         # month_value = self.args.month_predict
#         ## train_months, test_months = ['month'+str(month_value-4), 'month'+str(month_value-3), 'month'+str(month_value-2), 'month'+str(month_value-1)], 'month'+str(month_value)
#         # train_months, test_months = [month_value-4, month_value-3, month_value-2, month_value-1], month_value
#         # df_train = df_raw[df_raw['start'].isin(train_months)]
#         # df_test = df_raw[df_raw['start'] == test_months]
#
#         # 根据任务类型选取使用的数据集
#         df = df_train if self.set_type == 0 else df_test
#         # 划分输入和输出
#         grouped = df.groupby(['start', 'name'])
#         # grouped = df.groupby(['start', 'id'])
#         group_len = len(grouped)
#         data_selected = np.empty((group_len, self.seq_len + self.pred_len, len(final_columns)))
#         stamp_length = 2 if self.timeenc == 0 else 1
#         data_stamp = np.empty((group_len, self.seq_len + self.pred_len, stamp_length))
#         for index, group_pack in enumerate(grouped):
#             group = group_pack[1]
#             # 处理非时间特征数据
#             group_features = group[final_columns]
#             if group_features.shape[0] != 13:
#                 continue
#             data_selected[index] = group_features.values
#             # 处理时间特征数据，后续进行temporal_embedding
#             group_stamp = group[stamp_columns]
#             # group_stamp['date'] = pd.to_datetime(group_stamp.month,format='%Y%m')
#             group_stamp['date'] = group_stamp.month
#             if self.timeenc == 0:
#                 group_stamp['month'] = group_stamp.date.apply(lambda row: row.month, 1)
#                 group_stamp['year'] = group_stamp.date.apply(lambda row: row.year, 1)
#                 group_stamp = group_stamp.drop(['date'], 1).values
#             else:
#                 # group_stamp = time_features(pd.to_datetime(group_stamp['date'].values), freq=self.freq)
#                 group_stamp = group_stamp['date']
#                 # .apply(extract_number)
#                 # group_stamp = group_stamp.transpose(1, 0)
#             # data_stamp[index] = group_stamp
#             data_stamp[index] = group_stamp.values.reshape(-1, 1)
#
#         print('input_shape', data_stamp.shape)
#         self.data = data_selected
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         sample = self.data[index]
#         sample_stamp = self.data_stamp[index]
#         s_begin = 0
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = sample[s_begin:s_end]
#         seq_y = sample[r_begin:r_end]
#         seq_x_mark = sample_stamp[s_begin:s_end]
#         seq_y_mark = sample_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data)
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

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

        # ===================== 新增：按"渠道"划分源域/目标域 =====================
        # 统计各渠道样本量，样本量最少的渠道作为目标域，其余为源域
        if '渠道' in df_raw.columns:
            channel_counts = df_raw.groupby('渠道')['name'].nunique()
            total_stores = channel_counts.sum()
            threshold = total_stores * 0.2  # 门店数低于总数20%的渠道为目标域

            target_channels = channel_counts[channel_counts < threshold].index.tolist()
            source_channels = channel_counts[channel_counts >= threshold].index.tolist()

            # 若没有任何渠道低于20%，则取门店数最少的渠道作为兜底
            if len(target_channels) == 0:
                target_channels = [channel_counts.idxmin()]
                source_channels = [c for c in channel_counts.index if c not in target_channels]
                print(f'警告：无渠道门店数低于20%阈值({threshold:.0f}家)，兜底取最小渠道：{target_channels}')

            print(f'渠道门店数分布：\n{channel_counts}')
            print(f'门店总数：{total_stores}，20%阈值：{threshold:.0f}')
            print(f'目标域渠道（共{channel_counts[target_channels].sum()}家门店）：{target_channels}')
            print(f'源域渠道（共{channel_counts[source_channels].sum()}家门店）：{source_channels}')

            df_target = df_raw[df_raw['渠道'].isin(target_channels)].copy()
            df_source = df_raw[df_raw['渠道'].isin(source_channels)].copy()
        else:
            print('警告：数据集中未找到"渠道"列，所有数据作为目标域，源域为空')
            df_target = df_raw.copy()
            df_source = pd.DataFrame(columns=df_raw.columns)
        # ======================================================================

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

        # ===================== 目标域：按原有逻辑划分训练集/测试集 =====================
        df_target = df_target[final_columns + group_columns + stamp_columns]
        month_value = sorted(df_target['start'].unique())
        train_months = month_value[:-1]
        test_month = month_value[-1]
        df_target_train = df_target[df_target['start'].isin(train_months)]
        df_target_test = df_target[df_target['start'] == test_month]

        # 当前flag对应的目标域数据
        # val复用test逻辑（与原代码一致）
        if self.set_type == 0:
            df_target_current = df_target_train
        else:
            df_target_current = df_target_test

        # ===================== 源域：时间范围与目标域训练集保持一致 =====================
        if len(df_source) > 0:
            df_source = df_source[final_columns + group_columns + stamp_columns]
            df_source_aligned = df_source[df_source['start'].isin(train_months)]
        else:
            df_source_aligned = pd.DataFrame(columns=df_target_current.columns)

        stamp_length = 2 if self.timeenc == 0 else 1

        # ===================== 构建目标域样本 =====================
        self.data, self.data_stamp = self._build_samples(
            df_target_current, final_columns, stamp_columns, stamp_length
        )

        # ===================== 构建源域样本（训练和val/test均使用train_months范围的源域） =====================
        if len(df_source_aligned) > 0:
            self.source_data, self.source_stamp = self._build_samples(
                df_source_aligned, final_columns, stamp_columns, stamp_length
            )
        else:
            # 无源域时，用目标域数据代替（退化为无域适应）
            self.source_data = self.data.copy()
            self.source_stamp = self.data_stamp.copy()

        print(f'目标域样本数：{len(self.data)}，源域样本数：{len(self.source_data)}')

    def _build_samples(self, df, final_columns, stamp_columns, stamp_length):
        """将分组后的DataFrame构建为numpy数组样本"""
        grouped = df.groupby(['start', 'name'])
        group_len = len(grouped)
        data_selected = np.empty((group_len, self.seq_len + self.pred_len, len(final_columns)), dtype=np.float32)
        data_stamp = np.empty((group_len, self.seq_len + self.pred_len, stamp_length), dtype=np.float32)
        # data_selected = np.empty((group_len, self.seq_len + self.pred_len, len(final_columns)))
        # data_stamp = np.empty((group_len, self.seq_len + self.pred_len, stamp_length))

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
                group_stamp['month_feat'] = group_stamp['date'].apply(lambda row: row if not hasattr(row, 'month') else row.month)
                group_stamp['year_feat'] = group_stamp['date'].apply(lambda row: row if not hasattr(row, 'year') else row.year)
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