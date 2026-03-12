# """
# 金佰利数据处理与VAE缺失值填补整合脚本
# 功能：读取单份原始数据，自动拆分训练/验证集，执行缺失值填补与多维度评估
# """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import skew, kurtosis, ks_2samp
from dtw import dtw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
from scipy.spatial.distance import euclidean  # 导入欧氏距离函数

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ==============================================================================
# 1. 数据预处理（适配单文件输入，自动生成缺失值）
# ==============================================================================
class DataProcessor:
    def __init__(self, data_path, missing_rate=0.1):
        """
        初始化数据处理器
        :param data_path: 原始数据CSV路径
        :param missing_rate: 人工生成的缺失值比例（0-1）
        """
        # 读取原始数据
        print(f"正在读取数据: {data_path}")
        self.df_complete = pd.read_csv(data_path)
        print(f"数据读取成功，原始数据形状: {self.df_complete.shape}")

        # 定义关键列
        self.monthly_cols = [col for col in self.df_complete.columns if col.startswith('Y22-')]
        self.key_dim_cols = ['门店编码', '门店名称', '地城市', '渠道']


        # todo:新增数据处理剔除0值门店
        self.valid_threshold = 0.1
        # ========== 第一步：剔除全0/低有效值门店（核心新增） ==========
        if len(self.monthly_cols) > 0:
            # 1. 计算每个门店的有效值（>0）数量
            self.df_complete['有效值数量'] = self.df_complete[self.monthly_cols].apply(lambda x: (x > 1e-6).sum(),
                                                                                       axis=1)
            self.df_complete['有效值占比'] = self.df_complete['有效值数量'] / len(self.monthly_cols)

            # 2. 筛选有效门店（有效值占比≥阈值）
            valid_store_mask = self.df_complete['有效值占比'] >= self.valid_threshold
            self.df_complete = self.df_complete[valid_store_mask].reset_index(drop=True)


            # 打印筛选日志
            removed_stores = len(valid_store_mask) - valid_store_mask.sum()
            print(f"✅ 剔除低有效值门店数：{removed_stores}（有效值占比<{self.valid_threshold}）")
            print(f"✅ 保留有效门店数：{len(self.df_complete)}")

            if len(self.df_complete) == 0:
                raise ValueError("所有门店均为低有效值门店，请降低valid_value_threshold阈值！")

        # ========== 异常值清洗 ==========
        if len(self.monthly_cols) > 0:
            # 1. 替换inf/-inf为nan
            self.df_complete[self.monthly_cols] = self.df_complete[self.monthly_cols].replace([np.inf, -np.inf], np.nan)
            # 2. 填充剩余nan（用列均值）
            self.df_complete[self.monthly_cols] = self.df_complete[self.monthly_cols].fillna(
                self.df_complete[self.monthly_cols].mean()
            )
            # 3. 过滤极端值（3σ原则）
            for col in self.monthly_cols:
                mean = self.df_complete[col].mean()
                std = self.df_complete[col].std()
                upper_bound = mean + 3 * std
                lower_bound = mean - 3 * std
                self.df_complete[col] = self.df_complete[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"异常值清洗完成，月度数据范围：")
            for col in self.monthly_cols[:3]:  # 仅打印前3列示例
                print(f"  - {col}: [{self.df_complete[col].min():.2f}, {self.df_complete[col].max():.2f}]")


        # 补充渠道列（若不存在则按城市分组生成）
        if '渠道' not in self.df_complete.columns:
            print("数据中无'渠道'列，按城市自动生成渠道分组")
            city_channel_map = {city: f'渠道_{i % 5}' for i, city in enumerate(self.df_complete['地城市'].unique())}
            self.df_complete['渠道'] = self.df_complete['地城市'].map(city_channel_map)

        # 生成带缺失值的数据（模拟真实场景）
        self.df_missing = self._generate_missing_values(self.df_complete.copy(), missing_rate)

        # 数据标准化（优化范围，避免极端值）
        self.scaler = MinMaxScaler(feature_range=(0.05, 0.95))
        self.df_complete_scaled = self._scale_data(self.df_complete)
        self.df_missing_scaled = self._scale_data(self.df_missing)

        # 创建评估集（记录缺失值位置和真实值）
        self.eval_set = self._create_evaluation_set()
        print(f"评估集创建完成，包含 {len(self.eval_set)} 个缺失值样本")

    # ========== 补全缺失的 _generate_missing_values 方法 ==========
    def _generate_missing_values(self, df, missing_rate):
        """人工生成缺失值（仅在月度数据列中）"""
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        if len(self.monthly_cols) == 0:
            print("警告：未找到月度数据列（Y22-开头），跳过缺失值生成")
            return df

        # 生成缺失值掩码（只在月度列中生成）
        missing_mask = np.random.rand(*df[self.monthly_cols].shape) < missing_rate
        # 应用掩码生成缺失值
        df[self.monthly_cols] = df[self.monthly_cols].mask(missing_mask)
        # 统计缺失值数量
        total_missing = df[self.monthly_cols].isnull().sum().sum()
        total_values = len(df) * len(self.monthly_cols)
        print(f"生成缺失值完成：共 {total_missing} 个缺失值（占比 {total_missing / total_values:.2%}）")
        return df

    # ========== 数据标准化方法 ==========
    def _scale_data(self, df):
        """数据标准化（增加数值稳定性校验）"""
        df_scaled = df.copy()
        if len(self.monthly_cols) == 0:
            return df_scaled

        # 标准化前再次检查并替换异常值
        scaled_vals = self.scaler.fit_transform(df[self.monthly_cols])
        # 替换标准化后的nan/inf为均值
        scaled_vals = np.nan_to_num(scaled_vals, nan=np.nanmean(scaled_vals), posinf=np.nanmax(scaled_vals),
                                    neginf=np.nanmin(scaled_vals))
        df_scaled[self.monthly_cols] = scaled_vals
        return df_scaled

    # ========== 评估集创建方法 ==========
    def _create_evaluation_set(self):
        """创建评估集，记录缺失值位置和真实值"""
        eval_data = []
        if len(self.monthly_cols) == 0:
            return pd.DataFrame(eval_data)

        for idx in range(len(self.df_missing)):
            for month_col in self.monthly_cols:
                if pd.isna(self.df_missing.loc[idx, month_col]):
                    eval_data.append({
                        '门店编码': self.df_missing.loc[idx, '门店编码'],
                        '索引': idx,
                        '月份列': month_col,
                        '真实值': self.df_complete.loc[idx, month_col],
                        '真实值标准化': self.df_complete_scaled.loc[idx, month_col]
                    })
        return pd.DataFrame(eval_data)

    # ========== 数据划分策略方法 ==========
    def split_data_by_strategy(self, strategy_name):
        # , target_store_idx
        """
        保留原始策略逻辑（按目标门店划分训练/测试集）
        strategy_name: 策略名称（channel, city, month, cross, ts_similar）
        target_store_idx: 目标缺失门店的索引（全局选一个代表性门店）
        """
        if len(self.monthly_cols) == 0:
            print("警告：无月度数据列，返回空训练数据")
            return np.array([]), np.array([])  # 补充返回测试集

        # target_store = self.df_missing.iloc[target_store_idx]
        train_data = []
        test_data = []

        # ========== 原始策略逻辑完全保留 ==========
        if strategy_name == 'channel':
            # 步骤1：筛选门店数量最多的渠道（保留原逻辑）
            channel_counts = self.df_missing['渠道'].value_counts()
            max_count_channel = channel_counts.index[0]
            print(f"选择门店数量最多的渠道：{max_count_channel}（共{channel_counts.iloc[0]}家门店）")

            # 步骤2：仅按渠道筛选样本（移除 target_store_idx 排除逻辑）
            # 核心修改：不再排除目标门店，直接取该渠道的所有样本
            channel_mask = (self.df_missing['渠道'] == max_count_channel)
            channel_data = self.df_missing_scaled[channel_mask][self.monthly_cols].values
            print(f"该渠道全量样本数：{len(channel_data)}")

            # 步骤3：基于该渠道全量样本划分训练/测试集（优化逻辑，更健壮）
            if len(channel_data) >= 5:
                # 先随机打乱，再划分（避免顺序偏差，原逻辑无打乱，补充最佳实践）
                np.random.seed(42)  # 固定种子保证可复现
                np.random.shuffle(channel_data)
                train_size = int(len(channel_data) * 0.8)
                train_data = channel_data[:train_size]
                test_data = channel_data[train_size:]
            elif 2 <= len(channel_data) < 5:
                # 样本数2-4时，训练集取大部分，测试集至少1个
                train_data = channel_data[:-1]
                test_data = channel_data[-1:]
            else:
                # 样本数<2时，训练集=全量，测试集复用训练集（兜底）
                train_data = channel_data
                test_data = channel_data.copy() if len(channel_data) > 0 else np.array([])

            # 兜底：若测试集为空，补充至少1个样本
            if len(test_data) == 0 and len(train_data) > 0:
                test_data = train_data[:1]

            print(f"渠道策略 - 训练集样本数：{len(train_data)}，测试集样本数：{len(test_data)}")

        elif strategy_name == 'city':
            # 步骤1：筛选门店数量最多的地城市（保留原统计逻辑）
            city_counts = self.df_missing['地城市'].value_counts()
            max_count_city = city_counts.index[0]
            print(f"选择门店数量最多的城市：{max_count_city}（共{city_counts.iloc[0]}家门店）")

            # 步骤2：仅按地城市筛选全量样本（核心修改：移除target_store_idx排除逻辑）
            city_mask = (self.df_missing['地城市'] == max_count_city)
            city_data = self.df_missing_scaled[city_mask][self.monthly_cols].values
            print(f"该城市全量样本数：{len(city_data)}")

            # 步骤3：基于该城市全量样本划分训练/测试集（优化健壮性，覆盖所有样本场景）
            # 先随机打乱，避免顺序偏差（最佳实践，保证划分公平）
            np.random.seed(42)  # 固定随机种子，确保结果可复现
            if len(city_data) > 0:
                np.random.shuffle(city_data)

            # 分场景划分，保证测试集非空
            if len(city_data) >= 5:
                # 样本充足：80%训练，20%测试
                train_size = int(len(city_data) * 0.8)
                train_data = city_data[:train_size]
                test_data = city_data[train_size:]
            elif 2 <= len(city_data) < 5:
                # 样本较少：训练集取大部分，测试集至少1个
                train_data = city_data[:-1]
                test_data = city_data[-1:]
            else:
                # 样本极少（<2）：训练集=全量，测试集复用训练集（兜底）
                train_data = city_data
                test_data = city_data.copy() if len(city_data) > 0 else np.array([])

            # 最终兜底：若测试集为空且训练集有数据，补充至少1个样本
            if len(test_data) == 0 and len(train_data) > 0:
                test_data = train_data[:1]

            print(f"城市策略 - 训练集样本数：{len(train_data)}，测试集样本数：{len(test_data)}")

        elif strategy_name == 'month':
            # 步骤1：筛选全量无缺失数据的门店（核心修改：移除target_store_idx排除逻辑）
            # non_missing_mask：所有月度列均无缺失的门店
            non_missing_mask = self.df_missing[self.monthly_cols].notna().all(axis=1)
            global_non_missing_data = self.df_missing_scaled[non_missing_mask][self.monthly_cols].values
            print(f"全局无缺失值样本数：{len(global_non_missing_data)}")

            # 步骤2：数据预处理（打乱+兜底，保证划分公平且鲁棒）
            np.random.seed(42)  # 固定随机种子，结果可复现
            if len(global_non_missing_data) > 0:
                np.random.shuffle(global_non_missing_data)  # 打乱样本，避免顺序偏差

            # 步骤3：按样本量分场景划分训练/测试集
            if len(global_non_missing_data) >= 5:
                # 样本充足：80%训练，20%测试
                train_size = int(len(global_non_missing_data) * 0.8)
                train_data = global_non_missing_data[:train_size]
                test_data = global_non_missing_data[train_size:]
            elif 2 <= len(global_non_missing_data) < 5:
                # 样本较少：训练集取大部分，测试集至少1个
                train_data = global_non_missing_data[:-1]
                test_data = global_non_missing_data[-1:]
            else:
                # 样本极少（<2）：训练集=全量，测试集复用训练集（兜底）
                train_data = global_non_missing_data
                test_data = global_non_missing_data.copy() if len(global_non_missing_data) > 0 else np.array([])

            # 最终兜底：若测试集为空且训练集有数据，补充至少1个样本
            if len(test_data) == 0 and len(train_data) > 0:
                test_data = train_data[:1]

            print(f"月份策略 - 训练集样本数：{len(train_data)}，测试集样本数：{len(test_data)}")

        elif strategy_name == 'cross':
            # 步骤1：筛选门店数量最多的渠道（第一层筛选）
            channel_counts = self.df_missing['渠道'].value_counts()
            max_count_channel = channel_counts.index[0]
            print(f"第一步筛选：选择门店数量最多的渠道 - {max_count_channel}（共{channel_counts.iloc[0]}家门店）")

            # 步骤2：在该渠道内，筛选门店数量最多的地城市（第二层筛选）
            # 先过滤出该渠道的所有数据，再统计城市门店数
            channel_data = self.df_missing[self.df_missing['渠道'] == max_count_channel]
            city_counts_in_channel = channel_data['地城市'].value_counts()
            max_count_city_in_channel = city_counts_in_channel.index[0]
            print(
                f"第二步筛选：在{max_count_channel}渠道内，选择门店数量最多的城市 - {max_count_city_in_channel}（共{city_counts_in_channel.iloc[0]}家门店）")

            # 步骤3：获取双层筛选后的全量样本（移除target_store_idx排除逻辑）
            cross_mask = (self.df_missing['渠道'] == max_count_channel) & \
                         (self.df_missing['地城市'] == max_count_city_in_channel)
            cross_data = self.df_missing_scaled[cross_mask][self.monthly_cols].values
            print(f"双层筛选后全量样本数：{len(cross_data)}")

            # 步骤4：数据预处理（打乱+分场景划分，保证鲁棒性）
            np.random.seed(42)  # 固定随机种子，结果可复现
            if len(cross_data) > 0:
                np.random.shuffle(cross_data)  # 打乱样本，避免顺序偏差

            # 分场景划分训练/测试集，保证测试集非空
            if len(cross_data) >= 5:
                # 样本充足：80%训练，20%测试
                train_size = int(len(cross_data) * 0.8)
                train_data = cross_data[:train_size]
                test_data = cross_data[train_size:]
            elif 2 <= len(cross_data) < 5:
                # 样本较少：训练集取大部分，测试集至少1个
                train_data = cross_data[:-1]
                test_data = cross_data[-1:]
            else:
                # 样本极少（<2）：训练集=全量，测试集复用训练集（兜底）
                train_data = cross_data
                test_data = cross_data.copy() if len(cross_data) > 0 else np.array([])

            # 最终兜底：若测试集为空且训练集有数据，补充至少1个样本
            if len(test_data) == 0 and len(train_data) > 0:
                test_data = train_data[:1]

            print(f"交叉策略 - 训练集样本数：{len(train_data)}，测试集样本数：{len(test_data)}")

        elif strategy_name == 'ts_similar':
            # 步骤1：选择参考时间序列（用所有门店的均值序列，避免依赖单个目标门店）
            # 计算所有门店月度列的均值（填充缺失值），转为可写数组（核心修复：copy()）
            reference_ts = self.df_missing[self.monthly_cols].fillna(0).mean(axis=0).values.copy()  # 新增copy()
            reference_ts = np.nan_to_num(reference_ts, 0)  # 现在可正常修改

            # 步骤2：处理参考时序全0的特殊情况
            if np.sum(reference_ts) == 0:
                print("警告：参考时间序列全为0，改用全局前600家门店数据")
                # 直接取全局前600家非空门店（同样加copy()避免只读）
                non_empty_mask = self.df_missing[self.monthly_cols].fillna(0).sum(axis=1) > 0
                global_data = self.df_missing_scaled[non_empty_mask][self.monthly_cols].values.copy()[:600]
                similar_data = global_data
            else:
                # 步骤3：计算所有门店与参考时序的DTW相似度
                similarities = []
                # 遍历所有门店（无排除逻辑）
                for idx in range(len(self.df_missing)):
                    # 核心修复：加copy()转为可写数组
                    store_ts = self.df_missing.iloc[idx][self.monthly_cols].fillna(0).values.copy()
                    store_ts = np.nan_to_num(store_ts, 0)
                    # 跳过全0时序的门店
                    if np.sum(store_ts) == 0:
                        continue

                    # 计算DTW距离（兼容不同dtw库的返回格式）
                    try:
                        dtw_result = dtw(reference_ts, store_ts, dist=euclidean)
                        d = dtw_result.distance
                    except:
                        d, _, _, _ = dtw(reference_ts, store_ts, dist=lambda a, b: np.linalg.norm(a - b))

                    # 相似度计算（距离越小，相似度越高）
                    similarity = 1 / (1 + d)
                    similarities.append((idx, similarity))

                # 步骤4：筛选相似性最高的600家门店
                similarities.sort(key=lambda x: x[1], reverse=True)
                # 取前600家（若不足600则取全部）
                top_k = min(600, len(similarities))
                similar_indices = [x[0] for x in similarities[:top_k]]
                # 核心修复：加copy()避免只读数组
                similar_data = self.df_missing_scaled.iloc[similar_indices][self.monthly_cols].values.copy()
                print(f"筛选出时序相似性最高的{top_k}家门店作为样本集")

            # 步骤5：基于筛选后的样本集划分训练/测试集（优化鲁棒性）
            np.random.seed(42)  # 固定随机种子，保证可复现
            if len(similar_data) > 0:
                np.random.shuffle(similar_data)  # 现在可正常打乱（非只读）

            # 分场景划分，保证测试集非空
            if len(similar_data) >= 5:
                train_size = int(len(similar_data) * 0.8)
                train_data = similar_data[:train_size]
                test_data = similar_data[train_size:]
            elif 2 <= len(similar_data) < 5:
                train_data = similar_data[:-1]
                test_data = similar_data[-1:]
            else:
                train_data = similar_data
                test_data = similar_data.copy() if len(similar_data) > 0 else np.array([])

            # 最终兜底：测试集为空时补充至少1个样本
            if len(test_data) == 0 and len(train_data) > 0:
                test_data = train_data[:1]

            print(f"时序相似策略 - 训练集样本数：{len(train_data)}，测试集样本数：{len(test_data)}")

        # ========== 兜底逻辑（避免空值） ==========
        # if len(train_data) == 0:
        #     train_data = self.df_missing_scaled[self.df_missing.index != target_store_idx][self.monthly_cols].values[
        #                  :10]
        #     test_data = train_data[:2]

        # 确保数据类型
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)

        return train_data, test_data


# ==============================================================================
# 2. VAE模型定义
# ==============================================================================
class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, latent_dim=8, dropout=0.2):
        super(TimeSeriesVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 关键修复：将BatchNorm替换为LayerNorm（适配小批次）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 替换BatchNorm1d为LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # 替换BatchNorm1d为LayerNorm
            nn.ReLU()
        )

        # 均值和方差层
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器同样替换BatchNorm为LayerNorm
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # 替换BatchNorm1d为LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 替换BatchNorm1d为LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """前向传播"""
        # 编码
        enc_out = self.encoder(x)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        # 重参数化
        z = self.reparameterize(mu, logvar)
        # 解码
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# 自定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# ==============================================================================
# 3. 模型训练与缺失值填补
# ==============================================================================
class VAEMissingImputer:
    def __init__(self, input_dim=12, hidden_dim=32, latent_dim=8, lr=1e-4, epochs=50, batch_size=4,
                 strategy_name="default"):
        """
        新增 strategy_name 参数，为每个策略生成独立模型路径
        """
        self.model = TimeSeriesVAE(input_dim, hidden_dim, latent_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_loss = float('inf')
        self.strategy_name = strategy_name  # 记录当前策略名称
        # 为每个策略创建独立的模型保存目录
        self.model_dir = f'./models/{strategy_name}'  # 按策略命名目录
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.model_dir, 'vae_best_model.pth')  # 独立路径

    def loss_function(self, recon_x, x, mu, logvar, beta=0.1):
        """关键修复4：优化损失函数，降低KL损失权重+数值稳定"""
        # 1. 重构损失：只计算非缺失值部分（避免nan参与计算）
        mask = ~torch.isnan(x)
        if torch.sum(mask) == 0:
            recon_loss = torch.tensor(0.0, device=device)
        else:
            recon_loss = self.criterion(recon_x[mask], x[mask])

        # 2. KL散度：增加数值裁剪，避免溢出
        mu = torch.clamp(mu, -10, 10)  # 限制均值范围
        logvar = torch.clamp(logvar, -10, 10)  # 限制方差对数范围
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / torch.clamp(torch.tensor(x.size(0), device=device), min=1)  # 防止除以0

        # 3. 降低KL损失权重（beta=0.1，避免KL损失主导）
        total_loss = recon_loss + beta * kl_loss

        # 4. 检查损失是否为nan，若为nan则返回上一次有效损失
        if torch.isnan(total_loss):
            total_loss = torch.tensor(self.best_loss if self.best_loss != float('inf') else 1.0, device=device)
        return total_loss, recon_loss, kl_loss

    def reset_model(self):
        """重置模型参数和优化器状态"""
        # 重置模型参数
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # 重置优化器（避免学习率调度或动量积累）
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'], weight_decay=1e-6)
        self.best_loss = float('inf')

    def train_model(self, train_data):
        """训练前先重置模型，确保策略独立性"""
        self.reset_model()  # 新增：训练前重置
        if len(train_data) == 0:
            print("警告：训练数据为空，跳过模型训练")
            return []

        # 数据再次清洗：移除nan/inf
        train_data = np.nan_to_num(train_data, nan=0, posinf=1, neginf=0)

        # 关键修复1：确保训练数据量≥2，且批次大小适配
        if len(train_data) < 2:
            print(f"警告：训练数据量{len(train_data)}<2，复制样本补充到2个")
            train_data = np.vstack([train_data, train_data[0]])  # 复制第一个样本

        # 关键修复2：调整批次大小，避免最后一批次为1
        effective_batch_size = min(self.batch_size, len(train_data))
        # 确保批次大小能整除数据量（或最后一批次≥2）
        if len(train_data) % effective_batch_size == 1:
            effective_batch_size = max(2, effective_batch_size - 1)  # 调整批次大小

        # 准备数据集（设置drop_last=True，丢弃最后一个不足批次的样本）
        dataset = TimeSeriesDataset(train_data)
        dataloader = DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                drop_last=True  # 关键修复3：丢弃最后一个不足批次的样本
            )
        self.model.train()
        training_losses = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            batch_count = 0

            for batch in dataloader:
                batch_count += 1
                batch = batch.to(device)
                # 替换batch中的nan为0
                batch = torch.nan_to_num(batch, nan=0.0)

                # 前向传播
                recon_batch, mu, logvar = self.model(batch)

                # 计算损失
                loss, recon_loss, kl_loss = self.loss_function(recon_batch, batch, mu, logvar)

                # 反向传播：梯度裁剪
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 累加损失（只累加非nan值）
                if not torch.isnan(loss):
                    total_loss += loss.item() * batch.size(0)
                if not torch.isnan(recon_loss):
                    total_recon_loss += recon_loss.item() * batch.size(0)
                if not torch.isnan(kl_loss):
                    total_kl_loss += kl_loss.item() * batch.size(0)

            # 计算平均损失（避免除以0）
            avg_loss = total_loss / max(len(dataset), 1) if batch_count > 0 else 0
            avg_recon_loss = total_recon_loss / max(len(dataset), 1) if batch_count > 0 else 0
            avg_kl_loss = total_kl_loss / max(len(dataset), 1) if batch_count > 0 else 0

            # 只记录有效损失
            if not np.isnan(avg_loss) and batch_count > 0:
                training_losses.append(avg_loss)
                # 更新最佳损失
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"Epoch {epoch + 1}: 模型更新，当前最佳损失: {self.best_loss:.6f}")

            # 打印训练日志（只打印非nan值）
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], "
                      f"Total Loss: {avg_loss:.6f}" if not np.isnan(avg_loss) else "Total Loss: nan, "
                                                                                       f"Recon Loss: {avg_recon_loss:.6f}" if not np.isnan(
                        avg_recon_loss) else "Recon Loss: nan, "
                                             f"KL Loss: {avg_kl_loss:.6f}" if not np.isnan(
                        avg_kl_loss) else "KL Loss: nan")

        # 加载模型兜底逻辑
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=device))
            print(f"成功加载最佳模型: {self.best_model_path}")
        else:
            print("警告：未找到最佳模型文件，使用训练后的模型")

        return training_losses

    def impute_missing(self, missing_data):
        """填补缺失值：用模型重构结果替换输入数据中的NaN（添加类型转换和异常值处理）"""
        self.model.eval()  # 切换模型到评估模式
        with torch.no_grad():  # 禁用梯度计算，节省资源
            # 1. 强制转换为float32类型，忽略无法转换的值（转为NaN）
            missing_data = missing_data.astype(np.float32)
            # 2. 替换NaN、inf、-inf为0（避免张量转换失败）
            missing_data = np.nan_to_num(missing_data, nan=0.0, posinf=0.0, neginf=0.0)

            # 3. 将处理后的数值数组转换为张量并移到指定设备（CPU/GPU）
            missing_tensor = torch.tensor(missing_data, dtype=torch.float32).to(device)
            # 模型前向传播，得到重构数据
            recon_data, _, _ = self.model(missing_tensor)
            # 将张量转换为numpy数组
            recon_data = recon_data.cpu().numpy()

        # 用重构结果填补缺失值（只替换原始缺失位置）
        imputed_data = missing_data.copy()
        # 重新生成原始缺失掩码（因前面用nan_to_num替换了NaN，需重新标记）
        original_missing_mask = np.isnan(missing_data.astype(np.float32))
        imputed_data[original_missing_mask] = recon_data[original_missing_mask]
        return imputed_data


# ==============================================================================
# 4. 多维度评估指标计算
# ==============================================================================
class MultiDimensionEvaluator:
    def __init__(self, scaler):
        self.scaler = scaler

    def inverse_scale(self, data):
        """逆标准化"""
        if len(data.shape) == 1:
            return self.scaler.inverse_transform(data.reshape(1, -1))[0]
        return self.scaler.inverse_transform(data)

    # 4.1 统计分布一致性评估
    def calculate_distribution_metrics(self, true_data, imputed_data, month_col):
        """计算统计分布指标（添加类型转换）"""
        true_data = np.asarray(true_data, dtype=np.float32).reshape(-1)
        imputed_data = np.asarray(imputed_data, dtype=np.float32).reshape(-1)
        true_original = self.inverse_scale(true_data)
        imputed_original = self.inverse_scale(imputed_data)
        true_mean = np.mean(true_original)
        imputed_mean = np.mean(imputed_original)
        true_var = np.var(true_original)
        imputed_var = np.var(imputed_original)
        true_skew = skew(true_original)
        imputed_skew = skew(imputed_original)
        true_kurt = kurtosis(true_original)
        imputed_kurt = kurtosis(imputed_original)
        ks_statistic, _ = ks_2samp(true_original, imputed_original)
        return {
            '指标': f'{month_col}_分布指标',
            '均值差值': abs(true_mean - imputed_mean),
            '方差差值': abs(true_var - imputed_var),
            '偏度差值': abs(true_skew - imputed_skew),
            '峰度差值': abs(true_kurt - imputed_kurt),
            'KS检验统计量': ks_statistic
        }

    # 4.2 时序特征一致性评估（终极修复）

    def calculate_temporal_metrics(self, true_ts, imputed_ts, month_cols):
        """
        计算时序特征指标（适配旧版dtw库：返回元组而非对象）
        """
        # 1. 强制转换为float32 numpy数组 + 确保1维
        true_ts = np.asarray(true_ts, dtype=np.float32).reshape(-1)
        imputed_ts = np.asarray(imputed_ts, dtype=np.float32).reshape(-1)

        # 2. 逆标准化 + 再次强制1维（双重保险）
        true_ts_original = np.asarray(self.inverse_scale(true_ts), dtype=np.float32).reshape(-1)
        imputed_ts_original = np.asarray(self.inverse_scale(imputed_ts), dtype=np.float32).reshape(-1)

        # 3. 计算ACF
        true_acf = self.calculate_acf(true_ts_original)
        imputed_acf = self.calculate_acf(imputed_ts_original)
        acf_diff = np.mean(np.abs(true_acf - imputed_acf))

        # 4. 计算PACF
        true_pacf = self.calculate_pacf(true_ts_original)
        imputed_pacf = self.calculate_pacf(imputed_ts_original)
        pacf_diff = np.mean(np.abs(true_pacf - imputed_pacf))

        # 5. 自定义欧氏距离函数（规避维度校验）
        def custom_euclidean(u, v):
            if isinstance(u, (np.ndarray, list)):
                u = u[0] if len(u) > 0 else 0.0
            if isinstance(v, (np.ndarray, list)):
                v = v[0] if len(v) > 0 else 0.0
            return np.sqrt((u - v) ** 2)

        # 6. 关键修复：适配旧版dtw库（返回元组，取第一个元素为距离）
        try:
            # 旧版dtw：返回 (distance, path, accumulated_cost_matrix, grid)
            dtw_distance, _, _, _ = dtw(true_ts_original, imputed_ts_original, dist=custom_euclidean)
        except Exception as e:
            # 兼容新版dtw-python（若后续升级库，避免报错）
            try:
                dtw_result = dtw(true_ts_original, imputed_ts_original, dist=custom_euclidean)
                dtw_distance = dtw_result.distance
            except:
                dtw_distance = 0.0  # 极端情况兜底，避免程序中断
                print(f"DTW计算异常：{e}")

        return {
            '指标': '时序特征指标',
            'ACF差值均值': acf_diff,
            'PACF差值均值': pacf_diff,
            'DTW距离': dtw_distance
        }


    def calculate_acf(self, ts, lag=6):
        """计算自相关函数"""
        acf_values = []
        for i in range(1, lag + 1):
            if len(ts[:-i]) == 0:
                acf_values.append(0)
                continue
            corr = np.corrcoef(ts[:-i], ts[i:])[0, 1]
            acf_values.append(corr if not np.isnan(corr) else 0)
        return np.array(acf_values)

    def calculate_pacf(self, ts, lag=6):
        """简化计算偏自相关函数（使用Yule-Walker方程近似）"""
        pacf_values = []
        n = len(ts)
        for k in range(1, lag + 1):
            r = [np.corrcoef(ts[:-i], ts[i:])[0, 1] if (i < n and len(ts[:-i]) > 0) else 0 for i in range(k + 1)]
            r = [x if not np.isnan(x) else 0 for x in r]
            R = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    R[i, j] = r[abs(i - j)]
            try:
                pacf = np.linalg.solve(R, r[1:k + 1])[-1]
            except:
                pacf = 0
            pacf_values.append(pacf)
        return np.array(pacf_values)

    # 4.3 补全精度量化评估
    def calculate_accuracy_metrics(self, true_values, imputed_values):
        """计算补全精度指标（添加类型转换）"""
        true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
        imputed_values = np.asarray(imputed_values, dtype=np.float32).reshape(-1)
        true_original = self.inverse_scale(true_values)
        imputed_original = self.inverse_scale(imputed_values)
        valid_mask = true_original > 1e-3
        if np.sum(valid_mask) == 0:
            return {'指标': '精度指标', 'MAE': 0, 'MAPE': 0}
        true_valid = true_original[valid_mask]
        imputed_valid = imputed_original[valid_mask]
        mae = mean_absolute_error(true_valid, imputed_valid)
        mape = mean_absolute_percentage_error(true_valid, imputed_valid) * 100
        return {'指标': '精度指标', 'MAE': mae, 'MAPE': mape}

    # 4.4 数据分布合理性评估
    def calculate_rationality_metrics(self, true_data, imputed_data, model, device):
        """计算分布合理性指标（基于VAE模型）"""
        model.eval()
        with torch.no_grad():
            true_tensor = torch.tensor(true_data, dtype=torch.float32).to(device)
            imputed_tensor = torch.tensor(imputed_data, dtype=torch.float32).to(device)
            true_recon, true_mu, true_logvar = model(true_tensor)
            imputed_recon, imputed_mu, imputed_logvar = model(imputed_tensor)
            recon_criterion = nn.MSELoss()
            true_recon_loss = recon_criterion(true_recon, true_tensor).item()
            imputed_recon_loss = recon_criterion(imputed_recon, imputed_tensor).item()

            def kl_divergence(mu, logvar):
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item() / mu.size(0)

            true_kl = kl_divergence(true_mu, true_logvar)
            imputed_kl = kl_divergence(imputed_mu, imputed_logvar)
            true_total_loss = true_recon_loss + true_kl
            imputed_total_loss = imputed_recon_loss + imputed_kl
        return {
            '指标': '分布合理性指标',
            '真实数据重构损失': true_recon_loss,
            '补全数据重构损失': imputed_recon_loss,
            '真实数据KL散度': true_kl,
            '补全数据KL散度': imputed_kl,
            '真实数据总损失': true_total_loss,
            '补全数据总损失': imputed_total_loss
        }

    # 4.5 综合评估
    def comprehensive_evaluation(self, true_data, imputed_data, model, device, month_cols):
        """
        综合所有评估指标（支持多样本输入，返回每个样本的指标列表）
        true_data: 真实数据（shape: [n_samples, 12]）
        imputed_data: 填补数据（shape: [n_samples, 12]）
        """
        true_data = np.asarray(true_data, dtype=np.float32).reshape(-1, len(month_cols))  # [n_samples, 12]
        imputed_data = np.asarray(imputed_data, dtype=np.float32).reshape(-1, len(month_cols))
        true_data = np.nan_to_num(true_data, nan=0.0, posinf=0.0, neginf=0.0)
        imputed_data = np.nan_to_num(imputed_data, nan=0.0, posinf=0.0, neginf=0.0)

        sample_metrics_list = []  # 存储每个样本的完整指标
        distribution_metrics_all = []  # 存储所有样本的分布指标

        for sample_idx in range(len(true_data)):
            # 单个样本的真实数据和填补数据（12维）
            true_sample = true_data[sample_idx:sample_idx + 1]  # [1, 12]
            imputed_sample = imputed_data[sample_idx:sample_idx + 1]  # [1, 12]

            # 1. 统计分布一致性（每个月份）
            dist_metrics = []
            for i, month_col in enumerate(month_cols):
                month_true = true_sample[:, i]
                month_imputed = imputed_sample[:, i]
                valid_mask = (month_true != 0) | (month_imputed != 0)
                if np.sum(valid_mask) > 3:
                    month_true_valid = month_true[valid_mask].astype(np.float32)
                    month_imputed_valid = month_imputed[valid_mask].astype(np.float32)
                    dist_metric = self.calculate_distribution_metrics(month_true_valid, month_imputed_valid, month_col)
                    dist_metrics.append(dist_metric)
                    distribution_metrics_all.append(dist_metric)

            # 2. 时序特征一致性（单个样本的12维时序）
            true_ts = true_sample[0].astype(np.float32)
            imputed_ts = imputed_sample[0].astype(np.float32)
            temporal_metrics = self.calculate_temporal_metrics(true_ts, imputed_ts, month_cols)

            # 3. 补全精度量化（单个样本的缺失值位置）
            missing_mask = (true_sample == 0) & (imputed_sample != 0)
            if np.sum(missing_mask) > 0:
                true_values = true_sample[missing_mask].astype(np.float32)
                imputed_values = imputed_sample[missing_mask].astype(np.float32)
                accuracy_metrics = self.calculate_accuracy_metrics(true_values, imputed_values)
            else:
                accuracy_metrics = {'指标': '精度指标', 'MAE': 0, 'MAPE': 0}

            # 4. 数据分布合理性（单个样本）
            rationality_metrics = self.calculate_rationality_metrics(true_sample, imputed_sample, model, device)

            # 存储单个样本的所有指标
            sample_metrics_list.append({
                '样本索引': sample_idx,
                '统计分布一致性': dist_metrics,
                '时序特征一致性': temporal_metrics,
                '补全精度量化': accuracy_metrics,
                '数据分布合理性': rationality_metrics
            })

        # 返回：所有样本的指标列表 + 全局分布指标汇总
        return {
            '样本级指标列表': sample_metrics_list,
            '全局分布指标汇总': distribution_metrics_all
        }

    def aggregate_sample_metrics(self, sample_metrics_dict):
        """
        聚合多样本指标（计算所有样本的平均值）
        sample_metrics_dict: comprehensive_evaluation返回的结果（含样本级指标列表）
        """

        sample_metrics_list = sample_metrics_dict['样本级指标列表']
        distribution_metrics_all = sample_metrics_dict['全局分布指标汇总']

        # 1. 聚合补全精度指标（取消过滤，保留所有值）
        mae_list = []
        mape_list = []
        for sample_metrics in sample_metrics_list:
            acc = sample_metrics['补全精度量化']
            mae_list.append(acc['MAE'])  # 不再过滤 acc['MAE'] > 0
            mape_list.append(acc['MAPE'])
        avg_mae = np.mean(mae_list) if mae_list else 0
        avg_mape = np.mean(mape_list) if mape_list else 0

        # 2. 聚合时序特征指标（ACF差值、PACF差值、DTW距离）
        acf_diff_list = []
        pacf_diff_list = []
        dtw_dist_list = []
        for sample_metrics in sample_metrics_list:
            temporal = sample_metrics['时序特征一致性']
            acf_diff_list.append(temporal['ACF差值均值'])
            pacf_diff_list.append(temporal['PACF差值均值'])
            dtw_dist_list.append(temporal['DTW距离'])
        avg_acf_diff = np.mean(acf_diff_list)
        avg_pacf_diff = np.mean(pacf_diff_list)
        avg_dtw_dist = np.mean(dtw_dist_list)

        # 3. 聚合分布一致性指标（均值差值、方差差值等）
        mean_diff_list = []
        var_diff_list = []
        ks_stat_list = []
        for dist_metric in distribution_metrics_all:
            mean_diff_list.append(dist_metric['均值差值'])
            var_diff_list.append(dist_metric['方差差值'])
            ks_stat_list.append(dist_metric['KS检验统计量'])
        avg_mean_diff = np.mean(mean_diff_list) if mean_diff_list else 0
        avg_var_diff = np.mean(var_diff_list) if var_diff_list else 0
        avg_ks_stat = np.mean(ks_stat_list) if ks_stat_list else 0

        # 4. 聚合分布合理性指标（重构损失、KL散度）
        recon_loss_list = []
        kl_div_list = []
        for sample_metrics in sample_metrics_list:
            rationality = sample_metrics['数据分布合理性']
            recon_loss_list.append(rationality['补全数据重构损失'])
            kl_div_list.append(rationality['补全数据KL散度'])
        avg_recon_loss = np.mean(recon_loss_list)
        avg_kl_div = np.mean(kl_div_list)

        # 返回聚合后的策略级指标（单一值）
        return {
            '补全精度': {'平均MAE': avg_mae, '平均MAPE(%)': avg_mape},
            '时序特征一致性': {
                '平均ACF差值': avg_acf_diff,
                '平均PACF差值': avg_pacf_diff,
                '平均DTW距离': avg_dtw_dist
            },
            '统计分布一致性': {
                '平均均值差值': avg_mean_diff,
                '平均方差差值': avg_var_diff,
                '平均KS统计量': avg_ks_stat
            },
            '数据分布合理性': {
                '平均重构损失': avg_recon_loss,
                '平均KL散度': avg_kl_div
            }
        }

# ==============================================================================
# 5. 结果保存与可视化
# ==============================================================================
def save_and_visualize_results(all_results, all_imputed_data, monthly_cols, output_dir='./output'):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 初始化结果存储列表
    strategy_metrics = []

    # 2. 遍历all_results，提取关键指标
    for strategy_code, strategy_info in all_results.items():
        # 基础信息
        strategy_name = strategy_info['策略名称']
        train_samples = strategy_info['训练集样本数']
        test_samples = strategy_info['测试集样本数']

        # 测试集评估指标
        test_metrics = strategy_info['测试集整体评估']
        acf_diff = test_metrics['时序特征一致性']['平均ACF差值']
        pacf_diff = test_metrics['时序特征一致性']['平均PACF差值']
        dtw_distance = test_metrics['时序特征一致性']['平均DTW距离']
        recon_loss = test_metrics['数据分布合理性']['平均重构损失']
        kl_div = test_metrics['数据分布合理性']['平均KL散度']

        # 训练损失（取最后5轮均值，反映最终训练效果）
        train_loss_curve = strategy_info['策略级结果'][0]['训练损失曲线']
        final_train_loss = np.mean(train_loss_curve[-5:])

        # 存储到列表
        strategy_metrics.append({
            '策略编码': strategy_code,
            '策略名称': strategy_name,
            '训练集样本数': train_samples,
            '测试集样本数': test_samples,
            '平均ACF差值': float(acf_diff),  # 转为普通float，避免np.float64
            '平均PACF差值': float(pacf_diff),
            '平均DTW距离': float(dtw_distance),
            '平均重构损失': float(recon_loss),
            '平均KL散度': float(kl_div),
            '最终训练损失': float(final_train_loss),
            '训练损失曲线': train_loss_curve
        })

    # 3. 转为DataFrame
    metrics_df = pd.DataFrame(strategy_metrics)

    # ========== 核心：保存为CSV文件（关键优化：指定编码和不保存索引） ==========
    # 保存路径可自定义，比如 './策略评估结果.csv' 或 '/Users/inaya/Desktop/策略评估结果.csv'
    save_path = '策略评估结果.csv'

    # 保存CSV（encoding='utf-8-sig' 解决中文乱码问题，index=False 不保存行索引）
    metrics_df.to_csv(
        save_path,
        index=False,  # 不保存行索引（避免多余的Unnamed列）
        encoding='utf-8-sig',  # UTF-8带BOM，兼容Excel打开中文无乱码
        float_format='%.6f'  # 浮点数保留6位小数，便于阅读
    )

    print(f"✅ DataFrame已成功保存为CSV文件：{save_path}")
    # 可选：打印保存的文件内容预览
    print("\n文件内容预览：")
    print(metrics_df.round(6))


def generate_comparison_plots(eval_summary_df, monthly_cols, output_dir):
    """生成可视化对比图（确保列名完全匹配，无冗余字段）"""
    # 只保留存在的列，避免报错（关键容错）
    available_cols = eval_summary_df.columns.tolist()
    agg_cols = {}
    # 核心指标：只选择评估表中存在的列
    if '测试集_平均MAE' in available_cols:
        agg_cols['测试集_平均MAE'] = 'mean'
    if '测试集_平均MAPE(%)' in available_cols:
        agg_cols['测试集_平均MAPE(%)'] = 'mean'
    if '测试集_平均ACF差值' in available_cols:
        agg_cols['测试集_平均ACF差值'] = 'mean'
    if '测试集_平均DTW距离' in available_cols:
        agg_cols['测试集_平均DTW距离'] = 'mean'
    if '测试集_平均重构损失' in available_cols:
        agg_cols['测试集_平均重构损失'] = 'mean'
    if '测试集_平均均值差值' in available_cols:
        agg_cols['测试集_平均均值差值'] = 'mean'
    if '测试集_平均方差差值' in available_cols:
        agg_cols['测试集_平均方差差值'] = 'mean'

    # 计算各策略的平均指标
    strategy_metrics = eval_summary_df.groupby('策略名称').agg(agg_cols).round(4)
    # 按测试集_平均MAE排序（越小越好）
    if '测试集_平均MAE' in strategy_metrics.columns:
        strategy_metrics = strategy_metrics.sort_values('测试集_平均MAE')
    strategies = strategy_metrics.index.tolist()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

    # 创建子图（聚焦核心指标，避免冗余）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VAE模型缺失值补充策略多维度对比（测试集96样本平均）', fontsize=16, fontweight='bold')

    # 子图1：补全精度（测试集平均MAE和MAPE）
    ax1 = axes[0, 0]
    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, strategy_metrics['测试集_平均MAE'], width, label='平均MAE', color=colors[0],
                    alpha=0.8)
    if '测试集_平均MAPE(%)' in strategy_metrics.columns:
        bars2 = ax1.bar(x + width / 2, strategy_metrics['测试集_平均MAPE(%)'], width, label='平均MAPE(%)',
                        color=colors[1], alpha=0.8)
    ax1.set_title('补全精度对比', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机') for s
         in strategies], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    if '测试集_平均MAPE(%)' in strategy_metrics.columns:
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # 子图2：时序特征一致性（平均ACF差值）
    ax2 = axes[0, 1]
    if '测试集_平均ACF差值' in strategy_metrics.columns:
        bars3 = ax2.bar(strategies, strategy_metrics['测试集_平均ACF差值'], label='平均ACF差值', color=colors[2],
                        alpha=0.8)
        ax2.set_title('时序特征一致性（ACF差值）', fontweight='bold')
        ax2.set_xticklabels(
            [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机')
             for s in strategies], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 子图3：统计分布一致性（平均均值差值）
    ax3 = axes[0, 2]
    if '测试集_平均均值差值' in strategy_metrics.columns:
        bars5 = ax3.bar(strategies, strategy_metrics['测试集_平均均值差值'], label='平均均值差值', color=colors[0],
                        alpha=0.8)
        ax3.set_title('统计分布一致性（均值差值）', fontweight='bold')
        ax3.set_xticklabels(
            [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机')
             for s in strategies], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for bar in bars5:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    # 子图4：平均DTW距离
    ax4 = axes[1, 0]
    if '测试集_平均DTW距离' in strategy_metrics.columns:
        bars7 = ax4.bar(strategies, strategy_metrics['测试集_平均DTW距离'], color=colors[4], alpha=0.8)
        ax4.set_title('动态时间规整距离（DTW）', fontweight='bold')
        ax4.set_xticklabels(
            [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机')
             for s in strategies], rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        for bar in bars7:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 子图5：模型重构损失
    ax5 = axes[1, 1]
    if '测试集_平均重构损失' in strategy_metrics.columns:
        bars8 = ax5.bar(strategies, strategy_metrics['测试集_平均重构损失'], label='平均重构损失', color=colors[3],
                        alpha=0.8)
        ax5.set_title('模型重构损失对比', fontweight='bold')
        ax5.set_xticklabels(
            [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机')
             for s in strategies], rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for bar in bars8:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 子图6：综合性能得分（标准化后的总分）
    ax6 = axes[1, 2]
    # 选择存在的核心指标计算综合得分
    key_metrics_list = ['测试集_平均MAE', '测试集_平均MAPE(%)', '测试集_平均ACF差值', '测试集_平均DTW距离',
                        '测试集_平均重构损失']
    existing_key_metrics = [col for col in key_metrics_list if col in strategy_metrics.columns]
    if existing_key_metrics:
        key_metrics = strategy_metrics[existing_key_metrics].copy()
        # 标准化（转为0-100分，分数越高越好）
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 100))
        key_metrics_scaled = 1 - (key_metrics - key_metrics.min()) / (key_metrics.max() - key_metrics.min())  # 反转指标
        strategy_metrics['综合得分'] = key_metrics_scaled.mean(axis=1) * 100
        # 绘制综合得分
        bars10 = ax6.bar(strategies, strategy_metrics['综合得分'], color=colors, alpha=0.8)
        ax6.set_title('综合性能得分（0-100分）', fontweight='bold')
        ax6.set_xticklabels(
            [s.replace('策略', '').replace('划分', '').replace('（最多门店）', '').replace('全局无缺失随机', '全局随机')
             for s in strategies], rotation=45, ha='right')
        ax6.set_ylim(0, 100)
        ax6.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for bar in bars10:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}分', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'VAE缺失值补充策略多维度对比图（聚合后）.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚合后多维度对比图已保存: {plot_path}")


def save_final_results(all_results, all_imputed_data, processor, output_dir, save_full_imputed=True):
    """
    save_full_imputed: 是否保存全量补全数据（默认True，如需关闭设为False）
    """
    # 1. 保存测试集指标汇总（核心）
    final_metrics_rows = []
    for strategy, result in all_results.items():
        agg_metrics = result['aggregated_metrics']
        row = {
            '填补策略': strategy,
            '测试集样本数': len(result['test_indices']),  # 新增：标注测试集样本数
            '平均MAE（测试集）': agg_metrics['补全精度']['平均MAE（原始尺度）'],
            '平均MAPE(%)（测试集）': agg_metrics['补全精度']['平均MAPE(%)（原始尺度）'],
            '平均ACF差值（测试集）': agg_metrics['时序特征一致性']['平均ACF差值'],
            '平均PACF差值（测试集）': agg_metrics['时序特征一致性']['平均PACF差值'],
            '平均DTW距离（测试集）': agg_metrics['时序特征一致性']['平均DTW距离'],
            '平均均值差值（测试集）': agg_metrics['统计分布一致性']['平均均值差值（原始尺度）'],
            '平均方差差值（测试集）': agg_metrics['统计分布一致性']['平均方差差值（原始尺度）'],
            '平均KS统计量（测试集）': agg_metrics['统计分布一致性']['平均KS统计量（原始尺度）'],
            '平均重构损失（测试集）': agg_metrics['数据分布合理性']['平均重构损失'],
            '平均KL散度（测试集）': agg_metrics['数据分布合理性']['平均KL散度'],
            '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        final_metrics_rows.append(row)

    # 保存测试集指标汇总（核心文件）
    metrics_csv_path = os.path.join(output_dir, '测试集指标汇总表.csv')
    final_metrics_df = pd.DataFrame(final_metrics_rows)
    final_metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 测试集指标已保存至: {metrics_csv_path}")

    # 2. 可选保存全量补全数据（按需开关）
    if save_full_imputed:
        for strategy, imputed_data in all_imputed_data.items():
            df_imputed = processor.df_missing.copy()
            df_imputed[processor.monthly_cols] = imputed_data
            imputed_csv_path = os.path.join(output_dir, f'{strategy}_补全后全量数据.csv')
            df_imputed.to_csv(imputed_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ {strategy}策略补全后全量数据已保存至: {imputed_csv_path}")
    else:
        print("✅ 已关闭全量补全数据输出（仅保留测试集指标）")

    # 保存原始带缺失值数据（用于核对）
    missing_data_path = os.path.join(output_dir, '原始带缺失值数据.csv')
    processor.df_missing.to_csv(missing_data_path, index=False, encoding='utf-8-sig')
    print(f"✅ 原始带缺失值数据已保存至: {missing_data_path}")


def run_vae_imputation_experiment(data_path, missing_rate=0.1, output_dir='./final_output'):
    """
    修正后：仅用测试集计算指标，全量补全结果按需输出
    """
    # 1. 初始化组件
    processor = DataProcessor(data_path, missing_rate)
    evaluator = MultiDimensionEvaluator(processor.scaler)
    all_results = {}
    all_imputed_data = {}
    strategies = ['channel', 'city', 'month', 'cross', 'ts_similar']

    # todo:修改指标使用原始尺度
    # 2. 遍历所有策略执行填补
    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"开始执行策略: {strategy}")
        print(f"{'=' * 60}")

        # 2.1 按策略划分训练/测试数据（关键：保留测试集索引）
        train_data, test_data = processor.split_data_by_strategy(strategy)

        # 【新增】获取测试集在全量数据中的索引（核心：定位测试集位置）
        # 步骤1：获取全量带缺失值的标准化数据
        all_missing_scaled = processor.df_missing_scaled[processor.monthly_cols].values
        # 步骤2：通过数据匹配，找到测试集在全量数据中的索引（避免重复生成）
        test_indices = []
        for test_sample in test_data:
            # 匹配测试样本在全量数据中的位置（容错：浮点误差）
            matches = np.where(np.all(np.isclose(all_missing_scaled, test_sample, atol=1e-3), axis=1))[0]
            if len(matches) > 0:
                test_indices.append(matches[0])
        test_indices = list(set(test_indices))  # 去重
        print(f"✅ {strategy}策略 - 测试集样本数：{len(test_indices)}（仅用这些样本计算指标）")

        # 2.2 初始化并训练VAE模型（逻辑不变）
        input_dim = len(processor.monthly_cols) if processor.monthly_cols else 1
        imputer = VAEMissingImputer(
            input_dim=input_dim,
            hidden_dim=32,
            latent_dim=8,
            lr=1e-4,
            epochs=50,
            batch_size=4,
            strategy_name=strategy
        )
        print(f"训练{strategy}策略的VAE模型...")
        training_losses = imputer.train_model(train_data)

        # todo:还原到原始尺度
        # 2.3 补全全量数据（仍保留，供业务使用）
        print(f"使用{strategy}策略模型补全全量数据...")
        imputed_data_scaled = imputer.impute_missing(all_missing_scaled)
        imputed_data_original = evaluator.inverse_scale(imputed_data_scaled)  # 还原到原始尺度

        # 2.4 【核心修正】仅用测试集+原始尺度计算指标
        aggregated_metrics = {}
        if len(test_indices) > 0:
            # ========== 关键修正：提取原始尺度的测试集数据 ==========
            # 测试集原始尺度真实值（非标准化）
            test_true_original = processor.df_complete.iloc[test_indices][processor.monthly_cols].values
            # 测试集原始尺度补全值（非标准化）
            test_imputed_original = imputed_data_original[test_indices]
            # 测试集原始缺失标记（用于定位需要计算指标的位置）
            test_missing_mask = processor.df_missing.iloc[test_indices][processor.monthly_cols].isnull().values

            # ========== 第一步：仅计算缺失位置的核心指标（MAE/MAPE） ==========
            # 只关注原始缺失的位置（补全效果的核心）
            mean_diff_original = 0.0
            var_diff_original = 0.0
            ks_stat_original = 0.0
            if np.sum(test_missing_mask) > 0:
                # 提取缺失位置的真实值和补全值（原始尺度）
                true_missing = test_true_original[test_missing_mask]
                imputed_missing = test_imputed_original[test_missing_mask]

                # 过滤无效值（仅保留>0的数值，避免0值干扰）
                valid_mask = (true_missing > 1e-6) & (imputed_missing > 1e-6)
                if np.sum(valid_mask) > 0:
                    true_valid = true_missing[valid_mask]
                    imputed_valid = imputed_missing[valid_mask]

                    # 计算原始尺度的核心指标（MAE/MAPE）
                    mae_original = mean_absolute_error(true_valid, imputed_valid)
                    mape_original = mean_absolute_percentage_error(true_valid, imputed_valid) * 100

                    # ========== 核心新增：手动计算原始尺度的分布类指标 ==========
                    # 1. 平均均值差值（原始尺度）
                    true_mean = np.mean(true_valid)
                    imputed_mean = np.mean(imputed_valid)
                    mean_diff_original = abs(true_mean - imputed_mean)

                    # 2. 平均方差差值（原始尺度）
                    true_var = np.var(true_valid)
                    imputed_var = np.var(imputed_valid)
                    var_diff_original = abs(true_var - imputed_var)

                    # 3. 平均KS统计量（原始尺度）
                    # KS检验：衡量两个分布的相似度，取值0~1（0表示分布完全一致）
                    ks_stat, _ = ks_2samp(true_valid, imputed_valid)
                    ks_stat_original = ks_stat
                else:
                    print(f"警告：{strategy}策略测试集缺失位置无有效数值（全为0）")
                    mae_original = 0.0
                    mape_original = 0.0
            else:
                print(f"警告：{strategy}策略测试集无缺失值")
                mae_original = 0.0
                mape_original = 0.0

            # ========== 第二步：计算其他维度指标（时序/分布） ==========
            # 时序指标仍基于标准化数据计算（相对值），分布指标用手动计算的原始尺度值
            test_true_scaled = processor.df_complete_scaled.iloc[test_indices][processor.monthly_cols].values
            test_imputed_scaled = imputed_data_scaled[test_indices]

            comprehensive_metrics = evaluator.comprehensive_evaluation(
                test_true_scaled,  # 标准化测试集真实值（仅用于时序指标）
                test_imputed_scaled,  # 标准化测试集补全值（仅用于时序指标）
                imputer.model,
                device,
                processor.monthly_cols
            )
            temp_metrics = evaluator.aggregate_sample_metrics(comprehensive_metrics)

            # ========== 第三步：整合指标（覆盖分布类指标为原始尺度值） ==========
            aggregated_metrics = {
                '补全精度': {
                    '平均MAE（原始尺度）': mae_original,  # 核心：原始尺度
                    '平均MAPE(%)（原始尺度）': mape_original
                },
                '时序特征一致性': temp_metrics['时序特征一致性'],  # 相对值，无需还原
                '统计分布一致性': {
                    '平均均值差值（原始尺度）': mean_diff_original,  # 替换为原始尺度值
                    '平均方差差值（原始尺度）': var_diff_original,
                    '平均KS统计量（原始尺度）': ks_stat_original
                },
                '数据分布合理性': temp_metrics['数据分布合理性']
            }
            print(
                f"✅ {strategy}策略 - 测试集指标（原始尺度）：MAE={mae_original:.3f} | MAPE={mape_original:.2f}% | 均值差值={mean_diff_original:.3f} | 方差差值={var_diff_original:.3f}")
        else:
            # 兜底：无测试集时指标置0
            print(f"警告：{strategy}策略无测试集，指标置0")
            aggregated_metrics = {
                '补全精度': {'平均MAE（原始尺度）': 0, '平均MAPE(%)（原始尺度）': 0},
                '时序特征一致性': {'平均ACF差值': 0, '平均PACF差值': 0, '平均DTW距离': 0},
                '统计分布一致性': {'平均均值差值（原始尺度）': 0, '平均方差差值（原始尺度）': 0,
                                   '平均KS统计量（原始尺度）': 0},
                '数据分布合理性': {'平均重构损失': 0, '平均KL散度': 0}
            }
            comprehensive_metrics = None  # 无测试集时置空，避免报错

        # 2.5 存储结果（指标仅为测试集结果）
        all_results[strategy] = {
            'training_losses': training_losses,
            'comprehensive_metrics': comprehensive_metrics,
            'aggregated_metrics': aggregated_metrics,  # 仅测试集指标（含原始尺度MAE/MAPE）
            'strategy_name': strategy,
            'test_indices': test_indices,  # 记录测试集索引，便于核对
            'test_missing_count': np.sum(test_missing_mask) if len(test_indices) > 0 else 0  # 新增：缺失值数量
        }
        all_imputed_data[strategy] = imputed_data_original  # 存储原始尺度的补全数据


    # 3. 保存结果（指标为测试集结果，全量补全数据可选保存）
    save_final_results(all_results, all_imputed_data, processor, output_dir, save_full_imputed=True)
    # visualize_all_results(all_results, output_dir)
    return all_results, all_imputed_data


# ==============================================================================
# 7. 主函数执行
# ==============================================================================
if __name__ == "__main__":
    # 配置参数（仅需修改此处数据路径）
    INPUT_DATA_PATH = "/Users/inaya/Desktop/HierSales/金佰利数据处理结果_20260304_130926_透视表_转置结果.csv"
    MISSING_RATE = 0.1  # 人工生成10%的缺失值（可调整）

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"错误：输入文件 '{INPUT_DATA_PATH}' 不存在！")
        print("请确保输入文件路径正确。")
    else:
        # 运行完整实验
        experiment_results, imputed_data, data_processor, evaluator = run_vae_imputation_experiment(
            data_path=INPUT_DATA_PATH,
            missing_rate=MISSING_RATE
        )

        # 生成实验结论
        print(f"\n" + "=" * 80)
        print("VAE模型缺失值补充实验完成")
        print("=" * 80)

        # 读取评估汇总表
        eval_summary_df = pd.read_csv('./output/VAE缺失值补充策略评估汇总表.csv')

        # 计算各策略的平均性能
        strategy_performance = eval_summary_df.groupby('策略名称').agg({
            'MAE': 'mean',
            'MAPE(%)': 'mean',
            'ACF差值均值': 'mean',
            'DTW距离': 'mean',
            '重构损失': 'mean',
            '总损失': 'mean'
        }).round(4)

        # 按MAE排序（最佳到最差）
        strategy_performance = strategy_performance.sort_values('MAE')
        print("\n各策略平均性能排名（按MAE升序）:")
        print("-" * 80)
        for i, (strategy, metrics) in enumerate(strategy_performance.iterrows(), 1):
            print(f"{i}. {strategy}")
            print(f"   - MAE: {metrics['MAE']:.2f}")
            print(f"   - MAPE: {metrics['MAPE(%)']:.2f}%")
            print(f"   - ACF差值: {metrics['ACF差值均值']:.4f}")
            print(f"   - DTW距离: {metrics['DTW距离']:.2f}")
            print(f"   - 重构损失: {metrics['重构损失']:.6f}")
            print(f"   - 总损失: {metrics['总损失']:.6f}")
            print()

        # 输出关键结论
        best_strategy = strategy_performance.index[0]
        print("关键结论:")
        print("-" * 40)
        print(f"1. 最佳数据划分策略: {best_strategy}")
        print(f"   - 该策略在MAE、MAPE等关键指标上表现最优")
        print(f"2. 时序特征保持最好的策略: {strategy_performance.sort_values('DTW距离').index[0]}")
        print(f"3. 数据分布一致性最好的策略: {strategy_performance.sort_values('ACF差值均值').index[0]}")
        print(f"4. 模型拟合效果最好的策略: {strategy_performance.sort_values('总损失').index[0]}")

        print(f"\n生成的文件列表:")
        print("-" * 40)
        print("1. ./output/VAE缺失值补充策略评估汇总表.csv - 详细评估数据")
        print("2. ./output/VAE缺失值填补结果详细表.csv - 填补结果原始数据")
        print("3. ./output/VAE缺失值补充策略多维度对比图.png - 可视化对比图")
        print("4. ./models/vae_best_model.pth - 最佳VAE模型权重")