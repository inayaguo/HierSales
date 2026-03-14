import numpy as np
import os
import re
import time
import torch
import torch.nn as nn
import warnings
from torch import optim
import pandas as pd
from datetime import datetime

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')

# ── 结果写入工具（与 run_batch.py 共享同一个 CSV） ──────────────────────────
RESULT_DIR = os.path.join("output_hiersales", "result")
# 美赞
# RESULT_CSV = os.path.join(RESULT_DIR, "batch_experiment_results_mz_all_exp.csv")
# 美赞消融结果保存路径
RESULT_CSV = os.path.join(RESULT_DIR, "batch_experiment_results_mz_ab.csv")
# 金佰利
# RESULT_CSV = os.path.join(RESULT_DIR, "batch_experiment_result_all_exp.csv")

RESULT_COLUMNS = [
    "run_time",
    "model",
    "month",
    "domain_split",
    "experiment_mode",   # ← 新增：hierda / target_only / source_only
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


def _write_result_row(row: dict):
    """将单行结果追加写入 CSV；若文件/目录不存在则自动创建。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    row.setdefault("run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_new = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)
    write_header = not os.path.exists(RESULT_CSV)
    df_new.to_csv(RESULT_CSV, mode="a", header=write_header, index=False)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.loss_k = args.loss_k if hasattr(args, 'loss_k') else 2

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        output_scale_params = [p for n, p in self.model.named_parameters() if 'output_scale' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'output_scale' not in n]
        model_optim = optim.Adam([
            {'params': other_params, 'lr': self.args.learning_rate},
            {'params': output_scale_params, 'lr': self.args.learning_rate * 0.1}
        ])
        return model_optim

    def _select_criterion(self):
        return self.log_reg_obj

    def _select_criterion_mse(self):
        return nn.MSELoss()

    def log_reg_obj(self, preds, true):
        k = self.loss_k
        preds, true = preds[:, :, 0], true[:, :, 0]
        gap = abs(preds - true)
        gap_ratio = gap / torch.where(true == 0, torch.ones_like(true), true)
        loss = torch.where(gap_ratio > 0.2, k * gap_ratio, gap_ratio)
        loss = torch.squeeze(loss)
        loss = torch.sum(loss)
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds, trues = [], []
        self.model.eval()
        # eval 模式下清除上一步训练遗留的 extra_loss，避免影响 vali loss 计算
        if hasattr(self.model, 'extra_loss'):
            self.model.extra_loss = None
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark) in enumerate(vali_loader):
                batch_x       = batch_x.float().to(self.device)
                batch_y       = batch_y.float().to(self.device)
                batch_x_mark  = batch_x_mark.float().to(self.device)
                batch_y_mark  = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                else:
                    outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred    = outputs.detach().cpu()
                true    = batch_y.detach().cpu()
                loss    = criterion(pred, true)
                total_loss.append(loss)
                preds.append(pred.numpy())
                trues.append(true.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('vali shape:', preds.shape, trues.shape)
        self.cal_acc(preds, trues, phase="val", write_csv=False)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print('------------------loading dataset------------------')
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print('------------------finish with dataset------------------')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        if self.args.loss == 'Custom':
            criterion = self._select_criterion()
        else:
            criterion = self._select_criterion_mse()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss       = []
            extra_loss_accum = []   # 每 epoch 汇总 extra_loss，末尾打印一次

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                        f_dim   = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss    = criterion(outputs, batch_y)
                        if hasattr(self.model, 'extra_loss') and self.model.extra_loss is not None:
                            extra_loss_accum.append(self.model.extra_loss.item())
                            loss = loss + self.model.extra_loss
                else:
                    outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                    f_dim   = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss    = criterion(outputs, batch_y)
                    if hasattr(self.model, 'extra_loss') and self.model.extra_loss is not None:
                        extra_loss_accum.append(self.model.extra_loss.item())
                        loss = loss + self.model.extra_loss
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now   = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # extra_loss 汇总：每 epoch 打印一次均值，不刷屏
            if extra_loss_accum:
                print("  [{model}] avg extra_loss this epoch: {el:.6f}".format(
                    model=self.args.model,
                    el=float(np.mean(extra_loss_accum))
                ))
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"已加载最优模型：{best_model_path}")
        else:
            print("警告：未找到 checkpoint 文件，使用训练结束时的模型权重")

        return self.model

    def test(self, setting, test=0, return_metrics=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path))
            else:
                print(f"警告：checkpoint 不存在：{ckpt_path}")

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                else:
                    outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        metrics = self.cal_acc(preds, trues, phase="test", write_csv=True)

        if return_metrics:
            return metrics
        return

    # ────────────────────────────────────────────────────────────────────
    # cal_acc：计算指标，写入 CSV，并返回 dict
    # ────────────────────────────────────────────────────────────────────
    def cal_acc(self, preds, trues, phase="test", write_csv=True):
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

        preds_flat = preds.flatten()
        trues_flat = trues.flatten()

        valid_indices = preds_flat >= -100000
        filtered_trues = trues_flat[valid_indices]
        filtered_preds = preds_flat[valid_indices]

        df = pd.DataFrame({
            'true':    filtered_trues,
            'predict': filtered_preds,
            'loss_k':  [self.loss_k] * len(filtered_trues)
        })
        csv_name = f"{self.args.month_predict}_predict_k{self.loss_k}.csv"
        df.to_csv(csv_name, index=False)

        mae, mse, rmse, mape, mspe = metric(filtered_preds, filtered_trues)

        result_flags = []
        for i in range(len(filtered_trues)):
            if filtered_trues[i] != 0 and abs(filtered_trues[i] - filtered_preds[i]) / filtered_trues[i] <= 0.2:
                result_flags.append(1)
            else:
                result_flags.append(0)
        store_acc = sum(result_flags) / len(result_flags) if result_flags else 0

        print(f'===== [{self.args.model}] Loss k={self.loss_k} | phase={phase} =====')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, mape:{mape:.4f}, store_acc:{store_acc:.4f}')
        print(
            f'batch_size:{self.args.batch_size}, encoder_layers:{self.args.e_layers}, '
            f'lr:{self.args.learning_rate}, d_model:{self.args.d_model}'
        )

        metrics_dict = dict(
            mae       = round(float(mae),       4),
            mse       = round(float(mse),       4),
            rmse      = round(float(rmse),      4),
            mape      = round(float(mape),      4),
            mspe      = round(float(mspe),      4),
            store_acc = round(float(store_acc), 4),
        )

        hyper_dict = dict(
            model           = self.args.model,
            month           = getattr(self.args, 'month_predict',    ''),
            domain_split    = getattr(self.args, 'domain_split',     'district'),
            experiment_mode = getattr(self.args, 'experiment_mode',  'hierda'),  # ← 新增
            feature_select  = getattr(self.args, 'features',         ''),
            d_model         = self.args.d_model,
            e_layers        = self.args.e_layers,
            batch_size      = self.args.batch_size,
            learning_rate   = self.args.learning_rate,
            loss_type       = getattr(self.args, 'loss',    'Custom'),
            loss_k          = self.loss_k,
            seq_len         = self.args.seq_len,
            label_len       = self.args.label_len,
            pred_len        = self.args.pred_len,
            seed            = getattr(self.args, 'set_seed', ''),
            phase           = phase,
        )

        # 新增消融实验写入模块
        is_ablation = getattr(self.args, 'ablation_mode', 'full') != 'full' \
                      or getattr(self.args, 'freq_mode', 'both') != 'both' \
                      or getattr(self.args, 'granularity_levels', 3) != 3

        if write_csv and os.path.exists("output_hiersales") and not is_ablation:

        # if write_csv and os.path.exists("output_hiersales"):
            result_dir  = os.path.join("output_hiersales", "result")
            os.makedirs(result_dir, exist_ok=True)
            old_result_path = os.path.join(result_dir, "k_value_experiment.csv")
            old_header  = 'model,loss_k,month,d_model,encoder_layers,batch_size,learning_rate,mae,mse,rmse,mape,mspe,store_acc\n'
            old_data_row = (
                f'{self.args.model},{self.loss_k},{self.args.month_predict},'
                f'{self.args.d_model},{self.args.e_layers},{self.args.batch_size},'
                f'{self.args.learning_rate},{mae:.4f},{mse:.4f},{rmse:.4f},{mape:.4f},{mspe:.4f},{store_acc:.4f}\n'
            )
            if not os.path.exists(old_result_path):
                with open(old_result_path, 'w', encoding='utf-8') as f:
                    f.write(old_header)
            with open(old_result_path, 'a', encoding='utf-8') as f:
                f.write(old_data_row)

            _write_result_row({**hyper_dict, **metrics_dict})
            print(f"📄 指标已写入：{RESULT_CSV}")
        else:
            if not os.path.exists("output_hiersales"):
                print(f"基础目录 output_hiersales 不存在，跳过文件写入")

        return metrics_dict