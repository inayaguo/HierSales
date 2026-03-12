import numpy as np
import os
import re
import time
import torch
import torch.nn as nn
import warnings
from torch import optim
import pandas as pd

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from models.HierDA import Model as HierDA

warnings.filterwarnings('ignore')


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
        # 对 output_scale 单独设置更小的学习率
        output_scale_params = [p for n, p in self.model.named_parameters() if 'output_scale' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'output_scale' not in n]

        model_optim = optim.Adam([
            {'params': other_params, 'lr': self.args.learning_rate},
            {'params': output_scale_params, 'lr': self.args.learning_rate * 0.1}
        ])
        return model_optim
        # return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

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
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                else:
                    outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                preds.append(pred.numpy())
                trues.append(true.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('vali shape:', preds.shape, trues.shape)
        self.cal_acc(preds, trues)
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

        # ── 修复1：恢复 EarlyStopping，确保 checkpoint 被正确保存 ──
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
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_src, batch_x_src_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                        print(f'outputs[:,0,0] 前5个值: {outputs[:, 0, 0][:5].detach().cpu().numpy()}')
                        print(f'batch_y[:,0,0] 前5个值: {batch_y[:, 0, 0][:5].detach().cpu().numpy()}')

                        # 加这几行排查
                        print(f'batch_x_src has nan: {torch.isnan(batch_x_src).any()}')
                        print(f'batch_x has nan: {torch.isnan(batch_x).any()}')
                        print(f'outputs has nan: {torch.isnan(outputs).any()}')
                        print(f'batch_y has nan: {torch.isnan(batch_y).any()}')

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                            # 新增：打印各损失分项
                            print(f'main_loss={loss.item():.4f}, extra_loss={self.model.extra_loss.item():.4f}')
                            loss = loss + self.model.extra_loss
                        # if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                        #     loss = loss + self.model.extra_loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x_src, batch_x_src_mark, dec_inp, batch_y_mark, batch_x)
                    print(f'outputs[:,0,0] 前5个值: {outputs[:, 0, 0][:5].detach().cpu().numpy()}')
                    print(f'batch_y[:,0,0] 前5个值: {batch_y[:, 0, 0][:5].detach().cpu().numpy()}')

                    # 加这几行排查
                    print(f'batch_x_src has nan: {torch.isnan(batch_x_src).any()}')
                    print(f'batch_x has nan: {torch.isnan(batch_x).any()}')
                    print(f'outputs has nan: {torch.isnan(outputs).any()}')
                    print(f'batch_y has nan: {torch.isnan(batch_y).any()}')

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                        # 新增：打印各损失分项
                        print(f'main_loss={loss.item():.4f}, extra_loss={self.model.extra_loss.item():.4f}')
                        loss = loss + self.model.extra_loss
                    # if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                    #     loss = loss + self.model.extra_loss
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # ── 修复2：正确调用 early_stopping，保存 checkpoint ──
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ── 修复3：加载训练过程中保存的最优 checkpoint ──
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"已加载最优模型：{best_model_path}")
        else:
            print("警告：未找到 checkpoint 文件，使用训练结束时的模型权重")

        return self.model

    def test(self, setting, test=0):
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
        self.cal_acc(preds, trues)
        return

    def cal_acc(self, preds, trues):
        preds = preds[:, :, 0]
        trues = trues[:, :, 0]

        preds_flat = preds.flatten()
        trues_flat = trues.flatten()

        valid_indices = preds_flat >= -100000
        filtered_trues = trues_flat[valid_indices]
        filtered_preds = preds_flat[valid_indices]

        df = pd.DataFrame({
            'true': filtered_trues,
            'predict': filtered_preds,
            'loss_k': [self.loss_k] * len(filtered_trues)
        })
        csv_name = f"{self.args.month_predict}_predict_k{self.loss_k}.csv"
        df.to_csv(csv_name, index=False)

        mae, mse, rmse, mape, mspe = metric(filtered_preds, filtered_trues)
        result = []
        for i in range(len(filtered_trues)):
            if filtered_trues[i] != 0 and abs(filtered_trues[i] - filtered_preds[i]) / filtered_trues[i] <= 0.2:
                result.append(1)
            else:
                result.append(0)
        store_acc = sum(result) / len(result) if len(result) > 0 else 0

        print(f'===== Loss k={self.loss_k} =====')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, mape:{mape:.4f}, store_acc:{store_acc:.4f}')
        print(f'batch_size:{self.args.batch_size}, encoder_layers:{self.args.e_layers}, lr:{self.args.learning_rate}, d_model:{self.args.d_model}')

        base_dir = "output_hiersales"
        result_dir = os.path.join(base_dir, "result")
        result_path = os.path.join(result_dir, "k_value_experiment.csv")

        if os.path.exists(base_dir):
            os.makedirs(result_dir, exist_ok=True)
            header = 'model,loss_k,month,d_model,encoder_layers,batch_size,learning_rate,mae,mse,rmse,mape,mspe,store_acc\n'
            data_row = f'{self.args.model},{self.loss_k},{self.args.month_predict},{self.args.d_model},{self.args.e_layers},{self.args.batch_size},{self.args.learning_rate},{mae:.4f},{mse:.4f},{rmse:.4f},{mape:.4f},{mspe:.4f},{store_acc:.4f}\n'
            if not os.path.exists(result_path):
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(header)
            with open(result_path, 'a', encoding='utf-8') as f:
                f.write(data_row)
        else:
            print(f"基础目录 {base_dir} 不存在，无法写入文件")

        return