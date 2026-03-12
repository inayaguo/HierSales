# import numpy as np
# import os
# import re
# import time
# import torch
# import torch.nn as nn
# import warnings
# from torch import optim

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.metrics import metric
# from utils.tools import EarlyStopping, adjust_learning_rate, visual

# warnings.filterwarnings('ignore')


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         return self.log_reg_obj
    
#     def _select_criterion_mse(self):
#         return nn.MSELoss()
    
#     def log_reg_obj(self, preds, true):
#         k = 2
#         preds, true = preds[:, :, 0], true[:, :, 0]
#         # preds, true = preds[:, :, 0][:, 0], true[:, :, 0][:, 0]
#         gap = abs(preds - true)
#         gap_ratio = gap / torch.tensor(true)
#         loss = torch.where(gap_ratio > 0.2, k * gap_ratio, gap_ratio)
#         loss = torch.squeeze(loss)
#         loss = torch.sum(loss)
#         return loss
    
#     def vali(self, vali_data, vali_loader, setting, criterion):
#         total_loss = []
#         preds, trues = [], []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()
#                 loss = criterion(pred, true)
#                 total_loss.append(loss)
#                 preds.append(pred.numpy())
#                 trues.append(true.numpy())

#         preds = np.array(preds)
#         trues = np.array(trues)
        
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)
#         print('fedformer')
#         self.cal_acc(preds, trues)

#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss


#     def train(self, setting):
#         print('------------------loading dataset------------------')
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
#         print('------------------finish with dataset------------------')
#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()

#         # if self.args.model == 'Saleformer':
#         #     criterion = self._select_criterion()
#         # else:
#         #     criterion = self._select_criterion_mse()

#         criterion = self._select_criterion_mse()

#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
    
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)

#                     train_loss.append(loss.item())

#                 if (i + 1) % 10 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, setting, criterion)

#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
#                 epoch + 1, train_steps, train_loss, vali_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model


#     def cal_acc(self, preds, trues):
#         preds = preds[:, :, 0]
#         trues = trues[:, :, 0]
#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         result = []
#         for i in range(len(trues)):
#             if abs(trues[i] - preds[i]) / trues[i] <= 0.2:
#                 result.append(1)
#             else:
#                 result.append(0)
#         store_acc = sum(result) / len(result)

#         if 'mj' in self.args.data_path:
#             file_path = "result/total_result_0710_mj.csv"
#         elif 'kimberly' in self.args.data_path:
#             file_path = "result/total_result_0710_kim.csv"
#         else:
#             file_path = "result/total_result_0917_test.csv"
#         f = open(file_path, 'a')
#         if os.stat(file_path).st_size == 0:
#             f.write('model, dimension_model, encoder_layers, decoder_layers, batch_size, learning_rate, seed, month, mae, mape, mse, rmse, mspe, store_acc\n')
#         f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(f'{self.args.model}', f'{self.args.d_model}', f'{self.args.e_layers}', f'{self.args.d_layers}', f'{self.args.batch_size}', f'{self.args.learning_rate}', f'{self.args.set_seed}', f'{self.args.month_predict}', mae, mape, mse, rmse, mspe, store_acc))
#         f.write('\n')
#         f.close()

#         return
#
# import numpy as np
# import os
# import re
# import time
# import torch
# import torch.nn as nn
# import warnings
# from torch import optim
# import pandas as pd
#
# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.metrics import metric
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
#
# warnings.filterwarnings('ignore')
#
#
# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)
#
#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()
#
#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model
#
#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader
#
#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim
#
#     def _select_criterion(self):
#         return self.log_reg_obj
#
#     def _select_criterion_mse(self):
#         return nn.MSELoss()
#
#     def log_reg_obj(self, preds, true):
#         k = 2
#         preds, true = preds[:, :, 0], true[:, :, 0]
#         gap = abs(preds - true)
#         gap_ratio = gap / true
#         loss = torch.where(gap_ratio > 0.2, k * gap_ratio, gap_ratio)
#         loss = torch.squeeze(loss)
#         loss = torch.sum(loss)
#         return loss
#
#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         preds, trues = [], []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().cuda()
#                 # .to(self.device)
#                 batch_y = batch_y.float().cuda()
#                 # .to(self.device)
#
#                 batch_x_mark = batch_x_mark.float().cuda()
#                 # .to(self.device)
#                 batch_y_mark = batch_y_mark.float().cuda()
#                 # .to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()
#                 # .to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
#                 # .to(self.device)
#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()
#                 loss = criterion(pred, true)
#                 total_loss.append(loss)
#                 preds.append(pred.numpy())
#                 trues.append(true.numpy())
#
#         preds = np.array(preds)
#         trues = np.array(trues)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#
#         print('test shape:', preds.shape)
#         print('true shape:', trues.shape)
#         self.cal_acc(preds, trues)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss
#
#     def train(self, setting):
#         print('------------------loading dataset------------------')
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')
#         print('------------------finish with dataset------------------')
#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         time_now = time.time()
#
#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
#
#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion_mse()
#
#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()
#
#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []
#
#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().cuda()
#                 # .to(self.device)
#                 batch_y = batch_y.float().cuda()
#                 # .to(self.device)
#                 batch_x_mark = batch_x_mark.float().cuda()
#                 # .to(self.device)
#                 batch_y_mark = batch_y_mark.float().cuda()
#                 # .to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()
#                 # .to(self.device)
#
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
#                         # .to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                         # seq_x, seq_y, seq_x_mark, seq_y_mark
#
#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
#                     # .to(self.device)
#                     loss = criterion(outputs, batch_y)
#
#                     train_loss.append(loss.item())
#
#                 if (i + 1) % 10 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()
#
#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()
#
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             # test_loss = self.vali(test_data, test_loader, criterion)
#
#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
#                 epoch + 1, train_steps, train_loss, vali_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#             adjust_learning_rate(model_optim, epoch + 1, self.args)
#
#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))
#
#         return self.model
#
#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
#
#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().cuda()
#                 # .to(self.device)
#                 batch_y = batch_y.float().cuda()
#                 # .to(self.device)
#
#                 batch_x_mark = batch_x_mark.float().cuda()
#                 # .to(self.device)
#                 batch_y_mark = batch_y_mark.float().cuda()
#                 # .to(self.device)
#
#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()
#                 # .to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
#                 # .to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#
#                 pred = outputs
#                 true = batch_y
#                 preds.append(pred)
#                 trues.append(true)
#                 # if i % 20 == 0:
#                 #     input = batch_x.detach().cpu().numpy()
#                 #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#
#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds.shape)
#         print('true shape:', trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         self.cal_acc(preds, trues)
#         return
#
#     def cal_acc(self, preds, trues):
#         preds = preds[:, :, 0]
#         trues = trues[:, :, 0]
#
#         preds_flat = preds.flatten()
#         trues_flat = trues.flatten()
#
#         valid_indices = preds_flat >= -100000
#         # valid_indices = preds_flat >= 0
#
#         # 筛选有效的 true 和 predict
#         filtered_trues = trues_flat[valid_indices]
#         filtered_preds = preds_flat[valid_indices]
#
#         df = pd.DataFrame({
#             'true': filtered_trues,
#             'predict': filtered_preds
#         })
#         csv_name = str(self.args.month_predict)+ "_predict.csv"
#         # 输出到 CSV 文件
#         df.to_csv(csv_name, index=False)
#         mae, mse, rmse, mape, mspe = metric(filtered_preds, filtered_trues)
#         result = []
#         for i in range(len(filtered_trues)):
#             if abs(filtered_trues[i] - filtered_preds[i]) / filtered_trues[i] <= 0.2:
#                 result.append(1)
#             else:
#                 result.append(0)
#         store_acc = sum(result) / len(result)
#         print('mse:{}, mae:{}, mape:{}, store_acc:{}, batch_size:{}, feature_selection:{}, encoder_layers:{}, lr:{}, d_model:{}'.format(mse, mae, mape, store_acc, self.args.batch_size, self.args.features, self.args.e_layers, self.args.learning_rate, self.args.d_model))
#         # f = open("result/result_0918.csv", 'a')
#         # file_path = "result/clean_1008_0.3.csv"
#         train_month = re.sub('.csv', '', self.args.data_path.split('_')[-1])
#         # f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(f'{self.args.model}', train_month, mae, mape, 1 - mape, store_acc,
#         #                                             self.args.features, self.args.batch_size, self.args.e_layers, self.args.learning_rate, self.args.d_model))
#         # f.write('\n')
#         # f.close()
#         # f = open(file_path, 'a')
#         # if os.stat(file_path).st_size == 0:
#         #     f.write('model, dimension_model, encoder_layers, decoder_layers, batch_size, learning_rate, seed, month, mae, mape, mse, rmse, mspe, store_acc\n')
#         # f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(f'{self.args.model}', f'{self.args.d_model}', f'{self.args.e_layers}', f'{self.args.d_layers}', f'{self.args.batch_size}', f'{self.args.learning_rate}', f'{self.args.set_seed}', f'{self.args.month_predict}', mae, mape, mse, rmse, mspe, store_acc))
#         # f.write('\n')
#         # f.close()
#
#         return

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

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 新增：从args中获取损失函数惩罚系数k，默认2
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 改为使用实例变量self.loss_k
        return self.log_reg_obj

    def _select_criterion_mse(self):
        return nn.MSELoss()

    def log_reg_obj(self, preds, true):
        k = self.loss_k  # 动态获取k值
        preds, true = preds[:, :, 0], true[:, :, 0]
        gap = abs(preds - true)
        # 新增：处理true为0的情况，避免除以0
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                preds.append(pred.numpy())
                trues.append(true.numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('test shape:', preds.shape)
        print('true shape:', trues.shape)
        self.cal_acc(preds, trues)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def save_weights_to_csv(self, epoch, month, enc_core_weights, dec_core_weights):
        base_dir = "/kaggle/working"
        result_dir = os.path.join(base_dir, "result")
        result_path = os.path.join(result_dir, "weights_second_col_summary.csv")

        if os.path.exists(base_dir):
            os.makedirs(result_dir, exist_ok=True)

            # 提取特征贡献度（加异常处理）
            try:
                enc_second_col = enc_core_weights["feature_contribution"]
                enc_second_col = enc_second_col if isinstance(enc_second_col, (list, np.ndarray)) else [enc_second_col]
            except:
                enc_second_col = []

            try:
                dec_second_col = dec_core_weights["feature_contribution"]
                dec_second_col = dec_second_col if isinstance(dec_second_col, (list, np.ndarray)) else [dec_second_col]
            except:
                dec_second_col = []

            # 准备数据
            header = 'epoch,month,enc_core_second_col,dec_core_second_col\n'
            max_len = max(len(enc_second_col), len(dec_second_col))
            data_rows = []
            for i in range(max_len):
                enc_val = enc_second_col[i] if i < len(enc_second_col) else ""
                dec_val = dec_second_col[i] if i < len(dec_second_col) else ""
                data_row = f'{epoch},{month},{enc_val:.4f},{dec_val:.4f}\n' if isinstance(enc_val,
                                                                                          (int, float)) and isinstance(
                    dec_val, (int, float)) else f'{epoch},{month},{enc_val},{dec_val}\n'
                data_rows.append(data_row)

            # 写入CSV
            if not os.path.exists(result_path):
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(header)
            with open(result_path, 'a', encoding='utf-8') as f:
                f.writelines(data_rows)
            print(f"✅ 数据写入成功：{result_path}")
        else:
            print(f"❌ 基础目录 {base_dir} 不存在")

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
        # 新增：支持选择自定义损失函数或MSE
        if self.args.loss == 'Custom':
            criterion = self._select_criterion()
        else:
            criterion = self._select_criterion_mse()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 新增：设置参数保存/打印的间隔（可根据需求调整）
        save_interval = 5  # 每5个epoch保存一次参数、打印一次特征贡献度

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().cuda()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
                        loss = criterion(outputs, batch_y)
                        if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                            loss = loss + self.model.extra_loss
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
                    loss = criterion(outputs, batch_y)
                    if hasattr(self.model, "extra_loss") and self.model.extra_loss is not None:
                        loss = loss + self.model.extra_loss
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
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # ===================== 核心新增：调用参数保存和打印函数 =====================
            # 每 save_interval 个epoch执行一次，或最后一个epoch强制执行
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == self.args.train_epochs:
                print(f"\n---------- Epoch {epoch + 1}: 保存Embedding参数并打印特征贡献度 ----------")
                # 保存模型参数权重
                self.model.save_embedding_weights()
                # 打印特征贡献度（传入args作为configs）
                # self.model.print_feature_contribution(self.args)
                enc_core_weights, dec_core_weights = self.model.print_feature_contribution(self.args)
                self.save_weights_to_csv(epoch + 1, self.args.month_predict, enc_core_weights, dec_core_weights)

            # ==========================================================================

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                # 早停时也保存一次最终的参数
                print("\n---------- 早停触发：保存最终Embedding参数 ----------")
                self.model.save_embedding_weights()
                # self.model.print_feature_contribution(self.args)
                enc_core_weights, dec_core_weights = self.model.print_feature_contribution(self.args)
                self.save_weights_to_csv(epoch + 1, self.args.month_predict, enc_core_weights, dec_core_weights)
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 训练结束后，再次确认保存一次最优模型的参数
        print("\n---------- 训练完成：保存最优模型Embedding参数 ----------")
        self.model.save_embedding_weights()
        self.model.print_feature_contribution(self.args)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().cuda()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape)
        print('true shape:', trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
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

        # 保存预测结果（区分不同k值）
        df = pd.DataFrame({
            'true': filtered_trues,
            'predict': filtered_preds,
            'loss_k': [self.loss_k] * len(filtered_trues)  # 新增：记录当前k值
        })
        csv_name = f"{self.args.month_predict}_predict_k{self.loss_k}.csv"
        df.to_csv(csv_name, index=False)

        # 计算评估指标
        mae, mse, rmse, mape, mspe = metric(filtered_preds, filtered_trues)
        result = []
        for i in range(len(filtered_trues)):
            if filtered_trues[i] != 0 and abs(filtered_trues[i] - filtered_preds[i]) / filtered_trues[i] <= 0.2:
                result.append(1)
            else:
                result.append(0)
        store_acc = sum(result) / len(result) if len(result) > 0 else 0

        # 打印详细信息（包含k值）
        print(f'===== Loss k={self.loss_k} =====')
        print(f'mse:{mse:.4f}, mae:{mae:.4f}, mape:{mape:.4f}, store_acc:{store_acc:.4f}')
        print(
            f'batch_size:{self.args.batch_size}, encoder_layers:{self.args.e_layers}, lr:{self.args.learning_rate}, d_model:{self.args.d_model}')

        # 保存k值对比结果
        # 1. 定义基础路径和目标文件路径
        base_dir = "/kaggle/working"
        result_dir = os.path.join(base_dir, "result")
        result_path = os.path.join(result_dir, "k_value_experiment.csv")

        # 2. 检查并创建基础目录和子文件夹（确保路径存在）
        if os.path.exists(base_dir):  # 先判断kaggle/working是否存在
            os.makedirs(result_dir, exist_ok=True)  # 创建result子文件夹，已存在则不报错

            # 3. 准备写入内容（表头+数据行）
            header = 'model,loss_k,month,d_model,encoder_layers,batch_size,learning_rate,mae,mse,rmse,mape,mspe,store_acc\n'
            data_row = f'{self.args.model},{self.loss_k},{self.args.month_predict},{self.args.d_model},{self.args.e_layers},{self.args.batch_size},{self.args.learning_rate},{mae:.4f},{mse:.4f},{rmse:.4f},{mape:.4f},{mspe:.4f},{store_acc:.4f}\n'

            # 4. 写入文件（不存在则创建+写表头，存在则追加数据）
            if not os.path.exists(result_path):
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(header)  # 新文件先写表头
            with open(result_path, 'a', encoding='utf-8') as f:
                f.write(data_row)  # 追加写入数据行
        else:
            print(f"基础目录 {base_dir} 不存在，无法写入文件")

        # result_path = "/kaggle/working/result/k_value_experiment.csv"
        # if not os.path.exists(result_path):
        #     with open(result_path, 'w', encoding='utf-8') as f:
        #         f.write(
        #             'model,loss_k,month,d_model,encoder_layers,batch_size,learning_rate,mae,mse,rmse,mape,mspe,store_acc\n')
        #
        # with open(result_path, 'a', encoding='utf-8') as f:
        #     f.write(
        #         f'{self.args.model},{self.loss_k},{self.args.month_predict},{self.args.d_model},{self.args.e_layers},{self.args.batch_size},{self.args.learning_rate},{mae:.4f},{mse:.4f},{rmse:.4f},{mape:.4f},{mspe:.4f},{store_acc:.4f}\n')

        return