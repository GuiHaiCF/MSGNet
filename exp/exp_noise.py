from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Informer, Autoformer, DLinear, MSGNet, CrossGNN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim, autograd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

def add_gaussian_noise(x, snr_db):
    """添加高斯噪声"""
    signal_power = torch.mean(x**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'MSGNet': MSGNet,
            'CrossGNN':CrossGNN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    #flag = 'train' or 'val' or 'test'
    def _get_data(self, flag):
        # 获取数据集和数据加载器（flag指定训练/验证/测试）
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def test_noise(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()

        snr_levels = np.arange(0, 101, 10)  # 0-100dB步长10
        rmse_results = []
        with torch.no_grad():
                for snr in tqdm(snr_levels, desc="Testing SNR levels"):
                    epoch_loss = []
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                        batch_x = add_gaussian_noise(batch_x, snr).float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if 'Linear' in self.args.model:
                                    outputs = self.model(batch_x)
                                else:
                                    if self.args.output_attention:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                    else:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if 'Linear' in self.args.model:
                                    outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        # print(outputs.shape,batch_y.shape)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = torch.mean((outputs - batch_y)**2)
                        epoch_loss.append(loss.item())

                    # 转换MSE到RMSE
                    rmse = np.sqrt(np.mean(epoch_loss))
                    rmse_results.append(rmse)
                    print(f"SNR: {snr}dB | RMSE: {rmse:.4f}")

        return snr_levels, rmse_results