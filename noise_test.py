import argparse
import os
import time
from multiprocessing import freeze_support
from utils.tools import test_params_flop
import torch
from exp.exp_noise import Exp_Main
from utils.metrics import metric
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MSGNet for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='short_term_forecast',
                    help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,'
                         ' S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')


parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock/ScaleGraphBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
parser.add_argument('--tanhalpha', type=float, default=3, help='')

#GCN
parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
parser.add_argument('--gcn_depth', type=int, default=2, help='')
parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
parser.add_argument('--propalpha', type=float, default=0.3, help='')
parser.add_argument('--conv_channel', type=int, default=32, help='')
parser.add_argument('--skip_channel', type=int, default=32, help='')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default '
                                                              '1: value embedding + temporal embedding + positional embedding '
                                                              '2: value embedding + temporal embedding '
                                                              '3: value embedding + positional embedding '
                                                              '4: value embedding')
# CrossGNN 
parser.add_argument('--tk', type=int, default=10, help='top-k for time adjacency')
parser.add_argument('--scale_number', type=int, default=4, help='number of scales')
parser.add_argument('--tvechidden', type=int, default=1, help='time vector hidden dim')
parser.add_argument('--nvechidden', type=int, default=1, help='node vector hidden dim')
parser.add_argument('--anti_ood', type=int, default=1, help='simple strategy to solve data shift')
parser.add_argument('--use_tgcn', type=int, default=1, help='use cross-scale gnn')
parser.add_argument('--use_ngcn', type=int, default=1, help='use cross-variable gnn')
parser.add_argument('--hidden', type=int, default=8, help='channel dim')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    print(">>>>>>>train<<<<<<<<<<<<<<<<<<<<")
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
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
                                                                                                    args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    snr_levels, rmse_results = exp.test_noise(setting, test=1)

    dataset_name = Path(args.data_path).stem 
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, rmse_results, 
             marker='o', 
             linestyle='-', 
             color='#1f77b4',
             linewidth=2)
    
    # 样式设置
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title(f'{args.model}({dataset_name})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(0, 101, 10))

    # 保存结果
    os.makedirs(f"noise_96/{args.model}", exist_ok=True)
    plt.savefig(f"noise_96/{args.model}/{dataset_name}_{args.pred_len}.pdf", bbox_inches='tight')
    
    # 保存数值结果
    np.savetxt(f"noise_96/{args.model}/{dataset_name}_{args.pred_len}.csv",
               np.column_stack([snr_levels[::-1], rmse_results[::-1]]),
               delimiter=',',
               header='SNR(dB),RMSE',
               fmt='%.4f',
               comments='')

    torch.cuda.empty_cache()