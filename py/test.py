#!/usr/bin/env python
# coding: utf-8
import gc

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
from datetime import timedelta
import torchvision

sys.path.insert(0, '..')
torch.manual_seed(42)
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
batch_size = 4
day = 20
predict_days = 60  # tuning every time
categories = 2
year_list = np.arange(2000, 2020, 1)

use_gpu = True
device = 'cuda' if use_gpu else "cpu"
if use_gpu:
    from utils.gpu_tools import *

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(obj) for obj in select_gpu(query_gpu())])

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=True)


# In[6]:


images = []
label_df = []
for year in year_list:
    images.append(
        np.memmap(os.path.join(f"./img_data/monthly_{day}d", f"{day}d_month_has_vb_[{day}]_ma_{year}_images.dat"),
                  dtype=np.uint8, mode='r').reshape(
            (-1, IMAGE_HEIGHT[day], IMAGE_WIDTH[day])))
    label_df.append(pd.read_feather(
        os.path.join(f"./img_data/monthly_{day}d", f"{day}d_month_has_vb_[{day}]_ma_{year}_labels_w_delay.feather")))

images = np.concatenate(images)
label_df = pd.concat(label_df)

print(images.shape)
print(label_df.shape)


# ## build dataset

class MyDataset(Dataset):

    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


class Evaluation(object):
    def __init__(self, NV):
        self.NV = NV
        pass

    def get_annual_profit(self):  # 计算公式(1+r_year)^(day/365)=r
        days = (self.NV.index[-1] - self.NV.index[0]).days
        annual_profit = np.power(self.NV.iloc[-1] / self.NV.iloc[0], 1 / (days / 365)) - 1
        return annual_profit

    def get_max_withdraw(self, ):
        max_withdraw = 0
        max_withdraw_date = None
        peak = self.NV.iloc[0]
        i = 0
        for price in self.NV:
            if price > peak:
                peak = price
            withdraw = (peak - price) / peak
            if withdraw > max_withdraw:
                max_withdraw = withdraw
                max_withdraw_date = self.NV.index[i]
            i += 1

        return max_withdraw, max_withdraw_date

    def get_annual_volatility(self):
        # 计算每日收益率
        daily_returns = (self.NV - self.NV.shift(1)) / self.NV.shift(1)
        # 计算每日收益率的标准差，代表股票的日波动率
        daily_volatility = np.std(daily_returns)
        # 将日波动率转换为年波动率（假设一年有252个交易日）
        annual_volatility = daily_volatility * np.sqrt(252 / predict_days)
        return annual_volatility

    def get_sharpe(self):
        Er = self.get_annual_profit()
        sigma = self.get_annual_volatility()
        return Er / sigma

    def get_kamma(self):
        Er = self.get_annual_profit()
        withdraw, withdraw_date = self.get_max_withdraw()
        return Er / withdraw

    def generate_info(self):
        r = (self.NV.iloc[-1] - self.NV.iloc[0]) / self.NV.iloc[0]
        annual_r = self.get_annual_profit()
        sigma = self.get_annual_volatility()
        sharpe = self.get_sharpe()
        kamma = self.get_kamma()
        max_withdraw, max_withdraw_date = self.get_max_withdraw()
        return pd.Series(data=[r, annual_r, sigma, max_withdraw, max_withdraw_date, sharpe, kamma],
                         index=['period_return', 'annual_return', 'annual_volatility', 'max_drawback',
                                'max_drawback_date', 'Sharpe', 'Calmar'])


def eval_loop(dataloader, net, loss_fn):
    running_loss = 0.0
    total_loss = 0.0
    current = 0
    net.eval()
    target = []
    predict = []
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)
                y_pred = net(X)
                target.append(y.detach())
                predict.append(y_pred.detach())
                loss = loss_fn(y_pred, y.long())
                running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
                current += len(X)
                t.set_postfix({'running_loss': running_loss})
                total_loss += running_loss
    total_loss /= len(dataloader)
    return total_loss, torch.cat(predict), torch.cat(target)


two_cate_labels = (label_df.Ret_5d > 0) if predict_days == 5 else (label_df['Ret_20d'] > 0) if predict_days == 20 else (
        label_df.Ret_60d > 0)
labels = label_df.apply(
    lambda x: int(0) if ((x.Ret_20d > 0) and (x.Ret_20d > x.Ret_5d)) else int(1) if ((x.Ret_20d > 0) and (
            x.Ret_20d <= x.Ret_5d)) else int(2), axis=1) if categories == 3 else two_cate_labels  # Three categories

dataset = MyDataset(images, labels.values)
del images
gc.collect()
demonstrate_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

net_paths = [r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241029_115041\categories of 2 "
             r"baseline_epoch_3_train_0.718633_val_0.696167.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_175809\categories of 2 filter_3_3_epoch_5_train_0.701875_val_0.697265.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_190913\categories of 2 filter_7_3_epoch_4_train_0.712956_val_0.688309.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_195555\categories of 2 filters_32_epoch_3_train_0.717983_val_0.691144.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_202101\categories of 2 filters_128_epoch_3_train_0.720437_val_0.688457.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_214543\categories of 2 dropout_0_epoch_3_train_0.686332_val_0.708110.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_222835\categories of 2 dropout_025_epoch_3_train_0.706665_val_0.691760.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_231801\categories of 2 dropout_075_epoch_5_train_0.702405_val_0.692412.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_001651\categories of 2 layers_2_epoch_2_train_0.749750_val_0.683852.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_004204\categories of 2 layers_4_epoch_0_train_0.857160_val_0.698290.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_013434\categories of 2 act_relu_epoch_3_train_0.718570_val_0.695934.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_020621\categories of 2 bn_no_epoch_3_train_0.686493_val_0.687633.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_121205\categories of 2 maxpool_2_2_epoch_6_train_0.700384_val_0.696510.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_130913\categories of 2 ds_11_11_epoch_1_train_0.772438_val_0.703660.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241028_140817\categories of 2 ds_21_11_epoch_4_train_0.717940_val_0.691504.pt",
              r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241029_030251\categories of 2 ds_11_31_epoch_0_train_0.831172_val_0.688820.pt"]

# In[19]:


if __name__ == '__main__':
    # dataloader = iter(demonstrate_dataloader)
    # images, labels = next(dataloader)
    # use_images = images.reshape(batch_size,1,IMAGE_HEIGHT[day],IMAGE_WIDTH[day])
    # # show images
    # imshow(torchvision.utils.make_grid(use_images))  # 将多张图片排列成一个图片
    # for net_path in net_paths[0:]:
    net_path = (
        r"D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main\pt\20241027_172535 60d\categories of 2 baseline_epoch_1_train_0.757752_val_0.672125.pt")
    model_name = re.findall(r"\\([^\\]*?)_epoch", net_path)[0]
    net = torch.load(net_path)

    loss_fn = nn.CrossEntropyLoss()
    test_loss, y_pred, y_target = eval_loop(test_dataloader, net, loss_fn)

    #
    if categories == 3:
        predict_logit = (torch.nn.Softmax(dim=1)(y_pred)).cpu().numpy()
        predict_logit = np.apply_along_axis(func1d=lambda x: np.argmax(x),
                                            arr=predict_logit, axis=1)  # need to change when facing three categories
        accuracy = (predict_logit == y_target.cpu()).sum() / len(predict_logit)
    else:
        predict_logit = (torch.nn.Softmax(dim=1)(y_pred)[:, 1]).cpu().numpy()  # Choose 20ret>0
        accuracy = ((predict_logit >= 0.58) == y_target.cpu()).sum() / len(predict_logit)
    if predict_days == 60:
        date_range = pd.unique(label_df['Date'])
        use_range = date_range[::2]
        label_df.index = [i for i in range(len(label_df))]  # reindex
        label_df = label_df[label_df['Date'].isin(use_range)]
        predict_logit = predict_logit[label_df.index]
    print(f'{model_name}:Loss:{test_loss}, Accuracy:{accuracy}')
    period_ret = f'Ret_{predict_days}d'
    ret_baseline = label_df.groupby(['Date'])[period_ret].apply(lambda x: x.dropna().mean())

    threshold = 0.58
    flag = (predict_logit == 0) if categories == 3 else (predict_logit > threshold)
    # label_df['ret'] = flag * label_df[period_ret]
    label_filtered = label_df[flag]
    ret_cnn = label_filtered.groupby(['Date'])[period_ret].apply(lambda x: x.dropna().mean())

    # plt.scatter(label_filtered.groupby(['Date'])['ret'].count().index, label_filtered.groupby(['Date'])['ret'].count(),
    #             marker='+')

    log_ret_baseline = np.log10((ret_baseline + 1).cumprod().fillna(method='ffill'))  # 10倍
    log_ret_cnn = np.log10((ret_cnn + 1).cumprod().fillna(method='ffill'))
    fig = plt.figure()
    plt.plot(log_ret_baseline, label='baseline')
    plt.plot(log_ret_cnn, label='CNN')
    plt.plot(log_ret_cnn - log_ret_baseline, alpha=0.6, lw=2, label='exceed_ret')
    plt.legend()
    plt.title('Equal weighted', fontsize=16)
    fig.savefig(f'./pic/performance1 {model_name} predict {predict_days} days.png', dpi=300)
    # plt.show(block=True)
    plt.close()
    NV = (ret_cnn-ret_baseline + 1).cumprod()
    shift_days = 90 if predict_days == 60 else 30 if predict_days == 20 else 7
    NV = pd.concat([pd.Series(index=[NV.index[0] - timedelta(shift_days)], data=[1]), NV], axis=0)
    Ev = Evaluation(NV)

    result = Ev.generate_info()
    result['Loss'] = test_loss
    result['Accuracy'] = accuracy.numpy()
    result.to_csv(f'{predict_days}_{model_name}_result.csv')
    (1 + ret_cnn).cumprod().to_csv(f'{predict_days}_{model_name}_NV.csv')
    plt.plot((ret_cnn + 1).cumprod().fillna(method='ffill'), label='CNN_accumulate_ret')
    plt.plot((ret_cnn - ret_baseline + 1).cumprod().fillna(method='ffill'), label='exceed_accumulate_ret')
    plt.legend()
    plt.title('CNN_accumulate_ret', fontsize=16)
    plt.savefig(f'./pic/performance2 {model_name} predict {predict_days} days.png', dpi=300)
    # plt.show(block=True)
    plt.close()
    # ## Weighted by EWMA_Vol

    label_df['weighted_ret'] = 1 * label_df[period_ret] * label_df['EWMA_vol']
    label_df['weight'] = 1 * label_df['EWMA_vol']
    ret_baseline = (label_df.groupby(['Date'])['weighted_ret'].apply(lambda x: x.dropna().sum()) /
                    (label_df.groupby(['Date']).apply(lambda x: x.dropna(subset='weighted_ret')['weight'].sum())))

    threshold = 0.58
    flag = (predict_logit == 0) if categories == 3 else (predict_logit > threshold)
    label_df['weighted_ret'] = flag * label_df[period_ret] * label_df['EWMA_vol']
    label_df['weight'] = flag * label_df['EWMA_vol']
    ret_cnn = (label_df.groupby(['Date'])['weighted_ret'].apply(lambda x: x.dropna().sum()) /
               (label_df.groupby(['Date']).apply(lambda x: x.dropna(subset='weighted_ret')['weight'].sum())))

    log_ret_baseline = np.log10((ret_baseline + 1).cumprod().fillna(method='ffill'))  # 10倍
    log_ret_cnn = np.log10((ret_cnn + 1).cumprod().fillna(method='ffill'))
    plt.plot(log_ret_baseline, label='base_line')
    plt.plot(log_ret_cnn, label='CNN')
    plt.plot(log_ret_cnn - log_ret_baseline, alpha=0.6, lw=2, label='exceed_return')
    plt.legend()
    plt.title('EWMA_vol weighted', fontsize=16)
    plt.savefig(f'./pic/performance3 {model_name} predict {predict_days} days.png', dpi=300)
    # plt.show(block=True)
    plt.close()

    plt.plot((ret_cnn + 1).cumprod().fillna(method='ffill'), label='CNN_accumulate_ret')
    plt.plot((ret_cnn - ret_baseline + 1).cumprod().fillna(method='ffill'), label='exceed_accumulate_ret')
    plt.legend()
    plt.title('CNN_accumulate_ret EWMA_vol weighted', fontsize=16)
    plt.savefig(f'./pic/performance4 {model_name} predict {predict_days} days.png', dpi=300)
    # plt.show(block=True)
    plt.close()
    torch.cuda.empty_cache()
