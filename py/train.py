import gc
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from models import baseline, Three_cates,filter_3_3,filter_7_3,filters_32,filters_128,dropout_0,dropout_025,dropout_075,layers_2,layers_4,act_relu,bn_no,maxpool_2_2,ds_11_11,ds_21_11,ds_11_31


import os
import sys
from thop import profile as thop_profile
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

tb = SummaryWriter('C:/Users/12968/Desktop/tensorboard/runs/trend_recognition')
day = 20
categories = 2
predict_days = 20
use_gpu = True
use_ramdon_split = False
use_dataparallel = True
device = 'cuda' if use_gpu else 'cpu'
export_onnx = False
start_epoch = 0
min_val_loss = 1e9
last_min_ind = -1
early_stopping_epoch = 5
sys.path.insert(0, '..')
epochs = 100

if use_gpu:
    from utils.gpu_tools import *

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(obj) for obj in select_gpu(query_gpu())])
torch.manual_seed(42)
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# ## load data
# here we choose 1993-2001 data as our training(include validation) data, the remaining will be used in testing.

year_list = np.arange(1993, 2001, 1)
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


def train_loop(dataloader, net, loss_fn, optimizer, epoch):
    running_loss = 0.0
    current = 0
    net.train()

    with tqdm(dataloader) as t:
        for batch, (X, y) in enumerate(t):
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X)
            loss = loss_fn(y_pred, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
            current += len(X)
            t.set_postfix({'running_loss': running_loss})
            # tb.add_scalars(f'Training Loss' if categories == 2 else 'Training Loss with 3 Categories',
            #                {'Training': running_loss},
            #                batch + len(dataloader) * epoch)

    return running_loss


def val_loop(dataloader, net, loss_fn):
    running_loss = 0.0
    current = 0
    net.eval()

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)
                y_pred = net(X)
                loss = loss_fn(y_pred, y.long())

                running_loss += loss.item()
                running_loss = (len(X) * running_loss + loss.item() * current) / (len(X) + current)
                current += len(X)
    return running_loss


# Split method (not random split is recommended)
two_cate_labels = (label_df.Ret_5d > 0) if predict_days == 5 else (label_df.Ret_20d > 0) if predict_days == 20 else (
        label_df.Ret_60d > 0)
labels = label_df.apply(lambda x: int(0) if ((x.Ret_20d > 0) and (x.Ret_20d > x.Ret_5d)) else int(1) if ((x.Ret_20d > 0) and (
        x.Ret_20d <= x.Ret_5d)) else int(2), axis=1) if categories == 3 else two_cate_labels  # Three categories

if not use_ramdon_split:
    train_val_ratio = 0.7
    split_idx = int(images.shape[0] * 0.7)
    train_dataset = MyDataset(images[:split_idx], labels.values[:split_idx])  # 还是分类问题
    val_dataset = MyDataset(images[split_idx:], labels.values[split_idx:])
else:
    dataset = MyDataset(images, labels.values)
    train_val_ratio = 0.7
    train_dataset, val_dataset = random_split(dataset,
                                              [int(dataset.len * train_val_ratio),
                                               dataset.len - int(dataset.len * train_val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
    del dataset
    gc.collect()

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)
del train_dataset, val_dataset
gc.collect()

# ## models


net = baseline.Net().to(device)
net.apply(init_weights)

if export_onnx:
    import torch.onnx

    x = torch.randn([1, 1, 64, 60]).to(device)
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./cnn_baseline.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=False,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input_images'],  # the model's input names
                      output_names=['output_prob'],  # the model's output names
                      dynamic_axes={'input_images': {0: 'batch_size'},  # variable length axes
                                    'output_prob': {0: 'batch_size'}})

# ## Profiling

count = 0
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())
    count += parameters.numel()
print('total_parameters : {}'.format(count))

flops, params = thop_profile(net, inputs=(next(iter(train_dataloader))[0].to(device),))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')

inputs = next(iter(train_dataloader))[0].to(device)

with profile(activities=[
    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        net(inputs)

prof.export_chrome_trace("./trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# ## train


# net = torch.load('/home/clidg/proj_2/pt/baseline_epoch_10_train_0.6865865240322523_eval_0.686580_.pt')


if use_gpu and use_dataparallel and 'DataParallel' not in str(type(net)):
    net = net.to(device)
    net = nn.DataParallel(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
start_time = ''.join(datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S').split(':'))
os.mkdir('./pt/' + start_time + '/')

if __name__ == '__main__':
    for t in range(start_epoch, epochs):
        print(f"Epoch {t}\n-------------------------------")
        time.sleep(0.2)
        train_loss = train_loop(train_dataloader, net, loss_fn, optimizer, t)
        val_loss = val_loop(val_dataloader, net, loss_fn)
        tb.add_histogram("train_loss", train_loss, t)

        torch.save(net, './pt' + os.sep + start_time + os.sep + 'categories of {} baseline_epoch_{}_train_{:5f}_val_{:5f}.pt'.format(categories,t,
                                                                                                                    train_loss,
                                                                                                                    val_loss))
        if val_loss < min_val_loss:
            last_min_ind = t
            min_val_loss = val_loss
        elif t - last_min_ind >= early_stopping_epoch:
            break

    print('Done!')
    print('Best epoch: {}, val_loss: {}'.format(last_min_ind, min_val_loss))
