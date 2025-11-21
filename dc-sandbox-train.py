import os
import logging
import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader

import numpy as np 

from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict

import deepclean as dc 
import deepclean.timeseries as ts 
import deepclean.criterion 
import deepclean.nn as nn 
import deepclean.nn.utils as utils 
import deepclean.nn.net as net
from deepclean.timeseries import TimeSeriesDataset
import deepclean.logger as logger

train_dir = 'train_dir'
os.makedirs(train_dir, exist_ok=True)
log = os.path.join(train_dir, 'log.log')

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log, mode='a'),  # Write to file
        logging.StreamHandler()  # Write to console
    ]
)
logging.info('Create training directory: {}'.format(train_dir))
# For now, use CPU for training: 
device = dc.nn.utils.get_device('mps')
# Using params from deepclean-prod config: 
train_data = dc.timeseries.TimeSeriesSegmentDataset(kernel=8, stride=0.25, pad_mode='median')
val_data = dc.timeseries.TimeSeriesSegmentDataset(kernel=8, stride=0.25, pad_mode='median')

# TODO: Double check that the train/validation split is correct: 
train_data.read('data/processed/combined_data.npz', channels='SelectedChannels_110_130Hz.ini', start_time=1378403243, end_time=1378403243+3072, fs=2048)
val_data.read('data/processed/combined_data.npz', channels='SelectedChannels_110_130Hz.ini', start_time=1378403243, end_time=1378403243+3072+3072, fs=2048)

logging.info('Preprocessing ------------------------- ')
# bandpass filter: 
train_data = train_data.bandpass(110, 130, order=8, channels='target')
val_data = val_data.bandpass(110, 130, order=8, channels='target')

# filter pad default from deepclean-prod, is 5: 
filt_pad = 5 
fs = 2048 # from info.txt 
train_data.data = train_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]
val_data.data = val_data.data[:, int(filt_pad * fs):-int(filt_pad * fs)]

# normalization: 
mean = train_data.mean 
std = train_data.std 
train_data = train_data.normalize()
val_data = val_data.normalize(mean, std)

# read dataset into DataLoader 
batch_size = 32 
num_workers = 0 
train_loader = DataLoader(train_data, batch_size, num_workers)
val_loader = DataLoader(val_data, batch_size, num_workers)

# create model, loss function, optimizer, and lr scheduler
model = dc.nn.net.DeepClean(train_data.n_channels-1)
model = model.to(device)

fft_length = 2
overlap = None
psd_weight= 1.0
mse_weight= 0.0
coh_weight = 0.0 
tf_weight = 0.0 

criterion = dc.criterion.CompositePSDLoss(
    fs, 
    110, 
    130,
    fftlength=fft_length,
    overlap=overlap, 
    psd_weight=psd_weight, 
    mse_weight=mse_weight,
    reduction='sum', 
    device=device, 
    average='mean'
)

lr = 1e-3 
weight_decay = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

# start training: 
train_logger = dc.logger.Logger(outdir=train_dir, metrics=['loss'])
dc.nn.utils.train(
    train_loader, model, criterion, device, optimizer, lr_scheduler,
    val_loader=val_loader, max_epochs=3, logger=train_logger)
# max_epochs = 50