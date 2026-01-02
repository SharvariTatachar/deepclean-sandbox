import os
import torch 
import logging
from torch.utils.data import DataLoader
import numpy as np 
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
import deepclean as dc 
import deepclean.timeseries as ts 
from deepclean.timeseries import TimeSeriesDataset, TimeSeriesSegmentDataset
import deepclean.criterion 
import deepclean.nn as nn 
import deepclean.signal as signal
import deepclean.nn.utils as utils 
import deepclean.nn.net as net
import deepclean.logger as logger

# Set default tensor type (use CUDA if available, otherwise CPU)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("Using CUDA")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    print("CUDA not available, using CPU")

# Load data from combined .npz file
data = TimeSeriesDataset()
data.read('data/processed/combined_data.npz', channels='SelectedChannels_110_130Hz.ini')

print(f"Loaded {data.n_channels} channels")
# print(f"Data shape: {data.data.shape}")
print(f"Target channel (strain): {data.channels[data.target_idx]} (index {data.target_idx})")
# print(f"All channels: {list(data.channels)}")

os.makedirs('out_dir', exist_ok=True)
log = os.path.join('out_dir', 'log.log')
logging.basicConfig(filename=log, filemode='a', 
                    format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.info('Create output directory: {}'.format('out_dir'))

# Compute mean and std from the current dataset (before bandpass)
mean = data.mean  # Shape: (n_channels, 1) - mean per channel
std = data.std    # Shape: (n_channels, 1) - std per channel

# Filter parameters
clean_t0 = 1378403243
clean_duration = 3072 
filt_fl = 110
filt_fh = 130
filt_order = 8
batch_size = 32 
num_workers = 0 # TODO: change to 4 
# Apply bandpass filter (typically only on target channel for DeepClean)
preprocessed = data.bandpass(filt_fl, filt_fh, filt_order, channels='target')

# Normalize using the pre-computed mean and std
preprocessed = preprocessed.normalize(mean=mean, std=std)

print("Preprocessing done")

# Convert to TimeSeriesSegmentDataset for DataLoader (needs __len__ and __getitem__)
# Use same kernel and stride as training
kernel = 8  # seconds
stride = 0.25  # seconds
segment_dataset = TimeSeriesSegmentDataset(kernel=kernel, stride=stride, pad_mode='median')
# Copy the preprocessed data into the segment dataset
segment_dataset.data = preprocessed.data
segment_dataset.channels = preprocessed.channels
segment_dataset.t0 = preprocessed.t0
segment_dataset.fs = preprocessed.fs
segment_dataset.target_idx = preprocessed.target_idx

# TODO: add model and post processing here
device = dc.nn.utils.get_device('mps')
data_loader = DataLoader(segment_dataset, batch_size=batch_size, 
num_workers=num_workers, shuffle=False)

model = dc.nn.net.DeepClean(segment_dataset.n_channels-1) 
model = model.to(device)
logging.info(model)

checkpoint = dc.nn.utils.get_last_checkpoint(os.path.join('train_dir', 'models'))
logging.info('loading model from checkpoint: {}'.format(checkpoint))
model.load_state_dict(torch.load(checkpoint, map_location=device))

# Start cleaning
logging.info('Cleaning')
target = data.get_target()

# predict noise from auxiliary channels 
pred_batches = dc.nn.utils.evaluate(data_loader, model, device=device)

# post-processing the prediction 
# overlap-add to join together prediction 
noverlap = int((segment_dataset.kernel - segment_dataset.stride) * segment_dataset.fs)
pred = dc.signal.overlap_add(pred_batches, noverlap, 'hann')
# convert back to unit of target
pred *= std[segment_dataset.target_idx].ravel()
pred += mean[segment_dataset.target_idx].ravel()
# apply bandpass filter
pred = dc.signal.bandpass(pred, segment_dataset.fs, filt_fl, filt_fh, filt_order)
# because of data padding, we apply a cut to the output
pred = pred[:len(target)]

# subtract noise prediction from raw data
clean = target - pred 

out_file = os.path.join('out_dir', 'clean-{}-{}.h5'.format(clean_t0, clean_duration))
logging.info('Writing output to {}'.format(out_file))
series = TimeSeries(clean, t0=data.t0, sample_rate=data.fs, channel='H1:GDS-CALIB_STRAIN_DC', name='H1:GDS-CALIB_STRAIN_DC')
series.write(out_file)