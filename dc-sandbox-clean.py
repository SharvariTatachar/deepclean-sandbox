import torch 
from torch.utils.data import DataLoader

from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from timeseries import TimeSeriesDataset

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

# Compute mean and std from the current dataset (before bandpass)
mean = data.mean  # Shape: (n_channels, 1) - mean per channel
std = data.std    # Shape: (n_channels, 1) - std per channel

# Filter parameters
filt_fl = 110
filt_fh = 130
filt_order = 8

# Apply bandpass filter (typically only on target channel for DeepClean)
preprocessed = data.bandpass(filt_fl, filt_fh, filt_order, channels='target')

# Normalize using the pre-computed mean and std
preprocessed = preprocessed.normalize(mean=mean, std=std)

print("Preprocessing done")
