#!/usr/bin/env python3
"""
Preprocessing script for DeepClean data.
Combines individual .npz channel files into a single file for loading.
"""

import numpy as np
import argparse
import os
from pathlib import Path
from timeseries import TimeSeriesDataset

def load_info_file(info_path):
    """Load metadata from info.txt file."""
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    # Try to convert to float/int
                    if '.' in value:
                        info[key] = float(value.strip())
                    else:
                        info[key] = int(value.strip())
                except ValueError:
                    info[key] = value.strip()
    return info

def combine_npz_files(data_dir, channel_file, output_file, info_file=None):
    """
    Combine individual .npz channel files into a single .npz file.
    
    Note: The FIRST channel in the channel_file is assumed to be the TARGET
    (strain) that we want to clean. All other channels are WITNESS channels
    used to clean the strain.
    
    Parameters
    ----------
    data_dir : str
        Directory containing individual .npz files
    channel_file : str
        File containing list of channel names (one per line).
        First channel should be the strain/target channel.
    output_file : str
        Output .npz file path
    info_file : str, optional
        Path to info.txt file with metadata (t0, fs, etc.)
    """
    # Load channel list
    with open(channel_file, 'r') as f:
        channels = [line.strip() for line in f if line.strip()]
    
    if len(channels) == 0:
        raise ValueError("No channels found in channel file!")
    
    target_channel = channels[0]
    witness_channels = channels[1:]
    
    print(f"Target channel (strain): {target_channel}")
    print(f"Witness channels: {len(witness_channels)}")
    
    # Load metadata
    t0 = 0.0
    fs = 2048.0  # Default as mentioned by user
    
    if info_file and os.path.exists(info_file):
        info = load_info_file(info_file)
        t0 = info.get('time', info.get('t0', 0.0))
        fs = info.get('sample_rate', info.get('fs', 2048.0))
    
    # Load data from individual files
    data_dict = {}
    data_dir = Path(data_dir)
    
    print(f"\nLoading channels from {data_dir}...")
    
    # Helper function to load a single channel
    def load_channel(channel_name):
        """Load a single channel from .npz file."""
        # Convert channel name to filename format (replace : with _ and remove special chars)
        filename = channel_name.replace(':', '_').replace('-', '_') + '.npz'
        filepath = data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found for channel {channel_name}: {filepath}")
        
        with np.load(filepath, allow_pickle=True) as f:
            # Get the data array (could be under different keys)
            keys = list(f.keys())
            if len(keys) == 1:
                # Single key, use it
                data = f[keys[0]]
            elif 'data' in keys:
                data = f['data']
            elif channel_name in keys:
                data = f[channel_name]
            else:
                # Try to find a numeric array
                for key in keys:
                    if isinstance(f[key], np.ndarray):
                        data = f[key]
                        break
                else:
                    raise ValueError(f"Could not find data array in {filepath}. Available keys: {keys}")
        return data
    
    # Load target channel first
    print(f"\nLoading target channel: {target_channel}")
    try:
        data = load_channel(target_channel)
        data_dict[target_channel] = data
        print(f"  ✓ Loaded {target_channel}: shape {data.shape}")
    except Exception as e:
        print(f"  ✗ Error loading target channel {target_channel}: {e}")
        raise ValueError(f"Failed to load target channel {target_channel}!")
    
    # Load witness channels
    print(f"\nLoading {len(witness_channels)} witness channels...")
    for channel in witness_channels:
        try:
            data = load_channel(channel)
            data_dict[channel] = data
            print(f"  ✓ Loaded {channel}: shape {data.shape}")
        except Exception as e:
            print(f"  ✗ Error loading {channel}: {e}")
            continue
    
    # Check that all arrays have the same length
    lengths = [len(data) for data in data_dict.values()]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"Warning: Channels have different lengths. Truncating to minimum length: {min_len}")
        data_dict = {k: v[:min_len] for k, v in data_dict.items()}
    
    # Save combined file (preserve order: target first, then witnesses)
    print(f"\nSaving combined data to {output_file}...")
    save_dict = {}
    
    # Save target channel first
    save_dict[target_channel] = data_dict[target_channel]
    
    # Then save witness channels in order
    for channel in witness_channels:
        if channel in data_dict:
            save_dict[channel] = data_dict[channel]
    
    # Add metadata
    save_dict['t0'] = np.array(t0)
    save_dict['fs'] = np.array(fs)
    save_dict['sample_rate'] = np.array(fs)  # Also save as sample_rate for compatibility
    
    np.savez_compressed(output_file, **save_dict)
    print(f"✓ Saved {len(data_dict)} channels to {output_file}")
    print(f"  Metadata: t0={t0}, fs={fs} Hz")
    print(f"  Target: {target_channel}, Witnesses: {len(witness_channels)}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess DeepClean data by combining individual .npz files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing individual .npz channel files (default: data/processed)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        default='SelectedChannels_110_130Hz.ini',
        help='File containing channel list (default: SelectedChannels_110_130Hz.ini)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/combined_data.npz',
        help='Output combined .npz file (default: data/processed/combined_data.npz)'
    )
    parser.add_argument(
        '--info',
        type=str,
        default='info.txt',
        help='Info file with metadata (default: info.txt)'
    )
    parser.add_argument(
        '--test-load',
        action='store_true',
        help='Test loading the combined file after creation'
    )
    
    args = parser.parse_args()
    
    # Combine files
    output_file = combine_npz_files(
        args.data_dir,
        args.channels,
        args.output,
        args.info
    )
    
    # Test loading if requested
    if args.test_load:
        print("\nTesting load of combined file...")
        dataset = TimeSeriesDataset()
        with open(args.channels, 'r') as f:
            channels = [line.strip() for line in f if line.strip()]
        
        dataset.read(output_file, channels=channels)
        print(f"✓ Successfully loaded {dataset.n_channels} channels")
        print(f"  Data shape: {dataset.data.shape} (channels x samples)")
        print(f"  Sample rate: {dataset.fs} Hz")
        print(f"  Start time: {dataset.t0}")
        print(f"  Target channel: {dataset.channels[dataset.target_idx]} (index {dataset.target_idx})")
        print(f"  Witness channels: {dataset.n_channels - 1}")


if __name__ == '__main__':
    main()

