#!/usr/bin/env python3
"""
Preprocessing script for DeepClean data.
Combines individual .npz channel files into a single file for loading.
Adds optional scaling for the target (strain) channel to improve numerical stability.
"""

import numpy as np
import argparse
import os
from pathlib import Path
from deepclean.timeseries import TimeSeriesDataset

def load_info_file(info_path):
    """Load metadata from info.txt file."""
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    if '.' in value:
                        info[key] = float(value.strip())
                    else:
                        info[key] = int(value.strip())
                except ValueError:
                    info[key] = value.strip()
    return info

def combine_npz_files(
    data_dir,
    channel_file,
    output_file,
    info_file=None,
    target_scale=1.0,
    auto_target_scale=False,
    eps=1e-12,
    cast_float32=False,
):
    """
    Combine individual .npz channel files into a single .npz file,
    with optional scaling of the target channel.

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
    target_scale : float
        Multiply the target channel by this factor (applied after loading).
    auto_target_scale : bool
        If True, ignore target_scale and instead scale target so its std ≈ 1.
    eps : float
        Epsilon to avoid division by zero in auto scaling.
    cast_float32 : bool
        If True, cast all arrays to float32 before saving.
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
    fs = 2048.0
    if info_file and os.path.exists(info_file):
        info = load_info_file(info_file)
        t0 = info.get('time', info.get('t0', 0.0))
        fs = info.get('sample_rate', info.get('fs', 2048.0))

    # Helper to load one channel
    data_dir = Path(data_dir)
    def load_channel(channel_name):
        filename = channel_name.replace(':', '_').replace('-', '_') + '.npz'
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found for channel {channel_name}: {filepath}")
        with np.load(filepath, allow_pickle=True) as f:
            keys = list(f.keys())
            if len(keys) == 1:
                data = f[keys[0]]
            elif 'data' in keys:
                data = f['data']
            elif channel_name in keys:
                data = f[channel_name]
            else:
                data = None
                for key in keys:
                    if isinstance(f[key], np.ndarray):
                        data = f[key]
                        break
                if data is None:
                    raise ValueError(f"Could not find data array in {filepath}. Available keys: {keys}")
        return data

    # Load target channel
    print(f"\nLoading target channel: {target_channel}")
    try:
        target = load_channel(target_channel).astype(np.float64, copy=False)
        print(f"  ✓ Loaded {target_channel}: shape {target.shape}")
    except Exception as e:
        print(f"  ✗ Error loading target channel {target_channel}: {e}")
        raise ValueError(f"Failed to load target channel {target_channel}!")

    # Optional auto scaling of target to unit std
    applied_target_scale = float(target_scale)
    if auto_target_scale:
        std = float(np.std(target))
        # If std is extremely tiny (typical for LIGO strain ~1e-22 to 1e-18),
        # compute a scale that brings std to ~1
        applied_target_scale = 1.0 / max(std, eps)
        print(f"  ⤴ Auto target scale enabled. Original std={std:.3e} -> scale={applied_target_scale:.3e}")

    # Apply target scaling if not 1.0
    if applied_target_scale != 1.0:
        target = target * applied_target_scale
        new_std = float(np.std(target))
        print(f"  ✓ Applied target scale = {applied_target_scale:.6g}. New std ≈ {new_std:.3e}")
    else:
        print("  ↦ No target scaling applied.")

    # Load witness channels
    print(f"\nLoading {len(witness_channels)} witness channels...")
    data_dict = {target_channel: target}
    for ch in witness_channels:
        try:
            arr = load_channel(ch).astype(np.float64, copy=False)
            data_dict[ch] = arr
            print(f"  ✓ Loaded {ch}: shape {arr.shape}")
        except Exception as e:
            print(f"  ✗ Error loading {ch}: {e}")
            continue

    # Ensure same length
    lengths = [len(arr) for arr in data_dict.values()]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"Warning: Channels have different lengths. Truncating to minimum length: {min_len}")
        data_dict = {k: v[:min_len] for k, v in data_dict.items()}

    # Optional cast to float32 to reduce size / improve IO
    if cast_float32:
        for k in list(data_dict.keys()):
            data_dict[k] = data_dict[k].astype(np.float32)
        dtype_note = "float32"
    else:
        dtype_note = "float64"

    # Save combined file
    print(f"\nSaving combined data to {output_file} (dtype={dtype_note})...")
    save_dict = {}

    # Target first, then witnesses
    save_dict[target_channel] = data_dict[target_channel]
    for ch in witness_channels:
        if ch in data_dict:
            save_dict[ch] = data_dict[ch]

    # Metadata
    save_dict['t0'] = np.array(t0, dtype=np.float64)
    save_dict['fs'] = np.array(fs, dtype=np.float64)
    save_dict['sample_rate'] = np.array(fs, dtype=np.float64)
    save_dict['target_scale'] = np.array(applied_target_scale, dtype=np.float64)

    np.savez_compressed(output_file, **save_dict)
    print(f"✓ Saved {len(data_dict)} channels to {output_file}")
    print(f"  Metadata: t0={t0}, fs={fs} Hz, target_scale={applied_target_scale}")
    print(f"  Target: {target_channel}, Witnesses: {len(witness_channels)}")
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess DeepClean data by combining individual .npz files'
    )
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing individual .npz channel files (default: data/processed)')
    parser.add_argument('--channels', type=str, default='SelectedChannels_110_130Hz.ini',
                        help='File containing channel list (default: SelectedChannels_110_130Hz.ini)')
    parser.add_argument('--output', type=str, default='data/processed/combined_data.npz',
                        help='Output combined .npz file (default: data/processed/combined_data.npz)')
    parser.add_argument('--info', type=str, default='info.txt',
                        help='Info file with metadata (default: info.txt)')

    # NEW: scaling controls
    parser.add_argument('--target-scale', type=float, default=1.0,
                        help='Multiply the target (strain) by this factor (default: 1.0). '
                             'Use e.g. 1e18 to bring LIGO strain to O(1).')
    parser.add_argument('--auto-target-scale', action='store_true',
                        help='If set, ignore --target-scale and scale target so its std ≈ 1.')
    parser.add_argument('--eps', type=float, default=1e-12,
                        help='Epsilon for auto scaling to avoid division by zero (default: 1e-12).')
    parser.add_argument('--float32', action='store_true',
                        help='Cast arrays to float32 before saving (default: False).')

    parser.add_argument('--test-load', action='store_true',
                        help='Test loading the combined file after creation')

    args = parser.parse_args()

    output_file = combine_npz_files(
        args.data_dir,
        args.channels,
        args.output,
        args.info,
        target_scale=args.target_scale,
        auto_target_scale=args.auto_target_scale,
        eps=args.eps,
        cast_float32=args.float32,
    )

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
        print(f"  Target scale in file: {getattr(dataset, 'target_scale', 'N/A')}")

if __name__ == '__main__':
    main()

# Usage: 
# python3 combinechan.py --data-dir data/processed \
#   --channels SelectedChannels_110_130Hz.ini \
#   --output data/processed/combined_scaled.npz \
#   --target-scale 1e18