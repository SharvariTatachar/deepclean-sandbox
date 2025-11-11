import numpy as np
import sys
import os

def checkchans(npz_path):
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    strain = data['H1:GDS-CALIB_STRAIN']
    print(strain[:10]) 
    # print(f"✅ Loaded {npz_path}")
    # print(f"Contains {len(data.files)} arrays: {data.files}\n")

    # for key in data.files:
    #     arr = data[key]
    #     if not isinstance(arr, np.ndarray):
    #         print(f"⚠️  Skipping {key}: not a numpy array")
    #         continue

    #     # Basic stats
    #     num_nans = np.isnan(arr).sum()
    #     num_infs = np.isinf(arr).sum()
    #     mean = np.nanmean(arr)
    #     std = np.nanstd(arr)
    #     min_val = np.nanmin(arr)
    #     max_val = np.nanmax(arr)

    #     print(f"📈 Channel: {key}")
    #     print(f"   Shape: {arr.shape}")
    #     print(f"   Mean: {mean:.6f}, Std: {std:.6f}")
    #     print(f"   Min: {min_val:.6f}, Max: {max_val:.6f}")
    #     print(f"   NaNs: {num_nans}, Infs: {num_infs}")
    #     print("-" * 50)

    data.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python checkchans.py <path_to_npz>")
        sys.exit(1)
    checkchans(sys.argv[1])