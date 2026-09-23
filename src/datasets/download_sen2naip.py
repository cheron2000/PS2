import os
import numpy as np
import argparse
from datasets import load_dataset

def download_and_convert_sen2naip(subset_size=100, out_dir="data/sen2naip"):
    print(f"Loading SEN2NAIPv2 dataset from HuggingFace...")
    # Load dataset (streaming or downloading)
    # The dataset typically has 'image' or similar keys. 
    # Let's inspect it first
    dataset = load_dataset("tacofoundation/SEN2NAIPv2", split="train", streaming=True)
    
    lr_dir = os.path.join(out_dir, "lr")
    hr_dir = os.path.join(out_dir, "hr")
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    
    print(f"Extracting up to {subset_size} samples to {out_dir}...")
    
    count = 0
    for idx, item in enumerate(dataset):
        if count >= subset_size:
            break
            
        # Figure out the exact keys
        # According to TACO / SEN2NAIP specs, it might have "Sentinel2" and "NAIP" or similar.
        # We will attempt to find the right keys.
        if idx == 0:
            print("Dataset keys:", item.keys())
            
        # Common key names for SEN2NAIP
        s2_key = next((k for k in item.keys() if "s2" in k.lower() or "sentinel" in k.lower() or "lr" in k.lower()), None)
        naip_key = next((k for k in item.keys() if "naip" in k.lower() or "hr" in k.lower()), None)
        
        if s2_key is None or naip_key is None:
            raise ValueError(f"Could not automatically detect S2 and NAIP keys. Found keys: {item.keys()}")
            
        s2_data = item[s2_key]
        naip_data = item[naip_key]
        
        # Convert to numpy and ensure (C, H, W)
        if hasattr(s2_data, 'mode'):
            # It's a PIL image
            lr_arr = np.array(s2_data)
        else:
            lr_arr = np.array(s2_data)
            
        if hasattr(naip_data, 'mode'):
            hr_arr = np.array(naip_data)
        else:
            hr_arr = np.array(naip_data)
            
        # Often image arrays come as (H, W, C). We need (C, H, W)
        if lr_arr.ndim == 3 and lr_arr.shape[-1] <= 4:
            lr_arr = lr_arr.transpose(2, 0, 1)
        if hr_arr.ndim == 3 and hr_arr.shape[-1] <= 4:
            hr_arr = hr_arr.transpose(2, 0, 1)
            
        file_id = f"sample_{idx:05d}"
        
        np.save(os.path.join(lr_dir, f"{file_id}.npy"), lr_arr)
        np.save(os.path.join(hr_dir, f"{file_id}.npy"), hr_arr)
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} / {subset_size} samples...")
            
    print(f"Successfully extracted {count} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=20, help="Number of samples to download")
    parser.add_argument("--out", type=str, default="data/sen2naip", help="Output directory")
    args = parser.parse_args()
    
    download_and_convert_sen2naip(args.subset, args.out)
