import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def visualize():
    lr_dir = r"f:\Shreyash DOC\shreyash\PS2\data\sen2naip\lr"
    hr_dir = r"f:\Shreyash DOC\shreyash\PS2\data\sen2naip\hr"
    
    lr_files = sorted(glob(os.path.join(lr_dir, "*.npy")))
    hr_files = sorted(glob(os.path.join(hr_dir, "*.npy")))
    
    if not lr_files:
        print("No files found!")
        return
        
    lr_path = lr_files[0]
    hr_path = hr_files[0]
    
    print(f"Loading {lr_path}")
    lr_arr = np.load(lr_path)
    hr_arr = np.load(hr_path)
    
    print(f"LR shape: {lr_arr.shape}, min: {lr_arr.min()}, max: {lr_arr.max()}")
    print(f"HR shape: {hr_arr.shape}, min: {hr_arr.min()}, max: {hr_arr.max()}")
    
    # Extract RGB (assuming 4-band RGB-NIR or BGR-NIR)
    # Usually bands are R, G, B, NIR for NAIP, and B02, B03, B04, B08 for S2 (which is B, G, R).
    # Let's just try first 3 bands. If it's Sentinel-2, it might be BGR. 
    # Let's write a robust visualizer that handles potential 0-10000 or 0-255
    
    def to_rgb(arr):
        # input (C, H, W) -> output (H, W, 3)
        if arr.shape[0] >= 3:
            img = arr[:3, :, :].transpose(1, 2, 0)
        else:
            img = arr[0, :, :][:, :, None].repeat(3, axis=-1)
            
        # normalize to 0-1 for plotting
        if img.max() > 255:
            # likely 0-10000
            img = np.clip(img / 3000.0, 0, 1) # simple bright clipping
        elif img.max() > 1:
            img = img / 255.0
            
        return img
        
    lr_img = to_rgb(lr_arr)
    hr_img = to_rgb(hr_arr)
    
    # reverse channels if S2 is BGR
    # We will just plot as is for now.
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(lr_img)
    axes[0].set_title(f"LR Image\nShape: {lr_arr.shape}")
    axes[0].axis('off')
    
    axes[1].imshow(hr_img)
    axes[1].set_title(f"HR Image\nShape: {hr_arr.shape}")
    axes[1].axis('off')
    
    out_path = r"f:\Shreyash DOC\shreyash\PS2\results\sample_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    visualize()
