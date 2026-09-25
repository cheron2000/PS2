import os
import glob
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from src.model import SRModel
from src.infer import predict

st.set_page_config(page_title="Uncertainty-Aware Sentinel-2 SR", layout="wide", page_icon="🌍")

# -----------------
# Helper Functions
# -----------------
@st.cache_resource
def get_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        return None, None, device
    try:
        # Load checkpoint directly — weights_only=False is safe here because
        # we just trained this file locally (runs/demo/best.pt).
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state" in payload:
            state = payload["model_state"]
            config = dict(payload.get("model_config") or {})
        elif isinstance(payload, dict):
            state = payload
            config = {}
        else:
            raise ValueError("checkpoint must be a dict")

        allowed = {"in_channels", "out_channels", "base_channels",
                    "num_attn_blocks", "window_size", "num_heads", "scale"}
        config = {k: v for k, v in config.items() if k in allowed}
        model = SRModel(**config)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, config, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, device

def to_rgb(arr, low_pct=2, high_pct=98):
    """Convert (C, H, W) float array to (H, W, 3) RGB image for display.

    Uses per-channel percentile stretch so the output always spans [0, 1]
    regardless of the input value range (reflectance, DN, or model output).
    """
    # Select first 3 bands; if only 1 band, replicate to 3
    if arr.shape[0] >= 3:
        img = arr[:3].transpose(1, 2, 0).astype(np.float32)
        # Sentinel-2 bands are stored B, G, R — reverse to R, G, B for display
        img = img[:, :, ::-1].copy()
    else:
        img = np.stack([arr[0]] * 3, axis=-1).astype(np.float32)

    # Per-channel percentile stretch for proper visualization
    out = np.zeros_like(img)
    for c in range(3):
        lo = np.percentile(img[:, :, c], low_pct)
        hi = np.percentile(img[:, :, c], high_pct)
        if hi > lo:
            out[:, :, c] = (img[:, :, c] - lo) / (hi - lo)
        else:
            out[:, :, c] = 0.0
    return np.clip(out, 0, 1)

def visualize_uncertainty(variance_arr, low_pct=2, high_pct=98):
    """Convert (C, H, W) variance to a 2D heatmap using percentile stretch.

    The raw variance is highly skewed (a few high-variance outlier pixels pull
    the max far above the bulk), so min/max normalization makes 99% of the map
    appear black. Percentile stretch fixes this — same approach as to_rgb().
    """
    # Average variance across channels and work in log space so the
    # skewed distribution compresses into something visually useful.
    var_2d = variance_arr.mean(axis=0).astype(np.float64)

    # Log-transform to compress the long tail (add small epsilon to avoid log(0))
    log_var = np.log1p(var_2d)

    # Percentile stretch on the log-transformed values
    lo = np.percentile(log_var, low_pct)
    hi = np.percentile(log_var, high_pct)
    if hi > lo:
        var_norm = np.clip((log_var - lo) / (hi - lo), 0.0, 1.0)
    else:
        var_norm = np.zeros_like(log_var)

    # Apply magma colormap
    cmap = plt.get_cmap('magma')
    heatmap = cmap(var_norm)[..., :3]  # RGBA -> RGB, float64 in [0,1]
    return heatmap

# -----------------
# Main UI
# -----------------
def main():
    st.title("🛰️ Sentinel-2 Super-Resolution Framework")
    st.markdown("""
    This prototype demonstrates the **Uncertainty-Aware Super-Resolution** model (10m → 2.5m). 
    It enhances low-resolution Sentinel-2 imagery while providing a per-pixel uncertainty map, fulfilling the SIH26142 requirements.
    """)

    # Sidebar
    st.sidebar.header("Configuration")

    # Checkpoint configuration
    checkpoint_path = "runs/fixed/best.pt"
    st.sidebar.markdown(f"**Model Checkpoint:** `{checkpoint_path}`")

    model, config, device = get_model(checkpoint_path)
    if model is None:
        st.sidebar.error("Model checkpoint not found. Is it still training?")
    else:
        st.sidebar.success(f"Model loaded successfully! (Device: {device})")

    # Dataset selection
    data_dir = "data/sen2naip"
    lr_files = sorted(glob.glob(os.path.join(data_dir, "lr", "*.npy")))

    if not lr_files:
        st.error(f"No low-resolution image files found in {data_dir}/lr.")
        return

    # Extract clean names for the dropdown
    sample_names = [os.path.basename(f).replace('.npy', '') for f in lr_files]
    selected_sample = st.sidebar.selectbox("Select an Image Sample", sample_names)

    if selected_sample:
        lr_path = os.path.join(data_dir, "lr", f"{selected_sample}.npy")
        hr_path = os.path.join(data_dir, "hr", f"{selected_sample}.npy")

        lr_arr = np.load(lr_path)
        hr_arr = np.load(hr_path) if os.path.exists(hr_path) else None

        st.subheader("Data Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Low-Resolution Input (Sentinel-2, 10m)**\n\nShape: `{lr_arr.shape}`")
            st.image(to_rgb(lr_arr), use_container_width=True)

        with col2:
            if hr_arr is not None:
                st.markdown(f"**High-Resolution Ground Truth (NAIP, 2.5m)**\n\nShape: `{hr_arr.shape}`")
                st.image(to_rgb(hr_arr), use_container_width=True)
            else:
                st.warning("No High-Resolution ground truth found for this sample.")

        # Inference Section
        st.divider()
        st.subheader("Run Model Inference")

        if st.button("🚀 Run Super-Resolution", type="primary"):
            if model is None:
                st.error("Cannot run inference without a loaded model.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Ensure input is float32 and normalized to [0, 1] range for the model
                        input_data = np.clip(lr_arr.astype(np.float32) / 10000.0, 0.0, 1.0)

                        # Run inference
                        sr_mean, sr_variance = predict(model, input_data, device)

                        st.success("Inference Complete!")

                        res_col1, res_col2 = st.columns(2)

                        with res_col1:
                            st.markdown("**Super-Resolved Output (Model)**")
                            st.image(to_rgb(sr_mean), use_container_width=True)
                            st.caption(f"Display: percentile-stretched (p2–p98). Raw range: [{sr_mean.min():.3f}, {sr_mean.max():.3f}]")

                        with res_col2:
                            st.markdown("**Uncertainty Map (Variance)**")
                            st.image(visualize_uncertainty(sr_variance), use_container_width=True)
                            st.caption(f"Brighter = higher uncertainty. Display: log-scale p2–p98 stretch. Variance range: [{sr_variance.min():.4f}, {sr_variance.max():.4f}]")

                        with st.expander("Debug: Array statistics"):
                            st.text(f"SR mean shape: {sr_mean.shape}, min: {sr_mean.min():.4f}, max: {sr_mean.max():.4f}, mean: {sr_mean.mean():.4f}")
                            st.text(f"SR variance shape: {sr_variance.shape}, min: {sr_variance.min():.6f}, max: {sr_variance.max():.6f}")

                    except Exception as e:
                        st.error(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
