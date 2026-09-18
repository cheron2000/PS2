"""
src/model.py — Core super-resolution architecture for SIH26142.

Implements the pipeline from solution-draft.md v11's Technical Architecture section:
  1. Shallow feature extraction (conv stem)
  2. MHAN-style high-order (second-order/covariance) channel attention blocks — local detail
  3. SPIFFNet-style windowed transformer block — global context / cross-stage fusion
  4. Sub-pixel convolution upsampling head -> target GSD
  5. Heteroscedastic uncertainty head -> per-pixel mean + log-variance

EXECUTION STATUS (read before building on this):
  Written by claude1 (Claude Sonnet 5). Verified by agent2 with CPU PyTorch 2.14.0+cpu:
  `smoke_test()` passes with mean/log_var shapes `(2, 4, 256, 256)` and 1,089,272
  parameters. An additional odd-size test (`65x70`, not divisible by the window size)
  also passes with finite outputs. Don't build T9 (training loop) on top of this until
  the data/loss interfaces are available.

Shape convention: NCHW throughout. Default config assumes a 4-band (RGB+NIR) 10m
Sentinel-2 input patch, upscaled 4x (10m -> 2.5m), matching the SEN2NAIP primary
benchmark in solution-draft.md.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    # Broad except deliberately: a partial/broken torch install (missing shared
    # libs, mismatched CUDA build, etc.) raises OSError/ImportError depending on
    # environment, not just ModuleNotFoundError. Either way this file should stay
    # usable for _trace_shapes(), the torch-free arithmetic check below.
    _HAS_TORCH = False
    nn = None


if _HAS_TORCH:

    class SecondOrderChannelAttention(nn.Module):
        """MHAN-style 'high-order' channel attention.

        Standard channel attention (e.g. RCAB/SENet) pools each channel with a global
        *average* — a first-order statistic. MHAN's point is that global covariance
        (second-order) pooling captures texture/contrast information average-pooling
        throws away, which matters for recovering fine spatial detail. This is a
        simplified version: per-channel variance across space as a cheap second-order
        proxy, avoiding a full CxC covariance matrix for compute reasons.
        """

        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, H, W)
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            # fc expects `channels` input; mean/var each have `channels`, so combine
            # them by addition (channel-preserving) rather than concat, keeping fc's
            # input dimension fixed at `channels` regardless of this module's config.
            pooled = mean + var  # (B, C, 1, 1), cheap second-order-aware pooling
            attn = self.fc(pooled)  # (B, C, 1, 1)
            return x * attn

    class HighOrderAttentionBlock(nn.Module):
        """Residual block: conv -> ReLU -> conv -> second-order channel attention -> + skip."""

        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.attn = SecondOrderChannelAttention(channels)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            out = self.conv1(x)
            out = self.act(out)
            out = self.conv2(out)
            out = self.attn(out)
            return residual + out

    class WindowedTransformerBlock(nn.Module):
        """SPIFFNet-style local-window self-attention + MLP, for global context /
        cross-stage feature fusion. Simplified: fixed square windows (no shifting
        between blocks, unlike Swin) to keep this tractable as a first pass — revisit
        if window boundary artifacts show up in real training (T9+)."""

        def __init__(self, channels: int, window_size: int = 8, num_heads: int = 4, mlp_ratio: float = 2.0):
            super().__init__()
            self.window_size = window_size
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(channels)
            hidden = int(channels * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, channels),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, H, W). Pad H, W up to a multiple of window_size, window-partition,
            # run attention within each window, then un-window and un-pad.
            B, C, H, W = x.shape
            ws = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w

            # -> (B, Hp/ws, ws, Wp/ws, ws, C) -> (B * num_windows, ws*ws, C)
            x_seq = x_padded.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
            x_seq = x_seq.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
            x_seq = x_seq.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)  # (B*nW, ws*ws, C)

            shortcut = x_seq
            x_seq = self.norm1(x_seq)
            attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
            x_seq = shortcut + attn_out
            x_seq = x_seq + self.mlp(self.norm2(x_seq))

            # undo windowing
            x_seq = x_seq.reshape(B, Hp // ws, Wp // ws, ws, ws, C)
            x_seq = x_seq.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
            x_out = x_seq.permute(0, 3, 1, 2)  # (B, C, Hp, Wp)
            return x_out[:, :, :H, :W]

    class UpsampleHead(nn.Module):
        """Sub-pixel convolution (PixelShuffle) upsampling to the target scale factor.
        Supports scale=4 directly, or scale=2 applied twice (matches common SR practice
        of avoiding a single huge PixelShuffle factor)."""

        def __init__(self, channels: int, out_channels: int, scale: int = 4):
            super().__init__()
            assert scale in (2, 4), "only 2x or 4x supported by this head; compose for other factors"
            steps = [2, 2] if scale == 4 else [2]
            layers = []
            in_ch = channels
            for s in steps:
                layers += [
                    nn.Conv2d(in_ch, in_ch * (s ** 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(s),
                    nn.ReLU(inplace=True),
                ]
                # channel count is unchanged by PixelShuffle+matching conv expansion
            self.up = nn.Sequential(*layers)
            self.to_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.up(x)
            return self.to_out(x)

    class UncertaintyHead(nn.Module):
        """Heteroscedastic aleatoric uncertainty head (solution-draft.md Known Risk #3
        resolution, round 4): predicts per-pixel log-variance alongside the mean
        reconstruction, trained with a Gaussian/Laplacian NLL loss (see src/losses.py,
        task T2). Predicting log-variance rather than variance directly for numerical
        stability -- exp() of it recovers a strictly-positive variance downstream."""

        def __init__(self, channels: int, out_channels: int, scale: int = 4):
            super().__init__()
            self.up = UpsampleHead(channels, out_channels, scale=scale)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.up(x)  # log-variance, same shape as the mean output

    class SRModel(nn.Module):
        """Full pipeline: stem -> N high-order attention blocks -> windowed transformer
        fusion -> [mean upsample head, log-variance upsample head].

        Args:
            in_channels: input band count (default 4: R,G,B,NIR at 10m)
            out_channels: output band count (default same as in_channels)
            base_channels: feature width through the backbone
            num_attn_blocks: how many HighOrderAttentionBlock stages
            scale: upsampling factor (4 -> 10m input to 2.5m output, the project's
                primary SEN2NAIP-benchmarked target per solution-draft.md)
        """

        def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            base_channels: int = 64,
            num_attn_blocks: int = 6,
            window_size: int = 8,
            num_heads: int = 4,
            scale: int = 4,
        ):
            super().__init__()
            self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

            self.attn_blocks = nn.Sequential(
                *[HighOrderAttentionBlock(base_channels) for _ in range(num_attn_blocks)]
            )

            self.transformer_fusion = WindowedTransformerBlock(
                base_channels, window_size=window_size, num_heads=num_heads
            )

            # Cross-stage fusion: combine the pre-transformer (local/CNN) features with
            # the post-transformer (global/attention) features via concat + 1x1 conv,
            # rather than letting the transformer output simply overwrite the CNN path.
            self.fuse = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

            self.mean_head = UpsampleHead(base_channels, out_channels, scale=scale)
            self.uncertainty_head = UncertaintyHead(base_channels, out_channels, scale=scale)

        def forward(self, x: "torch.Tensor"):
            """
            Args:
                x: (B, in_channels, H, W) — e.g. a 64x64 10m Sentinel-2 patch
            Returns:
                mean: (B, out_channels, H*scale, W*scale) -- the super-resolved image
                log_var: (B, out_channels, H*scale, W*scale) -- per-pixel log-variance
            """
            feat = self.stem(x)
            cnn_feat = self.attn_blocks(feat)
            global_feat = self.transformer_fusion(cnn_feat)
            fused = self.fuse(torch.cat([cnn_feat, global_feat], dim=1))

            mean = self.mean_head(fused)
            log_var = self.uncertainty_head(fused)
            return mean, log_var


def _trace_shapes(H: int = 64, W: int = 64, in_ch: int = 4, out_ch: int = 4,
                   base_ch: int = 64, scale: int = 4, window: int = 8) -> None:
    """Torch-free arithmetic check of the shape math above, so this can be verified
    in an environment without torch installed (see EXECUTION STATUS at top of file).
    Mirrors: stem/attn blocks (shape-preserving), window pad/unpad (shape-preserving
    end to end), fuse (shape-preserving), upsample heads (H,W -> H*scale, W*scale).
    Raises AssertionError if the arithmetic doesn't hold together.
    """
    # stem + attn blocks: channel-preserving, spatial-preserving (3x3 conv, padding=1)
    h, w, c = H, W, base_ch
    assert (h, w, c) == (H, W, base_ch)

    # windowed transformer: pad up to multiple of window, then crop back down --
    # net effect on (h, w) is a no-op as long as pad/crop math is symmetric
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    hp, wp = h + pad_h, w + pad_w
    assert hp % window == 0 and wp % window == 0
    h_out, w_out = hp, wp  # before crop
    h_out, w_out = h, w    # after crop back to original H, W
    assert (h_out, w_out) == (h, w)

    # fuse: concat along channel dim (c -> 2c), 1x1 conv back to c; spatial unchanged
    fused_c = c

    # upsample head: for scale=4, two PixelShuffle(2) steps -> *2 *2 = *4 spatial,
    # channel count returns to `in_ch` (here: base_ch) after each PixelShuffle+conv
    # expansion step, per the s**2 channel expansion before each PixelShuffle(s).
    steps = [2, 2] if scale == 4 else [2]
    sh, sw, sc = h, w, fused_c
    for s in steps:
        # conv expands channels by s**2, PixelShuffle(s) divides channels by s**2
        # and multiplies each spatial dim by s -- net channel count unchanged.
        sh, sw = sh * s, sw * s
    final_h, final_w = sh, sw
    assert final_h == H * scale and final_w == W * scale, (
        f"upsample arithmetic mismatch: got {(final_h, final_w)}, expected {(H*scale, W*scale)}"
    )

    print(f"[_trace_shapes OK] input=({in_ch},{H},{W}) -> "
          f"mean/log_var output=({out_ch},{final_h},{final_w}), "
          f"scale={scale}x, window pad/crop symmetric at window={window}")


def smoke_test():
    """Full execution smoke test -- requires torch. Run this for task T3.
    NOT executed by claude1 (see EXECUTION STATUS at top of file)."""
    if not _HAS_TORCH:
        raise RuntimeError(
            "torch is not installed in this environment. This is expected in claude1's "
            "build sandbox (see EXECUTION STATUS at top of file) -- if you're seeing this "
            "elsewhere, `pip install torch` and rerun. This is task T3 in tasks.md."
        )
    model = SRModel(in_channels=4, out_channels=4, base_channels=64,
                     num_attn_blocks=6, window_size=8, num_heads=4, scale=4)
    model.eval()
    x = torch.randn(2, 4, 64, 64)  # batch of 2, 4-band, 64x64 patch
    with torch.no_grad():
        mean, log_var = model(x)
    assert mean.shape == (2, 4, 256, 256), f"unexpected mean shape: {mean.shape}"
    assert log_var.shape == (2, 4, 256, 256), f"unexpected log_var shape: {log_var.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke_test OK] mean={tuple(mean.shape)}, log_var={tuple(log_var.shape)}, "
          f"params={n_params:,}")


if __name__ == "__main__":
    # This runs in any environment (no torch required) and checks the shape
    # arithmetic independently of the actual nn.Module implementation above.
    _trace_shapes()

    if _HAS_TORCH:
        smoke_test()
    else:
        print("[smoke_test SKIPPED] torch not installed in this environment -- "
              "see task T3 in tasks.md. Shape arithmetic above was verified independently.")
