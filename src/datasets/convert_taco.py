"""
SEN2NAIPv2 cross-sensor TACO → .npy streaming converter.

Instead of downloading the full 9.7GB TACO file, this script:
1. Fetches the Parquet metadata index (tiny, ~few MB) via HuggingFace API
2. For each pair, fetches ONLY that pair's byte range via HTTP range request
3. Parses the TACO sub-item (which contains LR and HR GTiffs)
4. Saves LR/HR as .npy

Each pair is ~1.2MB; 200 pairs = ~240MB total download.

LR: Sentinel-2 (4-band RGBNIR, uint16) → float32 [0,1]
HR: NAIP (4-band RGBNIR, uint16)       → float32 [0,1]
Scale factor: 4 (10m S2 → 2.5m NAIP)

Usage:
    python src/datasets/convert_taco.py --n 200 --out data/sen2naip
"""

from __future__ import annotations
import argparse
import io
import re
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from rasterio.io import MemoryFile

warnings.filterwarnings("ignore")

HF_BASE_URL = "https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2/resolve/main"


def _fetch_bytes(url: str, start: int, length: int, retries: int = 3) -> bytes:
    """HTTP range-request, with retries."""
    end = start + length - 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "PS2-converter/2.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) != length:
                raise ValueError(f"Incomplete read: got {len(data)}, expected {length}")
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _resolve_hf_cdn_url(hf_path: str) -> str:
    """
    Follow the HuggingFace redirect to get the actual CDN URL (e.g. AWS/Xet).
    We need this because HF issues 302 redirects and urllib doesn't keep Range headers.
    """
    req = urllib.request.Request(hf_path, headers={"User-Agent": "PS2-converter/2.0"})
    req.add_header("Range", "bytes=0-0")
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    with opener.open(req, timeout=30) as resp:
        return resp.url


def _decode_tortilla_pair(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode a TORTILLA sub-item containing 'lr' and 'hr' GTiffs.
    """
    buf = io.BytesIO(raw)
    magic = buf.read(2)
    if magic not in (b"\x23\x79", b"\x57\x58"):
        raise ValueError(f"Not a TORTILLA sub-item, magic={magic.hex()}")

    footer_offset = int.from_bytes(buf.read(8), "little")
    footer_length = int.from_bytes(buf.read(8), "little")
    footer_raw = raw[footer_offset : footer_offset + footer_length]
    
    table = pq.read_table(pa.BufferReader(footer_raw))
    df = table.to_pandas()
    
    lr_arr, hr_arr = None, None
    for i, row in df.iterrows():
        id_ = row['tortilla:id']
        offset = int(row['tortilla:offset'])
        length = int(row['tortilla:length'])
        b = raw[offset:offset+length]
        
        with MemoryFile(b) as memfile:
            with memfile.open() as dataset:
                arr = dataset.read().astype(np.float32)
                # Return raw DNs; the dataset loader applies normalization
                if id_ == 'lr':
                    lr_arr = arr
                elif id_ == 'hr':
                    hr_arr = arr
                    
    if lr_arr is None or hr_arr is None:
        raise ValueError("Sub-item missing lr or hr")
        
    return lr_arr, hr_arr


def convert_streaming(out_dir: str, n: int = 200) -> None:
    """Stream-extract n pairs from SEN2NAIPv2 cross-sensor via HTTP range requests."""
    import tacoreader.v1 as tacoreader

    lr_dir = Path(out_dir) / "lr"
    hr_dir = Path(out_dir) / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching TACO metadata index from HuggingFace...")
    ds = tacoreader.load("hf://datasets/tacofoundation/SEN2NAIPv2/sen2naipv2-crosssensor.taco")
    total = len(ds)
    n = min(n, total)
    print(f"  {total} pairs available. Extracting {n}...")

    hf_taco_url = f"{HF_BASE_URL}/sen2naipv2-crosssensor.taco"
    print("  Resolving CDN URL...")
    try:
        cdn_url = _resolve_hf_cdn_url(hf_taco_url)
    except Exception:
        cdn_url = hf_taco_url
    print(f"  CDN URL: {cdn_url[:80]}...")

    saved = 0
    errors = 0
    t0 = time.time()

    for i in range(total):
        if saved >= n:
            break

        row = ds.iloc[i]
        sample_id = re.sub(r"[^\w\-]", "_", str(row["tortilla:id"]))
        offset = int(row["tortilla:offset"])
        length = int(row["tortilla:length"])

        lr_path = lr_dir / f"{sample_id}.npy"
        hr_path = hr_dir / f"{sample_id}.npy"

        # Skip if already converted
        if lr_path.exists() and hr_path.exists():
            saved += 1
            continue

        try:
            try:
                fresh_url = _resolve_hf_cdn_url(hf_taco_url)
            except Exception:
                fresh_url = hf_taco_url

            raw = _fetch_bytes(fresh_url, offset, length)
            lr_arr, hr_arr = _decode_tortilla_pair(raw)

            # Crop HR to exact multiple of scale (if needed)
            scale = 4
            h, w = lr_arr.shape[1], lr_arr.shape[2]
            hr_crop = hr_arr[:, :h * scale, :w * scale]

            np.save(lr_path, lr_arr)
            np.save(hr_path, hr_crop)

            saved += 1
            elapsed = time.time() - t0
            rate = saved / elapsed if elapsed > 0 else 0
            eta = (n - saved) / rate if rate > 0 else float("inf")

            if saved <= 5 or saved % 10 == 0:
                print(
                    f"  [{saved:3d}/{n}] {sample_id[:40]:40s} | "
                    f"LR={lr_arr.shape} HR={hr_crop.shape} | "
                    f"{rate:.1f} pairs/s | ETA {eta/60:.1f}min"
                )

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{i}] ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Saved {saved}/{n} pairs ({errors} errors).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stream-convert SEN2NAIPv2 TACO to .npy pairs")
    ap.add_argument("--out", default="data/sen2naip", help="Output directory")
    ap.add_argument("--n", type=int, default=200, help="Number of pairs to extract")
    args = ap.parse_args()
    convert_streaming(out_dir=args.out, n=args.n)
