"""
src/datasets/worldcover.py — ESA WorldCover land-cover loader (T13).

Split out from the original T8 (see tasks.md): feeds REAL land-cover
labels into src/eval_downstream.py's compare_downstream_utility(), which
was built against synthetic labels in T8 and needs no changes to accept
real ones from here.

ON-DISK FORMAT SUPPORTED HERE (real product, real format): ESA WorldCover
is distributed as Cloud-Optimized GeoTIFFs, one per 3x3-degree tile, e.g.
`ESA_WorldCover_10m_2021_v200_N00E006_Map.tif`, publicly hosted on AWS S3
(bucket `esa-worldcover`, also mirrored at
https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/...).
10m resolution, Sentinel-1/2 derived, 11 land-cover classes with fixed
value codes (WORLDCOVER_CLASSES below) — this schema is well-documented
and stable (unlike T5's TACO format or T6/T10's undocumented portal
schemas), so unlike those tasks, THIS module targets the real product
format directly rather than a placeholder convention.

KNOWN GAP: this sandbox's network allowlist doesn't include
esa-worldcover.org, AWS S3, or any GIS data host (same class of limitation
every prior data task hit) — so no real WorldCover tile could be
downloaded and tested against here. What IS tested here, for real: real
rasterio GeoTIFF I/O (read/write), real CRS/affine-transform handling,
and the class-remapping/reprojection logic — using a synthetically
generated but genuinely valid GeoTIFF, not just a numpy array. This is a
stronger test than T5/T6/T10 could manage for their real-format gaps,
because rasterio itself doesn't need network access, only the actual
WorldCover files do.

EXECUTION STATUS: written and tested by agent4 (formerly sonnet5) —
see test_worldcover.py. rasterio>=1.5 required (added to requirements.txt).
"""
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

# ESA WorldCover v200 class codes -> (consecutive label, name). Consecutive
# labels are what src/eval_downstream.py's confusion_matrix/IoU functions
# require (values in [0, num_classes)) — WorldCover's raw codes (10-100)
# don't satisfy that directly, so this remap is a required step, not
# cosmetic.
WORLDCOVER_CLASSES = {
    10:  (0, "Tree cover"),
    20:  (1, "Shrubland"),
    30:  (2, "Grassland"),
    40:  (3, "Cropland"),
    50:  (4, "Built-up"),
    60:  (5, "Bare / sparse vegetation"),
    70:  (6, "Snow and ice"),
    80:  (7, "Permanent water bodies"),
    90:  (8, "Herbaceous wetland"),
    95:  (9, "Mangroves"),
    100: (10, "Moss and lichen"),
}
NUM_WORLDCOVER_CLASSES = len(WORLDCOVER_CLASSES)
_RAW_TO_CONSECUTIVE = {raw: consecutive for raw, (consecutive, _name) in WORLDCOVER_CLASSES.items()}
NODATA_LABEL = 255  # WorldCover's own nodata convention; kept as-is, NOT remapped into [0, 11)


def remap_worldcover_classes(raw_labels: np.ndarray) -> np.ndarray:
    """raw_labels: integer array with WorldCover's raw class codes (10-100)
    and/or NODATA_LABEL (255). Returns consecutive labels in [0, 11) for
    known classes; unrecognized values (including nodata) map to -1 so
    callers can explicitly filter them rather than silently misclassifying
    nodata as a real class."""
    output = np.full(raw_labels.shape, -1, dtype=np.int64)
    for raw_code, consecutive_label in _RAW_TO_CONSECUTIVE.items():
        output[raw_labels == raw_code] = consecutive_label
    return output


def load_worldcover_tile(path: str):
    """Real rasterio GeoTIFF read. Returns (data, transform, crs) where
    data is (H, W) raw WorldCover class codes (not yet remapped — call
    remap_worldcover_classes() separately, keeping I/O and class-mapping
    as separate, independently testable steps)."""
    with rasterio.open(path) as src:
        data = src.read(1)  # WorldCover is single-band
        transform = src.transform
        crs = src.crs
    return data, transform, crs


def align_to_reference(
    worldcover_data: np.ndarray,
    worldcover_transform: Affine,
    worldcover_crs,
    reference_transform: Affine,
    reference_crs,
    reference_shape: tuple,
) -> np.ndarray:
    """Reproject/resample WorldCover data (10m, its own CRS) onto the
    exact pixel grid of a reference Sentinel-2/SR tile (reference_transform,
    reference_crs, reference_shape) — the actual "align to a given
    Sentinel-2 AOI" requirement from tasks.md's T13 description. Uses
    nearest-neighbor resampling deliberately: land-cover class codes are
    categorical, so bilinear/cubic resampling (appropriate for continuous
    reflectance) would silently invent nonexistent intermediate class
    values.

    Returns an array of shape reference_shape with WorldCover's raw class
    codes reprojected onto the reference grid; call
    remap_worldcover_classes() afterward for use with eval_downstream.py.
    """
    destination = np.zeros(reference_shape, dtype=worldcover_data.dtype)
    reproject(
        source=worldcover_data,
        destination=destination,
        src_transform=worldcover_transform,
        src_crs=worldcover_crs,
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        resampling=Resampling.nearest,
    )
    return destination


def write_geotiff(path: str, data: np.ndarray, transform: Affine, crs, dtype=None):
    """Write a single-band GeoTIFF — used by the test suite to create a
    synthetic-but-real WorldCover-shaped tile, and generally useful for
    exporting remapped/aligned label rasters for inspection."""
    if dtype is None:
        dtype = data.dtype
    from pathlib import Path
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(
        str(tmp_path), "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data.astype(dtype), 1)
    os.replace(tmp_path, path)
