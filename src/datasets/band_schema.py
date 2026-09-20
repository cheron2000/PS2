"""
src/datasets/band_schema.py — canonical band schemas and provenance
manifests (T20, narrowed from the external audit's original T14).

SCOPE NOTE — the audit's full T14 asked for real TACO/SEN2NAIP, SEN2Vénus/
SAFE-or-COG, and Cartosat FILE-FORMAT adapters. That genuinely can't be
built here: it needs real files from huggingface.co, the official
SEN2Vénus release, or Bhoonidhi — none reachable from this sandbox's
network allowlist, the exact same wall T5/T6/T10/T13 already hit. What
CAN be built without real data, and is real, useful work on its own:
  1. Canonical band schemas (name/wavelength/unit/scale-offset) for each
     product this project uses, as an explicit, checkable contract
     instead of an implicit assumption buried in code.
  2. Validation that actually rejects malformed input — specifically the
     audit's named concern ("Cartosat can repeat one band or truncate
     extra bands"): band-count mismatches and suspiciously-duplicated
     bands (a real symptom of a band-order bug, not a legitimate
     multispectral signal) are caught, not silently accepted.
  3. A provenance manifest (source, sensor, acquisition metadata,
     checksum) so a training run can record exactly what data it saw.
  4. Wired into src/datasets/sen2naip.py's actual loading path (see that
     file's diff), not just built in isolation — a schema nobody calls
     doesn't fix anything.

HONEST LIMITATION: the plain .npy files this project's loaders read (see
sen2naip.py's KNOWN GAP) carry no band metadata of their own. This module
can verify band COUNT and detect duplicate/degenerate bands, but it
cannot cryptographically confirm that band 0 of a given .npy array is
really centered at 492nm — that would need real per-file metadata (e.g.
a sidecar JSON or an actual TACO/COG file with embedded band
descriptions), which the real-format adapters (still not built, see
tasks.md) would need to provide. Documented here rather than silently
assumed.

Wavelength values below follow each sensor's own published band
specification (Sentinel-2: ESA; Cartosat-3: ISRO/ANTRIX; NAIP: USDA/FSA;
VENµS: CNES/ESA) — these are stable, standard technical specs, not
empirical measurements from data this sandbox has seen.

EXECUTION STATUS: written and tested by agent4 — see test_band_schema.py.
Pure numpy, no torch/network dependency.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class BandSchemaError(ValueError):
    """Raised when array data doesn't match its declared band schema —
    deliberately a distinct exception type, not a bare ValueError, so
    callers can catch schema violations specifically (e.g. to log them
    into dropped_pairs with a clear reason, as sen2naip.py now does)."""


@dataclass(frozen=True)
class Band:
    name: str
    wavelength_nm: Optional[float]  # None for non-optical/panchromatic-broadband bands
    unit: str = "reflectance"
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class BandSchema:
    product_name: str
    bands: tuple  # tuple of Band, not list — frozen dataclass needs hashable fields
    allow_extra_bands: bool = False
    duplicate_correlation_threshold: float = 0.999
    """Two bands with pairwise correlation above this are flagged as a
    likely band-order/repeat bug, not real spectral similarity — even
    visually similar bands like Red/NIR over vegetation rarely exceed
    ~0.97-0.98 correlation; 0.999+ is what an accidentally-repeated
    channel looks like, not a coincidence worth risking false positives over."""

    @property
    def num_bands(self) -> int:
        return len(self.bands)


# Sentinel-2 L2A, 4-band subset (R, G, B, NIR at 10m) used throughout this
# project. Central wavelengths per ESA's Sentinel-2 spectral response spec.
SENTINEL2_L2A_4BAND = BandSchema(
    product_name="Sentinel-2 L2A (B2,B3,B4,B8)",
    bands=(
        Band("blue", 492.4, unit="reflectance", scale=1 / 10000),
        Band("green", 559.8, unit="reflectance", scale=1 / 10000),
        Band("red", 664.6, unit="reflectance", scale=1 / 10000),
        Band("nir", 832.8, unit="reflectance", scale=1 / 10000),
    ),
)

# NAIP: USDA aerial imagery, R/G/B/NIR, no fixed narrow-band wavelength
# spec the way satellite sensors have (it's a broadband aerial camera) —
# wavelength_nm left as an approximate visible/NIR center, not a precise spec.
NAIP_4BAND = BandSchema(
    product_name="NAIP (R,G,B,NIR)",
    bands=(
        Band("red", 660.0, unit="DN", scale=1 / 255),
        Band("green", 560.0, unit="DN", scale=1 / 255),
        Band("blue", 480.0, unit="DN", scale=1 / 255),
        Band("nir", 850.0, unit="DN", scale=1 / 255),
    ),
)

# Cartosat-3 MX (multispectral), per ISRO's published band specification.
# Band-edge figures are ISRO's documented ranges; centers given here are the
# approximate midpoint, not a lab-measured spectral response curve.
CARTOSAT3_MX_4BAND = BandSchema(
    product_name="Cartosat-3 MX",
    bands=(
        Band("blue", 485.0, unit="DN"),
        Band("green", 555.0, unit="DN"),
        Band("red", 640.0, unit="DN"),
        Band("nir", 820.0, unit="DN"),
    ),
)

# SEN2Vénus: VENµS bands matched to the Sentinel-2-equivalent subset this
# project's loader uses (see sen2venus.py) — VENµS has 12 bands total;
# only the 4 used for this project's secondary 5m route are declared here.
SEN2VENUS_4BAND = BandSchema(
    product_name="SEN2Vénus (Sentinel-2-matched subset)",
    bands=(
        Band("blue", 490.0, unit="reflectance", scale=1 / 10000),
        Band("green", 555.0, unit="reflectance", scale=1 / 10000),
        Band("red", 620.0, unit="reflectance", scale=1 / 10000),
        Band("nir", 865.0, unit="reflectance", scale=1 / 10000),
    ),
)


def _band_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened bands. Used to detect
    accidental band duplication, not general similarity analysis."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        # A constant band correlates as undefined with anything; treat as
        # NOT a duplicate-detection match (a genuinely blank/nodata band
        # isn't the same failure mode as a copy-pasted channel), but this
        # is itself worth flagging separately — see validate_bands.
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def validate_bands(data: np.ndarray, schema: BandSchema) -> None:
    """data: (C, H, W). Raises BandSchemaError if:
      - band count doesn't match schema.num_bands (unless allow_extra_bands
        and data has >= num_bands)
      - any two bands are near-duplicates (likely a repeated/mis-mapped
        channel, per the audit's specific concern)
      - any band is exactly constant (a real signal band should never be
        perfectly flat; this usually means a missing/nodata-filled channel
        slipped through as if it were real data)
    Does NOT and cannot verify wavelength identity from array data alone —
    see this module's HONEST LIMITATION note.
    """
    if data.ndim != 3:
        raise BandSchemaError(f"expected (C, H, W) array, got shape {data.shape}")

    actual_bands = data.shape[0]
    expected_bands = schema.num_bands
    if schema.allow_extra_bands:
        if actual_bands < expected_bands:
            raise BandSchemaError(
                f"{schema.product_name}: expected at least {expected_bands} bands, got {actual_bands}"
            )
    elif actual_bands != expected_bands:
        raise BandSchemaError(
            f"{schema.product_name}: expected exactly {expected_bands} bands "
            f"({[b.name for b in schema.bands]}), got {actual_bands}"
        )

    for i in range(actual_bands):
        if np.std(data[i]) == 0:
            band_name = schema.bands[i].name if i < len(schema.bands) else f"band[{i}]"
            raise BandSchemaError(
                f"{schema.product_name}: band '{band_name}' (index {i}) is perfectly constant — "
                f"likely a missing/nodata channel, not a real signal"
            )

    for i in range(actual_bands):
        for j in range(i + 1, actual_bands):
            corr = _band_correlation(data[i], data[j])
            if corr >= schema.duplicate_correlation_threshold:
                name_i = schema.bands[i].name if i < len(schema.bands) else f"band[{i}]"
                name_j = schema.bands[j].name if j < len(schema.bands) else f"band[{j}]"
                raise BandSchemaError(
                    f"{schema.product_name}: bands '{name_i}' (index {i}) and '{name_j}' "
                    f"(index {j}) are near-identical (correlation={corr:.4f}) — likely a "
                    f"repeated/truncated channel, not a real duplicate spectral signal"
                )


@dataclass
class ProvenanceManifest:
    source_id: str
    product: str
    sensor: str
    checksum: str
    acquisition_date: Optional[str] = None
    processing_level: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "product": self.product,
            "sensor": self.sensor,
            "checksum": self.checksum,
            "acquisition_date": self.acquisition_date,
            "processing_level": self.processing_level,
            "extra": self.extra,
        }


def compute_checksum(data: np.ndarray) -> str:
    """SHA-256 of the array's raw bytes, prefixed with dtype/shape so two
    arrays with identical bytes but different interpreted shape/dtype
    don't collide."""
    header = f"{data.dtype}|{data.shape}|".encode("utf-8")
    return hashlib.sha256(header + np.ascontiguousarray(data).tobytes()).hexdigest()


def build_manifest(data: np.ndarray, source_id: str, product: str, sensor: str,
                    acquisition_date: str = None, processing_level: str = None,
                    **extra) -> ProvenanceManifest:
    """Convenience constructor: computes the checksum for you rather than
    making every caller remember to call compute_checksum() separately."""
    return ProvenanceManifest(
        source_id=source_id,
        product=product,
        sensor=sensor,
        checksum=compute_checksum(data),
        acquisition_date=acquisition_date,
        processing_level=processing_level,
        extra=extra,
    )
