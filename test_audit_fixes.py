"""
test_audit_fixes.py — torch-free verification of the arithmetic/logic in two
of the three 2026-09-19 audit fixes (see build-status.md for the full list).

What this CAN verify without torch: the variance unit-scaling formula, and
the train/val split-sizing logic (extracted here as pure functions mirroring
what's in infer.py / train.py, since importing those directly requires torch).

What this CANNOT verify: that torch.load(..., weights_only=True) actually
accepts a real checkpoint saved by this codebase's save_checkpoint(), or that
random_split + DataLoader wiring is bug-free at the PyTorch API level. Those
need execution with real torch -- flagged explicitly in build-status.md as
the next thing to verify, same pattern as T1/T3 earlier in this build.
"""
import numpy as np


def denormalise_mean_and_variance(mean: np.ndarray, variance: np.ndarray, divisor: float):
    """Mirrors the fixed logic in infer.py's reflectance branch. Var(a*X) = a^2 * Var(X)."""
    return mean * divisor, variance * (divisor ** 2)


def compute_split_sizes(n: int, val_fraction: float):
    """Mirrors split_dataset()'s sizing logic in train.py (pure arithmetic part)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train < 1:
        raise ValueError(f"val_fraction={val_fraction} leaves no training samples (n={n})")
    return n_train, n_val


def test_variance_scaling_matches_var_of_scaled_variable():
    rng = np.random.default_rng(0)
    raw_samples = rng.normal(loc=0.05, scale=0.01, size=100_000)  # model-space "reflectance"
    divisor = 10000.0

    true_var_of_raw = raw_samples.var()
    true_var_of_scaled = (raw_samples * divisor).var()  # ground truth: Var(aX)

    mean_scaled, variance_scaled = denormalise_mean_and_variance(
        mean=np.array([raw_samples.mean()]),
        variance=np.array([true_var_of_raw]),
        divisor=divisor,
    )

    # The fixed formula (variance * divisor**2) should closely match the
    # empirical variance of the actually-scaled data.
    ratio = float(variance_scaled[0]) / true_var_of_scaled
    assert abs(ratio - 1.0) < 1e-6, f"variance scaling formula off by ratio {ratio}"

    # And explicitly confirm the OLD (buggy) behavior -- raw variance with no
    # scaling at all -- would have been wrong by a factor of divisor**2,
    # which is the actual bug the audit flagged, not a rounding issue.
    old_buggy_ratio = true_var_of_raw / true_var_of_scaled
    assert abs(old_buggy_ratio - (1.0 / divisor ** 2)) < 1e-9
    print(f"[PASS] test_variance_scaling_matches_var_of_scaled_variable "
          f"(fixed ratio={ratio:.8f}, old buggy ratio would have been off by {divisor**2:.0f}x)")


def test_split_sizes_partition_exactly():
    for n, frac in [(100, 0.2), (10, 0.2), (5, 0.5), (1000, 0.01), (7, 0.3)]:
        n_train, n_val = compute_split_sizes(n, frac)
        assert n_train + n_val == n, (n, frac, n_train, n_val)
        assert n_train >= 1 and n_val >= 1
    print("[PASS] test_split_sizes_partition_exactly")


def test_split_rejects_degenerate_fractions():
    for bad_frac in [0.0, 1.0, -0.1, 1.5]:
        try:
            compute_split_sizes(100, bad_frac)
            raise AssertionError(f"expected ValueError for val_fraction={bad_frac}")
        except ValueError:
            pass
    # a val_fraction that would leave zero training samples should also raise
    try:
        compute_split_sizes(1, 0.99)  # n=1, n_val=max(1,0)=1 -> n_train=0
        raise AssertionError("expected ValueError when split leaves no training samples")
    except ValueError:
        pass
    print("[PASS] test_split_rejects_degenerate_fractions")


if __name__ == "__main__":
    test_variance_scaling_matches_var_of_scaled_variable()
    test_split_sizes_partition_exactly()
    test_split_rejects_degenerate_fractions()
    print("\nAll torch-free arithmetic checks passed. See build-status.md for "
          "what still needs real-torch execution verification.")
