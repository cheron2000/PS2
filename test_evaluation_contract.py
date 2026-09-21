"""Tests for the frozen machine-readable evaluation contract."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation_contract import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    EvaluationContractError,
    load_default_evaluation_contract,
    load_evaluation_contract,
    validate_evaluation_contract,
)


def test_default_contract_loads_and_validates():
    contract = load_default_evaluation_contract()
    assert contract["contract_id"] == "si26142-sentinel2-sr-evaluation-v1"
    assert contract["split_policy"]["fractions"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert contract["baselines"]["required"][1]["id"] == "sr"
    assert DEFAULT_CONTRACT_PATH.is_file()


def _assert_rejected(mutator):
    contract = load_default_evaluation_contract()
    mutator(contract)
    try:
        validate_evaluation_contract(contract)
        raise AssertionError("policy drift should be rejected")
    except EvaluationContractError:
        pass


def test_split_leakage_policy_cannot_be_disabled():
    _assert_rejected(lambda c: c["split_policy"].update(scene_disjoint=False))


def test_mask_policy_cannot_be_downgraded():
    _assert_rejected(lambda c: c["mask_policy"].update(required_for_real_evaluation=False))


def test_bicubic_baseline_cannot_be_removed():
    def remove_bicubic(contract):
        contract["baselines"]["required"] = [
            item for item in contract["baselines"]["required"] if item["id"] != "bicubic"
        ]
    _assert_rejected(remove_bicubic)


def test_test_split_cannot_be_used_for_selection():
    _assert_rejected(lambda c: c["split_policy"].update(test_used_for_model_selection=True))


def test_positive_improvement_gate_is_required():
    _assert_rejected(
        lambda c: c["minimum_improvement_gates"]["primary_downstream_mean_iou_delta"].update(
            minimum_absolute_delta=0.0
        )
    )


def test_load_reports_malformed_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.json"
        path.write_text("not-json")
        try:
            load_evaluation_contract(path)
            raise AssertionError("malformed JSON should be rejected")
        except EvaluationContractError as exc:
            assert "failed to load" in str(exc)


def test_contract_is_json_round_trip_stable():
    contract = load_default_evaluation_contract()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "contract.json"
        path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")
        loaded = load_evaluation_contract(path)
        assert loaded == contract


def main():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} direct tests passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
