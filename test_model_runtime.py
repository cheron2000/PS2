import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import torch
from model import SRModel

def test_invalid_model_configs_fail_at_construction():
    # NOTE (agent4): the original first case here — dict(base_channels=4,
    # num_attn_blocks=1, num_heads=1) — was not actually invalid. Verified
    # directly: it constructs a working model (base_channels=4 is evenly
    # divisible by num_heads=1, every other param passes SRModel's actual
    # validation rules) with a correct, finite forward pass. The test's
    # assumption was wrong, not the model — this was never caught earlier
    # because the whole function was silently never executed (see
    # build-status.md: file-corruption + missing-from-runner double bug).
    # Replaced with an invalid `scale` value, which exercises a genuinely
    # distinct validation path the other two cases don't cover.
    cases = [
        dict(base_channels=8, num_attn_blocks=1, num_heads=2, scale=3),
        dict(base_channels=10, num_attn_blocks=1, num_heads=4),
        dict(base_channels=8, num_attn_blocks=1, num_heads=2, window_size=0),
    ]
    for kwargs in cases:
        try:
            SRModel(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def main():
    torch.manual_seed(0)
    model = SRModel(in_channels=4, out_channels=4, base_channels=32, num_attn_blocks=2, window_size=8, num_heads=4, scale=4)
    model.eval()
    with torch.no_grad():
        mean, log_var = model(torch.randn(1, 4, 65, 70))
    assert mean.shape == (1, 4, 260, 280), mean.shape
    assert log_var.shape == mean.shape, log_var.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()
    print(f'extended runtime OK: mean={tuple(mean.shape)}, log_var={tuple(log_var.shape)}')
    test_invalid_model_configs_fail_at_construction()
    print('invalid model config rejection OK')

if __name__ == '__main__':
    main()