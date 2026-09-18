import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import torch
from model import SRModel

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

if __name__ == '__main__':
    main()
