"""T19 static regression checks for reproducibility helpers."""
import ast
from pathlib import Path

src = Path("src/train.py").read_text(encoding="utf-8")
tree = ast.parse(src)
names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
required = {
    "dataset_fingerprint",
    "write_run_manifest",
    "resume_from_checkpoint",
    "_rng_state",
    "_restore_rng_state",
}
missing = required - names
if missing:
    raise AssertionError(f"missing T19 helpers: {sorted(missing)}")

assert 'weights_only=True' in src
assert 'os.replace(tmp, path)' in src
assert '"dataset_fingerprint"' in src
assert '"rng_state"' in src
assert '--resume' in src
assert 'run_manifest.json' in src

print("T19 static checks passed")
