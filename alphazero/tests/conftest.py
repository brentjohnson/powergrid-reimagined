import sys
from pathlib import Path

# Make `import alphazero` work regardless of the invocation directory — the
# package lives at the repo root, not under `python/`, and isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
