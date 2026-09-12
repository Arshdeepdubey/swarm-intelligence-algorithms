import sys
from pathlib import Path

# Make `import swarm_fs` work without an editable install, both for pytest
# and for anyone poking around in a plain REPL from the repo root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
