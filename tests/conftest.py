import sys
from pathlib import Path

# Make `core.src...` importable, matching how the server imports it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
