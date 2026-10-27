"""The committed fixtures are exactly what scripts/generate_fixtures.py produces."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "generate_fixtures.py"
COMMITTED = ROOT / "src" / "mirkaso_copilot" / "fixtures"


def test_generator_reproduces_committed_fixtures(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], check=True, capture_output=True)
    generated = sorted(p.name for p in tmp_path.glob("*.json"))
    assert generated == sorted(p.name for p in COMMITTED.glob("*.json"))
    for name in generated:
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes(), name
