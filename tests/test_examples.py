import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Automatically find all subdirectories that contain a main.py
example_dirs = [d for d in EXAMPLES_DIR.iterdir() if d.is_dir() and (d / "main.py").exists()]


@pytest.mark.parametrize("example_dir", example_dirs, ids=[d.name for d in example_dirs])
def test_example_runs(example_dir):
    """Run main.py in each example directory with a given n_epoch argument."""
    main_script = example_dir / "main.py"
    result = subprocess.run(
        [sys.executable, str(main_script), "--n_epoch", "3"],
        cwd=example_dir,
        capture_output=True,
        text=True,
    )

    # Print stdout for debugging (pytest -s will show it)
    print(result.stdout)
    print(result.stderr)

    # Assert script exits successfully
    assert result.returncode == 0, f"{example_dir.name} failed:\n{result.stderr}"
