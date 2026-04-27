import re
from pathlib import Path


def find_matching_runs(exp_name: str, models_dir: str = "data/models") -> list:
    """Return sorted list of run dirs matching exp_name (exact or timestamped)."""
    base = Path(models_dir)
    if (base / exp_name).is_dir():
        return [exp_name]
    pattern = re.compile(rf"^{re.escape(exp_name)}_\d{{8}}_\d{{6}}$")
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and pattern.match(d.name)
    )
