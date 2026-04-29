import re
from pathlib import Path


def find_matching_runs(exp_name: str, models_dir: str = "data/models") -> list:
    """Return sorted list of run dirs matching exp_name (exact, timestamped, or sweep subdirs)."""
    base = Path(models_dir)
    exp_dir = base / exp_name

    if exp_dir.is_dir():
        if (exp_dir / "best_model.ckpt").is_file():
            return [exp_name]
        # Sweep parent: return subdirs that contain a checkpoint
        sweep_runs = sorted(
            f"{exp_name}/{d.name}"
            for d in exp_dir.iterdir()
            if d.is_dir() and (d / "best_model.ckpt").is_file()
        )
        if sweep_runs:
            return sweep_runs

    pattern = re.compile(rf"^{re.escape(exp_name)}_\d{{8}}_\d{{6}}$")
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and pattern.match(d.name)
    )
