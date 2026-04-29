#!/usr/bin/env python3
"""Adaptive CNN hyperparameter search using Optuna's TPE sampler.

Results accumulate in a SQLite study (resumable). Each trial trains the
optimized_cnn via scripts/train.py and reports the best validation MAE.

Usage:
    python scripts/adaptive_search.py --device cuda --n-trials 50
    python scripts/adaptive_search.py --device cuda --n-trials 100  # resume / extend
"""

import argparse
import subprocess
from pathlib import Path

import optuna
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive CNN hyperparameter search via Optuna")
    p.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Total trials to reach (including any already completed)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--store-prefix", default="data/models/adaptive_search")
    p.add_argument(
        "--db-path", default=None, help="SQLite DB path (default: <store-prefix>/optuna.db)"
    )
    p.add_argument("--study-name", default="cnn_adaptive_search")
    p.add_argument(
        "--timeout",
        type=float,
        default=60 * 60 * 10,  # 10 hours
        help="Stop after this many seconds regardless of --n-trials",
    )
    return p.parse_args()


def build_train_command(hp: dict, exp_name: str, store_prefix: str, device: str) -> list[str]:
    return [
        "python",
        "scripts/train.py",
        "+experiments=optimized_cnn",
        f"learning_rate={hp['learning_rate']}",
        f"model.weight_decay={hp['weight_decay']}",
        f"n_epochs={hp['n_epochs']}",
        f"network.hidden_dim={hp['hidden_dim']}",
        f"network.n_blocks={hp['n_blocks']}",
        f"exp_name={exp_name}",
        f"store_path={store_prefix}/{exp_name}",
        f"device={device}",
        "log_wandb=true",
        "use_timestamp=false",
    ]


def parse_val_loss(store_prefix: str, exp_name: str) -> float | None:
    log_file = Path(store_prefix) / exp_name / "train_log.csv"
    if not log_file.exists():
        return None
    df = pd.read_csv(log_file)
    epoch_rows = df[df["epoch"].notna()]
    if epoch_rows.empty:
        return None
    return float(epoch_rows["val/epoch_loss"].min())


def make_objective(store_prefix: str, device: str):
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-2),
            "n_epochs": trial.suggest_int("n_epochs", 20, 40),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "n_blocks": trial.suggest_categorical("n_blocks", [4, 6, 8, 12]),
        }
        exp_name = f"adaptive_t{trial.number:03d}"
        cmd = build_train_command(hp, exp_name, store_prefix, device)

        print(f"\n=== Trial {trial.number} ===")
        for k, v in hp.items():
            print(f"  {k}: {v}")

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"  CRASHED (returncode={proc.returncode})")
            return float("inf")

        val_loss = parse_val_loss(store_prefix, exp_name)
        if val_loss is None:
            print("  Could not parse results — treating as crashed")
            return float("inf")

        print(f"  val_loss: {val_loss:.4f}")
        return val_loss

    return objective


def print_summary(study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.value is not None and t.value < float("inf")]
    if not completed:
        print("\nNo completed trials yet.")
        return

    print(f"\n=== Best trial (of {len(completed)} completed) ===")
    best = study.best_trial
    print(f"  Trial {best.number}  val_loss={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    top = sorted(completed, key=lambda t: t.value)[:5]
    print(f"\n=== Top {len(top)} trials ===")
    for t in top:
        params_str = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  Trial {t.number:3d}  val_loss={t.value:.4f}  {params_str}")


def main() -> None:
    args = parse_args()
    Path(args.store_prefix).mkdir(parents=True, exist_ok=True)

    db_path = args.db_path or f"{args.store_prefix}/optuna.db"
    storage = f"sqlite:///{db_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="minimize",
        load_if_exists=True,
    )

    already_done = len(study.trials)
    remaining = max(0, args.n_trials - already_done)

    if remaining == 0:
        print(
            f"Study already has {already_done} trials (target: {args.n_trials}). Nothing to run."
        )
    else:
        print(
            f"Study '{args.study_name}': {already_done} existing trials, running {remaining} more"
        )
        study.optimize(
            make_objective(args.store_prefix, args.device),
            n_trials=remaining,
            timeout=args.timeout,
        )

    print_summary(study)


if __name__ == "__main__":
    main()
