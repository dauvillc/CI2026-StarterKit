#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

r"""
Run ensemble forecasts and submit averaged predictions to the portal.

Each experiment in ``ensemble_experiments`` is forecasted independently
(skipped if the netCDF already exists), then all per-region forecasts are
averaged pixel-wise before submission.

The ensemble experiment file (e.g. configs/experiments/my_ensemble.yaml)
lists experiment names; each name must have a trained checkpoint under
``data/models/<exp_name>/`` and a saved Hydra config at
``data/models/<exp_name>/hydra/config.yaml``.

Run from the starter_kit root directory::

    python scripts/submit_ens.py \
        +experiment=example_ensemble \
        email=you@example.com

Override device or skip forecasting::

    python scripts/submit_ens.py \
        +experiment=example_ensemble \
        device=cuda \
        skip_forecast=true \
        email=you@example.com
"""

# System modules
import logging
import sys
from pathlib import Path
from typing import Dict, List

# External modules
import hydra
import numpy as np
import requests
import xarray as xr

# Internal modules
from forecast import run_forecast
from omegaconf import DictConfig, OmegaConf
from starter_kit.utils import find_matching_runs

main_logger = logging.getLogger(__name__)

_REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]

_REGION_FILENAMES = {
    "era5_region1": "test_era5_region1.nc",
    "era5_region2": "test_era5_region2.nc",
    "aimip_region1": "test_aimip_region1.nc",
    "aimip_region2": "test_aimip_region2.nc",
}

_PORTAL_FIELDS = {
    "era5_region1": "file_era5_region1",
    "era5_region2": "file_era5_region2",
    "aimip_region1": "file_aimip_region1",
    "aimip_region2": "file_aimip_region2",
}


def get_network_config(exp_name: str) -> DictConfig:
    r"""
    Load the network sub-config saved by Hydra during training.

    Reads ``data/models/<exp_name>/hydra/config.yaml`` and returns the
    ``network`` node, which contains ``_target_`` and all hyperparameters
    needed to instantiate the network via ``hydra.utils.instantiate``.

    Parameters
    ----------
    exp_name : str
        Name of the trained experiment.

    Returns
    -------
    DictConfig
        Network configuration (``_target_`` plus constructor kwargs).

    Raises
    ------
    FileNotFoundError
        If the Hydra config file does not exist for ``exp_name``.
    """
    config_path = Path(f"data/models/{exp_name}/hydra/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Hydra config not found for experiment '{exp_name}': {config_path}"
        )
    # Load without resolving — the stored config has unresolved interpolations
    # (e.g. ${batch_size}) in other fields; only cfg.network is needed here
    # and it contains only concrete values.
    full_cfg = OmegaConf.load(config_path)
    return full_cfg.network


def _get_exp_forecast_path(exp_name: str, region: str) -> Path:
    return Path(f"data/forecasts/{exp_name}/{_REGION_FILENAMES[region]}")


def _ensure_forecast(
    exp_name: str,
    region: str,
    region_paths: DictConfig,
    cfg: DictConfig,
) -> Path:
    r"""
    Return the forecast path for one experiment/region, running it if absent.

    Parameters
    ----------
    exp_name : str
        Individual experiment name.
    region : str
        One of the four region keys.
    region_paths : DictConfig
        Region sub-config with ``input_path``.
    cfg : DictConfig
        Full ensemble config (provides ``device`` and ``data``).

    Returns
    -------
    Path
        Path to the existing (or freshly created) netCDF forecast file.
    """
    exp_output_path = _get_exp_forecast_path(exp_name, region)

    if exp_output_path.exists():
        main_logger.info(
            "Forecast already exists for %s / %s, skipping.",
            exp_name,
            region,
        )
        return exp_output_path

    main_logger.info("Forecasting %s / %s …", exp_name, region)
    network_cfg = get_network_config(exp_name)

    forecast_cfg = OmegaConf.create(
        {
            "network": OmegaConf.to_container(network_cfg, resolve=True),
            "ckpt_path": f"data/models/{exp_name}/best_model.ckpt",
            "input_path": str(region_paths.input_path),
            "output_path": str(exp_output_path),
            "device": cfg.device,
            "data": {
                "batch_size": cfg.data.batch_size,
                "num_workers": cfg.data.num_workers,
                "pin_memory": cfg.data.pin_memory,
            },
        }
    )
    run_forecast(forecast_cfg)
    return exp_output_path


def _average_forecasts(
    per_exp_paths: List[Path],
    output_path: str,
) -> None:
    r"""
    Average ``total_cloud_cover`` across multiple netCDF forecast files.

    Parameters
    ----------
    per_exp_paths : List[Path]
        Paths to per-experiment netCDF files (all must share lat/lon dims).
    output_path : str
        Destination path for the averaged netCDF file.
    """
    datasets = [xr.open_dataset(p) for p in per_exp_paths]
    try:
        stacked = np.stack([ds["total_cloud_cover"].values for ds in datasets], axis=0)
        avg = stacked.mean(axis=0)
        ref = datasets[0]
        result = xr.Dataset(
            {
                "total_cloud_cover": (
                    ["sample", "lat", "lon"],
                    avg,
                    {"long_name": "Total cloud cover", "units": "1"},
                )
            },
            coords={
                "sample": ref["sample"].values,
                "lat": ref["lat"].values,
                "lon": ref["lon"].values,
            },
        )
        result.to_netcdf(output_path)
    finally:
        for ds in datasets:
            ds.close()
    main_logger.info("Averaged forecast saved to %s", output_path)


def _run_all_ensemble_forecasts(cfg: DictConfig) -> None:
    r"""
    For every region, ensure all per-experiment forecasts exist then average.

    Parameters
    ----------
    cfg : DictConfig
        Full ensemble config tree.
    """
    ensemble_experiments = list(cfg.ensemble_experiments)
    expanded = []
    for name in ensemble_experiments:
        runs = find_matching_runs(name)
        if runs:
            expanded.extend(runs)
        else:
            main_logger.warning("No model runs found for exp_name=%s — skipping.", name)
    ensemble_experiments = expanded
    if not ensemble_experiments:
        main_logger.error("ensemble_experiments is empty — nothing to do.")
        sys.exit(1)

    for region in _REGIONS:
        region_paths = cfg.regions[region]
        per_exp_paths: List[Path] = []
        for exp_name in ensemble_experiments:
            path = _ensure_forecast(exp_name, region, region_paths, cfg)
            per_exp_paths.append(path)

        ensemble_output = Path(str(region_paths.output_path))
        ensemble_output.parent.mkdir(parents=True, exist_ok=True)
        main_logger.info("Averaging %d forecasts for %s …", len(per_exp_paths), region)
        _average_forecasts(per_exp_paths, str(ensemble_output))


def _collect_forecast_files(cfg: DictConfig) -> Dict[str, Path]:
    r"""
    Locate the four averaged forecast netCDF files defined in the config.

    Parameters
    ----------
    cfg : DictConfig
        Full ensemble config tree.

    Returns
    -------
    Dict[str, Path]
        Mapping from region name to absolute netCDF path.

    Raises
    ------
    FileNotFoundError
        If any of the four expected output files is missing.
    """
    files: Dict[str, Path] = {}
    missing = []
    for region in _REGIONS:
        output_path = cfg.regions[region].output_path
        path = Path(output_path)
        if not path.exists():
            missing.append(str(path))
        else:
            files[region] = path
    if missing:
        raise FileNotFoundError("Missing forecast files:\n" + "\n".join(f"  {p}" for p in missing))
    return files


def _submit_to_portal(
    email: str,
    portal_url: str,
    forecast_files: Dict[str, Path],
) -> None:
    r"""
    POST the four forecast files to the submission portal.

    Parameters
    ----------
    email : str
        Registered submitter email address.
    portal_url : str
        Base URL of the hackathon submission portal.
    forecast_files : Dict[str, Path]
        Mapping from region name to netCDF file path.

    Raises
    ------
    SystemExit
        If the server returns a non-2xx response.
    """
    url = portal_url.rstrip("/") + "/api/v1/submissions"
    handles = {}
    try:
        for region, path in forecast_files.items():
            handles[region] = open(path, "rb")
        files = {
            _PORTAL_FIELDS[region]: (
                path.name,
                handles[region],
                "application/octet-stream",
            )
            for region, path in forecast_files.items()
        }
        main_logger.info("Submitting to %s …", url)
        response = requests.post(
            url,
            data={"email": email},
            files=files,
            timeout=120,
        )
    finally:
        for handle in handles.values():
            handle.close()

    if response.ok:
        payload = response.json()
        unique_idx = payload.get("unique_idx")
        main_logger.info("Submission accepted.")
        main_logger.info("  unique_idx : %s", unique_idx)
        main_logger.info("  status     : %s", payload.get("status"))
        main_logger.info("  queue pos  : %s", payload.get("queue_position"))
        main_logger.info("  est. wait  : %s", payload.get("estimated_wait_formatted"))
        main_logger.info(
            "Check status with:\n  curl -s %s/api/v1/submissions/%s | python -m json.tool",
            portal_url.rstrip("/"),
            unique_idx,
        )
    else:
        main_logger.error(
            "Submission failed: %d %s",
            response.status_code,
            response.text,
        )
        sys.exit(1)


@hydra.main(
    config_path="../configs",
    config_name="submit_ens",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    r"""
    Entry point: ensemble-forecast all regions and submit to the portal.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree.
    """
    if not cfg.skip_forecast:
        _run_all_ensemble_forecasts(cfg)

    forecast_files = _collect_forecast_files(cfg)
    for region, path in forecast_files.items():
        main_logger.info("  %s: %s", region, path)

    _submit_to_portal(cfg.email, cfg.url_portal, forecast_files)


if __name__ == "__main__":
    main()
