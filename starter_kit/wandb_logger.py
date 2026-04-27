#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

import logging
from typing import Dict, Any, Optional

main_logger = logging.getLogger(__name__)


class WandbLogger:
    r'''
    Thin wrapper around the wandb API for experiment tracking.

    Initialises a wandb run on construction and exposes log() / finish()
    methods that mirror the CSVLogger interface so BaseModel can call both
    uniformly.
    '''

    def __init__(self, project: str, name: str, config: Dict[str, Any]) -> None:
        r'''
        Initialise a wandb run.

        Parameters
        ----------
        project : str
            W&B project name.
        name : str
            Run name (typically the experiment name).
        config : Dict[str, Any]
            Hyperparameter dictionary logged to the run.
        '''
        import wandb
        self._run = wandb.init(project=project, name=name, config=config)
        main_logger.debug(f"wandb run initialised: {self._run.url}")

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        r'''
        Log a dictionary of metrics to wandb.

        Parameters
        ----------
        log_dict : Dict[str, Any]
            Metric values to record.
        step : int, optional
            Global step for the log entry.
        '''
        import wandb
        wandb.log(log_dict, step=step)

    def finish(self) -> None:
        r'''
        Mark the wandb run as finished and upload remaining data.
        '''
        import wandb
        wandb.finish()
