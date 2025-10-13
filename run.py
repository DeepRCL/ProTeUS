#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
import sys

import omegaconf
from omegaconf import OmegaConf

# --- Imports: assume package-installed or run from repo root ---
from medAI.utils.reproducibility import set_global_seed
from src.train_sam import BKSAMExperiment
from src.train_medsam import BKMedSAMExperiment
from src.train_proteus import ProTeUSExperiment
from src.train_unet import UNetExperiment

try:
    import submitit  # optional, only needed when launching via SLURM/submitit
except Exception:  # pragma: no cover
    submitit = None  # type: ignore


# ------------------------------
# Utilities
# ------------------------------
def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        p: Path to the directory to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def _set_optional_env(key: str, value: str | None) -> None:
    """Set environment variable if value is not None or empty.
    
    Args:
        key: Environment variable name.
        value: Environment variable value to set.
    """
    if value is not None and value != "":
        os.environ[key] = value


def _snapshot_config(conf: omegaconf.DictConfig, out_dir: Path) -> Path:
    """Save a timestamped snapshot of the configuration.
    
    Args:
        conf: OmegaConf configuration dictionary.
        out_dir: Directory to save the configuration snapshot.
        
    Returns:
        Path to the saved configuration file.
    """
    _ensure_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"{conf.mode}_{timestamp}.yaml"
    OmegaConf.save(conf, out_path)
    return out_path


def _select_experiment(conf: omegaconf.DictConfig):
    """Select the appropriate experiment class based on configuration mode.
    
    Args:
        conf: OmegaConf configuration dictionary containing the mode.
        
    Returns:
        Initialized experiment instance.
        
    Raises:
        ValueError: If the mode is not recognized.
    """
    mode = str(conf.mode).lower()
    if "proteus" in mode:
        return ProTeUSExperiment(conf)
    if "medsam" in mode:
        return BKMedSAMExperiment(conf)
    if "bksam" in mode:
        return BKSAMExperiment(conf)
    if "unet" in mode:
        return UNetExperiment(conf)
    raise ValueError(
        f"Unknown mode '{conf.mode}'. "
        "Expected one of: 'proteus', 'medsam', 'bksam', 'unet' "
        "(optionally with suffixes like '-semi_sl')."
    )


# ------------------------------
# Submitit-compatible runner
# ------------------------------
class Main:
    """Submitit-compatible main runner class for ProTeUS experiments.
    
    This class handles the main execution flow for ProTeUS training experiments,
    including SLURM cluster compatibility, checkpoint directory setup, and
    experiment initialization.
    
    Attributes:
        conf: OmegaConf configuration dictionary.
    """
    
    def __init__(self, conf: omegaconf.DictConfig):
        """Initialize the main runner.
        
        Args:
            conf: OmegaConf configuration dictionary.
        """
        self.conf = conf

    def __call__(self):
        """Execute the main experiment run.
        
        Sets up SLURM environment, checkpoint directories, and runs the selected experiment.
        Handles cluster-specific configurations and Weights & Biases integration.
        """
        # SLURM context (best-effort)
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        user = os.getenv("USER")

        # Throttle tqdm a bit in cluster logs
        _set_optional_env("TQDM_MININTERVAL", "30")

        # Resume W&B by job id if present (safe: not a secret)
        if slurm_job_id:
            _set_optional_env("WANDB_RUN_ID", slurm_job_id)
            _set_optional_env("WANDB_RESUME", "allow")

        # Checkpoint directory: prefer /checkpoint on SLURM if it exists
        ckpt_dir = None
        if slurm_job_id and user and Path("/checkpoint").exists():
            ckpt_dir = Path("/checkpoint") / user / slurm_job_id
        else:
            # Fallback to configurable or local path
            cfg_ckpt = getattr(self.conf, "checkpoint_dir", None)
            ckpt_dir = Path(cfg_ckpt) if cfg_ckpt else Path("outputs/checkpoints")

        _ensure_dir(ckpt_dir)
        self.conf.checkpoint_dir = str(ckpt_dir)
        self.conf.slurm_job_id = slurm_job_id

        # Run experiment
        exp = _select_experiment(self.conf)
        exp.run()

    # Needed by submitit for fault-tolerance
    def checkpoint(self):
        """Create a checkpoint for fault tolerance with submitit.
        
        Returns:
            DelayedSubmission object for resuming the experiment.
            
        Raises:
            RuntimeError: If submitit is not available.
        """
        if submitit is None:
            raise RuntimeError("submitit is not available in this environment.")
        return submitit.helpers.DelayedSubmission(Main(self.conf))


# ------------------------------
# CLI
# ------------------------------
def main(argv: list[str] | None = None) -> int:
    """Main entry point for ProTeUS training experiments."""
    parser = argparse.ArgumentParser(
        description="ProTeUS: A Spatio-Temporal Enhanced Ultrasound-Based Framework for Prostate Cancer Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Use default proteus.yaml config
  python run.py -y medsam                 # Use medsam.yaml config
  python run.py -y config/custom.yaml    # Use custom config file
  python run.py -o data.batch_size=8 trainer.lr=1e-4  # Override config parameters
        """
    )
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        default="proteus",  
        help="Config name (without .yaml) under config/, or a direct path to a YAML file.",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="+",
        default=[],
        help="OmegaConf dotlist overrides, e.g. data.batch_size=4 trainer.lr=1e-4",
    )
    args = parser.parse_args(argv)

    # Resolve config path
    cfg_arg = Path(args.yaml)
    if cfg_arg.suffix.lower() in {".yml", ".yaml"} and cfg_arg.exists():
        cfg_path = cfg_arg
    else:
        cfg_path = Path("config") / f"{args.yaml}.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    conf = OmegaConf.load(str(cfg_path))

    # Apply overrides
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))

    # Set seed if provided
    seed = conf.get("seed", None)
    if seed is not None:
        set_global_seed(int(seed))

    # Snapshot the (possibly overridden) config
    snap_path = _snapshot_config(conf, Path("config/log"))
    logging.basicConfig(level=logging.INFO)
    logging.info("Saved config snapshot to %s", snap_path)

    # Execute (submitit-compatible object)
    runner = Main(conf)
    runner()  # direct call; if you use submitit, it will call runner() itself
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
