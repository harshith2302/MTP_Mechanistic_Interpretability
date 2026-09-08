"""Shared helpers: config loading, paths, git sha."""
import os
import subprocess

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    with open(path or os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def code_version():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def model_by_key(cfg, key):
    for m in cfg["models"]:
        if m["key"] == key:
            return m
    raise SystemExit(f"unknown model key {key!r}; have "
                     f"{[m['key'] for m in cfg['models']]}")
