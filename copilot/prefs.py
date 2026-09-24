"""Load preferences.yaml. Read on every request so edits apply without a restart."""
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parent.parent / "preferences.yaml"


def load_prefs() -> dict:
    return yaml.safe_load(PATH.read_text())
