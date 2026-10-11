from pathlib import Path
from typing import Any, Dict
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
