from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def save_run_metadata(output_dir: Path, runner) -> Path:
    """
    Save basic run metadata to output_dir/run_metadata.json, including
    environment info, registered models and metrics, and timestamp.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "models": list(getattr(runner, "models", {}).keys()),
        "metrics": [getattr(m, "name", type(m).__name__) for m in getattr(runner, "metrics", [])],
    }

    out_path = output_dir / "run_metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved run metadata to {out_path}")
    return out_path

