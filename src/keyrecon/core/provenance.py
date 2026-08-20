from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import pandas
        info["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import spacy
        info["spacy"] = spacy.__version__
    except Exception:
        info["spacy"] = None
    return info

def write_manifest(path: str | Path, payload: dict) -> None:
    out = dict(payload)
    out.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    out.setdefault("environment", environment_info())
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
