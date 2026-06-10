"""Audit logging utilities for API prediction events."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "reports" / "api_audit_log.jsonl"


def write_audit_log(record: Dict[str, Any]) -> None:
    """Append a prediction record to an audit log file in JSON Lines format."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
