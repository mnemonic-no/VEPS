"""SQLite cache for DistilBERT CVSS/CWE predictions keyed by NVD lastModified.

A CVE's BERT output is a deterministic function of its description, and NVD
re-stamps `cve.lastModified` whenever the entry changes. Caching predictions
keyed by `(cve_id, last_modified)` lets daily enrichment runs skip BERT for
every CVE that hasn't been re-analyzed since the previous run.
"""

import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Tuple, Type


class PredictionCache:
    """SQLite-backed cache mapping CVE ID to (lastModified, predictions JSON).

    Use as a context manager so the connection closes and commits even on
    error. Writes are buffered into the open transaction; ``flush()`` commits
    without closing (useful between files).
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS predictions (
            cve_id TEXT PRIMARY KEY,
            last_modified TEXT NOT NULL,
            predictions TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "PredictionCache":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def open(self) -> None:
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        # WAL keeps reads fast while writes are buffered; synchronous=NORMAL
        # is safe with WAL and avoids an fsync per commit.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self._SCHEMA)

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.commit()
        self._conn.close()
        self._conn = None

    def flush(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def get(self, cve_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return (last_modified, predictions) for ``cve_id`` or ``None``."""
        if self._conn is None:
            raise RuntimeError("PredictionCache is not open")
        row = self._conn.execute(
            "SELECT last_modified, predictions FROM predictions WHERE cve_id = ?",
            (cve_id,),
        ).fetchone()
        if row is None:
            return None
        last_modified, payload = row
        return last_modified, json.loads(payload)

    def put(
        self,
        cve_id: str,
        last_modified: str,
        predictions: Dict[str, Any],
    ) -> None:
        if self._conn is None:
            raise RuntimeError("PredictionCache is not open")
        self._conn.execute(
            "INSERT OR REPLACE INTO predictions (cve_id, last_modified, predictions) "
            "VALUES (?, ?, ?)",
            (cve_id, last_modified, json.dumps(predictions)),
        )

    def size(self) -> int:
        if self._conn is None:
            raise RuntimeError("PredictionCache is not open")
        (count,) = self._conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()
        return count

    def clear(self) -> None:
        if self._conn is None:
            raise RuntimeError("PredictionCache is not open")
        self._conn.execute("DELETE FROM predictions")
        self._conn.commit()
