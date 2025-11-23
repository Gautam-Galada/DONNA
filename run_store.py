import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Any, Dict

# You can change this if you want a different DB path
DEFAULT_DB_PATH = Path("donna2.db")


# ---------------------------------------------------------------------------
#  DDL: schema with dataset_id + seed_id
# ---------------------------------------------------------------------------

DDL_STATEMENTS = [
    # Keep referential integrity on
    "PRAGMA foreign_keys = ON;",

    # Datasets registry
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT    NOT NULL,
        version      TEXT,
        split_hash   TEXT    NOT NULL,
        note         TEXT
    );
    """,

    # Seed bank: single place where seeds live
    """
    CREATE TABLE IF NOT EXISTS seeds (
        seed_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        seed_value       INTEGER NOT NULL,
        numpy_rng_state  BLOB,
        torch_rng_state  BLOB,
        note             TEXT
    );
    """,

    # Runs table – NOTE: uses dataset_id + seed_id (no raw dataset/seed)
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id        TEXT    PRIMARY KEY,
        created_at    TEXT    NOT NULL,         -- ISO 8601 string
        dataset_id    INTEGER NOT NULL,         -- FK → datasets(dataset_id)
        model_name    TEXT    NOT NULL,
        lr            REAL    NOT NULL,
        batch_size    INTEGER NOT NULL,
        epochs        INTEGER NOT NULL,
        hidden        INTEGER,                  -- e.g. hidden size for MLP
        seed_id       INTEGER NOT NULL,         -- FK → seeds(seed_id)
        device        TEXT    NOT NULL,         -- cpu|cuda|mps|auto
        status        TEXT    NOT NULL,         -- created|running|done|failed
        best_val_acc  REAL,
        train_samples INTEGER,
        val_samples   INTEGER,
        notes         TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
        FOREIGN KEY (seed_id)    REFERENCES seeds(seed_id)
    );
    """,

    # Per-epoch metrics; latency_ms is optional (latency not a big constraint)
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT    NOT NULL,
        epoch       INTEGER NOT NULL,
        train_loss  REAL,
        val_loss    REAL,
        train_acc   REAL,
        val_acc     REAL,
        latency_ms  REAL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    """,

    # Artifacts (plots, checkpoints, reports, etc.)
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id    TEXT    NOT NULL,
        type      TEXT    NOT NULL,   -- ckpt|plot|report|confmat|other
        path      TEXT    NOT NULL,
        meta_json TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    """,
]


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------

def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Return a sqlite3 connection with foreign keys enabled.
    Lightweight, no ORM.
    """
    path = Path(db_path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(
    db_path: Path | str = DEFAULT_DB_PATH,
    extra_ddl: Optional[Iterable[str]] = None,
) -> sqlite3.Connection:
    """
    Create/upgrade the DB schema.

    This ensures:
    - Run schema uses dataset_id instead of dataset name
    - Runs link to seed_id instead of raw seed value
    """
    conn = connect(db_path)
    cur = conn.cursor()

    for stmt in DDL_STATEMENTS:
        cur.executescript(stmt)

    if extra_ddl:
        for stmt in extra_ddl:
            cur.executescript(stmt)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
#  RunStore: tiny wrapper to keep Donna code clean
# ---------------------------------------------------------------------------

class RunStore:
    """
    Lightweight helper around the SQLite schema.

    * Prioritizes simple workflow over fancy abstractions.
    * Uses dataset_id / seed_id (no raw strings in runs).
    * Latency is not a core concern; we just *store* latency_ms if you send it.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.conn = init_db(self.db_path)

    # -- internal helpers -----------------------------------------------------

    def _one(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.execute(query, params)
        return cur.fetchone()

    def _many(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.execute(query, params)
        return cur.fetchall()

    # -- dataset & seed registry ---------------------------------------------

    def get_or_create_dataset(
        self,
        name: str,
        version: str | None,
        split_hash: str,
        note: str | None = None,
    ) -> int:
        """
        Return dataset_id for (name, version, split_hash), creating if needed.
        """
        row = self._one(
            """
            SELECT dataset_id FROM datasets
            WHERE name = ?
              AND IFNULL(version,'') = IFNULL(?, '')
              AND split_hash = ?
            """,
            (name, version, split_hash),
        )
        if row:
            return int(row["dataset_id"])

        cur = self.conn.execute(
            """
            INSERT INTO datasets (name, version, split_hash, note)
            VALUES (?, ?, ?, ?)
            """,
            (name, version, split_hash, note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_or_create_seed(
        self,
        seed_value: int,
        numpy_rng_state: bytes | None = None,
        torch_rng_state: bytes | None = None,
        note: str | None = None,
    ) -> int:
        """
        Return seed_id for a given integer seed, creating if needed.
        """
        row = self._one(
            "SELECT seed_id FROM seeds WHERE seed_value = ?",
            (seed_value,),
        )
        if row:
            return int(row["seed_id"])

        cur = self.conn.execute(
            """
            INSERT INTO seeds (seed_value, numpy_rng_state, torch_rng_state, note)
            VALUES (?, ?, ?, ?)
            """,
            (seed_value, numpy_rng_state, torch_rng_state, note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # -- run lifecycle --------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        created_at: str,
        dataset_id: int,
        seed_id: int,
        config: Dict[str, Any],
        device: str,
        status: str = "created",
        notes: str | None = None,
    ) -> None:
        """
        Insert a new run row.

        config is expected to contain:
        - model_name
        - learning_rate
        - batch_size
        - num_epochs
        - hidden_size (optional)
        """
        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, created_at, dataset_id, model_name,
                lr, batch_size, epochs, hidden,
                seed_id, device, status,
                best_val_acc, train_samples, val_samples, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
            """,
            (
                run_id,
                created_at,
                dataset_id,
                config.get("model_name", "SimpleNet"),
                float(config.get("learning_rate")),
                int(config.get("batch_size")),
                int(config.get("num_epochs")),
                int(config.get("hidden_size", 0)) or None,
                seed_id,
                device,
                status,
                notes,
            ),
        )
        self.conn.commit()

    def update_run_status(
        self,
        run_id: str,
        status: str,
        best_val_acc: float | None = None,
        train_samples: int | None = None,
        val_samples: int | None = None,
    ) -> None:
        """
        Mark run as running/done/failed and optionally update summary stats.
        """
        self.conn.execute(
            """
            UPDATE runs
               SET status = ?,
                   best_val_acc = COALESCE(?, best_val_acc),
                   train_samples = COALESCE(?, train_samples),
                   val_samples = COALESCE(?, val_samples)
             WHERE run_id = ?
            """,
            (status, best_val_acc, train_samples, val_samples, run_id),
        )
        self.conn.commit()

    # -- metrics & artifacts --------------------------------------------------

    def log_epoch(
        self,
        run_id: str,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        train_acc: float | None = None,
        val_acc: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """
        Log one epoch’s metrics. latency_ms is optional.
        """
        self.conn.execute(
            """
            INSERT INTO metrics (
                run_id, epoch, train_loss, val_loss,
                train_acc, val_acc, latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, epoch, train_loss, val_loss, train_acc, val_acc, latency_ms),
        )
        self.conn.commit()

    def add_artifact(
        self,
        run_id: str,
        type_: str,
        path: str,
        meta_json: str | None = None,
    ) -> None:
        """
        Attach an artifact to a run (e.g. loss plot, checkpoint).
        """
        self.conn.execute(
            """
            INSERT INTO artifacts (run_id, type, path, meta_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, type_, path, meta_json),
        )
        self.conn.commit()

    # -- simple query helpers -------------------------------------------------

    def get_run_summary(self, run_id: str) -> Optional[dict]:
        """
        Return a single run row as a dict, or None if not found.
        """
        row = self._one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not row:
            return None
        return dict(row)

    def list_runs(self, limit: int = 50) -> list[dict]:
        """
        Return latest runs ordered by created_at DESC.
        """
        rows = self._many(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]


# Optional quick test: `python run_store.py`
if __name__ == "__main__":
    store = RunStore()
    print(f"DB initialized at {DEFAULT_DB_PATH.resolve()}")
    print("Existing runs:", store.list_runs(5))
