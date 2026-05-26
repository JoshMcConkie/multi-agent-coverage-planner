"""SQLite-backed storage for sweep results.

Persists per-cell sweep metrics to `results/sweeps.db` so that plotting,
ad-hoc querying, and cross-run comparisons can happen independently of
the (potentially long) sweep itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = Path("results/sweeps.db")

_PROJECT_ROOT: Path | None = None


def _project_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            _PROJECT_ROOT = parent
            return _PROJECT_ROOT

    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def _resolve_db_path(db_path: Path) -> Path:
    """Resolve *db_path* relative to the project root when not absolute."""
    if db_path.is_absolute():
        return db_path
    return _project_root() / db_path


SWEEP_ROW_COLUMNS: tuple[str, ...] = (
    "agents",
    "steps",
    "chunksize",
    "split_ratio_to_seq_min",
    "split_ratio_to_seq_mean",
    "split_ratio_to_seq_max_runtime",
    "split_ratio_to_seq_mean_runtime",
    "split_ratio_to_baseline",
    "seq_ratio_to_baseline",
    "split_ratio_to_baseline_runtime",
    "seq_ratio_to_baseline_runtime",
)


SWEEP_RESULT_COLUMNS: tuple[str, ...] = (
    "agents",
    "steps",
    "chunksize",
    "start_row",
    "start_col",
    "method",
    "score",
    "runtime",
)


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sweeps (
        sweep_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        grid_size     INTEGER NOT NULL,
        max_agents    INTEGER NOT NULL,
        max_steps     INTEGER NOT NULL,
        solve_optimal INTEGER NOT NULL,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        notes         TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sweep_rows (
        sweep_id                        INTEGER NOT NULL
            REFERENCES sweeps(sweep_id) ON DELETE CASCADE,
        agents                          INTEGER NOT NULL,
        steps                           INTEGER NOT NULL,
        chunksize                       INTEGER NOT NULL,
        split_ratio_to_seq_min          REAL,
        split_ratio_to_seq_mean         REAL,
        split_ratio_to_seq_max_runtime  REAL,
        split_ratio_to_seq_mean_runtime REAL,
        split_ratio_to_baseline         REAL,
        seq_ratio_to_baseline           REAL,
        split_ratio_to_baseline_runtime REAL,
        seq_ratio_to_baseline_runtime   REAL,
        PRIMARY KEY (sweep_id, agents, steps, chunksize)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sweep_results (
        sweep_id   INTEGER NOT NULL
            REFERENCES sweeps(sweep_id) ON DELETE CASCADE,
        agents     INTEGER NOT NULL,
        steps      INTEGER NOT NULL,
        chunksize  INTEGER NOT NULL,
        start_row  INTEGER NOT NULL,
        start_col  INTEGER NOT NULL,
        method     TEXT NOT NULL,
        score      REAL NOT NULL,
        runtime    REAL NOT NULL,
        PRIMARY KEY
            (sweep_id, agents, steps, chunksize, start_row, start_col, method)
    )
    """,
)


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating parent directories as needed.

    Relative *db_path* values are resolved from the project root (the
    directory containing ``pyproject.toml``), not the process cwd, so
    notebooks and CLI entrypoints share the same database file.
    """
    db_path = _resolve_db_path(Path(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""
    with conn:
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)


def create_sweep(
    conn: sqlite3.Connection,
    *,
    name: str,
    grid_size: int,
    max_agents: int,
    max_steps: int,
    solve_optimal: bool,
    notes: str | None = None,
) -> int:
    """Insert a new sweep header row and return its `sweep_id`."""
    with conn:
        cur = conn.execute(
            """
            INSERT INTO sweeps
                (name, grid_size, max_agents, max_steps, solve_optimal, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                int(grid_size),
                int(max_agents),
                int(max_steps),
                1 if solve_optimal else 0,
                notes,
            ),
        )
        return int(cur.lastrowid)


def insert_sweep_rows(
    conn: sqlite3.Connection,
    sweep_id: int,
    rows: list[dict],
) -> None:
    """Bulk-insert per-cell metric rows for an existing sweep.

    Any column missing from a given row dict is stored as NULL, which
    keeps the schema stable whether `solve_optimal` was True or False.
    """
    if not rows:
        return

    placeholders = ", ".join(["?"] * (1 + len(SWEEP_ROW_COLUMNS)))
    column_list = ", ".join(("sweep_id", *SWEEP_ROW_COLUMNS))
    sql = (
        f"INSERT INTO sweep_rows ({column_list}) VALUES ({placeholders})"
    )

    payload = [
        (sweep_id, *(row.get(col) for col in SWEEP_ROW_COLUMNS))
        for row in rows
    ]

    with conn:
        conn.executemany(sql, payload)


def insert_sweep_results(
    conn: sqlite3.Connection,
    sweep_id: int,
    results: list[dict],
) -> None:
    """Bulk-insert per-method, per-starting-cell raw `(score, runtime)` rows.

    Each result must supply every column in `SWEEP_RESULT_COLUMNS`. Rows
    corresponding to `EmptyResult` should be filtered out by the caller.
    """
    if not results:
        return

    placeholders = ", ".join(["?"] * (1 + len(SWEEP_RESULT_COLUMNS)))
    column_list = ", ".join(("sweep_id", *SWEEP_RESULT_COLUMNS))
    sql = (
        f"INSERT INTO sweep_results ({column_list}) VALUES ({placeholders})"
    )

    payload = [
        (sweep_id, *(row[col] for col in SWEEP_RESULT_COLUMNS))
        for row in results
    ]

    with conn:
        conn.executemany(sql, payload)


def list_sweeps(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return every sweep header as a DataFrame, newest first."""
    return pd.read_sql_query(
        "SELECT * FROM sweeps ORDER BY sweep_id DESC",
        conn,
    )


def _resolve_sweep_meta(
    conn: sqlite3.Connection,
    sweep_id: int | None,
    name: str | None,
) -> dict:
    """Resolve a sweep header by id, latest-with-name, or latest-overall."""
    if sweep_id is not None:
        meta_row = conn.execute(
            "SELECT * FROM sweeps WHERE sweep_id = ?",
            (sweep_id,),
        ).fetchone()
    elif name is not None:
        meta_row = conn.execute(
            """
            SELECT * FROM sweeps
            WHERE name = ?
            ORDER BY sweep_id DESC
            LIMIT 1
            """,
            (name,),
        ).fetchone()
    else:
        meta_row = conn.execute(
            "SELECT * FROM sweeps ORDER BY sweep_id DESC LIMIT 1"
        ).fetchone()

    if meta_row is None:
        raise LookupError(
            f"No sweep found (sweep_id={sweep_id!r}, name={name!r})."
        )

    return dict(meta_row)


def load_sweep_df(
    conn: sqlite3.Connection,
    sweep_id: int | None = None,
    name: str | None = None,
) -> tuple[int, dict, pd.DataFrame]:
    """Load a sweep's metadata and aggregate rows from `sweep_rows`.

    Resolution order:
        1. Explicit `sweep_id` if provided.
        2. Latest sweep with the given `name` if provided.
        3. Latest sweep overall.

    Returns:
        (sweep_id, metadata_dict, rows_df)
    """
    meta = _resolve_sweep_meta(conn, sweep_id, name)
    resolved_id = int(meta["sweep_id"])

    rows_df = pd.read_sql_query(
        """
        SELECT agents, steps, chunksize,
               split_ratio_to_seq_min,
               split_ratio_to_seq_mean,
               split_ratio_to_seq_max_runtime,
               split_ratio_to_seq_mean_runtime,
               split_ratio_to_baseline,
               seq_ratio_to_baseline,
               split_ratio_to_baseline_runtime,
               seq_ratio_to_baseline_runtime
        FROM sweep_rows
        WHERE sweep_id = ?
        ORDER BY agents, steps, chunksize
        """,
        conn,
        params=(resolved_id,),
    )

    return resolved_id, meta, rows_df


def load_sweep_results(
    conn: sqlite3.Connection,
    sweep_id: int,
) -> pd.DataFrame:
    """Return all raw per-method results for a single sweep."""
    return pd.read_sql_query(
        """
        SELECT agents, steps, chunksize,
               start_row, start_col,
               method, score, runtime
        FROM sweep_results
        WHERE sweep_id = ?
        ORDER BY agents, steps, chunksize, start_row, start_col, method
        """,
        conn,
        params=(int(sweep_id),),
    )


def load_sweep_raw_df(
    conn: sqlite3.Connection,
    sweep_id: int | None = None,
    name: str | None = None,
) -> tuple[int, dict, pd.DataFrame]:
    """Load a sweep's metadata and raw per-method results.

    Mirrors `load_sweep_df` but pulls from `sweep_results` so plotting
    code never touches SQL directly.

    Returns:
        (sweep_id, metadata_dict, raw_df)
    """
    meta = _resolve_sweep_meta(conn, sweep_id, name)
    resolved_id = int(meta["sweep_id"])
    raw_df = load_sweep_results(conn, resolved_id)
    return resolved_id, meta, raw_df
