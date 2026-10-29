from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from popdelta.population import Population

_TABLE_NAMES = ("households", "persons", "vehicles")
_DEFAULT_TABLE_KEYS: dict[str, list[str]] = {
    "households": ["household_id"],
    "persons": ["person_id"],
    "vehicles": ["vehicle_id"],
}
_DELTA_NAME = re.compile(r"^(\d+)_([^/]+)$")


@dataclass(frozen=True)
class PopulationStore:
    """On-disk population state and its ordered, visible deltas."""

    path: Path

    @property
    def current_state_path(self) -> Path:
        return self.path / "current_state"

    @property
    def deltas_path(self) -> Path:
        return self.path / "deltas"

    @classmethod
    def create(
        cls,
        path: Path,
        population: Population,
        table_keys: dict[str, str | tuple[str, ...] | list[str]] | None = None,
    ) -> PopulationStore:
        store = cls(path)
        if store.path.exists():
            raise FileExistsError(f"Population store already exists: {store.path}")
        raw_keys = {**_DEFAULT_TABLE_KEYS, **(table_keys or {})}
        keys = {name: [v] if isinstance(v := raw_keys[name], str) else list(v) for name in _TABLE_NAMES}
        population = Population(
            **{name: cls._normalize_keyed(population.table(name), keys[name]) for name in _TABLE_NAMES}
        )
        store.current_state_path.mkdir(parents=True)
        store.deltas_path.mkdir()
        initial_delta_path = store.deltas_path / "1_initial_commit"
        initial_delta_path.mkdir()
        state = {
            name: {
                "key": keys[name],
                "index": f"1_initial_commit/{name}.parquet",
                "columns": dict.fromkeys(population.table(name).columns, f"1_initial_commit/{name}.parquet"),
            }
            for name in _TABLE_NAMES
        }
        (initial_delta_path / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "Initial commit",
                    "parents": [],
                    "tables": {name: [] for name in _TABLE_NAMES},
                    "state": state,
                },
                indent=2,
            )
            + "\n"
        )
        store._write_population(population)
        for name in _TABLE_NAMES:
            population.table(name).to_parquet(initial_delta_path / f"{name}.parquet", index=False)
        return store

    def read(self) -> Population:
        tables = {
            name: pd.read_parquet(self.current_state_path / f"{name}.parquet")
            for name in _TABLE_NAMES
        }
        return Population(**tables)

    def play_commits(
        self,
        upto: str | int | None = None,
        *,
        columns: dict[str, list[str]] | None = None,
    ) -> Population:
        target_delta = self._resolve_delta(upto)
        manifest = self._load_manifest_state(target_delta)
        return self._read_from_state(manifest, columns=columns)

    def apply_delta(self, name: str) -> None:
        delta_path = self.deltas_path / name
        manifest_path = delta_path / "manifest.json"
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
        if "state" in manifest:
            self._write_population(self._read_from_state(manifest["state"]))
            return
        deltas = self.delta_names()
        idx = deltas.index(name)
        parents = [deltas[idx - 1]] if idx > 0 else []
        parent_state = self._load_manifest_state(parents[0] if parents else name)
        population = self._read_from_state(parent_state) if parents else self.read()
        tables = {table_name: population.table(table_name).copy() for table_name in _TABLE_NAMES}
        seen_keys = self._collect_historical_keys(deltas[:idx], parent_state, manifest["tables"])

        for table_name, operations in manifest["tables"].items():
            if table_name not in tables:
                raise ValueError(f"Unknown population table: {table_name}")
            key_cols = list(parent_state[table_name]["key"])
            tables[table_name] = self._apply_operations(
                tables[table_name], operations, delta_path, key_cols, seen_keys.get(table_name)
            )

        updated_population = Population(**tables)
        manifest["parents"] = parents
        manifest["state"] = self._build_next_state(
            name, delta_path, parent_state, manifest["tables"], updated_population
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self._write_population(updated_population)

    def delta_names(self) -> list[str]:
        deltas = []
        for path in self.deltas_path.iterdir():
            if not path.is_dir():
                continue
            match = _DELTA_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Delta directory must be named '<number>_<name>': {path.name}")
            deltas.append((int(match.group(1)), path.name))
        numbers = [number for number, _ in deltas]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Delta directory IDs must be unique")
        return [name for _, name in sorted(deltas)]

    def create_delta(self, name: str) -> Path:
        deltas = self.delta_names()
        if deltas and not self._has_state(deltas[-1]):
            raise ValueError(f"Delta {deltas[-1]!r} has not been committed; commit or discard it first")
        numbers = [self._delta_id(delta) for delta in deltas]
        next_id = max(numbers, default=0) + 1
        delta_path = self.deltas_path / f"{next_id}_{name}"
        delta_path.mkdir()
        return delta_path

    def discard_current_delta(self) -> None:
        deltas = self.delta_names()
        if not deltas or self._has_state(deltas[-1]):
            raise ValueError("No uncommitted delta to discard")
        shutil.rmtree(self.deltas_path / deltas[-1])

    def add_to_current_delta(
        self,
        table: str,
        rows: dict[str, pd.DataFrame] | None = None,
        columns: dict[str, Any] | None = None,
        table_data: pd.DataFrame | None = None,
    ) -> None:
        if int(bool(rows)) + int(bool(columns)) + int(table_data is not None) > 1:
            raise ValueError("Row, column, and table replacement operations must be added to the delta separately")
        if table not in _TABLE_NAMES:
            raise ValueError(f"Unknown population table: {table}")
        delta_names = self.delta_names()
        if not delta_names:
            raise ValueError("No current delta exists")
        delta_path = self.deltas_path / delta_names[-1]
        manifest_path = delta_path / "manifest.json"
        if manifest_path.exists():
            manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
        else:
            manifest = {"name": delta_names[-1], "tables": {}}
        tables = cast(dict[str, list[dict[str, Any]]], manifest.setdefault("tables", {}))
        operations = tables.setdefault(table, [])
        key_cols = self._table_keys_for_latest(delta_names, table)
        if table_data is not None:
            if not isinstance(table_data, pd.DataFrame):
                raise TypeError("Replacement table must be a pandas DataFrame")
            normalized = self._normalize_keyed(table_data, key_cols)
            file_name = f"{table}/replace_table_{len(operations)}.parquet"
            (delta_path / table).mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(delta_path / file_name, index=False)
            operations.append({"replace_table": file_name})
        else:
            self._add_row_operations(table, delta_path, operations, rows or {})
            self._add_column_operations(table, delta_path, operations, columns or {}, key_cols)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    @staticmethod
    def _add_row_operations(
        table: str, delta_path: Path, operations: list[dict[str, Any]], rows: dict[str, pd.DataFrame]
    ) -> None:
        for action, frame in rows.items():
            if action not in {"add", "delete"}:
                raise ValueError(f"Unsupported row action: {action}")
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Delta rows must be pandas DataFrames")
            operation = "add_rows" if action == "add" else "delete_rows"
            file_name = f"{table}/{action}_rows_{len(operations)}.parquet"
            (delta_path / table).mkdir(parents=True, exist_ok=True)
            frame.to_parquet(delta_path / file_name, index=False)
            operations.append({operation: file_name})

    @staticmethod
    def _add_column_operations(
        table: str,
        delta_path: Path,
        operations: list[dict[str, Any]],
        columns: dict[str, Any],
        key_cols: list[str] | None = None,
    ) -> None:
        for action, values in columns.items():
            if action == "delete":
                if not isinstance(values, list):
                    raise TypeError("Deleted columns must be a list")
                PopulationStore._reject_key_columns(values, key_cols)
                operations.append({"delete_columns": values})
                continue
            if action == "replace":
                if isinstance(values, pd.Series):
                    column, frame = str(values.name), values.to_frame()
                elif isinstance(values, pd.DataFrame):
                    column, frame = PopulationStore._extract_column_frame(values, key_cols)
                else:
                    raise TypeError("Replacement column must be a pandas Series or DataFrame")
                operation_value: Any = {column: f"{table}/replace_{column}_{len(operations)}.parquet"}
                operation = "replace_columns"
                file_name = operation_value[column]
            elif action == "add":
                if not isinstance(values, pd.DataFrame):
                    raise TypeError("Added columns must be a one-column or keyed pandas DataFrame")
                column, frame = PopulationStore._extract_column_frame(values, key_cols)
                operation, file_name = "add_columns", f"{table}/add_{column}_{len(operations)}.parquet"
                operation_value = file_name
            else:
                raise ValueError(f"Unsupported column action: {action}")
            PopulationStore._reject_key_columns([column], key_cols)
            (delta_path / table).mkdir(parents=True, exist_ok=True)
            frame.to_parquet(delta_path / file_name, index=False)
            operations.append({operation: operation_value})

    @staticmethod
    def _reject_key_columns(cols: list[str], key_cols: list[str] | None) -> None:
        if key_cols and (hit := set(cols) & set(key_cols)):
            raise ValueError(f"Key column(s) {sorted(hit)} cannot be modified with column operations; use table_data")

    @staticmethod
    def _extract_column_frame(values: pd.DataFrame, key_cols: list[str] | None) -> tuple[str, pd.DataFrame]:
        if len(values.columns) == 1:
            return str(values.columns[0]), values
        if key_cols and set(key_cols).issubset(values.columns) and len(values.columns) == len(key_cols) + 1:
            col = next(str(c) for c in values.columns if c not in key_cols)
            return col, PopulationStore._normalize_keyed(values[[*key_cols, col]], key_cols)
        raise ValueError(f"Column DataFrame must contain 1 value column or key {key_cols} plus 1 value column")

    def commit_current_delta(self) -> Population:
        delta_names = self.delta_names()
        if not delta_names:
            raise ValueError("No current delta exists")
        self.apply_delta(delta_names[-1])
        return self.read()

    def apply_deltas(self) -> None:
        for name in self.delta_names():
            self.apply_delta(name)

    @staticmethod
    def _apply_operations(
        frame: pd.DataFrame,
        operations: list[dict[str, Any]],
        delta_path: Path,
        key_cols: list[str] | None = None,
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        for operation in operations:
            if len(operation) != 1:
                raise ValueError(f"Each operation must contain exactly one action: {operation}")
            action, value = next(iter(operation.items()))
            if action == "replace_table":
                frame = PopulationStore._replace_table(frame, delta_path / value, key_cols, seen_keys)
            elif action == "add_rows":
                frame = PopulationStore._add_rows(frame, delta_path / value, key_cols, seen_keys)
            elif action == "delete_rows":
                frame = PopulationStore._delete_rows(frame, delta_path / value, key_cols)
            elif action == "replace_columns":
                frame = PopulationStore._replace_columns(frame, value, delta_path, key_cols)
            elif action == "add_columns":
                frame = PopulationStore._apply_column_file(frame, delta_path / value, key_cols, expected_col=None)
            elif action == "delete_columns":
                PopulationStore._reject_key_columns(value, key_cols)
                frame = frame.drop(columns=value)
            else:
                raise ValueError(f"Unsupported delta action: {action}")

        return frame

    @staticmethod
    def _replace_table(
        frame: pd.DataFrame,
        path: Path,
        key_cols: list[str] | None = None,
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raw = pd.read_parquet(path)
        new_frame = PopulationStore._normalize_keyed(raw, key_cols) if key_cols else raw
        if key_cols and seen_keys is not None and not seen_keys.empty:
            retired = seen_keys.merge(frame[key_cols], on=key_cols, how="left", indicator=True)
            retired_keys = retired.loc[retired["_merge"] == "left_only", key_cols]
            if not new_frame[key_cols].merge(retired_keys, on=key_cols, how="inner").empty:
                raise ValueError(f"Replacement table reuses previously deleted keys in {key_cols}")
        return new_frame

    @staticmethod
    def _add_rows(
        frame: pd.DataFrame,
        path: Path,
        key_cols: list[str] | None = None,
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        added = pd.read_parquet(path)
        if key_cols and seen_keys is not None and not seen_keys.empty:
            if not added[key_cols].merge(seen_keys, on=key_cols, how="inner").empty:
                raise ValueError(f"Added rows reuse existing or previously deleted keys in {key_cols}")
        combined = pd.concat([frame, added], ignore_index=True)
        return PopulationStore._normalize_keyed(combined, key_cols) if key_cols else combined

    @staticmethod
    def _delete_rows(frame: pd.DataFrame, path: Path, key_cols: list[str] | None = None) -> pd.DataFrame:
        deleted_rows = pd.read_parquet(path)
        if not set(deleted_rows.columns).issubset(frame.columns):
            raise ValueError("Deleted rows must contain columns from the table")
        merged = frame.merge(deleted_rows.drop_duplicates(), on=list(deleted_rows.columns), how="left", indicator=True)
        result = merged.loc[merged["_merge"] == "left_only", frame.columns].reset_index(drop=True)
        return PopulationStore._normalize_keyed(result, key_cols) if key_cols else result

    @staticmethod
    def _replace_columns(
        frame: pd.DataFrame, replacements: dict[str, str], delta_path: Path, key_cols: list[str] | None = None
    ) -> pd.DataFrame:
        for column, file_name in replacements.items():
            frame = PopulationStore._apply_column_file(frame, delta_path / file_name, key_cols, expected_col=column)
        return frame

    @staticmethod
    def _apply_column_file(
        frame: pd.DataFrame,
        path: Path,
        key_cols: list[str] | None,
        *,
        expected_col: str | None,
    ) -> pd.DataFrame:
        col_df = pd.read_parquet(path)
        if key_cols is not None and set(key_cols).issubset(col_df.columns) and len(col_df.columns) == len(key_cols) + 1:
            column = next(c for c in col_df.columns if c not in key_cols)
            if expected_col is not None and column != expected_col:
                raise ValueError(f"Replacement file for {expected_col!r} must contain only that column")
            if len(col_df) != len(frame):
                raise ValueError(f"Column {column!r} has {len(col_df)} values; expected {len(frame)}")
            normalized_col = PopulationStore._normalize_keyed(col_df[[*key_cols, column]], key_cols)
            merged = frame[key_cols].merge(normalized_col, on=key_cols, how="left", indicator=True)
            if (merged["_merge"] != "both").any():
                raise ValueError(f"Column {column!r} keys do not match current table keys")
            frame[column] = merged[column].array
        else:
            if expected_col is not None and list(col_df.columns) != [expected_col]:
                raise ValueError(f"Replacement file for {expected_col!r} must contain only that column")
            if len(col_df) != len(frame) or len(col_df.columns) != 1:
                if expected_col is not None:
                    raise ValueError(
                        f"Replacement column {expected_col!r} has {len(col_df)} values; expected {len(frame)}"
                    )
                raise ValueError("Added column must contain exactly one value for each current row")
            column = str(col_df.columns[0])
            PopulationStore._reject_key_columns([column], key_cols)
            frame[column] = col_df[column].array
        return frame

    @staticmethod
    def _build_next_state(
        delta_name: str,
        delta_path: Path,
        parent_state: dict[str, Any],
        operations_by_table: dict[str, list[dict[str, Any]]],
        updated_population: Population,
    ) -> dict[str, Any]:
        next_state = copy.deepcopy(parent_state)
        for table_name, operations in operations_by_table.items():
            if not operations:
                continue
            table_state = next_state[table_name]
            key_cols = list(table_state["key"])
            updated_frame = updated_population.table(table_name)
            if len(operations) == 1 and "replace_table" in operations[0]:
                rel_state = f"{delta_name}/{operations[0]['replace_table']}"
                table_state["index"] = rel_state
                table_state["columns"] = dict.fromkeys(updated_frame.columns, rel_state)
                continue

            op_kinds = {next(iter(op)) for op in operations}
            if "replace_table" in op_kinds or "add_rows" in op_kinds:
                state_file = f"{table_name}/state.parquet"
                (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                updated_frame.to_parquet(delta_path / state_file, index=False)
                rel_state = f"{delta_name}/{state_file}"
                table_state["index"] = rel_state
                table_state["columns"] = dict.fromkeys(updated_frame.columns, rel_state)
                continue

            if "delete_rows" in op_kinds:
                index_file = f"{table_name}/index.parquet"
                (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                updated_frame[key_cols].to_parquet(delta_path / index_file, index=False)
                rel_index = f"{delta_name}/{index_file}"
                table_state["index"] = rel_index
                for k_col in key_cols:
                    table_state["columns"][k_col] = rel_index

            PopulationStore._update_column_pointers(
                delta_name, delta_path, table_name, operations, table_state, key_cols, updated_frame
            )

        return next_state

    @staticmethod
    def _update_column_pointers(
        delta_name: str,
        delta_path: Path,
        table_name: str,
        operations: list[dict[str, Any]],
        table_state: dict[str, Any],
        key_cols: list[str],
        updated_frame: pd.DataFrame,
    ) -> None:
        for op in operations:
            action, value = next(iter(op.items()))
            if action == "delete_columns":
                for col in value:
                    table_state["columns"].pop(col, None)
                continue
            if action not in {"replace_columns", "add_columns"}:
                continue
            items = value.items() if action == "replace_columns" else [(None, value)]
            for col_hint, rel_op_path in items:
                op_df = pd.read_parquet(delta_path / rel_op_path)
                col = col_hint or next(str(c) for c in op_df.columns if c not in key_cols)
                if set(key_cols).issubset(op_df.columns) and len(op_df.columns) == len(key_cols) + 1:
                    table_state["columns"][col] = f"{delta_name}/{rel_op_path}"
                else:
                    state_col_rel = f"{table_name}/col_{col}.parquet"
                    (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                    updated_frame[[*key_cols, col]].to_parquet(delta_path / state_col_rel, index=False)
                    table_state["columns"][col] = f"{delta_name}/{state_col_rel}"

    def _read_from_state(
        self,
        state: dict[str, Any],
        *,
        columns: dict[str, list[str]] | None = None,
    ) -> Population:
        if columns is not None and (unknown := set(columns) - set(_TABLE_NAMES)):
            raise ValueError(f"Unknown population table(s): {sorted(unknown)}")

        tables: dict[str, pd.DataFrame] = {}
        for table_name in _TABLE_NAMES:
            table_state = state[table_name]
            key_cols: list[str] = list(table_state["key"])
            col_map: dict[str, str] = table_state["columns"]
            req_cols = list(columns[table_name]) if columns and table_name in columns else list(col_map.keys())
            if missing := [col for col in req_cols if col not in col_map]:
                raise ValueError(f"Unknown column(s) {missing} for table {table_name!r}")
            target_cols = [*(k for k in key_cols if k not in req_cols), *req_cols]

            base_df = pd.read_parquet(self.deltas_path / table_state["index"], columns=key_cols).copy()
            cols_by_source: dict[str, list[str]] = {}
            for col in target_cols:
                if col not in key_cols:
                    cols_by_source.setdefault(col_map[col], []).append(col)

            for rel_source, source_cols in cols_by_source.items():
                piece = pd.read_parquet(self.deltas_path / rel_source, columns=[*key_cols, *source_cols])
                base_df = base_df.merge(piece, on=key_cols, how="left", validate="one_to_one")

            tables[table_name] = base_df[target_cols].reset_index(drop=True)

        return Population(**tables)

    def _collect_historical_keys(
        self,
        prior_deltas: list[str],
        fallback_state: dict[str, Any],
        operations_by_table: dict[str, list[dict[str, Any]]],
    ) -> dict[str, pd.DataFrame]:
        seen: dict[str, pd.DataFrame] = {}
        for name, ops in operations_by_table.items():
            if not any("add_rows" in op or "replace_table" in op for op in ops):
                continue
            key_cols = list(fallback_state[name]["key"])
            index_paths = dict.fromkeys(self._load_manifest_state(d)[name]["index"] for d in prior_deltas)
            frames = [pd.read_parquet(self.deltas_path / rel, columns=key_cols) for rel in index_paths]
            seen[name] = (
                pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=key_cols)
            )
        return seen

    def _table_keys_for_latest(self, delta_names: list[str], table: str) -> list[str]:
        for delta in reversed(delta_names):
            manifest_path = self.deltas_path / delta / "manifest.json"
            if manifest_path.exists() and "state" in (data := json.loads(manifest_path.read_text())):
                return list(data["state"][table]["key"])
        return list(_DEFAULT_TABLE_KEYS[table])

    def _has_state(self, delta_name: str) -> bool:
        manifest_path = self.deltas_path / delta_name / "manifest.json"
        return manifest_path.exists() and "state" in json.loads(manifest_path.read_text())

    def _load_manifest_state(self, delta_name: str) -> dict[str, Any]:
        manifest_path = self.deltas_path / delta_name / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Delta {delta_name!r} has not been committed")
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
        if "state" not in manifest:
            if "tables" in manifest and self._has_state(self.delta_names()[0]):
                raise ValueError(f"Delta {delta_name!r} has not been committed")
            raise ValueError(f"Delta {delta_name!r} predates state manifests; recreate the store")
        return cast(dict[str, Any], manifest["state"])

    def _resolve_delta(self, upto: str | int | None) -> str:
        deltas = self.delta_names()
        if upto is None:
            committed = [d for d in deltas if self._has_state(d)]
            return committed[-1] if committed else deltas[-1]
        if upto in deltas:
            return str(upto)
        target = str(upto)
        matches = [
            d
            for d in deltas
            if d.split("_", 1)[1] == target or (target.isdigit() and self._delta_id(d) == int(target))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous delta reference {upto!r}: {matches}")
        raise ValueError(f"Unknown delta reference: {upto!r}")

    @staticmethod
    def _normalize_keyed(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
        if missing := [k for k in key_cols if k not in frame.columns]:
            raise ValueError(f"Missing key column(s): {missing}")
        if frame[key_cols].isna().any().any() or frame.duplicated(subset=key_cols).any():
            raise ValueError(f"Duplicate or null key values in {key_cols}")
        return frame.sort_values(key_cols, kind="mergesort").reset_index(drop=True)

    def _write_population(self, population: Population) -> None:
        for name in _TABLE_NAMES:
            population.table(name).to_parquet(self.current_state_path / f"{name}.parquet", index=False)

    @staticmethod
    def _delta_id(name: str) -> int:
        match = _DELTA_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Delta directory must be named '<number>_<name>': {name}")
        return int(match.group(1))
