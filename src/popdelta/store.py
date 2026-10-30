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

    @property
    def store_metadata_path(self) -> Path:
        return self.path / "store.json"

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
        store.store_metadata_path.write_text(
            cls._dump_json(
                {
                    "format": 1,
                    "tables": {name: {"key": keys[name]} for name in _TABLE_NAMES},
                }
            )
        )
        initial_delta_path = store.deltas_path / "1_initial_commit"
        initial_delta_path.mkdir()
        state: dict[str, Any] = {}
        for name in _TABLE_NAMES:
            rel_path = f"1_initial_commit/{name}.parquet"
            non_key_cols = [str(c) for c in population.table(name).columns if c not in keys[name]]
            state[name] = {
                "index": rel_path,
                "files": {rel_path: non_key_cols} if non_key_cols else {},
            }
        (initial_delta_path / "manifest.json").write_text(
            cls._dump_json(
                {
                    "name": "Initial commit",
                    "parents": [],
                    "tables": {name: [] for name in _TABLE_NAMES},
                    "state": state,
                }
            )
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
        parent_name = deltas[idx - 1]
        parent_state = self._load_manifest_state(parent_name)
        population = self._read_from_state(parent_state)
        tables = {table_name: population.table(table_name).copy() for table_name in _TABLE_NAMES}
        keys_by_table = self._table_keys()
        seen_keys = self._collect_historical_keys(deltas[:idx], keys_by_table, manifest["tables"])

        for table_name, operations in manifest["tables"].items():
            if table_name not in tables:
                raise ValueError(f"Unknown population table: {table_name}")
            tables[table_name] = self._apply_operations(
                tables[table_name], operations, delta_path, keys_by_table[table_name], seen_keys.get(table_name)
            )

        updated_population = Population(**tables)
        manifest["parents"] = [parent_name]
        manifest["state"] = self._build_next_state(
            name, delta_path, parent_name, parent_state, keys_by_table, manifest["tables"], updated_population
        )
        manifest_path.write_text(self._dump_json(manifest))
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

    def create_delta(self, name: str, *, metadata: dict[str, Any] | None = None) -> Path:
        deltas = self.delta_names()
        if deltas and not self._has_state(deltas[-1]):
            raise ValueError(f"Delta {deltas[-1]!r} has not been committed; commit or discard it first")
        numbers = [self._delta_id(delta) for delta in deltas]
        next_id = max(numbers, default=0) + 1
        delta_path = self.deltas_path / f"{next_id}_{name}"
        manifest_text: str | None = None
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("Delta metadata must be a dictionary")
            manifest_text = self._dump_json({"name": delta_path.name, "metadata": metadata, "tables": {}})
        delta_path.mkdir()
        if manifest_text is not None:
            (delta_path / "manifest.json").write_text(manifest_text)
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
        manifest = (
            cast(dict[str, Any], json.loads(manifest_path.read_text()))
            if manifest_path.exists()
            else {"name": delta_names[-1], "tables": {}}
        )
        operations = cast(dict[str, list[dict[str, Any]]], manifest.setdefault("tables", {})).setdefault(table, [])
        key_cols = self._table_keys()[table]
        if table_data is not None:
            self._add_table_replacement(table, delta_path, operations, table_data, key_cols)
        else:
            self._add_row_operations(table, delta_path, operations, rows or {})
            active_cols = self._active_non_key_columns(table, delta_path, operations, key_cols) if columns else []
            self._add_column_operations(table, delta_path, operations, columns or {}, key_cols, active_cols)
        manifest_path.write_text(self._dump_json(manifest))

    def _active_non_key_columns(
        self,
        table: str,
        delta_path: Path,
        operations: list[dict[str, Any]],
        key_cols: list[str],
    ) -> list[str]:
        committed = [d for d in self.delta_names() if self._has_state(d)]
        table_state = self._resolve_table_state(self._load_manifest_state(committed[-1]), table)
        active = [col for file_cols in table_state["files"].values() for col in file_cols]
        for op in operations:
            action, val = next(iter(op.items()))
            if action == "replace_table":
                active = [str(c) for c in pd.read_parquet(delta_path / val).columns if c not in key_cols]
            elif action == "delete_columns":
                active = [c for c in active if c not in val]
            elif action in {"replace_columns", "add_columns"}:
                cols = val["columns"]
                active = [*(c for c in active if c not in cols), *cols]
        return active

    @staticmethod
    def _add_table_replacement(
        table: str,
        delta_path: Path,
        operations: list[dict[str, Any]],
        table_data: pd.DataFrame,
        key_cols: list[str],
    ) -> None:
        if not isinstance(table_data, pd.DataFrame):
            raise TypeError("Replacement table must be a pandas DataFrame")
        # TODO: Record commit-time table effect summaries (rows/columns added, deleted, changed).
        normalized = PopulationStore._normalize_keyed(table_data, key_cols)
        file_name = f"{table}/replace_table_{len(operations)}.parquet"
        (delta_path / table).mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(delta_path / file_name, index=False)
        operations.append({"replace_table": file_name})

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
        key_cols: list[str],
        active_cols: list[str],
    ) -> None:
        for action, values in columns.items():
            if action == "delete":
                if not isinstance(values, list):
                    raise TypeError("Deleted columns must be a list")
                PopulationStore._reject_key_columns(values, key_cols)
                active_cols[:] = [c for c in active_cols if c not in values]
                operations.append({"delete_columns": values})
                continue
            val_cols, frame = PopulationStore._prepare_column_input(action, values, key_cols, active_cols)
            operation = "replace_columns" if action == "replace" else "add_columns"
            stem = f"{action}_{val_cols[0]}" if len(val_cols) == 1 else action
            file_name = f"{table}/{stem}_{len(operations)}.parquet"
            (delta_path / table).mkdir(parents=True, exist_ok=True)
            frame.to_parquet(delta_path / file_name, index=False)
            operations.append({operation: {"file": file_name, "columns": val_cols}})

    @staticmethod
    def _prepare_column_input(
        action: str,
        values: Any,
        key_cols: list[str],
        active_cols: list[str],
    ) -> tuple[list[str], pd.DataFrame]:
        if action == "replace":
            if isinstance(values, pd.Series):
                val_cols, frame = [str(values.name)], values.to_frame()
            elif isinstance(values, pd.DataFrame):
                val_cols, frame = PopulationStore._extract_column_frame(values, key_cols)
            else:
                raise TypeError("Replacement column must be a pandas Series or DataFrame")
        elif action == "add":
            if not isinstance(values, pd.DataFrame):
                raise TypeError("Added columns must be a pandas DataFrame")
            val_cols, frame = PopulationStore._extract_column_frame(values, key_cols)
        else:
            raise ValueError(f"Unsupported column action: {action}")
        PopulationStore._reject_key_columns(val_cols, key_cols)
        PopulationStore._validate_column_op(val_cols, active_cols, is_replace=(action == "replace"))
        if action == "add":
            active_cols.extend(val_cols)
        return val_cols, frame

    @staticmethod
    def _validate_column_op(val_cols: list[str], existing_cols: list[str] | pd.Index, *, is_replace: bool) -> None:
        if is_replace:
            if missing := [c for c in val_cols if c not in existing_cols]:
                raise ValueError(f"Cannot replace unknown column(s) {missing}")
        elif existing := [c for c in val_cols if c in existing_cols]:
            raise ValueError(f"Cannot add existing column(s) {existing}")

    @staticmethod
    def _reject_key_columns(cols: list[str], key_cols: list[str]) -> None:
        if hit := set(cols) & set(key_cols):
            raise ValueError(f"Key column(s) {sorted(hit)} cannot be modified with column operations; use table_data")

    @staticmethod
    def _extract_column_frame(values: pd.DataFrame, key_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
        if values.columns.empty:
            raise ValueError("Column DataFrame must contain at least one value column")
        present_keys = [k for k in key_cols if k in values.columns]
        val_cols = [str(c) for c in values.columns if c not in key_cols]
        if not val_cols:
            PopulationStore._reject_key_columns([str(c) for c in values.columns], key_cols)
        if present_keys:
            if len(present_keys) < len(key_cols):
                missing = [k for k in key_cols if k not in values.columns]
                raise ValueError(f"Missing key column(s): {missing}")
            return val_cols, PopulationStore._normalize_keyed(values[[*key_cols, *val_cols]], key_cols)
        if len(val_cols) > 1:
            raise ValueError(f"Multi-column DataFrames must include key column(s) {key_cols}")
        return val_cols, values

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
    def _iter_column_op_files(action: str, value: Any) -> list[tuple[list[str] | None, str]]:
        if isinstance(value, dict) and isinstance(value.get("file"), str) and isinstance(value.get("columns"), list):
            return [([str(c) for c in value["columns"]], value["file"])]
        if action == "replace_columns" and isinstance(value, dict):
            return [([str(col)], str(file_name)) for col, file_name in value.items()]
        if action == "add_columns" and isinstance(value, str):
            return [(None, value)]
        raise ValueError(f"Invalid {action} specification: {value!r}")

    @staticmethod
    def _apply_operations(
        frame: pd.DataFrame,
        operations: list[dict[str, Any]],
        delta_path: Path,
        key_cols: list[str],
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if seen_keys is None:
            seen_keys = frame[key_cols]
        for operation in operations:
            if len(operation) != 1:
                raise ValueError(f"Each operation must contain exactly one action: {operation}")
            action, value = next(iter(operation.items()))
            if action == "replace_table":
                frame = PopulationStore._replace_table(frame, delta_path / value, key_cols, seen_keys)
                seen_keys = pd.concat([seen_keys, frame[key_cols]], ignore_index=True).drop_duplicates()
            elif action == "add_rows":
                frame = PopulationStore._add_rows(frame, delta_path / value, key_cols, seen_keys)
                seen_keys = pd.concat([seen_keys, frame[key_cols]], ignore_index=True).drop_duplicates()
            elif action == "delete_rows":
                frame = PopulationStore._delete_rows(frame, delta_path / value, key_cols)
            elif action in {"replace_columns", "add_columns"}:
                for expected_cols, file_name in PopulationStore._iter_column_op_files(action, value):
                    frame = PopulationStore._apply_column_file(
                        frame,
                        delta_path / file_name,
                        key_cols,
                        expected_cols=expected_cols,
                        is_replace=(action == "replace_columns"),
                    )
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
        key_cols: list[str],
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        new_frame = PopulationStore._normalize_keyed(pd.read_parquet(path), key_cols)
        if seen_keys is not None and not seen_keys.empty:
            retired = seen_keys.merge(frame[key_cols], on=key_cols, how="left", indicator=True)
            retired_keys = retired.loc[retired["_merge"] == "left_only", key_cols]
            if not new_frame[key_cols].merge(retired_keys, on=key_cols, how="inner").empty:
                raise ValueError(f"Replacement table reuses previously deleted keys in {key_cols}")
        return new_frame

    @staticmethod
    def _add_rows(
        frame: pd.DataFrame,
        path: Path,
        key_cols: list[str],
        seen_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        added = pd.read_parquet(path)
        if seen_keys is not None and not seen_keys.empty:
            if not added[key_cols].merge(seen_keys, on=key_cols, how="inner").empty:
                raise ValueError(f"Added rows reuse existing or previously deleted keys in {key_cols}")
        combined = pd.concat([frame, added], ignore_index=True)
        return PopulationStore._normalize_keyed(combined, key_cols)

    @staticmethod
    def _delete_rows(frame: pd.DataFrame, path: Path, key_cols: list[str]) -> pd.DataFrame:
        deleted_rows = pd.read_parquet(path)
        if not set(deleted_rows.columns).issubset(frame.columns):
            raise ValueError("Deleted rows must contain columns from the table")
        merged = frame.merge(deleted_rows.drop_duplicates(), on=list(deleted_rows.columns), how="left", indicator=True)
        result = merged.loc[merged["_merge"] == "left_only", frame.columns].reset_index(drop=True)
        return PopulationStore._normalize_keyed(result, key_cols)

    @staticmethod
    def _apply_column_file(
        frame: pd.DataFrame,
        path: Path,
        key_cols: list[str],
        *,
        expected_cols: list[str] | None,
        is_replace: bool,
    ) -> pd.DataFrame:
        col_df = pd.read_parquet(path)
        is_keyed = set(key_cols).issubset(col_df.columns) and len(col_df.columns) > len(key_cols)
        val_cols = [str(c) for c in col_df.columns if not (is_keyed and c in key_cols)]
        cols_ref = expected_cols or val_cols
        label: Any = cols_ref[0] if len(cols_ref) == 1 else cols_ref
        if expected_cols is not None and val_cols != expected_cols:
            raise ValueError(f"Replacement file for {label!r} must contain only that column")
        PopulationStore._reject_key_columns(val_cols, key_cols)
        PopulationStore._validate_column_op(val_cols, frame.columns, is_replace=is_replace)
        if len(col_df) != len(frame) or not val_cols:
            if is_replace or is_keyed:
                prefix = "Column" if is_keyed else "Replacement column"
                raise ValueError(f"{prefix} {label!r} has {len(col_df)} values; expected {len(frame)}")
            raise ValueError("Added column must contain exactly one value for each current row")
        if is_keyed:
            normalized_col = PopulationStore._normalize_keyed(col_df[[*key_cols, *val_cols]], key_cols)
            merged = frame[key_cols].merge(normalized_col, on=key_cols, how="left", indicator=True)
            if (merged["_merge"] != "both").any():
                raise ValueError(f"Column {label!r} keys do not match current table keys")
            source_df = merged
        else:
            source_df = col_df
        for col in val_cols:
            if col in frame.columns:
                frame = frame.drop(columns=[col])
            frame[col] = source_df[col].array
        return frame

    def _build_next_state(
        self,
        delta_name: str,
        delta_path: Path,
        parent_name: str,
        parent_state: dict[str, Any],
        keys_by_table: dict[str, list[str]],
        operations_by_table: dict[str, list[dict[str, Any]]],
        updated_population: Population,
    ) -> dict[str, Any]:
        next_state: dict[str, Any] = {}
        for table_name in _TABLE_NAMES:
            operations = operations_by_table.get(table_name, [])
            if not operations:
                parent_entry = parent_state[table_name]
                next_state[table_name] = parent_entry if isinstance(parent_entry, str) else parent_name
                continue

            table_state = copy.deepcopy(self._resolve_table_state(parent_state, table_name))
            key_cols = keys_by_table[table_name]
            updated_frame = updated_population.table(table_name)
            non_key_cols = [str(c) for c in updated_frame.columns if c not in key_cols]
            if len(operations) == 1 and "replace_table" in operations[0]:
                rel_state = f"{delta_name}/{operations[0]['replace_table']}"
                next_state[table_name] = {
                    "index": rel_state,
                    "files": {rel_state: non_key_cols} if non_key_cols else {},
                }
                continue

            op_kinds = {next(iter(op)) for op in operations}
            if "replace_table" in op_kinds or "add_rows" in op_kinds:
                state_file = f"{table_name}/state.parquet"
                (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                updated_frame.to_parquet(delta_path / state_file, index=False)
                rel_state = f"{delta_name}/{state_file}"
                next_state[table_name] = {
                    "index": rel_state,
                    "files": {rel_state: non_key_cols} if non_key_cols else {},
                }
                continue

            if "delete_rows" in op_kinds:
                index_file = f"{table_name}/index.parquet"
                (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                updated_frame[key_cols].to_parquet(delta_path / index_file, index=False)
                table_state["index"] = f"{delta_name}/{index_file}"

            self._update_column_pointers(
                delta_name, delta_path, table_name, operations, table_state, key_cols, updated_frame
            )
            next_state[table_name] = table_state

        return next_state

    @staticmethod
    def _remove_columns_from_files(files: dict[str, list[str]], cols: list[str]) -> None:
        drop = set(cols)
        for file_path in list(files):
            remaining = [c for c in files[file_path] if c not in drop]
            if remaining:
                files[file_path] = remaining
            else:
                del files[file_path]

    @staticmethod
    def _assign_columns_to_file(files: dict[str, list[str]], rel_path: str, cols: list[str]) -> None:
        PopulationStore._remove_columns_from_files(files, cols)
        files.setdefault(rel_path, []).extend(cols)

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
        files: dict[str, list[str]] = table_state["files"]
        for op_idx, op in enumerate(operations):
            action, value = next(iter(op.items()))
            if action == "delete_columns":
                PopulationStore._remove_columns_from_files(files, list(value))
                continue
            if action not in {"replace_columns", "add_columns"}:
                continue
            for expected_cols, rel_op_path in PopulationStore._iter_column_op_files(action, value):
                op_df = pd.read_parquet(delta_path / rel_op_path)
                is_keyed = set(key_cols).issubset(op_df.columns) and len(op_df.columns) > len(key_cols)
                cols = expected_cols or [str(c) for c in op_df.columns if c not in key_cols]
                if is_keyed:
                    target_rel = f"{delta_name}/{rel_op_path}"
                else:
                    state_col_rel = f"{table_name}/col_{op_idx}.parquet"
                    (delta_path / table_name).mkdir(parents=True, exist_ok=True)
                    live_cols = [c for c in cols if c in updated_frame.columns]
                    if live_cols:
                        updated_frame[[*key_cols, *live_cols]].to_parquet(delta_path / state_col_rel, index=False)
                    target_rel = f"{delta_name}/{state_col_rel}"
                PopulationStore._assign_columns_to_file(files, target_rel, cols)

    def _read_from_state(
        self,
        state: dict[str, Any],
        *,
        columns: dict[str, list[str]] | None = None,
    ) -> Population:
        if columns is not None and (unknown := set(columns) - set(_TABLE_NAMES)):
            raise ValueError(f"Unknown population table(s): {sorted(unknown)}")

        keys_by_table = self._table_keys()
        tables: dict[str, pd.DataFrame] = {}
        for table_name in _TABLE_NAMES:
            table_state = self._resolve_table_state(state, table_name)
            key_cols = keys_by_table[table_name]
            files_map: dict[str, list[str]] = table_state["files"]
            non_key_cols = [col for file_cols in files_map.values() for col in file_cols]
            available_cols = [*key_cols, *non_key_cols]
            req_cols = list(columns[table_name]) if columns and table_name in columns else available_cols
            if missing := [col for col in req_cols if col not in available_cols]:
                raise ValueError(f"Unknown column(s) {missing} for table {table_name!r}")
            target_cols = [*(k for k in key_cols if k not in req_cols), *req_cols]
            wanted_non_key = set(target_cols) - set(key_cols)

            base_df = pd.read_parquet(self.deltas_path / table_state["index"], columns=key_cols).copy()
            for rel_source, source_cols in files_map.items():
                needed = [col for col in source_cols if col in wanted_non_key]
                if not needed:
                    continue
                piece = pd.read_parquet(self.deltas_path / rel_source, columns=[*key_cols, *needed])
                base_df = base_df.merge(piece, on=key_cols, how="left", validate="one_to_one")

            tables[table_name] = base_df[target_cols].reset_index(drop=True)

        return Population(**tables)

    def _collect_historical_keys(
        self,
        prior_deltas: list[str],
        keys_by_table: dict[str, list[str]],
        operations_by_table: dict[str, list[dict[str, Any]]],
    ) -> dict[str, pd.DataFrame]:
        seen: dict[str, pd.DataFrame] = {}
        for name, ops in operations_by_table.items():
            if not any("add_rows" in op or "replace_table" in op for op in ops):
                continue
            key_cols = keys_by_table[name]
            index_paths = dict.fromkeys(
                self._resolve_table_state(self._load_manifest_state(d), name)["index"] for d in prior_deltas
            )
            frames = [pd.read_parquet(self.deltas_path / rel, columns=key_cols) for rel in index_paths]
            seen[name] = (
                pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=key_cols)
            )
        return seen

    def _table_keys(self) -> dict[str, list[str]]:
        if not self.store_metadata_path.exists():
            raise ValueError("Population store predates format 1 (missing store.json); recreate the store")
        meta = cast(dict[str, Any], json.loads(self.store_metadata_path.read_text()))
        if meta.get("format") != 1:
            raise ValueError(f"Unsupported population store format: {meta.get('format')!r}")
        return {name: list(meta["tables"][name]["key"]) for name in _TABLE_NAMES}

    def _resolve_table_state(self, state: dict[str, Any], table_name: str) -> dict[str, Any]:
        entry = state[table_name]
        if isinstance(entry, str):
            target_state = self._load_manifest_state(entry)
            resolved = target_state.get(table_name)
            if not isinstance(resolved, dict):
                raise ValueError(
                    f"State pointer for {table_name!r} in {entry!r} must resolve to a state block in one hop"
                )
            return resolved
        if isinstance(entry, dict):
            return entry
        raise ValueError(f"Invalid state entry for table {table_name!r}: {entry!r}")

    def _has_state(self, delta_name: str) -> bool:
        manifest_path = self.deltas_path / delta_name / "manifest.json"
        return manifest_path.exists() and "state" in json.loads(manifest_path.read_text())

    def _load_manifest_state(self, delta_name: str) -> dict[str, Any]:
        if not self._has_state(delta_name):
            raise ValueError(f"Delta {delta_name!r} has not been committed")
        manifest = cast(dict[str, Any], json.loads((self.deltas_path / delta_name / "manifest.json").read_text()))
        return cast(dict[str, Any], manifest["state"])

    def _resolve_delta(self, upto: str | int | None) -> str:
        deltas = self.delta_names()
        if upto is None:
            return [d for d in deltas if self._has_state(d)][-1]
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
        ordered_cols = [*key_cols, *(c for c in frame.columns if c not in key_cols)]
        return frame[ordered_cols].sort_values(key_cols, kind="mergesort").reset_index(drop=True)

    @staticmethod
    def _dump_json(data: dict[str, Any]) -> str:
        raw = json.dumps(data, indent=2)
        compact = re.sub(
            r"\[(?:\s*\"(?:[^\"\\]|\\.)*\"\s*,?)*\s*\]",
            lambda m: json.dumps(json.loads(m.group(0))),
            raw,
        )
        return compact + "\n"

    def _write_population(self, population: Population) -> None:
        for name in _TABLE_NAMES:
            population.table(name).to_parquet(self.current_state_path / f"{name}.parquet", index=False)

    @staticmethod
    def _delta_id(name: str) -> int:
        match = _DELTA_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Delta directory must be named '<number>_<name>': {name}")
        return int(match.group(1))
