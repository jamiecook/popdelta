from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from popdelta.population import Population

_TABLE_NAMES = ("households", "persons", "vehicles")
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
    def create(cls, path: Path, population: Population) -> PopulationStore:
        store = cls(path)
        if store.path.exists():
            raise FileExistsError(f"Population store already exists: {store.path}")
        store.current_state_path.mkdir(parents=True)
        store.deltas_path.mkdir()
        initial_delta_path = store.deltas_path / "1_initial_commit"
        initial_delta_path.mkdir()
        (initial_delta_path / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "Initial commit",
                    "tables": {name: [] for name in _TABLE_NAMES},
                },
                indent=2,
            )
            + "\n"
        )
        store._write_population(population)
        for name in _TABLE_NAMES:
            getattr(population, name).to_parquet(initial_delta_path / f"{name}.parquet", index=False)
        return store

    def read(self) -> Population:
        tables = {
            name: pd.read_parquet(self.current_state_path / f"{name}.parquet")
            for name in _TABLE_NAMES
        }
        return Population(**tables)

    def apply_delta(self, name: str) -> None:
        delta_path = self.deltas_path / name
        manifest_path = delta_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        population = self.read()
        tables = {name: getattr(population, name).copy() for name in _TABLE_NAMES}

        for table_name, operations in manifest["tables"].items():
            if table_name not in tables:
                raise ValueError(f"Unknown population table: {table_name}")
            tables[table_name] = self._apply_operations(tables[table_name], operations, delta_path)

        self._write_population(Population(**tables))

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
        delta_path = self.deltas_path / f"{len(self.delta_names()) + 1}_{name}"
        delta_path.mkdir()
        return delta_path

    def apply_deltas(self) -> None:
        for name in self.delta_names():
            self.apply_delta(name)

    @staticmethod
    def _apply_operations(frame: pd.DataFrame, operations: list[dict[str, Any]], delta_path: Path) -> pd.DataFrame:
        for operation in operations:
            if len(operation) != 1:
                raise ValueError(f"Each operation must contain exactly one action: {operation}")
            action, value = next(iter(operation.items()))
            if action == "add_rows":
                frame = PopulationStore._add_rows(frame, delta_path / value)
            elif action == "delete_rows":
                frame = PopulationStore._delete_rows(frame, delta_path / value)
            elif action == "replace_columns":
                frame = PopulationStore._replace_columns(frame, value, delta_path)
            elif action == "delete_columns":
                frame = frame.drop(columns=value)
            else:
                raise ValueError(f"Unsupported delta action: {action}")

        return frame

    @staticmethod
    def _add_rows(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        return pd.concat([frame, pd.read_parquet(path)], ignore_index=True)

    @staticmethod
    def _delete_rows(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        deleted_rows = pd.read_parquet(path)
        if set(deleted_rows.columns) != set(frame.columns):
            raise ValueError("Deleted rows must contain every table column")
        merged = frame.merge(deleted_rows.drop_duplicates(), how="left", indicator=True)
        return merged.loc[merged["_merge"] == "left_only", frame.columns].reset_index(drop=True)

    @staticmethod
    def _replace_columns(frame: pd.DataFrame, replacements: dict[str, str], delta_path: Path) -> pd.DataFrame:
        for column, file_name in replacements.items():
            replacement = pd.read_parquet(delta_path / file_name)
            if list(replacement.columns) != [column]:
                raise ValueError(f"Replacement file for {column!r} must contain only that column")
            if len(replacement) != len(frame):
                raise ValueError(f"Replacement column {column!r} has {len(replacement)} values; expected {len(frame)}")
            frame[column] = replacement[column].to_numpy()
        return frame

    def _write_population(self, population: Population) -> None:
        for name in _TABLE_NAMES:
            getattr(population, name).to_parquet(self.current_state_path / f"{name}.parquet", index=False)
