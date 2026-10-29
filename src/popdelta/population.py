from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Population:
    """A collection of related population tables at one point in time."""

    households: pd.DataFrame
    persons: pd.DataFrame
    vehicles: pd.DataFrame

    @classmethod
    def from_dir(cls, path: Path) -> Population:
        return cls(
            households=pd.read_parquet(path / "households.parquet"),
            persons=pd.read_parquet(path / "persons.parquet"),
            vehicles=pd.read_parquet(path / "vehicles.parquet"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Population):
            return NotImplemented
        return all(
            left.equals(right)
            for left, right in zip(
                (self.households, self.persons, self.vehicles),
                (other.households, other.persons, other.vehicles),
                strict=True,
            )
        )
