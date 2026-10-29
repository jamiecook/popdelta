# popdelta

`popdelta` is a toolkit for representing population evolution as explicit, composable changes.

A population is a collection of related tables, such as households, persons, and vehicles. A
population delta describes how one population state becomes the next: rows may be added or removed,
and columns may be added, removed, or replaced. This makes an evolution from one simulation year to
the next inspectable and reproducible, much like a commit records a change to source code. Like a
commit, each delta also records the full state it produces, so any earlier population can be read
back directly without replaying history.

## Installation

```bash
pip install popdelta
```

For local development:

```bash
uv venv
uv sync --extra dev
```

## Population model

`Population` is a lightweight immutable container for related population tables. It does not impose
a schema, derived fields, or domain-specific assumptions; those belong to the deltas and validation
rules that will be added on top of it. `PopulationStore` does require a key that identifies each row
of each table. The defaults are `household_id`, `person_id`, and `vehicle_id`, and
`PopulationStore.create(..., table_keys={"vehicles": ("household_id", "vehicle_id")})` overrides
them, including with composite keys.

```python
import pandas as pd

from popdelta import Population

population = Population(
    households=pd.DataFrame(...),
    persons=pd.DataFrame(...),
    vehicles=pd.DataFrame(...),
)
```

## On-disk layout

`PopulationStore` keeps the materialized state and its visible delta history together. A complete
example starts with two households, adds one person, then changes columns in a second delta:

```python
from pathlib import Path

import pandas as pd

from popdelta import Population, PopulationStore

population = Population(
    households=pd.DataFrame({"household_id": [1, 2]}),
    persons=pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "household_id": [1, 1, 2, 2, 2],
            "age": [20, 18, 45, 42, 12],
            "obsolete_field": [True, True, False, False, False],
        }
    ),
    vehicles=pd.DataFrame({"vehicle_id": [1, 2], "household_id": [1, 2]}),
)

store = PopulationStore.create(Path("population"), population)
store.create_delta("second_commit")
store.add_to_current_delta(
    table="persons",
    rows={"add": pd.DataFrame({"person_id": [6], "household_id": [2], "age": [0], "obsolete_field": [False]})},
)
pop2 = store.commit_current_delta()

current_ages = pop2.persons[["person_id", "age"]].copy()
store.create_delta("column_manipulation")
store.add_to_current_delta(
    table="persons",
    columns={
        "replace": current_ages.assign(age=current_ages["age"] + 2),
        "delete": ["obsolete_field"],
        "add": pd.DataFrame({"person_id": pop2.persons["person_id"], "new_column": 1}),
    },
)
pop3 = store.commit_current_delta()
```

After this code runs, the directory contains the materialized latest state and both committed
deltas:

```text
population/
├── store.json
├── current_state/
│   ├── households.parquet
│   ├── persons.parquet
│   └── vehicles.parquet
└── deltas/
    ├── 1_initial_commit/
    │   ├── households.parquet
    │   ├── manifest.json
    │   ├── persons.parquet
    │   └── vehicles.parquet
    ├── 2_second_commit/
    │   ├── manifest.json
    │   └── persons/
    │       ├── add_rows_0.parquet
    │       └── state.parquet
    └── 3_column_manipulation/
        ├── manifest.json
        └── persons/
            ├── add_new_column_2.parquet
            └── replace_age_0.parquet
```

`store.json` is written once at store creation and records the primary key column(s) of each table.
Each delta has a JSON manifest. The `tables` operation list records what changed and preserves order,
so a replacement column can cover rows created by an earlier `add_rows` operation. The `state` block
records the resulting table state: for each table modified by the delta, `index` names the file
holding its current primary keys and `files` maps each Parquet file to the active non-key columns it
owns. Tables the delta did not touch are recorded as a single string naming the delta that last
modified them, so resolving any table requires at most one extra manifest read and nothing is copied:

```json
{
  "name": "3_column_manipulation",
  "parents": ["2_second_commit"],
  "tables": {
    "persons": [
      {
        "replace_columns": {
          "file": "persons/replace_age_0.parquet",
          "columns": ["age"]
        }
      },
      {
        "delete_columns": ["obsolete_field"]
      },
      {
        "add_columns": {
          "file": "persons/add_new_column_2.parquet",
          "columns": ["new_column"]
        }
      }
    ]
  },
  "state": {
    "households": "1_initial_commit",
    "persons": {
      "index": "2_second_commit/persons/state.parquet",
      "files": {
        "2_second_commit/persons/state.parquet": ["household_id"],
        "3_column_manipulation/persons/replace_age_0.parquet": ["age"],
        "3_column_manipulation/persons/add_new_column_2.parquet": ["new_column"]
      }
    },
    "vehicles": "1_initial_commit"
  }
}
```

`add_rows` and `delete_rows` refer to Parquet tables. Deleting rows writes only a new key index, and
existing column files stay valid. A column file contains the table's key columns plus one or more
named columns (a multi-column DataFrame writes a single column-set file), and it is matched to rows
by key rather than by position. A plain Series (or single-column DataFrame without keys) is still
accepted and is assigned in key order. Tables are always stored and returned sorted by key, with key
columns first and newly added or replaced columns appended at the end. Keys must be unique and
non-null, and a key that has been deleted cannot be reused by a later delta.

Models that write complete tables can commit them directly instead of computing a diff, and
`create_delta(..., metadata=...)` can record optional model or run provenance in the manifest:

```python
store.create_delta("2025_fleet", metadata={"model": "atlas"})
store.add_to_current_delta(table="vehicles", table_data=new_vehicles)
store.commit_current_delta()
```

Applying a delta updates `current_state/`, which is a materialized copy of the latest delta. The
manifests remain the source of truth:

```python
store.apply_delta("3_column_manipulation")
```

Delta directories start with a unique numeric ID followed by an underscore and descriptive name.
`store.delta_names()` returns them in numeric order, and `store.apply_deltas()` applies every delta
in that order.

## Reading earlier states

Any committed population can be read straight from its manifest:

```python
pop_second = store.play_commits(upto="second_commit")
```

`upto` accepts a full delta directory name (`"2_second_commit"`), a numeric ID or year (`2` or
`2025`), or a descriptive name (`"second_commit"`). An ambiguous reference raises an error instead
of guessing. Pass `columns=` to read only what you need. Only the files holding the requested
columns are opened, and primary key columns are always included automatically:

```python
ages = store.play_commits(upto="second_commit", columns={"persons": ["age"]})
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

`popdelta` is in the initial design phase. The population model is intentionally small while the
API for recording, applying, validating, and composing population deltas is developed.
