# popdelta

`popdelta` is a toolkit for representing population evolution as explicit, composable changes.

A population is a collection of related tables, such as households, persons, and vehicles. A
population delta describes how one population state becomes the next: rows may be added or removed,
and columns may be added, removed, or replaced. This makes an evolution from one simulation year to
the next inspectable and reproducible, much like a commit records a change to source code.

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
an ID scheme, schema, derived fields, or domain-specific assumptions; those belong to the deltas and
validation rules that will be added on top of it.

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

current_age = pop2.persons["age"].copy()
store.create_delta("column_manipulation")
store.add_to_current_delta(
    table="persons",
    columns={
        "replace": current_age + 2,
        "delete": ["obsolete_field"],
        "add": pd.DataFrame({"new_column": [1, 1, 1, 1, 1, 1]}),
    },
)
pop3 = store.commit_current_delta()
```

After this code runs, the directory contains the materialized latest state and both committed
deltas:

```text
population/
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
    │       └── add_rows_0.parquet
    └── 3_column_manipulation/
        ├── manifest.json
        └── persons/
            ├── add_new_column_1.parquet
            └── replace_age_0.parquet
```

Each delta has a JSON manifest. The table operation list preserves order, so a replacement column
can contain exactly one value for each row created by an earlier `add_rows` operation:

```json
{
  "name": "Evolve to 2030",
  "date_generated": "2026-10-12T09:00:00",
  "tables": {
    "persons": [
      {"add_rows": "persons/add_rows_0.parquet"},
      {"delete_rows": "persons/deleted_rows.parquet"},
      {"replace_columns": {"age": "persons/replacement_age.parquet"}},
      {"delete_columns": ["obsolete_field"]}
    ]
  }
}
```

`add_rows` and `delete_rows` refer to Parquet tables. A replacement Parquet file contains one named
column and must have exactly one value for every row present when its operation is reached. Applying
a delta updates `current_state/`:

```python
store.apply_delta("evolve-to-2030")
```

Delta directories start with a unique numeric ID followed by an underscore and descriptive name.
`store.delta_names()` returns them in numeric order, and `store.apply_deltas()` applies every delta
in that order.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

`popdelta` is in the initial design phase. The population model is intentionally small while the
API for recording, applying, validating, and composing population deltas is developed.
