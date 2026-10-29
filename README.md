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

`PopulationStore` keeps the materialized state and its visible delta history together:

```text
population/
├── current_state/
│   ├── households.parquet
│   ├── persons.parquet
│   └── vehicles.parquet
└── deltas/
    ├── 1_initial-population/
    └── 2_evolve-to-2030/
        ├── manifest.json
        └── persons/
            ├── added_rows.parquet
            └── replacement_age.parquet
```

The manifest uses a list so operations run in order. This matters when an operation changes the
number of rows before a later replacement column is applied:

```json
{
  "name": "Evolve to 2030",
  "date_generated": "2026-10-12T09:00:00",
  "tables": {
    "persons": [
      {"add_rows": "persons/added_rows.parquet"},
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
