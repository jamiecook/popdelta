import json

import pandas as pd
import pandas.testing as pdt
import pytest

from popdelta import Population, PopulationStore


def make_population() -> Population:
    return Population(
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


def write_delta(store: PopulationStore, name: str, operations: list[dict], files: dict[str, pd.DataFrame]) -> None:
    delta_path = store.deltas_path / name
    delta_path.mkdir(parents=True)
    for file_name, frame in files.items():
        path = delta_path / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    (delta_path / "manifest.json").write_text(
        json.dumps({"name": name, "date_generated": "2026-10-12T09:00:00", "tables": {"persons": operations}})
    )


def test_create_initializes_population_directory(tmp_path):
    population_dir = tmp_path / "population"
    population = make_population()

    store = PopulationStore.create(population_dir, population)

    assert (population_dir / "current_state").exists()
    assert (population_dir / "deltas").exists()
    assert (population_dir / "deltas" / "1_initial_commit").exists()
    assert store.delta_names() == ["1_initial_commit"]
    assert Population.from_dir(population_dir / "deltas" / "1_initial_commit") == population

    store.create_delta("second_commit")
    assert store.delta_names() == ["1_initial_commit", "2_second_commit"]
    assert (population_dir / "deltas" / "2_second_commit").exists()
    assert len(list((population_dir / "deltas" / "2_second_commit").iterdir())) == 0

    store.add_to_current_delta(
        table="persons",
        rows={"add": pd.DataFrame({"person_id": [6], "household_id": [2], "age": [0], "obsolete_field": [False]})},
    )

    # Before committing the current delta, the original population should remain unchanged.
    assert population.persons.shape[0] == 5

    # Running the commit
    # 1) applies the changes to the current state,
    # 2) persists everything to the filesystem and
    # 3) returns the new population.
    new_pop = store.commit_current_delta()
    assert new_pop.persons.shape[0] == 6
    assert store.delta_names() == ["1_initial_commit", "2_second_commit"]
    assert Population.from_dir(population_dir / "current_state") == new_pop


def test_delta_directories_are_sorted_by_numeric_id(tmp_path):
    store = PopulationStore.create(tmp_path / "population", make_population())
    write_delta(store, "2_age", [], {})
    write_delta(store, "10_future", [], {})

    assert store.delta_names() == ["1_initial_commit", "2_age", "10_future"]


def test_apply_delta_runs_operations_in_manifest_order(tmp_path):
    store = PopulationStore.create(tmp_path / "population", make_population())
    write_delta(
        store,
        "2_evolve-to-2030",
        [
            {"add_rows": "persons/added_rows.parquet"},
            {"replace_columns": {"age": "persons/replacement_age.parquet"}},
            {"delete_columns": ["obsolete_field"]},
        ],
        {
            "persons/added_rows.parquet": pd.DataFrame(
                {"person_id": [6], "household_id": [2], "age": [0], "obsolete_field": [False]}
            ),
            "persons/replacement_age.parquet": pd.DataFrame({"age": [25, 23, 50, 47, 17, 5]}),
        },
    )

    store.apply_delta("2_evolve-to-2030")

    pdt.assert_frame_equal(
        store.read().persons,
        pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5, 6],
                "household_id": [1, 1, 2, 2, 2, 2],
                "age": [25, 23, 50, 47, 17, 5],
            }
        ),
    )


def test_replace_column_requires_a_value_for_each_current_row(tmp_path):
    store = PopulationStore.create(tmp_path / "population", make_population())
    write_delta(
        store,
        "2_invalid-age-update",
        [{"replace_columns": {"age": "persons/replacement_age.parquet"}}],
        {"persons/replacement_age.parquet": pd.DataFrame({"age": [25]})},
    )

    with pytest.raises(ValueError, match="Replacement column 'age' has 1 values; expected 5"):
        store.apply_delta("2_invalid-age-update")
