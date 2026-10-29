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


def replay_operations_from_genesis(store: PopulationStore, upto: str) -> Population:
    """Reference sequential operation replay from 1_initial_commit without using manifest['state']."""
    deltas = store.delta_names()
    store_meta = json.loads((store.path / "store.json").read_text())
    key_cols = {name: list(store_meta["tables"][name]["key"]) for name in ("households", "persons", "vehicles")}
    initial = Population.from_dir(store.deltas_path / deltas[0])
    tables = {name: initial.table(name).copy() for name in ("households", "persons", "vehicles")}
    for delta_name in deltas[1 : deltas.index(upto) + 1]:
        delta_path = store.deltas_path / delta_name
        manifest = json.loads((delta_path / "manifest.json").read_text())
        for table_name, ops in manifest.get("tables", {}).items():
            tables[table_name] = PopulationStore._apply_operations(
                tables[table_name], ops, delta_path, key_cols=key_cols[table_name]
            )
    return Population(**tables)


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
    pop2 = store.commit_current_delta()
    assert pop2.persons.shape[0] == 6
    assert store.delta_names() == ["1_initial_commit", "2_second_commit"]
    assert Population.from_dir(population_dir / "current_state") == pop2

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
    assert pop3.persons.shape[0] == 6
    assert "3_column_manipulation" in store.delta_names()
    assert "new_column" in pop3.persons.columns
    assert "obsolete_field" not in pop3.persons.columns
    assert all(pop3.persons["age"] == current_age + 2)

    store.create_delta("conflicting rows cols")
    with pytest.raises(ValueError):
        # Ordering is important and we cant' establish it with keyword args
        # if row delete happens first, we can't add 6 age values when there are only 5 rows in the current state.
        store.add_to_current_delta(
            table="persons",
            columns={"replace": current_age + 3},
            rows={"delete": pd.DataFrame({"person_id": [6]})},
        )

    # If we sequence them correctly, we can first delete the row and then replace the column.
    store.add_to_current_delta(table="persons", columns={"replace": current_age + 3})
    store.add_to_current_delta(table="persons", rows={"delete": pd.DataFrame({"person_id": [6]})})
    pop4 = store.commit_current_delta()

    # Verify that the row with person_id 6 has been deleted
    assert 6 not in pop4.persons["person_id"].values
    assert all(pop4.persons["age"] == current_age[:-1] + 3)

    # Verify play_commits(upto=...) matches every historical snapshot and equals full replay
    assert store.play_commits(upto="1_initial_commit") == population
    assert store.play_commits(upto="second_commit") == pop2
    assert store.play_commits(upto=3) == pop3
    assert store.play_commits(upto="4_conflicting rows cols") == pop4
    for delta_name in store.delta_names():
        assert store.play_commits(upto=delta_name) == replay_operations_from_genesis(store, delta_name)


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


def test_keyed_columns_whole_table_updates_and_play_commits(tmp_path):
    store = PopulationStore.create(
        tmp_path / "population",
        make_population(),
        table_keys={"vehicles": ("household_id", "vehicle_id")},
    )

    # Commit 2 ("2025"): Keyed column replacement supplied in reverse person_id order + value column first
    store.create_delta("2025")
    reversed_ages = pd.DataFrame(
        {
            "age": [13, 43, 46, 19, 21],
            "person_id": [5, 4, 3, 2, 1],
        }
    )
    store.add_to_current_delta(table="persons", columns={"replace": reversed_ages})
    store.add_to_current_delta(
        table="persons",
        columns={
            "add": pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4, 5],
                    "status": pd.Categorical(["ft", "pt", "ft", "ret", "dep"], categories=["ft", "pt", "ret", "dep"]),
                }
            )
        },
    )
    store.add_to_current_delta(
        table="persons",
        columns={"add": pd.DataFrame({"score": pd.Series([10, pd.NA, 30, 40, 50], dtype="Int64")})},
    )
    # Deleting or replacing a key column via column operations is rejected
    with pytest.raises(ValueError, match="Key column.*cannot be modified with column operations"):
        store.add_to_current_delta(table="persons", columns={"delete": ["person_id"]})
    with pytest.raises(ValueError, match="Key column.*cannot be modified with column operations"):
        bad_key_series = pd.Series([10, 20, 30, 40, 50], name="person_id")
        store.add_to_current_delta(table="persons", columns={"replace": bad_key_series})

    store.add_to_current_delta(table="persons", rows={"delete": pd.DataFrame({"person_id": [3]})})
    pop_2025 = store.commit_current_delta()

    # Headless operation file (add_score_2.parquet) remains an immutable 1-column artifact on disk
    assert list(pd.read_parquet(store.deltas_path / "2_2025/persons/add_score_2.parquet").columns) == ["score"]

    # Even though ages were supplied in reverse order, keyed alignment matches each person_id accurately
    pdt.assert_frame_equal(
        pop_2025.persons[["person_id", "age"]],
        pd.DataFrame({"person_id": [1, 2, 4, 5], "age": [21, 19, 43, 13]}),
    )
    assert isinstance(pop_2025.persons["status"].dtype, pd.CategoricalDtype)
    assert str(pop_2025.persons["score"].dtype) == "Int64"

    # Row deletion only wrote an updated index in 2_2025 while inheriting household_id from 1_initial_commit
    manifest_2025 = json.loads((store.deltas_path / "2_2025" / "manifest.json").read_text())
    assert manifest_2025["tables"]["persons"][0] == {
        "replace_columns": {"file": "persons/replace_age_0.parquet", "columns": ["age"]}
    }
    assert manifest_2025["tables"]["persons"][2] == {
        "add_columns": {"file": "persons/add_score_2.parquet", "columns": ["score"]}
    }
    assert manifest_2025["state"]["households"] == "1_initial_commit"
    assert manifest_2025["state"]["vehicles"] == "1_initial_commit"
    assert manifest_2025["state"]["persons"]["index"] == "2_2025/persons/index.parquet"
    assert manifest_2025["state"]["persons"]["files"]["1_initial_commit/persons.parquet"] == [
        "household_id",
        "obsolete_field",
    ]
    assert manifest_2025["state"]["persons"]["files"]["2_2025/persons/replace_age_0.parquet"] == ["age"]
    assert manifest_2025["state"]["persons"]["files"]["2_2025/persons/col_2.parquet"] == ["score"]

    # Commit 3 ("2030"): Whole-table replacement for vehicles (e.g. from a fleet model)
    store.create_delta("2030")
    new_vehicles = pd.DataFrame(
        {
            "vehicle_id": [10, 1, 2],
            "household_id": [2, 1, 2],
            "powertrain": ["bev", "icev", "phev"],
        }
    )
    store.add_to_current_delta(table="vehicles", table_data=new_vehicles)
    pop_2030 = store.commit_current_delta()

    # Single replace_table op points state directly at replace_table_0.parquet without duplicating state.parquet
    manifest_2030 = json.loads((store.deltas_path / "3_2030" / "manifest.json").read_text())
    assert manifest_2030["state"]["households"] == "1_initial_commit"
    assert manifest_2030["state"]["persons"] == "2_2025"
    assert manifest_2030["state"]["vehicles"] == {
        "index": "3_2030/vehicles/replace_table_0.parquet",
        "files": {"3_2030/vehicles/replace_table_0.parquet": ["powertrain"]},
    }
    assert not (store.deltas_path / "3_2030/vehicles/state.parquet").exists()

    # Arbitrary commit checkout via play_commits(upto="2025") and integer year play_commits(upto=2025)
    replayed_2025 = store.play_commits(upto="2025")
    assert replayed_2025 == pop_2025
    assert store.play_commits(upto=2025) == pop_2025
    assert store.play_commits(upto="2030") == pop_2030

    # Projected column checkout reads only requested columns and always retains key columns
    projected = store.play_commits(upto="2025", columns={"persons": ["age"]})
    assert list(projected.persons.columns) == ["person_id", "age"]
    assert list(projected.households.columns) == ["household_id"]

    # Verify tree-based play_commits matches sequential operation replay across all commits (including dtypes)
    for delta_name in store.delta_names():
        assert store.play_commits(upto=delta_name) == replay_operations_from_genesis(store, delta_name)


def test_reusing_deleted_key_raises_error(tmp_path):
    store = PopulationStore.create(tmp_path / "population", make_population())

    store.create_delta("delete_person_5")
    store.add_to_current_delta(table="persons", rows={"delete": pd.DataFrame({"person_id": [5]})})
    store.commit_current_delta()

    store.create_delta("recycle_person_5")
    store.add_to_current_delta(
        table="persons",
        rows={"add": pd.DataFrame({"person_id": [5], "household_id": [1], "age": [0], "obsolete_field": [False]})},
    )
    with pytest.raises(ValueError, match="reuse existing or previously deleted keys"):
        store.commit_current_delta()

    # Uncommitted delta is skipped by default play_commits(), raises clear error if requested explicitly,
    # blocks creating a new delta until discarded, and can be cleanly discarded to retry
    assert store.play_commits() == store.play_commits(upto="delete_person_5")
    with pytest.raises(ValueError, match="has not been committed"):
        store.play_commits(upto="recycle_person_5")
    with pytest.raises(ValueError, match="has not been committed; commit or discard it first"):
        store.create_delta("another_delta")
    store.discard_current_delta()
    assert store.delta_names() == ["1_initial_commit", "2_delete_person_5"]

    # Also verify whole-table replacement (table_data) rejects reusing retired keys
    store_replace = PopulationStore.create(tmp_path / "population_replace", make_population())
    store_replace.create_delta("delete_person_5")
    store_replace.add_to_current_delta(table="persons", rows={"delete": pd.DataFrame({"person_id": [5]})})
    store_replace.commit_current_delta()

    store_replace.create_delta("replace_table_with_retired_person_5")
    store_replace.add_to_current_delta(table="persons", table_data=make_population().persons)
    with pytest.raises(ValueError, match="Replacement table reuses previously deleted keys"):
        store_replace.commit_current_delta()


def test_multi_column_set_files_and_one_hop_state_compaction(tmp_path):
    store = PopulationStore.create(tmp_path / "population", make_population())

    # store.json holds table key definitions once at the store root
    store_meta = json.loads((store.path / "store.json").read_text())
    assert store_meta == {
        "format": 1,
        "tables": {
            "households": {"key": ["household_id"]},
            "persons": {"key": ["person_id"]},
            "vehicles": {"key": ["vehicle_id"]},
        },
    }

    # Add a multi-column employment set in one operation -> writes a single Parquet file and manifest entry
    store.create_delta("2025_employment", metadata={"model": "demos", "run_id": "run_001"})
    employment_cols = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "employment_status": ["ft", "pt", "ft", "ret", "dep"],
            "industry": ["tech", "edu", "mfg", "none", "none"],
            "hours_worked": [40, 20, 45, 0, 0],
        }
    )
    store.add_to_current_delta(table="persons", columns={"add": employment_cols})
    store.commit_current_delta()

    manifest_2 = json.loads((store.deltas_path / "2_2025_employment" / "manifest.json").read_text())
    assert manifest_2["metadata"] == {"model": "demos", "run_id": "run_001"}
    assert manifest_2["tables"]["persons"] == [
        {
            "add_columns": {
                "file": "persons/add_0.parquet",
                "columns": ["employment_status", "industry", "hours_worked"],
            }
        }
    ]
    assert manifest_2["state"]["households"] == "1_initial_commit"
    assert manifest_2["state"]["vehicles"] == "1_initial_commit"
    assert manifest_2["state"]["persons"]["files"] == {
        "1_initial_commit/persons.parquet": ["household_id", "age", "obsolete_field"],
        "2_2025_employment/persons/add_0.parquet": ["employment_status", "industry", "hours_worked"],
    }

    # Replace all remaining columns from 1_initial_commit/persons.parquet + delete obsolete_field:
    # 1_initial_commit/persons.parquet should drop out of `files` once none of its non-key columns remain live
    store.create_delta("2030_update")
    store.add_to_current_delta(
        table="persons",
        columns={
            "replace": pd.DataFrame(
                {
                    "person_id": [1, 2, 3, 4, 5],
                    "household_id": [1, 1, 2, 2, 1],
                    "age": [25, 23, 50, 47, 17],
                }
            ),
            "delete": ["obsolete_field"],
        },
    )
    pop_3 = store.commit_current_delta()

    manifest_3 = json.loads((store.deltas_path / "3_2030_update" / "manifest.json").read_text())
    # Untouched tables still point directly to 1_initial_commit (one-hop invariant, never chained via 2_2025_employment)
    assert manifest_3["state"]["households"] == "1_initial_commit"
    assert manifest_3["state"]["vehicles"] == "1_initial_commit"
    assert manifest_3["state"]["persons"]["index"] == "1_initial_commit/persons.parquet"
    assert manifest_3["state"]["persons"]["files"] == {
        "2_2025_employment/persons/add_0.parquet": ["employment_status", "industry", "hours_worked"],
        "3_2030_update/persons/replace_0.parquet": ["household_id", "age"],
    }

    # Selective projection of one column from a multi-column set file
    industry_only = store.play_commits(upto="2030_update", columns={"persons": ["industry"]})
    assert list(industry_only.persons.columns) == ["person_id", "industry"]
    assert list(industry_only.persons["industry"]) == ["tech", "edu", "mfg", "none", "none"]
    assert store.play_commits(upto="2030_update") == pop_3
    for delta_name in store.delta_names():
        assert store.play_commits(upto=delta_name) == replay_operations_from_genesis(store, delta_name)

    # Enforcing the one-hop invariant: a chained pointer raises ValueError
    manifest_3["state"]["households"] = "2_2025_employment"
    (store.deltas_path / "3_2030_update" / "manifest.json").write_text(json.dumps(manifest_3))
    with pytest.raises(ValueError, match="must resolve to a state block in one hop"):
        store.play_commits(upto="2030_update")


def test_column_validation_and_store_format_checks(tmp_path):
    store = PopulationStore.create(
        tmp_path / "population",
        make_population(),
        table_keys={"vehicles": ("household_id", "vehicle_id")},
    )

    # Non-serializable metadata fails before creating an orphaned delta directory
    with pytest.raises(TypeError):
        store.create_delta("bad_meta", metadata={"bad": {1, 2, 3}})
    assert store.delta_names() == ["1_initial_commit"]

    # Round-trip _dump_json with strings containing brackets, commas, and escaped quotes
    tricky = {
        "note": 'values like ["a", "b"], [1, 2], and "escaped" quotes',
        "list": ["alpha", 'beta "quoted"', "gamma, [delta]"],
    }
    assert json.loads(PopulationStore._dump_json(tricky)) == tricky

    store.create_delta("validation_checks")

    # Headless multi-column DataFrames are rejected (require keys for column sets)
    with pytest.raises(ValueError, match="Multi-column DataFrames must include key column"):
        store.add_to_current_delta(
            table="persons",
            columns={"add": pd.DataFrame({"col_a": [1, 2, 3, 4, 5], "col_b": [6, 7, 8, 9, 10]})},
        )

    # Partial composite key reports the missing key column clearly
    with pytest.raises(ValueError, match=r"Missing key column\(s\): \['vehicle_id'\]"):
        store.add_to_current_delta(
            table="vehicles",
            columns={"add": pd.DataFrame({"household_id": [1, 2], "powertrain": ["bev", "icev"]})},
        )

    # Enforce add vs. replace column existence semantics
    with pytest.raises(ValueError, match=r"Cannot replace unknown column\(s\) \['nonexistent'\]"):
        store.add_to_current_delta(
            table="persons",
            columns={"replace": pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "nonexistent": [1, 2, 3, 4, 5]})},
        )
    with pytest.raises(ValueError, match=r"Cannot add existing column\(s\) \['age'\]"):
        store.add_to_current_delta(
            table="persons",
            columns={"add": pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "age": [21, 19, 46, 43, 13]})},
        )

    # Verify store.json format version check and missing store.json error
    store.store_metadata_path.write_text(json.dumps({"format": 99, "tables": {}}))
    with pytest.raises(ValueError, match="Unsupported population store format: 99"):
        store.play_commits()
    store.store_metadata_path.unlink()
    with pytest.raises(ValueError, match="Population store predates format 1"):
        store.play_commits()
