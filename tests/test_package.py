import pandas as pd

import popdelta
from popdelta import Population


def test_version():
    assert popdelta.__version__ == "0.1.0"


def test_population_retains_supplied_tables():
    households = pd.DataFrame({"id": [1]})
    persons = pd.DataFrame({"id": [1], "household_id": [1]})
    vehicles = pd.DataFrame({"id": [1], "household_id": [1]})

    population = Population(households, persons, vehicles)

    assert population.households is households
    assert population.persons is persons
    assert population.vehicles is vehicles
