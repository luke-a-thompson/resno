from enum import StrEnum


class Driver(StrEnum):
    fBM = "fBM"


class RDE(StrEnum):
    fOU = "fOU"


driver_path_locations = {
    "fBM": "data/drivers/fbm_paths",
}

driver_rough_path_locations = {
    "fBM": "data/rough_paths/fbm_rough_paths",
}

rde_solution_locations = {
    "fOU": "data/solutions/fOU_solutions",
}
