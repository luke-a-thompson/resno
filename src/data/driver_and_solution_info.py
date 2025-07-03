from enum import StrEnum


class Driver(StrEnum):
    fBM = "fBM"


class RDE(StrEnum):
    fOU = "fOU"


rde_locations = {
    "fOU": "data/fOU",
}
