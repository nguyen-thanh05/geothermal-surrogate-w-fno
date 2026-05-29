from dataclasses import dataclass


@dataclass
class NormConstants:
    TEMP_MIN: float = 20.0
    TEMP_MAX: float = 185.0
    PRES_MIN: float = 1900.0
    PRES_MAX: float = 68000.0
    ACTION_MIN: float = 0.0
    ACTION_MAX: float = 5000.0
    ENERGY_MAX: float = 2.9e12

    POR_MIN_MATRIX: float = 0.0
    POR_MAX_MATRIX: float = 1.0
    POR_MIN_FRAC: float = 0.0
    POR_MAX_FRAC: float = 1.0
    PERM_MIN_MATRIX: float = 0.0
    PERM_MAX_MATRIX: float = 1.0
    PERM_MIN_FRAC: float = 0.0
    PERM_MAX_FRAC: float = 1.0

    @property
    def TEMP_RANGE(self):
        return self.TEMP_MAX - self.TEMP_MIN

    @property
    def PRES_RANGE(self):
        return self.PRES_MAX - self.PRES_MIN

    @property
    def ACTION_RANGE(self):
        return self.ACTION_MAX - self.ACTION_MIN


HOMO_CONSTANTS = NormConstants(
    PRES_MIN=1900.0, PRES_MAX=68000.0,
)

HETERO_CONSTANTS = NormConstants(
    PRES_MIN=1300.0, PRES_MAX=70000.0,
    POR_MIN_MATRIX=0.03, POR_MAX_MATRIX=0.07,
    POR_MIN_FRAC=0.002, POR_MAX_FRAC=0.008,
    PERM_MIN_MATRIX=0.05, PERM_MAX_MATRIX=0.12,
    PERM_MIN_FRAC=3.0, PERM_MAX_FRAC=190.0,
)


def get_constants(heterogeneous: bool) -> NormConstants:
    return HETERO_CONSTANTS if heterogeneous else HOMO_CONSTANTS


WELL_COORDS = [
    [31, 15], [45, 4], [56, 15], [45, 27], [18, 27],
    [4, 15], [18, 4], [18, 15], [45, 15],
]

N_STEPS = 156
TEST_INDICES = list(range(350, 400))

CHANNEL_NAMES = ['Temp Formation', 'Temp Frac', 'Pres Formation', 'Pres Frac']
CHANNEL_UNITS = ['°C', '°C', 'kPa', 'kPa']
CHANNEL_KEYS = ['temp_form', 'temp_frac', 'pres_form', 'pres_frac']
