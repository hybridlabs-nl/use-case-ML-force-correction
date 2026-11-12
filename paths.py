"""Data paths."""

from pathlib import Path

# `data/` at the repo root, which must contain `data_x_wp2_202512/` (git-ignored)
DATA_ROOT = Path(__file__).parent / "data"

DATA_DECAY_CL = DATA_ROOT / "data_x_wp2_202512/dataB/decay_cl"
DATA_DECAY_CL_TEST = DATA_ROOT / "data_x_wp2_202512/dataB/decay_cl_test"
