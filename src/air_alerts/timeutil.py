import numpy as np
import pandas as pd

# Stand-in end time for an alert that is still open: later than any real timestamp.
OPEN_END = np.iinfo(np.int64).max


def to_ns(values) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds. pandas 3 infers s/us/ns units, so normalise first."""
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def ends_to_ns(values) -> np.ndarray:
    """Like to_ns, but NaT (alert still open) becomes OPEN_END instead of a huge negative number."""
    idx = pd.DatetimeIndex(values).as_unit("ns")
    return np.where(idx.isna(), OPEN_END, idx.asi8)
