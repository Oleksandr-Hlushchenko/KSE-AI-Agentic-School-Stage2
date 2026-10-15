import pandas as pd
import pytest

from air_alerts.data import build_episodes, load_events


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    return load_events()


@pytest.fixture(scope="session")
def episodes(events) -> pd.DataFrame:
    return build_episodes(events)[0]


def make_events(rows: list[tuple[str, int]]) -> pd.DataFrame:
    """Synthetic event log from (UTC time, state) pairs."""
    return pd.DataFrame({
        "t": pd.to_datetime([r[0] for r in rows], utc=True, format="ISO8601"),
        "state": [r[1] for r in rows],
    })
