import pytest

from speed_caps import load_speed_cap_cache, resolve_speed_caps
from traffic_signals import load_signal_cache, resolve_signal_counts


SIGNAL_ROW = {
    "from_stop_id": "a",
    "to_stop_id": "b",
    "n_signals": "1",
    "source": "osm",
    "snap_radius_m": "30",
    "relaxed_snap_radius_m": "60",
    "cluster_radius_m": "35",
    "length_m": "100",
    "fetched_utc": "2026-01-01T00:00:00Z",
}

SPEED_ROW = {
    "from_stop_id": "a",
    "to_stop_id": "b",
    "speed_cap_ms": "13.889",
    "speed_cap_kmh": "50",
    "source": "osm",
    "coverage_frac": "1",
    "snap_radius_m": "25",
    "sample_step_m": "20",
    "length_m": "100",
    "fetched_utc": "2026-01-01T00:00:00Z",
}


def test_valid_signal_cache_row_loads(write_csv):
    path = write_csv("signals.csv", [SIGNAL_ROW])

    cache = load_signal_cache(path)

    assert cache[("a", "b")]["n_signals"] == 1
    assert cache[("a", "b")]["source"] == "osm"


def test_missing_signal_cache_file_returns_empty_dict(tmp_path):
    assert load_signal_cache(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"n_signals": "-1"}, "non-negative"),
        ({"n_signals": "bad"}, "could not convert"),
        ({"source": "manual"}, "invalid signal cache source"),
        ({"snap_radius_m": "0"}, "snap_radius_m must be positive"),
        ({"length_m": "-1"}, "length_m must be non-negative"),
    ],
)
def test_invalid_signal_cache_rows_are_rejected(write_csv, patch, message):
    row = {**SIGNAL_ROW, **patch}
    path = write_csv("signals.csv", [row])

    with pytest.raises(ValueError, match=message):
        load_signal_cache(path)


def test_duplicate_signal_cache_stop_pairs_are_rejected(write_csv):
    path = write_csv("signals.csv", [SIGNAL_ROW, {**SIGNAL_ROW, "n_signals": "2"}])

    with pytest.raises(ValueError, match="duplicate signal cache row"):
        load_signal_cache(path)


def test_signal_cache_parameter_mismatch_refreshes_row():
    cache = {("a", "b"): {**SIGNAL_ROW, "snap_radius_m": 99.0, "n_signals": 3}}

    counts, sources, stats = resolve_signal_counts(
        [("a", "b", 100.0)],
        {"a": (51.0, -8.0), "b": (51.001, -8.0)},
        snap_radius_m=30.0,
        cache=cache,
        fetch_fn=lambda *args, **kwargs: [],
        verbose=False,
    )

    assert stats["cache_hit"] == 0, "mismatched snap radius should bypass cache"
    assert counts[("a", "b")] == 0
    assert sources[("a", "b")] == "osm"


def test_signal_cache_refresh_bypasses_matching_cache():
    cache = {
        ("a", "b"): {
            **SIGNAL_ROW,
            "snap_radius_m": 30.0,
            "relaxed_snap_radius_m": 60.0,
            "cluster_radius_m": 35.0,
            "n_signals": 3,
        }
    }

    counts, _, stats = resolve_signal_counts(
        [("a", "b", 100.0)],
        {"a": (51.0, -8.0), "b": (51.001, -8.0)},
        snap_radius_m=30.0,
        relaxed_snap_radius_m=60.0,
        cluster_radius_m=35.0,
        cache=cache,
        refresh=True,
        fetch_fn=lambda *args, **kwargs: [],
        verbose=False,
    )

    assert stats["cache_hit"] == 0
    assert counts[("a", "b")] == 0, "refresh should ignore stale cached count"


def test_valid_speed_cap_cache_row_loads(write_csv):
    path = write_csv("speed_caps.csv", [SPEED_ROW])

    cache = load_speed_cap_cache(path)

    assert cache[("a", "b")]["speed_cap_ms"] == pytest.approx(13.889)
    assert cache[("a", "b")]["source"] == "osm"


def test_missing_speed_cap_cache_file_returns_empty_dict(tmp_path):
    assert load_speed_cap_cache(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"speed_cap_ms": "0"}, "speed_cap_ms must be positive"),
        ({"speed_cap_ms": "bad"}, "could not convert"),
        ({"speed_cap_kmh": "-1"}, "speed_cap_kmh must be non-negative"),
        ({"source": "manual"}, "invalid speed-cap cache source"),
        ({"coverage_frac": "1.5"}, r"coverage_frac must be in \[0, 1\]"),
        ({"snap_radius_m": "0"}, "snap_radius_m must be positive"),
        ({"sample_step_m": "0"}, "sample_step_m must be positive"),
        ({"length_m": "-1"}, "length_m must be non-negative"),
    ],
)
def test_invalid_speed_cap_cache_rows_are_rejected(write_csv, patch, message):
    row = {**SPEED_ROW, **patch}
    path = write_csv("speed_caps.csv", [row])

    with pytest.raises(ValueError, match=message):
        load_speed_cap_cache(path)


def test_duplicate_speed_cap_cache_stop_pairs_are_rejected(write_csv):
    path = write_csv("speed_caps.csv", [SPEED_ROW, {**SPEED_ROW, "speed_cap_ms": "9"}])

    with pytest.raises(ValueError, match="duplicate speed-cap cache row"):
        load_speed_cap_cache(path)


def test_speed_cap_cache_parameter_mismatch_refreshes_row():
    cache = {("a", "b"): {**SPEED_ROW, "snap_radius_m": 99.0, "sample_step_m": 20.0}}

    caps, sources, stats = resolve_speed_caps(
        [("a", "b", 100.0)],
        {"a": (51.0, -8.0), "b": (51.001, -8.0)},
        snap_radius_m=25.0,
        sample_step_m=20.0,
        cache=cache,
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        verbose=False,
    )

    assert stats["cache_hit"] == 0, "mismatched snap radius should bypass speed cache"
    assert caps[("a", "b")] == pytest.approx(50.0 / 3.6)
    assert sources[("a", "b")] == "fallback"


def test_speed_cap_cache_refresh_bypasses_matching_cache():
    cache = {("a", "b"): {**SPEED_ROW, "snap_radius_m": 25.0, "sample_step_m": 20.0}}

    caps, _, stats = resolve_speed_caps(
        [("a", "b", 100.0)],
        {"a": (51.0, -8.0), "b": (51.001, -8.0)},
        snap_radius_m=25.0,
        sample_step_m=20.0,
        cache=cache,
        refresh=True,
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        verbose=False,
    )

    assert stats["cache_hit"] == 0
    assert caps[("a", "b")] == pytest.approx(50.0 / 3.6)
