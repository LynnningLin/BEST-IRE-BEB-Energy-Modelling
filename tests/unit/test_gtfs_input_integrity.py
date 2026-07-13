import pytest

from gtfs_to_segment import (
    build_segments,
    gtfs_time_to_seconds,
    haversine_m,
    iter_valid_stop_pairs,
    load_shapes_for_trips,
    load_small_tables,
    stop_times_for_trip,
    validate_gtfs_feed,
)


def test_valid_minimal_gtfs_feed_passes_integrity_checks(minimal_gtfs_zip):
    summary = validate_gtfs_feed(minimal_gtfs_zip())

    assert summary["routes"] == 1
    assert summary["trips"] == 1
    assert summary["stop_times"] == 3


def test_missing_required_gtfs_file_produces_clear_error(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(omit={"stops.txt"})

    with pytest.raises(ValueError, match="missing required file.*stops.txt"):
        load_small_tables(feed)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "stop_times.txt": [
                    {
                        "trip_id": "t1",
                        "arrival_time": "08:00:00",
                        "departure_time": "08:00:00",
                        "stop_id": "missing",
                        "stop_sequence": "1",
                    }
                ]
            },
            "unknown stop_id",
        ),
        (
            {
                "stop_times.txt": [
                    {
                        "trip_id": "missing",
                        "arrival_time": "08:00:00",
                        "departure_time": "08:00:00",
                        "stop_id": "s1",
                        "stop_sequence": "1",
                    }
                ]
            },
            "unknown trip_id",
        ),
        (
            {
                "trips.txt": [
                    {
                        "route_id": "missing",
                        "service_id": "svc",
                        "trip_id": "t1",
                        "direction_id": "0",
                        "shape_id": "shape1",
                    }
                ]
            },
            "unknown route_id",
        ),
    ],
)
def test_gtfs_cross_table_references_are_validated(
    minimal_gtfs_zip, overrides, message
):
    feed = minimal_gtfs_zip(overrides=overrides)

    with pytest.raises(ValueError, match=message):
        validate_gtfs_feed(feed)


def test_stop_sequences_must_increase_within_trip(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:05:00",
                    "departure_time": "08:05:00",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:00",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
            ]
        }
    )

    with pytest.raises(ValueError, match="stop_sequence values must increase"):
        validate_gtfs_feed(feed)


def test_scheduled_travel_times_must_be_positive(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:05:00",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:04:59",
                    "departure_time": "08:06:00",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
            ]
        }
    )

    with pytest.raises(ValueError, match="non-positive travel time"):
        validate_gtfs_feed(feed)


def test_gtfs_times_above_24_hours_parse_to_next_day_seconds():
    assert gtfs_time_to_seconds("25:10:00") == 25 * 3600 + 10 * 60


def test_negative_dwell_is_clamped_after_segment_build(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:00",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:05:00",
                    "departure_time": "08:04:00",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
            ]
        }
    )
    _, _, stops, _, _ = load_small_tables(feed)

    segments = build_segments(stop_times_for_trip(feed, "t1"), stops)

    assert segments[0].dwell_s == 0.0, "negative GTFS dwell should clamp to zero"


def test_colocated_stops_do_not_create_zero_length_segments(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "stops.txt": [
                {"stop_id": "s1", "stop_name": "A", "stop_lat": "51.0", "stop_lon": "-8.0"},
                {"stop_id": "s2", "stop_name": "B", "stop_lat": "51.0", "stop_lon": "-8.0"},
            ],
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:00",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:05:00",
                    "departure_time": "08:05:00",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
            ],
        },
        omit={"shapes.txt"},
    )
    _, _, stops, _, _ = load_small_tables(feed)

    pairs = list(iter_valid_stop_pairs(stop_times_for_trip(feed, "t1"), stops))

    assert pairs == [], "co-located stops should not produce invalid zero-length pairs"


def test_shape_distance_is_used_when_valid_shape_exists(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "shapes.txt": [
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.0",
                    "shape_pt_lon": "-8.0",
                    "shape_pt_sequence": "1",
                },
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.001",
                    "shape_pt_lon": "-8.0",
                    "shape_pt_sequence": "2",
                },
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.001",
                    "shape_pt_lon": "-7.999",
                    "shape_pt_sequence": "3",
                },
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.0",
                    "shape_pt_lon": "-7.999",
                    "shape_pt_sequence": "4",
                },
            ]
        }
    )
    routes, trips, stops, _, _ = load_small_tables(feed)
    rows = stop_times_for_trip(feed, "t1")
    shapes, trip_shape = load_shapes_for_trips(feed, trips, ["t1"])

    pairs = list(iter_valid_stop_pairs(rows, stops, shapes[trip_shape["t1"]]))

    straight = haversine_m(51.0, -8.0, 51.0, -7.999)
    assert pairs[0][2] > straight, "valid shapes should override straight-line distance"
    assert routes.iloc[0]["route_id"] == "r1"


def test_haversine_distance_is_used_when_shape_data_are_absent(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(omit={"shapes.txt"})
    _, _, stops, _, _ = load_small_tables(feed)

    pairs = list(iter_valid_stop_pairs(stop_times_for_trip(feed, "t1"), stops))

    assert pairs[0][2] == pytest.approx(haversine_m(51.0, -8.0, 51.0, -7.999))


def test_invalid_shape_data_falls_back_to_haversine(minimal_gtfs_zip):
    feed = minimal_gtfs_zip(
        overrides={
            "shapes.txt": [
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "not-a-number",
                    "shape_pt_lon": "-8.0",
                    "shape_pt_sequence": "1",
                }
            ]
        }
    )
    _, trips, stops, _, _ = load_small_tables(feed)
    shapes, trip_shape = load_shapes_for_trips(feed, trips, ["t1"])

    pairs = list(
        iter_valid_stop_pairs(
            stop_times_for_trip(feed, "t1"), stops, shapes.get(trip_shape["t1"])
        )
    )

    assert pairs[0][2] == pytest.approx(haversine_m(51.0, -8.0, 51.0, -7.999))
