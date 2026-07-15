import pytest

from gtfs_to_segment import (
    build_segments,
    load_shapes_for_trips,
    resolve_route_id,
    service_ids_for_date,
    stop_times_for_trip,
    trips_for_route,
)


def test_gtfs_tables_select_trip_and_construct_ordered_segments(
    integration_gtfs_zip, integration_tables, service_date
):
    routes, trips, stops, calendar, calendar_dates = integration_tables

    route_id, route_long_name = resolve_route_id(routes, "208")
    service_ids = service_ids_for_date(calendar, calendar_dates, service_date)
    trip_ids = trips_for_route(trips, route_id, direction_id=0, service_ids=service_ids)
    rows = stop_times_for_trip(integration_gtfs_zip, trip_ids[0])
    shapes, shape_by_trip = load_shapes_for_trips(integration_gtfs_zip, trips, trip_ids)
    segments = build_segments(rows, stops, shape_points=shapes[shape_by_trip[trip_ids[0]]])

    assert route_id == "r1"
    assert route_long_name == "Synthetic 208"
    assert trip_ids == ["t1"]
    assert trips.set_index("trip_id").loc["t1", "direction_id"] == 0
    assert [row["stop_id"] for row in rows] == ["s1", "sdup", "s2", "s3"]
    assert len(segments) == 2, "co-located duplicate stop should be dropped"
    assert [seg.length_m for seg in segments] == pytest.approx([500.0, 500.0], abs=0.1)
    assert [seg.run_time_s for seg in segments] == [110, 100]
    assert [seg.dwell_s for seg in segments] == [20, 15]
    assert segments[0].from_stop_departure_time == "08:00:10"
    assert segments[1].to_stop_arrival_time == "08:04:00"
