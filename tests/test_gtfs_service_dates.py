import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault(
    "srtm", types.SimpleNamespace(get_data=lambda *args, **kwargs: None)
)

from gtfs_to_segment import resolve_gtfs_service_date, service_ids_for_date  # noqa: E402


def _calendar_2026_only():
    return pd.DataFrame(
        [
            {
                "service_id": "monday_service",
                "monday": 1,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0,
                "start_date": "20260615",
                "end_date": "20260615",
            },
            {
                "service_id": "tuesday_early",
                "monday": 0,
                "tuesday": 1,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0,
                "start_date": "20260616",
                "end_date": "20260623",
            },
            {
                "service_id": "tuesday_late",
                "monday": 0,
                "tuesday": 1,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 0,
                "sunday": 0,
                "start_date": "20260714",
                "end_date": "20261215",
            },
            {
                "service_id": "saturday_service",
                "monday": 0,
                "tuesday": 0,
                "wednesday": 0,
                "thursday": 0,
                "friday": 0,
                "saturday": 1,
                "sunday": 0,
                "start_date": "20260613",
                "end_date": "20260613",
            },
        ]
    )


def test_service_date_remaps_to_same_month_day_in_feed_year() -> None:
    requested = date(2021, 5, 6)

    assert resolve_gtfs_service_date(
        _calendar_2026_only(), None, requested
    ) == date(2026, 5, 6)


def test_out_of_range_remap_uses_one_nearest_real_weekday_service() -> None:
    requested = date(2025, 4, 14)  # Monday, but 2026-04-14 is Tuesday.

    assert service_ids_for_date(
        _calendar_2026_only(), None, requested
    ) == {"tuesday_early"}


def test_out_of_range_weekend_uses_real_weekend_service() -> None:
    requested = date(2025, 3, 14)  # Friday, but 2026-03-14 is Saturday.

    assert service_ids_for_date(
        _calendar_2026_only(), None, requested
    ) == {"saturday_service"}


def test_exact_exception_date_is_used_before_fallback() -> None:
    calendar_dates = pd.DataFrame(
        [
            {
                "service_id": "summer_special",
                "date": "20260712",
                "exception_type": 1,
            }
        ]
    )

    assert service_ids_for_date(
        _calendar_2026_only(), calendar_dates, date(2025, 7, 12)
    ) == {"summer_special"}
