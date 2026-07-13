import csv
import sys
import types
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault(
    "srtm", types.SimpleNamespace(get_data=lambda *args, **kwargs: None)
)


def pytest_configure(config):
    """Keep pytest temp files inside the writable repo workspace on Windows."""
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp is None:
        basetemp = PROJECT_ROOT / "test_tmp" / "pytest"
        config.option.basetemp = str(basetemp)
    Path(basetemp).expanduser().parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def write_csv(tmp_path):
    def _write(name, rows, fieldnames=None):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if fieldnames is None and rows:
            fieldnames = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames or [])
            writer.writeheader()
            writer.writerows(rows)
        return path

    return _write


@pytest.fixture
def minimal_gtfs_zip(tmp_path):
    def _build(overrides=None, omit=()):
        tables = {
            "routes.txt": [
                {"route_id": "r1", "route_short_name": "208", "route_long_name": "Route 208"}
            ],
            "trips.txt": [
                {
                    "route_id": "r1",
                    "service_id": "svc",
                    "trip_id": "t1",
                    "direction_id": "0",
                    "shape_id": "shape1",
                }
            ],
            "stops.txt": [
                {"stop_id": "s1", "stop_name": "A", "stop_lat": "51.0", "stop_lon": "-8.0"},
                {"stop_id": "s2", "stop_name": "B", "stop_lat": "51.0", "stop_lon": "-7.999"},
                {"stop_id": "s3", "stop_name": "C", "stop_lat": "51.001", "stop_lon": "-7.999"},
            ],
            "stop_times.txt": [
                {
                    "trip_id": "t1",
                    "arrival_time": "08:00:00",
                    "departure_time": "08:00:10",
                    "stop_id": "s1",
                    "stop_sequence": "1",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:05:00",
                    "departure_time": "08:05:20",
                    "stop_id": "s2",
                    "stop_sequence": "2",
                },
                {
                    "trip_id": "t1",
                    "arrival_time": "08:10:00",
                    "departure_time": "08:10:20",
                    "stop_id": "s3",
                    "stop_sequence": "3",
                },
            ],
            "calendar.txt": [
                {
                    "service_id": "svc",
                    "monday": "1",
                    "tuesday": "1",
                    "wednesday": "1",
                    "thursday": "1",
                    "friday": "1",
                    "saturday": "0",
                    "sunday": "0",
                    "start_date": "20250101",
                    "end_date": "20251231",
                }
            ],
            "shapes.txt": [
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.0",
                    "shape_pt_lon": "-8.0",
                    "shape_pt_sequence": "1",
                },
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.0",
                    "shape_pt_lon": "-7.999",
                    "shape_pt_sequence": "2",
                },
                {
                    "shape_id": "shape1",
                    "shape_pt_lat": "51.001",
                    "shape_pt_lon": "-7.999",
                    "shape_pt_sequence": "3",
                },
            ],
        }
        if overrides:
            tables.update(overrides)

        path = tmp_path / "gtfs.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for filename, rows in tables.items():
                if filename in omit:
                    continue
                if rows:
                    fieldnames = list(rows[0])
                elif filename in tables:
                    fieldnames = []
                else:
                    fieldnames = []
                lines = []
                if fieldnames:
                    lines.append(",".join(fieldnames))
                    for row in rows:
                        lines.append(",".join(str(row.get(k, "")) for k in fieldnames))
                zf.writestr(filename, "\n".join(lines) + "\n")
        return path

    return _build
