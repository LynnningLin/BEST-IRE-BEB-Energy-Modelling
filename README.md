# BEST-IRE BEB Energy Modelling

Python project for battery electric bus (BEB) energy modelling in the BEST-IRE
project.

## Project Structure

```text
.
├── configs/              # Model and scenario configuration files
├── data/
│   ├── raw/              # Original input data, not committed
│   └── processed/        # Cleaned/generated data, not committed
├── notebooks/            # Exploratory analysis notebooks
├── scripts/              # Command-line utilities and one-off workflows
├── src/best_ire_beb/     # Reusable Python package code
└── tests/                # Automated tests
```

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project with development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional local settings:

```bash
cp .env.example .env
```

## Check The Setup

Run tests:

```bash
pytest
```

Run formatting and lint checks:

```bash
black --check src tests
ruff check src tests
```

## Configuration

Project defaults live in `configs/model.yaml`. This file now owns the shared
model settings, including BEB vehicle and battery parameters, GTFS input paths,
passenger-loading data paths, weather data paths, climate-control/HVAC settings,
SRTM cache location, and output paths.

Paths in the YAML file are resolved relative to the project root unless they
are absolute. You can point scripts at a different config with `--config`:

```bash
python scripts/gtfs_to_segment.py 208 --config configs/model.yaml
python scripts/beb_soc_model.py --config configs/model.yaml
```

Routes are selected by short name or exact `route_id`. Operators can share a short
name, so an ambiguous name is rejected: `W4` matches both Waterford and Dublin, and
the configured Waterford route is `WFRD W4 c b`. Route CSV filenames follow the
selector, so that route writes `route_WFRD_W4_c_b_trips_<date>.csv`.

`gtfs.shape_repairs` removes phantom road from GTFS shapes before distances are
measured. Each entry names two coincident shape points, so a feed update fails
loudly rather than cutting real road. `--compare-fidelities` also refuses any
interval scheduled above 120 km/h, which signals a length defect.

OSM signal and speed-cap fetches use the `simulation.osm_snapshot_utc` map state.
A failed fetch is never cached as data: the run warns, uses the configured estimate
for that run only, and retries next time. `--compare-fidelities` refuses to run on
those estimates.

Signal counts come from `scripts/signal_inventory.py`, which matches each shape to
the directed OSM road graph and counts the distinct stop points a bus must obey on
one traversal. It writes the signal caches and the placement registry, working
offline from the Overpass responses saved in `paths.osm_responses_dir`:

```bash
python scripts/signal_inventory.py --date 2026-07-09
```

With `traffic_signals.inventory_frozen: true` a run stops on any stop pair outside
that inventory, so the older proximity count cannot mix in. Set it to false to run
other routes on the proximity count.

GTFS runs save all selected routes under `paths.gtfs_output_dir` in a folder
named `traffic_signals_off` when signals are disabled, or
`traffic_signals_on_<motion.signal_fidelity>` when enabled (for example,
`data/processed/traffic_signals_on_F2_spatial/`). CLI signal overrides also apply
to the folder name. Route CSV filenames keep their route and date; optional
signal-event CSVs are saved alongside them. Repeating the same settings and date
overwrites the corresponding CSVs.

The new `F1-U` and `F1-J` modes use canonical-pattern placements from an explicit
accepted registry (`paths.signal_placement_registry`) and require
`motion.solver_mode: trip_continuous`. `F1_frequency` retains the historical
per-trip algorithm. Use `--compare-fidelities --solver-mode trip_continuous` to
write matched F0/F1-U/F1-J/F2 results, including unavailable cells. New U/J outputs
also distinguish placement and behaviour indices; unknown inventories are
reported as blocked. See the local `docs/signal_frequency_fidelity.md` for the
registry contract and reproducible constructed examples.

## Notes

- Put original datasets in `data/raw/`.
- Put generated or cleaned datasets in `data/processed/`.
- Local documentation (`docs/`), investigation snapshots (`convergence_test/`),
  generated outputs, report figures, elevation tiles, caches, and temporary test
  runs are ignored by Git. These files can remain on disk for local reference.
- Reusable model code and automated regression tests remain part of the project.
