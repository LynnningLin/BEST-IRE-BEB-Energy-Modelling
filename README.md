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

## Notes

- Put original datasets in `data/raw/`.
- Put generated or cleaned datasets in `data/processed/`.
- Data files are ignored by Git by default to avoid accidentally committing
  large or sensitive files.
