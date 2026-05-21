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

## Notes

- Put original datasets in `data/raw/`.
- Put generated or cleaned datasets in `data/processed/`.
- Data files are ignored by Git by default to avoid accidentally committing
  large or sensitive files.
