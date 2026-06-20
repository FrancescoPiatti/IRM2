# Installation

## Runtime

```bash
# Inside your virtualenv
pip install torch torchsde pandas numpy matplotlib optuna
```

The repository's existing `irm/` venv already contains these dependencies.

## Building the docs

```bash
pip install -r docs/requirements.txt
cd docs
make html
# Open docs/build/html/index.html in a browser
```

## Running the tests

```bash
pip install pytest
pytest tests/ -q
```

The test suite is intentionally fast (a couple of seconds) and exercises the
data loader, type contracts, configs, neural backbones, BondNet, the pricer
(including a hand-checked CTD reduction), and `ShortRateModel`. It does
**not** run training.

## Repository layout

```
IRM2/
├── data/                   CSV market data
├── docs/                    this Sphinx documentation
├── experiments_fra/         end-to-end training scripts (expensive)
├── src/                     library code
├── tests/                   pytest suite
├── CLAUDE.md                Claude Code's operating guide
├── project_description.md   mathematical / algorithmic blueprint
├── review_notes.md          known issues + quality notes
├── future_pricing_plan.md   design decisions behind the futures pricer
└── optimization_plan.md     prioritised performance backlog
```
