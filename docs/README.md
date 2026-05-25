# IRM2 documentation

Sphinx source tree for the IRM2 user manual + API reference.

## Build

```bash
pip install -r docs/requirements.txt
cd docs
make html
open build/html/index.html
```

## Layout

```
docs/
├── Makefile
├── requirements.txt
└── source/
    ├── conf.py
    ├── index.rst
    ├── overview.md
    ├── install.md
    ├── quickstart.md
    ├── concepts/
    │   ├── dataflow.md
    │   ├── types.md
    │   ├── pricing.md
    │   └── training.md
    ├── examples/
    │   ├── 01_load_market_data.md
    │   ├── 02_build_and_train_model.md
    │   ├── 03_evaluate_and_inspect.md
    │   └── 04_futures_pricing_walkthrough.md
    └── api/
        ├── configs.rst
        ├── dataloaders.rst
        ├── models.rst
        ├── finance.rst
        ├── training.rst
        └── types.rst
```

API pages are auto-generated from in-code docstrings (NumPy style — `napoleon`
extension). Concept pages are hand-written Markdown via `myst_parser`. Add new
examples under `source/examples/` and link them from `source/index.rst`.
