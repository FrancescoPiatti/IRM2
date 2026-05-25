# full_test_run

End-to-end smoke runs that exercise the **full training loop**, the
**Optuna grid search**, and the **result analyser** against the bundled
`data2/` corpus.

These are **not** picked up by the default pytest collection — the file
names don't match `test_*.py`, so `pytest tests/` ignores this folder.
They are meant to be invoked **by hand**, since each one takes several
minutes to hours depending on the configuration.

## Available scripts

| Script | What it exercises |
|---|---|
| `fullrun_SimpleSimple_diag.py` | Simple encoder + simple Neural SDE, diagonal noise. |
| `fullrun_SimpleSimple_gen.py`  | Simple encoder + simple Neural SDE, general noise. |
| `fullrun_HierOU_gen.py`         | Hierarchical encoder + OU Neural SDE, general noise. |
| `grid_SimpleSimple_v1.py`       | Optuna grid search over a small parameter grid. |

## How to run

From the repository root:

```bash
python -m tests.full_test_run.fullrun_SimpleSimple_diag
python -m tests.full_test_run.fullrun_SimpleSimple_gen
python -m tests.full_test_run.fullrun_HierOU_gen
python -m tests.full_test_run.grid_SimpleSimple_v1
```

Each script writes its artefacts under `results/<run_name>` and prints a
short summary via `ResultAnalyzer` at the end.
