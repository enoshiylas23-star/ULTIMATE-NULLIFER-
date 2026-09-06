# Nullius

*Nullius in verba* — "on no one's word." The Royal Society's motto, adopted as an engineering
ethos: **do not take claims on authority. Verify them.**

Nullius is an open-source, locally executable system intended to behave less like a chatbot and
more like a **computational scientist**: it formalizes questions, forms hypotheses, checks whether
statistical assumptions hold, runs experiments, tries to falsify its own conclusions, and reports
honestly about uncertainty.

**Working concept:** *Open Autonomous Data Scientist.*

> The objective is **not** to build a chatbot that generates data-science code.
> The objective is an autonomous scientific reasoning and experimentation system that treats a
> wrong-but-confident answer as a failure and an honest "insufficient evidence" as a success.

## Status

| Phase | Component | Status |
| --- | --- | --- |
| 0 | Architecture & interfaces | ✅ Designed — see [`docs/`](docs/README.md) |
| 1 | **Data Understanding Engine (data profiler)** | ✅ Implemented & tested |
| 2 | Statistical analysis engine | ⬜ Planned — see [roadmap](docs/06-development-roadmap.md) |

Nothing outside this table is claimed to work. See the [roadmap](docs/06-development-roadmap.md)
for the development order and the no-faking rule.

## Design principles

1. **Correctness over appearance.** Never optimize for an impressive-looking answer.
2. **Statistical validity first.** Assumptions are checked before methods are trusted; conclusions
   are reported with uncertainty.
3. **Reproducibility by construction.** Dataset hashes, seeds, configurations, and dependency
   versions travel with every experiment.
4. **Traceability.** Every important conclusion is traceable to evidence: assumptions, data,
   transformations, experiments, algorithms.
5. **Local-first.** Private data never has to leave the machine. LLMs are optional plug-ins, not
   the engine.
6. **Falsification.** The system attempts to disprove its own conclusions (a built-in critic),
   and refuses causal claims the analysis cannot support.
7. **No pretending.** An unfinished component is documented as unfinished. (Engineering rule 22.)

## Quick start (Phase 1)

Phase 1 is the **Data Understanding Engine**: given a dataset it produces a `Data Understanding
Report` — schema, missingness, duplicates, distributions, outliers, skew, categorical imbalance,
correlations, identifier candidates, and *heuristic* leakage/proxy flags — without ever modifying
the original data.

```bash
# From the repository root (no install required):
python -m nullius data profile --help

# Profile a CSV, optionally naming a target column for imbalance/leak heuristics:
python -m nullius data profile path/to/data.csv --target churn

# Keep the machine-readable report:
python -m nullius data profile path/to/data.csv --out report.json
```

Programmatic use:

```python
import pandas as pd
from nullius.data import profile_dataframe

df = pd.read_csv("data.csv")
report = profile_dataframe(df, name="customer_data", target="churn")
print(report.render_markdown())
```

The report is a structured `DataUnderstandingReport` (JSON-serializable) plus a human-readable
markdown rendering. Nullius analyzes a read-only snapshot: **no transformation is ever applied to
the input frame**, and every numerical method used is documented with its assumptions.

## Repository layout

```
docs/               Architecture, interfaces, and design decisions
src/nullius/        Package source (src layout)
tests/              Test suite, including failure-first adversarial datasets
```

Source package layout (current and planned):

```
src/nullius/
  core/          Domain records shared across components (planned)
  data/          Phase 1: Data Understanding Engine  ✅
  stats/         Phase 2: statistical analysis engine (planned)
  math/          Phase 5: mathematical reasoning layer (planned)
  models/        Baseline ML engine (planned)
  experiments/   Experiment registry and lifecycle (planned)
  critic/        Model critic + robustness engine (planned)
  memory/        Persistent experiment memory (planned)
  execution/     Sandboxed code execution (planned)
  reports/       Report generation (planned)
  llm/           Provider adapters — optional, engine works without them (planned)
```

## Why "Nullius"?

The original motto is a statement of method, not of skepticism for its own sake: evidence is
produced by experiment and demonstration, then published so others may verify it. Nullius the
system does the same — it prefers "insufficient evidence" over a confident wrong answer, and
"these assumptions are violated" over "here is a beautiful model."

## License

Apache-2.0. See [LICENSE](LICENSE).
