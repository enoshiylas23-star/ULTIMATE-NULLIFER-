# 06 · Development roadmap

## 1. The rule (spec §21–22)

Build sequentially. Never attempt the whole system at once. At every phase: **write tests,
benchmark, document assumptions, create examples, measure failures, keep backwards
compatibility.** Never pretend an unfinished component works.

## 2. MVP definition

The MVP is **Phase 1 + Phase 2 + Phase 3 on tabular data**, delivered through a CLI that:

1. takes a CSV + an optional target + a natural-language question (phase 6 planner optional at MVP);
2. profiles the data and *refuses or warns* when quality problems would invalidate naive analysis;
3. runs assumption-checked statistical analyses and baseline models with honest validation;
4. emits a reproducible report + code manifest.

An MVP run on an adversarial dataset must **flag** leakage/imbalance/assumption violations rather
than produce an impressive but invalid model — that failure-first behavior is a release
criterion, not a stretch goal.

## 3. Phase plan with exit criteria

| # | Phase | Deliverable | Exit criteria (all must hold) |
| --- | --- | --- | --- |
| 1 | **Data profiler** | `nullius.data` — Data Understanding Report (schema, missingness, duplicates, distributions, outliers, skew, imbalance, correlations, identifiers, leakage heuristics) | ✅ Implemented. No mutation of input; JSON + markdown output; every adversarial fixture in §5 is flagged; CLI works on CSV/parquet; tests green |
| 2 | **Statistical engine** | `nullius.stats` — assumption checks, CIs, hypothesis tests w/ pre-checks, effect sizes, bootstrap, multiple-testing adjustment | Every emitted test result carries an assumption-check log; refusal paths tested; power framing for null results |
| 3 | **Baseline ML engine** | `nullius.models` — baselines + few families, out-of-sample evaluation, calibration, learning curves | Nested/strategy-correct validation only; results reproducible from manifest; baselines always present; leakage flags from ph.1 block bad experiments |
| 4 | **Experiment tracking** | `nullius.experiments` — registry, lifecycle, contamination guards | Registry append-only; quarantined test unreachable by tuning; temporal data cannot use random CV |
| 5 | **Math reasoning layer** | `nullius.math` — ontology, MethodRecords, assumption checker wiring into ph.2–3 | Every method in reports carries MethodRecord; assumption violations surfaced before claims |
| 6 | **AI planner** | NL question → formal AnalysisPlan; LLM adapters (local-first) | Engine works with `none` provider; plan shown before analysis (OBJECTIVE/TARGET/ASSUMPTIONS/…); plans re-run deterministically |
| 7 | **Autonomous loop** | PLAN→EXECUTE→OBSERVE→CRITIQUE→MODIFY→REPEAT with sandbox | Sandbox enforces §3 of security doc; runs are reproducible; budget limits stop runaway loops |
| 8 | **Critic + robustness** | falsification pass, robustness sweep, stability verdicts | Critic independently re-derives flaws on seeded adversarial cases; stability verdicts have tests |
| 9 | **Experiment memory** | persistent memory consulted before new experiments | Prior failures prevent repeat experiments; memory survives restart; lineage queryable |
| 10 | **Deployment + monitoring** | model artifact + drift/concept-drift monitoring plan | Human-gated deploy; monitoring uses registered baseline; drift alerts traced to artifact version |

Each phase ends with: docs updated, examples in `examples/`, benchmark entry, and the
failure-first catalog (§5) extended.

## 4. Benchmarking (spec §19)

Benchmark suite grows with each phase (`benchmarks/`). It measures Nullius against traditional
AutoML, human-written baseline pipelines, and existing open-source ML systems on:

- predictive performance (with uncertainty, on held-out data);
- statistical validity (assumption violations caught, CI coverage, calibration);
- robustness (conclusions stable across seeds/splits/subgroups?);
- experiment efficiency (experiments per correct conclusion);
- reproducibility (rerun fidelity);
- computational cost;
- error & hallucination rate (claims not supported by registry evidence);
- leakage detection rate (how many of the §5 traps were caught);
- quality of explanations (accuracy of attributed causes, calibrated importance).

**No claim of "replaces data scientists" without these measurements.**

## 5. Failure-first dataset catalog (spec §20)

Tests in `tests/` include *generators* for datasets engineered to fool naive pipelines. Each is
expected to be flagged or handled correctly; a phase is not done until its fixtures pass:

| Trap | Phase that must catch it | Test fixture |
| --- | --- | --- |
| Leakage (target duplicated under another name / derived column) | 1 (flag) · 4 (block) · 8 (critic) | `leaky_target` |
| Temporal leakage (future info in features) | 1 · 4 · 8 | `temporal_leak` |
| Missing data (incl. encoded as "" / "NA" strings) | 1 | `missing_encodings` |
| Duplicated rows | 1 | `duplicate_rows` |
| Confounding (associational effect flips with adjustment) | 2 · 5 · 8 | `confounded` |
| Class imbalance | 1 (flag) · 3 (metrics) | `imbalanced` |
| Distribution shift train/serve | 3 · 10 | `shifted` |
| Misleading correlation (nonsense but significant) | 2 · 5 (multiplicity, assumptions) | `spurious` |
| Irrelevant features (needle-in-haystack with many noise cols) | 3 · 4 | `many_noise` |
| Adversarial columns (misleading names, "ID", constant, near-constant) | 1 | `adversarial_cols` |
| Small samples | 2 · 5 | `tiny_sample` |
| High-dimensional data (p ≫ n) | 3 · 5 | `wide_data` |

These are implemented as pytest fixtures generating in-memory DataFrames (no committed data
blobs), so the suite stays fast and auditable.

## 6. Live status

| Item | Status |
| --- | --- |
| Name / identity | Nullius (working concept: Open Autonomous Data Scientist) |
| Architecture & interfaces (docs 01–06) | ✅ |
| Phase 1 data profiler | ✅ implemented; fixtures above for duplicate rows, missing encodings, adversarial cols, leaky target, imbalanced, tiny/wide covered |
| Phases 2–10 | Planned — work proceeds phase-by-phase per §3 |

## 7. What "done" looks like for each release

A release ships when: all exit criteria for its phases hold; benchmark numbers are recorded next
to the release notes; docs reflect reality; and the failure-first fixtures for those phases pass.
