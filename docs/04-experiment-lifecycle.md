# 04 · Experiment lifecycle

## 1. Experiment spec (spec §5)

Every experiment is registered **before** it runs with:

```
experiment_id          hypothesis        dataset_version    features
target                 preprocessing     algorithm          hyperparameters
validation_strategy    random_seed       metrics            statistical_tests
result                 uncertainty       limitations
```

Registration is append-only: an experiment spec, once written to the registry, is immutable.
Results are appended as separate records. This is what makes a later audit possible: what was
planned is always distinguishable from what happened.

## 2. Lifecycle state machine

```
                    ┌────────────────────────── CRITIQUE failed ──────────────┐
                    ▼                                                        │
PROPOSED → REGISTERED → PLANNED → EXECUTING → OBSERVED → CRITIQUED → RESOLVED
                          │            │                        │
                          └──(human edit)→ REGISTERED           ├─ critiqued-ok  → CONCLUDED
                                                               └─ critiqued-caveats → CONCLUDED_WITH_CAVEATS
```

Transitions:

| From | To | Condition |
| --- | --- | --- |
| PROPOSED | REGISTERED | spec validated (fields complete, dataset version exists, features ⊆ schema, target not in features) |
| REGISTERED | PLANNED | validation strategy chosen from data structure (IID/grouped/temporal/imbalanced) |
| PLANNED | EXECUTING | sandbox approved code plan; manifest captured |
| EXECUTING | OBSERVED | run completed; metrics + statistical tests + uncertainty recorded |
| OBSERVED | CRITIQUED | critic pass complete (doc 02 §2.7) |
| CRITIQUED | CONCLUDED | stability gate passed or explicitly waived with reasons recorded |
| CRITIQUED | (back to PLANNED) | critic found fixable flaw → modified experiment (new spec, new id) |

Continuation is always a **new experiment id** that references the parent (`derived_from`).
Memory records the lineage so failures are not blindly repeated (spec §11).

## 3. Validation strategy selection

Deterministic rules (no blind default of plain KFold):

| Data structure detected | Default strategy | Rationale |
| --- | --- | --- |
| IID, no grouping, no ordering | Stratified K-fold (classification) / K-fold (regression) | standard |
| Grouped observations (repeated subjects, clusters) | GroupKFold / leave-one-group-out | keep groups intact; else leakage via within-group correlation |
| Temporal / ordered rows | Walk-forward / blocked time-series CV (expanding or rolling) | no random CV on time series; temporal leakage guard |
| Strong class imbalance | Stratified folds; report PR-AUC, calibration; never plain accuracy | §7 metric rules |
| Very small n | Repeated nested CV + bootstrap CIs, or LOOCV with diagnostics | honest generalization error |
| Unclear structure | Profiler evidence + explicit note; strategy chosen conservatively (grouped/temporal if ambiguous) | safe default |

The chosen strategy and its rationale are part of the experiment spec and rendered in the report.

## 4. Contamination guards

1. **Test-set quarantine.** When a holdout is quarantined (`quarantine_test`), the registry
   refuses any experiment whose spec touches it. Period.
2. **No tuning on the final test set.** Model selection happens on validation folds; the final
   test is touched once, at the end, to *estimate* generalization — and then only to report,
   never to iterate.
3. **Nested validation** where selection is part of the pipeline (hyperparameter search inside
   the outer folds).
4. **Preprocessing inside folds.** Every fit/transform (imputation, scaling, feature selection,
   encoding) is learned on the training portion of each fold only. Leakage through preprocessing
   is a critic check, not a style suggestion.
5. **Temporal guard**: when rows are ordered, shuffle-based methods are rejected unless a
   justification is recorded.
6. **Feature/leak review** before any modeling: identifier columns, target duplicates, and
   post-hoc columns flagged by the profiler (Phase 1 heuristics) are surfaced to the planner.

## 5. Registry schema (JSONL on disk)

```json
{"type": "experiment_spec", "id": "exp_…", "spec": {…}, "created_utc": "…"}
{"type": "experiment_result", "id": "exp_…", "metrics": {…}, "tests": […], "created_utc": "…"}
{"type": "critique", "id": "exp_…", "findings": […], "verdict": "…"}
{"type": "dataset_version", "id": "ds_…", "sha256": "…", …}
{"type": "hypothesis", "id": "hyp_…", …}
{"type": "memory_note", "…"}         // lessons, rejected approaches
```

Append-only files under a local `runs/` or user-chosen directory; human-readable JSONL; trivially
diffable and grep-able. No server required (local-first, spec §18).

## 6. Reproducibility manifest (spec §12)

Captured automatically at EXECUTING:

- code (the exact plan executed) + package version of Nullius;
- dependency versions (pinned manifest of the environment that ran it);
- dataset sha256 + schema snapshot;
- configuration (full ExperimentSpec);
- random seeds (all sources; if an algorithm uses hidden randomness, it is pinned or rejected);
- environment: OS, python, hardware summary (CPU/GPU), locale-independent;
- preprocessing pipeline (hashable PipelineSpec);
- experiment metadata (registry id, lineage).

A researcher reproduces by: `nullius run --manifest <manifest.json>` in a pinned environment.

## 7. Human-in-the-loop gates in the lifecycle (spec §14)

Autonomous by default *inside* the loop: hypothesis generation, profiling, experiments, model
comparison, critique — all run without interruption. Confirmation is required for:

| Action | Gate |
| --- | --- |
| Deleting or overwriting user data | confirm |
| Deploying a model / touching production | confirm (separate deploy plan) |
| Any external communication | confirm |
| Publishing results | confirm |
| Quarantined test-set release for re-use | confirm + reason recorded |
| Costly compute (est. budget above user threshold) | confirm |

## 8. Gate before a conclusion is accepted

A conclusion may be promoted to the report only when:

1. the critic's falsification attempts have been run and recorded;
2. robustness verdict is not "sensitive" without explanation;
3. uncertainty is attached (CI/bootstraps);
4. language policy passed (no unsupported causality);
5. the conclusion is traceable to evidence links.

Otherwise the conclusion is recorded as `inconclusive` — which is a complete, valid outcome.
