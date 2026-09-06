# 01 · System architecture

## 1. Vision

Nullius is an open-source, locally executable **autonomous data-science system**. Given a dataset
and a natural-language research/business question it should be able to:

understand the data → formalize the question → hypothesize → check assumptions → design
experiments → fit and compare models → validate rigorously → **attempt to falsify its own
conclusions** → produce reproducible code and a scientific report with honest uncertainty.

It is *not* a code-generating chatbot. Code generation is incidental; scientific method is the
product. The design target is that an incorrect result can be investigated and traced back to the
assumptions, data, transformations, experiments, or algorithms that produced it.

## 2. Non-negotiable properties

| Property | Consequence |
| --- | --- |
| **Correctness** | Wrong-but-confident answers are the worst failure mode. Honest "insufficient evidence" is a success. |
| **Statistical validity** | Assumptions checked before methods trusted; significance ≠ practical importance; effect sizes + uncertainty reported. |
| **Reproducibility** | Every artifact carries dataset hash, seeds, config, dependency versions, pipeline code. |
| **Traceability** | Every conclusion links to evidence records (assumptions → data → transformation → experiment → result). |
| **No leakage/overfitting** | Quarantined holdout; registry prevents repeated tuning against the test set. |
| **Transparency** | Reasoning artifacts are inspectable; LLM prose is never the source of truth. |
| **Local-first** | Core engine runs with no LLM and no network. Cloud is optional. |
| **Honesty** | "These assumptions are violated" beats "here is a beautiful model". Unfinished components are documented as unfinished. |

## 3. Scope boundaries

Nullius **does**:

- descriptive and inferential analysis with assumption checking;
- baseline-driven predictive modeling with rigorous validation;
- causal *analysis* (assumption-explicit, language-disciplined) — never unqualified causal claims;
- experiment management, memory, critic/robustness passes, reproducible reporting;
- monitored deployment of models *when the user confirms*.

Nullius **does not**:

- replace the judgment of a human statistician for consequential decisions;
- run arbitrary generated code outside its sandbox (see security model);
- delete data, deploy, or publish without explicit confirmation;
- claim capabilities without benchmarks.

## 4. System context

```
                        ┌──────────────────────────────────────────────┐
                        │                  USER / HOST                  │
                        │   question (NL) · dataset · confirmations    │
                        └───────────▲──────────────────▲───────────────┘
                                    │ reports / asks   │ confirm gates
┌─────────────────────────────┐     │                  │     ┌─────────────────────────┐
│         ORCHESTRATOR        │◄────┴──────────────────┴─────│    HUMAN GATE (HITL)    │
│  (planner: phases 6–7)      │                              └─────────────────────────┘
└──────┬──────┬──────┬────────┘
       │      │      │
┌──────▼──┐ ┌─▼──────┴─────┐ ┌───────────────▼───────────────┐
│ DATA    │ │ STATISTICS / │ │  EXPERIMENT ENGINE / REGISTRY  │
│ layer   │ │ MATH layer   │ │  (modeling, validation)        │
│ (ph.1)  │ │ (ph.2,5)     │ └──────┬───────────────┬─────────┘
└────┬────┘ └──────┬────────┘       │               │
     │             │                │               │
┌────▼─────────────▼─────────┐ ┌────▼─────────┐ ┌───▼──────────────┐
│  MEMORY (ph.9)             │ │ CRITIC (ph.8)│ │ EXECUTION SANDBOX│
│  datasets·hypotheses·      │ │ robustness   │ │ (ph.7+, secure)  │
│  experiments·failures      │ │ falsification│ │                  │
└────────────────────────────┘ └──────────────┘ └──────────────────┘
   (optional)  ┌────────────────────────┐
   LLM layer   │ llm/ adapters — local  │   Reasoning engine works
               │ or remote providers    │   with NO LLM attached.
               └────────────────────────┘
```

- The **core data-science engine** (everything except `llm/` and parts of the orchestrator) runs
  without any LLM. It is a library + CLI + deterministic pipeline engine.
- The **orchestrator** (Phase 6–7) may be LLM-driven *planning* on top of deterministic,
  validated primitives. Plans are stored, reviewed, and executed through the same registry as
  everything else — an LLM never executes code directly, it proposes plans.

## 5. Component architecture

Modules below are listed with the phase that introduces them (see roadmap doc 06). Contracts for
each component live in doc 02.

| Module | Responsibility | Phase |
| --- | --- | --- |
| `data/` | Data Understanding Engine: profiling, quality, leakage heuristics, versioning hooks | 1 ✅ |
| `stats/` | Statistical engine: tests, estimators, CIs, bootstrap, power, effect sizes | 2 |
| `models/` | Baseline + candidate ML families, calibration, learning curves | 3 |
| `experiments/` | Experiment registry, lifecycle state machine, contamination guards | 4 |
| `math/` | Mathematical reasoning layer: concept ontology, assumption checker, method records | 5 |
| `core/` | Domain records shared everywhere (`DatasetVersion`, `Hypothesis`, `MethodRecord`, …) | 1+ (grows) |
| `agents/` or orchestrator | Planner: NL question → formal analysis specification | 6 |
| — (loop) | Autonomous loop PLAN→EXECUTE→OBSERVE→CRITIQUE→MODIFY | 7 |
| `critic/` | Independent critic + robustness engine | 8 |
| `memory/` | Persistent experiment memory | 9 |
| `deploy/` | Deployment + monitoring (drift, concept drift) | 10 |
| `reports/` | Scientific report builder | grows 2→10 |
| `execution/` | Sandboxed code execution, reproducibility manifests | 7 |
| `llm/` | Provider adapters (multi-provider, local-first) | 6 |
| `ui/`, `api/` | Interfaces on top of the core | later |
| `cli/` (`nullius.cli`) | Human interface to every phase as it lands | 1 ✅ |

Design rule: **modules depend downward on `core` records and on deterministic compute
(`stats`, `models`, `math`); no module reaches into the LLM layer except the planner.**
This keeps the science trustworthy regardless of model provider.

## 6. Data flow (end-to-end target)

```
 question + dataset
   │
   ▼
┌─────────────────────────── NL INTERFACE (ph.6) ───────────────────────────┐
│ Display before major analysis: OBJECTIVE · TARGET · ASSUMPTIONS ·         │
│ DATA REQUIRED · ANALYSIS PLAN · EXPERIMENT PLAN  ──── human may edit      │
└───────────────────────────┬───────────────────────────────────────────────┘
   │
   ▼
1. REGISTER DATASET        → DatasetVersion{n, schema, sha256}          (mem: ph.9)
   │
   ▼
2. DATA UNDERSTANDING      → Data Understanding Report; quality flags    (data: ph.1 ✅)
   │                         no mutation of source
   ▼
3. FORMALIZE QUESTION      → objective, target, estimand, assumptions    (core records)
   │
   ▼
4. HYPOTHESIZE             → H1..Hk {null, alt, variables, confounders,  (ph.6; spec §4)
   │                         method, assumptions, expected evidence}
   │
   ▼
5. ASSUMPTION CHECK        → theory validator; distributional checks;    (math: ph.5, stats: ph.2)
   │                         independence, sample size, leakage review
   ▼
6. DESIGN EXPERIMENT       → ExperimentSpec registered BEFORE running     (experiments: ph.4)
   │                         validation strategy chosen from data structure
   │                         (IID? grouped? temporal?) — never default KFold blindly
   ▼
7. EXECUTE                 → sandboxed; manifest {code, versions, seed,  (execution: ph.7)
   │                         dataset hash}; metrics + uncertainty
   ▼
8. CRITIQUE & ROBUSTNESS   → critic attempts falsification (leakage,     (critic: ph.8)
   │                         overfitting, assumption violations, drift);
   │                         robustness sweep across seeds/splits/models
   ▼
9. CONCLUDE                → only if stability gate passes; language     (reports, core)
   │                         policy: association ≠ causation
   ▼
10. REPORT + MEMORY        → scientific report (16 sections, spec §16);  (reports ph., mem ph.9)
                             update experiment memory: successes,
                             failures, rejected hypotheses
```

Deployment (ph.10) is a separate, human-gated pipeline consuming an approved model artifact and a
monitoring plan.

## 7. Technology choices

Rationale in doc on interfaces; summary:

| Concern | Choice | Why |
| --- | --- | --- |
| Language | Python ≥ 3.11 | Scientific ecosystem; team familiarity; single-language system |
| Tabular compute | pandas + numpy | Phase-1 profiling; later candidate: polars for >RAM data |
| Statistics | scipy | Broad, maintained distribution/test coverage |
| Modeling (later) | scikit-learn first; statsmodels for inference; optional lightgbm/xgboost; torch optional | Baseline-first policy; interpretability before complexity |
| Experiment tracking | Own registry on disk (JSONL + hashed artifacts) | No service dependency; local-first; trivially inspectable |
| LLM | Adapters; OpenAI-compatible + llama.cpp/Ollama local first | Engine never depends on a provider |
| Execution | Subprocess sandbox w/ resource limits + fs allowlist | See security model |
| Tests | pytest | Standard |
| Packaging | pyproject/setuptools, src layout | Standard |

Rule: **no new runtime dependency enters without (a) a reason, (b) a license check, and
(c) a benchmark or test justifying it.** LLMs are never a hard dependency.

## 8. Repository structure

See root README. `src/nullius/` mirrors the component table; `docs/`, `tests/`, plus a planned
`examples/` and `benchmarks/` (ph. 2+). Datasets for failure-first tests live in
`tests/fixtures/` as *generators* (code), not committed data blobs.

## 9. Versioning and compatibility

- SemVer. Pre-1.0: minor = breaking.
- The public surface is the `core` records + the component entry points documented in
  doc 02; everything else is private to its package.
- Every release records dependency pins (`uv lock`-style manifest or pip freeze artifact) in the
  reproducibility manifest schema.
- Backwards compatibility: a result from Nullius 0.x must be reproducible with Nullius 0.(x+1)
  when the manifest pins are honored; migration tooling where records change.

## 10. Anti-principles (what we refuse to do)

- Report a single metric with no uncertainty or baseline.
- Tune repeatedly on the final holdout.
- Claim causality from associational data.
- Print p-values without checking their assumptions.
- Let an LLM execute code it wrote, unchecked.
- Say a component works when it has no tests.
- Optimize for how the report *looks*.
