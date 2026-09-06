# 02 · Component interfaces

This document defines the typed contracts between components. It is written **before**
implementation of the later phases so that Phase 1 code and future phases share vocabulary.
Signatures are normative for planning; dataclasses will live in `nullius/core/records.py` and
grow with each phase. Python 3.11+ syntax.

## 1. Core domain records (`nullius/core`)

### 1.1 Dataset

```python
@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str            # stable logical id, e.g. "customer_data"
    version: str               # "v1", "v2", … or content-hash short
    sha256: str                # content hash of the canonical source file
    n_rows: int
    schema: tuple[ColumnSpec, ...]
    source_path: str | None    # provenance; None if built in-session
    created_utc: datetime

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    stored_dtype: str          # as read from file (e.g. "str", "float64")
    inferred_kind: ColumnKind  # NUMERIC | CATEGORICAL | BOOLEAN | DATETIME | TEXT | IDENTIFIER | OTHER
    n_missing: int
    n_unique: int
```

`ColumnKind` and the Phase-1 profile records live in `nullius/data/` today and migrate to `core`
when a second consumer needs them (no premature centralization).

### 1.2 Hypothesis (spec §4)

```python
@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str            # e.g. "Longer contract duration is associated with lower churn"
    null: str
    alternative: str
    variables: tuple[str, ...]
    target: str | None
    confounders: tuple[str, ...] = ()
    method_plan: MethodRecord | None = None   # chosen statistical method + why
    expected_evidence: str = ""
    # outcome fields filled by the experiment engine:
    result: str | None = None
    effect_size: float | None = None
    effect_size_type: str | None = None
    uncertainty: Interval | None = None
    conclusion: str | None = None
    status: Literal["proposed", "tested", "rejected", "supported_with_caveats", "inconclusive"] = "proposed"
```

Rules: hypotheses are **not** generated from correlations alone (spec §4); each carries its null,
its confounders, and the assumptions its chosen method requires.

### 1.3 Experiment (spec §5) — see doc 04 for lifecycle

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis_id: str | None
    dataset_version_id: str
    features: tuple[str, ...]
    target: str
    preprocessing: PipelineSpec          # hashable, recorded, versioned
    algorithm: str
    hyperparameters: Mapping[str, Any]
    validation_strategy: ValidationSpec  # IID/grouped/temporal + k + seed
    random_seed: int
    metrics: tuple[str, ...]
    statistical_tests: tuple[str, ...] = ()
    # results appended as ExperimentResult (mutable registry record)
```

### 1.4 MethodRecord (spec §1) — see doc 03

Every mathematical/statistical method used records: METHOD, ASSUMPTIONS, WHY IT APPLIES,
MATHEMATICAL BASIS, LIMITATIONS, DATA REQUIREMENTS, RESULT, CONFIDENCE/UNCERTAINTY.

### 1.5 Conclusion

```python
@dataclass(frozen=True)
class Conclusion:
    conclusion_id: str
    statement: str                     # already filtered by language policy
    evidence: tuple[EvidenceLink, ...] # traceability: every claim → evidence
    stability: Literal["stable", "sensitive", "insufficient_evidence"]
    caveats: tuple[str, ...]
    verdict: Literal["supported", "not_supported", "inconclusive"]
```

## 2. Component contracts

Each component below exposes one or two public entry points; everything else is private.

### 2.1 Data Understanding Engine — `nullius.data` (Phase 1, implemented)

```python
def profile_dataframe(
    df: pd.DataFrame,
    *,
    name: str | None = None,
    source_path: str | None = None,
    target: str | None = None,
    max_correlation_columns: int = 200,
) -> DataUnderstandingReport: ...

def hash_file(path: Path) -> str            # sha256 of raw bytes
def load_dataset(path: Path, **kwargs) -> tuple[pd.DataFrame, str]  # + sha256
```

Output: `DataUnderstandingReport` (JSON-serializable, markdown-renderable). It never mutates the
input; every numeric method is documented with assumptions (spec §3).

### 2.2 Statistical engine — `nullius.stats` (Phase 2)

```python
def describe_estimand(...)                    # formalize target/estimand
def assumption_checks(series_or_model, ...)   # distributional/independence checks
def confidence_interval(estimator, method, ...) -> Interval
def hypothesis_test(test_spec, data, ...) -> TestResult   # + assumption check log
def effect_size(...)
def power_analysis(...)
def bootstrap_ci(...)
def multiple_testing_adjust(method, pvalues, ...) -> AdjustedPValues
```

Rule: a `TestResult` is **not emitted** unless its assumption-check log passes (or violations are
explicitly recorded and severity-assessed) (spec §2).

### 2.3 Model layer — `nullius.models` (Phase 3)

```python
def suggest_baselines(task_type, n, feature_structure, ...) -> list[ModelSpec]
def fit_and_evaluate(spec, data_version, ...) -> ModelResult   # out-of-sample only
def calibration_report(...) / learning_curve(...)
def select(...)  # model selection with uncertainty, not just point metrics
```

Selection policy (spec §6): baselines first; complexity justified by measured gain; algorithm
choice informed by data size, dimensionality, target type, temporal structure, interpretability
requirements, deployment environment.

### 2.4 Experiment manager — `nullius.experiments` (Phase 4)

```python
class ExperimentRegistry:
    def register(spec: ExperimentSpec) -> experiment_id   # immutable after registration
    def start(id) / def append_result(id, result)         # no spec mutation
    def get(id) / def list(filters) / def quarantine_test(...)
    def guard_no_test_contamination(features, dataset)    # raises on quarantined touch
```

### 2.5 Mathematical reasoning layer — `nullius.math` (Phase 5; see doc 03)

```python
def choose_method(question_kind, data_props, target_kind, ...) -> MethodRecord
def check_assumptions(method: MethodRecord, data_props) -> AssumptionReport
def ontology_lookup(concept)   # e.g. "condition number", "stationarity", "confounder"
```

### 2.6 Orchestrator / planner — `nullius.agents` (Phase 6)

```python
def formalize(question: str, data_report, ...) -> AnalysisPlan   # OBJECTIVE/TARGET/ASSUMPTIONS/…
class AnalysisPlan: ...        # plan is a document + structured spec, human-editable
def execute_plan(plan, registry, ...) -> PlanResult
```

The LLM, when present, proposes plans via `llm.Provider`; the deterministic engine executes them.
Plans are versioned and re-runnable without the LLM.

### 2.7 Critic — `nullius.critic` (Phase 8)

```python
class Critic:
    def attempt_falsification(analysis, dataset_version, ...) -> Critique
@dataclass Critique:
    findings: tuple[CritiqueFinding, ...]     # leakage, overfitting, confounding, drift, …
    verdict: Literal["passed", "failed", "caveats"]
```

The critic is **independent**: it re-reads the data and the pipeline manifest, and does not share
the analyst's fitted state. Robustness engine:

```python
def robustness_sweep(analysis, *, seeds, splits, feature_removals, model_families, ...) -> RobustnessReport
# verdict: "Conclusion stable" | "Conclusion sensitive to assumptions"
```

### 2.8 Memory — `nullius.memory` (Phase 9)

```python
class Memory:
    def remember(record)                 # datasets, hypotheses, experiments, failures
    def query(kind, filters) -> list
    def dont_repeat(failure_signature) -> list[prior_failure]   # consult before experiments
```

Failure memory is consulted by the planner **before** proposing experiments (spec §11).

### 2.9 Execution sandbox — `nullius.execution` (Phase 7; see doc 05)

```python
class Sandbox:
    def run(plan: CodePlan, *, resources=..., network=False, allowlist=..., timeout=...) -> RunResult
```

### 2.10 LLM adapters — `nullius.llm` (Phase 6)

```python
class Provider(Protocol):
    name: str
    def complete(self, prompt, *, temperature=0.0, json_schema=None) -> Completion
    def available(self) -> bool
def load(config) -> Provider     # "ollama:llama3", "openai:gpt-4o", "none" → NullProvider
```

`NullProvider` returns structured "no model configured" responses so the engine degrades
gracefully. Adapters are thin: they translate messages only; no science logic lives here.

## 3. Cross-cutting interface rules

1. Every public function that consumes data takes a `DatasetVersion` or validates a sha256, never
   an anonymous frame, once past Phase 1.
2. Randomness is explicit: seed is an argument or part of the spec; no global `np.random`.
3. Time/versions: `created_utc` on all records; registry is append-only.
4. JSON round-trip: all core records implement `to_dict()`/`from_dict()` (or dataclass
   `asdict` + schema check) so memory, reports, and the API share one serialization.
5. Optional imports: scikit-learn/statsmodels/torch/LLM SDKs are imported lazily; importing
   `nullius.data` must not require them.
