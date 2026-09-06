# 03 · Mathematical reasoning layer

## 1. Purpose

A dedicated layer that applies mathematical theory *when it is relevant*, rather than treating
every problem as a generic machine-learning problem. It covers the areas listed in spec §1:
linear algebra, calculus, probability, statistics, information theory, optimization, ML theory,
causal inference, time series, numerical methods.

Two hard rules:

1. **No name-dropping.** Invoking a concept "to sound intelligent" is a bug. Every concept used
   must change what the system does: a check it runs, a method it selects, a limitation it reports.
2. **Every method is recorded.** For each method used, the system records METHOD, ASSUMPTIONS,
   WHY IT APPLIES, MATHEMATICAL BASIS, LIMITATIONS, DATA REQUIREMENTS, RESULT,
   CONFIDENCE/UNCERTAINTY (spec §1).

## 2. MethodRecord

```python
@dataclass(frozen=True)
class MethodRecord:
    method: str                    # canonical name, e.g. "Welch's t-test"
    why_it_applies: str            # matched to question kind + data properties
    mathematical_basis: str        # reference + formula-level sketch
    assumptions: tuple[Assumption, ...]   # each: statement, testable? how?
    limitations: tuple[str, ...]
    data_requirements: tuple[str, ...]    # sample size, measurement type, missingness…
    result: str | None = None
    confidence: Interval | None = None
    uncertainty: str | None = None
```

`Assumption` carries `severity_if_violated` so the assumption checker can distinguish
"violation invalidates the method" from "violation degrades it mildly."

## 3. Concept ontology

The layer maintains a machine-readable ontology over the areas of spec §1. Each concept node
records: definition; when it is *relevant* (trigger conditions, e.g. "multicollinearity is
relevant when fitting linear models with |corr| > 0.9 or condition number > 30"); which methods
depend on it; which checks test it. Examples of intended entries:

- linear algebra: rank, conditioning (`cond(A)`), positive-definiteness (needed for valid
  covariance → Mahalanobis distances, Gaussian assumptions), SVD (PCA whitening, low-rank
  structure), projection (least squares geometry, residual orthogonality);
- probability: likelihood (model comparison), posterior (Bayesian updating),
  conditional expectation (regression as E[Y|X] under squared error);
- statistics: estimator bias/variance/consistency (when reporting means/effects), CLT
  applicability (n, independence, tail behavior), bootstrap validity (exchangeability);
- information theory: entropy/KL/MI (feature screening, categorical association, drift
  detection — e.g. PSI/KL between train and serving distributions);
- optimization: convexity (guarantees), regularization (bias–variance), convergence
  diagnostics (gradient norms, dual gaps);
- ML theory: bias–variance, generalization (→ honest CV, nested CV), calibration
  (probability outputs), class imbalance (metric choice, not resampling-by-default);
- causal inference: confounders, selection bias, DAG assumptions, backdoor adjustment,
  instruments; **what cannot be identified from the data alone**;
- time series: stationarity (ADF/KPSS role), ACF/PACF (order selection), temporal leakage
  (walk-forward only; no random CV);
- numerical methods: floating-point stability, conditioning, overflow in softmax/log-sum-exp,
  catastrophic cancellation, convergence diagnostics.

## 4. Assumption checking (Theory Validator)

Checks are staged:

1. **Pre-method checks** (before a test/estimator/model is trusted):
   - distributional assumptions (shape, discreteness, boundedness);
   - independence / grouped / temporal structure (→ split strategy);
   - sample size vs method requirements;
   - variance assumptions (homoscedasticity for classical tests);
   - measurement type (nominal/ordinal/interval/ratio) vs method;
   - missingness mechanism assumptions (MCAR/MAR/MNAR) vs chosen imputation.
2. **Post-fit diagnostics** for models (before interpretation):
   - residuals: shape, heteroskedasticity, autocorrelation (when temporal);
   - multicollinearity (VIF / condition number);
   - influential observations (Cook's distance / DFBETAS);
   - specification checks (functional form, omitted variables where testable).
3. **Validation-strategy checks** before CV: IID? grouped (→ GroupKFold / leave-group-out)?
   temporal (→ blocked/walk-forward)? imbalanced (→ stratified where valid)?
4. **Causal-claim gate**: before any causal wording is emitted, the system must exhibit the
   explicit causal assumptions (DAG or the identification argument, confounders, selection
   mechanism) and check identifiability. Otherwise the language policy in §6 applies.

Violations are not fatal by default: they downgrade the *claim*, not necessarily the *analysis*.
The report must state what is still valid under the violated assumptions, or say nothing is.

## 5. Language policy (spec §2, §20)

The system never states *"X causes Y"* unless the analysis supports a causal interpretation
(design or explicit identification assumptions + sensitivity analysis). Permitted phrasings:

- "X is associated with Y under the assumptions of this analysis."
- "Under the causal model M (assumptions listed), the estimated effect of X on Y is …, with
  uncertainty …; the estimate is sensitive to assumption A (see robustness)."
- "This study cannot establish causality from this data."

Forbidden phrasings: "X causes Y" from cross-sectional association; "the model proves…";
significance = importance; "this feature is the most important driver" (unless effect/
importance measured and uncertainty reported — and never as causal).

## 6. Statistical honesty rules

1. **Context for every metric**: metric + CI + baseline + sample size. A single number without
   context is a bug (spec §7).
2. **Significance ≠ practical importance**: report effect sizes and raw units; a tiny effect
   with p < 0.05 is reported as tiny, with the decision-relevant interpretation.
3. **Multiple testing**: multiplicity accounted for when a hypothesis family is explored
   (Bonferroni–Holm/BH with the family defined in the plan; family is registered before testing).
4. **Power & sample size**: when a null result is reported as evidence of "no effect", the system
   must report whether the design had power to detect a meaningful effect (equivalence framing
   preferred: report the effect size the data can rule out).
5. **Uncertainty everywhere**: CIs/bootstraps for estimators; predictive intervals for models;
   calibration curves for probabilities.
6. **Pre-registration discipline**: hypotheses and analysis plans are registered before results
   are seen where feasible; post-hoc analyses are labeled as exploratory, and their conclusions
   cannot be stated with confirmatory strength.
7. **"Insufficient evidence" is a first-class result.**

## 7. When the math layer refuses

The layer refuses (returns `insufficient_evidence` / `assumptions_violated`) when:
- the method's core assumptions fail and no robust alternative exists;
- sample size cannot support the requested inference;
- the target of inference is not identifiable from the data and design;
- the data cannot answer the question (wrong level of aggregation, e.g. ecological inference for
  an individual-level claim; no counterfactual variation for a treatment effect).

Refusal is a designed outcome, not a failure mode.
