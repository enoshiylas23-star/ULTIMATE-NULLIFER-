# 05 · Security model

## 1. Threat model

Nullius executes user data pipelines and (later) model training, and will — when the user asks —
write and run analysis code. The primary risks:

1. **Untrusted data.** A dataset (CSV/parquet/…) is attacker-influenced input. A column name,
   cell value, or metadata string must never be executed, and must never break out of a sandbox.
2. **Generated code.** Any code the system generates (Phase 7+) is untrusted until inspected and
   constrained. The system must never blindly execute arbitrary generated code on the user's OS.
3. **Model artifacts.** Pickle-style model files can execute code on load; treat as untrusted.
4. **LLM layer.** Remote prompts may contain user data (privacy), and LLM output may be
   manipulated (prompt injection via dataset contents or web text).
5. **Dependency chain.** Scientific wheels are large and occasionally malicious or broken.
6. **Human error.** The system deleting or overwriting user data, or deploying by mistake.

## 2. Principles

- **Least privilege.** Processes get the minimum filesystem, network, and CPU/memory they need.
- **Inspectability first.** All generated code is inspectable before and after execution; logs
  are append-only.
- **Local-first privacy.** No data leaves the machine unless the user opts into a remote service
  for a specific step (and is told exactly what will be sent).
- **Confirmation for consequences.** High-impact actions (deletion, deployment, external
  communication, publishing) always require explicit human confirmation.
- **No untrusted deserialization.** No `pickle`/`joblib` from untrusted sources; prefer safe
  formats (JSON, parquet, ONNX, safetensors) and sandboxed loading otherwise.
- **Deterministic environment.** Reproducibility manifests pin the environment; sandbox enforces
  resource limits so a runaway experiment cannot harm the host.

## 3. Execution sandbox design (Phase 7)

`nullius.execution.Sandbox.run(plan, *, resources, network, fs_allowlist, timeout)`:

| Concern | Control |
| --- | --- |
| Isolation | Run in a subprocess/container (platform-dependent: `subprocess` + OS user/container on Linux; containers when available). Never in-process. |
| Filesystem | Allowlist of readable paths (the dataset dir + `runs/`), write-only scratch dir, everything else blocked. No access to credentials, home dotfiles, or system dirs. |
| Network | Off by default. Opt-in per run; when on, egress logged and scoped where possible. |
| Resource limits | CPU time, wall-clock timeout, memory cap, disk quota, process count. |
| Data exfil | Output files are checksummed; any path outside the scratch dir is refused. |
| Audit | Every run records: code hash, manifest, exit status, resource usage, warnings. |

Realistic local-only MVP (no container daemon): OS-level subprocess with `resource` limits,
a restricted cwd, and a filesystem wrapper (or, where available, `bwrap`/Docker). The sandbox
policy is documented per-platform and *fails closed*: if isolation cannot be guaranteed, the run
is refused rather than executed unsafely.

## 4. Human-in-the-loop matrix (spec §14)

| Action class | Default | Notes |
| --- | --- | --- |
| Profiling, EDA, transforms on a copy | autonomous | never mutates originals |
| Hypothesis generation, experiments, model comparison | autonomous | inside registry discipline |
| Sandboxed execution of generated code | autonomous | only in sandbox; code shown in report |
| Deleting / overwriting user data | **confirm** | destructive actions |
| Deploying models / touching production | **confirm** | separate deploy plan + rollback |
| External communications (email, webhooks, publishing) | **confirm** | what/where shown verbatim |
| Releasing quarantined test set for tuning | **confirm** | reason recorded |
| Remote LLM calls with private data | **confirm once per dataset** | shows exactly which fields are sent |
| Installing packages / modifying environment | **confirm** | pin + license check first |

## 5. Prompt-injection posture

Dataset cells and web text are **data, not instructions**. The planner's system prompt states
this; more importantly, no LLM output is ever executed or trusted as a result — LLM output is
*plans and prose*, validated and executed by deterministic code (assumption checks, registry,
sandbox). A claim produced by prose is not evidence; only registry records with artifacts are.

## 6. Secret and provenance handling

- Secrets live in environment variables / OS keychain; never in registry, reports, or manifests.
- The audit log records who/what initiated each action (`planner`, `user`, `critic`), its inputs
  (hashes), and its outputs — so an incorrect result can be traced to its origin (core principle).
