# Design documentation

Design artifacts are produced before implementation (engineering rule 22) and are updated as
components land. Order of reading: architecture → interfaces → the three domain designs
(mathematical reasoning, experiment lifecycle, security) → roadmap.

| Doc | Covers |
| --- | --- |
| [01-architecture.md](01-architecture.md) | Vision, principles, system architecture, data flow, technology choices, repository structure |
| [02-component-interfaces.md](02-component-interfaces.md) | Typed contracts between components; core domain records |
| [03-mathematical-reasoning.md](03-mathematical-reasoning.md) | Math layer, method records, assumption checking, honesty rules |
| [04-experiment-lifecycle.md](04-experiment-lifecycle.md) | Experiment specification, state machine, registry, contamination guards |
| [05-security-model.md](05-security-model.md) | Sandbox, permissions, human-in-the-loop matrix |
| [06-development-roadmap.md](06-development-roadmap.md) | MVP definition, phases 1–10, benchmarking, failure-first test catalog, live status |
