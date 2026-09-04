# Research Projects

This repository is a monorepo: each research effort lives under `projects/` with its own package, dependencies, tests, configs, and Kaggle instructions.

## Projects

| Project | Status | Location |
|---|---|---|
| Logit-Guided Evidence Routing (ICLR 2027) | Phase 01 Kaggle pilot ready | [`projects/logit_evidence_routing`](projects/logit_evidence_routing) |

## Branch workflow

Use one feature branch per project. The ICLR project is developed on `feat/iclr`:

```bash
git switch feat/iclr
git add projects/logit_evidence_routing README.md .gitignore
git commit -m "Implement initial LGER routing pipeline"
git push -u origin feat/iclr
```

Each project README contains its own local and Kaggle commands.
