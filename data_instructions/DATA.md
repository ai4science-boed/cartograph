# Public Data Instructions

Most experiments in the paper are synthetic or simulation-based and run
without external files. Two retrospective checks require public data.

## A-Lab Retrospective Audit

Expected paths:

```text
data/alab/original_supplementary_data/
  20230502 Synthesis Results with Recipes.csv
  Refinement-Table.xlsx
```

Download the original supplementary data for Szymanski et al., Nature 2023
and extract the ZIP into `data/alab/original_supplementary_data/`. The paper
also uses the 2026 author correction as the source of external
post-correction labels; the audit script operationalizes those labels through
the corrected manual conclusions in `Refinement-Table.xlsx`.

The script intentionally excludes ordering ambiguity alone from the
`inconclusive` label because that is a crystallographic refinement issue
orthogonal to whether the target phase was synthesized.

## EPA CvTdb Retrospective

Expected path:

```text
data/cvtdb_v2_0_0_no_audit.sqlite
```

Download the public EPA CompTox-PK CvTdb v2.0.0 SQLite database and place it
at the path above before running the EPA scripts.

## Frozen Outputs

The `outputs/` directory contains the frozen outputs used for manuscript
numbers, so reviewers can inspect the reported values even before downloading
the public datasets.

