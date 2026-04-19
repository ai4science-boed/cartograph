# Final And Latest Results: A-opt Pivot

Run timestamp (UTC): `2026-03-28T07:13:38.817657+00:00`

## Bottom Line

The unresolved A-opt upgrade is now the strongest algorithmic version of the
project. It gives a real high-dimensional advantage on the replicated cascade
benchmark, while preserving the low-dimensional PK/EPA near-tie story and the
failure-case refusal behavior.

## Headline Numbers

- Cascade d=8: hidden-best raw `0.02` vs A-opt `0.65`; regret raw `19.9404` vs A-opt `0.0102`
- Cascade d=16: hidden-best raw `0.07` vs A-opt `0.72`; regret raw `2.8322` vs A-opt `0.0142`
- PK: A-opt vs disagreement `1W / 6T / 0L`, A-opt vs raw `0W / 7T / 0L`
- EPA: A-opt vs raw `0W / 4T / 0L`, with degenerate unresolved space on `2/4` series
- Failure benchmark: raw refusal/control `True/True`, A-opt refusal/control `True/True`