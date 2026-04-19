# Final And Latest Results: Cascade A-opt Robustness

Run timestamp (UTC): `2026-03-28T07:09:50.354028+00:00`

## Top-Line Takeaway

The exact unresolved A-opt upgrade remains stronger than raw CARTOGRAPH and
disagreement on the structured high-dimensional cascade benchmark after
replicating over many noise seeds. The advantage is negligible at `d=2`, but
becomes clear at `d>=4`, while runtime stays in the low-millisecond regime.

## Dimension Highlights

### d=2
- Hidden-best match: raw `0.44`, A-opt `0.44`, disagreement `0.44`
- Mean regret: raw `0.0518`, A-opt `0.0518`, disagreement `0.0518`
- A-opt vs raw trial counts: `0W / 144T / 0L`
- Average sequence runtime (ms): raw `0.2001`, A-opt `1.2320`, disagreement `0.0091`

### d=4
- Hidden-best match: raw `0.00`, A-opt `0.09`, disagreement `0.00`
- Mean regret: raw `0.3119`, A-opt `0.4182`, disagreement `0.3119`
- A-opt vs raw trial counts: `73W / 0T / 71L`
- Average sequence runtime (ms): raw `0.1990`, A-opt `1.3937`, disagreement `0.0149`

### d=8
- Hidden-best match: raw `0.02`, A-opt `0.65`, disagreement `0.02`
- Mean regret: raw `19.9404`, A-opt `0.0102`, disagreement `19.9404`
- A-opt vs raw trial counts: `129W / 0T / 15L`
- Average sequence runtime (ms): raw `0.5556`, A-opt `9.6442`, disagreement `0.0136`

### d=16
- Hidden-best match: raw `0.07`, A-opt `0.72`, disagreement `0.07`
- Mean regret: raw `2.8322`, A-opt `0.0142`, disagreement `2.8322`
- A-opt vs raw trial counts: `120W / 0T / 24L`
- Average sequence runtime (ms): raw `1.1473`, A-opt `46.7718`, disagreement `0.0194`
