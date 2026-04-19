# CARTOGRAPH: Duffing Exact Check – Theorem 2 Validation

**Date:** 2026-03-27T19:15:58.391511+00:00

## System

Duffing oscillator: `x'' + dx' + ax + bx^3 = g*cos(wt)`

| Parameter | Value |
|-----------|-------|
| alpha | 1.0 |
| beta | 0.2 |
| delta | 0.3 |
| gamma | 0.5 |
| omega | 1.2 |

Shared basis: ['x', 'x^3', "x'", 'cos(wt)']

Ground-truth coefficients a*: `[-1.0, -0.2, -0.3, 0.5]`

## Models

| Model | Support mask | Omitted terms |
|-------|-------------|---------------|
| A (no x^3) | [1, 0, 1, 1] | ['x^3'] |
| B (no x') | [1, 1, 0, 1] | ["x'"] |
| C (no cos) | [1, 1, 1, 0] | ['cos(wt)'] |
| D (no x^3,cos) | [1, 0, 1, 0] | ['x^3', 'cos(wt)'] |

## Results

### Coverage (Proposition 1)

- Full coverage: **True**
- Union of supports: `[1, 1, 1, 1]`

### Controversial terms

- Terms omitted by at least one model: **['x^3', "x'", 'cos(wt)']**
- Count: **3**
- Theorem 2 prediction: rank(H) = **3**

### Rank progression

Trajectory-domain rank of H (recovery requires rank >= 3):

| Experiments | rank(H) | Sufficient? |
|------------|--------|------------|
| e1 | 3 | Yes |
| e1, e2 | 5 | Yes |
| e1, e2, e3 | 7 | Yes |
| e1, e2, e3, e4 | 9 | Yes |
| e1, e2, e3, e4, e5 | 11 | Yes |
| e1, e2, e3, e4, e5, e6 | 13 | Yes |

rank(H) >= 3 achieved with **1** experiment(s). Trajectory-domain rank continues growing to 13 as more experiments add independent trajectory patterns.

### Recovery of controversial component

| Term | Recovered | Ground truth |
|------|-----------|-------------|
| x^3 | -0.200000 | -0.200000 |
| x' | -0.300000 | -0.300000 |
| cos(wt) | 0.500000 | 0.500000 |

Recovery error (L2): **0.00e+00**

### Conclusion

Theorem 2 is validated: in the exact omission-only setting with 4 models,
the disagreement matrix H achieves trajectory-domain rank **13** (>= **3** controversial terms),
confirming that behavioral access suffices for recovery of the controversial component.
The controversial coefficients are recovered exactly (error = 0.00e+00).
