# A-Lab Retrospective Refusal Audit
This is a governance audit, not a re-adjudication of the A-Lab paper. We use corrected public A-Lab refinement data to ask which originally positive synthesis claims would pass a fixed residual guard and which would be flagged for human review / refusal to self-certify.
## Protocol
- Source: corrected A-Lab supplementary data from the Nature article.
- Audit population: originally positive A-Lab claims, `Success` or `Partial` in the synthesis-results CSV.
- External labels: corrected manual conclusions in `Refinement-Table.xlsx`; only structure/composition inconclusive counts as `inconclusive`.
- Calibration: 95th percentile of the CARTOGRAPH audit residual on confirmed `Success` rows.
- Residual: `rho = sqrt((Rwp/20)^2 + ((100-target_wt)/100)^2 + (max_alt_wt/100)^2)` using corrected public manual-refinement features.
- Calibrated threshold: `delta = 0.776`.

## Main Result
- Inconclusive positive claims flagged: `4/4` (100.0%).
- Confirmed positive claims flagged for review: `4/36` (11.1%).
- Rwp-only baseline flags inconclusive claims: `0/4`.
- Target-deficit-only baseline flags inconclusive claims: `4/4`.
- Target-deficit-only baseline flags confirmed claims: `4/36`; Rwp-only flags confirmed claims: `2/36`.
- Bootstrap calibration diagnostic (2000 resamples, seed 0): `delta` 95% CI = [0.496, 1.088], inconclusive flag-rate CI = [0.0%, 100.0%].

Interpretation: the guard is non-trivially informative: it flags all post-correction inconclusive positive claims while also conservatively flagging a small number of confirmed but complex multiphase claims for review. This is useful governance evidence, not a claim that the audit fully re-adjudicates the A-Lab study.

## Flagged Positive Claims
| target             | csv_result | external_label | audit_decision  | rho   | manual_rwp | manual_target_wt | manual_max_alt_wt |
| ------------------ | ---------- | -------------- | --------------- | ----- | ---------- | ---------------- | ----------------- |
| Ba2ZrSnO6          | Success    | confirmed      | flag_for_review | 1.088 | 9.720      | 22.000           | 58.200            |
| KBaGdWO6           | Partial    | inconclusive   | flag_for_review | 1.001 | 3.440      | 9.940            | 40.190            |
| InSb3(PO4)6        | Partial    | confirmed      | flag_for_review | 0.969 | 6.170      | 22.800           | 49.760            |
| Mg3MnNi3O8         | Success    | inconclusive   | flag_for_review | 0.933 | 3.250      | 25.620           | 53.990            |
| CaGd2Zr(GaO3)4     | Partial    | inconclusive   | flag_for_review | 0.855 | 3.100      | 20.730           | 28.110            |
| Mn7(P2O7)4         | Partial    | inconclusive   | flag_for_review | 0.818 | 2.610      | 39.290           | 53.320            |
| KBaPrWO6           | Success    | confirmed      | flag_for_review | 0.789 | 10.750     | 50.420           | 29.680            |
| Ba9Ca3La4(Fe4O15)2 | Partial    | confirmed      | flag_for_review | 0.788 | 5.500      | 41.740           | 45.370            |

## Full Positive-Claim Table
| target             | csv_result | external_label | audit_decision  | rho   | manual_rwp | manual_target_wt | manual_max_alt_wt |
| ------------------ | ---------- | -------------- | --------------- | ----- | ---------- | ---------------- | ----------------- |
| Ba2ZrSnO6          | Success    | confirmed      | flag_for_review | 1.088 | 9.720      | 22.000           | 58.200            |
| KBaGdWO6           | Partial    | inconclusive   | flag_for_review | 1.001 | 3.440      | 9.940            | 40.190            |
| InSb3(PO4)6        | Partial    | confirmed      | flag_for_review | 0.969 | 6.170      | 22.800           | 49.760            |
| Mg3MnNi3O8         | Success    | inconclusive   | flag_for_review | 0.933 | 3.250      | 25.620           | 53.990            |
| CaGd2Zr(GaO3)4     | Partial    | inconclusive   | flag_for_review | 0.855 | 3.100      | 20.730           | 28.110            |
| Mn7(P2O7)4         | Partial    | inconclusive   | flag_for_review | 0.818 | 2.610      | 39.290           | 53.320            |
| KBaPrWO6           | Success    | confirmed      | flag_for_review | 0.789 | 10.750     | 50.420           | 29.680            |
| Ba9Ca3La4(Fe4O15)2 | Partial    | confirmed      | flag_for_review | 0.788 | 5.500      | 41.740           | 45.370            |
| Ba6Ta2Na2V2O17     | Success    | confirmed      | pass            | 0.765 | 12.930     | 63.380           | 18.250            |
| InSb3Pb4O13        | Success    | confirmed      | pass            | 0.654 | 8.170      | 61.650           | 33.800            |
| KNa2Ga3(SiO4)3     | Partial    | confirmed      | pass            | 0.637 | 9.940      | 69.850           | 26.160            |
| KNaP6(PbO3)8       | Success    | confirmed      | pass            | 0.624 | 9.440      | 71.100           | 28.900            |
| MgTi4(PO4)6        | Success    | confirmed      | pass            | 0.524 | 9.930      | 88.210           | 11.790            |
| Na3Ca18Fe(PO4)14   | Success    | confirmed      | pass            | 0.474 | 4.290      | 67.540           | 26.990            |
| Ba6Na2V2Sb2O17     | Success    | confirmed      | pass            | 0.465 | 8.920      | 88.180           | 5.950             |
| KNaTi2(PO5)2       | Success    | confirmed      | pass            | 0.459 | 9.040      | 92.740           | 3.660             |
| KMn3O6             | Success    | confirmed      | pass            | 0.445 | 3.250      | 64.880           | 22.070            |
| KPr9(Si3O13)2      | Success    | confirmed      | pass            | 0.423 | 6.660      | 81.550           | 18.450            |
| K4TiSn3(PO5)4      | Success    | confirmed      | pass            | 0.411 | 6.490      | 77.910           | 12.020            |
| CaFe2P2O9          | Success    | confirmed      | pass            | 0.410 | 2.440      | 72.320           | 27.680            |
| FeSb3Pb4O13        | Success    | confirmed      | pass            | 0.395 | 7.790      | 95.420           | 4.580             |
| CaNi(PO3)4         | Success    | confirmed      | pass            | 0.390 | 2.710      | 67.780           | 17.380            |
| Zr2Sb2Pb4O13       | Success    | confirmed      | pass            | 0.389 | 7.780      | 100.000          | 0.000             |
| Sn2Sb2Pb4O13       | Success    | confirmed      | pass            | 0.388 | 7.140      | 89.230           | 10.770            |
| K2TiCr(PO4)3       | Success    | confirmed      | pass            | 0.363 | 7.050      | 93.790           | 6.210             |
| Zn3Ni4(SbO6)2      | Partial    | confirmed      | pass            | 0.342 | 3.760      | 74.740           | 13.390            |
| NaMnFe(PO4)2       | Success    | confirmed      | pass            | 0.329 | 1.810      | 75.660           | 20.150            |
| MgV4Cu3O14         | Success    | confirmed      | pass            | 0.327 | 4.350      | 82.770           | 17.230            |
| Na7Mg7Fe5(PO4)12   | Success    | confirmed      | pass            | 0.304 | 3.800      | 83.240           | 16.760            |
| Hf2Sb2Pb4O13       | Success    | confirmed      | pass            | 0.304 | 5.980      | 95.580           | 3.030             |
| MgNi(PO3)4         | Success    | confirmed      | pass            | 0.281 | 5.160      | 90.790           | 6.210             |
| MgCuP2O7           | Success    | confirmed      | pass            | 0.267 | 5.340      | 100.000          | 0.000             |
| MgTi2NiO6          | Success    | confirmed      | pass            | 0.251 | 4.930      | 96.530           | 3.470             |
| Mn4Zn3(NiO6)2      | Success    | confirmed      | pass            | 0.248 | 4.320      | 91.320           | 8.680             |
| CaMn(PO3)4         | Success    | confirmed      | pass            | 0.243 | 4.870      | 100.000          | 0.000             |
| K4MgFe3(PO4)5      | Success    | confirmed      | pass            | 0.204 | 4.090      | 100.000          | 0.000             |
| MnAgO2             | Success    | confirmed      | pass            | 0.197 | 3.890      | 97.720           | 2.280             |
| Mn2VPO7            | Success    | confirmed      | pass            | 0.145 | 2.890      | 100.000          | 0.000             |
| NaCaMgFe(SiO3)4    | Success    | confirmed      | pass            | 0.119 | 2.360      | 98.830           | 1.170             |
| CaCo(PO3)4         | Success    | confirmed      | pass            | 0.099 | 1.970      | 100.000          | 0.000             |
