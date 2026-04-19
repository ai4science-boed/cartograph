# EPA Real-Data Retrospective With Local T-opt Baseline

Run timestamp (UTC): `2026-04-13T19:28:54.275464+00:00`
Dataset: `/Users/neelshah/Downloads/ai4science/data/cvtdb_v2_0_0_no_audit.sqlite`
Cohort rule: oral series, at least `10` points, max time <= `24.0` h, one series per chemical
Successful series: `8`
Degenerate unresolved-space series: `3`
Nondegenerate active subset: `5`
Identification margin (BIC gap): `2.00`

## Cohort-Level Summary

| Split | Initial Margin | CART Margin | Disagreement Margin | T-opt Margin | Random E[margin] | CART Hit | Disagreement Hit | T-opt Hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All series | -15.301 | 7.703 | 7.686 | 8.576 | 4.288 | 50.00% | 37.50% | 62.50% |
| Active subset | 4.988 | 13.554 | 13.527 | 13.554 | 7.990 | 80.00% | 60.00% | 80.00% |

## Pairwise One-Step Oracle-Margin Results

- All series, CARTOGRAPH vs Disagreement: `1W / 7T / 0L`
- All series, CARTOGRAPH vs T-opt: `0W / 7T / 1L`
- All series, T-opt vs Disagreement: `2W / 6T / 0L`
- Active subset, CARTOGRAPH vs Disagreement: `1W / 4T / 0L`
- Active subset, CARTOGRAPH vs T-opt: `0W / 5T / 0L`
- Active subset, T-opt vs Disagreement: `1W / 4T / 0L`

## Working Vs Skipped

- Successful evaluations: `8`
- Skipped after fit/evaluation failure: `88`

### Skipped Series

- `Chloroform oral` (series `65743`): `RuntimeError: Failed to fit model B`
- `Miconazole oral` (series `32548`): `RuntimeError: Failed to fit model A`
- `Miconazole oral` (series `32549`): `RuntimeError: Failed to fit model A`
- `Quinidine gluconate oral` (series `98239`): `RuntimeError: Failed to fit model A`
- `Disulfate ion oral` (series `98443`): `RuntimeError: Failed to fit model A`
- `Disulfate ion oral` (series `98460`): `RuntimeError: Failed to fit model A`
- `Chlorogenic acid oral` (series `32476`): `RuntimeError: Failed to fit model A`
- `Chlorogenic acid oral` (series `32477`): `RuntimeError: Failed to fit model A`
- `D-Carnosine oral` (series `32482`): `RuntimeError: Failed to fit model A`
- `D-Carnosine oral` (series `32483`): `RuntimeError: Failed to fit model A`
- `Ciprofloxacin oral` (series `32652`): `RuntimeError: Failed to fit model A`
- `Ciprofloxacin oral` (series `32653`): `RuntimeError: Failed to fit model A`
- `Isopsoralen oral` (series `32522`): `RuntimeError: Failed to fit model A`
- `Caffeine, citrated oral` (series `98278`): `RuntimeError: Failed to fit model A`
- `5,5-Diphenylhydantoin oral` (series `97578`): `RuntimeError: Failed to fit model A`
- `5,5-Diphenylhydantoin oral` (series `97579`): `RuntimeError: Failed to fit model A`
- `5,5-Diphenylhydantoin oral` (series `97580`): `RuntimeError: Failed to fit model A`
- `5,5-Diphenylhydantoin oral` (series `97581`): `RuntimeError: Failed to fit model A`
- `Diclofenac oral` (series `97557`): `RuntimeError: Failed to fit model A`
- `Acetaminophen oral` (series `32436`): `RuntimeError: Failed to fit model A`
- `Acetaminophen oral` (series `32437`): `RuntimeError: Failed to fit model A`
- `Nilvadipine oral` (series `98157`): `RuntimeError: Failed to fit model A`
- `Nilvadipine oral` (series `98158`): `RuntimeError: Failed to fit model A`
- `Nilvadipine oral` (series `98159`): `RuntimeError: Failed to fit model A`
- `Sulfasalazine oral` (series `32602`): `RuntimeError: Failed to fit model A`
- `Sulfasalazine oral` (series `32603`): `RuntimeError: Failed to fit model A`
- `Dabigatran oral` (series `32484`): `RuntimeError: Failed to fit model A`
- `Dabigatran oral` (series `32485`): `RuntimeError: Failed to fit model A`
- `Diltiazem oral` (series `97989`): `RuntimeError: Failed to fit model A`
- `Diltiazem oral` (series `97993`): `RuntimeError: Failed to fit model A`
- `Diltiazem oral` (series `97994`): `RuntimeError: Failed to fit model A`
- `Diazepam oral` (series `32664`): `RuntimeError: Failed to fit model A`
- `Diazepam oral` (series `32665`): `RuntimeError: Failed to fit model A`
- `Fexofenadine oral` (series `32678`): `RuntimeError: Failed to fit model A`
- `Fexofenadine oral` (series `32679`): `RuntimeError: Failed to fit model A`
- `Oxycodone oral` (series `32714`): `RuntimeError: Failed to fit model A`
- `Oxycodone oral` (series `32715`): `RuntimeError: Failed to fit model A`
- `Quinotolast oral` (series `32584`): `RuntimeError: Failed to fit model A`
- `Quinotolast oral` (series `32585`): `RuntimeError: Failed to fit model A`
- `Alprazolam oral` (series `97829`): `RuntimeError: Failed to fit model A`
- `Alprazolam oral` (series `97832`): `RuntimeError: Failed to fit model A`
- `Alprazolam oral` (series `97835`): `RuntimeError: Failed to fit model A`
- `Nitrazepam oral` (series `32710`): `RuntimeError: Failed to fit model A`
- `Nitrazepam oral` (series `32711`): `RuntimeError: Failed to fit model A`
- `Methyl methacrylate oral` (series `32544`): `RuntimeError: Failed to fit model A`
- `Methyl methacrylate oral` (series `32545`): `RuntimeError: Failed to fit model A`
- `Aspirin oral` (series `32450`): `RuntimeError: Failed to fit model A`
- `Aspirin oral` (series `32451`): `RuntimeError: Failed to fit model A`
- `Terguride oral` (series `32614`): `RuntimeError: Failed to fit model A`
- `Terguride oral` (series `32615`): `RuntimeError: Failed to fit model A`
- `Diisobutyl ketone oral` (series `32666`): `RuntimeError: Failed to fit model A`
- `Diisobutyl ketone oral` (series `32667`): `RuntimeError: Failed to fit model A`
- `Ellagic acid oral` (series `32674`): `RuntimeError: Failed to fit model A`
- `Ellagic acid oral` (series `32675`): `RuntimeError: Failed to fit model A`
- `Disulfate ion oral` (series `98453`): `RuntimeError: Failed to fit model A`
- `Disulfate ion oral` (series `98470`): `RuntimeError: Failed to fit model A`
- `Apixaban oral` (series `32446`): `RuntimeError: Failed to fit model A`
- `Apixaban oral` (series `32447`): `RuntimeError: Failed to fit model A`
- `Rivaroxaban oral` (series `32588`): `RuntimeError: Failed to fit model A`
- `Rivaroxaban oral` (series `32589`): `RuntimeError: Failed to fit model A`
- `Tedizolid oral` (series `32746`): `RuntimeError: Failed to fit model A`
- `Tedizolid oral` (series `32747`): `RuntimeError: Failed to fit model A`
- `Pranlukast oral` (series `32576`): `RuntimeError: Failed to fit model A`
- `Pranlukast oral` (series `32577`): `RuntimeError: Failed to fit model A`
- `Theophylline oral` (series `32618`): `RuntimeError: Failed to fit model A`
- `Theophylline oral` (series `32619`): `RuntimeError: Failed to fit model A`
- `Diltiazem oral` (series `97995`): `RuntimeError: Failed to fit model A`
- `Lemildipine oral` (series `32532`): `RuntimeError: Failed to fit model A`
- `Lemildipine oral` (series `32533`): `RuntimeError: Failed to fit model A`
- `Methylphenidate oral` (series `32702`): `RuntimeError: Failed to fit model A`
- `Methylphenidate oral` (series `32703`): `RuntimeError: Failed to fit model A`
- `Zolpidem oral` (series `32758`): `RuntimeError: Failed to fit model A`
- `Zolpidem oral` (series `32759`): `RuntimeError: Failed to fit model A`
- `Pitavastatin oral` (series `32572`): `RuntimeError: Failed to fit model A`
- `Pitavastatin oral` (series `32573`): `RuntimeError: Failed to fit model A`
- `Triazolam oral` (series `32752`): `RuntimeError: Failed to fit model A`
- `Triazolam oral` (series `32753`): `RuntimeError: Failed to fit model A`
- `Morphine oral` (series `32704`): `RuntimeError: Failed to fit model A`
- `Morphine oral` (series `32705`): `RuntimeError: Failed to fit model A`
- `Alprazolam oral` (series `97828`): `RuntimeError: Failed to fit model A`
- `Alprazolam oral` (series `97834`): `RuntimeError: Failed to fit model A`
- `Digoxin oral` (series `32490`): `RuntimeError: Failed to fit model A`
- `Digoxin oral` (series `32491`): `RuntimeError: Failed to fit model A`
- `Domitroban oral` (series `32494`): `RuntimeError: Failed to fit model A`
- `Domitroban oral` (series `32495`): `RuntimeError: Failed to fit model A`
- `Scopoletin oral` (series `32734`): `RuntimeError: Failed to fit model A`
- `Scopoletin oral` (series `32735`): `RuntimeError: Failed to fit model A`
- `Ondansetron hydrochloride dihydrate oral` (series `97570`): `RuntimeError: Failed to fit model A`

## Per-Series Results

| Series | Oracle | Initial Margin | Unresolved Dim | CART Pick | CART Margin | Disagreement Pick | Disagreement Margin | T-opt Pick | T-opt Margin | Hidden Best |
|---|---|---:|---:|---|---:|---|---:|---|---:|---|
| Dichloromethane oral | A | -71.065 | 0 | E1 | -4.637 | E1 | -4.637 | E1 | -4.637 | E3 |
| 1,2-Dichloroethane oral | A | -78.091 | 0 | E1 | -3.678 | E1 | -3.678 | E3 | 3.308 | E3 |
| Trichloroethylene oral | C | 32.083 | 1 | E1 | 55.558 | E1 | 55.558 | E1 | 55.558 | E2 |
| Benzo[a]pyrene oral | B | 3.654 | 1 | E1 | 8.868 | E1 | 8.868 | E1 | 8.868 | E1 |
| Chloroform oral | A | -14.379 | 1 | E1 | -0.933 | E1 | -0.933 | E1 | -0.933 | E1 |
| Methyl tert-butyl ether oral | A | 1.803 | 0 | E1 | 2.166 | E1 | 2.166 | E1 | 2.166 | E2 |
| Valproic acid oral | A | 1.792 | 1 | E1 | 2.197 | E1 | 2.197 | E1 | 2.197 | E1 |
| Glycine oral | A | 1.792 | 1 | E1 | 2.081 | E2 | 1.946 | E1 | 2.081 | E1 |

## Artifact Paths

- `figure_1_method_summary`: `/Users/neelshah/Downloads/ai4science/outputs/real_data_topt/figure_1_method_summary.png`