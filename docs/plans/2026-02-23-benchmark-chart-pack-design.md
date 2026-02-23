# Benchmark Chart Pack Design

## Goal
Create presentation-ready benchmark visuals (executive + analytical) from existing dual-market artifacts using seaborn.

## Audience
- Product managers / executives with short attention window.
- Technical reviewers who need metric context behind headline winners.

## Output Packs

### Executive Pack (3 charts)
1. Champion scorecard: concise market/task/model winners with primary metric values.
2. Forecast headline chart: sMAPE by model across PJM/NP (lower is better), winners highlighted.
3. Rally headline chart: PR-AUC by model across PJM/NP (higher is better), winners highlighted.

### Analytical Pack (2 charts)
1. Forecast efficiency chart: MAE vs sMAPE scatter, sized/colored by RMSE, faceted by market.
2. Rally reliability chart: model calibration curves (predicted probability vs observed rate), faceted by market.

## Visual Style
- Seaborn whitegrid with restrained color palette and high contrast labels.
- Simple typography hierarchy and direct chart subtitles.
- Explicit "higher/lower is better" cues in titles.
- Export high-resolution PNGs suitable for slides.

## File Layout
- Script entrypoint: `scripts/generate_charts.py`
- Chart module: `src/time_copilot_demo/charts.py`
- Output directory: `reports/charts/`
- Optional markdown index: `reports/charts/README.md`

## Success Criteria
- One command generates all 5 charts from `artifacts/dual_market`.
- Charts are legible and self-explanatory with minimal narration.
- Tests verify generation and expected filenames.
