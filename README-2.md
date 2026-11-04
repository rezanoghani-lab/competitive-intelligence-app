# Golden Set + Evaluation Guide

## What this is for
- Regression testing: run after EVERY prompt/model/config change
- Calibration baseline: scores per case, tracked in results_log.csv over versions

## Create your own cases (aim for 10-15 total)
Pick products where the competitive landscape is UNAMBIGUOUS and you can
verify ground truth yourself in 15 minutes:
  - must_include: competitors any correct analysis MUST find
  - nice_to_include: good-to-have, not required
  - required_facts: structural checks (no nulls, sourced claims)
Good candidates: Zoom (video conferencing), Notion (workspace docs),
Spotify (music streaming), Shopify (e-commerce), Duolingo (language learning)

## How to feed it to the app - two options
OPTION A (recommended): keep evals OUT of the demo app.
  Run:  python evals/run_evals.py
  Edit the import line to point at your pipeline function. App stays fast/free-tier.

OPTION B (portfolio showcase): add an 'Evaluation' mode in Streamlit:
  mode = st.sidebar.selectbox('Mode', ['Demo', 'Evaluation'])
  if mode == 'Evaluation':
      run golden set headlessly, show metrics table + results_log.csv trend chart
  This tab IS your calibration dashboard for recruiters. Guard against abuse:
  only run it on button click, since 15 cases x full pipeline can burn quota.

## Cadence
- Before/after any prompt change: full golden set, compare scores
- Weekly: review eval failures + human thumbs-down feedback together
- Only quote calibration numbers (accuracy-by-confidence) with n>=30 human judgments
