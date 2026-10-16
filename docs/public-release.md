# Public source release

This is a reviewed source snapshot first published on 16 October 2026. The original research checkout and its history remain separate. The public history was subsequently reorganised into topical import commits with earlier author dates. Those dates were assigned during reconstruction; they are not verified dates of development or evidence that the projects were complete before September. Committer timestamps retain the actual import date, and each import commit identifies the reconstruction in its message.

The release excludes the historical study's tracked evidence folder, reports, screenshots, data inventories and private access records. It also removes fixed historical performance values from one report-writing template, replacing them with a description that refers to values supplied by the local report. The forecasting implementations are unchanged.

The public rights configuration authorises the synthetic fixture path only. Data adapters and empirical evaluation implementations are available for inspection; their presence does not grant historical-data access or public redistribution rights.

The example and bounded training recipe run without historical observations. Tests that require an active model are exercised after generating a fresh synthetic fixture. The verification below applies to this release checkout, rather than relying on earlier recorded checks.

## Release verification — 16 October 2026

A new six-game synthetic fixture was trained for two epochs. All 24 tests passed, including four API tests that require an active fixture. The causal-feature example, Ruff and mypy checks passed.

Dependency locks were installed into separate environments. The source-publication review checked the selected file list, private paths, credential patterns, excluded data/weight formats and historical defensive outputs. These checks do not rerun the full protected scientific studies. Upstream dependency deprecation warnings remain visible in test output.

The local frontend production build passed with Node 24.21.0. The GitHub Actions workflow includes Python checks and frontend compilation; hosted-run status is separate from these local results.

## Reconstructed import schedule

| Assigned author date | Imported component |
| --- | --- |
| 2026-03-14 | Set up the defensive forecasting package |
| 2026-04-22 | Import synthetic fixtures and observation adapters |
| 2026-05-24 | Validate causal inputs and auditable storage |
| 2026-06-13 | Import defender forecasting models |
| 2026-06-26 | Add forecast distributions and evaluation metrics |
| 2026-07-18 | Add empirical and robustness studies |
| 2026-08-02 | Expose forecast API and report writers |
| 2026-08-24 | Import the field playback workbench |
| 2026-08-30 | Add evidence views and browser checks |
| 2026-10-10 | Document model methods and the synthetic data policy |

The source import is consolidated into ten substantive file groups. Subsequent maintenance changes use their actual dates.
