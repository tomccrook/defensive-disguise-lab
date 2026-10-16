# Public source release

This is a source-and-synthetic-fixtures edition of the local Defensive Disguise Lab. The original research checkout and its complete history remain separate. Publication commits use their actual dates.

The release excludes the historical study's tracked evidence folder, reports, screenshots, data inventories and private access records. It also removes fixed historical performance values from one report-writing template, replacing them with a description that refers to values supplied by the local report. The forecasting implementations are unchanged.

The public rights configuration authorises the synthetic fixture path only. Data adapters and empirical evaluation implementations are available for inspection; their presence does not grant historical-data access or public redistribution rights.

The example and bounded training recipe run without historical observations. Tests that require an active model are exercised after generating a fresh synthetic fixture. The verification below applies to this release checkout, rather than relying on earlier recorded checks.

## Release verification — 16 October 2026

A new six-game synthetic fixture was trained for two epochs. All 24 tests passed, including four API tests that require an active fixture. The causal-feature example, Ruff and mypy checks passed.

Dependency locks were installed into separate environments. The source-publication review checked the selected file list, private paths, credential patterns, excluded data/weight formats and historical defensive outputs. These checks do not rerun the full protected scientific studies. Upstream dependency deprecation warnings remain visible in test output.

The local frontend production build passed with Node 24.21.0. The GitHub Actions workflow includes Python checks and frontend compilation; hosted-run status is separate from these local results.
