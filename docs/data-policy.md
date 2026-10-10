# Public data boundary

The reproducible path in this repository uses `disguise_lab.fixtures.generate`: hand-authored formations and seeded random values. The generated entities, dates and team labels describe fictional scenes, not historical plays.

Included:

- Python implementations, configuration schemas and tests.
- Synthetic fixture generation, model training and evaluation code.
- Source adapters that can be inspected without supplying data.
- A local forecast/reveal API and its research workbench.

Excluded:

- Historical tracking rows, player metadata and recovered data downloads.
- Historical fitted models, predictions, empirical metrics and report figures.
- Private access approvals, rules confirmations and source inventories.
- Credentials, caches, active-run pointers and local research databases.

`configs/rights.json` describes this public synthetic release. It does not carry over any historical-data authorisation from another checkout. Historical-source commands need separately obtained inputs and an independent access/publication decision. A mirror's code licence does not by itself establish rights over its data.

Keep new datasets and generated outputs in the ignored `data/`, `artifacts/` and `checkpoints/` directories. Before publishing a new example, confirm that it was independently generated and that its figures and metadata do not include private research outputs.
