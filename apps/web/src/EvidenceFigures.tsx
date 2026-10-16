export function EvidenceFigures({ timing }: { timing: boolean }) {
  const figures = timing
    ? [
        [
          "nfl-timing",
          "Same target and anchor: score improvement and belief-update size",
        ],
        [
          "nfl-sensitivities",
          "Frozen-model sensitivity checks on a predeclared subset",
        ],
      ]
    : [
        [
          "nfl-scores-coverage",
          "Held-out proper scores and measured marginal coverage",
        ],
        [
          "nfl-calibration",
          "Validation and test calibration, including shared mixtures",
        ],
      ];
  return (
    <section
      aria-label="Scientific evidence figures"
      className="evidence-figures"
    >
      {figures.map(([name, caption]) => (
        <figure key={name}>
          <img
            loading="lazy"
            width={1600}
            height={640}
            src={`/api/figures/${name}`}
            alt={caption}
          />
          <figcaption>
            {caption}. Numerical results and source records are in the research
            report.
          </figcaption>
        </figure>
      ))}
    </section>
  );
}
