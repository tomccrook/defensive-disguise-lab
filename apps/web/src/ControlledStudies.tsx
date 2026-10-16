import { familyLabels, fixed } from "./types";
import type { Metrics, Family } from "./types";
export function ControlledStudies({ metrics }: { metrics: Metrics }) {
  return (
    <>
      {metrics.opponent_primary && (
        <section>
          <h3>Unfamiliar opponents</h3>
          <p>
            ARI/SEA whole games held out; BUF/SF validation. This overlaps the
            chronological protocol and is not an independent replication.
          </p>
          <div className="table-scroll">
            <table>
              <caption>Opponent holdout · primary endpoint</caption>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Energy ↓ (yd)</th>
                  <th>80% marginal coverage</th>
                  <th>Games / plays</th>
                </tr>
              </thead>
              <tbody>
                {metrics.opponent_primary.map((r) => (
                  <tr key={r.family}>
                    <th scope="row">{familyLabels[r.family]}</th>
                    <td>{fixed(r.energy_score)}</td>
                    <td>{(r.marginal_coverage_80 * 100).toFixed(1)}%</td>
                    <td>
                      {r.games} / {r.plays}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {metrics.ablations && (
        <details>
          <summary>Controlled ablations and mixture heads</summary>
          <p>
            Gaussian heads use nominal ellipses. Shared mixtures use marginal
            density sets; the component is shared across defenders, with
            conditional independence. Controls use the same physical targets and
            scoring cohorts.
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                All registered chronological variants · same primary endpoint
              </caption>
              <thead>
                <tr>
                  <th>Variant</th>
                  <th>Energy ↓ (yd)</th>
                  <th>NLL</th>
                  <th>Measured 80% region coverage</th>
                </tr>
              </thead>
              <tbody>
                {metrics.all_variants?.map((r) => (
                  <tr key={r.family}>
                    <th scope="row">
                      {familyLabels[r.family as Family] ??
                        r.family.replaceAll("_", " ")}
                    </th>
                    <td>{fixed(r.energy_score)}</td>
                    <td>{fixed(r.nll)}</td>
                    <td>{(100 * r.marginal_coverage_80).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-scroll">
            <table>
              <caption>
                Predeclared variant comparisons · difference in energy score
              </caption>
              <thead>
                <tr>
                  <th>Comparison</th>
                  <th>Difference (yd)</th>
                  <th>Game bootstrap 95% interval</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.ablations).map(([name, r]) => (
                  <tr key={name}>
                    <th scope="row">{name.replaceAll("_", " ")}</th>
                    <td>{fixed(r.difference)}</td>
                    <td>{r.interval.map(fixed).join(" to ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}
