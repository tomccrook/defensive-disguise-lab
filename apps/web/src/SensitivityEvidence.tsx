import type { Metrics } from "./types";
import { fixed } from "./types";
export function SensitivityEvidence({ metrics }: { metrics: Metrics }) {
  const evidence = metrics.robustness;
  if (!evidence) return null;
  return (
    <details>
      <summary>
        Sensitivity, numerical precision and distinct play groups
      </summary>
      <p>
        Predeclared 256-play subset. Sensor bias shifts permitted coordinates
        and anchors consistently by about 0.1 yd. Six-frame history retains the
        observed anchor and is a model stress, outside the production schema.
        Graph radii change allowed edges without refitting or recalibration.
        These checks cannot select a model.
      </p>
      <div className="table-scroll">
        <table>
          <caption>Frozen-model sensitivities · primary endpoint</caption>
          <thead>
            <tr>
              <th>Model / stress</th>
              <th>Energy (yd)</th>
              <th>Change (yd)</th>
              <th>Game bootstrap 95% interval</th>
            </tr>
          </thead>
          <tbody>
            {evidence.stress.map((r) => (
              <tr key={`${r.family}/${r.stress}`}>
                <th scope="row">
                  {r.family} / {r.stress.replaceAll("_", " ")}
                </th>
                <td>{fixed(r.energy_score)}</td>
                <td>{fixed(r.change_from_unmodified.difference)}</td>
                <td>
                  {r.change_from_unmodified.interval.map(fixed).join(" to ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Monte Carlo precision uses a fixed 128-play subset, separately from
        game-sampling uncertainty.
      </p>
      <div className="table-scroll">
        <table>
          <caption>Graph minus pooled energy · four fixed random seeds</caption>
          <thead>
            <tr>
              <th>Draws</th>
              <th>Seed</th>
              <th>Difference (yd)</th>
            </tr>
          </thead>
          <tbody>
            {evidence.monte_carlo_precision.map((r) => (
              <tr key={`${r.draws}/${r.seed}`}>
                <th scope="row">{r.draws}</th>
                <td>{r.seed}</td>
                <td>{fixed(r.graph_minus_pooled)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-scroll">
        <table>
          <caption>
            Descriptive concept associations · different measures can overlap
          </caption>
          <thead>
            <tr>
              <th>Model / measures</th>
              <th>Spearman ρ</th>
              <th>Top-decile overlap</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(evidence.concept_distinctions).flatMap(
              ([family, pairs]) =>
                Object.entries(pairs).map(([name, r]) => (
                  <tr key={`${family}/${name}`}>
                    <th scope="row">
                      {family} / {name.replaceAll("_", " ")}
                    </th>
                    <td>
                      {r.spearman === null
                        ? "Undefined (tied values)"
                        : fixed(r.spearman)}
                    </td>
                    <td>
                      {r.top_decile_overlap_fraction === null
                        ? "Undefined (tied values)"
                        : `${(100 * r.top_decile_overlap_fraction).toFixed(1)}%`}
                    </td>
                  </tr>
                )),
            )}
          </tbody>
        </table>
      </div>
      <p className="helper">
        Associations and selected groups do not establish intent, quarterback
        perception or a causal information effect.
      </p>
    </details>
  );
}
