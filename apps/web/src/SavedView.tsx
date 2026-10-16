import { familyLabels } from "./types";
import type { Family, Forecast } from "./types";

export function SavedView({
  saved,
  onClear,
}: {
  saved: { family: Family; forecast: Forecast }[];
  onClear: () => void;
}) {
  return (
    <section className="panel">
      <h2>Saved endpoint comparisons</h2>
      <p>
        Saved comparisons retain input hashes and model distributions within
        this page session. Seeing a later outcome makes replay exploratory.
      </p>
      {!saved.length ? (
        <p>No saved forecasts yet.</p>
      ) : (
        <ul className="case-list">
          {saved.map((s, i) => (
            <li key={`${s.forecast.model_id}-${i}`}>
              <span>
                {familyLabels[s.family]} · cutoff {s.forecast.cutoff} s · target
                +{s.forecast.horizon} s · {s.forecast.input_hash.slice(0, 12)}
              </span>
              <details>
                <summary>Inspect saved endpoints and versions</summary>
                <p className="helper">
                  Dataset {s.forecast.dataset_id} · model {s.forecast.model_id}{" "}
                  · calibration {s.forecast.calibration_id} · target{" "}
                  {s.forecast.target_version} · anchor{" "}
                  {s.forecast.anchor_seconds} s
                </p>
                <div className="table-scroll">
                  <table>
                    <caption>
                      Saved marginal endpoints; outcomes excluded
                    </caption>
                    <thead>
                      <tr>
                        <th>Defender</th>
                        <th>Mean Δx / Δy (yd)</th>
                        <th>Covariance trace (yd²)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.forecast.defenders.map((d) => (
                        <tr key={d.entity_id}>
                          <th scope="row">{d.entity_id}</th>
                          <td>
                            {d.mean_displacement
                              .map((v) => v.toFixed(2))
                              .join(" / ")}
                          </td>
                          <td>
                            {(d.covariance[0][0] + d.covariance[1][1]).toFixed(
                              2,
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </li>
          ))}
        </ul>
      )}
      <button disabled={!saved.length} onClick={onClear}>
        Clear saved forecasts
      </button>
    </section>
  );
}
