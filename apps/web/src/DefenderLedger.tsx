import { familyLabels, fixed } from "./types";
import type { Family, Forecast, Reveal } from "./types";
export function DefenderLedger({
  forecast,
  reveal,
  family,
  selected,
  setSelected,
}: {
  forecast: Forecast;
  reveal: Reveal | null;
  family: Family;
  selected: string;
  setSelected: (id: string) => void;
}) {
  return (
    <section className="panel" aria-labelledby="defenders-heading">
      <div className="panel-heading">
        <h2 id="defenders-heading">Defender forecast ledger</h2>
        <span>{familyLabels[family]} · yards</span>
      </div>
      <div className="table-scroll">
        <table>
          <caption className="sr-only">
            Numerical alternative to the field. Select a defender to inspect its
            probability region.
          </caption>
          <thead>
            <tr>
              <th scope="col">Defender</th>
              <th scope="col">Anchor x / y</th>
              <th scope="col">Forecast Δx / Δy</th>
              <th scope="col">Spread yd²</th>
              {reveal && (
                <>
                  <th scope="col">Actual Δx / Δy</th>
                  <th scope="col">Endpoint error yd</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {forecast.defenders.map((d, index) => {
              const outcome = reveal?.outcomes.find(
                (o) => o.entity_id === d.entity_id,
              )?.displacement;
              return (
                <tr key={d.entity_id}>
                  <th scope="row">
                    <span aria-hidden="true">D{index + 1} · </span>
                    <button
                      aria-pressed={selected === d.entity_id}
                      onClick={() => setSelected(d.entity_id)}
                    >
                      {d.entity_id}
                    </button>
                  </th>
                  <td>{d.anchor.map((n) => n.toFixed(2)).join(" / ")}</td>
                  <td>
                    {d.mean_displacement.map((n) => n.toFixed(2)).join(" / ")}
                  </td>
                  <td>{fixed(d.covariance[0][0] + d.covariance[1][1])}</td>
                  {reveal && (
                    <>
                      <td>
                        {outcome
                          ? outcome.map((n) => n.toFixed(2)).join(" / ")
                          : "Missing target"}
                      </td>
                      <td>
                        {outcome
                          ? fixed(
                              Math.hypot(
                                outcome[0] - d.mean_displacement[0],
                                outcome[1] - d.mean_displacement[1],
                              ),
                            )
                          : "Masked"}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="helper">
        A sharp failure may reflect model weakness, omitted context or bad
        measurements. It does not prove defensive intent or quarterback
        confusion.
      </p>
      {!!forecast.defenders.find((d) => d.entity_id === selected)?.modes
        ?.length && (
        <div className="table-scroll">
          <table>
            <caption>
              Selected defender component hypotheses. The main ledger shows
              moment means and total spread; component weights are shared across
              the play.
            </caption>
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Probability</th>
                <th scope="col">Conditional Δx / Δy (yd)</th>
              </tr>
            </thead>
            <tbody>
              {forecast.defenders
                .find((d) => d.entity_id === selected)
                ?.modes.map((m, i) => (
                  <tr key={i}>
                    <th scope="row">{i + 1}</th>
                    <td>{(100 * m.weight).toFixed(1)}%</td>
                    <td>
                      {m.mean_displacement.map((n) => n.toFixed(2)).join(" / ")}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Input, model and calibration provenance</summary>
        <dl className="provenance">
          <dt>Dataset</dt>
          <dd>{forecast.dataset_id}</dd>
          <dt>Model</dt>
          <dd>{forecast.model_id}</dd>
          <dt>Input SHA-256</dt>
          <dd>{forecast.input_hash}</dd>
          <dt>Calibration</dt>
          <dd>{forecast.calibration_id}</dd>
          <dt>Split</dt>
          <dd>{forecast.split_id}</dd>
          <dt>Features / target</dt>
          <dd>
            {forecast.feature_version} / {forecast.target_version}
          </dd>
        </dl>
      </details>
    </section>
  );
}
