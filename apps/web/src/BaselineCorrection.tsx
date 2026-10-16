import type { Metric } from "./types";
import { fixed } from "./types";
export type Correction = {
  primary: (Omit<Metric, "family"> & { family: string })[];
  validation: (Omit<Metric, "family"> & { family: string })[];
};
export function BaselineCorrection({ result }: { result: Correction }) {
  return (
    <section className="notice">
      <h3>Post-exposure anchor control</h3>
      <p>
        The original boosted model predicts the full anchor-relative target.
        Neural heads explicitly add observed cutoff-to-anchor displacement. This
        additional control gives the booster the same target re-expression; tree
        settings and calibration policy stay fixed. New covariance scales use
        the same validation block only. It was registered after the original
        test was exposed, so it is an exploratory corrective audit. Original
        models, primary result and demo selection are preserved. The
        history-aware residual booster did not improve its validation score.
      </p>
      <div className="table-scroll">
        <table>
          <caption>
            Separate corrective audit · same −0.1s / +1s target; 28 test games
          </caption>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Validation energy ↓</th>
              <th>Test energy ↓ (yd)</th>
              <th>Measured 80% coverage</th>
            </tr>
          </thead>
          <tbody>
            {result.primary.map((r) => (
              <tr key={r.family}>
                <th scope="row">{r.family.replaceAll("_", " ")}</th>
                <td>
                  {fixed(
                    result.validation.find((v) => v.family === r.family)!
                      .energy_score,
                  )}
                </td>
                <td>{fixed(r.energy_score)}</td>
                <td>{(100 * r.marginal_coverage_80).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="helper">
        These results are not new untouched test confirmation. Both target
        parameterisations and all trial records remain available.
      </p>
    </section>
  );
}
