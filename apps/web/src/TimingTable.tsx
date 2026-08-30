import { familyLabels, fixed } from "./types";
import type { Metrics, Family } from "./types";
export function TimingTable({
  metrics,
  families,
}: {
  metrics: Metrics;
  families: Family[];
}) {
  return (
    <>
      <h3>Early → late on one common cohort</h3>
      <p>
        Same +1.0 s endpoint and −1.0 s anchor. Belief change is a symmetric
        marginal Gaussian KL in nats; realised density improvement uses the
        observed outcome.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Mean belief change (nats)</th>
              <th>Late − early energy (yd)</th>
              <th>Log density improvement</th>
            </tr>
          </thead>
          <tbody>
            {families.map((family) => {
              const rows = metrics.timing.filter((t) => t.family === family);
              const average = (
                k:
                  | "belief_change_nats"
                  | "late_minus_early_energy"
                  | "log_density_improvement",
              ) => rows.reduce((s, r) => s + r[k], 0) / rows.length;
              return (
                <tr key={family}>
                  <th scope="row">{familyLabels[family]}</th>
                  <td>{fixed(average("belief_change_nats"))}</td>
                  <td>{fixed(average("late_minus_early_energy"))}</td>
                  <td>{fixed(average("log_density_improvement"))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
