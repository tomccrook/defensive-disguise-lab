import { useState } from "react";
import { familyLabels, fixed, get } from "./types";
import { TimingTable } from "./TimingTable";
import { ControlledStudies } from "./ControlledStudies";
import { EvidenceFigures } from "./EvidenceFigures";
import { SensitivityEvidence } from "./SensitivityEvidence";
import { BaselineCorrection } from "./BaselineCorrection";
import { TrialHistory } from "./TrialHistory";
import type { DataInfo, Metrics } from "./types";

export function EvidenceView({
  data,
  metrics,
  timing = false,
}: {
  data: DataInfo;
  metrics: Metrics | null;
  timing?: boolean;
}) {
  const [report, setReport] = useState(""),
    [reportError, setReportError] = useState("");
  async function openReport(name: string) {
    setReport("");
    setReportError("");
    try {
      const r = await get<{ markdown: string }>(`evidence/${name}`);
      setReport(r.markdown);
    } catch (e) {
      setReportError((e as Error).message);
    }
  }
  return (
    <section className="panel evidence-view">
      <span className="eyebrow">
        {timing ? "Timing / models" : "Evaluation / data"}
      </span>
      <h2>
        {timing
          ? "What did more information add?"
          : "Evidence before confidence"}
      </h2>
      <p className="notice">
        {data.scope} Verified NFL games in this study: {data.verified_nfl_games}
        ; verified Bengals plays: {data.verified_bengals_plays}.
      </p>
      <p>
        The separately audited public sample is one 2017 NE–KC game with zero
        Bengals plays. It is excluded from training and evaluation.
      </p>
      {!metrics ? (
        <p role="status">Loading metrics…</p>
      ) : (
        <>
          <div className="table-scroll">
            <table>
              <caption>
                Locked{" "}
                {data.dataset.synthetic ? "synthetic" : "2022 chronological"}{" "}
                test · −0.1 s cutoff · +1.0 s target · −1.0 s anchor ·{" "}
                {metrics.primary[0]?.plays} plays / {metrics.primary[0]?.games}{" "}
                game clusters
              </caption>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Energy score ↓ (yd)</th>
                  <th>Endpoint error (yd)</th>
                  <th>NLL (yard density)</th>
                  <th>Nominal 80% coverage</th>
                </tr>
              </thead>
              <tbody>
                {metrics.primary.map((r) => (
                  <tr key={r.family}>
                    <th scope="row">{familyLabels[r.family]}</th>
                    <td>{fixed(r.energy_score)}</td>
                    <td>{fixed(r.endpoint_error)}</td>
                    <td>{fixed(r.nll)}</td>
                    <td>{(r.marginal_coverage_80 * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Graph minus pooled energy score:{" "}
            <strong>{fixed(metrics.graph_minus_pooled.difference)} yd</strong> ·
            game bootstrap 95% interval [
            {metrics.graph_minus_pooled.interval.map(fixed).join(", ")}]. Only{" "}
            {metrics.graph_minus_pooled.games} test games as resampling
            clusters.
          </p>
          <p className="helper">
            Lower scores are better. Energy score averages 64 independent
            Gaussian samples per defender, then defenders within play, then
            plays. Uncertainty resamples entire games. Variance scaling uses
            validation only. Compare measured coverage with the nominal level;
            an ellipse is a marginal region for one defender.
          </p>
          {timing && (
            <>
              <TimingTable
                metrics={metrics}
                families={data.models.filter((f) =>
                  metrics.timing.some((r) => r.family === f),
                )}
              />
            </>
          )}
        </>
      )}
      <div className="evidence-columns">
        <div>
          <h3>Four separate measurements</h3>
          <dl>
            <dt>Ambiguity</dt>
            <dd>Covariance trace before outcome.</dd>
            <dt>Belief change</dt>
            <dd>Early/late distribution comparison.</dd>
            <dt>Surprise</dt>
            <dd>Realised negative log density.</dd>
            <dt>Overconfident failure</dt>
            <dd>
              Validation-sharp region miss; qualification depends on
              calibration.
            </dd>
          </dl>
        </div>
        <div>
          <h3>Data lineage</h3>
          <dl>
            <dt>Dataset games / eligible plays</dt>
            <dd>
              {data.dataset.games} / {data.dataset.plays}
            </dd>
            <dt>Tracking rows</dt>
            <dd>{data.dataset.rows.toLocaleString()}</dd>
            <dt>Split games</dt>
            <dd>
              {["train", "validation", "test"]
                .map(
                  (s) =>
                    `${s}: ${Object.values(data.split.games).filter((v) => v === s).length}`,
                )
                .join(" · ")}
            </dd>
            <dt>Source status</dt>
            <dd>{data.access_status}</dd>
            <dt>Execution</dt>
            <dd>Mac CPU · Windows GPU pending · $0 cloud</dd>
          </dl>
        </div>
      </div>
      {metrics && <ControlledStudies metrics={metrics} />}
      {metrics?.baseline_correction && (
        <BaselineCorrection result={metrics.baseline_correction} />
      )}
      {metrics && <SensitivityEvidence metrics={metrics} />}
      {!data.dataset.synthetic && <EvidenceFigures timing={timing} />}
      <h3>Open the audit trail</h3>
      <TrialHistory />
      <div className="report-links">
        {[
          "research-report",
          "bengals-cases",
          "interview",
          "data-access",
          "milestones",
          "ui-design-decisions",
          "empirical-plan",
        ].map((name) => (
          <button key={name} onClick={() => openReport(name)}>
            {name.replaceAll("-", " ")}
          </button>
        ))}
      </div>
      {reportError && <p role="alert">{reportError}</p>}
      {report && (
        <pre className="report-text" tabIndex={0}>
          {report}
        </pre>
      )}
    </section>
  );
}
