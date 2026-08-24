import type { Ref } from "react";
import type { Play, DataInfo } from "./types";

export function WorkbenchHeader({
  titleRef,
  error,
  onEvidence,
  onRetry,
  data,
}: {
  titleRef: Ref<HTMLHeadingElement>;
  error: string;
  onEvidence: () => void;
  onRetry: () => void;
  data: DataInfo | null;
}) {
  return (
    <>
      <header className="page-heading">
        <div>
          <span className="eyebrow">Spatial / temporal forecasting</span>
          <h1 ref={titleRef} tabIndex={-1}>
            Defensive Forecast Lab
          </h1>
          <p>
            Forecast early movement. Reveal the outcome. Inspect the evidence.
          </p>
        </div>
        <span className="badge synthetic-badge">
          {data?.dataset.synthetic === false
            ? "HISTORICAL NFL · 2022"
            : "SYNTHETIC FIXTURES"}
        </span>
      </header>
      <div className="gate-banner">
        <strong>
          {data?.dataset.synthetic === false
            ? "Locked historical study"
            : "Synthetic software fixtures"}
        </strong>
        <span>
          {data?.dataset.synthetic === false
            ? "2022 weeks 1–9 · league-wide training · later whole-game tests · private local replay"
            : "Fixture results demonstrate the research pipeline; no historical NFL findings in this run."}
        </span>
        <button onClick={() => onEvidence()}>Inspect data evidence →</button>
      </div>
      {error && (
        <div className="panel" role="alert">
          <h2>Evidence unavailable</h2>
          <p>{error}</p>
          <button onClick={() => onRetry()}>Retry connection</button>
        </div>
      )}
    </>
  );
}

export function SampleLine({
  selected,
  count,
}: {
  selected: Play | undefined;
  count: number;
}) {
  return (
    <div className="sample-line">
      <span>
        {selected
          ? `${selected.offense} offence · ${selected.defense} defence`
          : "Select a play"}{" "}
        ·{" "}
        <strong>
          {selected?.split === "test"
            ? selected.synthetic
              ? "HELD-OUT SYNTHETIC"
              : "HELD-OUT GAME"
            : "NO SAMPLE"}
        </strong>
      </span>
      <span>
        {count} available plays in this view · season:{" "}
        {selected?.season ?? "synthetic / none"}
      </span>
    </div>
  );
}
