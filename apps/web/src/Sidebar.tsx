export type View = "forecast" | "timing" | "cases" | "evaluation";
const navigation: { id: View; name: string; number: string }[] = [
  { id: "forecast", name: "Forecast & reveal", number: "01" },
  { id: "timing", name: "Timing & models", number: "02" },
  { id: "cases", name: "Held-out cases", number: "03" },
  { id: "evaluation", name: "Evaluation & data", number: "04" },
];

export function Sidebar({
  view,
  onNavigate,
  data,
}: {
  view: View;
  onNavigate: (v: View) => void;
  data: DataInfo | null;
}) {
  return (
    <aside className="sidebar">
      <a className="brand" href="#main">
        <span className="brand-mark" aria-hidden="true">
          DF
        </span>
        <span>
          Defensive
          <br />
          <strong>Forecast Lab</strong>
        </span>
      </a>
      <span className="eyebrow sidebar-label">Research workbench</span>
      <nav aria-label="Research views">
        {navigation.map((n) => (
          <button
            key={n.id}
            aria-current={view === n.id ? "page" : undefined}
            onClick={() => onNavigate(n.id)}
          >
            <span>{n.number}</span>
            {n.name}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="status-dot" aria-hidden="true" />
        <strong>
          {data?.dataset.synthetic === false
            ? "Historical research"
            : "Synthetic mode"}
        </strong>
        <p>
          {data?.dataset.synthetic === false
            ? "2022 partial season. League-wide study; optional team slices. No live feed."
            : "Software fixtures. Inspect the source and evaluation evidence for this run."}
        </p>
        <p>
          Independent personal research.
          <br />
          No NFL or team affiliation.
        </p>
      </div>
    </aside>
  );
}
import type { DataInfo } from "./types";
