import { useEffect, useState } from "react";
import { familyLabels, get } from "./types";
import type { Family } from "./types";

type Trial = {
  trial_id: string;
  status: string;
  config: {
    family?: Family;
    variant?: string;
    protocol?: string;
    epochs: number;
    code_commit?: string;
    source_tree_hash?: string;
  };
  result: {
    seconds?: number;
    parameters?: number | null;
    history?: unknown[];
  } | null;
};
export function TrialHistory() {
  const [trials, setTrials] = useState<Trial[]>([]);
  const [error, setError] = useState("");
  const [maximum, setMaximum] = useState(24);
  useEffect(() => {
    const ac = new AbortController();
    get<{ trials: Trial[]; max_trials?: number }>("jobs", ac.signal)
      .then((r) => {
        setTrials(r.trials);
        setMaximum(r.max_trials ?? 24);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => ac.abort();
  }, []);
  return (
    <details>
      <summary>
        Preserved trial history · {trials.length} / {maximum} trials
      </summary>
      <p>
        CPU research runs; every attempt counts toward the development budget.
        Full identities, histories and provenance are available in the research
        evidence export. An empty hash marks a legacy trial.
      </p>
      {error && <p role="alert">{error}</p>}
      <div className="table-scroll">
        <table>
          <caption>
            Completed and failed attempts remain in the registry
          </caption>
          <thead>
            <tr>
              <th>Trial</th>
              <th>Model</th>
              <th>Status</th>
              <th>Epochs</th>
              <th>Seconds</th>
              <th>Parameters</th>
              <th>Code</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t) => (
              <tr key={t.trial_id}>
                <th scope="row" title={t.trial_id}>
                  {t.trial_id.slice(0, 8)}
                </th>
                <td>
                  {t.config.family
                    ? familyLabels[t.config.family]
                    : t.config.variant?.replaceAll("_", " ")}{" "}
                  {t.config.protocol ?? "synthetic"}
                </td>
                <td>{t.status}</td>
                <td>{t.result?.history?.length ?? "—"}</td>
                <td>{t.result?.seconds?.toFixed(2) ?? "—"}</td>
                <td>{t.result?.parameters ?? "—"}</td>
                <td
                  title={
                    t.config.source_tree_hash ??
                    "Legacy trial; source hash was not recorded"
                  }
                >
                  {t.config.code_commit?.slice(0, 7) ?? "unrecorded"} ·{" "}
                  {t.config.source_tree_hash?.slice(0, 8) ?? "unrecorded"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
