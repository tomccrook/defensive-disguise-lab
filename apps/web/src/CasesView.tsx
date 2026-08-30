import { useEffect, useState } from "react";
import { get } from "./types";
import type { Family } from "./types";

export function CasesView({
  onSelect,
}: {
  onSelect: (game: string, play: string, model?: Family) => void;
}) {
  const [cases, setCases] = useState<
      {
        kind: string;
        game_id: string;
        play_id: string;
        note: string;
        model_family?: Family;
      }[]
    >([]),
    [error, setError] = useState("");
  const [synthetic, setSynthetic] = useState(true);
  useEffect(() => {
    const ac = new AbortController();
    get<{ cases: typeof cases; synthetic: boolean }>("cases", ac.signal)
      .then((r) => {
        setCases(r.cases);
        setSynthetic(r.synthetic);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => ac.abort();
  }, []);
  return (
    <section className="panel">
      <span className="eyebrow">League / opponent cases</span>
      <h2>Illustrate a forecast. Keep the claim bounded.</h2>
      <p className="notice">
        {synthetic
          ? "Retrospective synthetic case selection. Team labels are illustrative."
          : "Retrospective held-out 2022 cases. The model was selected on validation; case rules were registered before test evaluation."}
      </p>
      <p>
        Cases are selected by registered rules after evaluation: seeded ordinary
        play, greatest predictive spread, and greatest realised NLL. A selected
        case cannot establish overall skill or intent.
      </p>
      {error && <p role="alert">{error}</p>}
      {!cases.length && !error && <p role="status">No cases in this cohort.</p>}
      <ul className="case-list">
        {cases.map((c) => (
          <li key={c.kind}>
            <div>
              <span className="eyebrow">{c.kind}</span>
              <h3>
                {c.game_id} / {c.play_id}
              </h3>
              <p>
                {c.note} · {synthetic ? "synthetic" : "historical"} test.
              </p>
            </div>
            <button
              onClick={() => onSelect(c.game_id, c.play_id, c.model_family)}
            >
              Open replay
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
