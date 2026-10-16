import { familyLabels, signed } from "./types";
import type { Family, Play } from "./types";
type Props = {
  cohort: string;
  setCohort: (v: string) => void;
  playKey: string;
  setPlayKey: (v: string) => void;
  plays: Play[];
  cutoff: number;
  setCutoff: (v: number) => void;
  horizon: number;
  setHorizon: (v: number) => void;
  family: Family;
  models: Family[];
  setFamily: (v: Family) => void;
};
export function SelectionBar({
  cohort,
  setCohort,
  playKey,
  setPlayKey,
  plays,
  cutoff,
  setCutoff,
  horizon,
  setHorizon,
  family,
  models,
  setFamily,
}: Props) {
  return (
    <section className="control-bar" aria-label="Forecast selection">
      <label>
        Study view
        <select value={cohort} onChange={(e) => setCohort(e.target.value)}>
          <option value="bengals_offense">
            CIN offence vs opposing defences
          </option>
          <option value="bengals_defense">CIN defending — separate view</option>
          <option value="league">League-wide · all teams</option>
        </select>
      </label>
      <label>
        Held-out play
        <select
          value={playKey}
          onChange={(e) => setPlayKey(e.target.value)}
          disabled={!plays.length}
        >
          {plays.map((p) => (
            <option
              key={`${p.game_id}/${p.play_id}`}
              value={`${p.game_id}/${p.play_id}`}
            >
              {p.offense} vs {p.defense} · {p.game_id} / {p.play_id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Pre-snap cutoff
        <select
          value={cutoff}
          onChange={(e) => setCutoff(Number(e.target.value))}
        >
          {[-1, -0.5, -0.1].map((c) => (
            <option key={c} value={c}>
              {signed(c)} s
            </option>
          ))}
        </select>
      </label>
      <label>
        Target horizon
        <select
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
        >
          <option value={0.5}>+0.5 s</option>
          <option value={1}>+1.0 s</option>
        </select>
      </label>
      <label>
        Forecast model
        <select
          value={family}
          onChange={(e) => setFamily(e.target.value as Family)}
        >
          {models.map((f) => (
            <option value={f} key={f}>
              {familyLabels[f]}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
