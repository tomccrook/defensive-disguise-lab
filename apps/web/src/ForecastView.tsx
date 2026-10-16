import { useEffect, useState } from "react";
import { Field } from "./Field";
import { get, signed } from "./types";
import { DefenderLedger } from "./DefenderLedger";
import { PlaybackControls } from "./PlaybackControls";
import type { Family, Forecast, Reveal, Sample } from "./types";

type Props = {
  sample: Sample;
  forecast: Forecast;
  family: Family;
  synthetic: boolean;
  onSave: (f: Forecast) => void;
  onReveal: () => void;
};
export function ForecastView({
  sample,
  forecast,
  family,
  synthetic,
  onSave,
  onReveal,
}: Props) {
  const [reveal, setReveal] = useState<Reveal | null>(null),
    [error, setError] = useState("");
  const [busy, setBusy] = useState(false),
    [time, setTime] = useState(sample.cutoff),
    [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1),
    [selected, setSelected] = useState(forecast.defenders[0].entity_id);
  const [reduced, setReduced] = useState(
    window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!playing || reduced) return;
    let frame = 0,
      previous = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.1) * speed;
      previous = now;
      setTime((t) => {
        const end = reveal ? sample.horizon : sample.cutoff;
        const next = Math.min(t + dt, end);
        if (next === end) setPlaying(false);
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, reduced, reveal, sample, speed]);
  async function showOutcome() {
    setBusy(true);
    setError("");
    try {
      const outcome = await get<Reveal>("reveal", undefined, {
        input_hash: forecast.input_hash,
      });
      setReveal(outcome);
      setTime(sample.horizon);
      onReveal();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const chosen =
    forecast.defenders.find((d) => d.entity_id === selected) ??
    forecast.defenders[0];
  const trace = chosen.covariance[0][0] + chosen.covariance[1][1];
  return (
    <>
      <div className="workspace-grid">
        <section className="panel field-panel" aria-labelledby="field-heading">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">
                {reveal ? "Observed outcome" : "Forecast frozen"}
              </span>
              <h2 id="field-heading">
                {reveal ? "Test the belief" : "What happens after the snap?"}
              </h2>
            </div>
            <span className="badge">
              {reveal ? "REVEALED" : "FUTURE HIDDEN"}
            </span>
          </div>
          <Field
            sample={sample}
            forecast={forecast}
            reveal={reveal}
            time={time}
            selected={chosen.entity_id}
          />
          <PlaybackControls
            sample={sample}
            reveal={reveal}
            reduced={reduced}
            time={time}
            playing={playing}
            speed={speed}
            setTime={setTime}
            setPlaying={setPlaying}
            setSpeed={setSpeed}
          />{" "}
          <p className="helper">
            10 Hz {synthetic ? "synthetic" : "historical source"} observations.
            Endpoints only: connecting lines are visual guides, not learned
            trajectories.
          </p>
        </section>
        <aside className="insight-panel" aria-label="Forecast evidence">
          <div className="panel">
            <span className="eyebrow">01 / Observations</span>
            <h2>One fixed target</h2>
            <dl>
              <dt>Input cutoff</dt>
              <dd>{signed(sample.cutoff)} s</dd>
              <dt>Common anchor</dt>
              <dd>−1.0 s</dd>
              <dt>Target after snap</dt>
              <dd>+{sample.horizon.toFixed(1)} s</dd>
              <dt>Player histories</dt>
              <dd>22 / 22</dd>
              <dt>History allowance</dt>
              <dd>2.0 s</dd>
            </dl>
          </div>
          <div className="panel">
            <span className="eyebrow">02 / Belief</span>
            <h2>
              {chosen.entity_id}{" "}
              {chosen.modes?.length ? "moment mean" : "endpoint"}
            </h2>
            <p className="large-number">
              {chosen.mean_displacement[0].toFixed(2)} <small>Δx yd</small>
            </p>
            <p className="large-number">
              {chosen.mean_displacement[1].toFixed(2)} <small>Δy yd</small>
            </p>
            <p>Spread: {trace.toFixed(2)} yd² covariance trace.</p>
            <p className="helper">
              {forecast.region_note} Coverage is measured in Evaluation.
            </p>
          </div>
          <div className="reveal-block">
            <span className="eyebrow">03 / Evidence</span>
            <h2>{reveal ? "Outcome available" : "Ready to test it?"}</h2>
            <p>
              {reveal
                ? "Realised movement can now be compared with the frozen forecast."
                : "Inspect the permitted history and probability region before revealing the future."}
            </p>
            <button
              className="primary"
              disabled={busy || !!reveal}
              onClick={showOutcome}
            >
              {busy
                ? "Retrieving outcome…"
                : reveal
                  ? "Outcome revealed"
                  : "Reveal actual movement"}
            </button>
            <button onClick={() => onSave(forecast)}>Save this forecast</button>
            <p role="status" aria-live="polite">
              {reveal ? "Actual movement revealed." : ""}
            </p>
            {error && <p role="alert">{error}</p>}
          </div>
        </aside>
      </div>
      <DefenderLedger
        forecast={forecast}
        reveal={reveal}
        family={family}
        selected={selected}
        setSelected={setSelected}
      />
    </>
  );
}
