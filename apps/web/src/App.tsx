import { useEffect, useRef, useState } from "react";
import { EvidenceView } from "./EvidenceViews";
import { CasesView } from "./CasesView";
import { SavedView } from "./SavedView";
import { ForecastView } from "./ForecastView";
import { get } from "./types";
import { Sidebar } from "./Sidebar";
import type { View } from "./Sidebar";
import { WorkbenchHeader, SampleLine } from "./WorkbenchHeader";
import { SelectionBar } from "./SelectionBar";
import type {
  DataInfo,
  Family,
  Forecast,
  Metrics,
  Play,
  Sample,
} from "./types";

export default function App() {
  const [view, setView] = useState<View>("forecast"),
    [cohort, setCohort] = useState("league");
  const [data, setData] = useState<DataInfo | null>(null),
    [metrics, setMetrics] = useState<Metrics | null>(null);
  const [plays, setPlays] = useState<Play[]>([]),
    [playKey, setPlayKey] = useState("");
  const [cutoff, setCutoff] = useState(-0.1),
    [horizon, setHorizon] = useState(1),
    [family, setFamily] = useState<Family>("geometry");
  const [payload, setPayload] = useState<{
    key: string;
    sample: Sample;
    forecast: Forecast;
  } | null>(null);
  const [error, setError] = useState(""),
    [dataError, setDataError] = useState(""),
    [notice, setNotice] = useState("");
  const [retry, setRetry] = useState(0),
    [saved, setSaved] = useState<{ family: Family; forecast: Forecast }[]>([]);
  const [exposed, setExposed] = useState<Set<string>>(new Set());
  const requestSequence = useRef(0),
    title = useRef<HTMLHeadingElement>(null);
  const pendingCase = useRef<string | null>(null);
  const initialModel = useRef(false);
  const key = `${cohort}:${playKey}:${cutoff}:${horizon}:${family}`;
  const ready = payload?.key === key;
  useEffect(() => {
    const ac = new AbortController();
    setDataError("");
    get<DataInfo>("data", ac.signal)
      .then((r) => {
        setData(r);
        if (!initialModel.current) {
          setFamily(r.selected_demo_family ?? "geometry");
          initialModel.current = true;
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") setDataError(e.message);
      });
    get<Metrics>("metrics", ac.signal)
      .then(setMetrics)
      .catch((e) => {
        if (e.name !== "AbortError") setDataError(e.message);
      });
    return () => ac.abort();
  }, [retry]);
  useEffect(() => {
    const ac = new AbortController();
    setPlays([]);
    setPlayKey("");
    setPayload(null);
    setError("");
    get<{ plays: Play[] }>(`plays?view=${cohort}&held_out=true`, ac.signal)
      .then((r) => {
        setPlays(r.plays);
        if (r.plays.length) {
          const requested = r.plays.find(
            (p) => `${p.game_id}/${p.play_id}` === pendingCase.current,
          );
          const chosen = requested ?? r.plays[0];
          setPlayKey(`${chosen.game_id}/${chosen.play_id}`);
          pendingCase.current = null;
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => ac.abort();
  }, [cohort, retry]);
  useEffect(() => {
    setPayload(null);
    setError("");
    setNotice("");
    if (!playKey) return;
    const ac = new AbortController(),
      sequence = ++requestSequence.current;
    const run = async () => {
      const sample = await get<Sample>(
        `sample/${playKey}?cutoff=${cutoff}&horizon=${horizon}`,
        ac.signal,
      );
      const forecast = await get<Forecast>("forecast", ac.signal, {
        family,
        sample,
      });
      if (sequence === requestSequence.current && !ac.signal.aborted)
        setPayload({ key, sample, forecast });
    };
    run().catch((e) => {
      if (e.name !== "AbortError" && sequence === requestSequence.current)
        setError(e.message);
    });
    return () => ac.abort();
  }, [key, retry]);
  const selected = plays.find((p) => `${p.game_id}/${p.play_id}` === playKey);
  function navigate(next: View) {
    setView(next);
    requestAnimationFrame(() => title.current?.focus());
  }
  function selectCase(game: string, play: string, model?: Family) {
    setFamily(model ?? data?.selected_demo_family ?? "geometry");
    setCutoff(-0.1);
    setHorizon(1);
    setExposed((s) => new Set([...s, `${game}/${play}`]));
    pendingCase.current = `${game}/${play}`;
    if (cohort === "league") {
      setPlayKey(pendingCase.current);
      pendingCase.current = null;
    }
    setCohort("league");
    setView("forecast");
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to workbench
      </a>
      <Sidebar view={view} onNavigate={navigate} data={data} />
      <main id="main">
        <WorkbenchHeader
          titleRef={title}
          error={dataError}
          onEvidence={() => navigate("evaluation")}
          onRetry={() => setRetry((r) => r + 1)}
          data={data}
        />
        {view === "forecast" && (
          <>
            <SelectionBar
              models={
                data?.models ?? [
                  "persistence",
                  "velocity",
                  "geometry",
                  "temporal",
                  "pooled",
                  "graph",
                ]
              }
              cohort={cohort}
              setCohort={setCohort}
              playKey={playKey}
              setPlayKey={setPlayKey}
              plays={plays}
              cutoff={cutoff}
              setCutoff={setCutoff}
              horizon={horizon}
              setHorizon={setHorizon}
              family={family}
              setFamily={setFamily}
            />
            <SampleLine selected={selected} count={plays.length} />
            {exposed.has(playKey) && (
              <p className="notice">
                Exploratory replay: this play has already been revealed or
                selected retrospectively. Changing the cutoff is an explanation,
                not a fresh blind test.
              </p>
            )}
            {notice && (
              <p role="status" className="saved-notice">
                {notice}
              </p>
            )}
            {error ? (
              <section className="panel" role="alert">
                <h2>Forecast unavailable</h2>
                <p>{error}</p>
                <button onClick={() => setRetry((r) => r + 1)}>
                  Retry forecast
                </button>
              </section>
            ) : ready && payload ? (
              <ForecastView
                synthetic={selected?.synthetic ?? true}
                key={key}
                sample={payload.sample}
                forecast={payload.forecast}
                family={family}
                onReveal={() => setExposed((s) => new Set([...s, playKey]))}
                onSave={(forecast) => {
                  setSaved((s) => [...s, { family, forecast }]);
                  setNotice(
                    "Forecast saved with its input/model/calibration versions.",
                  );
                }}
              />
            ) : (
              <section
                className="panel loading-panel"
                role="status"
                aria-busy={!!playKey}
              >
                <h2>
                  {!playKey && plays.length === 0
                    ? "Waiting for an eligible cohort"
                    : "Preparing causal forecast…"}
                </h2>
                <p>
                  {!playKey
                    ? "No eligible play selected. Check the view or API status."
                    : "Reading permitted history and freezing the model distribution."}
                </p>
              </section>
            )}
            <SavedView saved={saved} onClear={() => setSaved([])} />
          </>
        )}
        {(view === "evaluation" || view === "timing") && data && (
          <EvidenceView
            data={data}
            metrics={metrics}
            timing={view === "timing"}
          />
        )}
        {view === "cases" && <CasesView onSelect={selectCase} />}
        <footer>
          Offline snap-conditioned study · idealised player tracking view ·
          endpoint forecasts in yards · future snap alignment is retrospective
        </footer>
      </main>
    </div>
  );
}
