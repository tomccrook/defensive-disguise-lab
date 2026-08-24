import type { Dispatch, SetStateAction } from "react";
import { signed } from "./types";
import type { Sample, Reveal } from "./types";
type Props = {
  sample: Sample;
  reveal: Reveal | null;
  reduced: boolean;
  time: number;
  playing: boolean;
  speed: number;
  setTime: Dispatch<SetStateAction<number>>;
  setPlaying: Dispatch<SetStateAction<boolean>>;
  setSpeed: (v: number) => void;
};
export function PlaybackControls({
  sample,
  reveal,
  reduced,
  time,
  playing,
  speed,
  setTime,
  setPlaying,
  setSpeed,
}: Props) {
  const min = sample.cutoff - 2,
    max = reveal ? sample.horizon : sample.cutoff;
  return (
    <>
      {" "}
      <div className="playback">
        <button
          onClick={() => {
            setTime(min);
            setPlaying(!reduced);
          }}
          disabled={reduced}
        >
          Replay history
        </button>
        <button
          onClick={() => {
            setTime(sample.cutoff);
            setPlaying(!reduced);
          }}
          disabled={!reveal || reduced}
        >
          Play reveal
        </button>
        <button onClick={() => setPlaying(false)} disabled={!playing}>
          Pause
        </button>
        <label>
          Speed
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
          </select>
        </label>
      </div>
      {reduced && (
        <p className="helper">
          Reduced motion: static endpoints and manual stepping enabled.
        </p>
      )}
      <div className="scrubber">
        <label htmlFor="data-time">Data time relative to snap</label>
        <output>{signed(time)} s</output>
        <input
          id="data-time"
          type="range"
          min={min}
          max={max}
          step={0.1}
          value={time}
          onChange={(e) => {
            setPlaying(false);
            setTime(Number(e.target.value));
          }}
        />
        <label className="sr-only" htmlFor="data-time-number">
          Data time in seconds
        </label>
        <input
          id="data-time-number"
          type="number"
          min={min}
          max={max}
          step={0.1}
          value={Number(time.toFixed(1))}
          onChange={(e) => {
            setPlaying(false);
            setTime(Math.max(min, Math.min(max, Number(e.target.value))));
          }}
        />
        <button
          onClick={() =>
            setTime((t) => Math.max(min, Number((t - 0.1).toFixed(1))))
          }
        >
          −0.1 s
        </button>
        <button
          onClick={() =>
            setTime((t) => Math.min(max, Number((t + 0.1).toFixed(1))))
          }
        >
          +0.1 s
        </button>
      </div>
    </>
  );
}
