import { useEffect, useRef } from "react";
import type { Forecast, Reveal, Sample } from "./types";

type Props = {
  sample: Sample;
  forecast: Forecast;
  reveal: Reveal | null;
  time: number;
  selected: string;
};
const FIELD_WIDTH = 160 / 3;

export function Field({ sample, forecast, reveal, time, selected }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  // Camera uses permitted geometry only, including before reveal.
  const left = Math.max(0, Math.min(50, sample.line_of_scrimmage - 25));
  const right = left + 70;
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const paint = () => {
      const width = element.clientWidth,
        height = (width * FIELD_WIDTH) / 70;
      const dpr = window.devicePixelRatio || 1;
      element.width = width * dpr;
      element.height = height * dpr;
      const c = element.getContext("2d");
      if (!c) return;
      c.scale(dpr, dpr);
      const scale = width / 70;
      const X = (x: number) => (x - left) * scale,
        Y = (y: number) => height - y * scale;
      c.fillStyle = "#1b2927";
      c.fillRect(0, 0, width, height);
      c.lineWidth = 1;
      for (let x = Math.ceil(left / 5) * 5; x <= right; x += 5) {
        c.strokeStyle = x % 10 === 0 ? "#60736c" : "#354c43";
        c.beginPath();
        c.moveTo(X(x), 0);
        c.lineTo(X(x), height);
        c.stroke();
        if (x % 10 === 0 && x >= 10 && x <= 110) {
          c.fillStyle = "#bac6bc";
          c.font = "12px ui-monospace, monospace";
          const yardline = String(Math.min(x - 10, 110 - x));
          c.fillText(yardline, X(x) + 4, 20);
          c.fillText(yardline, X(x) + 4, height - 10);
        }
        for (const y of [0.7, FIELD_WIDTH - 0.7, 23.58, 29.75]) {
          c.beginPath();
          c.moveTo(X(x), Y(y) - 3);
          c.lineTo(X(x), Y(y) + 3);
          c.stroke();
        }
      }
      c.strokeStyle = "#e6c37b";
      c.setLineDash([6, 5]);
      c.beginPath();
      c.moveTo(X(sample.line_of_scrimmage), 0);
      c.lineTo(X(sample.line_of_scrimmage), height);
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = "#e6c37b";
      c.font = "11px system-ui";
      c.fillText("LOS", X(sample.line_of_scrimmage) + 5, 38);
      c.fillStyle = "#bac6bc";
      c.fillText("OFFENCE →", width - 100, height - 30);
      for (const e of sample.entities) {
        const allowed = e.observations.filter(
          (p) => p.t <= Math.min(time, sample.cutoff) + 1e-7,
        );
        if (!allowed.length) continue;
        c.strokeStyle = e.side === "defense" ? "#b57e5c" : "#a0adb7";
        c.globalAlpha = 0.55;
        c.beginPath();
        allowed.forEach((p, i) =>
          i ? c.lineTo(X(p.x), Y(p.y)) : c.moveTo(X(p.x), Y(p.y)),
        );
        c.stroke();
        c.globalAlpha = 1;
        const p = allowed[allowed.length - 1],
          radius = e.entity_id === selected ? 8 : 6;
        c.fillStyle = e.side === "defense" ? "#f5a168" : "#e5ecef";
        c.strokeStyle = "#121719";
        c.lineWidth = 2;
        c.beginPath();
        if (e.side === "defense") c.arc(X(p.x), Y(p.y), radius, 0, Math.PI * 2);
        else c.rect(X(p.x) - radius, Y(p.y) - radius, radius * 2, radius * 2);
        c.fill();
        c.stroke();
        c.lineWidth = 1;
        if (e.side === "defense") {
          c.fillStyle = "#f8e0ca";
          c.font = "10px ui-monospace, monospace";
          const index = forecast.defenders.findIndex(
            (d) => d.entity_id === e.entity_id,
          );
          c.fillText(
            e.entity_id === selected ? e.entity_id : `D${index + 1}`,
            X(p.x) + 9,
            Y(p.y) - 6,
          );
        }
      }
      for (const d of forecast.defenders) {
        // Render component hypotheses; a mixture's moment mean can fall between modes.
        const modes = d.modes?.length ? d.modes : [{ ...d, weight: 1 }];
        for (const mode of modes) {
          const x = d.anchor[0] + mode.mean_displacement[0],
            y = d.anchor[1] + mode.mean_displacement[1];
          c.globalAlpha = Math.max(0.25, mode.weight);
          c.strokeStyle = "#f5a168";
          c.setLineDash([3, 3]);
          c.beginPath();
          c.arc(X(x), Y(y), 7, 0, Math.PI * 2);
          c.stroke();
          c.setLineDash([]);
          if (d.entity_id === selected) {
            const [[a, b], [, v]] = mode.covariance;
            const root = Math.sqrt((a - v) ** 2 + 4 * b * b);
            const major =
              Math.sqrt(Math.max((a + v + root) / 2, 0) * 3.218875825) * scale;
            const minor =
              Math.sqrt(Math.max((a + v - root) / 2, 0) * 3.218875825) * scale;
            c.fillStyle = "rgba(245,161,104,.10)";
            c.strokeStyle = "#f5a168";
            c.beginPath();
            c.ellipse(
              X(x),
              Y(y),
              major,
              minor,
              -Math.atan2(2 * b, a - v) / 2,
              0,
              Math.PI * 2,
            );
            c.fill();
            c.stroke();
          }
        }
        c.globalAlpha = 1;
      }
      if (reveal) {
        for (const points of Object.values(reveal.future)) {
          const visible = points.filter((p) => p.t <= time + 1e-7);
          if (!visible.length) continue;
          c.strokeStyle = "#87d4cc";
          c.lineWidth = 2;
          c.beginPath();
          visible.forEach((p, i) =>
            i ? c.lineTo(X(p.x), Y(p.y)) : c.moveTo(X(p.x), Y(p.y)),
          );
          c.stroke();
          const p = visible[visible.length - 1];
          c.beginPath();
          c.moveTo(X(p.x) - 5, Y(p.y) - 5);
          c.lineTo(X(p.x) + 5, Y(p.y) + 5);
          c.moveTo(X(p.x) + 5, Y(p.y) - 5);
          c.lineTo(X(p.x) - 5, Y(p.y) + 5);
          c.stroke();
          c.lineWidth = 1;
        }
      }
    };
    const observer = new ResizeObserver(paint);
    observer.observe(element);
    paint();
    return () => observer.disconnect();
  }, [sample, forecast, reveal, time, selected, left, right]);
  return (
    <figure className="field-figure">
      <canvas
        ref={canvas}
        aria-label="Scaled field with observed players and marginal endpoint forecasts. Defender values are in the table below."
        role="img"
      />
      <figcaption>
        <span>
          70 × 53⅓ yd window · coordinates {left.toFixed(1)}–{right.toFixed(1)}
        </span>
        <span>
          Circle: defence · square: offence · dashed:{" "}
          {forecast.distribution === "shared_component_mixture"
            ? "component hypotheses"
            : "forecast"}
          {reveal ? " · × actual" : ""}
        </span>
      </figcaption>
    </figure>
  );
}
