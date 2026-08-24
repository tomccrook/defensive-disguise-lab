export type Family =
  | "persistence"
  | "velocity"
  | "geometry"
  | "temporal"
  | "pooled"
  | "graph"
  | "pooled_mixture"
  | "graph_mixture";
export type Point = { t: number; x: number; y: number };
export type Entity = {
  entity_id: string;
  side: "offense" | "defense";
  observations: Point[];
  anchor: Point;
};
export type Sample = {
  dataset_id: string;
  game_id: string;
  play_id: string;
  cutoff: number;
  horizon: number;
  anchor_seconds: number;
  line_of_scrimmage: number;
  entities: Entity[];
  feature_version: string;
};
export type Defender = {
  entity_id: string;
  anchor: [number, number];
  mean_displacement: [number, number];
  covariance: [[number, number], [number, number]];
  modes: {
    weight: number;
    mean_displacement: [number, number];
    covariance: [[number, number], [number, number]];
  }[];
};
export type Forecast = {
  status: "ok";
  input_hash: string;
  model_id: string;
  dataset_id: string;
  calibration_id: string;
  feature_version: string;
  target_version: string;
  split_id: string;
  cutoff: number;
  anchor_seconds: number;
  horizon: number;
  units: "yards";
  distribution: string;
  region_note: string;
  defenders: Defender[];
};
export type Reveal = {
  outcomes: { entity_id: string; displacement: [number, number] | null }[];
  future: Record<string, Point[]>;
  note: string;
};
export type Play = {
  game_id: string;
  play_id: string;
  date: string;
  season: number | null;
  offense: string;
  defense: string;
  split: string;
  synthetic: boolean;
};
export type DataInfo = {
  dataset: {
    dataset_id: string;
    games: number;
    plays: number;
    rows: number;
    cadence_hz: number;
    season: number | null;
    synthetic: boolean;
    weeks?: number[];
  };
  split: { split_id: string; games: Record<string, string> };
  scope: string;
  access_status: string;
  verified_nfl_games: number;
  verified_bengals_plays: number;
  models: Family[];
  selected_demo_family?: Family;
};
export type Metric = {
  family: Family;
  energy_score: number;
  nll: number;
  endpoint_error: number;
  marginal_coverage_80: number;
  games: number;
  plays: number;
};
export type Timing = {
  family: Family;
  game_id: string;
  play_id: string;
  belief_change_nats: number;
  late_minus_early_energy: number;
  log_density_improvement: number;
};
export type Metrics = {
  baseline_correction?: import("./BaselineCorrection").Correction;
  primary: Metric[];
  graph_minus_pooled: {
    difference: number;
    interval: [number, number];
    games: number;
    status: string;
  };
  timing: Timing[];
  scope: string;
  opponent_primary?: Metric[];
  opponent_graph_minus_pooled?: Metrics["graph_minus_pooled"];
  all_variants?: (Omit<Metric, "family"> & { family: string })[];
  ablations?: Record<string, Metrics["graph_minus_pooled"]>;
  robustness?: {
    stress: {
      family: string;
      stress: string;
      plays: number;
      energy_score: number;
      change_from_unmodified: Metrics["graph_minus_pooled"];
    }[];
    monte_carlo_precision: {
      draws: number;
      seed: number;
      graph_minus_pooled: number;
    }[];
    concept_distinctions: Record<
      string,
      Record<
        string,
        {
          spearman: number | null;
          top_decile_overlap_fraction: number | null;
        }
      >
    >;
  };
};
export const familyLabels: Record<Family, string> = {
  persistence: "Position persistence",
  velocity: "Constant velocity",
  geometry: "Geometry / boosted trees",
  temporal: "Individual GRU",
  pooled: "GRU + pooled context",
  graph: "Temporal graph",
  pooled_mixture: "Pooled · shared mixture",
  graph_mixture: "Graph · shared mixture",
};
export const signed = (n: number) => `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
export const fixed = (n: number) => n.toFixed(3);

export async function get<T>(
  url: string,
  signal?: AbortSignal,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/${url}`, {
    signal,
    ...(body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      result.detail?.message ??
        result.detail?.status ??
        `Request failed (${response.status})`,
    );
  return result as T;
}
