import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("blind forecast, numeric parity, reveal and stale selections", async ({
  page,
}) => {
  let revealRequests = 0;
  page.on("request", (r) => {
    if (r.url().includes("/api/reveal")) revealRequests++;
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toHaveCount(0);
  expect(revealRequests).toBe(0);
  await page
    .getByRole("combobox", { name: "Forecast model", exact: true })
    .selectOption("geometry");
  const forecastResponse = await page.request.get("/api/plays");
  const play = (await forecastResponse.json()).plays[0];
  const sample = await (
    await page.request.get(
      `/api/sample/${play.game_id}/${play.play_id}?cutoff=-0.1&horizon=1.0`,
    )
  ).json();
  const forecast = await (
    await page.request.post("/api/forecast", {
      data: { family: "geometry", sample },
    })
  ).json();
  const first = forecast.defenders[0];
  await expect(
    page.getByRole("row").filter({
      has: page.getByRole("button", { name: first.entity_id, exact: true }),
    }),
  ).toContainText(
    first.mean_displacement.map((n: number) => n.toFixed(2)).join(" / "),
  );
  await page.screenshot({
    path: "../../artifacts/browser-nfl/browser-forecast.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1280, height: 1100 });
  await page.screenshot({
    path: "../../artifacts/browser-nfl/browser-preview.png",
  });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByRole("button", { name: "Reveal actual movement" }).click();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toBeVisible();
  expect(revealRequests).toBe(1);
  await page.screenshot({
    path: "../../artifacts/browser-nfl/browser-reveal.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Save this forecast" }).click();
  await expect(
    page.getByText("Forecast saved with its input/model/calibration versions."),
  ).toBeVisible();
  await page
    .getByText("Inspect saved endpoints and versions", { exact: true })
    .click();
  await expect(
    page.getByText("Saved marginal endpoints; outcomes excluded", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText(new RegExp(`model ${forecast.model_id}`)),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Pre-snap cutoff", exact: true })
    .selectOption("-1");
  await page
    .getByRole("combobox", { name: "Forecast model", exact: true })
    .selectOption("graph");
  await page
    .getByRole("combobox", { name: "Pre-snap cutoff", exact: true })
    .selectOption("-0.5");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Defender forecast ledger" }),
  ).toBeVisible();
  await expect(page.getByText("Temporal graph · yards")).toBeVisible();
  await expect(page.getByText("−1.0 s", { exact: true })).toBeVisible();
  await expect(page.getByText(/Exploratory replay: this play/)).toBeVisible();
});

test("keyboard, reduced motion, responsive layouts and accessibility", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Replay history" }),
  ).toBeDisabled();
  const d = page
    .locator("section[aria-labelledby=defenders-heading] tbody button")
    .nth(1);
  await d.focus();
  await page.keyboard.press("Enter");
  await expect(d).toHaveAttribute("aria-pressed", "true");
  const scrub = page.getByLabel("Data time relative to snap", { exact: true });
  await scrub.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(
    page.getByLabel("Data time in seconds", { exact: true }),
  ).toHaveValue("-0.2");
  const result = await new AxeBuilder({ page }).analyze();
  expect(result.violations).toEqual([]);
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: `../../artifacts/browser-nfl/browser-${width}.png`,
      fullPage: true,
    });
  }
});

test("timing, cases, source evidence, playback and empty/error states", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Reveal actual movement" }).click();
  await page.getByRole("button", { name: "Play reveal" }).click();
  await expect(
    page.getByLabel("Data time in seconds", { exact: true }),
  ).not.toHaveValue("1");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("button", { name: "02 Timing & models" }).click();
  await expect(
    page.getByRole("heading", { name: "Early → late on one common cohort" }),
  ).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "04 Evaluation & data" }).click();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "data access", exact: true }).click();
  await expect(page.locator(".report-text")).toContainText(/2022|removed/);
  await page.getByRole("button", { name: "03 Held-out cases" }).click();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "Open replay" }).first().click();
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toHaveCount(0);
  await page.route("**/api/plays?*", (route) =>
    route.fulfill({ json: { status: "empty_cohort", plays: [] } }),
  );
  await page
    .getByRole("combobox", { name: "Study view", exact: true })
    .selectOption("bengals_defense");
  await expect(page.getByText("Waiting for an eligible cohort")).toBeVisible();
});

test("delayed old response cannot replace the current forecast; errors recover", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  let release!: () => void;
  const started = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/sample/**", async (route) => {
    if (route.request().url().includes("cutoff=-1&")) {
      const response = await route.fetch();
      release();
      await new Promise((resolve) => setTimeout(resolve, 500));
      await route.fulfill({ response }).catch(() => {}); // Old request was intentionally aborted.
    } else await route.continue();
  });
  const cutoff = page.getByRole("combobox", {
    name: "Pre-snap cutoff",
    exact: true,
  });
  await cutoff.selectOption("-1");
  await started;
  await cutoff.selectOption("-0.5");
  await expect(
    page.getByLabel("Data time in seconds", { exact: true }),
  ).toHaveValue("-0.5");
  await page.waitForTimeout(600); // Deliberately let the stale response arrive.
  await expect(
    page.getByLabel("Data time in seconds", { exact: true }),
  ).toHaveValue("-0.5");
  await expect(page.getByText("FUTURE HIDDEN", { exact: true })).toBeVisible();
  await page.route("**/api/forecast", (route) =>
    route.fulfill({
      status: 503,
      json: { detail: { status: "model_unavailable" } },
    }),
  );
  const model = page.getByRole("combobox", {
    name: "Forecast model",
    exact: true,
  });
  await model.selectOption(
    (await model.inputValue()) === "graph" ? "pooled" : "graph",
  );
  await expect(
    page.getByRole("heading", { name: "Forecast unavailable" }),
  ).toBeVisible();
  await expect(
    page.getByText("model_unavailable", { exact: true }),
  ).toBeVisible();
  await page.unroute("**/api/forecast");
  await page.getByRole("button", { name: "Retry forecast" }).click();
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
});

test("historical mixture modes, evidence figures and exact retrospective case", async ({
  page,
}) => {
  const info = await (await page.request.get("/api/data")).json();
  test.skip(
    info.dataset.synthetic,
    "Requires the frozen historical run; fixture checks are separate.",
  );
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await page
    .getByRole("combobox", { name: "Forecast model", exact: true })
    .selectOption("geometry");
  await expect(
    page.getByText("Geometry / boosted trees · yards"),
  ).toBeVisible();
  const response = page.waitForResponse(
    (r) =>
      r.url().includes("/api/forecast") &&
      r.request().postDataJSON()?.family === "graph_mixture",
  );
  await page
    .getByRole("combobox", { name: "Forecast model", exact: true })
    .selectOption("graph_mixture");
  const prediction = await (await response).json();
  expect(prediction.distribution).toBe("shared_component_mixture");
  expect(prediction.defenders[0].modes).toHaveLength(3);
  expect(
    prediction.defenders[0].modes.reduce(
      (sum: number, m: { weight: number }) => sum + m.weight,
      0,
    ),
  ).toBeCloseTo(1, 5);
  await expect(
    page.getByText(/Selected defender component hypotheses/),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toHaveCount(0);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.screenshot({
    path: "../../artifacts/browser-nfl/browser-mixture.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "04 Evaluation & data" }).click();
  await expect(
    page.getByRole("heading", { name: "Post-exposure anchor control" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Held-out proper scores and measured marginal coverage",
    }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".evidence-figures img")
        .first()
        .evaluate((e: HTMLImageElement) => e.complete && e.naturalWidth > 0),
    )
    .toBe(true);
  await page
    .getByText("Controlled ablations and mixture heads", { exact: true })
    .click();
  await expect(
    page.getByText("Graph · shared mixture", { exact: true }),
  ).toBeVisible();
  await page
    .getByText("Sensitivity, numerical precision and distinct play groups", {
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("table", { name: /Frozen-model sensitivities/ }),
  ).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.screenshot({
    path: "../../artifacts/browser-nfl/browser-evaluation.png",
    fullPage: true,
  });
  const selected = (await (await page.request.get("/api/cases")).json())
    .cases[0];
  await page.getByRole("button", { name: "03 Held-out cases" }).click();
  await page.getByRole("button", { name: "Open replay" }).first().click();
  await expect(
    page.getByRole("combobox", { name: "Held-out play", exact: true }),
  ).toHaveValue(`${selected.game_id}/${selected.play_id}`);
  await expect(
    page.getByRole("combobox", { name: "Forecast model", exact: true }),
  ).toHaveValue(selected.model_family);
  await expect(page.getByText(/Exploratory replay/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Reveal actual movement" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("columnheader", { name: "Actual Δx / Δy" }),
  ).toHaveCount(0);
});
