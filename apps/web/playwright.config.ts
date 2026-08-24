import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:5178",
    viewport: { width: 1440, height: 1100 },
    trace: "retain-on-failure",
  },
  reporter: "list",
  webServer: [
    {
      command: "cd ../.. && uv run --locked disguise-lab serve",
      url: "http://127.0.0.1:8008/api/health",
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5178",
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
});
