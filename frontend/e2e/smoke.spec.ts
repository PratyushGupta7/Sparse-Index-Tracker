import { test, expect, type Page } from "@playwright/test";

const ROUTES = ["/", "/research", "/backtest", "/api", "/invest"] as const;

async function expectNoConsoleErrors(page: Page, fn: () => Promise<void>) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  await fn();
  // Filter out third-party noise (clearbit/companieslogo 404s, etc.)
  const filtered = errors.filter(
    (e) =>
      !/clearbit|companieslogo|favicon|hydration text content|_vercel\/insights|_vercel\/speed-insights|MIME type|Failed to load resource/i.test(
        e
      )
  );
  expect(filtered, filtered.join("\n")).toEqual([]);
}

for (const route of ROUTES) {
  test(`renders ${route} without console errors`, async ({ page }) => {
    await expectNoConsoleErrors(page, async () => {
      const r = await page.goto(route);
      expect(r?.status()).toBeLessThan(400);
      await page.waitForLoadState("networkidle");
    });
  });
}

test("/ landing page renders the hero copy", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Replicate the S&P 500 with/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /Try it live/i })).toBeVisible();
});

test("/backtest renders equity chart + stat cards", async ({ page }) => {
  await page.goto("/backtest");
  await expect(page.getByRole("heading", { name: /Walk-forward 2018/ })).toBeVisible();
  await expect(page.getByText("Sharpe", { exact: true }).first()).toBeVisible();
});

test("/invest form validates and submits against a mocked API", async ({ page }) => {
  await page.route("**/api/proxy/**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        mode: "live_retrained",
        index: "sp500",
        benchmark: "SPY",
        capital: 100000,
        total_invested: 99500,
        residual_cash: 500,
        utilization_pct: "99.5%",
        n_stocks_bought: 12,
        price_date: "2026-05-12",
        total_time_seconds: 1.2,
        model: {
          train_period: "2026-01-01 - 2026-05-12",
          r2_train: 0.962,
          te_train_pct: 3.4,
          active_stocks: 12,
          universe_size: 500,
          converged: true,
          iterations: 220,
          solve_time_ms: 142.0,
          solver_iterations: 220,
        },
        allocations: [
          {
            ticker: "AAPL",
            shares: 10,
            price: 178.4,
            weight: 0.07,
            allocated: 7000,
            actual_cost: 1784,
          },
        ],
        warnings: null,
      }),
    });
  });

  await page.goto("/invest");
  await page.getByLabel(/Capital/).fill("100000");
  await page.getByRole("button", { name: /Get allocations/ }).click();
  await expect(page.getByRole("cell", { name: "AAPL" })).toBeVisible({ timeout: 8000 });
  await expect(page.getByText("Invested", { exact: true })).toBeVisible();
});
