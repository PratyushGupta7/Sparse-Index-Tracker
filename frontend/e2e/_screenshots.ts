import { chromium } from "@playwright/test";
import path from "node:path";
import fs from "node:fs/promises";

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000";
const OUT = path.resolve(__dirname, "../../docs/images/frontend");

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
  const page = await ctx.newPage();

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "landing.png"), fullPage: false });

  await page.goto(`${BASE}/backtest`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "backtest.png"), fullPage: false });

  await page.goto(`${BASE}/research`, { waitUntil: "networkidle" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, "research.png"), fullPage: false });

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
        n_stocks_bought: 8,
        price_date: "2026-05-12",
        total_time_seconds: 1.2,
        model: {
          train_period: "2026-01-01 - 2026-05-12",
          r2_train: 0.962,
          te_train_pct: 3.4,
          active_stocks: 8,
          universe_size: 500,
          converged: true,
          iterations: 220,
          solve_time_ms: 142,
          solver_iterations: 220,
        },
        allocations: [
          { ticker: "AAPL", shares: 12, price: 178.4, weight: 0.18, allocated: 18000, actual_cost: 2140.8 },
          { ticker: "MSFT", shares: 5, price: 412.6, weight: 0.16, allocated: 16000, actual_cost: 2063.0 },
          { ticker: "NVDA", shares: 6, price: 905.1, weight: 0.14, allocated: 14000, actual_cost: 5430.6 },
          { ticker: "AMZN", shares: 18, price: 178.0, weight: 0.12, allocated: 12000, actual_cost: 3204.0 },
          { ticker: "GOOG", shares: 22, price: 162.0, weight: 0.10, allocated: 10000, actual_cost: 3564.0 },
          { ticker: "META", shares: 12, price: 502.0, weight: 0.10, allocated: 10000, actual_cost: 6024.0 },
          { ticker: "TSLA", shares: 24, price: 220.0, weight: 0.10, allocated: 10000, actual_cost: 5280.0 },
          { ticker: "JPM", shares: 50, price: 198.0, weight: 0.10, allocated: 10000, actual_cost: 9900.0 },
        ],
        warnings: null,
      }),
    });
  });
  await page.goto(`${BASE}/invest`, { waitUntil: "networkidle" });
  await page.getByLabel(/Capital/).fill("100000");
  await page.getByRole("button", { name: /Get allocations/ }).click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(OUT, "invest.png"), fullPage: false });

  await browser.close();
  console.log("screenshots → docs/images/frontend/");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
