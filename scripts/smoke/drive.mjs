// Drives the dashboard through a full rollout -> train -> eval -> cancel
// cycle and captures screenshots. Run via scripts/smoke/run.sh.
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

// Playwright may live in a global install; resolve it from NODE_PATH.
const require = createRequire(path.join(process.env.NODE_PATH ?? process.cwd(), "/"));
const { chromium } = require("playwright");

const API = process.env.SMOKE_API ?? "http://127.0.0.1:8000";
const UI = process.env.SMOKE_UI ?? "http://127.0.0.1:5173";
const OUT = process.env.SMOKE_OUT ?? "artifacts/smoke";
fs.mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const api = async (method, p, body) => {
  const res = await fetch(`${API}${p}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch { /* not json */ }
  return { status: res.status, json, text };
};
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const report = { steps: [] };
const record = (name, detail) => { report.steps.push({ name, ...detail }); log(name, JSON.stringify(detail)); };

async function waitFor(fn, timeoutMs, label) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const v = await fn();
    if (v) return v;
    await sleep(250);
  }
  throw new Error(`timeout waiting for ${label}`);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: "en-US" });
page.on("pageerror", (e) => log("PAGE ERROR", e.message));
let shot = 0;
const snap = async (name) => {
  shot += 1;
  const file = path.join(OUT, `${String(shot).padStart(2, "0")}-${name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  log("screenshot", file);
  return file;
};

try {
  // 1. Empty overview, health.
  const health = await api("GET", "/health");
  const config = await api("GET", "/api/config");
  record("health", { status: health.status, config: config.json });
  await page.goto(UI, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Overview");
  await snap("overview-empty");

  // 2. Queue one rollout from the UI, the rest through the API.
  await page.goto(`${UI}/rollouts`, { waitUntil: "networkidle" });
  await page.fill("input[placeholder='Task prompt']", "Explore this repository and summarise what it does.");
  await page.click("[data-testid=queue-rollout]");
  const prompts = [
    ["Read the README and report the project's purpose.", ["readme", "horizon"]],
    ["List the crates in this workspace.", ["crates"]],
    ["Find every mention of 'horizon' in the tree.", ["horizon"]],
    ["Run ls and describe the top-level layout.", ["Cargo.toml"]],
    ["Write a short findings note and finish.", ["findings"]],
    ["Summarise the Python bridge package.", ["bridge"]],
    ["Describe how jobs are leased.", ["lease"]],
  ];
  const runIds = [];
  for (const [prompt, criteria] of prompts) {
    const r = await api("POST", "/api/runs", { prompt, success_criteria: criteria, horizon: 5 });
    if (r.status !== 202) throw new Error(`create run failed: ${r.text}`);
    runIds.push(r.json.run_id);
  }
  record("queued", { count: runIds.length + 1 });

  // 3. Capture the table and the overview while runs are in flight.
  await waitFor(async () => (await api("GET", "/api/stats")).json.in_flight >= 2, 15000, "running jobs");
  await sleep(1200);
  await snap("rollouts-live");
  await page.goto(UI, { waitUntil: "networkidle" });
  await page.waitForSelector(".uplot", { timeout: 15000 });
  await sleep(800);
  await snap("overview-live");

  // 4. Wait for everything to settle.
  const settled = await waitFor(async () => {
    const s = (await api("GET", "/api/stats")).json;
    return s.queued === 0 && s.in_flight === 0 ? s : null;
  }, 180000, "all rollouts terminal");
  record("rollouts-settled", { completed: settled.rollouts_completed, failed: settled.rollouts_failed, p95_policy_ms: settled.policy_latency.p95_ms, tokens_per_s: settled.tokens_per_s });
  await sleep(1500);
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector(".uplot");
  await sleep(500);
  await snap("overview-settled");

  // 5. Run detail: conversation, reward tab with rescoring, events tab.
  const runs = (await api("GET", "/api/runs?status=completed")).json;
  const detailRun = runs[0];
  await page.goto(`${UI}/rollouts/${detailRun.id}`, { waitUntil: "networkidle" });
  await page.waitForSelector(".turn-card");
  await snap("run-detail-conversation");
  await page.click("button[role=tab]:has-text('Reward')");
  await page.waitForSelector(".data-table");
  await page.click(".panel-head .rt-SelectTrigger");
  await page.click("[role=option]:has-text('coding-v1')");
  await page.click("[data-testid=rescore]");
  await page.waitForSelector("text=coding-v1 · rubric", { timeout: 10000 });
  await snap("run-detail-reward");
  await page.click("button[role=tab]:has-text('Events')");
  await page.waitForSelector(".tl-row");
  await snap("run-detail-events");
  record("rescored", { run: detailRun.id });

  // 6. Train from the training page.
  await page.goto(`${UI}/training`, { waitUntil: "networkidle" });
  const rows = page.locator("[data-testid=sample-row]");
  await rows.first().waitFor();
  const n = Math.min(await rows.count(), 4);
  for (let i = 0; i < n; i++) await rows.nth(i).click();
  await page.click("[data-testid=start-training]");
  await page.waitForURL(/\/training\/trun-/, { timeout: 10000 });
  const trainingId = page.url().split("/training/")[1];
  await waitFor(async () => (await api("GET", `/api/training-runs/${trainingId}`)).json.metrics.length >= 3, 20000, "training metrics");
  await snap("training-live");
  const trained = await waitFor(async () => {
    const r = (await api("GET", `/api/training-runs/${trainingId}`)).json;
    return r.status === "completed" || r.status === "failed" ? r : null;
  }, 60000, "training terminal");
  record("training", { id: trainingId, status: trained.status, steps: trained.metrics.length, adapter: trained.adapter_out });
  await sleep(1000);
  await snap("training-completed");
  await page.goto(`${UI}/training`, { waitUntil: "networkidle" });
  await page.waitForSelector(".data-table");
  await snap("training-list");

  // 7. Eval the adapter from the Adapters page.
  await page.goto(`${UI}/adapters`, { waitUntil: "networkidle" });
  await page.click(`[data-testid=eval-${trained.adapter_out}]`);
  const adapter = await waitFor(async () => {
    const a = (await api("GET", `/api/adapters/${trained.adapter_out}`)).json;
    return a.eval_reports.length > 0 ? a : null;
  }, 120000, "eval report");
  record("eval", { adapter: adapter.adapter.id, eval_score: adapter.adapter.eval_score, reports: adapter.eval_reports.length });
  await sleep(1200);
  await snap("adapters-evaluated");

  // 8. Cancel a live rollout from its detail page.
  const slow = await api("POST", "/api/runs", { prompt: "This one gets cancelled mid-flight.", horizon: 8 });
  const slowId = slow.json.run_id;
  await page.goto(`${UI}/rollouts/${slowId}`, { waitUntil: "networkidle" });
  await waitFor(async () => (await api("GET", `/api/runs/${slowId}`)).json.manifest.status === "running", 15000, "slow run running");
  await sleep(900);
  await page.click("[data-testid=cancel-run]");
  const cancelled = await waitFor(async () => {
    const r = (await api("GET", `/api/runs/${slowId}`)).json;
    return r.manifest.status === "cancelled" ? r : null;
  }, 15000, "cancelled");
  await sleep(800);
  await snap("run-cancelled");
  record("cancel", { run: slowId, status: cancelled.manifest.status, steps: cancelled.manifest.step_count });

  // 9. Fleet, jobs, events, rollouts table, final overview.
  await page.goto(`${UI}/fleet`, { waitUntil: "networkidle" });
  await page.waitForSelector(".data-table");
  await snap("fleet");
  await page.goto(`${UI}/jobs`, { waitUntil: "networkidle" });
  await page.waitForSelector(".data-table");
  await snap("jobs");
  await page.goto(`${UI}/events`, { waitUntil: "networkidle" });
  await page.waitForSelector(".tl-row");
  await snap("events");
  await page.goto(`${UI}/rollouts`, { waitUntil: "networkidle" });
  await page.waitForSelector(".data-table");
  await snap("rollouts-table");
  await page.goto(UI, { waitUntil: "networkidle" });
  await page.waitForSelector(".uplot");
  await sleep(600);
  await snap("overview-final");

  const events = (await api("GET", "/api/events?since=0&limit=1000")).json;
  const kinds = {};
  for (const e of events) kinds[e.kind] = (kinds[e.kind] ?? 0) + 1;
  const jobs = (await api("GET", "/api/jobs?limit=100")).json;
  const fleet = (await api("GET", "/api/fleet")).json;
  record("summary", { events: events.length, kinds, jobs: jobs.reduce((m, j) => ({ ...m, [j.status]: (m[j.status] ?? 0) + 1 }), {}), fleet: fleet.map((n) => [n.id, n.status]) });
  report.ok = true;
} catch (err) {
  report.ok = false;
  report.error = String(err?.stack ?? err);
  log("FAILED", report.error);
  try { await snap("failure"); } catch { /* ignore */ }
} finally {
  fs.writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));
  await browser.close();
}
process.exit(report.ok ? 0 : 1);
