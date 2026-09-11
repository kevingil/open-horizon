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
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
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
  // 1. Empty dashboard, health.
  const health = await api("GET", "/health");
  const config = await api("GET", "/api/config");
  record("health", { status: health.status, config: config.json });
  await page.goto(UI, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Launch rollout");
  await snap("dashboard-empty");

  // 2. Queue one rollout from the UI, the rest through the API.
  await page.fill(".launch-input", "Explore this repository and summarise what it does.");
  await page.click("text=Queue rollout");
  const prompts = [
    ["Read the README and report the project's purpose.", ["readme", "horizon"]],
    ["List the crates in this workspace.", ["crates"]],
    ["Find every mention of 'horizon' in the tree.", ["horizon"]],
    ["Run ls and describe the top-level layout.", ["Cargo.toml"]],
    ["Write a short findings note and finish.", ["findings"]],
  ];
  const runIds = [];
  for (const [prompt, criteria] of prompts) {
    const r = await api("POST", "/api/runs", { prompt, success_criteria: criteria, horizon: 5 });
    if (r.status !== 202) throw new Error(`create run failed: ${r.text}`);
    runIds.push(r.json.run_id);
  }
  record("queued", { count: runIds.length + 1 });

  // 3. Capture the dashboard while runs are in flight.
  await waitFor(async () => (await api("GET", "/api/dashboard")).json.jobs.running >= 2, 15000, "running jobs");
  await sleep(1500);
  await page.waitForSelector(".run-progress", { timeout: 15000 });
  await snap("dashboard-live");

  // 4. Wait for everything to settle.
  const settled = await waitFor(async () => {
    const d = (await api("GET", "/api/dashboard")).json;
    return d.jobs.queued === 0 && d.jobs.running === 0 ? d : null;
  }, 120000, "all rollouts terminal");
  record("rollouts-settled", {
    completed: settled.jobs.completed, failed: settled.jobs.failed,
    rewards: settled.runs.map((r) => [r.id, r.status, r.terminal_reward, r.tokens, Number(r.cost_usd.toFixed(4))]),
  });
  await sleep(500);
  await snap("dashboard-settled");

  // 5. Run detail with trajectory + reward signals; rescore with coding-v1.
  const detailRun = settled.runs.find((r) => r.status === "completed");
  await page.goto(`${UI}/runs/${detailRun.id}`, { waitUntil: "networkidle" });
  await page.waitForSelector(".signal-table");
  await snap("run-detail");
  await page.selectOption(".rescore select", "coding-v1");
  await page.click(".rescore button");
  await page.waitForSelector("text=Rubric: coding-v1", { timeout: 10000 });
  await snap("run-detail-rescored");
  record("rescored", { run: detailRun.id });

  // 6. Train from the dashboard selection.
  await page.goto(UI, { waitUntil: "networkidle" });
  const boxes = page.locator(".run-select input:not([disabled])");
  const n = Math.min(await boxes.count(), 4);
  for (let i = 0; i < n; i++) await boxes.nth(i).check();
  await page.waitForSelector("text=Train from selection");
  await snap("dashboard-selection");
  await page.click("text=Train from selection");
  await page.waitForURL(/\/training\//, { timeout: 10000 });
  const trainingId = page.url().split("/training/")[1];
  await waitFor(async () => (await api("GET", `/api/training-runs/${trainingId}`)).json.metrics.length >= 3, 20000, "training metrics");
  await snap("training-live");
  const trained = await waitFor(async () => {
    const r = (await api("GET", `/api/training-runs/${trainingId}`)).json;
    return r.status === "completed" || r.status === "failed" ? r : null;
  }, 60000, "training terminal");
  record("training", { id: trainingId, status: trained.status, steps: trained.metrics.length, adapter: trained.adapter_out });
  await page.waitForSelector(".badge-completed", { timeout: 10000 });
  await snap("training-completed");

  // 7. Eval the adapter from the Adapters page.
  await page.goto(`${UI}/adapters`, { waitUntil: "networkidle" });
  const card = page.locator(".adapter-card", { hasText: trained.adapter_out });
  await card.waitFor();
  await card.locator("button").click();
  await card.locator("text=/eval -?\\d/").waitFor({ timeout: 90000 });
  const adapter = (await api("GET", `/api/adapters/${trained.adapter_out}`)).json;
  record("eval", { adapter: adapter.adapter.id, eval_score: adapter.adapter.eval_score, reports: adapter.eval_reports.length });
  await snap("adapters-evaluated");

  // 8. Cancel a live rollout from its detail page.
  const slow = await api("POST", "/api/runs", { prompt: "This one gets cancelled mid-flight.", horizon: 8 });
  const slowId = slow.json.run_id;
  await page.goto(`${UI}/runs/${slowId}`, { waitUntil: "networkidle" });
  await waitFor(async () => (await api("GET", `/api/runs/${slowId}`)).json.manifest.status === "running", 15000, "slow run running");
  await sleep(700);
  await page.click("text=Cancel rollout");
  const cancelled = await waitFor(async () => {
    const r = (await api("GET", `/api/runs/${slowId}`)).json;
    return r.manifest.status === "cancelled" ? r : null;
  }, 15000, "cancelled");
  await page.waitForSelector(".badge-cancelled", { timeout: 10000 });
  await snap("run-cancelled");
  record("cancel", { run: slowId, status: cancelled.manifest.status, steps: cancelled.manifest.step_count });

  // 9. Training list + final dashboard with charts.
  await page.goto(`${UI}/training`, { waitUntil: "networkidle" });
  await page.waitForSelector(".run-card");
  await snap("training-list");
  await page.goto(UI, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Reward over time");
  await snap("dashboard-final");

  const events = (await api("GET", "/api/events?since=0&limit=1000")).json;
  const kinds = {};
  for (const e of events) kinds[e.kind] = (kinds[e.kind] ?? 0) + 1;
  const jobs = (await api("GET", "/api/jobs?limit=100")).json;
  record("summary", { events: events.length, kinds, jobs: jobs.reduce((m, j) => ({ ...m, [j.status]: (m[j.status] ?? 0) + 1 }), {}) });
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
