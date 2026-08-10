import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { _test as resultsApi } from "../api/results.mjs";
import { _test as analysisApi } from "../api/analyze.mjs";
import modelsApi, { _test as modelSearchApi } from "../api/models.mjs";
import testLauncher, { _test as launcherApi } from "../api/test.mjs";

const legacyHtml = await readFile(new URL("../index.html", import.meta.url), "utf8");
const dashboardHtml = await readFile(new URL("../dashboard.html", import.meta.url), "utf8");

function assertInlineScriptParses(html, filename) {
  const match = html.match(/<script>([\s\S]*?)<\/script>/);
  assert.ok(match, `${filename} precisa conter script inline`);
  new vm.Script(match[1], { filename });
}

assertInlineScriptParses(legacyHtml, "legacy-index-inline.js");
assertInlineScriptParses(dashboardHtml, "dashboard-v2-inline.js");
assert.match(dashboardHtml, /comparison_group_id/);
assert.match(dashboardHtml, /legacy-unverified/);
assert.match(dashboardHtml, /If-None-Match/);
assert.match(dashboardHtml, /30000/);
assert.doesNotMatch(dashboardHtml, /refresh=\$\{Date\.now\(\)\}/);
assert.match(dashboardHtml, /Quality gate é eliminatório/);
assert.match(dashboardHtml, /Painel legado/);

const base = {
  timestamp_utc: "2026-08-09T10:00:00Z",
  model_id: "Qwen/Qwen2.5-0.5B",
  status: "PASS",
};
const riftOld = {
  ...base,
  run_id: "run-rift-old",
  battery_id: "P1_Q4_LINEAR_BASE_PLUS_REF_4BIT",
  technology: "RIFT",
};
const riftNew = {
  ...base,
  timestamp_utc: "2026-08-09T11:00:00Z",
  run_id: "run-rift-new",
  battery_id: "P1_Q4_LINEAR_BASE_PLUS_REF_4BIT",
  technology: "RIFT",
  schema_version: 1,
  comparison_group_id: "cmp-0123456789abcdef01234567",
  comparison_context: {
    protocol: "LINEAR_REFERENCE_V2",
    model_id: "Qwen/Qwen2.5-0.5B",
    target_layer_request: "auto",
    gpu: "NVIDIA T4, 15360, driver",
  },
};
const cascade = {
  ...base,
  timestamp_utc: "2026-08-09T11:01:00Z",
  run_id: "run-cascade",
  battery_id: "P1_CASCADE_GATED_F0_PLUS_F1",
  technology: "CASCADE",
  schema_version: 1,
  comparison_group_id: "cmp-0123456789abcdef01234567",
};

const validated = resultsApi.validateHistory([riftOld, riftNew, cascade], "test");
assert.equal(validated.length, 3);
assert.equal(validated[0].schema_version, 1);
assert.equal(validated[0].technology, "RIFT");
assert.equal(validated[1].comparison_group_id, "cmp-0123456789abcdef01234567");
assert.throws(() => resultsApi.validateHistory([
  { ...riftOld, technology: "UNKNOWN" },
], "test"));
assert.throws(() => resultsApi.validateHistory([
  { ...riftNew, comparison_group_id: "bad group id!" },
], "test"));

// Histórico agora é append-only por run_id+technology+battery_id.
const merged = resultsApi.mergeHistories([riftOld], [riftNew, cascade]);
assert.equal(merged.length, 3);
assert.deepEqual(merged.map((record) => record.run_id), ["run-rift-old", "run-rift-new", "run-cascade"]);
const idempotent = resultsApi.mergeHistories(merged, [riftNew]);
assert.equal(idempotent.length, 3);
assert.equal(resultsApi.recordKey(riftNew), "run-rift-new\u0000RIFT\u0000P1_Q4_LINEAR_BASE_PLUS_REF_4BIT");

const analysisModels = analysisApi.validatePayload({
  models: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    technologies: {
      RIFT: {
        battery_id: "P1_RIFT",
        status: "PASS",
        output_cosine: 0.995,
        output_nrmse: 0.003,
        quality_gate_pass: true,
        disk_reduction_pct: 80,
        ram_reduction_pct: 60,
        operation_speedup_x: 0.8,
      },
      CASCADE: {
        battery_id: "P1_CASCADE",
        status: "PASS",
        output_cosine: 0.996,
        output_nrmse: 0.002,
        quality_gate_pass: true,
        disk_reduction_pct: 75,
        ram_reduction_pct: 70,
        operation_speedup_x: 0.9,
      },
    },
  }],
});
const ranking = analysisApi.buildRanking(analysisModels);
assert.equal(ranking.length, 2);
assert.equal(ranking[0].position, 1);
assert.ok(ranking.every((entry) => entry.score >= 0 && entry.score <= 100));
const analysis = analysisApi.validateGeminiAnalysis({
  global_summary: "Comparação concluída.",
  analyses: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    recommendation: "CASCADE",
    confidence: 80,
    summary: "Melhor equilíbrio nesta bateria.",
    decisive_metrics: ["RAM"],
    caveats: ["Caminho de referência"],
  }],
}, analysisModels);
assert.equal(analysis.analyses[0].recommendation, "CASCADE");
analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://dashboard.example" },
}));
assert.throws(() => analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://attacker.example" },
})));

assert.equal(modelSearchApi.normalizeSearch("  qwen  "), "qwen");
assert.throws(() => modelSearchApi.normalizeSearch("<script>"));
const normalizedModels = modelSearchApi.normalizeModelResults([
  { id: "Qwen/Qwen2.5-0.5B", downloads: 500, pipeline_tag: "text-generation" },
  { id: "private/model", downloads: 9999, private: true },
]);
assert.equal(normalizedModels.length, 1);
assert.equal((await modelsApi.fetch(new Request("https://dashboard.example/api/models?q=x"))).status, 400);

assert.equal(launcherApi.BENCHMARK_PROTOCOL, "LINEAR_REFERENCE_V2");
const launcherModel = launcherApi.normalizeModel("Qwen/Qwen2.5-0.5B");
const launcher = launcherApi.buildLauncher({
  technology: "cascade",
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  targetLayer: "auto",
  device: "cuda",
  publish: "required",
  ref: "main",
});
assert.match(launcher, /cascade_m0_phase1_test_v030_auto_batteries\.py/);
assert.match(launcher, /LINEAR_REFERENCE_V2/);
assert.match(launcher, /comparison_group_id/);
assert.match(launcher, /RIFT_COMPARISON_GROUP_ID/);
assert.match(launcher, /RIFT_COMPARISON_CONTEXT_JSON/);
assert.match(launcher, /install_result_enricher/);
assert.match(launcher, /REQUEST_LEVEL/);
assert.match(launcher, /Dependências ausentes serão instaladas pela própria bateria/);
assert.doesNotMatch(launcher, /subprocess\.check_call\(\[sys\.executable, "-m", "pip", "install"/);
assert.ok(
  launcher.indexOf("enforce_compatibility()") < launcher.indexOf("urlopen(request"),
  "compatibility guard precisa rodar antes do download",
);
const launcherResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=cascade&model=Qwen%2FQwen2.5-0.5B&device=cuda",
));
assert.equal(launcherResponse.status, 200);
assert.match(launcherResponse.headers.get("content-type"), /text\/x-python/);

const vercelConfig = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
assert.ok(vercelConfig.rewrites.some((r) => r.source === "/" && r.destination === "/dashboard.html"));
assert.ok(vercelConfig.rewrites.some((r) => r.source === "/legacy" && r.destination === "/index.html"));
assert.ok(vercelConfig.rewrites.some((r) => r.source === "/cascade/:model*"));
const dataHeaders = vercelConfig.headers.find((entry) => entry.source === "/data/(.*)");
assert.ok(dataHeaders.headers.some((header) => header.key === "Cache-Control" && /s-maxage=30/.test(header.value)));

const schema = JSON.parse(await readFile(new URL("../data/record-schema-example.json", import.meta.url), "utf8"));
assert.equal(schema.schema_version, 1);
assert.equal(schema.benchmark_protocol, "LINEAR_REFERENCE_V2");
assert.ok(schema.comparison_group_id);
assert.equal(schema.implementation.kind, "REFERENCE|NATIVE|SIMULATED");

console.log("dashboard smoke: OK");
