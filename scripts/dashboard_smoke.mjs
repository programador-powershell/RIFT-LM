import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { _test as api } from "../api/results.mjs";
import { _test as analysisApi } from "../api/analyze.mjs";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);

assert.ok(match, "index.html precisa conter o script principal");
new vm.Script(match[1], { filename: "index-inline.js" });
assert.match(html, /America\/Sao_Paulo/);
assert.match(html, /Comparador RIFT × CASCADE/);
assert.match(html, /Gemini 2\.5 Flash/);
assert.match(html, /Ranking das baterias/);
assert.match(html, /cascade_m0_phase1_test_v030_auto_batteries\.py/);
assert.doesNotMatch(html, /RIFT_GITHUB_TOKEN\s*=/);
assert.doesNotMatch(html, /API_GOOGLE\s*=/);

const rift = { run_id: "run-rift", battery_id: "P1", technology: "RIFT" };
const cascade = { run_id: "run-cascade", battery_id: "P1", technology: "CASCADE" };
assert.equal(api.validateHistory([rift, cascade], "test").length, 2);
assert.equal(api.mergeHistories([rift], [cascade]).length, 2);

const models = analysisApi.validatePayload({
  models: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    technologies: {
      RIFT: {
        battery_id: "P1_RIFT",
        status: "EXPERIMENTAL_FAIL",
        output_cosine: 0.9803,
        output_nrmse: 0.0068,
        quality_gate_pass: false,
        disk_reduction_pct: 87.5,
        ram_reduction_pct: -7.7,
        operation_speedup_x: 0.021,
      },
      CASCADE: {
        battery_id: "P1_CASCADE",
        status: "EXPERIMENTAL_PASS",
        output_cosine: 0.972,
        output_nrmse: 0.011,
        quality_gate_pass: true,
        disk_reduction_pct: 74.2,
        ram_reduction_pct: 73.8,
        operation_speedup_x: 0.389,
      },
    },
  }],
});

const ranking = analysisApi.buildRanking(models);
assert.equal(ranking.length, 2);
assert.equal(ranking[0].position, 1);
assert.ok(ranking.every((entry) => entry.score >= 0 && entry.score <= 100));
assert.ok(ranking.every((entry) => entry.coverage_pct === 100));

const analysis = analysisApi.validateGeminiAnalysis({
  global_summary: "CASCADE apresentou o melhor equilíbrio nesta bateria.",
  analyses: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    recommendation: "CASCADE",
    confidence: 82,
    summary: "A qualidade permaneceu próxima e RAM e latência foram melhores.",
    decisive_metrics: ["RAM", "latência"],
    caveats: ["Kernel nativo não medido"],
  }],
}, models);
assert.equal(analysis.analyses[0].recommendation, "CASCADE");

const singleTechnology = analysisApi.validatePayload({
  models: [{ model_id: "synthetic/rift-b0", technologies: { RIFT: { output_cosine: 0.98 } } }],
});
const forcedInconclusive = analysisApi.validateGeminiAnalysis({
  global_summary: "Teste incompleto.",
  analyses: [{
    model_id: "synthetic/rift-b0",
    recommendation: "RIFT",
    confidence: 99,
    summary: "Somente RIFT foi medido.",
    decisive_metrics: [],
    caveats: [],
  }],
}, singleTechnology);
assert.equal(forcedInconclusive.analyses[0].recommendation, "INCONCLUSIVO");

analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://dashboard.example" },
}));
assert.throws(() => analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://attacker.example" },
})));

const geminiBody = analysisApi.buildGeminiBody("benchmark");
assert.equal(geminiBody.generationConfig.responseFormat.text.mimeType, "application/json");

console.log("dashboard smoke test: PASS");
