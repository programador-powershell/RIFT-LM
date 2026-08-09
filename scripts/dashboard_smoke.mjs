import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { _test as api } from "../api/results.mjs";
import { _test as analysisApi } from "../api/analyze.mjs";
import testLauncher, { _test as launcherApi } from "../api/test.mjs";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);

assert.ok(match, "index.html precisa conter o script principal");
new vm.Script(match[1], { filename: "index-inline.js" });
assert.match(html, /America\/Sao_Paulo/);
assert.match(html, /Comparador RIFT × CASCADE × AETHER × SPECTRA/);
assert.match(html, /Gemini 2\.5 Flash/);
assert.match(html, /Ranking das baterias/);
assert.match(html, /curl -fsSL/);
assert.doesNotMatch(html, /RIFT_GITHUB_TOKEN\s*=/);
assert.doesNotMatch(html, /API_GOOGLE\s*=/);

const rift = { run_id: "run-rift", battery_id: "P1", technology: "RIFT" };
const cascade = { run_id: "run-cascade", battery_id: "P1", technology: "CASCADE" };
const aether = { run_id: "run-aether", battery_id: "P1", technology: "AETHER" };
const spectra = { run_id: "run-spectra", battery_id: "P1", technology: "SPECTRA" };
assert.equal(api.validateHistory([rift, cascade, aether, spectra], "test").length, 4);
assert.equal(api.mergeHistories([rift], [cascade, aether, spectra]).length, 4);

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
      AETHER: {
        battery_id: "P1_AETHER",
        status: "EXPERIMENTAL_FAIL",
        output_cosine: 0.987,
        output_nrmse: 0.0045,
        quality_gate_pass: false,
        disk_reduction_pct: 92.5,
        ram_reduction_pct: 60,
        operation_speedup_x: 0.583,
      },
      SPECTRA: {
        battery_id: "P1_SPECTRA",
        status: "EXPERIMENTAL_PASS",
        output_cosine: 0.991,
        output_nrmse: 0.0038,
        quality_gate_pass: true,
        disk_reduction_pct: 90.8,
        ram_reduction_pct: 63.1,
        operation_speedup_x: 0.7,
      },
    },
  }],
});

const ranking = analysisApi.buildRanking(models);
assert.equal(ranking.length, 4);
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

const unavailableTechnology = analysisApi.validateGeminiAnalysis({
  global_summary: "Resposta inconsistente.",
  analyses: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    recommendation: "AETHER",
    confidence: 90,
    summary: "AETHER não foi medido.",
    decisive_metrics: [],
    caveats: [],
  }],
}, analysisApi.validatePayload({
  models: [{
    model_id: "Qwen/Qwen2.5-0.5B",
    technologies: { RIFT: { output_cosine: 0.98 }, CASCADE: { output_cosine: 0.97 } },
  }],
}));
assert.equal(unavailableTechnology.analyses[0].recommendation, "INCONCLUSIVO");

analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://dashboard.example" },
}));
assert.throws(() => analysisApi.enforceSameOrigin(new Request("https://dashboard.example/api/analyze", {
  headers: { Origin: "https://attacker.example" },
})));

const geminiBody = analysisApi.buildGeminiBody("benchmark");
assert.equal(geminiBody.generationConfig.responseFormat.text.mimeType, "application/json");

const kimi = launcherApi.normalizeModel("Kimi-K3");
assert.equal(kimi.modelId, "moonshotai/Kimi-K3");
assert.equal(kimi.trustRemoteCode, true);
const launcherResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=aether&model=Kimi-K3",
));
assert.equal(launcherResponse.status, 200);
assert.match(launcherResponse.headers.get("content-type"), /text\/x-python/);
const launcher = await launcherResponse.text();
assert.match(launcher, /aether_m0_phase1_test_v100_auto_batteries\.py/);
assert.match(launcher, /moonshotai\/Kimi-K3/);
assert.match(launcher, /--trust-remote-code/);
assert.match(launcher, /https:\/\/rift-lm\.vercel\.app\/api\/results/);
const untrustedOriginLauncher = launcherApi.buildLauncher({
  technology: "aether",
  model: launcherApi.normalizeModel("Kimi-K3"),
  origin: "https://proxy.example",
});
assert.match(untrustedOriginLauncher, /https:\/\/rift-lm\.vercel\.app\/api\/results/);
assert.doesNotMatch(untrustedOriginLauncher, /proxy\.example\/api\/results/);
const invalidLauncher = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=unknown&model=Kimi-K3",
));
assert.equal(invalidLauncher.status, 400);

const spectraLauncherResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=spectra&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(spectraLauncherResponse.status, 200);
const spectraLauncher = await spectraLauncherResponse.text();
assert.match(spectraLauncher, /SPECTRA_Colab_Test_M0\.py/);
assert.match(spectraLauncher, /--mode.*phase1/s);

const vercelConfig = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
assert.ok(vercelConfig.rewrites.some((rewrite) => rewrite.source === "/aether/:model*"));
assert.ok(vercelConfig.rewrites.some((rewrite) => rewrite.source === "/spectra/:model*"));
const aetherScript = await readFile(
  new URL("../aether_m0_phase1_test_v100_auto_batteries.py", import.meta.url),
  "utf8",
);
assert.match(aetherScript, /technology": "AETHER"/);
assert.match(aetherScript, /P1_AETHER_HQR_PLUS_TADDS_DYNAMIC/);
const spectraScript = await readFile(new URL("../SPECTRA_Colab_Test_M0.py", import.meta.url), "utf8");
assert.match(spectraScript, /technology": "SPECTRA"/);
assert.match(spectraScript, /P1_SPECTRA_HQR_PLUS_TADDS_DYNAMIC/);
assert.match(spectraScript, /P1_SPECTRA_DRIFT_CONTRACT_REF/);

console.log("dashboard smoke test: PASS");
