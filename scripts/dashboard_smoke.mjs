import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { _test as api } from "../api/results.mjs";
import { _test as analysisApi } from "../api/analyze.mjs";
import modelSearch, { _test as modelSearchApi } from "../api/models.mjs";
import testLauncher, { _test as launcherApi } from "../api/test.mjs";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);

assert.ok(match, "index.html precisa conter o script principal");
new vm.Script(match[1], { filename: "index-inline.js" });
assert.match(html, /America\/Sao_Paulo/);
assert.match(html, /Comparador RIFT × CASCADE × AETHER × SPECTRA × WINNER/);
assert.match(html, /Gemini 2\.5 Flash/);
assert.match(html, /Ranking das baterias/);
assert.match(html, /curl -fsSL/);
assert.match(html, /Fila serial de baterias/);
assert.match(html, /class="launcherDisclosure"/);
assert.match(html, /setInterval\(\(\)=>\{if\(state\.dataSource==="published"/);
assert.match(html, /Copiar lista serial/);
assert.match(html, /subprocess\.Popen/);
assert.match(html, /nvidia-smi/);
assert.match(html, /wait_for_resource_release/);
assert.match(html, /from google\.colab import userdata/);
assert.match(html, /env=CHILD_ENV/);
assert.match(html, /RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres/);
assert.match(html, /COLAB_BLOCK_REASONS/);
assert.match(html, /Teste não gerado: modelo incompatível/);
assert.match(html, /Garantindo dependências \(sentencepiece, tiktoken\)/);
assert.match(html, /deepseek-v4/);
assert.match(html, /nvfp4/);
assert.doesNotMatch(html, /RIFT_GITHUB_TOKEN\s*=/);
assert.doesNotMatch(html, /API_GOOGLE\s*=/);

const rift = { run_id: "run-rift", model_id: "Qwen/Qwen2.5-0.5B", battery_id: "P1", technology: "RIFT" };
const cascade = { run_id: "run-cascade", model_id: "Qwen/Qwen2.5-0.5B", battery_id: "P1", technology: "CASCADE" };
const aether = { run_id: "run-aether", model_id: "Qwen/Qwen2.5-0.5B", battery_id: "P1", technology: "AETHER" };
const spectra = { run_id: "run-spectra", model_id: "Qwen/Qwen2.5-0.5B", battery_id: "P1", technology: "SPECTRA" };
const winner = { run_id: "run-winner", model_id: "Qwen/Qwen2.5-0.5B", battery_id: "P1", technology: "WINNER" };
assert.equal(api.validateHistory([rift, cascade, aether, spectra, winner], "test").length, 5);
assert.equal(api.mergeHistories([rift], [cascade, aether, spectra, winner]).length, 5);
const replaced = api.mergeHistories(
  [{ ...rift, run_id: "old", battery_id: "OLD_PRIMARY" }, { ...rift, run_id: "old", battery_id: "OLD_B0" }],
  [{ ...rift, run_id: "optimized", battery_id: "NEW_PRIMARY" }, { ...rift, run_id: "optimized", battery_id: "NEW_B0" }],
);
assert.deepEqual(replaced.map((record) => record.run_id), ["optimized", "optimized"]);

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
      WINNER: {
        battery_id: "P1_WINNER",
        status: "EXPERIMENTAL_PASS",
        output_cosine: 0.994,
        output_nrmse: 0.003,
        quality_gate_pass: true,
        disk_reduction_pct: 85,
        ram_reduction_pct: 67,
        operation_speedup_x: 0.467,
      },
    },
  }],
});

const ranking = analysisApi.buildRanking(models);
assert.equal(ranking.length, 5);
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

assert.equal(modelSearchApi.normalizeSearch("  qwen  "), "qwen");
assert.throws(() => modelSearchApi.normalizeSearch("<script>"));
const modelSearchResults = modelSearchApi.normalizeModelResults([
  { id: "example/vision", downloads: 9999, pipeline_tag: "image-classification" },
  { id: "Qwen/Qwen2.5-0.5B", downloads: 500, pipeline_tag: "text-generation", likes: 20 },
  { id: "trl-internal-testing/tiny-Qwen", downloads: 500000, pipeline_tag: "text-generation" },
  { id: "private/model", downloads: 100000, private: true },
]);
assert.equal(modelSearchResults.length, 1);
assert.equal(modelSearchResults[0].id, "Qwen/Qwen2.5-0.5B");
assert.equal((await modelSearch.fetch(new Request("https://dashboard.example/api/models?q=x"))).status, 400);
assert.equal((await modelSearch.fetch(new Request("https://dashboard.example/api/models?q=qwen", { method: "POST" }))).status, 405);

const kimi = launcherApi.normalizeModel("Kimi-K3");
assert.equal(kimi.modelId, "moonshotai/Kimi-K3");
assert.equal(kimi.trustRemoteCode, true);
assert.equal(kimi.compatibility.colabSupported, false);
assert.equal(kimi.compatibility.transformersVersion, "4.56.2");
assert.equal(kimi.compatibility.minimumPackedWeightBytes, 1_400_000_000_000);
const canonicalKimi = launcherApi.normalizeModel("moonshotai/Kimi-K3");
assert.equal(canonicalKimi.compatibility.colabSupported, false);
const kimiGguf = launcherApi.normalizeModel("community/Kimi-K3-GGUF");
assert.equal(kimiGguf.modelId, "community/Kimi-K3-GGUF");
assert.equal(kimiGguf.compatibility.colabSupported, false);
const qwenGguf = launcherApi.normalizeModel("Qwen/Qwen3-8B-GGUF");
assert.equal(qwenGguf.compatibility.colabSupported, false);
assert.match(qwenGguf.compatibility.reason, /checkpoint Transformers original/);
const deepseekV4 = launcherApi.normalizeModel("deepseek-ai/DeepSeek-V4-Flash-0731");
assert.equal(deepseekV4.compatibility.colabSupported, false);
assert.equal(deepseekV4.compatibility.minimumPackedWeightBytes, 142_000_000_000);
const qwenNvfp4 = launcherApi.normalizeModel("example/Qwen3.6-27B-Text-NVFP4-MTP");
assert.equal(qwenNvfp4.compatibility.colabSupported, false);
assert.match(qwenNvfp4.compatibility.reason, /NVFP4\/MTP/);
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
assert.match(launcher, /BLOQUEADO: este modelo não é compatível/);
assert.match(launcher, /transformers==" \+ required_transformers/);
assert.match(launcher, /minimumPackedWeightBytes/);
assert.match(launcher, /sentencepiece>=0\.2\.0/);
assert.match(launcher, /tiktoken>=0\.7\.0/);
assert.match(launcher, /RIFT_INGEST_TOKEN não chegou ao subprocesso/);
assert.match(
  launcher,
  /enforce_compatibility\(\)\nenforce_publish_settings\(\)\nensure_tokenizer_dependencies\(\)\nprint\("\[LAUNCHER\] Baixando bateria versionada:/,
);
assert.ok(
  launcher.indexOf("enforce_compatibility()") < launcher.indexOf("urlopen(request"),
  "a proteção de recursos precisa executar antes de qualquer download",
);
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
const winnerLauncherResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=winner&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(winnerLauncherResponse.status, 200);
const winnerLauncher = await winnerLauncherResponse.text();
assert.match(winnerLauncher, /winner_m0_phase1_test_v080_auto_batteries\.py/);
assert.match(winnerLauncher, /--mode.*phase1/s);

const vercelConfig = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
assert.ok(vercelConfig.rewrites.some((rewrite) => rewrite.source === "/aether/:model*"));
assert.ok(vercelConfig.rewrites.some((rewrite) => rewrite.source === "/spectra/:model*"));
assert.ok(vercelConfig.rewrites.some((rewrite) => rewrite.source === "/winner/:model*"));
const aetherScript = await readFile(
  new URL("../aether_m0_phase1_test_v100_auto_batteries.py", import.meta.url),
  "utf8",
);
assert.match(aetherScript, /technology": "AETHER"/);
assert.match(aetherScript, /P1_AETHER_HQR_PLUS_TADDS_DYNAMIC/);
assert.match(aetherScript, /message = "Configure " \+ " e "\.join\(missing\)/);
assert.match(aetherScript, /ensure_import\("sentencepiece"\)/);
assert.match(aetherScript, /ensure_import\("tiktoken"\)/);
const spectraScript = await readFile(new URL("../SPECTRA_Colab_Test_M0.py", import.meta.url), "utf8");
assert.match(spectraScript, /technology": "SPECTRA"/);
assert.match(spectraScript, /ensure_import\("sentencepiece"\)/);
assert.match(spectraScript, /ensure_import\("tiktoken"\)/);
assert.match(spectraScript, /P1_SPECTRA_HQR_PLUS_TADDS_DYNAMIC/);
assert.match(spectraScript, /P1_SPECTRA_DRIFT_CONTRACT_REF/);
assert.match(spectraScript, /message = "Configure " \+ " e "\.join\(missing\)/);
const cascadeScript = await readFile(
  new URL("../cascade_m0_phase1_test_v030_auto_batteries.py", import.meta.url),
  "utf8",
);
assert.match(cascadeScript, /message = "Configure " \+ " e "\.join\(missing\)/);
const winnerScript = await readFile(
  new URL("../winner_m0_phase1_test_v080_auto_batteries.py", import.meta.url),
  "utf8",
);
assert.match(winnerScript, /"technology": "WINNER"/);
assert.match(winnerScript, /P1_WINNER_F0_PLUS_LS/);
assert.match(winnerScript, /--self-test/);
const winnerCmake = await readFile(new URL("../winner_cpp/CMakeLists.txt", import.meta.url), "utf8");
assert.match(winnerCmake, /winner_self_test/);
assert.doesNotMatch(winnerCmake, /-march=native.*FORCE/);

console.log("dashboard smoke test: PASS");
