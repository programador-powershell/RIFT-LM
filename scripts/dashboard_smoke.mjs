import assert from "node:assert/strict";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import resultsHandler, { _test as resultsApi, selectWinnerArchitecture } from "../api/results.mjs";
import { _test as analysisApi } from "../api/analyze.mjs";
import modelsApi, { _test as modelSearchApi } from "../api/models.mjs";
import testLauncher, { _test as launcherApi } from "../api/test.mjs";
import geyserApi, { _test as geyserLauncherApi } from "../api/geyser.mjs";
import runnerHandler, { _test as runnerQueueApi } from "../api/runner.mjs";
import converterHandler, { _test as converterApi } from "../api/converter.mjs";
import { _test as realLauncherApi } from "../api/real-test.mjs";
import { rawBaseUrl, resolveRef, resolveRepo, _test as repoLib } from "../api/_lib/repo.mjs";

// Determinismo da cadeia repo-agnóstica (§14.1): sem NENHUMA env de repo/ref
// configurada, toda resolução precisa cair no fallback legado
// "programador-powershell/RIFT-LM" + ref "main". Limpa as envs no início para
// que o smoke não dependa do ambiente da máquina/CI.
for (const name of [
  "GITHUB_REPO",
  "RIFT_GITHUB_REPOSITORY",
  "RIFT_GITHUB_BRANCH",
  "VERCEL_GIT_REPO_OWNER",
  "VERCEL_GIT_REPO_SLUG",
  "VERCEL_GIT_COMMIT_SHA",
]) {
  delete process.env[name];
}

const LEGACY_REPO_FALLBACK = "programador-powershell/RIFT-LM";
// §20 regra 1 — duplicata eliminada + árvore canônica (declarados no topo:
// usados também pelo bloco do conversor §26, que roda antes do bloco §20).
const BANNED_CONVERTER_DUPLICATE = "cascade-model-converter";
const CANONICAL_REPO_PATH_RE = /^(engines|batteries|core|scripts|data)\//;

const legacyHtml = await readFile(new URL("../index.html", import.meta.url), "utf8");
// §24.1 — painel ÚNICO: dashboard.html foi DELETADO do repositório (as rotas
// /v2 e /legacy deixaram de existir). O assert canônico é a AUSÊNCIA do arquivo.
await assert.rejects(
  () => stat(new URL("../dashboard.html", import.meta.url)),
  { code: "ENOENT" },
  "dashboard.html deveria ter sido DELETADO do repositório (§24.1 — painel único)",
);
// Árvore canônica (docs/C3_CONTRACTS_V1.md §20): baterias M0 em engines/<tech>/,
// launcher GEYSER em engines/geyser/ e baterias multi-motor em batteries/.
const geyserLauncherPy = await readFile(new URL("../engines/geyser/geyser_launcher.py", import.meta.url), "utf8");
const comparePublisherPy = await readFile(new URL("../batteries/compare_generations_publisher.py", import.meta.url), "utf8");
const capabilityPy = await readFile(new URL("../batteries/capability_eval_auto_batteries.py", import.meta.url), "utf8");
const riftPy = await readFile(new URL("../engines/rift/rift_m0_phase1_test_v035_auto_batteries.py", import.meta.url), "utf8");
const aetherPy = await readFile(new URL("../engines/aether/aether_m0_phase1_test_v100_auto_batteries.py", import.meta.url), "utf8");
const spectraPy = await readFile(new URL("../engines/spectra/SPECTRA_Colab_Test_M0.py", import.meta.url), "utf8");
const winnerPy = await readFile(new URL("../engines/winner/winner_m0_phase1_test_v080_auto_batteries.py", import.meta.url), "utf8");
const cascadeC2Py = await readFile(new URL("../engines/cascade/cascade_c2_e2e_auto_batteries.py", import.meta.url), "utf8");
const ggufPy = await readFile(new URL("../batteries/gguf_e2e_auto_batteries.py", import.meta.url), "utf8");
const finalPy = await readFile(new URL("../batteries/final_phase_auto_batteries.py", import.meta.url), "utf8");
const runnerPy = await readFile(new URL("../scripts/real_benchmark_runner.py", import.meta.url), "utf8");
// MicroLM (§22): bateria própria + arquivos VERBATIM do usuário — a leitura
// já é o assert de existência dos 4 arquivos + o script da bateria.
const microlmPy = await readFile(new URL("../engines/microlm/microlm_m0_auto_batteries.py", import.meta.url), "utf8");
const microlmModelPy = await readFile(new URL("../engines/microlm/model.py", import.meta.url), "utf8");
const microlmTestModelPy = await readFile(new URL("../engines/microlm/test_model.py", import.meta.url), "utf8");
const microlmChangesMd = await readFile(new URL("../engines/microlm/CHANGES.md", import.meta.url), "utf8");
const microlmDiagramSvg = await readFile(new URL("../engines/microlm/diagram.svg", import.meta.url), "utf8");

function assertInlineScriptParses(html, filename) {
  const scriptTags = html.match(/<script[\s>]/g) || [];
  assert.equal(scriptTags.length, 1, `${filename} precisa ter exatamente um bloco <script>`);
  const match = html.match(/<script>([\s\S]*?)<\/script>/);
  assert.ok(match, `${filename} precisa conter script inline`);
  new vm.Script(match[1], { filename });
}

assertInlineScriptParses(legacyHtml, "legacy-index-inline.js");

// Extrai uma função nomeada (`function nome(...) { ... }`) do fonte por
// contagem de chaves — usado para exercitar o comportamento do painel único.
function extractFunction(source, name, filename) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${filename} sem definição de function ${name}(`);
  // Pula a lista de parâmetros por contagem de parênteses — parâmetros com
  // destructuring (ex.: {collapsible=false}) têm chaves que NÃO são o corpo.
  let parens = 0;
  let paramsEnd = source.indexOf("(", start);
  for (; paramsEnd < source.length; paramsEnd += 1) {
    if (source[paramsEnd] === "(") parens += 1;
    else if (source[paramsEnd] === ")") {
      parens -= 1;
      if (parens === 0) break;
    }
  }
  const braceStart = source.indexOf("{", paramsEnd);
  let depth = 0;
  for (let index = braceStart; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(start, index + 1);
    }
  }
  throw new Error(`${filename}: chaves desbalanceadas em function ${name}`);
}

// ---------------------------------------------------------------------------
// Identidade visual (docs/C3_CONTRACTS_V1.md §10) — "Observatório LLM" no
// painel único; o nome antigo "Test Observatory" não existe mais.
// ---------------------------------------------------------------------------

assert.ok(legacyHtml.includes("<title>Observatório LLM</title>"), "index.html sem título Observatório LLM");
assert.ok(
  legacyHtml.includes("RIFT · CASCADE · AETHER · SPECTRA · GEYSER · WINNER"),
  "index.html sem subtítulo de linhagem com GEYSER",
);
assert.doesNotMatch(legacyHtml, /Test Observatory/, "index.html ainda contém o nome antigo");
assert.doesNotMatch(legacyHtml, /Todas as 5 tecnologias/, "index.html ainda fala em 5 tecnologias");
assert.ok(legacyHtml.includes("--geyser:#0d9488"), "index.html sem a cor teal do GEYSER (§10)");
assert.ok(legacyHtml.includes("Todas as 6 tecnologias"));
assert.doesNotMatch(legacyHtml, /modelSearchResults/);
assert.doesNotMatch(legacyHtml, /renderModelSearchResults/);

// ---------------------------------------------------------------------------
// Níveis de bateria (§8) — batteryLevel é camada de EXIBIÇÃO com implementação
// ÚNICA no painel (index.html, §24.1); os battery_ids permanecem imutáveis.
// As fixtures fixam o comportamento sobre os ids reais.
// ---------------------------------------------------------------------------

const legacyBatteryLevel = vm.runInNewContext(
  `(${extractFunction(legacyHtml, "batteryLevel", "index.html")})`,
  {},
  { filename: "legacy-batteryLevel.js" },
);
const BATTERY_LEVEL_FIXTURES = [
  ["CAP_X", 4],
  ["CAP_INTELLIGENCE", 4],
  ["CAP_CODING", 4],
  ["CAP_AGENTIC", 4],
  ["C3_X", 3],
  ["C3_RIFT_LINEAR_F0_GATE_F1", 3],
  ["C3_SPECTRA_FULLMODEL_E2E_TOKS", 3],
  // Fases finais (§16, FINAL_PHASE_V1): prefixos C4_/C5_/C6_ → Nível 5.
  ["C4_RIFT_SECOND_FAMILY", 5],
  ["C5_AETHER_REPR_BLOCKS", 5],
  ["C6_CASCADE_COMPILE_EXECUTE", 5],
  ["c6_spectra_compile_execute", 5],
  ["P1_CASCADE_C1_BLOCK_GATED", 2],
  ["P1_CASCADE_C0_PIPELINE", 2],
  ["P1_CASCADE_C2_E2E", 2],
  // Comparação de gerações (§18): /^CMP_/ → Nível 2 (case-insensitive).
  ["CMP_CASCADE_GENERATIONS", 2],
  ["CMP_RIFT_GENERATIONS", 2],
  ["cmp_rift_generations", 2],
  ["cmp_geyser_generations", 2],
  ["P1_Q4_LINEAR_BASE_2BIT", 1],
  ["G1_GEYSER_ZDC_LUT", 1],
  ["B0_GEYSER_PHYSICS_BANDWIDTH", 1],
  ["G5_GEYSER_ELASTIC_KV", 1],
  ["", 1],
];
for (const [batteryId, expectedLevel] of BATTERY_LEVEL_FIXTURES) {
  assert.equal(
    legacyBatteryLevel(batteryId),
    expectedLevel,
    `index.html batteryLevel(${JSON.stringify(batteryId)}) != ${expectedLevel}`,
  );
}
assert.ok(legacyHtml.includes("BATTERY_LEVEL_LABELS"));
assert.ok(legacyHtml.includes("batteryFriendlyName"));

// ---------------------------------------------------------------------------
// §19.5 (7º lote) — batteryFriendlyName no painel único: título visível
// SEMPRE 'N<nível> · <nome amigável PT-BR>' (id cru só em tooltip). A
// implementação é composta (helpers + mapas const) e as fixtures fixam o
// comportamento sobre TODOS os exemplos do contrato.
// ---------------------------------------------------------------------------

function extractConstObject(source, name, filename) {
  const match = source.match(new RegExp(`const ${name}=\\{[\\s\\S]*?\\n\\};`));
  assert.ok(match, `${filename} sem const ${name}`);
  return match[0];
}

// Variante para arrays const de UMA linha (ex.: const BIT_CLASS_ORDER=[...];)
// — extractConstObject só casa objetos multi-linha.
function extractConstArray(source, name, filename) {
  const match = source.match(new RegExp(`const ${name}=\\[[^\\]]*\\];`));
  assert.ok(match, `${filename} sem const ${name}`);
  return match[0];
}

const legacyFriendlyContext = {};
vm.createContext(legacyFriendlyContext);
vm.runInContext(
  [
    extractFunction(legacyHtml, "batteryLevel", "index.html"),
    extractFunction(legacyHtml, "batteryFriendlyHumanize", "index.html"),
    extractFunction(legacyHtml, "batteryFriendlySuffix", "index.html"),
    extractFunction(legacyHtml, "batteryFriendlyName", "index.html"),
    extractConstObject(legacyHtml, "BATTERY_FRIENDLY_NAMES", "index.html"),
    extractConstObject(legacyHtml, "C3_STEP_FRIENDLY_NAMES", "index.html"),
    "this.batteryFriendlyName=batteryFriendlyName;",
  ].join("\n"),
  legacyFriendlyContext,
  { filename: "legacy-batteryFriendlyName.js" },
);
const legacyBatteryFriendlyName = legacyFriendlyContext.batteryFriendlyName;

// Exemplos canônicos do contrato §19.5 (fixtures espelhadas em
// scripts/battery_group_key_fixtures.mjs).
const BATTERY_FRIENDLY_NAME_FIXTURES = [
  ["P1_CASCADE_C2_E2E_TOKS", "N2 · Velocidade ponta a ponta"],
  ["B0_BINARY_IR_FOUNDATION", "N1 · Fundação do formato"],
  ["B0_GGUF_RUNTIME_SETUP", "N1 · Fundação do formato"],
  ["B0_WINNER_CPP_BUILD_SELF_TEST", "N1 · Fundação do formato"],
  ["P1_Q4_LINEAR_BASE_2BIT", "N1 · Compressão base"],
  ["P1_Q4_LINEAR_BASE_PLUS_REF_4BIT", "N1 · Compressão + refinamento"],
  ["P1_CASCADE_GATED_F0_PLUS_F1", "N1 · Compressão + refinamento"],
  ["P1_WINNER_F0_PLUS_LS", "N1 · Compressão + refinamento"],
  ["P1_AETHER_PIO_POLICY_SIM", "N1 · Política simulada"],
  ["P1_SPECTRA_PIO_POLICY_SIM", "N1 · Política simulada"],
  ["P1_CASCADE_PREFETCH_SIM", "N1 · Política simulada"],
  ["B0_GEYSER_PHYSICS_BANDWIDTH", "N1 · Banda de memória"],
  ["G1_GEYSER_ZDC_LUT", "N1 · Compressão + kernel LUT"],
  ["G2_GEYSER_RRS_SALIENCE", "N1 · Saliência e cache"],
  ["G3_GEYSER_BURST", "N1 · Rajada especulativa"],
  ["G4_GEYSER_EQC", "N1 · Controlador de qualidade"],
  ["G5_GEYSER_ELASTIC_KV", "N1 · Cache KV compacto"],
  // MicroLM (§22.4): as 5 baterias N1 do modelo de referência, espelhadas
  // nos DOIS painéis (case-insensitive como as demais).
  ["B0_MICROLM_NOOP_INIT", "N1 · Init no-op exato"],
  ["P1_MICROLM_DECODE_PARITY", "N1 · Paridade de decode"],
  ["P1_MICROLM_DECODE_TOKS", "N1 · Velocidade de decode"],
  ["P1_MICROLM_TRAINS_FROM_INIT", "N1 · Treina do init"],
  ["P1_MICROLM_UNIT_CHECKS", "N1 · Checagens de unidade"],
  ["p1_microlm_decode_toks", "N1 · Velocidade de decode"],
  ["P1_RIFT_E2E_TOKS", "N1 · Velocidade ponta a ponta"],
  ["P1_GGUF_E2E_TOKS", "N1 · Velocidade ponta a ponta"],
  ["P1_CASCADE_C0_PIPELINE", "N2 · Linear (4 caminhos)"],
  ["P1_CASCADE_C1_BLOCK_GATED", "N2 · Bloco real"],
  ["CMP_RIFT_GENERATIONS", "N2 · Comparação de gerações"],
  ["cmp_geyser_generations", "N2 · Comparação de gerações"],
  ["C3_RIFT_BUNDLE_M0_FREEZE", "N3 · Congelamento do formato"],
  ["C3_RIFT_STAGE_PAGE_M0_FREEZE", "N3 · Tabela de estágios"],
  ["C3_AETHER_IR_WRITER", "N3 · Escritor de grafo"],
  ["C3_CASCADE_CPP_BUNDLE_READER", "N3 · Leitor C++"],
  ["C3_RIFT_LINEAR_ORIGINAL", "N3 · Linear original"],
  ["C3_RIFT_LINEAR_F0_ONLY", "N3 · Linear base"],
  ["C3_RIFT_LINEAR_F0_PLUS_F1_ALWAYS", "N3 · Linear completo"],
  ["C3_RIFT_LINEAR_F0_GATE_F1", "N3 · Linear inteligente (gate)"],
  ["C3_SPECTRA_BLOCK_F0_GATE_F1", "N3 · Bloco inteligente (gate)"],
  ["C3_RIFT_C1_DECISION", "N3 · Decisão de aprovação"],
  ["C3_SPECTRA_BLOCKS4_GATED", "N3 · Quatro blocos"],
  ["C3_RIFT_FULLMODEL_E2E_TOKS", "N3 · Modelo completo"],
  ["CAP_CODING", "N4 · Coding"],
  ["CAP_INTELLIGENCE", "N4 · Intelligence"],
  ["CAP_AGENTIC", "N4 · Agentic"],
  ["CAP_DEEPSEARCH_QA", "N4 · DeepSearch QA"],
  ["CAP_MCP_ATLAS", "N4 · MCP-Atlas"],
  ["CAP_TAU3_BENCH", "N4 · τ³-Bench"],
  ["CAP_SWE_BENCH", "N4 · SWE-Bench"],
  ["C4_RIFT_SECOND_FAMILY", "N5 · Segunda família"],
  ["C5_AETHER_REPR_BLOCKS", "N5 · Blocos representativos"],
  ["C6_CASCADE_COMPILE_EXECUTE", "N5 · Compilar e executar"],
];
for (const [batteryId, expectedName] of BATTERY_FRIENDLY_NAME_FIXTURES) {
  assert.equal(
    legacyBatteryFriendlyName(batteryId),
    expectedName,
    `index.html batteryFriendlyName(${JSON.stringify(batteryId)}) != ${JSON.stringify(expectedName)}`,
  );
}
// Fallback humanizado (§19.5) para id desconhecido: prefixo N<nível> e NUNCA
// o id cru (o id bruto vive somente em tooltip/title).
{
  const fallback = legacyBatteryFriendlyName("X9_FOO_BAR_TEST");
  assert.match(fallback, /^N1 · /, "index.html: fallback humanizado sem prefixo de nível");
  assert.ok(
    !fallback.includes("X9_FOO_BAR_TEST"),
    "index.html: fallback humanizado não pode exibir o id cru",
  );
}

// ---------------------------------------------------------------------------
// §19.1/§19.2/§19.3/§19.4 (7º lote) — o painel mostra DADOS e GRÁFICOS:
// explicações vivem atrás do badge ⓘ (componente infoBadge único por página),
// título visível "Comparação de modelos" sem meta-linguagem, botões
// "+ Adicionar modelo" e "adicionar todos (≤6)" lado a lado e o card
// "Comparador auditável" extinto.
// ---------------------------------------------------------------------------

// §19.1 — infoBadge: UMA implementação por página + wiring do popover.
assert.equal(
  (legacyHtml.match(/function infoBadge\(/g) || []).length,
  1,
  "index.html precisa de exatamente uma function infoBadge (§19.1)",
);
assert.ok(
  (legacyHtml.match(/data-info-toggle/g) || []).length >= 10,
  "index.html sem badges ⓘ data-info-toggle nas seções (§19.1)",
);
assert.ok(legacyHtml.includes("closeInfoPopovers"), "index.html sem wiring closeInfoPopovers (§19.1)");

// §19.2 — título da seção SEM meta-linguagem. A palavra "openrouter" pode
// sobreviver apenas em comentários de código (CSS /*...*/ ou JS //...),
// nunca em markup/strings visíveis.
assert.ok(
  legacyHtml.includes('sectionTitle">Comparação de modelos<span'),
  "index.html sem o título visível 'Comparação de modelos' com badge ⓘ (§19.2)",
);
{
  const withoutComments = legacyHtml
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n")
    .filter((line) => !/^\s*\/\//.test(line))
    .join("\n");
  assert.doesNotMatch(
    withoutComments,
    /estilo openrouter/i,
    'index.html: meta-linguagem "estilo openrouter" fora de comentário de código (§19.2)',
  );
}

// §19.3 — botões lado a lado na mesma linha flex, limite 6.
assert.ok(legacyHtml.includes("const COMPARE_ADD_ALL_LIMIT=6"), "index.html: COMPARE_ADD_ALL_LIMIT != 6 (§19.3)");
assert.ok(legacyHtml.includes("adicionar todos (≤6)"), "index.html sem o atalho 'adicionar todos (≤6)' (§19.3)");
{
  const legacyCompareBar = legacyHtml.match(/<section class="card compareBar"[\s\S]*?<\/section>/);
  assert.ok(legacyCompareBar, "index.html sem a seção compareBar (§19.3)");
  assert.ok(
    legacyCompareBar[0].includes('id="addCompareModelBtn"') && legacyCompareBar[0].includes("data-add-all-models"),
    "index.html: os dois botões precisam viver na mesma linha flex da compareBar (§19.3)",
  );
}

// §19.4/§19.6 — card "Comparador auditável" e colunas por modelo extintos
// (modelos selecionados são SEÇÕES empilhadas); portado do painel resumido.
for (const banned of [
  /Comparador auditável/,
  /Vencedor verificado por grupo/,
  /renderCompareColumn/,
  /compareColumns/,
]) {
  assert.doesNotMatch(legacyHtml, banned, `index.html ainda contém ${banned} (§19.4/§19.6)`);
}

// ---------------------------------------------------------------------------
// batteryGroupKey (§13.1) — a função de agrupamento dos cards vive SOMENTE no
// painel único (index.html, §24.1) e segue as regras ordenadas (a primeira que
// casar vence; §16 acrescentou C4/C5/C6 antes do fallback e §18 colocou
// /^CMP_/ como a PRIMEIRA regra da cadeia — os registros CMP_* continuam no
// histórico mesmo com a seção própria removida, §24.2).
// Fixtures espelham scripts/battery_group_key_fixtures.mjs.
// ---------------------------------------------------------------------------

// batteryGroupKey pode delegar em batteryLevel — injeta a versão extraída do
// próprio painel no contexto do vm.
const legacyBatteryGroupKey = vm.runInNewContext(
  `(${extractFunction(legacyHtml, "batteryGroupKey", "index.html")})`,
  { batteryLevel: legacyBatteryLevel },
  { filename: "legacy-batteryGroupKey.js" },
);
const BATTERY_GROUP_KEY_FIXTURES = [
  // Regra 1 (§18, a PRIMEIRA da cadeia — antes de /^CAP_/): /^CMP_/ →
  // 'E2E · comparação de gerações' (case-insensitive como as demais).
  ["CMP_RIFT_GENERATIONS", "E2E · comparação de gerações"],
  ["CMP_CASCADE_GENERATIONS", "E2E · comparação de gerações"],
  ["CMP_GEYSER_GENERATIONS", "E2E · comparação de gerações"],
  ["cmp_aether_generations", "E2E · comparação de gerações"],
  // Regra 2: /^CAP_/ → 'Capacidades'
  ["CAP_INTELLIGENCE", "Capacidades"],
  ["CAP_CODING", "Capacidades"],
  ["CAP_AGENTIC", "Capacidades"],
  // Regra 3: /_E2E_TOKS$/ → 'E2E · tok/s modelo completo' (antes de B0/C3)
  ["P1_CASCADE_C2_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["P1_RIFT_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["P1_AETHER_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["P1_SPECTRA_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["P1_WINNER_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["P1_GGUF_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["C3_RIFT_FULLMODEL_E2E_TOKS", "E2E · tok/s modelo completo"],
  ["C3_SPECTRA_FULLMODEL_E2E_TOKS", "E2E · tok/s modelo completo"],
  // Regra 4: /^B0_/ → 'B0 · Fundação'
  ["B0_BINARY_IR_FOUNDATION", "B0 · Fundação"],
  ["B0_GEYSER_PHYSICS_BANDWIDTH", "B0 · Fundação"],
  ["B0_GGUF_RUNTIME_SETUP", "B0 · Fundação"],
  ["B0_WINNER_CPP_BUILD_SELF_TEST", "B0 · Fundação"],
  // Regra 5: sufixos _SIM → 'P1 · políticas simuladas'
  ["P1_AETHER_PIO_POLICY_SIM", "P1 · políticas simuladas"],
  ["P1_SPECTRA_PIO_POLICY_SIM", "P1 · políticas simuladas"],
  ["P1_CASCADE_PREFETCH_SIM", "P1 · políticas simuladas"],
  // Regra 6: _C0_ (nível 2) → 'C0 · Linear 4 caminhos'
  ["P1_CASCADE_C0_PIPELINE", "C0 · Linear 4 caminhos"],
  // Regra 7: _C1_ (nível 2) → 'C1 · Bloco real'
  ["P1_CASCADE_C1_BLOCK_GATED", "C1 · Bloco real"],
  // Regra 8: C3_<TECH>_<REST> → 'C3 · REST' (mesmo passo agrupa entre techs)
  ["C3_RIFT_LINEAR_F0_GATE_F1", "C3 · LINEAR F0 GATE F1"],
  ["C3_AETHER_LINEAR_F0_GATE_F1", "C3 · LINEAR F0 GATE F1"],
  ["C3_CASCADE_BLOCK_F0_GATE_F1", "C3 · BLOCK F0 GATE F1"],
  ["C3_SPECTRA_BLOCKS4_GATED", "C3 · BLOCKS4 GATED"],
  ["C3_RIFT_C1_DECISION", "C3 · C1 DECISION"],
  // Regras 9-11 (§16, FINAL_PHASE_V1): fases finais C4/C5/C6 ANTES do
  // fallback; C6_* NÃO cai na regra 2 porque o id não termina em _E2E_TOKS.
  ["C4_RIFT_SECOND_FAMILY", "C4 · Segunda família"],
  ["C4_SPECTRA_SECOND_FAMILY", "C4 · Segunda família"],
  ["C5_AETHER_REPR_BLOCKS", "C5 · Blocos representativos"],
  ["C5_CASCADE_REPR_BLOCKS", "C5 · Blocos representativos"],
  ["C6_CASCADE_COMPILE_EXECUTE", "C6 · Compilar+Executar"],
  ["C6_RIFT_COMPILE_EXECUTE", "C6 · Compilar+Executar"],
  // Case-insensitive nas DUAS implementações (o ingest não força maiúsculas).
  ["cap_coding", "Capacidades"],
  ["p1_rift_e2e_toks", "E2E · tok/s modelo completo"],
  ["p1_cascade_c1_block_gated", "C1 · Bloco real"],
  ["c6_cascade_compile_execute", "C6 · Compilar+Executar"],
  // MicroLM (§22): B0_MICROLM_* cai na regra 4 (B0 · Fundação);
  // P1_MICROLM_* caem no fallback — inclusive DECODE_TOKS, que NÃO termina
  // em _E2E_TOKS (nenhuma regra nova foi inventada para o MicroLM).
  ["B0_MICROLM_NOOP_INIT", "B0 · Fundação"],
  ["P1_MICROLM_DECODE_PARITY", "P1 · Codec principal"],
  ["P1_MICROLM_DECODE_TOKS", "P1 · Codec principal"],
  ["P1_MICROLM_TRAINS_FROM_INIT", "P1 · Codec principal"],
  ["P1_MICROLM_UNIT_CHECKS", "P1 · Codec principal"],
  // Regra 12 (fallback): resto → 'P1 · Codec principal'
  ["P1_Q4_LINEAR_BASE_2BIT", "P1 · Codec principal"],
  ["P1_Q4_LINEAR_BASE_PLUS_REF_4BIT", "P1 · Codec principal"],
  ["G1_GEYSER_ZDC_LUT", "P1 · Codec principal"],
  ["G5_GEYSER_ELASTIC_KV", "P1 · Codec principal"],
  ["P1_WINNER_F0_PLUS_LS", "P1 · Codec principal"],
  ["", "P1 · Codec principal"],
];
for (const [batteryId, expectedGroup] of BATTERY_GROUP_KEY_FIXTURES) {
  assert.equal(
    legacyBatteryGroupKey(batteryId),
    expectedGroup,
    `index.html batteryGroupKey(${JSON.stringify(batteryId)}) != ${JSON.stringify(expectedGroup)}`,
  );
}

// ---------------------------------------------------------------------------
// index.html (painel ÚNICO, servido em "/" — §23.1/§24.1) — winner dinâmico,
// série C3, gráficos de medição portados do painel resumido e limpezas
// ---------------------------------------------------------------------------

const C3_TECH_LABELS = ["RIFT", "AETHER", "CASCADE", "SPECTRA"];
const c3PrimaryIds = C3_TECH_LABELS.flatMap((tech) => [
  `C3_${tech}_LINEAR_F0_GATE_F1`,
  `C3_${tech}_BLOCK_F0_GATE_F1`,
  `C3_${tech}_BLOCKS4_GATED`,
  `C3_${tech}_FULLMODEL_E2E_TOKS`,
]);
for (const marker of [
  "selectWinnerArchitecture",
  "Tok/s medido (e2e)",
  "somente medições reais de model.generate",
  "Bateria C3 (metodologia 16 passos)",
  "/c3/",
  "/api/results",
  "/data/rift_test_batteries.json",
  "dataSourceBadge",
  "tokSeriesCard",
  "tokSeriesChart",
  "launcherOrigin",
  "10000",
  "cascade_c0_phase1_auto_batteries.py",
  "cascade_c1_block_auto_batteries.py",
  "cascade_c2_e2e_auto_batteries.py",
  ...c3PrimaryIds,
  // GEYSER na política do winner (§1/§7).
  'ELIGIBLE=["RIFT","AETHER","CASCADE","SPECTRA","GEYSER"]',
  'PRIORITY=["CASCADE","RIFT","AETHER","SPECTRA","GEYSER"]',
  "GEYSER M0 (G0)",
  "geyser_launcher.py",
  "retestGeyserNote",
  "geyserCompareStatus",
  "geyserCurlCommand",
  "geyserCurlExample",
  "capLauncherUrl",
  // Célula Colab CURTA (§14.3): o builder emite APENAS o bootstrap; a fila
  // pesada (deps, limpeza prévia, série C, VRAM) vive no /runner.py servidor.
  "buildColabBootstrapCell",
  "buildSerialColabCode",
  "buildSingleColabCode",
  "# Observatório LLM — fila de baterias (célula curta)",
  '"/runner.py"',
  "variavel — edite a vontade",
  // CAP excluído dos rankings/winner + seção própria (§9).
  'CAP_TECHNOLOGY="CAP"',
  "capabilitiesCard",
  "capProbeBtn",
  "capProbeNote",
  "capCharts",
  "renderCapabilities",
  "copyCapProbeCell",
  "CAP_INTELLIGENCE",
  "CAP_CODING",
  "CAP_AGENTIC",
  "não é MMLU/HumanEval/SWE-bench completos",
  // Probes de benchmark agêntico (§15): 7 categorias em ordem fixa, ids de
  // contêiner espelhados nos dois dashboards, display_name lido do registro e
  // rotulagem honesta das 4 categorias novas (caption estática + renderizada).
  "CAP_CATEGORIES",
  "capabilityDisplayName",
  "capIntelligence",
  "capCoding",
  "capAgentic",
  "capDeepsearchQa",
  "capMcpAtlas",
  "capTau3Bench",
  "capSweBench",
  "CAP_DEEPSEARCH_QA",
  "CAP_MCP_ATLAS",
  "CAP_TAU3_BENCH",
  "CAP_SWE_BENCH",
  "DeepSearch QA",
  "MCP-Atlas",
  "τ³-Bench",
  "SWE-Bench",
  "probes inspirados em DeepSearch QA, MCP-Atlas, τ³-Bench e SWE-Bench (não são os benchmarks oficiais completos)",
  "#capCharts{display:grid",
  // Seletor de modelos estilo OpenRouter (redesign A6).
  "modelPicker",
  "modelPickerList",
  "modelPickerDetails",
  "fitFilterInput",
  "renderModelPickerList",
  "renderModelPickerDetails",
  "focusPickerModel",
  "togglePickerModel",
  "usePickerModel",
  "modelParamsBadge",
  "retestModelsOrField",
  // Histórico recolhível (últimos 100 registros).
  "<details",
  "historyDetails",
  "historyCount",
  // Cards por modelo agrupados por bateria (§13.1): "Antes" uma vez no topo,
  // uma linha por tecnologia e alternância de visão métricas/qualidade.
  "batteryGroupKey",
  "renderGroupCard",
  '"Antes" uma vez no topo',
  "data-quality-toggle",
  "data-view-metrics",
  "data-view-quality",
  "groupMetricChart",
  "groupGainChart",
  "groupBaseline",
  // Execução sempre de TODAS as tecnologias (§13.3) — sem botão por tech.
  "retestAllBtn",
  "retestAllNote",
  // §13.4/§21.2: nenhuma mensagem textual de IA/Gemini no Ranking Geral — só o
  // destaque ★ inline nas linhas.
  "IA recomenda",
  // Baterias por série (§21.1/§21.2/§21.3): UM card por série A–E com botão
  // único de rodada + "Rodar todas as séries" (substitui o "Teste reforçado";
  // os cards CASCADE·Série C e GEYSER foram absorvidos nos ⓘ das séries).
  'sectionTitle">Baterias por série',
  'id="seriesSection"',
  'id="seriesNote"',
  'id="runAllSeriesBtn"',
  "Rodar todas as séries",
  ...["A", "B", "C", "D", "E"].map((serie) => `Bateria · Série ${serie}`),
  ...["A", "B", "C", "D", "E"].map((serie) => `Rodar Série ${serie}`),
  "BATTERY_SERIES",
  "seriesTechs",
  "copySeriesCell",
  "wireSeriesButtons",
  '"c-series"',
  // Lista de modelos (§21.2 item 4): renderização agrupada unificada (§19.6),
  // TODOS os modelos do histórico com entrada recolhível por modelo.
  'sectionTitle">Lista de modelos',
  'id="modelList"',
  "renderModelList",
  "MODEL_LIST_OPEN_DEFAULT",
  "modelGroupChevron",
  // GGUF / Muse Glimmer 2-bit (§11): launcher, cor âmbar e sugestão.
  "ggufLauncherUrl",
  "GGUF_SUGGESTED_MODEL",
  "GGUF_SUGGESTED_QUANT",
  "--gguf:#d97706",
  "unsloth/Muse-Glimmer-30B-GGUF",
  "Muse Glimmer 2-bit (T4)",
  // Fases finais C4/C5/C6 (§16): nível 5, rota /final/ e ids primários C6.
  "Nível 5 · Fase final",
  "Todas as baterias (N1 → N5)",
  "FINAL_PHASE_V1",
  "finalLauncherUrl",
  "finalPhaseNote",
  "/final/",
  "FINAL_PHASE_PRIMARY_BATTERY_IDS",
  "C4 · Segunda família",
  "C5 · Blocos representativos",
  "C6 · Compilar+Executar",
  ...["RIFT", "AETHER", "CASCADE", "SPECTRA"].map((tech) => `C6_${tech}_COMPILE_EXECUTE`),
  // Comparação de modelos estilo openrouter/compare (§17): mesma chave de
  // localStorage do dashboard.html, modal com busca/preview e chips removíveis.
  "observatorio_selected_models",
  "+ Adicionar modelo",
  "Adicionar à comparação",
  "addCompareModelBtn",
  "compareModalOverlay",
  "compareModalList",
  "compareModalPreview",
  "compareModalSearch",
  "selectedModelChips",
  "compareModelCatalog",
  "compareEmptyStateHtml",
  "normalizeSelectedModels",
  "data-remove-compare-model",
  "data-add-compare-model",
  "data-open-compare-modal",
  "data-add-all-models",
  "COMPARE_ADD_ALL_LIMIT",
  // §24.2 — a seção "Comparação de gerações" foi REMOVIDA da UI, mas os
  // registros CMP_* permanecem no histórico e caem na regra de agrupamento
  // /^CMP_/ (nível 2, grupo próprio) — o marcador do grupo PERMANECE.
  "E2E · comparação de gerações",
  // MicroLM (§22): 7ª tecnologia tipo MODELO — constante, badge, cor, opção
  // no filtro, model_id fixo, aliases e MICROLM antes de CAP na ordem de
  // exibição por grupo; diagrama como link (ⓘ), nunca imagem inline.
  'MICROLM_TECHNOLOGY="MICROLM"',
  '<span class="badge techMicrolm">MICROLM</span>',
  '<option value="MICROLM">MICROLM</option>',
  "--microlm:#c026d3",
  "microlm/MicroLM-22M-v0.2",
  "MICROLM (modelo de referência)",
  "microlm_tok_s",
  "microlm_ram_bytes",
  "microlm_disk_bytes",
  '"MICROLM","CAP"',
  "7 tecnologias",
  "engines/microlm/diagram.svg",
  // §23.3 — largura fluida; §23.4 — classe de bit por card com badge.
  "max-width:min(1880px,96vw)",
  "batteryBitClass",
  "BIT_CLASS_ORDER",
  "badge bitClass",
  // §24.1 — gráficos do painel resumido PORTADOS para a seção "Gráficos" do
  // painel único: tok/s por modelo (baseline × candidato), speedup por bateria
  // (marcador 1,0×), redução de disco/RAM por tecnologia e latência mediana
  // por bateria primária. A série temporal de tok/s continua ÚNICA
  // (tokSeriesChart, marcador acima) — SEM um segundo gráfico temporal.
  'id="measurementCharts"',
  "renderMeasurementCharts",
  "measureChartToksPorModelo",
  "measureChartSpeedup",
  "measureChartReducao",
  "measureChartLatencia",
  "measureChartCard",
  "measureGainDisk",
  "measureGainRam",
  "measureOpCandMs",
  "chartToksModelo",
  "chartSpeedup",
  "chartReducao",
  "chartLatencia",
  "hbarChart",
  "niceTicks",
  "fmtTick",
  "fmtMsCompact",
  "latestBy",
  "avgList",
  ".measureCharts",
  "chartLegend",
  'sectionTitle">Gráficos',
  "tok/s (tokens por segundo) — maior é melhor",
  "speedup da operação (×) — acima de 1,0× é mais rápido",
  "1,0×",
  "redução média (%) — positivo = candidato consome menos",
  "latência mediana (ms) — menor é melhor",
  // §24.3 — regra geral de relevância: cabeçalho compacto com contagem de
  // grupos comparáveis, cards de grupo exigem ≥2 linhas de tecnologia e os
  // gráficos de capacidade exigem ≥2 modelos por categoria.
  "grupo(s) comparável(is)",
  ".filter(card=>card.records.length>=2)",
  "ms.size>=2",
]) {
  assert.ok(legacyHtml.includes(marker), `index.html sem marcador: ${marker}`);
}
// §24.1/§24.2 — extintos no painel único: o botão de navegação para o painel
// resumido (/v2 e /legacy não existem mais) e TODO o aparato da seção
// "Comparação de gerações" (card, fetch do relatório estático, detecção por
// schemes_tensor e tetos PROJETADOS). O grupo CMP_* do histórico permanece.
for (const banned of [
  /href="\/v2"/,
  /href="\/legacy"/,
  /Painel resumido/,
  /Comparação de gerações \(e2e real\)/,
  /compareGenerationsCard/,
  /compareGenerationsContent/,
  /compareGenerationsNote/,
  /isCompareGenerationsReport/,
  /compareGenerationsFromReport/,
  /compareGenerationsRowsFromRecords/,
  /renderCompareGenerations/,
  /loadCompareGenerationsReport/,
  /fmtCompareNumber/,
  /schemes_tensor/,
  /\/data\/compare_generations_report\.json/,
  /Tetos tok\/s — PROJETADO/,
  /chartToksTempo/,
]) {
  assert.doesNotMatch(legacyHtml, banned, `index.html ainda contém ${banned} (removido pela §24.1/§24.2)`);
}
// §23.5 — o card "WINNER executa"/"Arquitetura do WINNER (seleção dinâmica)"
// foi REMOVIDO do index.html (a função selectWinnerArchitecture permanece para
// rankings/battery — marcador positivo acima).
for (const banned of [/WINNER executa:/, /winnerArchCard/, /winnerArchContent/]) {
  assert.doesNotMatch(legacyHtml, banned, `index.html ainda contém ${banned} (removido pela §23.5)`);
}
// §22.1 — MICROLM NUNCA nas listas de elegibilidade do winner do painel
// legado (negative assert além dos literais exatos ELIGIBLE/PRIORITY).
assert.doesNotMatch(
  legacyHtml,
  /ELIGIBLE=\[[^\]]*MICROLM/,
  "index.html: MICROLM não pode entrar em ELIGIBLE (§22.1)",
);
assert.doesNotMatch(
  legacyHtml,
  /PRIORITY=\[[^\]]*MICROLM/,
  "index.html: MICROLM não pode entrar em PRIORITY (§22.1)",
);
assert.doesNotMatch(legacyHtml, /demoBtn/);
assert.doesNotMatch(legacyHtml, /Carregar exemplo/);
// Botões por tecnologia extintos (§13.3) + IA em painel dedicado extinta
// (§13.4): retest é sempre de TODAS as tecnologias e a IA é inline.
for (const banned of [
  /retestGeyserBtn/,
  /retestGeyserFromRanking/,
  /retestCascadeBtn/,
  /retestWinnerBtn/,
  /technologyInput/,
  /aiAnalysisPanel/,
  /analyzeBtn/,
]) {
  assert.doesNotMatch(legacyHtml, banned, `index.html ainda contém ${banned} (removido pela §13.3/§13.4)`);
}
assert.equal(
  legacyHtml.split('$("copyCommandBtn").addEventListener').length - 1,
  1,
  "index.html precisa registrar copyCommandBtn exatamente uma vez",
);

// ---------------------------------------------------------------------------
// §21 (9º lote) — estrutura de seções do painel legado, baterias por série
// (A–E) com TECHS da célula curta, botão único "+ Adicionar modelo" e lista de
// modelos unificada sem battery_id cru como título.
// ---------------------------------------------------------------------------

// §21.2 — ordem das seções: Ranking Geral → Ranking WINNER → Baterias por
// série → (Comparação de modelos) → Lista de modelos fechando a página.
{
  const sectionOrder = [
    'sectionTitle">Ranking geral',
    'sectionTitle">Ranking WINNER',
    'sectionTitle">Baterias por série',
    'sectionTitle">Comparação de modelos',
    'sectionTitle">Lista de modelos',
  ];
  const positions = sectionOrder.map((marker) => legacyHtml.indexOf(marker));
  positions.forEach((position, index) => {
    assert.notEqual(position, -1, `index.html sem a seção "${sectionOrder[index]}" (§21.2)`);
    if (index > 0) {
      assert.ok(
        positions[index - 1] < position,
        `index.html: seção fora da ordem do §21.2: "${sectionOrder[index]}"`,
      );
    }
  });
  // Gatilhos de rodada: exatamente A–E + "all" (Rodar todas as séries).
  const runSeries = [...legacyHtml.matchAll(/data-run-series="([^"]+)"/g)].map((m) => m[1]).sort();
  assert.deepEqual(runSeries, ["A", "B", "C", "D", "E", "all"], "index.html: gatilhos data-run-series != A–E + all (§21.2)");
  // §23.2 — o Comparador desce para DEPOIS da Lista de modelos e ANTES do
  // Histórico consolidado (modelos com gráficos primeiro).
  const comparadorAt = legacyHtml.indexOf("Comparador RIFT × CASCADE");
  const modelListAt = legacyHtml.indexOf('sectionTitle">Lista de modelos');
  const historyAt = legacyHtml.indexOf("Histórico consolidado");
  assert.notEqual(comparadorAt, -1, "index.html sem o Comparador RIFT × CASCADE (§23.2)");
  assert.notEqual(historyAt, -1, "index.html sem o Histórico consolidado (§23.2)");
  assert.ok(
    comparadorAt > modelListAt,
    "index.html: o Comparador precisa vir DEPOIS da Lista de modelos (§23.2)",
  );
  assert.ok(
    comparadorAt < historyAt,
    "index.html: o Comparador precisa vir ANTES do Histórico consolidado (§23.2)",
  );
}

// §21.3 — cards independentes CASCADE·Série C, GEYSER e "Teste reforçado"
// extintos, junto com o botão global antigo, o gráfico por rodada e QUALQUER
// mensagem textual de IA/Gemini. Comentários de código não contam.
{
  const withoutComments = legacyHtml
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n")
    .filter((line) => !/^\s*\/\//.test(line))
    .join("\n");
  for (const banned of [
    /Teste reforçado/,
    /reinforcedBtns/,
    /reinforcedNote/,
    /wireReinforcedButtons/,
    /techRoundChart/,
    /Score médio da rodada/,
    /rankingCascadeCard/,
    /rankingGeyserCard/,
    /c3Btns/,
    /c3Note\b/,
    /data-c3-tech/,
    /copyC3Cell/,
    /wireC3Buttons/,
    /runAllBatteriesBtn/,
    /Rodar todas as baterias/,
    /aiInlineStatus/,
    /renderAIInline/,
    /IA \(Gemini/,
    /data-retest-tech/,
  ]) {
    assert.doesNotMatch(withoutComments, banned, `index.html ainda contém ${banned} (extinto pela §21.3)`);
  }
}

// §21.1 — mapeamento fixo série → TECHS da célula curta, espelhado nos dois
// dashboards e restrito a tokens aceitos pelo /runner.py.
{
  const SMOKE_ORIGIN = "https://rift-lm.vercel.app";
  const EXPECTED_SERIES_TECHS = {
    // §22: a Série A (Nível 1) ganhou o MicroLM como 7ª tecnologia.
    A: ["rift", "cascade", "aether", "spectra", "winner", "geyser", "microlm"],
    B: ["c-series"],
    C: ["c3"],
    D: ["cap"],
    E: ["final"],
  };
  for (const token of [...new Set(Object.values(EXPECTED_SERIES_TECHS).flat())]) {
    assert.ok(
      runnerQueueApi.KNOWN_TECHNOLOGIES.includes(token),
      `token de série desconhecido do /runner.py: ${token} (§21.1)`,
    );
  }
  // index.html: BATTERY_SERIES + seriesTechs (A–E + "all").
  const legacySeriesContext = {};
  vm.createContext(legacySeriesContext);
  vm.runInContext(
    [
      extractConstObject(legacyHtml, "BATTERY_SERIES", "index.html"),
      extractFunction(legacyHtml, "seriesTechs", "index.html"),
      "this.seriesTechs=seriesTechs;",
    ].join("\n"),
    legacySeriesContext,
    { filename: "legacy-seriesTechs.js" },
  );
  // (spread: arrays vindos do vm têm outro Array.prototype — deepStrictEqual
  // cross-realm falharia mesmo com valores idênticos.)
  for (const [serie, techs] of Object.entries(EXPECTED_SERIES_TECHS)) {
    assert.deepEqual(
      [...legacySeriesContext.seriesTechs(serie)],
      techs,
      `index.html: seriesTechs(${JSON.stringify(serie)}) difere do §21.1`,
    );
  }
  assert.deepEqual([...legacySeriesContext.seriesTechs("all")], ["all"], 'index.html: seriesTechs("all") != ["all"] (§21.1)');
  // Células geradas por série: a linha TECHS carrega os tokens da série
  // (B → "c-series", C → "c3", D → "cap", E → "final") e a célula continua
  // curta (bootstrap §14.3).
  const legacySeriesCell = vm.runInNewContext(
    `(${extractFunction(legacyHtml, "buildColabBootstrapCell", "index.html")})`,
    { launcherOrigin: SMOKE_ORIGIN },
    { filename: "legacy-series-cell.js" },
  );
  for (const [serie, techs] of Object.entries(EXPECTED_SERIES_TECHS)) {
    const legacyCell = legacySeriesCell(["Qwen/Qwen2.5-0.5B"], legacySeriesContext.seriesTechs(serie));
    assert.ok(
      legacyCell.includes(`TECHS  = ${JSON.stringify(techs)}`),
      `index.html: célula da Série ${serie} sem TECHS  = ${JSON.stringify(techs)} (§21.1)`,
    );
    assert.ok(
      legacyCell.split("\n").length <= 25,
      `célula da Série ${serie} deixou de ser curta (> 25 linhas)`,
    );
  }
}

// §21.4 — exatamente UM botão "+ Adicionar modelo" na página (na linha de
// ações); o estado vazio da comparação é SÓ texto.
{
  assert.equal(
    (legacyHtml.match(/>\+ Adicionar modelo<\/button>/g) || []).length,
    1,
    'index.html: precisa haver exatamente UM botão "+ Adicionar modelo" (§21.4)',
  );
  assert.equal(
    (legacyHtml.match(/id="addCompareModelBtn"/g) || []).length,
    1,
    "index.html: addCompareModelBtn precisa existir exatamente uma vez (§21.4)",
  );
  const emptyStateSource = extractFunction(legacyHtml, "compareEmptyStateHtml", "index.html");
  assert.doesNotMatch(
    emptyStateSource,
    /data-open-compare-modal|\+ Adicionar modelo<\/button>/,
    "index.html: o estado vazio da comparação não pode repetir o botão (§21.4)",
  );
}

// §21.5 — NENHUM battery_id cru como título visível na Lista de modelos:
// renderização real (fixtures) de renderModelGroup/renderGroupCard com stubs
// pragmáticos nos gráficos; ids crus só podem sobreviver em atributos (title).
{
  const modelListContext = {
    // Stubs: gráficos/linhas/datas não participam de nenhum título visível.
    groupLines: (records) => records.map((rec) => ({ rec, label: rec.technology })),
    analysisForModel: () => null,
    technologyBadge: () => "techRift",
    fmtDate: (value) => String(value || "—"),
    groupMetricChart: () => "",
    groupGainChart: () => "",
    groupQualityRows: () => "",
    isPrimary: () => false,
    BATTERY_LEVEL_LABELS: {
      1: "Nível 1 · Fundação (M0)",
      2: "Nível 2 · Série C (bloco/e2e)",
      3: "Nível 3 · Metodologia C3",
      4: "Nível 4 · Capacidades",
      5: "Nível 5 · Fase final",
    },
  };
  vm.createContext(modelListContext);
  vm.runInContext(
    [
      extractFunction(legacyHtml, "esc", "index.html"),
      extractFunction(legacyHtml, "batteryLevel", "index.html"),
      extractFunction(legacyHtml, "batteryGroupKey", "index.html"),
      extractFunction(legacyHtml, "batteryFriendlyHumanize", "index.html"),
      extractFunction(legacyHtml, "batteryFriendlySuffix", "index.html"),
      extractConstObject(legacyHtml, "BATTERY_FRIENDLY_NAMES", "index.html"),
      extractConstObject(legacyHtml, "C3_STEP_FRIENDLY_NAMES", "index.html"),
      extractConstObject(legacyHtml, "GROUP_FRIENDLY_TITLES", "index.html"),
      extractFunction(legacyHtml, "groupFriendlyTitle", "index.html"),
      // §23.4: renderModelGroup/renderGroupCard particionam por classe de bit
      // (badge) — a função e a ordem fixa entram no contexto do fixture render.
      extractFunction(legacyHtml, "batteryBitClass", "index.html"),
      extractConstArray(legacyHtml, "BIT_CLASS_ORDER", "index.html"),
      extractFunction(legacyHtml, "renderGroupCard", "index.html"),
      extractFunction(legacyHtml, "renderModelGroup", "index.html"),
      extractFunction(legacyHtml, "groupRowsByModel", "index.html"),
      "this.renderModelGroup=renderModelGroup;this.groupRowsByModel=groupRowsByModel;",
    ].join("\n"),
    modelListContext,
    { filename: "legacy-model-list-render.js" },
  );
  const fixtureRow = (model, battery, technology, timestamp) => ({
    model,
    battery,
    technology,
    timestamp,
    status: "PASS",
    source: "fixture",
    quality: {},
  });
  // §24.3 — cada card (grupo × classe de bit) exige ≥2 LINHAS de tecnologia;
  // por isso as fixtures dão 2 tecnologias a cada grupo renderizável e mantêm
  // grupos de 1 tecnologia (CAP e o modelo "org/solo") para provar a OMISSÃO.
  const modelListRows = [
    fixtureRow("Qwen/Qwen2.5-0.5B", "P1_WINNER_F0_PLUS_LS", "WINNER", "2026-08-10T10:00:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "G1_GEYSER_ZDC_LUT", "GEYSER", "2026-08-10T10:01:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "B0_GEYSER_PHYSICS_BANDWIDTH", "GEYSER", "2026-08-10T10:05:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "B0_BINARY_IR_FOUNDATION", "RIFT", "2026-08-10T10:06:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "P1_CASCADE_C2_E2E_TOKS", "CASCADE", "2026-08-10T10:10:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "C3_RIFT_FULLMODEL_E2E_TOKS", "RIFT", "2026-08-10T10:11:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "C3_RIFT_LINEAR_F0_GATE_F1", "RIFT", "2026-08-10T10:15:00Z"),
    fixtureRow("Qwen/Qwen2.5-0.5B", "C3_AETHER_LINEAR_F0_GATE_F1", "AETHER", "2026-08-10T10:16:00Z"),
    // CAP tem SEMPRE uma única "tecnologia" (CAP) → card omitido pela §24.3.
    fixtureRow("Qwen/Qwen2.5-0.5B", "CAP_CODING", "CAP", "2026-08-10T10:20:00Z"),
    fixtureRow("meta-llama/Llama-3.2-1B", "CMP_RIFT_GENERATIONS", "RIFT", "2026-08-10T11:00:00Z"),
    fixtureRow("meta-llama/Llama-3.2-1B", "CMP_CASCADE_GENERATIONS", "CASCADE", "2026-08-10T11:01:00Z"),
    fixtureRow("meta-llama/Llama-3.2-1B", "C6_CASCADE_COMPILE_EXECUTE", "CASCADE", "2026-08-10T11:05:00Z"),
    fixtureRow("meta-llama/Llama-3.2-1B", "C6_RIFT_COMPILE_EXECUTE", "RIFT", "2026-08-10T11:06:00Z"),
    // Modelo com UMA tecnologia só: nenhum grupo comparável → só o cabeçalho
    // compacto (0 grupo(s)), sem card e sem estado vazio (§24.3).
    fixtureRow("org/solo", "P1_Q4_LINEAR_BASE_2BIT", "RIFT", "2026-08-10T12:00:00Z"),
  ];
  const renderedModelList = modelListContext
    .groupRowsByModel(modelListRows)
    .map((modelRows, index) =>
      modelListContext.renderModelGroup(modelRows, { collapsible: true, open: index < 3, removable: false }))
    .join("");
  // Modo lista: entrada <details> recolhível por modelo, sem botão de remover.
  assert.ok(renderedModelList.includes('<details class="card modelGroup"'), "lista de modelos sem entrada <details> por modelo (§21.2)");
  assert.ok(renderedModelList.includes("modelGroupChevron"), "lista de modelos sem chevron de collapse (§21.2)");
  assert.doesNotMatch(renderedModelList, /data-remove-compare-model/, "lista de modelos não pode ter botão de remover (§21.2)");
  // Nomes amigáveis §19.5/§21.5 presentes nos títulos dos grupos (≥2 techs).
  for (const friendly of [
    "N1 · Compressão de pesos",
    "N1 · Fundação do formato",
    "N2 · Velocidade ponta a ponta",
    "N3 · Linear inteligente (gate)",
    "N2 · Comparação de gerações",
    "N5 · Compilar e executar",
  ]) {
    assert.ok(renderedModelList.includes(friendly), `lista de modelos sem o título amigável: ${friendly} (§21.5)`);
  }
  // §24.3 — omissões: o card de CAP (1 "tecnologia") não renderiza; o modelo
  // de 1 tecnologia mostra só o cabeçalho compacto com a contagem 0 e o id da
  // bateria dele não aparece em lugar nenhum (nem em title de card).
  assert.ok(
    !renderedModelList.includes("N4 · Capacidades"),
    "§24.3: card de Capacidades com 1 linha de tecnologia deveria ser OMITIDO",
  );
  assert.ok(
    renderedModelList.includes("0 grupo(s) comparável(is)"),
    "§24.3: modelo sem grupo comparável deveria mostrar o cabeçalho compacto com contagem 0",
  );
  assert.ok(
    !renderedModelList.includes("P1_Q4_LINEAR_BASE_2BIT"),
    "§24.3: card de grupo com 1 tecnologia deveria ser OMITIDO (sem estado vazio)",
  );
  // Sanidade: os ids crus continuam disponíveis, mas SOMENTE em tooltip/title.
  assert.ok(renderedModelList.includes("C3_RIFT_LINEAR_F0_GATE_F1"), "fixtures não exercitaram os ids crus (§21.5)");
  const RAW_BATTERY_ID_RE = /\b(?:B0_|P1_|C3_|CAP_|CMP_|C[456]_)[A-Z0-9_]+/i;
  const summaries = renderedModelList.match(/<summary[\s\S]*?<\/summary>/g) || [];
  assert.ok(summaries.length >= 9, "render da lista de modelos sem os summaries esperados (§21.5)");
  for (const summary of summaries) {
    const visibleText = summary
      .replace(/\s(?:title|data-[\w-]+|aria-[\w-]+)="[^"]*"/g, "")
      .replace(/<[^>]*>/g, " ");
    assert.doesNotMatch(
      visibleText,
      RAW_BATTERY_ID_RE,
      `§21.5: battery_id cru como título visível na lista de modelos: ${visibleText.trim().slice(0, 120)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// §23.4 (11º lote) — batteryBitClass(battery_id, metrics) no painel único:
// dentro de um grupo é DESLEAL misturar precisões — um card por
// (recurso × classe de bit) com badge. A cadeia canônica de tokens do id
// (TERNARY > 2BIT/INT2 > 4BIT/INT4, case-insensitive) é fixada por fixtures;
// só as classes normalizadas {'2-bit','ternário','4-bit','baixo-bit'} são
// emitidas ('INT4' do texto do contrato é normalizado para '4-bit').
// ---------------------------------------------------------------------------

{
  const legacyBitClass = vm.runInNewContext(
    `(${extractFunction(legacyHtml, "batteryBitClass", "index.html")})`,
    {},
    { filename: "legacy-batteryBitClass.js" },
  );
  const BIT_CLASSES = ["2-bit", "ternário", "4-bit", "baixo-bit"];
  // BIT_CLASS_ORDER: as 4 classes canônicas (a ordem de exibição é layout).
  {
    const order = vm.runInNewContext(
      `(${extractConstArray(legacyHtml, "BIT_CLASS_ORDER", "index.html").replace(/^const BIT_CLASS_ORDER=/, "").replace(/;$/, "")})`,
      {},
      { filename: "index.html-bit-class-order.js" },
    );
    assert.deepEqual([...order].sort(), [...BIT_CLASSES].sort(), "index.html: BIT_CLASS_ORDER != as 4 classes canônicas (§23.4)");
  }
  // Fixtures canônicas (ids reais dos 75 registros + tokens do contrato):
  // [battery_id, metrics, classe esperada].
  const BATTERY_BIT_CLASS_FIXTURES = [
    // Tokens do battery_id — 2BIT vence o prefixo Q4 (armadilha clássica).
    ["P1_Q4_LINEAR_BASE_2BIT", undefined, "2-bit"],
    ["P1_Q4_LINEAR_BASE_PLUS_REF_4BIT", undefined, "4-bit"],
    ["SELFTEST_Q4_BASE_PLUS_REF_4BIT", undefined, "4-bit"],
    // TERNARY vence 2BIT no mesmo id (case-insensitive).
    ["P1_AETHER_HQR_TERNARY_2BIT", undefined, "ternário"],
    ["P1_SPECTRA_HQR_TERNARY_2BIT", undefined, "ternário"],
    ["P1_WINNER_F0_TERNARY_2BIT", undefined, "ternário"],
    ["p1_winner_f0_ternary_2bit", undefined, "ternário"],
    // INT4/INT2 do §23.4 normalizados para as classes canônicas.
    ["G1_GEYSER_DRAFT_INT4_PROXY", undefined, "4-bit"],
    ["G1_GEYSER_DRAFT_INT2_HOT", undefined, "2-bit"],
    // Fallback por métricas (bits efetivos medidos): ≤3 → 2-bit, ≤5 → 4-bit,
    // acima → baixo-bit.
    ["CMP_RIFT_GENERATIONS", { bits_effective: 2.4 }, "2-bit"],
    ["CMP_GEYSER_GENERATIONS", { bits_effective: 3.7 }, "4-bit"],
    ["G2_GEYSER_RRS_SALIENCE", { bits_effective: 8 }, "baixo-bit"],
    // Sem token e sem bits medidos → 'baixo-bit' (fallback honesto).
    ["P1_WINNER_F0_PLUS_LS", undefined, "baixo-bit"],
    ["P1_AETHER_HQR_PLUS_TADDS_DYNAMIC", undefined, "baixo-bit"],
    ["B0_GEYSER_PHYSICS_BANDWIDTH", undefined, "baixo-bit"],
    ["P1_MICROLM_DECODE_TOKS", undefined, "baixo-bit"],
    ["", undefined, "baixo-bit"],
  ];
  for (const [batteryId, metrics, expected] of BATTERY_BIT_CLASS_FIXTURES) {
    assert.equal(
      legacyBitClass(batteryId, metrics),
      expected,
      `index.html batteryBitClass(${JSON.stringify(batteryId)}) != ${JSON.stringify(expected)} (§23.4)`,
    );
    assert.ok(BIT_CLASSES.includes(legacyBitClass(batteryId, metrics)), "classe fora do conjunto canônico (§23.4)");
  }
}

// ---------------------------------------------------------------------------
// Célula Colab CURTA (§14.3) — o painel emite exatamente o bootstrap do
// contrato (Secrets → BASE → MODELS → TECHS → exec de /runner.py); toda a
// lógica pesada vive no orquestrador servido em GET /runner.py.
// ---------------------------------------------------------------------------

{
  const SMOKE_ORIGIN = "https://rift-lm.vercel.app";
  const legacyBuildCell = vm.runInNewContext(
    `(${extractFunction(legacyHtml, "buildColabBootstrapCell", "index.html")})`,
    { launcherOrigin: SMOKE_ORIGIN },
    { filename: "legacy-buildColabBootstrapCell.js" },
  );
  // TEMPLATE EXATO do contrato §14.3 (com BASE = launcherOrigin preenchido).
  const expectedCell = [
    "# Observatório LLM — fila de baterias (célula curta)",
    "from google.colab import userdata",
    "import os, urllib.request",
    'for k in ("RIFT_INGEST_TOKEN", "HF_TOKEN"):',
    "    try:",
    '        v = str(userdata.get(k) or "").strip()',
    "        if v: os.environ[k] = v",
    "    except Exception: pass",
    'if len(os.environ.get("RIFT_INGEST_TOKEN", "")) < 32:',
    '    raise SystemExit("Configure o Secret RIFT_INGEST_TOKEN (>=32 chars) no Colab")',
    `BASE = "${SMOKE_ORIGIN}"`,
    'MODELS = ["Qwen/Qwen2.5-0.5B"]   # variavel — edite a vontade',
    'TECHS  = ["all"]              # ou lista: rift,cascade,aether,spectra,winner,geyser,microlm,c3,final,cap,gguf',
    'exec(urllib.request.urlopen(BASE + "/runner.py").read().decode("utf-8"))',
  ].join("\n");
  const legacyCell = legacyBuildCell(["Qwen/Qwen2.5-0.5B"], ["all"]);
  assert.equal(legacyCell, expectedCell, "index.html: célula curta difere do template §14.3");
  assert.ok(expectedCell.split("\n").length <= 25, "célula curta deixou de ser curta (> 25 linhas)");
  assert.ok(legacyCell.includes("/runner.py"), "index.html: célula sem exec de /runner.py");
  assert.ok(legacyCell.includes("MODELS = "), "index.html: célula sem variável MODELS");
  assert.ok(legacyCell.includes("TECHS  = "), "index.html: célula sem variável TECHS");
  assert.doesNotMatch(
    legacyCell,
    /raw\.githubusercontent/,
    "index.html: célula curta não pode apontar para raw.githubusercontent (repo/ref são do servidor)",
  );
  assert.doesNotMatch(
    legacyCell,
    /RIFT_GITHUB_REPOSITORY/,
    "index.html: repo/ref são exportados pelo /runner.py, não pela célula curta",
  );
  // Variante GGUF: TECHS=["gguf"] continua sendo uma célula curta válida.
  const ggufCell = legacyBuildCell(["unsloth/Muse-Glimmer-30B-GGUF"], ["gguf"]);
  assert.ok(ggufCell.includes('TECHS  = ["gguf"]'), "variante GGUF sem TECHS=[\"gguf\"]");
  assert.ok(ggufCell.split("\n").length <= 25);
  // Fila com vários modelos continua sendo UMA célula curta (MODELS é a variável).
  const serialCell = legacyBuildCell(["org/a", "org/b"], ["all"]);
  assert.ok(serialCell.includes('"org/a"') && serialCell.includes('"org/b"'));
  assert.ok(serialCell.split("\n").length <= 25, "fila serial precisa continuar em UMA célula curta");
}

// Legados aposentados (§14): o painel não embute raw.githubusercontent nas
// células (repo/ref resolvidos no servidor) e o select "Modelo sugerido" do
// card de fila foi removido (§14.4). EXCEÇÃO ÚNICA (§22.4): o diagrama do
// MicroLM vira link (dentro do ⓘ, não imagem inline) — exatamente UMA URL raw
// na página, terminando em /engines/microlm/diagram.svg. Qualquer outra URL
// raw (script/launcher/célula) continua proibida.
{
  const rawUrls = legacyHtml.match(/raw\.githubusercontent[^\s"'<>)\\]*/g) || [];
  assert.equal(
    rawUrls.length,
    1,
    "index.html: exatamente UMA URL raw.githubusercontent é permitida — o diagrama do MicroLM (§14.1/§22.4)",
  );
  assert.match(
    rawUrls[0],
    /\/engines\/microlm\/diagram\.svg$/,
    `index.html: a única URL raw permitida é o diagrama do MicroLM (§22.4), não ${rawUrls[0]}`,
  );
  assert.doesNotMatch(legacyHtml, /Modelo sugerido/, 'index.html ainda tem o select "Modelo sugerido" (§14.4)');
}

// ---------------------------------------------------------------------------
// api/results.mjs — validação/merge (inalterado) + whitelist de campos
// ---------------------------------------------------------------------------

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

// Whitelist de campos de topo: chaves desconhecidas são descartadas (sem
// spread arbitrário) e contadas em dropped_unknown_keys.
const whitelisted = resultsApi.whitelistRecordFields({
  run_id: "x",
  battery_id: "y",
  rift_tok_s: 10,
  evil_injected_key: 1,
});
assert.deepEqual(Object.keys(whitelisted.kept).sort(), ["battery_id", "rift_tok_s", "run_id"]);
assert.equal(whitelisted.dropped, 1);
const dropStats = { droppedKeys: 0 };
const [cleanRecord] = resultsApi.validateHistory(
  [{ ...riftOld, unexpected_top_level_key: true }],
  "test",
  dropStats,
);
assert.equal(dropStats.droppedKeys, 1);
assert.ok(!("unexpected_top_level_key" in cleanRecord));

// GEYSER e CAP no enum de tecnologias do ingest (§7/§9). O timestamp do
// GEYSER usa offset ISO-8601 '+00:00' (não sufixo 'Z') e precisa ser aceito.
const geyserRecord = {
  run_id: "geyser-20260810T120000Z-0123abcd",
  battery_id: "G1_GEYSER_ZDC_LUT",
  technology: "GEYSER",
  timestamp_utc: "2026-08-10T12:00:00+00:00",
  status: "PASS",
  model_id: "Qwen/Qwen2.5-0.5B",
  geyser_tok_s: null,
};
const capRecord = {
  run_id: "20260810T120000Z-89abcdef",
  battery_id: "CAP_INTELLIGENCE",
  technology: "CAP",
  timestamp_utc: "2026-08-10T12:00:00+00:00",
  status: "PASS",
  model_id: "Qwen/Qwen2.5-0.5B",
};
const newTechRecords = resultsApi.validateHistory([geyserRecord, capRecord], "test");
assert.equal(newTechRecords.length, 2);
assert.equal(newTechRecords[0].technology, "GEYSER");
assert.equal(newTechRecords[1].technology, "CAP");
assert.equal(resultsApi.inferTechnology({ battery_id: "CAP_AGENTIC" }), "CAP");
assert.equal(
  resultsApi.inferTechnology({ battery_id: "G1_GEYSER_ZDC_LUT", spec: "GEYSER-LM v0.1" }),
  "GEYSER",
);
// Alias geyser_* entra na whitelist; cap_* NÃO (CAP não tem campos de topo).
const aliasCheck = resultsApi.whitelistRecordFields({ run_id: "x", geyser_tok_s: 1, cap_tok_s: 2 });
assert.deepEqual(Object.keys(aliasCheck.kept).sort(), ["geyser_tok_s", "run_id"]);
assert.equal(aliasCheck.dropped, 1);

// GGUF no ingest (§11): enum, alias gguf_* e model_id com sufixo ':<tag>'
// opcional (CAP-sobre-GGUF publica 'org/modelo-GGUF:UD-Q2_K_XL').
assert.ok(resultsApi.TECHNOLOGIES.has("GGUF"), "api/results.mjs sem GGUF no enum de technology (§11)");
assert.match("gguf_tok_s", resultsApi.RECORD_ALIAS_PREFIX_RE);
assert.ok(resultsApi.RECORD_MODEL_ID_RE.test("Qwen/Qwen2.5-0.5B"));
assert.ok(resultsApi.RECORD_MODEL_ID_RE.test("unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL"));
assert.ok(!resultsApi.RECORD_MODEL_ID_RE.test("unsloth/Muse-Glimmer-30B-GGUF:"), "sufixo ':' vazio não pode passar");
assert.ok(!resultsApi.RECORD_MODEL_ID_RE.test("sem-barra:UD-Q2_K_XL"), "model_id sem org/ não pode passar");
const ggufRecord = {
  run_id: "gguf-20260810T120000Z-0123abcd",
  battery_id: "P1_GGUF_E2E_TOKS",
  technology: "GGUF",
  timestamp_utc: "2026-08-10T12:00:00Z",
  status: "PASS",
  model_id: "unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL",
  gguf_tok_s: 12.5,
};
const [validGguf] = resultsApi.validateHistory([ggufRecord], "test");
assert.equal(validGguf.technology, "GGUF");
assert.equal(validGguf.model_id, "unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL");
const ggufAlias = resultsApi.whitelistRecordFields({
  run_id: "x",
  gguf_tok_s: 1,
  gguf_ram_bytes: 2,
  gguf_disk_bytes: 3,
});
assert.deepEqual(
  Object.keys(ggufAlias.kept).sort(),
  ["gguf_disk_bytes", "gguf_ram_bytes", "gguf_tok_s", "run_id"],
);
assert.equal(ggufAlias.dropped, 0);

// MICROLM no ingest (§22.1): enum, alias microlm_*, prefixos reservados
// B0_/P1_MICROLM_ no inferTechnology e model_id fixo do modelo de referência.
assert.ok(resultsApi.TECHNOLOGIES.has("MICROLM"), "api/results.mjs sem MICROLM no enum de technology (§22.1)");
assert.match("microlm_tok_s", resultsApi.RECORD_ALIAS_PREFIX_RE);
assert.equal(resultsApi.inferTechnology({ battery_id: "B0_MICROLM_NOOP_INIT" }), "MICROLM");
assert.equal(resultsApi.inferTechnology({ battery_id: "P1_MICROLM_DECODE_TOKS" }), "MICROLM");
assert.ok(resultsApi.RECORD_MODEL_ID_RE.test("microlm/MicroLM-22M-v0.2"));
const microlmRecord = {
  run_id: "microlm-20260811T120000Z-0123abcd",
  battery_id: "P1_MICROLM_DECODE_TOKS",
  technology: "MICROLM",
  timestamp_utc: "2026-08-11T12:00:00Z",
  status: "PASS",
  model_id: "microlm/MicroLM-22M-v0.2",
  microlm_tok_s: 3.2,
};
const [validMicrolm] = resultsApi.validateHistory([microlmRecord], "test");
assert.equal(validMicrolm.technology, "MICROLM");
assert.equal(validMicrolm.model_id, "microlm/MicroLM-22M-v0.2");
const microlmAlias = resultsApi.whitelistRecordFields({
  run_id: "x",
  microlm_tok_s: 1,
  microlm_ram_bytes: 2,
  microlm_disk_bytes: 3,
});
assert.deepEqual(
  Object.keys(microlmAlias.kept).sort(),
  ["microlm_disk_bytes", "microlm_ram_bytes", "microlm_tok_s", "run_id"],
);
assert.equal(microlmAlias.dropped, 0);

// scripts/validate_data.mjs espelha o enum e os aliases (§22.1).
{
  const validateDataSource = await readFile(new URL("./validate_data.mjs", import.meta.url), "utf8");
  for (const marker of ['"MICROLM"', "microlm_tok_s", "microlm_ram_bytes", "microlm_disk_bytes"]) {
    assert.ok(validateDataSource.includes(marker), `scripts/validate_data.mjs sem marcador MICROLM: ${marker} (§22.1)`);
  }
}

// Histórico agora é append-only por run_id+technology+battery_id.
const merged = resultsApi.mergeHistories([riftOld], [riftNew, cascade]);
assert.equal(merged.length, 3);
assert.deepEqual(merged.map((record) => record.run_id), ["run-rift-old", "run-rift-new", "run-cascade"]);
const idempotent = resultsApi.mergeHistories(merged, [riftNew]);
assert.equal(idempotent.length, 3);
assert.equal(resultsApi.recordKey(riftNew), "run-rift-new\u0000RIFT\u0000P1_Q4_LINEAR_BASE_PLUS_REF_4BIT");

// ---------------------------------------------------------------------------
// Política do WINNER dinâmico (docs/C3_CONTRACTS_V1.md §1) — função pura
// exportada por api/results.mjs e espelhada nos dashboards e em Python.
// ---------------------------------------------------------------------------

const WINNER_ARCHS = ["RIFT", "AETHER", "CASCADE", "SPECTRA", "GEYSER"];
assert.deepEqual(resultsApi.WINNER_ELIGIBLE_TECHS, ["RIFT", "AETHER", "CASCADE", "SPECTRA", "GEYSER"]);
assert.deepEqual(resultsApi.WINNER_TIE_ORDER, ["CASCADE", "RIFT", "AETHER", "SPECTRA", "GEYSER"]);
assert.equal(
  resultsApi.WINNER_TIE_ORDER.at(-1),
  "GEYSER",
  "GEYSER precisa ser o último da ordem de desempate (§1 passo 5)",
);
const winnerBase = {
  comparison_role: "primary",
  status: "PASS",
  battery_id: "C3_X_FULLMODEL_E2E_TOKS",
  timestamp_utc: "2026-08-09T10:00:00Z",
};

// Sem dados -> incumbente CASCADE (selection_basis default_incumbent).
assert.equal(selectWinnerArchitecture([]), "CASCADE");
assert.equal(selectWinnerArchitecture(null), "CASCADE");

// Mais modelos otimizados vence.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "RIFT", model_id: "org/a" },
  { ...winnerBase, technology: "RIFT", model_id: "org/b" },
  { ...winnerBase, technology: "CASCADE", model_id: "org/a" },
]), "RIFT");

// synthetic/, status reprovado e quality gate false não contam.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "RIFT", model_id: "synthetic/x" },
  { ...winnerBase, technology: "RIFT", model_id: "org/f", status: "FAIL" },
  { ...winnerBase, technology: "RIFT", model_id: "org/g", quality: { full_local_gate_pass: false } },
  { ...winnerBase, technology: "AETHER", model_id: "org/a", status: "EXPERIMENTAL_PASS" },
]), "AETHER");

// Registro diagnóstico (comparison_role != primary) não conta.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "RIFT", model_id: "org/a", comparison_role: null },
]), "CASCADE");

// Empate em nº de modelos: maior score médio (SCORE_WEIGHTS) desempata.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "SPECTRA", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.999, nrmse: 0.001 } } },
  { ...winnerBase, technology: "RIFT", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.2, nrmse: 0.09 } } },
]), "SPECTRA");

// Empate persistente -> ordem fixa CASCADE, RIFT, AETHER, SPECTRA, GEYSER.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "SPECTRA", model_id: "org/a" },
  { ...winnerBase, technology: "CASCADE", model_id: "org/b" },
]), "CASCADE");

// GEYSER é elegível (§7): mais modelos otimizados via G1_GEYSER_ZDC_LUT vence.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "GEYSER", model_id: "org/a", battery_id: "G1_GEYSER_ZDC_LUT" },
  { ...winnerBase, technology: "GEYSER", model_id: "org/b", battery_id: "G1_GEYSER_ZDC_LUT" },
  { ...winnerBase, technology: "CASCADE", model_id: "org/a" },
]), "GEYSER");

// GEYSER é o ÚLTIMO no desempate persistente: SPECTRA ganha de GEYSER.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "SPECTRA", model_id: "org/a" },
  { ...winnerBase, technology: "GEYSER", model_id: "org/b", battery_id: "G1_GEYSER_ZDC_LUT" },
]), "SPECTRA");

// CAP NUNCA entra na seleção do winner (§9), mesmo se marcado como primário.
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "CAP", model_id: "org/a", battery_id: "CAP_INTELLIGENCE" },
  { ...winnerBase, technology: "CAP", model_id: "org/b", battery_id: "CAP_CODING" },
]), "CASCADE");

// MICROLM NUNCA entra na seleção do winner (§22.1) — é MODELO de referência,
// não otimizador — mesmo com registro primário/PASS; e nunca desloca uma
// tecnologia elegível com menos modelos.
assert.ok(!resultsApi.WINNER_ELIGIBLE_TECHS.includes("MICROLM"), "MICROLM não pode ser elegível (§22.1)");
assert.ok(!resultsApi.WINNER_TIE_ORDER.includes("MICROLM"), "MICROLM não pode estar na ordem de desempate (§22.1)");
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "MICROLM", model_id: "microlm/MicroLM-22M-v0.2", battery_id: "P1_MICROLM_DECODE_TOKS" },
]), "CASCADE");
assert.equal(selectWinnerArchitecture([
  { ...winnerBase, technology: "MICROLM", model_id: "org/a", battery_id: "P1_MICROLM_DECODE_TOKS" },
  { ...winnerBase, technology: "MICROLM", model_id: "org/b", battery_id: "P1_MICROLM_DECODE_TOKS" },
  { ...winnerBase, technology: "AETHER", model_id: "org/a" },
]), "AETHER");

assert.equal(resultsApi.winnerRecordScore({}), null);
assert.equal(typeof resultsApi.winnerScoreComponents, "function");

// Contra o histórico real do repositório o resultado precisa ser uma das
// cinco arquiteturas elegíveis (dados novos podem mudar o vencedor, mas
// nunca para fora do enum).
const publishedHistory = JSON.parse(
  await readFile(new URL("../data/rift_test_batteries.json", import.meta.url), "utf8"),
);
assert.ok(WINNER_ARCHS.includes(selectWinnerArchitecture(publishedHistory)));

// §24.1 — o espelhamento da política do winner agora é index.html ↔
// api/results.mjs ↔ winner_m0 Python (4 implementações, com as fórmulas de
// api/analyze.mjs): a versão do painel único precisa escolher a MESMA
// arquitetura que a exportada por api/results.mjs em todos os cenários do §1
// e sobre o histórico real publicado.
{
  const legacySelectWinner = vm.runInNewContext(
    `(${extractFunction(legacyHtml, "selectWinnerArchitecture", "index.html")})`,
    {},
    { filename: "legacy-selectWinnerArchitecture.js" },
  );
  const WINNER_EQUIVALENCE_FIXTURES = [
    [],
    [
      { ...winnerBase, technology: "RIFT", model_id: "org/a" },
      { ...winnerBase, technology: "RIFT", model_id: "org/b" },
      { ...winnerBase, technology: "CASCADE", model_id: "org/a" },
    ],
    [
      { ...winnerBase, technology: "RIFT", model_id: "synthetic/x" },
      { ...winnerBase, technology: "RIFT", model_id: "org/f", status: "FAIL" },
      { ...winnerBase, technology: "RIFT", model_id: "org/g", quality: { full_local_gate_pass: false } },
      { ...winnerBase, technology: "AETHER", model_id: "org/a", status: "EXPERIMENTAL_PASS" },
    ],
    [{ ...winnerBase, technology: "RIFT", model_id: "org/a", comparison_role: null }],
    [
      { ...winnerBase, technology: "SPECTRA", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.999, nrmse: 0.001 } } },
      { ...winnerBase, technology: "RIFT", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.2, nrmse: 0.09 } } },
    ],
    [
      { ...winnerBase, technology: "SPECTRA", model_id: "org/a" },
      { ...winnerBase, technology: "GEYSER", model_id: "org/b", battery_id: "G1_GEYSER_ZDC_LUT" },
    ],
    [
      { ...winnerBase, technology: "CAP", model_id: "org/a", battery_id: "CAP_INTELLIGENCE" },
      { ...winnerBase, technology: "MICROLM", model_id: "microlm/MicroLM-22M-v0.2", battery_id: "P1_MICROLM_DECODE_TOKS" },
    ],
    publishedHistory,
  ];
  for (const [index, records] of WINNER_EQUIVALENCE_FIXTURES.entries()) {
    const legacyResult = legacySelectWinner(records);
    assert.equal(
      legacyResult.architecture,
      selectWinnerArchitecture(records),
      `winner divergente entre index.html e api/results.mjs no cenário ${index} (§24.1)`,
    );
    assert.ok(
      ["published_history", "env_override", "default_incumbent"].includes(legacyResult.selection_basis),
      `index.html: selection_basis fora do enum do §1 no cenário ${index}`,
    );
  }
}

// ---------------------------------------------------------------------------
// §25 — score canônico v2 ("computador convencional": 4 núcleos, 8 GB de RAM
// livre, SEM GPU — a RAM é a restrição dura). Pesos novos cosine 25 ·
// nrmse 10 · gate 5 · ram 30 · speedup 20 · disk 10 espelhados em
// api/analyze.mjs, api/results.mjs, index.html e winner_m0 Python; as
// normalizações do §1 e o fator de coverage (0.65 + 0.35 × coverage) NÃO
// mudam. Valores esperados recalculados rodando as implementações reais.
// ---------------------------------------------------------------------------

const SCORE_WEIGHTS_V2_LITERAL =
  "const SCORE_WEIGHTS={output_cosine:25,output_nrmse:10,disk_reduction_pct:10,ram_reduction_pct:30,operation_speedup_x:20,quality_gate_pass:5};";
assert.ok(
  legacyHtml.includes("const WEIGHTS={cosine:25,nrmse:10,disk:10,ram:30,speedup:20,gate:5};"),
  "index.html: WEIGHTS do selectWinnerArchitecture fora dos pesos do §25",
);
assert.ok(legacyHtml.includes(SCORE_WEIGHTS_V2_LITERAL), "index.html: SCORE_WEIGHTS fora dos pesos do §25");
// Textos visíveis dos ⓘ de ranking: composição 40/30/20/10 + objetivo declarado.
assert.ok(
  legacyHtml.includes(
    "Score composto de referência: Qualidade 40% • RAM 30% • latência da operação 20% • disco 10% • ajuste por cobertura. 0 → pior resultado medido; 100 → referência ideal. Pesos calibrados para PC convencional: 4 núcleos, 8 GB de RAM livre, sem GPU.",
  ),
  "index.html: ⓘ do Ranking geral sem a composição/objetivo do §25",
);
assert.ok(
  legacyHtml.includes("Qualidade 40% • RAM 30% • latência da operação 20% • disco 10% • cobertura"),
  "index.html: ⓘ do Ranking WINNER sem a composição do §25",
);
assert.equal(
  legacyHtml.split("Pesos calibrados para PC convencional: 4 núcleos, 8 GB de RAM livre, sem GPU.").length - 1,
  2,
  "index.html: a frase do objetivo (§25) precisa aparecer exatamente 2× (os dois ⓘ de ranking)",
);
// Pesos/textos ANTIGOS extintos — qualquer um deles reintroduziria o score v1.
for (const banned of [
  /cosine:25,nrmse:15/,
  /output_nrmse:15/,
  /disk_reduction_pct:20/,
  /ram_reduction_pct:15/,
  /qualidade 45%/i,
  /disco 20%/,
  /RAM 15%/,
  /latência 20% • cobertura/,
]) {
  assert.doesNotMatch(legacyHtml, banned, `index.html ainda contém pesos antigos ${banned} (§25)`);
}

// winner_m0 Python (4ª implementação espelhada) — pesos §25 literais; o
// npm test não executa Python, então o espelho é pinado por marcador.
for (const marker of [
  '"output_cosine": 25',
  '"output_nrmse": 10',
  '"disk_reduction_pct": 10',
  '"ram_reduction_pct": 30',
  '"operation_speedup_x": 20',
  '"quality_gate_pass": 5',
]) {
  assert.ok(winnerPy.includes(marker), `winner_m0 Python sem o peso §25: ${marker}`);
}
for (const banned of [/"output_nrmse": 15/, /"disk_reduction_pct": 20/, /"ram_reduction_pct": 15/]) {
  assert.doesNotMatch(winnerPy, banned, `winner_m0 Python ainda contém peso antigo ${banned} (§25)`);
}

// winnerRecordScore (api/results.mjs) — valores EXATOS com os pesos §25 sobre
// as fixtures do desempate (só qualidade → coverage 40%, fator 0.79):
// SPECTRA: (99.95×25 + 99×10 + 100×5)/40 = 99.7… → ×0.79 = 78.7778125;
// RIFT:    (60×25 + 10×10 + 100×5)/40 = 52.5   → ×0.79 = 41.475.
{
  const spectraTie = { ...winnerBase, technology: "SPECTRA", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.999, nrmse: 0.001 } } };
  const riftTie = { ...winnerBase, technology: "RIFT", model_id: "org/a", quality: { full_local_gate_pass: true, output: { cosine: 0.2, nrmse: 0.09 } } };
  const spectraScore = resultsApi.winnerRecordScore(spectraTie);
  const riftScore = resultsApi.winnerRecordScore(riftTie);
  assert.ok(
    Math.abs(spectraScore - 78.7778125) < 1e-9,
    `winnerRecordScore(SPECTRA tie) = ${spectraScore} != 78.7778125 (pesos §25)`,
  );
  assert.ok(
    Math.abs(riftScore - 41.475) < 1e-9,
    `winnerRecordScore(RIFT tie) = ${riftScore} != 41.475 (pesos §25)`,
  );
}

// scoreAnalysisTechnology (index.html) — harness com o SCORE_WEIGHTS literal
// já assertado acima: registro rico (coverage 100%) e cobertura parcial
// só-RAM (coverage 30% → fator 0.755).
{
  const scoreHarness = {};
  vm.createContext(scoreHarness);
  vm.runInContext(
    [
      SCORE_WEIGHTS_V2_LITERAL,
      extractFunction(legacyHtml, "clamp", "index.html"),
      extractFunction(legacyHtml, "normalizedScoreMetric", "index.html"),
      extractFunction(legacyHtml, "scoreAnalysisTechnology", "index.html"),
      "this.scoreAnalysisTechnology=scoreAnalysisTechnology;",
    ].join("\n"),
    scoreHarness,
    { filename: "legacy-score-v2.js" },
  );
  // cosine 99.75×25 + nrmse 97×10 + disk 80×10 + ram 60×30 + speedup 80×20 +
  // gate 100×5 = 8163.75/100 → 81.6375 exato → toFixed(2) = 81.64.
  const rich = scoreHarness.scoreAnalysisTechnology("Qwen/Qwen2.5-0.5B", "RIFT", {
    output_cosine: 0.995,
    output_nrmse: 0.003,
    quality_gate_pass: true,
    disk_reduction_pct: 80,
    ram_reduction_pct: 60,
    operation_speedup_x: 0.8,
  });
  assert.equal(rich.score, 81.64, "index.html scoreAnalysisTechnology: score != 81.64 (pesos §25)");
  assert.equal(rich.raw_score, 81.64, "index.html scoreAnalysisTechnology: raw_score != 81.64 (pesos §25)");
  assert.equal(rich.coverage_pct, 100, "index.html scoreAnalysisTechnology: coverage_pct != 100");
  // Só RAM presente: raw 50, coverage 30% → 50 × (0.65 + 0.35×0.30) = 37.75.
  const ramOnly = scoreHarness.scoreAnalysisTechnology("Qwen/Qwen2.5-0.5B", "RIFT", { ram_reduction_pct: 50 });
  assert.equal(ramOnly.score, 37.75, "index.html scoreAnalysisTechnology (só-RAM): score != 37.75 (§25)");
  assert.equal(ramOnly.coverage_pct, 30, "index.html scoreAnalysisTechnology (só-RAM): coverage_pct != 30");
}

// ---------------------------------------------------------------------------
// GET /api/results — endpoint público com ETag/304 e fallback deploy_bundle
// ---------------------------------------------------------------------------

{
  const savedEnv = {};
  for (const name of ["GITHUB_REPO", "RIFT_GITHUB_REPOSITORY", "RIFT_GITHUB_BRANCH", "RIFT_GITHUB_DATA_PATH"]) {
    savedEnv[name] = process.env[name];
    delete process.env[name];
  }
  assert.equal(resultsApi.historyRepository(), "programador-powershell/RIFT-LM");
  assert.equal(resultsApi.historyBranch(), "main");
  assert.equal(resultsApi.historyDataPath(), "data/rift_test_batteries.json");

  // Sem rede (fetch stub que falha) o GET precisa cair no bundle do deploy.
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline (stub do smoke test)");
  };
  try {
    const getResponse = await resultsHandler.fetch(
      new Request("https://dashboard.example/api/results"),
    );
    assert.equal(getResponse.status, 200);
    assert.equal(getResponse.headers.get("x-history-source"), "deploy_bundle");
    assert.match(getResponse.headers.get("cache-control"), /s-maxage=15/);
    assert.match(getResponse.headers.get("cache-control"), /stale-while-revalidate=60/);
    const etag = getResponse.headers.get("etag");
    assert.match(etag, /^"sha256-[0-9a-f]{32}"$/);
    const payload = await getResponse.json();
    assert.ok(Array.isArray(payload.records));
    assert.equal(payload.count, payload.records.length);
    assert.ok(Number.isFinite(Date.parse(payload.generated_at)));
    assert.equal(payload.source, "deploy_bundle");

    const cachedResponse = await resultsHandler.fetch(
      new Request("https://dashboard.example/api/results", {
        headers: { "If-None-Match": etag },
      }),
    );
    assert.equal(cachedResponse.status, 304);

    const headResponse = await resultsHandler.fetch(
      new Request("https://dashboard.example/api/results", { method: "HEAD" }),
    );
    assert.equal(headResponse.status, 200);
    assert.equal(await headResponse.text(), "");
  } finally {
    globalThis.fetch = realFetch;
    for (const [name, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }

  // Método não permitido anuncia GET, HEAD, POST.
  const methodNotAllowed = await resultsHandler.fetch(
    new Request("https://dashboard.example/api/results", { method: "PUT" }),
  );
  assert.equal(methodNotAllowed.status, 405);
  assert.equal(methodNotAllowed.headers.get("allow"), "GET, HEAD, POST");

  // POST continua protegido por Bearer >= 32 chars (token dummy de teste).
  const savedToken = process.env.RIFT_INGEST_TOKEN;
  process.env.RIFT_INGEST_TOKEN = "smoke-dummy-ingest-token-0123456789abcdef";
  try {
    const unauthorized = await resultsHandler.fetch(
      new Request("https://dashboard.example/api/results", {
        method: "POST",
        body: "[]",
        headers: { Authorization: "Bearer wrong" },
      }),
    );
    assert.equal(unauthorized.status, 401);
  } finally {
    if (savedToken === undefined) delete process.env.RIFT_INGEST_TOKEN;
    else process.env.RIFT_INGEST_TOKEN = savedToken;
  }
}

// ---------------------------------------------------------------------------
// api/analyze.mjs e api/models.mjs (inalterados)
// ---------------------------------------------------------------------------

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
// Valores EXATOS com os pesos §25 (coverage 100% → fator 1): CASCADE
// (99.8×25 + 98×10 + 75×10 + 70×30 + 90×20 + 100×5)/100 = 86.25 vence RIFT
// (99.75×25 + 97×10 + 80×10 + 60×30 + 80×20 + 100×5)/100 = 81.6375 → 81.64.
assert.equal(ranking[0].technology, "CASCADE", "buildRanking: CASCADE deveria liderar (pesos §25)");
assert.equal(ranking[0].score, 86.25, "buildRanking: score do CASCADE != 86.25 (pesos §25)");
assert.equal(ranking[0].raw_score, 86.25, "buildRanking: raw_score do CASCADE != 86.25 (pesos §25)");
assert.equal(ranking[0].coverage_pct, 100);
assert.equal(ranking[1].technology, "RIFT", "buildRanking: RIFT deveria ser o 2º (pesos §25)");
assert.equal(ranking[1].score, 81.64, "buildRanking: score do RIFT != 81.64 (pesos §25)");
assert.equal(ranking[1].raw_score, 81.64, "buildRanking: raw_score do RIFT != 81.64 (pesos §25)");
assert.equal(ranking[1].coverage_pct, 100);
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

// ---------------------------------------------------------------------------
// api/test.mjs — launcher M0 (inalterado) + série C3 + override ?arch=
// ---------------------------------------------------------------------------

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
// Árvore canônica (§20): o SCRIPT_URL do launcher M0 embute engines/<tech>/.
assert.ok(
  launcher.includes("engines/cascade/cascade_m0_phase1_test_v030_auto_batteries.py"),
  "launcher M0 cascade sem o caminho engines/cascade/ no SCRIPT_URL (§20)",
);
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

// Override de arquitetura do WINNER (?arch=) — injeta RIFT_WINNER_ARCH.
assert.equal(launcherApi.normalizeWinnerArch("rift"), "RIFT");
assert.equal(launcherApi.normalizeWinnerArch(""), null);
assert.throws(() => launcherApi.normalizeWinnerArch("bogus"));
const winnerLauncher = launcherApi.buildLauncher({
  technology: "winner",
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  winnerArch: "RIFT",
  ref: "main",
});
assert.ok(winnerLauncher.includes('os.environ["RIFT_WINNER_ARCH"] = "RIFT"'));
assert.ok(
  winnerLauncher.includes("engines/winner/winner_m0_phase1_test_v080_auto_batteries.py"),
  "launcher M0 winner sem o caminho engines/winner/ no SCRIPT_URL (§20)",
);
const archRouteResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?technology=winner&model=Qwen%2FQwen2.5-0.5B&arch=spectra",
));
assert.equal(archRouteResponse.status, 200);
assert.ok((await archRouteResponse.text()).includes('os.environ["RIFT_WINNER_ARCH"] = "SPECTRA"'));

// Série C3 (metodologia 16 passos).
assert.equal(launcherApi.C3_BENCHMARK_PROTOCOL, "C3_METHODOLOGY_V1");
// §20 regra 2: o nome LOCAL no Colab não muda; o caminho no REPO agora vive em
// batteries/ e o pacote em core/cascade/ (pares repo_path → local_path).
assert.equal(launcherApi.C3_SCRIPT, "c3_methodology_auto_batteries.py");
assert.equal(launcherApi.C3_SCRIPT_REPO_PATH, "batteries/c3_methodology_auto_batteries.py");
assert.equal(launcherApi.CAP_SCRIPT_REPO_PATH, "batteries/capability_eval_auto_batteries.py");
assert.equal(launcherApi.GGUF_SCRIPT_REPO_PATH, "batteries/gguf_e2e_auto_batteries.py");
assert.equal(launcherApi.FINAL_SCRIPT_REPO_PATH, "batteries/final_phase_auto_batteries.py");
// 21 pares (a duplicata cascade-model-converter/ foi eliminada — §20 regra 1;
// o conversor tem cópia única em core/cascade/converter/).
assert.equal(launcherApi.C3_PACKAGE_FILES.length, 21);
assert.ok(launcherApi.C3_PACKAGE_FILES.every(
  (pair) => Array.isArray(pair) && pair.length === 2,
), "C3_PACKAGE_FILES precisa ser lista de pares [repo_path, local_path] (§20)");
assert.ok(launcherApi.C3_PACKAGE_FILES.every(
  ([repoPath]) => repoPath.startsWith("core/cascade/"),
), "todo repo_path do pacote C3 precisa viver em core/cascade/ (§20)");
assert.ok(launcherApi.C3_PACKAGE_FILES.every(
  ([, localPath]) => localPath.startsWith("cascade/"),
), "todo local_path do pacote C3 continua no layout Colab cascade/ (§20 regra 2)");
for (const [repoPath, localPath] of [
  ["core/cascade/runtime/cleanup.py", "cascade/runtime/cleanup.py"],
  ["core/cascade/runtime/cpp/mmap_smoke.cpp", "cascade/runtime/cpp/mmap_smoke.cpp"],
  ["core/cascade/runtime/cpp/mmap_bundle.hpp", "cascade/runtime/cpp/mmap_bundle.hpp"],
  ["core/cascade/compiler/decompose.py", "cascade/compiler/decompose.py"],
  ["core/cascade/converter/cascade_converter.py", "cascade/converter/cascade_converter.py"],
  ["core/cascade/converter/convert_model.sh", "cascade/converter/convert_model.sh"],
  ["core/cascade/converter/requirements.txt", "cascade/converter/requirements.txt"],
]) {
  assert.ok(
    launcherApi.C3_PACKAGE_FILES.some(([r, l]) => r === repoPath && l === localPath),
    `C3_PACKAGE_FILES sem o par ${repoPath} -> ${localPath} (§20)`,
  );
}
assert.equal(launcherApi.normalizeBattery("c3"), "c3");
assert.equal(launcherApi.normalizeBattery(""), null);
assert.throws(() => launcherApi.normalizeBattery("c4"));

const c3Launcher = launcherApi.buildC3Launcher({
  technology: "rift",
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  ref: "main",
});
for (const marker of [
  "C3_METHODOLOGY_V1",
  "c3_methodology_auto_batteries.py",
  "cascade/runtime/cleanup.py",
  // §20: download por pares (repo_path → local_path); o script vem de
  // batteries/ e o pacote de core/cascade/, mantendo o layout local do Colab.
  'SCRIPT_REPO_PATH = "batteries/c3_methodology_auto_batteries.py"',
  "download(SCRIPT_REPO_PATH, SCRIPT_NAME)",
  "for repo_path, local_path in PACKAGE_FILES:",
  "core/cascade/runtime/cleanup.py",
  "core/cascade/compiler/decompose.py",
  '"--technology", TECHNOLOGY, "--model", MODEL_ID',
  "[C3]",
  "enforce_publish_settings",
  "enforce_transport_security",
  "RIFT_INGEST_TOKEN",
  "https://raw.githubusercontent.com/programador-powershell/RIFT-LM/main",
]) {
  assert.ok(c3Launcher.includes(marker), `launcher C3 sem marcador: ${marker}`);
}
assert.throws(
  () => launcherApi.buildC3Launcher({ technology: "winner", model: launcherModel, origin: "https://x" }),
  /A série C3 aceita apenas rift, aether, cascade ou spectra/,
);

const c3Response = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=c3&technology=cascade&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(c3Response.status, 200);
assert.match(c3Response.headers.get("content-type"), /text\/x-python/);
assert.match(c3Response.headers.get("content-disposition"), /c3-cascade-Qwen-Qwen2\.5-0\.5B\.py/);

const c3WinnerResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=c3&technology=winner&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(c3WinnerResponse.status, 400);
assert.match(
  (await c3WinnerResponse.json()).error,
  /A série C3 aceita apenas rift, aether, cascade ou spectra/,
);

// ---------------------------------------------------------------------------
// api/test.mjs — bateria de capacidades (§9): battery=cap + limpeza prévia
// ---------------------------------------------------------------------------

assert.equal(launcherApi.CAP_BENCHMARK_PROTOCOL, "CAPABILITY_PROBE_V1");
assert.equal(launcherApi.CAP_SCRIPT, "capability_eval_auto_batteries.py");
assert.deepEqual(launcherApi.COLAB_PRECLEAN_PATHS, [
  "/content/geyser_m0_test_output",
  "/content/cap_test_output",
  "/content/gguf_test_output",
  "/content/final_test_output",
  "/content/microlm_m0_test_output",
]);
assert.equal(launcherApi.normalizeBattery("cap"), "cap");
assert.throws(
  () => launcherApi.normalizeBattery("c4"),
  /use c3, cap ou omita o parâmetro/,
);
assert.throws(
  () => launcherApi.normalizeBattery("c4"),
  /final dispara a fase final FINAL_PHASE_V1/,
);

const capLauncher = launcherApi.buildCapLauncher({
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  publish: "required",
  ref: "main",
});
for (const marker of [
  "[CAP]",
  "CAPABILITY_PROBE_V1",
  // §20: a bateria de capacidades vive em batteries/ no repositório.
  "batteries/capability_eval_auto_batteries.py",
  '"--model", MODEL_ID',
  "enforce_publish_settings",
  "enforce_transport_security",
  "preclean_workspace",
  "PRECLEAN_PATHS",
  "/content/geyser_m0_test_output",
  "/content/cap_test_output",
  "/content/gguf_test_output",
  "RIFT_INGEST_TOKEN",
  "32",
  "https://raw.githubusercontent.com/programador-powershell/RIFT-LM/main",
]) {
  assert.ok(capLauncher.includes(marker), `launcher CAP sem marcador: ${marker}`);
}

const capResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=cap&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(capResponse.status, 200);
assert.match(capResponse.headers.get("content-type"), /text\/x-python/);
assert.match(capResponse.headers.get("content-disposition"), /cap-Qwen-Qwen2\.5-0\.5B\.py/);

// capability_eval_auto_batteries.py — contrato §9 (ids imutáveis + protocolo)
// + backend llamacpp opcional (CAP-sobre-GGUF, §11) + probes de benchmark
// agêntico (§15): 4 battery_ids novos, ordem fixa CATEGORY_ORDER e rotulagem
// honesta "probe leve inspirado em <benchmark> — NÃO é o benchmark oficial".
for (const marker of [
  "CAPABILITY_PROBE_V1",
  "CAP_INTELLIGENCE",
  "CAP_CODING",
  "CAP_AGENTIC",
  "CAP_DEEPSEARCH_QA",
  "CAP_MCP_ATLAS",
  "CAP_TAU3_BENCH",
  "CAP_SWE_BENCH",
  "probe leve inspirado",
  "CATEGORY_ORDER",
  "RIFT_INGEST_TOKEN",
  '"--backend"',
  '"--server-url"',
  "llamacpp_generate",
  '"--model-id-label"',
]) {
  assert.ok(capabilityPy.includes(marker), `capability_eval_auto_batteries.py sem marcador: ${marker}`);
}

// ---------------------------------------------------------------------------
// api/test.mjs — bateria GGUF (§11): battery=gguf, quant validado, célula com
// llama.cpp pinado + probe CAP via llama-server
// ---------------------------------------------------------------------------

assert.equal(launcherApi.GGUF_BENCHMARK_PROTOCOL, "GGUF_RUNTIME_V1");
assert.equal(launcherApi.GGUF_SCRIPT, "gguf_e2e_auto_batteries.py");
assert.equal(launcherApi.GGUF_DEFAULT_QUANT, "UD-Q2_K_XL");
assert.equal(launcherApi.GGUF_OUTPUT_DIR, "/content/gguf_test_output");
assert.equal(launcherApi.normalizeBattery("gguf"), "gguf");
assert.equal(launcherApi.normalizeQuant(""), "UD-Q2_K_XL");
assert.equal(launcherApi.normalizeQuant(null), "UD-Q2_K_XL");
assert.equal(launcherApi.normalizeQuant("Q4_K_M"), "Q4_K_M");
assert.throws(() => launcherApi.normalizeQuant("!"), /quant inválido/);
assert.throws(() => launcherApi.normalizeQuant("x"), /quant inválido/);
assert.throws(() => launcherApi.normalizeQuant("a".repeat(33)), /quant inválido/);

const ggufLauncher = launcherApi.buildGgufLauncher({
  model: launcherApi.normalizeModel("unsloth/Muse-Glimmer-30B-GGUF"),
  origin: "https://rift-lm.vercel.app",
  quant: launcherApi.normalizeQuant(null),
  publish: "required",
  ref: "main",
});
for (const marker of [
  "[GGUF]",
  "GGUF_RUNTIME_V1",
  "gguf_e2e_auto_batteries.py",
  "capability_eval_auto_batteries.py",
  // §20: downloads por par (repo_path → nome local inalterado no Colab).
  'GGUF_SCRIPT_REPO_PATH = "batteries/gguf_e2e_auto_batteries.py"',
  'CAP_SCRIPT_REPO_PATH = "batteries/capability_eval_auto_batteries.py"',
  "download(GGUF_SCRIPT_REPO_PATH, GGUF_SCRIPT)",
  "download(CAP_SCRIPT_REPO_PATH, CAP_SCRIPT)",
  '"--model", MODEL_ID, "--quant", QUANT',
  '"--backend", "llamacpp", "--server-url", CAP_SERVER_URL',
  "http://127.0.0.1:8081",
  "/content/gguf_test_output",
  "UD-Q2_K_XL",
  "import_colab_secrets",
  "enforce_transport_security",
  "enforce_publish_settings",
  "preclean_workspace",
  "RIFT_INGEST_TOKEN",
  "32",
  // O download do binário llama.cpp pinado + gguf fica dentro da bateria.
  "acontece DENTRO de gguf_e2e_auto_batteries.py",
]) {
  assert.ok(ggufLauncher.includes(marker), `launcher GGUF sem marcador: ${marker}`);
}

const ggufResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=gguf&model=unsloth%2FMuse-Glimmer-30B-GGUF",
));
assert.equal(ggufResponse.status, 200);
assert.match(ggufResponse.headers.get("content-type"), /text\/x-python/);
assert.match(
  ggufResponse.headers.get("content-disposition"),
  /gguf-unsloth-Muse-Glimmer-30B-GGUF-UD-Q2_K_XL\.py/,
);
const ggufQuantResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=gguf&model=unsloth%2FMuse-Glimmer-30B-GGUF&quant=Q4_K_M",
));
assert.equal(ggufQuantResponse.status, 200);
assert.match(
  ggufQuantResponse.headers.get("content-disposition"),
  /gguf-unsloth-Muse-Glimmer-30B-GGUF-Q4_K_M\.py/,
);
const ggufBadQuant = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=gguf&model=unsloth%2FMuse-Glimmer-30B-GGUF&quant=%21",
));
assert.equal(ggufBadQuant.status, 400);
assert.match((await ggufBadQuant.json()).error, /quant inválido/);

// gguf_e2e_auto_batteries.py — contrato §11 (protocolo + ids imutáveis).
for (const marker of [
  "GGUF_RUNTIME_V1",
  "B0_GGUF_RUNTIME_SETUP",
  "P1_GGUF_E2E_TOKS",
  "UD-Q2_K_XL",
]) {
  assert.ok(ggufPy.includes(marker), `gguf_e2e_auto_batteries.py sem marcador: ${marker}`);
}

// ---------------------------------------------------------------------------
// api/test.mjs — fase final C4/C5/C6 (§16): battery=final gera a célula
// FINAL_PHASE_V1 (final_phase_auto_batteries.py + pacote cascade/, preclean
// /content/final_test_output); technology=all percorre as 4 tecnologias.
// ---------------------------------------------------------------------------

assert.equal(launcherApi.FINAL_BENCHMARK_PROTOCOL, "FINAL_PHASE_V1");
assert.equal(launcherApi.FINAL_SCRIPT, "final_phase_auto_batteries.py");
assert.deepEqual(launcherApi.FINAL_ALL_TECHNOLOGIES, ["rift", "aether", "cascade", "spectra"]);
assert.equal(launcherApi.normalizeBattery("final"), "final");

const finalLauncher = launcherApi.buildFinalLauncher({
  technology: "rift",
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  ref: "main",
});
for (const marker of [
  "[FINAL]",
  "FINAL_PHASE_V1",
  "final_phase_auto_batteries.py",
  // §20: script em batteries/ + pacote em core/cascade/, baixados por pares.
  'SCRIPT_REPO_PATH = "batteries/final_phase_auto_batteries.py"',
  "download(SCRIPT_REPO_PATH, SCRIPT_NAME)",
  "for repo_path, local_path in PACKAGE_FILES:",
  "core/cascade/runtime/cpp/mmap_bundle.hpp",
  '"--technology", technology, "--model", MODEL_ID',
  "/content/final_test_output",
  "cascade/runtime/cpp/mmap_bundle.hpp",
  "import_colab_secrets",
  "enforce_compatibility",
  "enforce_transport_security",
  "enforce_publish_settings",
  "preclean_workspace",
  "/content/final_run",
  "RIFT_INGEST_TOKEN",
  "32",
]) {
  assert.ok(finalLauncher.includes(marker), `launcher FINAL sem marcador: ${marker}`);
}
assert.throws(
  () => launcherApi.buildFinalLauncher({ technology: "winner", model: launcherModel, origin: "https://x" }),
  /A fase final aceita apenas rift, aether, cascade ou spectra/,
);
// technology=all → fila serial das 4 tecnologias bakeada na célula.
const finalAllLauncher = launcherApi.buildFinalLauncher({
  technology: "all",
  model: launcherModel,
  origin: "https://rift-lm.vercel.app",
  ref: "main",
});
assert.ok(
  finalAllLauncher.includes('["rift","aether","cascade","spectra"]'),
  "launcher FINAL all sem a fila serial das 4 tecnologias",
);

for (const tech of ["rift", "aether", "cascade", "spectra", "all"]) {
  const finalResponse = await testLauncher.fetch(new Request(
    `https://rift-lm.vercel.app/api/test?battery=final&technology=${tech}&model=Qwen%2FQwen2.5-0.5B`,
  ));
  assert.equal(finalResponse.status, 200, `battery=final&technology=${tech} deveria responder 200`);
  assert.match(finalResponse.headers.get("content-type"), /text\/x-python/);
  assert.match(
    finalResponse.headers.get("content-disposition"),
    new RegExp(`final-${tech}-Qwen-Qwen2\\.5-0\\.5B\\.py`),
  );
}
const finalWinnerResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=final&technology=winner&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(finalWinnerResponse.status, 400);
assert.match(
  (await finalWinnerResponse.json()).error,
  /A fase final aceita apenas rift, aether, cascade ou spectra/,
);

// final_phase_auto_batteries.py — contrato §16 (protocolo, ids imutáveis por
// tecnologia, primária SOMENTE no C6 com e2e medido, pesos originais fora do
// caminho quente e publicação incremental endurecida).
for (const marker of [
  "FINAL_PHASE_V1",
  'f"C4_{tech_upper}_SECOND_FAMILY"',
  'f"C5_{tech_upper}_REPR_BLOCKS"',
  'f"C6_{tech_upper}_COMPILE_EXECUTE"',
  "comparison_role",
  "original_weights_freed",
  "bundle_bytes",
  "--second-model",
  "--large-model",
  "RIFT_RESULTS_ENDPOINT",
  "RIFT_INGEST_TOKEN",
]) {
  assert.ok(finalPy.includes(marker), `final_phase_auto_batteries.py sem marcador: ${marker}`);
}

// ---------------------------------------------------------------------------
// api/test.mjs — bateria MicroLM (§22): battery=microlm SEM parâmetro de
// modelo (o MicroLM É o modelo, referência fixa microlm/MicroLM-22M-v0.2);
// a célula MICROLM_M0_V1 baixa a bateria auto-contida + model.py verbatim
// de engines/microlm/, pinados no repo/ref resolvidos (§14.1).
// ---------------------------------------------------------------------------

assert.equal(launcherApi.MICROLM_BENCHMARK_PROTOCOL, "MICROLM_M0_V1");
assert.equal(launcherApi.MICROLM_SCRIPT, "microlm_m0_auto_batteries.py");
assert.equal(launcherApi.MICROLM_SCRIPT_REPO_PATH, "engines/microlm/microlm_m0_auto_batteries.py");
assert.equal(launcherApi.MICROLM_MODEL_ID, "microlm/MicroLM-22M-v0.2");
assert.deepEqual(launcherApi.MICROLM_MODEL_FILES, [["engines/microlm/model.py", "model.py"]]);
// api/runner.mjs espelha o model_id fixo do passo microlm (§22.3).
assert.equal(runnerQueueApi.MICROLM_MODEL_ID, launcherApi.MICROLM_MODEL_ID);
assert.equal(launcherApi.normalizeBattery("microlm"), "microlm");

const microlmLauncher = launcherApi.buildMicrolmLauncher({
  origin: "https://rift-lm.vercel.app",
  publish: "required",
  ref: "main",
});
for (const marker of [
  "[MICROLM]",
  "MICROLM_M0_V1",
  'SCRIPT_REPO_PATH = "engines/microlm/microlm_m0_auto_batteries.py"',
  "download(SCRIPT_REPO_PATH, SCRIPT_NAME)",
  "for repo_path, local_path in MODEL_FILES:",
  "engines/microlm/model.py",
  "microlm/MicroLM-22M-v0.2",
  "/content/microlm_run",
  "/content/microlm_m0_test_output",
  "import_colab_secrets",
  "enforce_transport_security",
  "enforce_publish_settings",
  "preclean_workspace",
  "RIFT_INGEST_TOKEN",
  "32",
]) {
  assert.ok(microlmLauncher.includes(marker), `launcher MICROLM sem marcador: ${marker}`);
}
// ?publish=off é repassado à bateria como --publish off (§22.3).
const microlmOffLauncher = launcherApi.buildMicrolmLauncher({
  origin: "https://rift-lm.vercel.app",
  publish: "off",
  ref: "main",
});
assert.ok(microlmOffLauncher.includes('PUBLISH_MODE = "off"'), "launcher MICROLM publish=off sem PUBLISH_MODE");
assert.ok(microlmOffLauncher.includes('"--publish", "off"'), "launcher MICROLM publish=off sem repasse --publish off");

// Rota sem modelo = caminho VÁLIDO; ?model= explícito → 400 com mensagem clara.
const microlmResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=microlm",
));
assert.equal(microlmResponse.status, 200);
assert.match(microlmResponse.headers.get("content-type"), /text\/x-python/);
assert.match(microlmResponse.headers.get("content-disposition"), /microlm-m0\.py/);
assert.ok((await microlmResponse.text()).includes("MICROLM_M0_V1"));
const microlmModelResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=microlm&model=Qwen%2FQwen2.5-0.5B",
));
assert.equal(microlmModelResponse.status, 400);
assert.match(
  (await microlmModelResponse.json()).error,
  /não aceita parâmetro de modelo/,
);
const microlmOffResponse = await testLauncher.fetch(new Request(
  "https://rift-lm.vercel.app/api/test?battery=microlm&publish=off",
));
assert.equal(microlmOffResponse.status, 200);
assert.ok((await microlmOffResponse.text()).includes('"--publish", "off"'));

// microlm_m0_auto_batteries.py — contrato §22.2: protocolo, os 5 battery_ids,
// technology MICROLM NUNCA primário e publisher endurecido §5 (HTTPS +
// token >= 32; token jamais logado).
for (const marker of [
  'BENCHMARK_PROTOCOL = "MICROLM_M0_V1"',
  'TECH_UPPER = "MICROLM"',
  'MODEL_ID = "microlm/MicroLM-22M-v0.2"',
  "B0_MICROLM_NOOP_INIT",
  "P1_MICROLM_DECODE_PARITY",
  "P1_MICROLM_DECODE_TOKS",
  "P1_MICROLM_TRAINS_FROM_INIT",
  "P1_MICROLM_UNIT_CHECKS",
  '"comparison_role": None',
  '"eligible_for_primary_ranking": False',
  "RIFT_INGEST_TOKEN",
  "RIFT_RESULTS_ENDPOINT",
  "https://",
  "perf_counter_ns",
  "VmRSS",
]) {
  assert.ok(microlmPy.includes(marker), `microlm_m0_auto_batteries.py sem marcador: ${marker}`);
}
// Arquivos VERBATIM do usuário em engines/microlm/ (§22 — cópias intocadas).
assert.ok(microlmModelPy.includes("class MicroLM"), "engines/microlm/model.py sem class MicroLM (§22)");
assert.ok(microlmModelPy.includes("class AttentionCache"), "engines/microlm/model.py sem class AttentionCache (§22)");
assert.ok(microlmModelPy.includes("def decode_step"), "engines/microlm/model.py sem decode_step (§22)");
assert.ok(microlmTestModelPy.includes("def test_"), "engines/microlm/test_model.py sem a suíte pytest (§22)");
assert.ok(microlmChangesMd.length > 0, "engines/microlm/CHANGES.md vazio (§22)");
assert.ok(microlmDiagramSvg.includes("<svg"), "engines/microlm/diagram.svg sem markup SVG (§22.4)");

// ---------------------------------------------------------------------------
// §26 — Card do Conversor: rewrites (§26.1/§26.2), runner local auto-contido
// GET /converter.py (api/converter.mjs), célula Colab battery=converter
// (api/test.mjs) e card operacional no index.html (§26.3). Segurança §26.4:
// NENHUM token literal em código/URL nos corpos gerados.
// ---------------------------------------------------------------------------

{
  const SMOKE_ORIGIN = "https://rift-lm.vercel.app";

  // §26.1/§26.2 — rewrites na vercel.json: /converter.py (runner local) ANTES
  // do wildcard /converter/:model* (célula Colab), como no dev_server.
  const vercelConfig = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
  const rewriteSources = vercelConfig.rewrites.map((rewrite) => rewrite.source);
  const runnerRewrite = vercelConfig.rewrites.find((rewrite) => rewrite.source === "/converter.py");
  const cellRewrite = vercelConfig.rewrites.find((rewrite) => rewrite.source === "/converter/:model*");
  assert.ok(runnerRewrite, "vercel.json sem o rewrite /converter.py (§26.1)");
  assert.equal(runnerRewrite.destination, "/api/converter", "vercel.json: /converter.py deve apontar para /api/converter (§26.1)");
  assert.ok(cellRewrite, "vercel.json sem o rewrite /converter/:model* (§26.2)");
  assert.equal(
    cellRewrite.destination,
    "/api/test?battery=converter&model=:model*",
    "vercel.json: /converter/:model* deve apontar para battery=converter (§26.2)",
  );
  assert.ok(
    rewriteSources.indexOf("/converter.py") < rewriteSources.indexOf("/converter/:model*"),
    "vercel.json: /converter.py precisa vir ANTES do wildcard /converter/:model* (§26.1)",
  );

  // §26.1 — marcadores do módulo: 4 arquivos do conversor (cópia única §20),
  // snapshot só de pesos+config/tokenizer e defaults pacote-por-pacote.
  assert.equal(converterApi.RUNNER_FILENAME, "cascade-converter-runner.py");
  assert.deepEqual(converterApi.CONVERTER_REPO_FILES, [
    "core/cascade/converter/__init__.py",
    "core/cascade/converter/cascade_converter.py",
    "core/cascade/converter/CASCADE_DIR_FORMAT_v0.1.txt",
    "core/cascade/converter/requirements.txt",
  ]);
  assert.equal(converterApi.CONVERTER_LOCAL_DIR, "cascade/converter");
  assert.deepEqual(converterApi.HF_ALLOW_PATTERNS, ["*.safetensors", "*.json", "tokenizer*", "*.model"]);
  assert.equal(converterApi.DEFAULT_DISK_BUDGET_GB, 75);
  assert.match(converterApi.PIP_INSTALL_HINT, /^pip install 'torch' 'safetensors>=0\.4' 'numpy>=1\.26' 'huggingface_hub>=0\.24,<1'$/);

  // Runner gerado (§26.1): CLI espelhando a CLI REAL do conversor
  // (core/cascade/converter/cascade_converter.py: subcomando convert +
  // --input/--output/--model-id/--group-size/--ranks/--disk-budget-gb/
  // --resume/--delete-source-shards/--publish) e fluxo download → convert →
  // upload. Tokens: lidos SOMENTE do ambiente, nunca impressos.
  const converterRunner = converterApi.buildRunner({ origin: SMOKE_ORIGIN });
  for (const marker of [
    `REPO_BASE = "https://raw.githubusercontent.com/${LEGACY_REPO_FALLBACK}/main"`,
    "add_mutually_exclusive_group(required=True)",
    "parser.set_defaults(resume=True)",
    '"--disk-budget-gb", type=float, default=75',
    '"--publish", choices=["on", "off"], default="off"',
    "core/cascade/converter/cascade_converter.py",
    "snapshot_download(",
    "allow_patterns=HF_ALLOW_PATTERNS",
    "create_repo(repo_id=hf_repo, exist_ok=True)",
    "upload_folder(repo_id=hf_repo",
    "pip install 'torch' 'safetensors>=0.4' 'numpy>=1.26' 'huggingface_hub>=0.24,<1'",
    // argv do subprocesso = CLI real do conversor (subcomando convert).
    '"convert",',
    '"--input", str(input_dir)',
    '"--output", str(output_dir)',
    '"--model-id", args.model',
    '"--group-size", str(args.group_size)',
    '"--ranks", args.ranks',
    '"--delete-source-shards"',
    "RIFT_ALLOW_LOCAL_CLEANUP",
    'env.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)',
    'env.setdefault("RIFT_SOURCE_REF", SOURCE_REF)',
  ]) {
    assert.ok(converterRunner.includes(marker), `runner do conversor sem marcador: ${marker}`);
  }
  // Ordem do fluxo: baixar conversor → baixar modelo → converter → subir.
  const downloadConverterAt = converterRunner.indexOf("converter_script = download_converter(");
  const downloadModelAt = converterRunner.indexOf("input_dir = download_model(");
  const convertAt = converterRunner.indexOf("subprocess.call(command");
  const uploadAt = converterRunner.indexOf("upload_output(args.hf_repo");
  assert.ok(
    downloadConverterAt !== -1 && downloadConverterAt < downloadModelAt
      && downloadModelAt < convertAt && convertAt < uploadAt,
    "runner do conversor fora da ordem download → convert → upload (§26.1)",
  );
  // Segurança (§26.4): nenhum token literal/Bearer no corpo gerado e nenhuma
  // linha de fila (a rota do conversor não usa MODELS/TECHS do §14.3).
  assert.doesNotMatch(converterRunner, /hf_[A-Za-z0-9]{30,}/, "runner do conversor com token HF literal (§26.4)");
  assert.doesNotMatch(converterRunner, /Bearer/, "runner do conversor não fala Bearer — o publish é do conversor (§26.4)");
  assert.doesNotMatch(converterRunner, /^MODELS\s*=|^TECHS\s*=/m, "runner do conversor não usa fila MODELS/TECHS (§14.3)");
  assertCanonicalDownloads(converterRunner, "runner do conversor");
  // Repo-agnóstico (§14.1): com repo/ref explícitos nenhum traço do fallback.
  const renamedConverterRunner = converterApi.buildRunner({
    repo: "someone/llm-battery-test",
    ref: "main",
    origin: "https://llm-battery-test.vercel.app",
  });
  assert.ok(
    renamedConverterRunner.includes('REPO_BASE = "https://raw.githubusercontent.com/someone/llm-battery-test/main"'),
    "runner do conversor não seguiu o repo renomeado (§14.1)",
  );
  assert.ok(
    !renamedConverterRunner.includes(LEGACY_REPO_FALLBACK),
    "runner do conversor com repo explícito não pode conter o fallback legado (§14.1)",
  );

  // Handler HTTP: GET 200 text/x-python com Content-Disposition attachment
  // (botão de download); HEAD = variante sem corpo; POST → 405 Allow: GET, HEAD.
  const converterGet = await converterHandler.fetch(new Request(`${SMOKE_ORIGIN}/converter.py`));
  assert.equal(converterGet.status, 200);
  assert.match(converterGet.headers.get("content-type"), /text\/x-python/);
  assert.match(
    converterGet.headers.get("content-disposition"),
    /^attachment; filename="cascade-converter-runner\.py"$/,
  );
  const converterGetBody = await converterGet.text();
  assert.ok(converterGetBody.includes("add_mutually_exclusive_group(required=True)"));
  assert.doesNotMatch(converterGetBody, /hf_[A-Za-z0-9]{30,}/, "GET /converter.py com token HF literal (§26.4)");
  const converterHead = await converterHandler.fetch(new Request(`${SMOKE_ORIGIN}/converter.py`, { method: "HEAD" }));
  assert.equal(converterHead.status, 200);
  assert.equal((await converterHead.text()).length, 0, "HEAD /converter.py precisa vir sem corpo");
  const converterPost = await converterHandler.fetch(new Request(`${SMOKE_ORIGIN}/converter.py`, { method: "POST" }));
  assert.equal(converterPost.status, 405);
  assert.equal(converterPost.headers.get("allow"), "GET, HEAD");

  // §26.2 — battery=converter (api/test.mjs): constantes, validação de
  // hf_repo/publish e célula Colab que baixa {origin}/converter.py.
  assert.equal(launcherApi.CONVERTER_BENCHMARK_PROTOCOL, "CONVERTER_STATIC_V1");
  assert.equal(launcherApi.CONVERTER_RUNNER_ROUTE, "/converter.py");
  assert.equal(launcherApi.CONVERTER_RUNNER_LOCAL_PATH, "/content/cascade_converter_runner.py");
  assert.deepEqual(launcherApi.CONVERTER_PIP_PACKAGES, [
    "torch",
    "safetensors>=0.4",
    "numpy>=1.26",
    "huggingface_hub>=0.24,<1",
  ]);
  assert.equal(launcherApi.normalizeBattery("converter"), "converter");
  assert.ok(launcherApi.HF_REPO_RE.test("usuario/qwen-cascade"));
  assert.ok(!launcherApi.HF_REPO_RE.test("sem-barra"));
  assert.equal(launcherApi.normalizeHfRepo(""), null);
  assert.equal(launcherApi.normalizeHfRepo(null), null);
  assert.equal(launcherApi.normalizeHfRepo("usuario/qwen-cascade"), "usuario/qwen-cascade");
  assert.throws(() => launcherApi.normalizeHfRepo("inválido"), /hf_repo inválido/);
  assert.equal(launcherApi.normalizeConverterPublish(""), "off");
  assert.equal(launcherApi.normalizeConverterPublish(null), "off");
  assert.equal(launcherApi.normalizeConverterPublish("on"), "on");
  assert.throws(() => launcherApi.normalizeConverterPublish("auto"), /publish precisa ser on ou off/);

  const converterCellLauncher = launcherApi.buildConverterLauncher({
    model: launcherApi.normalizeModel("Qwen/Qwen2.5-0.5B"),
    origin: SMOKE_ORIGIN,
    hfRepo: "usuario/qwen-cascade",
    publish: "on",
    ref: "main",
  });
  for (const marker of [
    'BENCHMARK_PROTOCOL = "CONVERTER_STATIC_V1"',
    'RUNNER_URL = ORIGIN + "/converter.py"',
    '["--model", MODEL_ID, "--output", output_dir]',
    '"--hf-repo", HF_REPO',
    '"--publish", "on"',
    "enforce_hf_token",
    "import_colab_secrets",
    "enforce_compatibility",
    "enforce_transport_security",
    "/content/cascade_converter_runner.py",
    'HF_REPO = "usuario/qwen-cascade"',
    'PUBLISH_MODE = "on"',
    'os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)',
    'os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)',
  ]) {
    assert.ok(converterCellLauncher.includes(marker), `célula do conversor sem marcador: ${marker}`);
  }
  assert.doesNotMatch(converterCellLauncher, /hf_[A-Za-z0-9]{30,}/, "célula do conversor com token HF literal (§26.4)");
  assertCanonicalDownloads(converterCellLauncher, "célula do conversor");
  // A saída /content/<nome>-cascade é o PRODUTO — nunca entra no preclean.
  assert.ok(
    launcherApi.COLAB_PRECLEAN_PATHS.every((precleanPath) => !precleanPath.includes("-cascade")),
    "a saída CASCADE-DIR do conversor não pode entrar em COLAB_PRECLEAN_PATHS (§26)",
  );
  // Sem hf_repo: célula converte sem upload e sem exigência de HF_TOKEN.
  const converterCellNoUpload = launcherApi.buildConverterLauncher({
    model: launcherApi.normalizeModel("Qwen/Qwen2.5-0.5B"),
    origin: SMOKE_ORIGIN,
    hfRepo: null,
    publish: "off",
    ref: "main",
  });
  assert.ok(converterCellNoUpload.includes("HF_REPO = None"));
  assert.ok(converterCellNoUpload.includes('PUBLISH_MODE = "off"'));
  // §29.3 — keep_source=on repassa --keep-source-passthrough; default é off e a
  // guarda em runtime (`if KEEP_SOURCE_MODE == "on":`) precisa continuar lá.
  assert.ok(converterCellNoUpload.includes('KEEP_SOURCE_MODE = "off"'),
    "célula do conversor sem keep_source precisa declarar KEEP_SOURCE_MODE = off (§29.3)");
  assert.ok(converterCellLauncher.includes('if KEEP_SOURCE_MODE == "on":'),
    "célula do conversor precisa da guarda de runtime do keep_source (§29.3)");
  const converterCellKeepSource = launcherApi.buildConverterLauncher({
    model: launcherApi.normalizeModel("Qwen/Qwen2.5-0.5B"),
    origin: SMOKE_ORIGIN,
    hfRepo: null,
    publish: "off",
    keepSource: "on",
    ref: "main",
  });
  assert.ok(converterCellKeepSource.includes('KEEP_SOURCE_MODE = "on"'));
  assert.ok(converterCellKeepSource.includes('ARGS += ["--keep-source-passthrough"]'),
    "keep_source=on precisa repassar --keep-source-passthrough ao conversor (§29.3)");
  assert.equal(launcherApi.normalizeConverterKeepSource(null), "off");
  assert.equal(launcherApi.normalizeConverterKeepSource("ON"), "on");
  assert.throws(() => launcherApi.normalizeConverterKeepSource("auto"),
    /keep_source precisa ser on ou off/);
  // O repasse de --hf-repo é guardado em runtime por `if HF_REPO:` — sem repo
  // de destino a célula NÃO pode bakear um destino literal.
  assert.ok(!converterCellNoUpload.includes('HF_REPO = "'), "célula sem hf_repo não pode bakear destino literal");
  assert.ok(converterCellNoUpload.includes("if HF_REPO:"), "célula sem hf_repo precisa manter a guarda if HF_REPO:");

  // Rotas: 200 com filename converter-<slug>.py; hf_repo/publish inválidos e
  // modelo ausente respondem 400 com mensagem clara.
  const converterRouteOk = await testLauncher.fetch(new Request(
    `${SMOKE_ORIGIN}/api/test?battery=converter&model=Qwen%2FQwen2.5-0.5B&hf_repo=usuario%2Fqwen-cascade&publish=on`,
  ));
  assert.equal(converterRouteOk.status, 200);
  assert.match(converterRouteOk.headers.get("content-type"), /text\/x-python/);
  assert.match(converterRouteOk.headers.get("content-disposition"), /converter-Qwen-Qwen2\.5-0\.5B\.py/);
  const converterRouteBody = await converterRouteOk.text();
  assert.ok(converterRouteBody.includes('HF_REPO = "usuario/qwen-cascade"'));
  assert.doesNotMatch(converterRouteBody, /hf_[A-Za-z0-9]{30,}/, "rota do conversor com token HF literal (§26.4)");
  const converterRouteBadRepo = await testLauncher.fetch(new Request(
    `${SMOKE_ORIGIN}/api/test?battery=converter&model=Qwen%2FQwen2.5-0.5B&hf_repo=sem-barra`,
  ));
  assert.equal(converterRouteBadRepo.status, 400);
  assert.match((await converterRouteBadRepo.json()).error, /hf_repo inválido/);
  const converterRouteBadPublish = await testLauncher.fetch(new Request(
    `${SMOKE_ORIGIN}/api/test?battery=converter&model=Qwen%2FQwen2.5-0.5B&publish=auto`,
  ));
  assert.equal(converterRouteBadPublish.status, 400);
  assert.match((await converterRouteBadPublish.json()).error, /publish precisa ser on ou off/);
  const converterRouteNoModel = await testLauncher.fetch(new Request(
    `${SMOKE_ORIGIN}/api/test?battery=converter`,
  ));
  assert.equal(converterRouteNoModel.status, 400);
  assert.match((await converterRouteNoModel.json()).error, /Modelo inválido/);

  // §26.3 — card operacional no index.html (sempre visível, isento da §24.3):
  // ids, botões lado a lado, link de download e helpers da célula curta.
  for (const marker of [
    'sectionTitle">Conversor de modelos',
    'id="converterCard"',
    'id="converterModelInput"',
    'id="converterHfRepoInput"',
    'id="converterPublishInput"',
    'id="converterKeepSourceInput"',
    'id="converterCopyCellBtn"',
    'id="converterDownloadBtn"',
    'id="converterNote"',
    'href="/converter.py" download',
    "Baixar script (rodar no PC)",
    "Repo de destino no Hugging Face",
    "seu-usuario/&lt;modelo&gt;-cascade",
    "Publicar registro no dashboard",
    "buildConverterCell",
    "copyConverterCell",
    "syncConverterModelField",
    "HF_REPO_PATTERN",
    "--disk-budget-gb 75",
    // §29.3: o controle do keep_source e o repasse na URL da célula curta.
    "Não copiar tensores fora do CASCADE",
    'params.set("keep_source","on")',
  ]) {
    assert.ok(legacyHtml.includes(marker), `index.html sem marcador do card do conversor: ${marker}`);
  }
  // Ordem (§21.2 estendida): Baterias por série → Conversor de modelos →
  // Gráficos de medição (a fixture de ordem existente segue intacta).
  const seriesSectionAt = legacyHtml.indexOf('sectionTitle">Baterias por série');
  const converterSectionAt = legacyHtml.indexOf('sectionTitle">Conversor de modelos');
  const measurementChartsAt = legacyHtml.indexOf('id="measurementCharts"');
  assert.ok(
    seriesSectionAt !== -1 && seriesSectionAt < converterSectionAt && converterSectionAt < measurementChartsAt,
    "index.html: o card do conversor precisa ficar entre Baterias por série e os Gráficos (§26.3)",
  );
  // O botão do conversor NÃO participa dos gatilhos de série (data-run-series
  // continua exatamente A–E + all — assert canônico no bloco do §21.2).
  assert.doesNotMatch(
    legacyHtml,
    /converterCopyCellBtn[^>]*data-run-series/,
    "o botão do conversor não pode usar data-run-series (§26.3)",
  );

  // Fixture da célula curta do card (§26.3, mesmo extractFunction do smoke):
  // GET /converter/<modelo>?hf_repo=...&publish=on|off — sem MODELS/TECHS,
  // sem token literal e com raise SystemExit SOMENTE quando há upload/publish.
  const buildConverterCellFixture = vm.runInNewContext(
    `(${extractFunction(legacyHtml, "buildConverterCell", "index.html")})`,
    { launcherOrigin: SMOKE_ORIGIN, URLSearchParams },
    { filename: "legacy-buildConverterCell.js" },
  );
  const converterCellFull = buildConverterCellFixture("Qwen/Qwen2.5-0.5B", "usuario/qwen-cascade", true);
  for (const marker of [
    "/converter/Qwen/Qwen2.5-0.5B",
    "hf_repo=usuario%2Fqwen-cascade",
    "publish=on",
    "from google.colab import userdata",
  ]) {
    assert.ok(converterCellFull.includes(marker), `célula curta do conversor sem marcador: ${marker}`);
  }
  assert.doesNotMatch(converterCellFull, /hf_[A-Za-z0-9]{20,}/, "célula curta do conversor com token HF literal (§26.4)");
  assert.doesNotMatch(converterCellFull, /Bearer/, "célula curta do conversor não pode citar Bearer (§26.4)");
  assert.doesNotMatch(converterCellFull, /^MODELS\s*=|^TECHS\s*=/m, "célula curta do conversor não usa fila MODELS/TECHS (§14.3)");
  assert.ok(converterCellFull.split("\n").length <= 25, "célula curta do conversor deixou de ser curta (> 25 linhas)");
  const converterCellMinimal = buildConverterCellFixture("Qwen/Qwen2.5-0.5B", "", false);
  assert.ok(converterCellMinimal.includes("publish=off"));
  assert.ok(!converterCellMinimal.includes("hf_repo="), "sem repo de destino a célula não envia hf_repo");
  assert.ok(!converterCellMinimal.includes("raise SystemExit"), "sem upload/publish a célula não exige Secrets");
}

// ---------------------------------------------------------------------------
// §14.2 — TODA célula-launcher gerada exporta o repo/ref resolvidos no
// servidor (RIFT_GITHUB_REPOSITORY / RIFT_SOURCE_REF) para os subprocessos
// Python. setdefault: preserva override manual; o runner.py hard-seta antes.
// ---------------------------------------------------------------------------

for (const [source, name] of [
  [launcher, "M0 cascade"],
  [winnerLauncher, "M0 winner"],
  [c3Launcher, "C3"],
  [capLauncher, "CAP"],
  [ggufLauncher, "GGUF"],
  [finalLauncher, "FINAL"],
  [microlmLauncher, "MICROLM"],
]) {
  for (const marker of [
    'GITHUB_REPOSITORY = "',
    'os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)',
    'os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)',
  ]) {
    assert.ok(source.includes(marker), `launcher ${name} sem export do §14.2: ${marker}`);
  }
}

// ---------------------------------------------------------------------------
// §20 (árvore canônica) — asserts NEGATIVOS sobre TODO corpo gerado: nenhuma
// referência à duplicata eliminada cascade-model-converter/ e nenhuma URL de
// download raw apontando para script na RAIZ do repositório (todo caminho de
// código baixado começa em engines/, batteries/, core/ ou scripts/).
// ---------------------------------------------------------------------------

function assertCanonicalDownloads(body, name) {
  assert.ok(
    !body.includes(BANNED_CONVERTER_DUPLICATE),
    `${name}: referência à duplicata eliminada ${BANNED_CONVERTER_DUPLICATE}/ (§20 regra 1)`,
  );
  // URLs raw literais: o path após <owner>/<repo>/<ref>/ precisa começar na
  // árvore canônica (nunca mais script solto na raiz).
  const rawUrlRe = /https:\/\/raw\.githubusercontent\.com\/[\w.-]+\/[\w.-]+\/[\w.-]+\/([\w./-]+)/g;
  for (const [, repoPath] of body.matchAll(rawUrlRe)) {
    assert.match(
      repoPath,
      CANONICAL_REPO_PATH_RE,
      `${name}: URL de download fora da árvore canônica (§20): ${repoPath}`,
    );
  }
  // Formatos legados de download (pré-§20): listas FLAT de caminhos na raiz
  // (script solto ou pacote cascade/ direto como repo_path) e o download por
  // nome local sem par. Nos corpos novos só existem pares (repo → local).
  for (const legacyMarker of [
    '"cascade_c0_phase1_auto_batteries.py",', // flat list: script da raiz como repo_path
    '"cascade/__init__.py",',                 // flat list: pacote como repo_path (pares terminam em "])
    "download(SCRIPT_NAME)",                  // download sem par repo→local
  ]) {
    assert.ok(
      !body.includes(legacyMarker),
      `${name}: download ainda usa caminho de script na raiz do repo (§20): ${legacyMarker}`,
    );
  }
}
for (const [source, name] of [
  [launcher, "M0 cascade"],
  [winnerLauncher, "M0 winner"],
  [c3Launcher, "C3"],
  [capLauncher, "CAP"],
  [ggufLauncher, "GGUF"],
  [finalLauncher, "FINAL"],
  [microlmLauncher, "MICROLM"],
]) {
  assertCanonicalDownloads(source, `launcher ${name}`);
}

// api/real-test.mjs — launcher REAL_MEASUREMENT_V3 repo-agnóstico: com repo/ref
// explícitos (repo renomeado) NENHUM traço do fallback legado sobra na célula;
// sem envs (limpas no topo do smoke) o default cai no fallback documentado.
{
  const realLauncherParams = {
    technology: "rift",
    model: launcherApi.normalizeModel("Qwen/Qwen2.5-0.5B"),
    targetLayer: "auto",
    device: "cuda",
    publish: "required",
    trustRemoteCode: false,
    iterations: 50,
    warmup: 10,
    origin: "https://llm-battery-test.vercel.app",
  };
  const renamedLauncher = realLauncherApi.buildLauncher({
    ...realLauncherParams,
    repo: "someone/llm-battery-test",
    ref: "main",
  });
  assert.ok(renamedLauncher.includes(
    "https://raw.githubusercontent.com/someone/llm-battery-test/main/scripts/real_benchmark_runner.py",
  ), "launcher real-test não seguiu o repo renomeado (§14.1)");
  assert.ok(renamedLauncher.includes(
    'os.environ.setdefault("RIFT_GITHUB_REPOSITORY", "someone/llm-battery-test")',
  ));
  assert.ok(renamedLauncher.includes('os.environ.setdefault("RIFT_SOURCE_REF", "main")'));
  assert.ok(
    !renamedLauncher.includes(LEGACY_REPO_FALLBACK),
    "launcher real-test com repo explícito não pode conter o fallback legado",
  );
  const fallbackLauncher = realLauncherApi.buildLauncher(realLauncherParams);
  assert.ok(fallbackLauncher.includes(
    `https://raw.githubusercontent.com/${LEGACY_REPO_FALLBACK}/main/scripts/real_benchmark_runner.py`,
  ), "sem envs, o launcher real-test precisa cair no fallback legado + main");
  assertCanonicalDownloads(renamedLauncher, "launcher real-test (repo renomeado)");
  assertCanonicalDownloads(fallbackLauncher, "launcher real-test (fallback)");
}

// ---------------------------------------------------------------------------
// api/geyser.mjs — GET /geyser/:model* serve o launcher com placeholder
// substituído (§7); GET-only (HEAD e POST respondem 405 com Allow: GET)
// ---------------------------------------------------------------------------

assert.equal(geyserLauncherApi.LAUNCHER_FILENAME, "geyser_launcher.py");
// §20: o arquivo vive em engines/geyser/ no repositório (nome servido inalterado).
assert.equal(geyserLauncherApi.LAUNCHER_REPO_PATH, "engines/geyser/geyser_launcher.py");
assert.equal(geyserLauncherApi.MODEL_ID_PLACEHOLDER, "__GEYSER_MODEL_ID__");
{
  const geyserGet = await geyserApi.fetch(new Request(
    "https://rift-lm.vercel.app/api/geyser?model=Qwen%2FQwen2.5-0.5B",
  ));
  assert.equal(geyserGet.status, 200);
  assert.match(geyserGet.headers.get("content-type"), /text\/plain/);
  assert.equal(geyserGet.headers.get("cache-control"), "no-store");
  assert.equal(geyserGet.headers.get("x-content-type-options"), "nosniff");
  assert.match(geyserGet.headers.get("content-disposition"), /inline; filename="geyser_launcher\.py"/);
  const geyserBody = await geyserGet.text();
  assert.ok(!geyserBody.includes("__GEYSER_MODEL_ID__"), "placeholder não substituído no /api/geyser");
  assert.ok(geyserBody.includes("Qwen/Qwen2.5-0.5B"), "modelo da rota ausente no launcher servido");

  for (const method of ["HEAD", "POST"]) {
    const blocked = await geyserApi.fetch(new Request(
      "https://rift-lm.vercel.app/api/geyser?model=Qwen%2FQwen2.5-0.5B",
      { method },
    ));
    assert.equal(blocked.status, 405, `/api/geyser precisa ser GET-only (${method})`);
    assert.equal(blocked.headers.get("allow"), "GET");
  }

  const badModel = await geyserApi.fetch(new Request(
    "https://rift-lm.vercel.app/api/geyser?model=..%2F..%2Fetc",
  ));
  assert.equal(badModel.status, 400);
}

// geyser_launcher.py — contrato de publicação §7 (placeholder + schema v2)
// + promoção do tok/s medido do G3 (Adendo E2E_TOKS_V1 do §12)
// + base científica v0.2.0 do usuário (§18.1: merge, não substituição —
// atualizações de ciência NUNCA removem a camada de records/publicação).
for (const marker of [
  "__GEYSER_MODEL_ID__",
  '"records"',
  "GEYSER_M0_G0_V1",
  "GEYSER_INGEST_TOKEN",
  "G1_GEYSER_ZDC_LUT",
  "G3_GEYSER_BURST",
  "g3_baseline_tok_s",
  "g3_candidate_tok_s",
  "python_reference_wall_clock",
  "is_primary = is_g1 or is_g3",
  // Ciência v0.2.0 (§18.1): draft proxy INT4g32 com disclosure H1, probe do
  // draft INT2 quente, KV KIVI-classe com bits medidos vs assintóticos,
  // tau_by_k e greedy_equivalence gateando a promoção do tok/s do G3.
  "GEYSER-LM v0.2.0",
  "draft_probe_int2_hot",
  "tau_by_k",
  "kv_bits_asymptotic_long_ctx",
  "greedy_equivalence",
]) {
  assert.ok(geyserLauncherPy.includes(marker), `geyser_launcher.py sem marcador: ${marker}`);
}

// ---------------------------------------------------------------------------
// Comparação de gerações (§18.2) — publicador stdlib-only em batteries/ converte o
// relatório cross-tech em registros CMP_<TECH>_GENERATIONS (schema v2) e a
// cópia real data/compare_generations_report.json (base visual dos dashboards)
// precisa parsear e conter as chaves do formato do artefato.
// ---------------------------------------------------------------------------

for (const marker of [
  "COMPARE_GENERATIONS_V1",
  "CMP_",
  "_GENERATIONS",
  "--selftest",
  "schemes_tensor",
  "RIFT_INGEST_TOKEN",
]) {
  assert.ok(comparePublisherPy.includes(marker), `compare_generations_publisher.py sem marcador: ${marker}`);
}

const compareReport = JSON.parse(
  await readFile(new URL("../data/compare_generations_report.json", import.meta.url), "utf8"),
);
assert.ok(
  compareReport && typeof compareReport === "object" && !Array.isArray(compareReport),
  "data/compare_generations_report.json precisa ser um objeto JSON",
);
assert.ok(
  compareReport.schemes_tensor && typeof compareReport.schemes_tensor === "object",
  "data/compare_generations_report.json sem a chave schemes_tensor (detecção de upload §18.2)",
);
assert.ok(
  compareReport.e2e && typeof compareReport.e2e === "object",
  "data/compare_generations_report.json sem o bloco e2e",
);
assert.ok(
  Object.keys(compareReport.e2e).some((key) => key.toUpperCase() === "ORIGINAL"),
  "data/compare_generations_report.json sem a referência medida ORIGINAL no e2e",
);

// ---------------------------------------------------------------------------
// Baterias E2E tok/s MEDIDO (§12 + Adendo E2E_TOKS_V1) — cada tecnologia tem
// sua bateria P1_<TECH>_E2E_TOKS com model.generate real e metrics.e2e; o
// runner de bancada preserva os tok/s medidos dessas baterias.
// ---------------------------------------------------------------------------

const E2E_SCRIPT_FIXTURES = [
  [riftPy, "rift_m0_phase1_test_v035_auto_batteries.py", "P1_RIFT_E2E_TOKS"],
  [aetherPy, "aether_m0_phase1_test_v100_auto_batteries.py", "P1_AETHER_E2E_TOKS"],
  [spectraPy, "SPECTRA_Colab_Test_M0.py", "P1_SPECTRA_E2E_TOKS"],
  [winnerPy, "winner_m0_phase1_test_v080_auto_batteries.py", "P1_WINNER_E2E_TOKS"],
];
for (const [source, filename, batteryId] of E2E_SCRIPT_FIXTURES) {
  for (const marker of [
    batteryId,
    '"e2e"',
    '"measured"',
    "baseline_tok_s",
    "candidate_tok_s",
    // Disclaimer obrigatório: velocidade do runtime Python de referência.
    "não representa kernel",
    // Guard compartilhado da fila serial (mesmo env nas quatro techs).
    "RIFT_E2E_MAX_PARAMS",
  ]) {
    assert.ok(source.includes(marker), `${filename} sem marcador E2E: ${marker}`);
  }
}

// cascade_c2 — bateria e2e patcheia TODAS as nn.Linear dos blocos com o codec
// real (candidato real no C2) e restaura o modelo ao final.
for (const marker of [
  "P1_CASCADE_C2_E2E_TOKS",
  "patch_all_block_linears",
  "_CascadeLinearWithBias",
  "restore_patched_linears",
  "python_reference_model_generate",
]) {
  assert.ok(cascadeC2Py.includes(marker), `cascade_c2_e2e_auto_batteries.py sem marcador: ${marker}`);
}

// scripts/real_benchmark_runner.py — passthrough: baterias *_E2E_TOKS com
// metrics.e2e.measured=true preservam baseline/candidate_tok_s medidos.
for (const marker of [
  "_E2E_TOKS",
  "E2E_TOKS_V1",
  'battery_id.endswith("_E2E_TOKS")',
  "REAL_MEASUREMENT_V3 + E2E_TOKS_V1",
]) {
  assert.ok(runnerPy.includes(marker), `real_benchmark_runner.py sem marcador: ${marker}`);
}

// ---------------------------------------------------------------------------
// vercel.json — rewrites (incl. /c3/) e cabeçalhos de segurança (CSP + HSTS)
// ---------------------------------------------------------------------------

const vercelConfig = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
// §24.1 — painel ÚNICO: "/" serve index.html; as rotas /v2 e /legacy foram
// EXTINTAS junto com dashboard.html (nenhum rewrite pode apontar para ele).
assert.ok(
  vercelConfig.rewrites.some((r) => r.source === "/" && r.destination === "/index.html"),
  'vercel.json sem rewrite "/" -> /index.html (painel único, §24.1)',
);
assert.ok(
  !vercelConfig.rewrites.some((r) => r.source === "/v2"),
  "vercel.json: a rota /v2 deveria ter sido extinta (§24.1)",
);
assert.ok(
  !vercelConfig.rewrites.some((r) => r.source === "/legacy"),
  "vercel.json: a rota /legacy deveria ter sido extinta (§24.1)",
);
assert.ok(
  !vercelConfig.rewrites.some((r) => String(r.destination || "").includes("dashboard.html")),
  "vercel.json: nenhum rewrite pode apontar para dashboard.html (arquivo deletado, §24.1)",
);
// scripts/dev_server.mjs espelha o mesmo destino de página (paridade §24.1):
// "/" serve index.html e NENHUM traço de dashboard.html//v2//legacy sobra.
const devServerSource = await readFile(new URL("./dev_server.mjs", import.meta.url), "utf8");
assert.match(
  devServerSource,
  /pathname === "\/"\s*\?\s*"index\.html"/,
  'dev_server.mjs: "/" precisa servir index.html (§24.1)',
);
assert.doesNotMatch(
  devServerSource,
  /dashboard\.html/,
  "dev_server.mjs: nenhuma referência a dashboard.html pode sobrar (§24.1)",
);
assert.doesNotMatch(
  devServerSource,
  /"\/v2"|"\/legacy"/,
  "dev_server.mjs: as rotas /v2 e /legacy foram extintas (§24.1)",
);
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/runner.py" && r.destination === "/api/runner",
), "vercel.json sem rewrite /runner.py -> /api/runner (§14.3)");
assert.ok(vercelConfig.rewrites.some((r) => r.source === "/cascade/:model*"));
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/c3/:tech/:model*" && /\/api\/test\?battery=c3/.test(r.destination),
), "vercel.json sem rewrite /c3/:tech/:model* -> /api/test?battery=c3");
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/geyser/:model*" && /\/api\/geyser\?model=/.test(r.destination),
), "vercel.json sem rewrite /geyser/:model* -> /api/geyser");
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/cap/:model*" && /\/api\/test\?battery=cap/.test(r.destination),
), "vercel.json sem rewrite /cap/:model* -> /api/test?battery=cap");
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/gguf/:model*" && /\/api\/test\?battery=gguf/.test(r.destination),
), "vercel.json sem rewrite /gguf/:model* -> /api/test?battery=gguf");
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/final/:tech/:model*" && /\/api\/test\?battery=final/.test(r.destination),
), "vercel.json sem rewrite /final/:tech/:model* -> /api/test?battery=final (§16)");
assert.ok(vercelConfig.rewrites.some(
  (r) => r.source === "/microlm" && r.destination === "/api/test?battery=microlm",
), "vercel.json sem rewrite /microlm -> /api/test?battery=microlm SEM parâmetro de modelo (§22.3)");
// O .vercelignore exclui *.py do upload, então functions/includeFiles NÃO pode
// ser usado (o build da Vercel falha com "pattern doesn't match any Serverless
// Functions" quando o alvo não existe no deployment). api/geyser.mjs serve o
// launcher via fallback GitHub raw no repo/ref resolvidos (§14.1).
assert.equal(
  vercelConfig.functions,
  undefined,
  "vercel.json não pode ter bloco functions (o .vercelignore remove *.py; o launcher vem do GitHub raw)",
);

// §28: orçamento de RAM pela máquina + tabela por largura de bits + card.
const converterSource = await readFile(new URL("../core/cascade/converter/cascade_converter.py", import.meta.url), "utf8");
for (const marker of [
  "detect_total_ram_bytes", "auto_target_ram_bytes", "OS_RESERVE_BYTES",
  "memory_by_bits", "REPORT_BIT_WIDTHS", "GGUFSource", "LADDERS",
  "INT2_GROUP_ASYMMETRIC_MINMAX", "residency_report",
  "RAM NECESSÁRIA POR LARGURA DE BITS", '"converter": {',
]) {
  assert.ok(converterSource.includes(marker), `cascade_converter.py sem marcador §27/§28: ${marker}`);
}
assert.ok(/--target-ram-gb", type=float, default=0.0/.test(converterSource),
  "--target-ram-gb precisa ter default 0 (auto pela RAM da máquina, §28)");
assert.ok(/--ram-budget-mb", type=float, default=0.0/.test(converterSource),
  "--ram-budget-mb precisa ter default 0 (auto pela RAM da máquina, §28)");
for (const marker of [
  "renderConvertedModels", "CONVERTER_BATTERY_ID", "CASCADE_MODEL_CONVERSION",
  "convertedModels", "bitsTable", "converterRows", "Modelos convertidos",
  "memory_by_bits", "noFitFill",
]) {
  assert.ok(legacyHtml.includes(marker), `index.html sem marcador do card de convertidos (§28): ${marker}`);
}
assert.ok(legacyHtml.includes("renderMeasurementCharts();renderConvertedModels();"),
  "renderConvertedModels precisa estar no pipeline de render (§28)");

// §29: decisões auditáveis por tensor (escada auto, guarda de bytes,
// passthrough sem cópia, piso de energia do F1 derivado do gate).
for (const marker of [
  "LOW_BIT_SOURCE_RE", "source_is_low_bit", "resolve_ladder_mode",
  "projected_f0_bytes", "byte_expansion_guard", "projected_byte_expansion",
  "byte_expansion_with_f1", "write_external_stage", "SOURCE_EXTERNAL",
  "external_bytes", "requires_source_file", "bundle_requires_source",
  "required_capture_fraction", "F1_ENERGY_SAFETY", "residual_not_low_rank",
  "below_gate_requirement", "below_explicit_floor", "captured_fraction",
  "required_fraction",
  // Com auto a escada é por tensor: o relatório precisa dizer o que foi USADO.
  "codec_ladder_resolved", "dominant_ladder_mode",
  // Despacho de formato por magic bytes: o resolve() do --input segue o symlink
  // do cache do HF até um blob SEM extensão (§29.8).
  "sniff_container", 'head == b"GGUF"',
]) {
  assert.ok(converterSource.includes(marker), `cascade_converter.py sem marcador §29: ${marker}`);
}
assert.ok(/--codec-ladder", choices=sorted\(set\(LADDERS\) \| \{"auto"\}\), default="auto"/.test(converterSource),
  "--codec-ladder precisa ter default auto (escolhe a escada pela fonte, §29.1)");
assert.ok(/--f1-min-energy", type=float, default=0\.0/.test(converterSource),
  "--f1-min-energy precisa ter default 0 (o piso padrão vem do gate, §29.4)");
for (const flag of ["--allow-byte-expansion", "--keep-source-passthrough"]) {
  assert.ok(converterSource.includes(`"${flag}", action="store_true"`),
    `${flag} precisa existir como flag opt-in (§29.2/§29.3)`);
}
// A guarda e o abort não podem afrouxar o gate: quality_pass segue exigindo as
// duas métricas e o fallback continua sendo passthrough exato (§29.5).
assert.ok(/return m\["cosine"\] >= cosine_min and m\["nrmse"\] <= nrmse_max/.test(converterSource),
  "quality_pass não pode ser afrouxado pelas otimizações de custo (§29.5)");
// §29.8 — make_source NÃO pode voltar a despachar só pelo sufixo: o magic byte
// vem primeiro e o sufixo é apenas fallback.
assert.ok(/kind = sniff_container\(input_path\) or input_path\.suffix/.test(converterSource),
  "make_source precisa despachar por magic bytes antes do sufixo (§29.8)");

// §30 — Confidence Gate v1: o threshold NÃO pode voltar a sair do percentil do
// próprio batch (com batch 1 a máscara vira `score >= score`, sempre True).
const gateSource = await readFile(new URL("../core/cascade/runtime/confidence_gate.py", import.meta.url), "utf8");
for (const marker of [
  "GateCalibrator", "fixed_threshold", "min_batch_for_batch_percentile",
  "UNCALIBRATED_SMALL_BATCH_F0_ONLY", "ACTIVATION_SCORE_V1_CALIBRATED",
  "threshold_source", "explicit_argument", "fixed_calibrated",
  "batch_percentile_prefill", "warning",
]) {
  assert.ok(gateSource.includes(marker), `confidence_gate.py sem marcador §30.1: ${marker}`);
}
assert.ok(gateSource.includes('"features"') || gateSource.includes('"features":'),
  "o meta do gate precisa manter `features` (compat com o v0)");
assert.ok(/int\(x\.shape\[0\]\) >= int\(cfg\.min_batch_for_batch_percentile\)/.test(gateSource),
  "o percentil do batch só vale a partir de min_batch_for_batch_percentile (§30.1)");
assert.ok(!/if threshold is None:\s*\n\s*pct = min/.test(gateSource),
  "REGRESSÃO §30.1: o gate voltou a cair direto no percentil do batch sem guarda");
// §30.3 — o kernel AVX2 antigo fica anotado, sem o loop morto.
const avx2Source = await readFile(new URL("../core/cascade/runtime/cpp/avx2_lowrank.cpp", import.meta.url), "utf8");
assert.ok(avx2Source.includes("SUPERSEDIDO"),
  "avx2_lowrank.cpp precisa declarar que foi supersedido pelo kernel do v2 (§30.3)");
assert.ok(!avx2Source.includes("const float* Vr = V + static_cast<size_t>(r) * in_f"),
  "REGRESSÃO §30.3: o loop morto voltou ao avx2_lowrank.cpp");
// §30.4 — o motor v2 está na árvore com o kernel e o guia de migração.
for (const relPath of [
  "../core/cascade/runtime_v2/q4k_linear.py",
  "../core/cascade/runtime_v2/kernels/kernels.c",
  "../core/cascade/runtime_v2/MIGRACAO.md",
]) {
  await readFile(new URL(relPath, import.meta.url), "utf8");
}

// §31 — conversor v2: orçamento de máquina sem subtração dupla, política de
// gate declarada e projeção com margem.
const convV2 = await readFile(new URL("../core/cascade/runtime_v2/convert.py", import.meta.url), "utf8");
assert.ok(convV2.includes("MACHINE_TOTAL_GIB = (16, 24, 32, 48)"),
  "MACHINE_TOTAL_GIB precisa ser RAM TOTAL da máquina, não orçamento (§31.5)");
assert.ok(!convV2.includes("MACHINE_CLASSES_GIB"),
  "REGRESSÃO §31.5: a tupla ambígua MACHINE_CLASSES_GIB voltou (subtração dupla)");
assert.ok(/budget = max\(total - 8(?:\.0)?, total \/ 2(?:\.0)?\)/.test(convV2),
  "o orçamento tem de sair da regra do §28 sobre a RAM total (§31.5)");
assert.ok(!/maquina_\{cls \+ 8/.test(convV2),
  "REGRESSÃO §31.5: rótulo de classe voltado a somar 8 sobre um orçamento");
// O rótulo da classe fala do TOTAL da máquina; o orçamento é campo, não sufixo.
assert.ok(/f"maquina_\{total:\.0f\}gb"/.test(convV2),
  "a chave da classe precisa ser maquina_<total>gb (§31.5)");
for (const marker of [
  "orcamento_gib", "folga_gib", "regra_orcamento", "kv_runtime_reserve_gib",
  "all_tensors_passed_gate", "below_gate_tensor_count", "below_gate_tensors",
  "gate_policy", "RESCUE_LAST_RUNG",
]) {
  assert.ok(convV2.includes(marker), `convert.py v2 sem marcador §31: ${marker}`);
}
// Docstring não pode voltar a listar as classes antigas nem afirmar 144 B para
// todo super-bloco (g=64 usa 138).
assert.ok(!convV2.includes("8/16/24/40"),
  "o docstring voltou a listar orçamentos como se fossem classes de máquina (§31.5)");
assert.ok(!convV2.includes("144 B/super-bloco prontos"),
  "o docstring precisa dizer 138 B em g=64 e 144 B em g=32 (§31.1)");
// Anti-regressão versionada junto com o conversor.
const residTest = await readFile(new URL("../core/cascade/runtime_v2/tests/test_residency.py", import.meta.url), "utf8");
assert.ok(residTest.includes('maquina_24gb"]["orcamento_gib"] == 16.0'),
  "test_residency.py precisa travar 24 GB total -> 16 GiB de orçamento (§31.5)");
assert.ok(residTest.includes("14.51"),
  "test_residency.py precisa cobrir o limiar exato do orçamento (§31.5)");
assert.ok(!convV2.includes("6.5 bpw"),
  "o docstring não pode prometer pior caso de 6.5 bpw: LADDER só tem degraus de 4 bits (§31.2)");
// O card precisa expor a MARGEM da projeção, não só o veredito.
for (const marker of ["Folga em 24 GB", "break_even_note", "projRes"]) {
  assert.ok(legacyHtml.includes(marker), `index.html sem a margem da projeção (§31.6): ${marker}`);
}
assert.ok(legacyHtml.includes("typeof projRaw===\"object\"?projRaw.central:projRaw"),
  "a projeção precisa aceitar número OU {best,central,worst} (§31.6)");

// §35 — conversão INTEGRAL da Muse: medição substitui projeção, e o veredito de
// 24 GB caiu. Nada disso pode ser suavizado depois.
const museReal = publishedHistory.find((r) => r.run_id === "cascade-convert-v21-muse30b-integral");
if (museReal) {
  const c = museReal.metrics.converter;
  assert.equal(c.format, "CASCADE-Q4K/2.1");
  assert.equal(c.sample_scope, null, "a conversão integral NÃO é amostra (§35.1)");
  assert.equal(c.integral_conversion.tensors_converted, c.integral_conversion.tensors_total,
    "integral significa 627/627 (§35.1)");
  assert.equal(c.integral_conversion.label, "MEDIDO");
  // O bundle tem de reconciliar com a soma dos degraus declarados.
  const rungs = c.selected_rungs;
  assert.equal(Object.values(rungs).reduce((a, b) => a + b, 0), c.integral_conversion.tensors_total,
    "a contagem por degrau tem de somar o total de tensores (§35.1)");
  // Veredito de 24 GB: NÃO CABE, e o gap é negativo.
  const res = c.residency_measured;
  assert.equal(res.label, "MEDIDO");
  assert.equal(res.verdict_24gb, "NAO CABE",
    "REGRESSÃO §35.2: o veredito de 24 GB da Muse foi medido como NÃO CABE");
  assert.ok(res.verdict_24gb_gap_gib < 0, "o gap do veredito tem de ser negativo (§35.2)");
  assert.equal(museReal.metrics.converter.fits_resident_in_target, false);
  assert.ok(res.classes.maquina_32gb.cabe === true && res.classes.maquina_24gb.cabe === false,
    "o alvo prático passou a ser a classe de 32 GB (§35.2)");
  // RSS de conversão: o ganho de 5.8x não pode ser perdido em edição.
  assert.ok(c.conversion_peak_rss_bytes < 0.6 * 1024 ** 3,
    "o pico de RSS medido foi 0,537 GiB (§35.3)");
  // Embedding: resolvido no conversor, aberto no executor.
  assert.match(c.fp32_embedding_status.converter, /RESOLVIDO/);
  assert.match(c.fp32_embedding_status.executor, /NÃO VERIFICADO/);
  assert.match(c.fp32_embedding_status.ppl_cost_of_quantized_embedding, /NÃO MEDIDO/);
  // Qualidade: gate 627/627 NÃO é certificação.
  assert.equal(museReal.quality.end_to_end_measured, false,
    "a PPL da Muse não foi medida: end_to_end_measured tem de ser false (§35.6)");
  assert.equal(museReal.quality.end_to_end_certified, false);
  assert.equal(c.quality_measured.all_tensors_passed_gate, true);
}

// §35.2 — o registro da projeção precisa apontar para a medição que o superou.
const museProj = publishedHistory.find((r) => r.run_id === "cascade-convert-v2-bf16-muse30b");
if (museProj) {
  const sup = museProj.metrics.converter.projection.superseded_by;
  assert.ok(sup, "a projeção de 15,08 GB precisa declarar que foi superada (§35.2)");
  assert.equal(sup.run_id, "cascade-convert-v21-muse30b-integral");
  assert.ok(sup.measured_bundle_bytes > sup.projected_bundle_bytes,
    "a medição ficou ACIMA da projeção: não inverter o sinal (§35.2)");
  assert.equal(museProj.metrics.converter.projection.residency.verdict_status, "RESOLVIDO_CONTRA");
}
// §35.5 — projeção superada vai para o FIM da lista de cards.
assert.ok(legacyHtml.includes("projeção SUPERADA por medição"),
  "index.html sem o selo de projeção superada (§35.5)");
assert.ok(legacyHtml.includes("const superseded=(r)=>Boolean(r.c&&r.c.projection&&r.c.projection.superseded_by)"),
  "converterRows precisa empurrar projeção superada para o fim (§35.5)");
assert.ok(legacyHtml.includes("tensores · MEDIDO"),
  "index.html sem o selo da conversão integral (§35.5)");

// §33 — o gate de cosseno é PRÉ-FILTRO, não veredito de qualidade.
for (const marker of [
  "PREFILTER_NOT_VERDICT", "GATE_ROLE", "end_to_end_validated",
  "gate_vs_ppl_evidence", "+29.1% de PPL",
]) {
  assert.ok(convV2.includes(marker), `convert.py sem marcador §33.1: ${marker}`);
}
assert.ok(/PRE-FILTRO, nao veredito/.test(convV2),
  "o aviso do gate como pré-filtro tem de ser impresso em toda conversão (§33.1)");

const pplRecord = publishedHistory.find((r) => r.battery_id === "P1_CASCADE_PPL_E2E");
if (pplRecord) {
  const m = pplRecord.metrics;
  assert.equal(pplRecord.benchmark_protocol, "PPL_E2E_V1");
  assert.equal(pplRecord.quality.end_to_end_measured, true,
    "o registro de PPL é a primeira medição end-to-end: precisa declarar isso");
  assert.equal(pplRecord.quality.end_to_end_certified, false,
    "medido não é certificado: a config default degrada 29,1% (§33.1)");
  // O achado central não pode ser diluído.
  assert.equal(m.gate_vs_ppl.all_tensors_passed_gate, true);
  assert.ok(m.gate_vs_ppl.ppl_delta_pct_with_all_passing > 25,
    "o registro precisa preservar que o gate aprovou tudo E a PPL degradou (§33.1)");
  // A soma da decomposição tem de fechar com o delta reportado.
  const d = m.perplexity.decomposition_ppl_points;
  const soma = d.peso_g32_data_free + d.degrau_g64 + d.ativacao_int8_grupo256 + d.cabeca_4_5bpw;
  assert.ok(Math.abs(soma - d.soma) < 1e-9, "a soma da decomposição de PPL não fecha");
  assert.ok(Math.abs((m.perplexity.baseline_bf16 + soma) - m.perplexity.cascade_default) < 1e-9,
    "baseline + decomposição precisa reproduzir a PPL da config default (§33.1)");
  // Duelo: empate declarado quando o efeito não pareado excede a margem.
  const duel = m.matched_class_duel;
  assert.ok(duel.unmatched_variable_known_cost_ppl > duel.margin_ppl,
    "se o efeito não pareado excede a margem, o duelo é empate (§33.2)");
  assert.match(duel.verdict, /EMPATE T[ÉE]CNICO/,
    "o duelo precisa ser registrado como empate, não vitória (§33.2)");
  // §34 — aqui o tok/s de topo é LEGÍTIMO: baseline e candidato correm no mesmo
  // runtime PyTorch, mesma máquina, mediana de 3. Mas tem de bater com a mediana
  // registrada em metrics.runtime, senão é número solto.
  const tk = m.runtime.tok_s;
  assert.equal(pplRecord.baseline_tok_s, tk.pytorch_bf16.median,
    "baseline_tok_s tem de ser a mediana registrada do BF16 (§34)");
  assert.equal(pplRecord.candidate_tok_s, tk.pytorch_cascade_full.median,
    "candidate_tok_s tem de ser a mediana registrada do CASCADE (§34)");
  for (const cfg of Object.values(tk)) {
    if (!cfg.runs) continue;
    const sorted = cfg.runs.slice().sort((a, b) => a - b);
    assert.equal(cfg.median, sorted[Math.floor(sorted.length / 2)],
      "a mediana declarada não é a mediana dos runs (§34)");
  }
  // A ablação tem de ser MEDIDA: cada componente é diferença de duas configs.
  const ladder = m.perplexity.ablation_ladder;
  assert.ok(ladder.length === 5, "a escada de ablação precisa das 5 configs medidas (§34)");
  assert.equal(ladder[0].delta_pct, 0, "a primeira linha da escada é o baseline");
  for (let i = 1; i < ladder.length; i += 1) {
    assert.ok(ladder[i].ppl > ladder[i - 1].ppl,
      "a escada de ablação tem de ser monotônica em PPL (§34)");
  }
  // RSS medido: o resultado é NEGATIVO e não pode ser maquiado.
  const mem = m.memory;
  assert.ok(mem.measured_candidate_rss_bytes > mem.measured_baseline_rss_bytes,
    "o RSS medido do CASCADE é MAIOR que o do baseline — não esconder (§34.2)");
  assert.ok(pplRecord.gains.ram_reduction_pct < 0,
    "ram_reduction_pct tem de ser NEGATIVO: o CASCADE gastou mais RAM (§34.2)");
  // Contabilidade de peso: o total inclui o embedding FP32.
  const wa = m.weight_accounting;
  assert.equal(wa.total_resident_weight_bytes,
    wa.q4k_bytes + wa.fp32_input_embedding_bytes,
    "o peso residente tem de somar q4k + embedding FP32 (§34.3)");
  assert.equal(pplRecord.candidate_disk_bytes, wa.total_resident_weight_bytes,
    "candidate_disk_bytes não pode ser só o bloco q4k (§34.3)");
  assert.ok(wa.ratio_vs_q4_0_x > 1,
    "o registro precisa preservar que o footprint real é MAIOR que o do q4_0 (§34.3)");
}

// §33.3 — a projeção de 24 GB da Muse ficou CONDICIONAL ao g64.
const convRec = publishedHistory.find((r) => r.run_id === "cascade-convert-v2-bf16-muse30b");
if (convRec) {
  const cond = convRec.metrics.converter.projection.residency.new_condition_from_ppl;
  assert.ok(cond, "a projeção de 24 GB precisa carregar a condição vinda da PPL (§33.3)");
  assert.equal(cond.verdict_status, "CONDICIONAL");
  assert.match(cond.consequence, /15,67|15.67/,
    "a condição precisa citar o bundle de 15,67 GB que não cabe (§33.3)");
}

// §29.10 — política Fase-1 nos DOIS esquemas de nome. O padrão HF não casa nada
// no ggml: sem as alternativas token_embd/output.weight/_exps a política era
// silenciosamente inerte em entrada .gguf.
for (const marker of [
  "EMBEDDING_NAME_RE", "OUTPUT_HEAD_NAME_RE", "MOE_NAME_RE",
  "token_embd", "_exps", "embedding_passthrough_phase1",
  "output_head_passthrough_phase1", "moe_passthrough_phase1",
]) {
  assert.ok(converterSource.includes(marker), `cascade_converter.py sem marcador §29.10: ${marker}`);
}
// A cabeça de saída do ggml é ANCORADA no início: `blk.N.attn_output.weight` é
// um linear legítimo e não pode cair na política.
assert.ok(converterSource.includes("|^output(?:\\.weight|\\.bias)?$"),
  "OUTPUT_HEAD_NAME_RE precisa ancorar `output.weight` no início, senão exclui attn_output (§29.10)");
assert.ok(!/"(?:embed_tokens|lm_head)" in lname/.test(converterSource),
  "REGRESSÃO §29.10: eligible_matrix voltou ao teste por substring só-HF");
assert.ok(!converterSource.includes("DEFAULT_EXCLUDE"),
  "DEFAULT_EXCLUDE era código morto com aparência de política oficial (§29.10)");

// §29.9 — footprint honesto: os campos de comparação do registro publicado NÃO
// podem ser bundle-vs-fonte quando parte dos tensores ficou em SOURCE_EXTERNAL
// (medido no Muse-Glimmer-30B: "85,51%" de bundle contra 6,5% reais).
for (const marker of [
  "required_disk_bytes", "required_disk_reduction_pct", "all_in_ram_bytes",
  'headline_metric": "all_in_ram_bytes',
]) {
  assert.ok(converterSource.includes(marker), `cascade_converter.py sem marcador §29.9: ${marker}`);
}
assert.ok(converterSource.includes('"candidate_disk_bytes": required_disk_bytes'),
  "candidate_disk_bytes precisa ser o disco EXIGIDO (bundle + externo), §29.9");
assert.ok(converterSource.includes('"rift_disk_bytes": required_disk_bytes'),
  "o alias legado rift_disk_bytes precisa acompanhar o disco exigido (§29.9)");
assert.ok(converterSource.includes('"estimated_candidate_bytes": all_in_ram_bytes'),
  "estimated_candidate_bytes precisa sair de all_in_ram_bytes, não do bundle (§29.9)");
assert.ok(!/"candidate_disk_bytes": int\(actual_bundle\)/.test(converterSource),
  "REGRESSÃO §29.9: candidate_disk_bytes voltou a ser o tamanho do bundle");
assert.ok(!/ram_reduction_pct[\s\S]{0,120}total_stage_bytes/.test(converterSource),
  "REGRESSÃO §29.9: ram_reduction_pct voltou a sair dos bytes de estágio do bundle");
// Painel: ranking e headline do card por all_in_ram_bytes.
for (const marker of [
  "function allInRam(", "TOTAL em RAM", "Redução real vs fonte",
  "depende da fonte", "Fora do bundle (na fonte)",
]) {
  assert.ok(legacyHtml.includes(marker), `index.html sem marcador do headline do conversor (§29.9): ${marker}`);
}
assert.ok(/return ra-rb;/.test(legacyHtml),
  "converterRows precisa ordenar pelo TOTAL em RAM (menor primeiro), não por data (§29.9)");

// §29.11 — conversão PARCIAL não pode parecer completa, e veredito de
// residência não pode ser afirmado sem alvo declarado.
for (const marker of [
  "sample_scope", "amostra", "PROJETADO", "alvo não declarado",
  "Modelo inteiro", "projection",
]) {
  assert.ok(legacyHtml.includes(marker), `index.html sem marcador da conversão parcial (§29.11): ${marker}`);
}
assert.ok(legacyHtml.includes("const temVeredito=c.fits_resident_in_target!=null&&n(c.target_ram_gb)!=null"),
  "o veredito só pode ser afirmado com fits_resident_in_target E target_ram_gb (§29.11)");
assert.ok(!/c\.fits_resident_in_target\?"cabe em":"não cabe em"\} \$\{esc\(String\(c\.target_ram_gb\)\)\} GB<\/span><\/h3>/.test(legacyHtml),
  "REGRESSÃO §29.11: veredito voltado a renderizar sem guarda (gerava 'não cabe em null GB')");

// O registro do teste BF16 no histórico precisa continuar rotulado como AMOSTRA
// e manter as projeções FORA dos campos de comparação (§28.2).
const bf16Record = publishedHistory.find((r) => String(r.run_id || "").includes("bf16sample"));
if (bf16Record) {
  const conv = bf16Record.metrics?.converter || {};
  assert.match(bf16Record.measurement_scope, /AMOSTRA de 3 das 52 camadas/,
    "o registro BF16 precisa declarar no measurement_scope que é amostra, não o modelo inteiro");
  assert.ok(conv.sample_scope?.layers_measured?.length === 3 && conv.sample_scope?.layers_total === 52,
    "sample_scope do registro BF16 precisa dizer quantas camadas foram medidas");
  assert.equal(conv.projection?.label, "PROJETADO",
    "a extrapolação do modelo inteiro precisa vir rotulada PROJETADO");
  // Os campos de topo referem-se à AMOSTRA: nenhum deles pode carregar a
  // projeção do modelo inteiro (ordem de grandeza 20 GB vs 890 MB).
  assert.ok(bf16Record.candidate_disk_bytes < 1e9,
    "candidate_disk_bytes do registro BF16 precisa ser o da amostra medida, nunca a projeção (§28.2)");
  assert.equal(bf16Record.candidate_disk_bytes, conv.bundle_bytes + conv.external_source_bytes,
    "candidate_disk_bytes precisa ser bundle + externo também neste registro (§29.9)");
  assert.equal(bf16Record.metrics.memory.estimated_candidate_bytes, conv.all_in_ram_bytes,
    "estimated_candidate_bytes precisa ser all_in_ram_bytes também neste registro (§29.9)");
  assert.equal(bf16Record.implementation.eligible_for_primary_ranking, false,
    "registro de conversão nunca é elegível para ranking primário");
  assert.equal(bf16Record.candidate_tok_s, null,
    "o conversor não mede tok/s: o campo de topo tem de ser null");
  assert.equal(bf16Record.candidate_ram_bytes, null,
    "RAM de topo é só RSS de inferência medido; a conversão não mede isso");
  assert.ok(conv.f0_effective_bits_per_weight === null,
    "bpw do F0 não é derivável deste relatório: precisa ser null, não um número inventado");
}

// §30.4 — o registro do motor v2 é SINTÉTICO: tok/s e RAM de topo têm de ser
// null, senão o número de kernel se mistura com o tok/s de model.generate.
const runtimeRecord = publishedHistory.find((r) => r.battery_id === "P1_CASCADE_RUNTIME_V2_KERNEL");
if (runtimeRecord) {
  assert.equal(runtimeRecord.benchmark_protocol, "RUNTIME_KERNEL_SYNTHETIC_V1");
  assert.equal(runtimeRecord.metrics.runtime.scope, "SYNTHETIC_STACK");
  for (const field of [
    "baseline_tok_s", "candidate_tok_s", "baseline_ram_bytes", "candidate_ram_bytes",
  ]) {
    assert.equal(runtimeRecord[field], null,
      `${field} do registro de kernel sintético precisa ser null (§30.4)`);
  }
  assert.equal(runtimeRecord.implementation.eligible_for_primary_ranking, false);
  assert.ok(runtimeRecord.metrics.runtime.paths.v2_kernel_c.tok_s > 0,
    "o tok/s medido do kernel vive em metrics.runtime, não no topo");
  // §31.8 — benchmark com mais de um run precisa publicar a FAIXA, e o campo de
  // ponto único tem de ser a mediana, nunca o melhor run.
  const rtm = runtimeRecord.metrics.runtime;
  assert.ok(rtm.runs >= 3 && Array.isArray(rtm.runs_detail) && rtm.runs_detail.length === rtm.runs,
    "o registro do kernel precisa declarar quantos runs e o detalhe de cada um (§31.8)");
  for (const key of ["tok_s_range", "speedup_vs_legacy_low_mem_range_x", "speedup_vs_legacy_default_range_x"]) {
    const r = rtm[key];
    assert.ok(r && r.min < r.max, `${key} precisa trazer a faixa medida (§31.8)`);
    assert.ok(r.median >= r.min && r.median <= r.max, `${key}.median fora da faixa`);
  }
  const best = Math.max(...rtm.runs_detail.map((d) => d.v2 / d.low_mem));
  assert.ok(rtm.speedup_vs_legacy_low_mem_x < best,
    "REGRESSÃO §31.8: o speedup de ponto único voltou a ser o melhor run em vez da mediana");
  assert.match(rtm.dispersion_note, /dispersão/i,
    "a dispersão entre runs precisa estar declarada no registro (§31.8)");
  assert.equal(runtimeRecord.metrics.gate.legacy_v0_activation_rate_batch1, 1.0,
    "o registro precisa preservar a taxa 1.0 do gate v0 como evidência do bug (§30.1)");
  assert.match(runtimeRecord.measurement_scope, /NÃO é model\.generate/,
    "o escopo precisa dizer explicitamente que não é model.generate");
}
// argparse interpola o help com %-formatting: um "%" literal não escapado faz
// `convert --help` levantar TypeError. Vale para todos os CLIs Python do repo.
const pythonCliFiles = [
  "../core/cascade/converter/cascade_converter.py",
  "../batteries/gguf_e2e_auto_batteries.py",
];
for (const relPath of pythonCliFiles) {
  const src = await readFile(new URL(relPath, import.meta.url), "utf8");
  for (const block of src.match(/add_argument\([\s\S]*?\n {4}\)/g) || []) {
    const code = block.split("\n").filter((line) => !/^\s*#/.test(line)).join("\n");
    const bad = code.replace(/%%/g, "").match(/%(?!\(default\)s)/);
    assert.ok(!bad, `${relPath}: "%" não escapado em help do argparse (use %%):\n${block}`);
  }
}

const converterReadme = await readFile(new URL("../core/cascade/converter/README.md", import.meta.url), "utf8");
for (const marker of [
  "Guarda de expansão de bytes", "--keep-source-passthrough",
  "Piso de energia do F1", "`auto` (padrão)",
]) {
  assert.ok(converterReadme.includes(marker), `README do conversor sem seção §29: ${marker}`);
}
const dirFormatDoc = await readFile(new URL("../core/cascade/converter/CASCADE_DIR_FORMAT_v0.1.txt", import.meta.url), "utf8");
for (const marker of [
  "SOURCE_EXTERNAL", "requires_source_file", "bundle_requires_source",
  "byte_expansion_guard", "f1_spectrum", "memory_by_bits",
]) {
  assert.ok(dirFormatDoc.includes(marker), `CASCADE_DIR_FORMAT sem campo novo (§29): ${marker}`);
}

const geyserApiSource = await readFile(new URL("../api/geyser.mjs", import.meta.url), "utf8");
for (const marker of ["rawBaseUrl(resolveRepo(), resolveRef())", "LAUNCHER_REPO_PATH", "GitHub raw"]) {
  assert.ok(geyserApiSource.includes(marker), `api/geyser.mjs sem fallback GitHub raw: ${marker}`);
}
const dataHeaders = vercelConfig.headers.find((entry) => entry.source === "/data/(.*)");
assert.ok(dataHeaders.headers.some((header) => header.key === "Cache-Control" && /s-maxage=30/.test(header.value)));

const globalHeaders = vercelConfig.headers.find((entry) => entry.source === "/(.*)").headers;
const globalHeaderValue = (key) => globalHeaders.find((header) => header.key === key)?.value || "";
const csp = globalHeaderValue("Content-Security-Policy");
assert.match(csp, /default-src 'self'/);
assert.match(csp, /connect-src 'self' https:\/\/raw\.githubusercontent\.com/);
assert.doesNotMatch(csp, /unsafe-eval/);
assert.match(globalHeaderValue("Strict-Transport-Security"), /max-age=\d+/);
assert.equal(globalHeaderValue("X-Content-Type-Options"), "nosniff");

// ---------------------------------------------------------------------------
// api/_lib/repo.mjs — cadeia repo-agnóstica (§14.1): GITHUB_REPO →
// RIFT_GITHUB_REPOSITORY → VERCEL_GIT_REPO_OWNER/VERCEL_GIT_REPO_SLUG →
// fallback legado; ref: RIFT_GITHUB_BRANCH → VERCEL_GIT_COMMIT_SHA → main.
// ---------------------------------------------------------------------------

{
  assert.equal(repoLib.LEGACY_REPOSITORY, LEGACY_REPO_FALLBACK);
  // Sem env nenhuma: fallback legado + main.
  assert.equal(resolveRepo({}), LEGACY_REPO_FALLBACK);
  assert.equal(resolveRef({}), "main");
  // Precedência da cadeia (§14.1).
  assert.equal(resolveRepo({ GITHUB_REPO: "a/b", RIFT_GITHUB_REPOSITORY: "c/d" }), "a/b");
  assert.equal(resolveRepo({
    RIFT_GITHUB_REPOSITORY: "c/d",
    VERCEL_GIT_REPO_OWNER: "owner",
    VERCEL_GIT_REPO_SLUG: "slug",
  }), "c/d");
  // Cenário do incidente: repo renomeado para llm-battery-test — o env
  // automático da Vercel resolve sem nenhum hardcode.
  assert.equal(resolveRepo({
    VERCEL_GIT_REPO_OWNER: "programador-powershell",
    VERCEL_GIT_REPO_SLUG: "llm-battery-test",
  }), "programador-powershell/llm-battery-test");
  // Normalização: URL https/ssh e sufixo .git viram owner/repo.
  assert.equal(resolveRepo({ GITHUB_REPO: "https://github.com/o/r.git" }), "o/r");
  assert.equal(resolveRepo({ GITHUB_REPO: "git@github.com:o/r.git" }), "o/r");
  // Valor inválido é pulado (nunca lança) → segue para o fallback.
  assert.equal(resolveRepo({ GITHUB_REPO: "sem-barra" }), LEGACY_REPO_FALLBACK);
  const sha = "0123456789abcdef0123456789abcdef01234567";
  assert.equal(resolveRef({ VERCEL_GIT_COMMIT_SHA: sha }), sha);
  assert.equal(resolveRef({ RIFT_GITHUB_BRANCH: "release/v2", VERCEL_GIT_COMMIT_SHA: sha }), "release/v2");
  assert.equal(resolveRef({ RIFT_GITHUB_BRANCH: "../etc", VERCEL_GIT_COMMIT_SHA: "not-a-sha" }), "main");
  assert.equal(rawBaseUrl("o/r", "main"), "https://raw.githubusercontent.com/o/r/main");
  // Marcadores da cadeia no fonte (documentação viva do §14.1).
  const repoLibSource = await readFile(new URL("../api/_lib/repo.mjs", import.meta.url), "utf8");
  for (const marker of [
    "GITHUB_REPO",
    "RIFT_GITHUB_REPOSITORY",
    "VERCEL_GIT_REPO_OWNER",
    "VERCEL_GIT_REPO_SLUG",
    "RIFT_GITHUB_BRANCH",
    "VERCEL_GIT_COMMIT_SHA",
    LEGACY_REPO_FALLBACK,
  ]) {
    assert.ok(repoLibSource.includes(marker), `api/_lib/repo.mjs sem marcador da cadeia §14.1: ${marker}`);
  }
}

// ---------------------------------------------------------------------------
// GET /runner.py → api/runner.mjs (§14.3): orquestrador completo em text/plain
// com origin/repo/ref resolvidos no servidor; GET/HEAD-only.
// ---------------------------------------------------------------------------

{
  assert.deepEqual(
    runnerQueueApi.ALL_TECHNOLOGIES,
    ["rift", "cascade", "aether", "spectra", "winner", "geyser", "microlm", "c-series", "c3", "final", "cap"],
    "expansão de TECHS=[\"all\"] divergente do contrato §13.3/§14.3/§16/§22.3",
  );
  assert.deepEqual(runnerQueueApi.KNOWN_TECHNOLOGIES, [...runnerQueueApi.ALL_TECHNOLOGIES, "gguf"]);
  assert.equal(runnerQueueApi.GGUF_DEFAULT_QUANT, "UD-Q2_K_XL");

  // §20: a série C viaja como pares [repo_path, local_path] — scripts em
  // engines/cascade/ e pacote em core/cascade/, layout local do Colab intacto.
  assert.ok(runnerQueueApi.CSERIES_SCRIPTS.every(
    ([repoPath, localPath]) =>
      repoPath.startsWith("engines/cascade/") && repoPath.endsWith("/" + localPath),
  ), "CSERIES_SCRIPTS precisa ser pares engines/cascade/<script> -> <script> (§20)");
  assert.ok(runnerQueueApi.CSERIES_SCRIPTS.some(
    ([r, l]) => r === "engines/cascade/cascade_c0_phase1_auto_batteries.py" && l === "cascade_c0_phase1_auto_batteries.py",
  ));
  assert.ok(runnerQueueApi.CSERIES_PACKAGE_FILES.every(
    ([repoPath, localPath]) =>
      repoPath.startsWith("core/cascade/") && localPath.startsWith("cascade/"),
  ), "CSERIES_PACKAGE_FILES precisa ser pares core/cascade/... -> cascade/... (§20)");
  assert.ok(runnerQueueApi.CSERIES_PACKAGE_FILES.some(
    ([r, l]) => r === "core/cascade/runtime/cleanup.py" && l === "cascade/runtime/cleanup.py",
  ));

  const runnerGet = await runnerHandler.fetch(new Request("https://rift-lm.vercel.app/runner.py"));
  assert.equal(runnerGet.status, 200);
  assert.match(runnerGet.headers.get("content-type"), /text\/plain/);
  assert.equal(runnerGet.headers.get("x-content-type-options"), "nosniff");
  assert.match(runnerGet.headers.get("content-disposition"), /inline; filename="runner\.py"/);
  assert.match(runnerGet.headers.get("cache-control"), /s-maxage=300/);
  const runnerBody = await runnerGet.text();
  for (const marker of [
    // Origin/repo/ref resolvidos e bakeados no script (envs limpas → fallback).
    'RIFT_RUNNER_ORIGIN = "https://rift-lm.vercel.app"',
    `RIFT_RUNNER_REPO = ${JSON.stringify(LEGACY_REPO_FALLBACK)}`,
    'RIFT_RUNNER_REF = "main"',
    // §14.2: o runner exporta (hard-set) as envs para TODOS os subprocessos.
    'os.environ["RIFT_GITHUB_REPOSITORY"] = RIFT_RUNNER_REPO',
    'os.environ["RIFT_SOURCE_REF"] = RIFT_RUNNER_REF',
    // Globals da célula curta com fallback em env.
    "RIFT_QUEUE_MODELS",
    "RIFT_QUEUE_TECHS",
    // Deps de tokenização pinadas (aceita todo tipo de tokenização).
    "transformers>=4.52.0",
    "sentencepiece>=0.2,<1",
    "tiktoken>=0.7,<1",
    // Passos da fila: C3 all em 1 passo, fase final via /final/all/ (§16),
    // GGUF com quant, GEYSER via curl, série C com scripts + pacote cascade/.
    "/c3/all/",
    "/final/all/",
    "?quant=",
    "geyser_launcher.py",
    "cascade_c0_phase1_auto_batteries.py",
    "cascade/runtime/cleanup.py",
    // §20: pares repo→local bakeados + loops de download/execução por par.
    "engines/cascade/cascade_c0_phase1_auto_batteries.py",
    "core/cascade/runtime/cleanup.py",
    "for repo_path, local_path in RIFT_CSERIES_SCRIPTS + RIFT_CSERIES_PKG_FILES:",
    "for _repo_path, script in RIFT_CSERIES_SCRIPTS:",
    // Passo microlm (§22.3): rota fixa /microlm sem segmento de modelo, com o
    // model_id de referência bakeado para o resumo da fila.
    'RIFT_MICROLM_MODEL_ID = "microlm/MicroLM-22M-v0.2"',
    'if tech == "microlm":',
    'RIFT_QUEUE_BASE + "/microlm"',
    // Limpeza prévia (movida da célula do painel para o runner servidor).
    "/content/geyser_m0_test_output",
    "/content/cap_test_output",
    "/content/gguf_test_output",
    "/content/final_test_output",
    "/content/final_run",
    "/content/microlm_run",
    "/content/microlm_m0_test_output",
    // Espera de liberação de VRAM entre passos + resumo PT-BR.
    "_rift_wait_for_resource_release",
    "Resumo da fila",
    "[FILA]",
    "RIFT_INGEST_TOKEN",
  ]) {
    assert.ok(runnerBody.includes(marker), `runner.py servido sem marcador: ${marker}`);
  }
  // A lista de expansão de "all" viaja bakeada no corpo do script.
  assert.ok(
    runnerBody.includes(JSON.stringify(runnerQueueApi.ALL_TECHNOLOGIES)),
    "runner.py sem a lista de expansão de TECHS=[\"all\"] bakeada",
  );

  // Repo renomeado (motivação do §14): o script gerado segue o novo repo e
  // NÃO carrega nenhum traço do fallback legado.
  const renamedRunner = runnerQueueApi.buildRunnerScript({
    origin: "https://llm-battery-test.vercel.app",
    repo: "someone/llm-battery-test",
    ref: "0123456789abcdef0123456789abcdef01234567",
  });
  assert.ok(renamedRunner.includes('RIFT_RUNNER_REPO = "someone/llm-battery-test"'));
  assert.ok(renamedRunner.includes(
    "https://raw.githubusercontent.com/someone/llm-battery-test/0123456789abcdef0123456789abcdef01234567",
  ));
  assert.ok(!renamedRunner.includes(LEGACY_REPO_FALLBACK), "runner de repo renomeado ainda contém o repo legado");

  // §20: nenhum download do runner fora da árvore canônica nem referência à
  // duplicata eliminada do conversor.
  assertCanonicalDownloads(runnerBody, "runner.py servido");
  assertCanonicalDownloads(renamedRunner, "runner.py (repo renomeado)");

  const runnerHead = await runnerHandler.fetch(
    new Request("https://rift-lm.vercel.app/runner.py", { method: "HEAD" }),
  );
  assert.equal(runnerHead.status, 200);
  assert.equal(await runnerHead.text(), "");
  for (const method of ["POST", "PUT", "DELETE"]) {
    const blocked = await runnerHandler.fetch(
      new Request("https://rift-lm.vercel.app/runner.py", { method, body: method === "DELETE" ? null : "x" }),
    );
    assert.equal(blocked.status, 405, `/api/runner precisa ser GET/HEAD-only (${method})`);
    assert.equal(blocked.headers.get("allow"), "GET, HEAD");
  }
}

// ---------------------------------------------------------------------------
// Varredura anti-hardcode (§14.1/§14.2): o fallback legado aparece SOMENTE em
// api/_lib/repo.mjs dentro de api/, e em cada .py no MÁXIMO uma vez, na forma
// documentada LEGACY_REPOSITORY = "..." ao lado da resolução por env.
// ---------------------------------------------------------------------------

{
  const repoRoot = new URL("..", import.meta.url);
  const allFiles = (await readdir(repoRoot, { recursive: true, withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(entry.parentPath, entry.name))
    .filter((file) => !/[\\/](node_modules|\.git)[\\/]/.test(file));

  const apiFiles = allFiles.filter((file) => /[\\/]api[\\/]/.test(file));
  assert.ok(apiFiles.length >= 8, "varredura de api/ não encontrou os handlers esperados");
  for (const file of apiFiles) {
    const content = await readFile(file, "utf8");
    const isRepoLib = /[\\/]api[\\/]_lib[\\/]repo\.mjs$/.test(file);
    if (isRepoLib) {
      // O fallback legado vive aqui como a constante documentada (comentários
      // que citam a cadeia §14.1 são permitidos).
      assert.match(
        content,
        /const LEGACY_REPOSITORY = "programador-powershell\/RIFT-LM";/,
        "api/_lib/repo.mjs deve declarar a constante LEGACY_REPOSITORY documentada",
      );
      continue;
    }
    assert.ok(
      !content.includes(LEGACY_REPO_FALLBACK),
      `${path.basename(file)}: repo legado hardcoded fora de api/_lib/repo.mjs (§14.1)`,
    );
  }

  // §20 regra 1: a duplicata cascade-model-converter/ foi ELIMINADA — nenhum
  // arquivo pode viver nesse diretório e nenhum código executável/painel/config
  // pode referenciá-lo (menções históricas em comentários .mjs e em docs/ do
  // contrato ficam fora do escopo — não são caminhos de download).
  for (const file of allFiles) {
    assert.ok(
      !file.includes(BANNED_CONVERTER_DUPLICATE),
      `${file}: o diretório ${BANNED_CONVERTER_DUPLICATE}/ deveria ter sido eliminado (§20 regra 1)`,
    );
  }
  const converterScanFiles = allFiles.filter((file) => /\.(py|html|sh|json)$/.test(file));
  assert.ok(converterScanFiles.length >= 12, "varredura §20 não encontrou os arquivos de código esperados");
  for (const file of converterScanFiles) {
    const content = await readFile(file, "utf8");
    assert.ok(
      !content.includes(BANNED_CONVERTER_DUPLICATE),
      `${path.basename(file)}: referência à duplicata eliminada ${BANNED_CONVERTER_DUPLICATE}/ (§20 regra 1)`,
    );
  }

  const pyFiles = allFiles.filter((file) => file.endsWith(".py"));
  assert.ok(pyFiles.length >= 10, "varredura não encontrou os scripts .py esperados");
  for (const file of pyFiles) {
    const content = await readFile(file, "utf8");
    const occurrences = content.split(LEGACY_REPO_FALLBACK).length - 1;
    assert.ok(
      occurrences <= 1,
      `${file}: ${occurrences} ocorrências do repo legado (máximo 1, o fallback documentado do §14.2)`,
    );
    if (occurrences === 1) {
      const line = content.split(/\r?\n/).find((candidate) => candidate.includes(LEGACY_REPO_FALLBACK));
      assert.match(
        line,
        /LEGACY_REPOSITORY\s*=\s*"programador-powershell\/RIFT-LM"/,
        `${file}: ocorrência do repo legado fora do padrão documentado LEGACY_REPOSITORY = "..." (§14.2)`,
      );
      assert.ok(
        content.includes("RIFT_GITHUB_REPOSITORY"),
        `${file}: fallback legado sem a resolução via env RIFT_GITHUB_REPOSITORY ao lado (§14.2)`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// data/record-schema-example.json — contrato preferencial agora é o v2
// ---------------------------------------------------------------------------

const schema = JSON.parse(await readFile(new URL("../data/record-schema-example.json", import.meta.url), "utf8"));
assert.equal(schema.schema_version, 2);
assert.equal(schema.benchmark_protocol, "LINEAR_REAL_MEASURED_V3");
assert.ok(schema.comparison_group_id);
assert.equal(schema.implementation.kind, "REFERENCE_MEASURED|NATIVE_MEASURED|SIMULATED");
assert.equal(schema.eligible_for_primary_ranking, true);
assert.ok(schema.gains && typeof schema.gains === "object", "schema example precisa do objeto gains");
assert.ok(schema.metrics?.memory && typeof schema.metrics.memory === "object", "schema example precisa de metrics.memory");

console.log("dashboard smoke: OK");
