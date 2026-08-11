// Fixtures de batteryGroupKey (contrato docs/C3_CONTRACTS_V1.md §13.1 + §16
// + §18) e de batteryFriendlyName (§19.5, 7º lote). Cobre os casos das regras
// ORDENADAS (a primeira que casar vence; §16 acrescentou as fases finais
// C4/C5/C6 antes do fallback e §18 colocou /^CMP_/ como a PRIMEIRA regra da
// cadeia), os nomes amigáveis PT-BR por nível ('N<nível> · <nome>') e o
// agrupamento sobre os dados reais publicados (data/rift_test_batteries.json).
// Zero dependências novas: node:assert + node:fs + node:vm.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const legacyHtml = await readFile(new URL("../index.html", import.meta.url), "utf8");

function extractFunction(source, name, filename) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${filename} sem definição de function ${name}(`);
  const braceStart = source.indexOf("{", start);
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

const batteryGroupKey = vm.runInNewContext(
  `(${extractFunction(legacyHtml, "batteryGroupKey", "index.html")})`,
  {},
  { filename: "legacy-batteryGroupKey.js" },
);
// inferTechnology depende das constantes do painel — espelha o contexto mínimo.
const inferTechnology = vm.runInNewContext(
  `(${extractFunction(legacyHtml, "inferTechnology", "index.html")})`,
  {
    CAP_TECHNOLOGY: "CAP",
    GGUF_TECHNOLOGY: "GGUF",
    TECHNOLOGIES: ["RIFT", "CASCADE", "AETHER", "SPECTRA", "GEYSER", "WINNER"],
  },
  { filename: "legacy-inferTechnology.js" },
);

// ---------------------------------------------------------------------------
// Os casos das regras ordenadas (primeira que casa vence).
// ---------------------------------------------------------------------------

// Regra 1 (§18, a PRIMEIRA da cadeia ordenada — antes de /^CAP_/): /^CMP_/ →
// 'E2E · comparação de gerações' (comparação de gerações e2e real).
assert.equal(batteryGroupKey("CMP_RIFT_GENERATIONS"), "E2E · comparação de gerações");
assert.equal(batteryGroupKey("CMP_CASCADE_GENERATIONS"), "E2E · comparação de gerações");
assert.equal(batteryGroupKey("CMP_AETHER_GENERATIONS"), "E2E · comparação de gerações");
assert.equal(batteryGroupKey("CMP_GEYSER_GENERATIONS"), "E2E · comparação de gerações");
assert.equal(batteryGroupKey("cmp_aether_generations"), "E2E · comparação de gerações");

// Regra 2: /^CAP_/ → 'Capacidades'
assert.equal(batteryGroupKey("CAP_INTELLIGENCE"), "Capacidades");
assert.equal(batteryGroupKey("CAP_CODING"), "Capacidades");
assert.equal(batteryGroupKey("CAP_AGENTIC"), "Capacidades");
// Case-insensitive (o ingest do painel não força maiúsculas).
assert.equal(batteryGroupKey("cap_coding"), "Capacidades");
assert.equal(batteryGroupKey("p1_rift_e2e_toks"), "E2E · tok/s modelo completo");

// Regra 3: /_E2E_TOKS$/ → 'E2E · tok/s modelo completo' — agrupa o MESMO passo
// entre techs e vem ANTES das regras B0/C3 (C3_*_FULLMODEL_E2E_TOKS cai aqui).
assert.equal(batteryGroupKey("P1_CASCADE_C2_E2E_TOKS"), "E2E · tok/s modelo completo");
assert.equal(batteryGroupKey("P1_RIFT_E2E_TOKS"), "E2E · tok/s modelo completo");
assert.equal(batteryGroupKey("P1_GGUF_E2E_TOKS"), "E2E · tok/s modelo completo");
assert.equal(batteryGroupKey("C3_RIFT_FULLMODEL_E2E_TOKS"), "E2E · tok/s modelo completo");
assert.equal(batteryGroupKey("C3_SPECTRA_FULLMODEL_E2E_TOKS"), "E2E · tok/s modelo completo");

// Regra 4: /^B0_/ → 'B0 · Fundação'
assert.equal(batteryGroupKey("B0_BINARY_IR_FOUNDATION"), "B0 · Fundação");
assert.equal(batteryGroupKey("B0_GEYSER_PHYSICS_BANDWIDTH"), "B0 · Fundação");
assert.equal(batteryGroupKey("B0_GGUF_RUNTIME_SETUP"), "B0 · Fundação");
assert.equal(batteryGroupKey("B0_WINNER_CPP_BUILD_SELF_TEST"), "B0 · Fundação");

// Regra 5: /(_SIM$|_POLICY_SIM$|PREFETCH_SIM$)/ → 'P1 · políticas simuladas'
assert.equal(batteryGroupKey("P1_AETHER_PIO_POLICY_SIM"), "P1 · políticas simuladas");
assert.equal(batteryGroupKey("P1_SPECTRA_PIO_POLICY_SIM"), "P1 · políticas simuladas");
assert.equal(batteryGroupKey("P1_CASCADE_PREFETCH_SIM"), "P1 · políticas simuladas");

// Regra 6: /_C0_/ (nível 2) → 'C0 · Linear 4 caminhos'
assert.equal(batteryGroupKey("P1_CASCADE_C0_PIPELINE"), "C0 · Linear 4 caminhos");

// Regra 7: /_C1_/ (nível 2) → 'C1 · Bloco real'
assert.equal(batteryGroupKey("P1_CASCADE_C1_BLOCK_GATED"), "C1 · Bloco real");

// Regra 8: /^C3_<TECH>_(REST)$/ → 'C3 · ' + REST com underscores→espaços;
// o MESMO passo agrupa entre techs; C3_*_C1_DECISION NÃO cai na regra 7 (nível 3).
assert.equal(batteryGroupKey("C3_RIFT_LINEAR_F0_GATE_F1"), "C3 · LINEAR F0 GATE F1");
assert.equal(batteryGroupKey("C3_AETHER_LINEAR_F0_GATE_F1"), "C3 · LINEAR F0 GATE F1");
assert.equal(batteryGroupKey("C3_CASCADE_BLOCK_F0_GATE_F1"), "C3 · BLOCK F0 GATE F1");
assert.equal(batteryGroupKey("C3_SPECTRA_BLOCKS4_GATED"), "C3 · BLOCKS4 GATED");
assert.equal(batteryGroupKey("C3_RIFT_C1_DECISION"), "C3 · C1 DECISION");
assert.equal(batteryGroupKey("C3_CASCADE_C1_DECISION"), "C3 · C1 DECISION");

// Regras 9-11 (§16, FINAL_PHASE_V1): fases finais C4/C5/C6 ANTES do fallback
// (nível 5 de exibição); C6_* NÃO cai na regra 3 porque o id não termina em
// _E2E_TOKS. Case-insensitive como as demais regras.
assert.equal(batteryGroupKey("C4_RIFT_SECOND_FAMILY"), "C4 · Segunda família");
assert.equal(batteryGroupKey("C4_SPECTRA_SECOND_FAMILY"), "C4 · Segunda família");
assert.equal(batteryGroupKey("C5_AETHER_REPR_BLOCKS"), "C5 · Blocos representativos");
assert.equal(batteryGroupKey("C5_CASCADE_REPR_BLOCKS"), "C5 · Blocos representativos");
assert.equal(batteryGroupKey("C6_CASCADE_COMPILE_EXECUTE"), "C6 · Compilar+Executar");
assert.equal(batteryGroupKey("C6_RIFT_COMPILE_EXECUTE"), "C6 · Compilar+Executar");
assert.equal(batteryGroupKey("c6_cascade_compile_execute"), "C6 · Compilar+Executar");

// MicroLM (§22): NENHUMA regra nova — B0_MICROLM_NOOP_INIT cai na regra 4
// (B0 · Fundação) e os P1_MICROLM_* caem no fallback 'P1 · Codec principal';
// P1_MICROLM_DECODE_TOKS NÃO cai na regra 3 porque não termina em _E2E_TOKS.
assert.equal(batteryGroupKey("B0_MICROLM_NOOP_INIT"), "B0 · Fundação");
assert.equal(batteryGroupKey("P1_MICROLM_DECODE_PARITY"), "P1 · Codec principal");
assert.equal(batteryGroupKey("P1_MICROLM_DECODE_TOKS"), "P1 · Codec principal");
assert.equal(batteryGroupKey("P1_MICROLM_TRAINS_FROM_INIT"), "P1 · Codec principal");
assert.equal(batteryGroupKey("P1_MICROLM_UNIT_CHECKS"), "P1 · Codec principal");
assert.equal(batteryGroupKey("p1_microlm_decode_toks"), "P1 · Codec principal");

// Regra 12 (fallback): resto (P1_*/G[1-5]_* nível 1) → 'P1 · Codec principal'
assert.equal(batteryGroupKey("P1_Q4_LINEAR_BASE_2BIT"), "P1 · Codec principal");
assert.equal(batteryGroupKey("P1_Q4_LINEAR_BASE_PLUS_REF_4BIT"), "P1 · Codec principal");
assert.equal(batteryGroupKey("G1_GEYSER_ZDC_LUT"), "P1 · Codec principal");
assert.equal(batteryGroupKey("G5_GEYSER_ELASTIC_KV"), "P1 · Codec principal");
assert.equal(batteryGroupKey("P1_WINNER_F0_PLUS_LS"), "P1 · Codec principal");
assert.equal(batteryGroupKey(""), "P1 · Codec principal");

// ---------------------------------------------------------------------------
// Agrupamento sobre os dados reais publicados.
// ---------------------------------------------------------------------------

const published = JSON.parse(
  await readFile(new URL("../data/rift_test_batteries.json", import.meta.url), "utf8"),
);
assert.ok(Array.isArray(published) && published.length > 0, "histórico publicado vazio");

const KNOWN_GROUPS = new Set([
  "E2E · comparação de gerações",
  "Capacidades",
  "E2E · tok/s modelo completo",
  "B0 · Fundação",
  "P1 · políticas simuladas",
  "C0 · Linear 4 caminhos",
  "C1 · Bloco real",
  "C4 · Segunda família",
  "C5 · Blocos representativos",
  "C6 · Compilar+Executar",
  "P1 · Codec principal",
]);
const groupsByModel = new Map();
const codecTechsAllModels = new Set();
for (const record of published) {
  const batteryId = String(record.battery_id || "");
  const key = batteryGroupKey(batteryId);
  assert.ok(typeof key === "string" && key.length > 0, `grupo vazio para ${batteryId}`);
  assert.ok(
    KNOWN_GROUPS.has(key) || key.startsWith("C3 · "),
    `grupo inesperado "${key}" para ${batteryId}`,
  );
  const model = String(record.model_id || "");
  const technology = inferTechnology(record);
  if (!groupsByModel.has(model)) groupsByModel.set(model, new Map());
  const groups = groupsByModel.get(model);
  if (!groups.has(key)) groups.set(key, new Set());
  groups.get(key).add(technology);
  if (key === "P1 · Codec principal") codecTechsAllModels.add(technology);
}

// Exemplo canônico do usuário: o card 'P1 · Codec principal' agrega VÁRIAS
// tecnologias como linhas do mesmo grupo para o mesmo modelo.
const qwen = groupsByModel.get("Qwen/Qwen2.5-0.5B");
assert.ok(qwen, "sem registros publicados de Qwen/Qwen2.5-0.5B");
const codecTechs = qwen.get("P1 · Codec principal");
assert.ok(codecTechs, "Qwen/Qwen2.5-0.5B sem grupo 'P1 · Codec principal'");
for (const tech of ["RIFT", "WINNER"]) {
  assert.ok(codecTechs.has(tech), `grupo 'P1 · Codec principal' (Qwen 0.5B) sem linha ${tech}`);
}
assert.ok(codecTechs.size >= 2, "o card 'P1 · Codec principal' deveria ter 2+ linhas de tecnologia");
// No histórico completo, RIFT/AETHER/SPECTRA/WINNER contribuem para o codec principal.
for (const tech of ["RIFT", "AETHER", "SPECTRA", "WINNER"]) {
  assert.ok(codecTechsAllModels.has(tech), `nenhum modelo tem linha ${tech} no codec principal`);
}
// P1_CASCADE_C2_E2E_TOKS (CASCADE) cai no grupo E2E, não no codec principal.
const e2eTechs = qwen.get("E2E · tok/s modelo completo");
assert.ok(e2eTechs && e2eTechs.has("CASCADE"), "grupo E2E sem a linha CASCADE (C2)");
// Fundação B0 agrega múltiplas techs no mesmo card.
const b0Techs = qwen.get("B0 · Fundação");
assert.ok(b0Techs && b0Techs.size >= 2, "grupo 'B0 · Fundação' deveria ter 2+ tecnologias");

// ---------------------------------------------------------------------------
// batteryFriendlyName (§19.5, 7º lote) — o título visível de bateria é SEMPRE
// 'N<nível> · <nome amigável PT-BR não-técnico>'; o battery_id cru vive só em
// tooltip. A implementação do index.html é composta (helpers + mapas const) —
// index.html é o painel ÚNICO (§24.1); aqui ficam os exemplos do contrato.
// ---------------------------------------------------------------------------

function extractConstObject(source, name, filename) {
  const match = source.match(new RegExp(`const ${name}=\\{[\\s\\S]*?\\n\\};`));
  assert.ok(match, `${filename} sem const ${name}`);
  return match[0];
}

const friendlyContext = {};
vm.createContext(friendlyContext);
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
  friendlyContext,
  { filename: "legacy-batteryFriendlyName.js" },
);
const batteryFriendlyName = friendlyContext.batteryFriendlyName;

// Exemplos canônicos do contrato §19.5 (espelhados no smoke, que também
// compara as duas implementações entre si).
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
  // MicroLM (§22.4): as 5 baterias N1 do modelo de referência (nomes
  // amigáveis espelhados nos dois painéis; paridade garantida no smoke).
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
    batteryFriendlyName(batteryId),
    expectedName,
    `batteryFriendlyName(${JSON.stringify(batteryId)}) != ${JSON.stringify(expectedName)}`,
  );
}
// Fallback humanizado para id desconhecido: prefixo de nível, nunca o id cru.
const fallbackName = batteryFriendlyName("X9_FOO_BAR_TEST");
assert.match(fallbackName, /^N1 · /, "fallback humanizado sem prefixo de nível");
assert.ok(!fallbackName.includes("X9_FOO_BAR_TEST"), "fallback humanizado não pode exibir o id cru");

// Nenhum battery_id publicado pode vazar cru para o título amigável.
for (const record of published) {
  const batteryId = String(record.battery_id || "");
  const friendly = batteryFriendlyName(batteryId);
  assert.match(friendly, /^N[1-5] · /, `nome amigável sem prefixo de nível para ${batteryId}`);
  assert.ok(
    !friendly.includes(batteryId),
    `nome amigável exibe o id cru de ${batteryId} (id só pode viver em tooltip)`,
  );
}

// ---------------------------------------------------------------------------
// batteryBitClass (§23.4, 11º lote) — classe de bit por card com badge: dentro
// de um grupo é desleal misturar precisões. Cadeia canônica de tokens do id
// (TERNARY > 2BIT/INT2 > 4BIT/INT4/Q4, case-insensitive) + fallback por bits
// efetivos medidos (≤3 → 2-bit, ≤5 → 4-bit) + 'baixo-bit' honesto. index.html
// é a implementação única do painel (§24.1); os exemplos do contrato vivem aqui.
// ---------------------------------------------------------------------------

const batteryBitClass = vm.runInNewContext(
  `(${extractFunction(legacyHtml, "batteryBitClass", "index.html")})`,
  {},
  { filename: "legacy-batteryBitClass.js" },
);

const BIT_CLASSES = new Set(["2-bit", "ternário", "4-bit", "baixo-bit"]);
const BATTERY_BIT_CLASS_FIXTURES = [
  // 2BIT vence o prefixo Q4 do id (armadilha clássica).
  ["P1_Q4_LINEAR_BASE_2BIT", undefined, "2-bit"],
  ["P1_Q4_LINEAR_BASE_PLUS_REF_4BIT", undefined, "4-bit"],
  ["SELFTEST_Q4_BASE_PLUS_REF_4BIT", undefined, "4-bit"],
  // TERNARY vence 2BIT no mesmo id, case-insensitive.
  ["P1_AETHER_HQR_TERNARY_2BIT", undefined, "ternário"],
  ["P1_SPECTRA_HQR_TERNARY_2BIT", undefined, "ternário"],
  ["P1_WINNER_F0_TERNARY_2BIT", undefined, "ternário"],
  ["p1_winner_f0_ternary_2bit", undefined, "ternário"],
  // INT4/INT2 normalizados para as classes canônicas ('INT4' → '4-bit').
  ["G1_GEYSER_DRAFT_INT4_PROXY", undefined, "4-bit"],
  ["G1_GEYSER_DRAFT_INT2_HOT", undefined, "2-bit"],
  // Fallback por métricas (bits efetivos medidos no topo de metrics).
  ["CMP_RIFT_GENERATIONS", { bits_effective: 2.4 }, "2-bit"],
  ["CMP_GEYSER_GENERATIONS", { bits_effective: 3.7 }, "4-bit"],
  ["G2_GEYSER_RRS_SALIENCE", { bits_effective: 8 }, "baixo-bit"],
  // Sem token e sem bits medidos → 'baixo-bit'.
  ["P1_WINNER_F0_PLUS_LS", undefined, "baixo-bit"],
  ["P1_AETHER_HQR_PLUS_TADDS_DYNAMIC", undefined, "baixo-bit"],
  ["P1_MICROLM_DECODE_TOKS", undefined, "baixo-bit"],
  ["", undefined, "baixo-bit"],
];
for (const [batteryId, metrics, expected] of BATTERY_BIT_CLASS_FIXTURES) {
  assert.equal(
    batteryBitClass(batteryId, metrics),
    expected,
    `batteryBitClass(${JSON.stringify(batteryId)}) != ${JSON.stringify(expected)} (§23.4)`,
  );
}
// Sobre os dados reais publicados (com as métricas reais de cada registro), a
// classe emitida é SEMPRE uma das 4 canônicas — nunca um token cru tipo INT4.
for (const record of published) {
  const bitClass = batteryBitClass(record.battery_id, record.metrics);
  assert.ok(
    BIT_CLASSES.has(bitClass),
    `classe de bit fora do conjunto canônico para ${record.battery_id}: ${bitClass} (§23.4)`,
  );
}

console.log("battery_group_key fixtures: OK");
