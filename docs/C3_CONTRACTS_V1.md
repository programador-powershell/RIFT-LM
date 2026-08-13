# Contratos de Engenharia — C3 / Winner Dinâmico / Schema v2

Este documento é a fonte da verdade para os contratos compartilhados entre as baterias Python,
o runtime C++ (`engines/winner/cpp/`, ex-`winner_cpp/`), as APIs (`api/*.mjs`) e os dashboards (`dashboard.html`, `index.html`).
Qualquer alteração nestes contratos exige atualização deste arquivo na mesma mudança.

## 1. Política de seleção da arquitetura vencedora (WINNER dinâmico)

O WINNER não é mais acoplado ao CASCADE. A arquitetura executada pelo WINNER é escolhida
dinamicamente a partir do histórico publicado (`data/rift_test_batteries.json` ou `GET /api/results`).

Regra determinística (`select_winner_architecture(records)` em Python e
`selectWinnerArchitecture(records)` em JS — implementações devem ser espelhadas):

1. Tecnologias elegíveis: `RIFT`, `AETHER`, `CASCADE`, `SPECTRA`, `GEYSER` (o próprio `WINNER`
   é excluído). `GEYSER` entrou em 2026-08-10 (§7).
2. "Modelo otimizado" de uma tecnologia = `model_id` distinto para o qual existe ao menos um
   registro com `technology == tech`, `comparison_role == "primary"`,
   `status ∈ {PASS, EXPERIMENTAL_PASS}` e (quando presente) `quality.full_local_gate_pass == true`.
   Registros com `model_id` iniciando em `synthetic/` são ignorados.
3. Vence a tecnologia com o MAIOR número de modelos otimizados.
4. Empate: vence a de maior score médio (pesos `SCORE_WEIGHTS` definidos no §25 — score
   canônico v2: cosine 25, nrmse 10, disk 10, ram 30, speedup 20, gate 5 — sobre o
   registro mais recente por par modelo|battery_id).
5. Empate persistente / sem dados: ordem de prioridade fixa
   `[CASCADE, RIFT, AETHER, SPECTRA, GEYSER]` (CASCADE é o incumbente).

Overrides: variável de ambiente/Secret `RIFT_WINNER_ARCH` (valores:
`RIFT|AETHER|CASCADE|SPECTRA|GEYSER`) força a arquitetura no script do WINNER; parâmetro
`?arch=` no launcher tem o mesmo efeito.
Todo registro do WINNER deve gravar `metrics.winner.architecture_selected`,
`metrics.winner.selection_basis` (`"published_history" | "env_override" | "default_incumbent"`)
e `metrics.winner.optimized_model_counts` (mapa tech→contagem no momento da seleção).

### Normalização do desempate (canônica — obrigatória em TODAS as implementações)

O score do passo 4 usa EXATAMENTE as fórmulas de `normalizedMetric`/`scoreTechnology` de
`api/analyze.mjs` (espelhadas em `winnerRecordScore` de `api/results.mjs`, `scoreRecord` de
`index.html` e `_normalized_score`/`_record_score` de
`winner_m0_phase1_test_v080_auto_batteries.py` — 4 implementações; dashboard.html foi
removido pelo §24.1). `clamp(v, a, b) = min(b, max(a, v))`;
métrica ausente/não numérica = `null` e fica FORA da média (não vale 0):

| Métrica (peso) | Normalização para 0..100 |
| --- | --- |
| `output_cosine` (25) | `clamp((v + 1) * 50, 0, 100)` |
| `output_nrmse` (10) | `100 * (1 - clamp(v / 0.1, 0, 1))` |
| `disk_reduction_pct` (10) | `clamp(v, 0, 100)` |
| `ram_reduction_pct` (30) | `clamp(v, 0, 100)` |
| `operation_speedup_x` (20) | `clamp(v, 0, 1) * 100` |
| `quality_gate_pass` (5) | `true → 100`, `false → 0`, ausente → `null` |

Pesos definidos no §25 (`SCORE_WEIGHTS_V2`, score canônico v2 — objetivo "computador
convencional"); esta tabela apenas os espelha. As fórmulas de normalização acima NÃO mudam.

Score do registro: `raw = Σ(norm_i × peso_i) / Σ(peso_i presentes)`;
`coverage = Σ(peso_i presentes) / 100`; `score = raw × (0.65 + 0.35 × coverage)`.
Registro sem NENHUMA métrica presente tem score `null` e não entra na média da tecnologia.
Score médio da tecnologia = média aritmética dos scores não nulos dos registros mais
recentes por par `modelo|battery_id`. É PROIBIDO usar outras curvas (ex.: cosine
`(v-0.90)/0.10`, nrmse `/0.08`, `50 + reduction/2`, `50 + 25*log2(speedup)`) ou omitir o
fator de coverage — qualquer mudança aqui exige atualizar este documento e as quatro
implementações na mesma mudança.

## 2. Terceira bateria (série C3) — metodologia de 16 passos

Script único: `c3_methodology_auto_batteries.py --technology rift|aether|cascade|spectra`
(auto-contido no estilo do repositório, reutiliza o pacote `cascade/` para INT4+low-rank,
IR, bundle, gate e runtimes). Codec F0 por tecnologia (flag `--codec`, default por tech):
`cascade → int4`, `rift → int2`, `aether → ternary`, `spectra → ternary` (SPECTRA adiciona a
métrica de contrato de drift). F1 (residual low-rank) e Confidence Gate v0 são comuns.

Mapeamento dos 16 passos → battery_ids (`<TECH>` ∈ {RIFT, AETHER, CASCADE, SPECTRA}):

| Passos | battery_id | Conteúdo |
| --- | --- | --- |
| 1, 5 | `C3_<TECH>_BUNDLE_M0_FREEZE` | Bundle M0 congelado (CSCD v0x0003, header 128B) + golden tests válidos/inválidos |
| 2 | `C3_<TECH>_STAGE_PAGE_M0_FREEZE` | Stage Table/Page ABI congelada (entrada de 24 bytes) + golden tests |
| 4 | `C3_<TECH>_IR_WRITER` | CASCADE-IR v3 write→validate→reload roundtrip |
| 6 | `C3_<TECH>_CPP_BUNDLE_READER` | Compila e executa o leitor C++ mmap sobre o bundle real (POSIX; `SKIPPED` no Windows) |
| 3, 7–12 | `C3_<TECH>_LINEAR_ORIGINAL`, `C3_<TECH>_LINEAR_F0_ONLY`, `C3_<TECH>_LINEAR_F0_PLUS_F1_ALWAYS`, `C3_<TECH>_LINEAR_F0_GATE_F1` | Linear real, 4 caminhos, cada um como REGISTRO PRÓPRIO |
| 11–12 | `C3_<TECH>_BLOCK_ORIGINAL`, `C3_<TECH>_BLOCK_F0_ONLY`, `C3_<TECH>_BLOCK_F0_PLUS_F1_ALWAYS`, `C3_<TECH>_BLOCK_F0_GATE_F1` | Bloco Transformer real, 4 caminhos |
| 14 | `C3_<TECH>_C1_DECISION` | Aprova/reprova C1: `PASS` sse cosine≥0.995 ∧ NRMSE≤0.05 no caminho gated (Linear e Bloco) ∧ golden tests PASS ∧ F1_skip_rate>0 |
| 15 | `C3_<TECH>_BLOCKS4_GATED` | Expansão para 4 blocos reais patchados (gated vs original) |
| 16 | `C3_<TECH>_FULLMODEL_E2E_TOKS` | Modelo pequeno completo: TODOS os blocos patchados; `baseline_tok_s` E `candidate_tok_s` REAIS via `model.generate` |

`comparison_role="primary"`: `C3_*_LINEAR_F0_GATE_F1`, `C3_*_BLOCK_F0_GATE_F1`,
`C3_*_BLOCKS4_GATED`, `C3_*_FULLMODEL_E2E_TOKS`. Os demais são diagnósticos (`comparison_role=null`).
O passo 13 (alimentar dashboard) é o publish incremental de cada registro.

Id diagnóstico adicional (fora dos 16 passos): `C3_<TECH>_LOAD_MODEL`, emitido com
`status=FAIL` quando o carregamento do modelo falha (registro de infraestrutura,
nunca primário). A decisão C1 reprova a expansão: os passos 15 e 16 só executam com
`C3_<TECH>_C1_DECISION = PASS`; com C1 reprovado ambos são emitidos como `SKIPPED`.

## 3. Regras de honestidade de medição (herda docs/REAL_BENCHMARK_PROTOCOL_V3.md)

- Latência: `time.perf_counter_ns` com warmup e `torch.cuda.synchronize()` quando CUDA.
- `baseline_tok_s`/`candidate_tok_s` de nível superior: SOMENTE de `model.generate` do modelo
  completo (baseline e candidato sob o mesmo protocolo). Proxies de Linear/bloco ficam apenas em
  `metrics.operation.*` com sufixo `_proxy` e NUNCA nos campos de comparação.
- `*_ram_bytes` de nível superior: SOMENTE RSS medido (thread de amostragem de `/proc/self/status`
  VmRSS a ~1ms, pico; método registrado em `metrics.memory.method`). Estimativas aritméticas vivem
  apenas em `metrics.memory.estimated_*`. Sem medição → `null`.
- `candidate_disk_bytes`: soma de `os.stat().st_size` de artefatos binários reais existentes.
- Sem atividade real capturada (fallback sintético): registro é rebaixado
  (`comparison_role=null`, nota `activation_source=synthetic_fallback`).
- Baterias simuladas: `status="SIMULATED"` + `implementation.kind="SIMULATED"` +
  `eligible_for_primary_ranking=false`.
- Campos obrigatórios do schema v2 em TODO registro novo: `schema_version: 2`,
  `benchmark_protocol` (`"C3_METHODOLOGY_V1"` para a série C3), `comparison_group_id`
  (`cmp-<sha256[:24]>` de `protocol|model_id|device|torch`), `comparison_context`,
  `implementation {kind: REFERENCE_MEASURED|NATIVE_MEASURED|SIMULATED, native: bool, simulated: bool}`.

## 4. Endpoints

- `POST /api/results` — inalterado (Bearer ≥32 chars, timing-safe), mas com whitelist de campos
  (sem spread de chaves arbitrárias) e caps existentes.
- `GET /api/results` — NOVO, público (dados já são públicos): retorna
  `{generated_at, count, records:[...]}` lendo o histórico do GitHub raw (fallback: arquivo
  estático do deploy). Cache: `s-maxage=15, stale-while-revalidate=60` + ETag. É a fonte
  primária dos dashboards; fallback client-side para `/data/rift_test_batteries.json`.
- Launchers: rota `/c3/:tech/:model*` → gera célula Colab da série C3 (mesmo padrão de
  segurança: `RIFT_INGEST_TOKEN` ≥32, HTTPS, pin de commit SHA quando `VERCEL_GIT_COMMIT_SHA`
  disponível).
- `GET /geyser/:model*` — serve `geyser_launcher.py` como `text/plain` com o placeholder
  `__GEYSER_MODEL_ID__` substituído pelo modelo da rota (contrato de uso:
  `curl -fsSL <url> -o /content/geyser_launcher.py && python /content/geyser_launcher.py`).
- `/cap/:model*` → gera célula Colab da bateria de capacidades (§9), mesmo padrão de segurança
  dos demais launchers.

## 5. Segurança (aplicável a todo o repositório)

- `vercel.json`: adicionar `Content-Security-Policy` (sem `unsafe-eval`; `script-src 'self'
  'unsafe-inline'` é aceitável pois os dashboards usam JS inline; `connect-src 'self'
  https://raw.githubusercontent.com`) e `Strict-Transport-Security`.
- Todo publisher Python: HTTPS obrigatório + token ≥32 chars (inclusive série C e C3).
- `pip install` automático: apenas quando `google.colab` importável OU `RIFT_AUTO_INSTALL=1`;
  versões PINADAS (ex.: `transformers==4.*` faixa testada) — nunca `-U` sem pino.
- Leitores binários C++ (`mmap_bundle.hpp`, `engines/winner/cpp/src/bundle.cpp`): validar TODOS os
  offsets/tamanhos contra o tamanho do arquivo antes de indexar; verificar CRC; rejeitar com erro
  claro em vez de ler fora dos limites.
- Limpeza destrutiva (`cleanup_colab_workspace`, `shutil.rmtree`): só executa quando
  `google.colab` importável ou caminho sob `/content`; nunca em execução local fora do Colab.
- Segredos: somente env vars / Colab Secrets / Vercel env. Nunca em arquivos versionados.

## 6. Codecs F0 (série C3)

- `int4`: `cascade/kernels/int4.py` (groupwise signed [-8,7], escala FP32, grupo 32).
- `int2`: mesmo esquema groupwise com 2 bits/peso, 4 níveis simétricos (RIFT).
- `ternary`: {-1,0,+1} com escala por linha + limiar de esparsidade (AETHER/SPECTRA).
Todos os codecs expõem `pack(...)`, `unpack(...)`, `linear(x, packed) -> y` de referência e
`packed_bytes` reais. Bundles gravam `stage.meta.codec` com o codec REAL do payload do
stage 0 (`INT4_GROUP` | `INT2_GROUP` | `TERNARY_ROWSCALE`; stage 1 é sempre
`FP32_LOWRANK`) — o passo 2 (`C3_<TECH>_STAGE_PAGE_M0_FREEZE`) falha o ABI check quando
o codec declarado diverge do codec da tecnologia.

## 7. GEYSER-LM (6ª tecnologia — 2026-08-10)

- Launcher: `engines/geyser/geyser_launcher.py` (§20; suite M0/G0 da spec GEYSER v0.1). Contrato de
  disparo DIFERENTE dos demais: a rota `GET /geyser/:model*` serve o PRÓPRIO arquivo Python
  (placeholder `__GEYSER_MODEL_ID__` substituído), executado com
  `curl -fsSL ... -o /content/geyser_launcher.py && python /content/geyser_launcher.py`.
- battery_ids: `B0_GEYSER_PHYSICS_BANDWIDTH`, `G1_GEYSER_ZDC_LUT`, `G2_GEYSER_RRS_SALIENCE`,
  `G3_GEYSER_BURST`, `G4_GEYSER_EQC`, `G5_GEYSER_ELASTIC_KV` (todas Nível 1, §8).
- Publicação: o launcher publica `{records:[...]}` no `POST /api/results` (Bearer ≥32 + HTTPS,
  como todos), convertendo cada bateria para o schema v2: `technology="GEYSER"`,
  `battery_id=<name>`, status `OK→PASS`, `FAILED→FAIL`, `SKIPPED→SKIPPED`,
  `benchmark_protocol="GEYSER_M0_G0_V1"`, `run_id="geyser-<RUN_TS>-<SESSION>"`.
  `comparison_role="primary"` SOMENTE em `G1_GEYSER_ZDC_LUT` (qualidade de codec sobre Linear
  real, análogo aos P1_*), com `quality.output={cosine,nrmse}` das ativações reais.
- Honestidade: tok/s de nível superior permanece `null` (tok/s Python de G3 é rotulado
  `SIMULADO EM PYTHON`; projeções BPAT são `PROJETADO` e ficam apenas em `metrics`). Aliases
  `geyser_*` aceitos pela whitelist do ingest.
- Limpeza: `/content/geyser_m0_test_output` DEVE constar nas listas de limpeza prévia das
  células Colab (fila serial e afins).

## 8. Níveis de bateria (camada de EXIBIÇÃO — ids nunca são renomeados)

Os `battery_id` históricos são imutáveis. Os dashboards agrupam e rotulam por NÍVEL, exibidos
em sequência (Nível 1 → 4), via função espelhada `batteryLevel(battery_id)`:

| Nível | Rótulo | Regra sobre o battery_id |
| --- | --- | --- |
| 1 | `Nível 1 · Fundação (M0)` | contém `_C0_`/`_C1_`/`_C2_`? não → prefixo `B0_`, `P1_`, `G1_`..`G5_` |
| 2 | `Nível 2 · Série C (bloco/e2e)` | contém `_C0_`, `_C1_`, `_C2_` (ex.: `P1_CASCADE_C1_BLOCK_GATED`) |
| 3 | `Nível 3 · Metodologia C3` | prefixo `C3_` |
| 4 | `Nível 4 · Capacidades` | prefixo `CAP_` |

Nome amigável = rótulo do nível + descrição curta derivada do id (ex.:
`P1_Q4_LINEAR_BASE_2BIT` → "N1 · RIFT · codec 2-bit em Linear real"). O id bruto permanece
visível como texto secundário/tooltip.

## 9. Bateria de capacidades (estilo OpenRouter compare) — `CAPABILITY_PROBE_V1`

Script: `capability_eval_auto_batteries.py --model <org/modelo>` (auto-contido; tarefas
embutidas e determinísticas; `model.generate` greedy). Três categorias, 3 registros:

- `CAP_INTELLIGENCE` — múltipla escolha estilo MMLU (subset embutido ≥20 questões);
- `CAP_CODING` — completar funções Python com asserts embutidos (≥8 tarefas, execução
  sandbox local com timeout e sem I/O);
- `CAP_AGENTIC` — instruções de function-calling/JSON: gerar chamada JSON válida contra
  schema embutido (≥8 tarefas; JSON 100% parseável é parte do score).

Score 0–100 em `metrics.capability.score` (+ breakdown por tarefa), `status=PASS` quando a
suíte executa completa (score é medida, não gate), `comparison_role=null` (NÃO entra na
seleção do winner), `benchmark_protocol="CAPABILITY_PROBE_V1"`.
Registros de capacidade são POR MODELO (avaliam o modelo baseline, não uma tecnologia de
otimização): CONTRATO — `technology="CAP"`, valor adicionado ao enum de validação
(`validate_data.mjs`, `api/results.mjs`) e EXCLUÍDO dos rankings de tecnologia e da política
do winner em todas as implementações. Dashboards exibem capacidades em seção própria com
gráficos de barras triplos por modelo (Intelligence · Coding · Agentic), como no OpenRouter
compare. Rotulagem obrigatória: "probe leve embutido — não é MMLU/HumanEval/SWE-bench
completos".

## 10. Identidade visual

Título público do painel: **"Observatório LLM"** (substitui "RIFT-LM, CASCADE, AETHER,
SPECTRA & WINNER Test Observatory"). Subtítulo curto com a linhagem:
`RIFT · CASCADE · AETHER · SPECTRA · GEYSER · WINNER`. Cor de série do GEYSER nos gráficos:
teal (`#0d9488` claro / `#2dd4bf` escuro), distinta das 5 existentes.

## 11. Muse Glimmer 2-bit / caminho GGUF (Colab T4) — `GGUF_RUNTIME_V1`

Modelo alvo: `unsloth/Muse-Glimmer-30B-GGUF`, quant `UD-Q2_K_XL` (~12–15 GB; T4 exige
llama.cpp com offload parcial CPU/GPU — `transformers` NÃO carrega 30B no T4).
Script: `gguf_e2e_auto_batteries.py --model unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q2_K_XL`.

- `B0_GGUF_RUNTIME_SETUP` — baixa binário oficial do llama.cpp (release PINADA por tag +
  verificação sha256) e o GGUF do quant escolhido via `huggingface_hub` (apenas os arquivos
  do quant — nunca o repositório inteiro); registra bytes reais de disco e tempo de setup.
- `P1_GGUF_E2E_TOKS` — tok/s REAL de decode via llama.cpp no T4 (prompt fixo, ≥3 medições,
  mediana; `-ngl` auto), RAM/VRAM medidas (RSS + nvidia-smi). O artefato 2-bit É o candidato:
  `candidate_tok_s` medido, `candidate_disk_bytes` = os.stat do .gguf; baseline BF16 não é
  executável no T4 → `baseline_tok_s=null` com nota (sem comparação inventada);
  `comparison_role=null`.
- `P1_GGUF_<TECH>_CODEC_TENSOR` (RIFT/AETHER/CASCADE/SPECTRA) — extrai UMA Linear REAL do
  GGUF (pacote `gguf` do llama.cpp, pinado; dequant do tensor para FP32) e roda o codec F0+F1
  da tecnologia sobre esse tensor real; ativação sintética FLAGADA (sem forward do modelo em
  torch) → `comparison_role=null` por protocolo, mas alimenta o dashboard.
- CAP no GGUF: `capability_eval_auto_batteries.py` ganha `--backend llamacpp
  --server-url http://127.0.0.1:<porta>` (llama-server, API OpenAI-compatível) — mesmos
  battery_ids CAP_*, `model_id="unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL"`.
- Dependências novas (SUJEITAS A HOMOLOGAÇÃO TI/SI, instaladas SÓ no Colab, pinadas):
  binário llama.cpp (release oficial ggml-org/llama.cpp) e pacote pip `gguf`.
- Rota: `/gguf/:model*` → célula Colab (mesmo padrão de segurança). Sugerido no painel como
  "Muse Glimmer 2-bit (T4)". A trava anti-GGUF dos launchers antigos NÃO se aplica a esta rota.

## 12. tok/s contado em TODAS as tecnologias — baterias E2E

Regra do V3 mantida (tok/s de topo só de `model.generate` sob mesmo protocolo) — o que muda
é que TODAS as tecnologias agora TÊM essa medição:

- Novos battery_ids `P1_RIFT_E2E_TOKS`, `P1_AETHER_E2E_TOKS`, `P1_SPECTRA_E2E_TOKS`,
  `P1_WINNER_E2E_TOKS` nos scripts M0 (CASCADE já tem `P1_CASCADE_C2_E2E_TOKS`):
  baseline = modelo original via `generate` (greedy, ≥2 warmup, ≥3 medições, mediana);
  candidato = MESMO `generate` com todas as Linear dos blocos executando o runtime de
  referência do codec da tecnologia (W denso fora do caminho quente); ambos MEDIDOS →
  `baseline_tok_s` E `candidate_tok_s` de topo; `comparison_role="primary"`;
  `measurement_scope` declara "runtime de referência Python".
- `P1_CASCADE_C2_E2E_TOKS` passa a medir também o candidato (mesma técnica).
- GEYSER `G3_GEYSER_BURST`: promove `vanilla_tok_s_py`→`baseline_tok_s` e
  `burst_tok_s_py`→`candidate_tok_s` (mesmo protocolo wall-clock python, equivalência greedy
  verificada); G3 vira `comparison_role="primary"` além do G1.
- `scripts/real_benchmark_runner.py`: deixa de anular tok/s quando o registro é `*_E2E_TOKS`
  com `metrics.e2e.measured=true` (continua anulando proxies).
- Adendo correspondente em docs/REAL_BENCHMARK_PROTOCOL_V3.md.

## 13. Regras de UI do painel (pedidos 2026-08-10, 2º lote)

1. Cards por modelo: UM card por GRUPO de bateria (ex.: "P1 · codec 2-bit"), com as
   tecnologias como LINHAS dentro de cada métrica (Throughput, RAM necessária, Espaço em
   disco, Ganhos medidos). O valor "Antes" aparece UMA vez no topo de cada métrica (não se
   repete por tecnologia; se os baselines divergirem entre techs, usar o mais recente e
   marcar "≈"). Grupo = mesmo nível (§8) + mesma classe de bateria; chave
   `batteryGroupKey(battery_id)` espelhada entre os dashboards.
2. Métricas de qualidade (Weight cosine, Output cosine, Quality gate, Estágio adaptativo):
   NÃO empilhar no card principal — o conjunto de badges do card (tech/status/fonte) vira um
   BOTÃO de alternância que troca a visualização do card para a visão de qualidade
   (por tecnologia), e volta.
3. Execução de testes: NÃO existe botão por tecnologia. Toda geração de célula/reteste roda
   TODAS as tecnologias (fila serial completa: M0 × 6 techs + série C + C3 × 4 + CAP +
   GEYSER), para o(s) modelo(s) escolhido(s). Dropdowns de tecnologia podem continuar
   existindo APENAS como filtro visual de gráficos.
4. Análise de IA: não é card. O retorno da IA vira DESTAQUE inline (ex.: ★ "IA recomenda")
   nas linhas/cards dos melhores modelos de cada teste/ranking.
5. Nenhum gráfico previamente existente pode ser removido/ocultado sem pedido explícito:
   restaurar os que sumiram no redesign (inventário: Throughput tok/s por bateria, RAM,
   Disco, resumo de ganhos, score médio por tecnologia/rodada, série temporal tok/s,
   speedup, latência, reduções, capacidades).
6. A base de dados publicada no GitHub (data/rift_test_batteries.json) permanece a fonte
   para verificação visual — não remover/filtrar registros históricos.

Nota §11 (enum): `B0_GGUF_RUNTIME_SETUP` e `P1_GGUF_E2E_TOKS` usam `technology="GGUF"`
(valor adicionado aos enums de `api/results.mjs` e `scripts/validate_data.mjs`; exibido como
série própria nos dashboards; NUNCA elegível na política do winner). As baterias
`P1_GGUF_<TECH>_CODEC_TENSOR` usam a `technology` da respectiva tecnologia.

## 14. Repo-agnóstico + células Colab curtas (pedido 2026-08-10, 3º lote)

Motivação: renomear o repositório GitHub (ex.: para `llm-battery-test`) quebrou dashboard e
launchers (404 no Colab) porque `programador-powershell/RIFT-LM` estava hardcoded.

1. RESOLUÇÃO DE REPO (server-side, cadeia única `resolveRepo()` espelhada em api/*):
   `GITHUB_REPO` → `RIFT_GITHUB_REPOSITORY` → `VERCEL_GIT_REPO_OWNER/VERCEL_GIT_REPO_SLUG`
   (env automático da Vercel) → fallback legado `programador-powershell/RIFT-LM`.
   Ref/branch: `RIFT_GITHUB_BRANCH` → `VERCEL_GIT_COMMIT_SHA` (pin preferido) → `main`.
2. PYTHON: nenhum script .py pode hardcodar owner/repo — todos leem
   `RIFT_GITHUB_REPOSITORY` e `RIFT_SOURCE_REF` do ambiente (com o mesmo fallback legado).
   As células geradas pelo servidor EXPORTAM essas envs (valores vindos de resolveRepo()).
3. CÉLULA COLAB CURTA (contrato do runner):
   - Nova rota `GET /runner.py` → `/api/runner` (text/plain): script orquestrador COMPLETO
     gerado no servidor (repo/SHA/origin já resolvidos), que: instala deps de tokenização
     pinadas (transformers/accelerate/tokenizers/sentencepiece/tiktoken — aceita todo tipo de
     tokenização), lê `MODELS`/`TECHS`/`BASE` do escopo chamador (globals) ou de env
     `RIFT_QUEUE_MODELS`/`RIFT_QUEUE_TECHS`, faz limpeza prévia, e para cada modelo×tech faz
     request em `{BASE}/{tech}/{model}` (launchers já existentes) e executa em subprocesso
     isolado (falha não interrompe a fila), com espera de liberação de VRAM entre passos.
   - A célula gerada pelos dashboards (individual E fila) é APENAS o bootstrap (~10 linhas):
     Secrets → `BASE = "<origin>"` → `MODELS = [...]` (variável) → `TECHS = ["all"]` ou lista
     → `exec(urlopen(BASE + "/runner.py").read().decode())`.
   - `TECHS=["all"]` = todas (M0×6 + série C + C3×4 + CAP + GEYSER + GGUF quando aplicável),
     coerente com §13.3.
4. UI: remover o select "Modelo sugerido" do card de fila; a seleção vem do picker/campo de
   modelo; as URLs amigáveis por tech (`{BASE}/rift/<org>/<modelo>` etc.) são geradas
   automaticamente a partir do modelo escolhido.

## 15. Probes de benchmark agêntico (pedido 2026-08-10, 4º lote) — extensão do §9

`capability_eval_auto_batteries.py` ganha 4 novas categorias (mesmo protocolo
`CAPABILITY_PROBE_V1`, `technology="CAP"`, score 0–100 em `metrics.capability.score`,
ambos os backends transformers|llamacpp):

| battery_id | Inspiração | Probe embutido (determinístico, ≥8 tarefas) |
| --- | --- | --- |
| `CAP_DEEPSEARCH_QA` | DeepSearch QA | QA multi-hop: mini-corpus de 2–3 passagens no prompt; resposta exige combinar fatos; score por match normalizado |
| `CAP_MCP_ATLAS` | MCP Atlas | Uso de ferramentas MCP: várias tool schemas; escolher a ferramenta certa e emitir chamada JSON válida com argumentos corretos (cenários multi-passo) |
| `CAP_TAU3_BENCH` | τ³-Bench | Agente de atendimento com política (domínio bancário/aéreo): próxima ação/tool call correta E conformidade com a política declarada |
| `CAP_SWE_BENCH` | SWE-Bench | Reparo de código: função Python com bug + teste falhando no prompt; corrigir a função; asserts embutidos em sandbox (mesmo sandbox do CAP_CODING) |

Rotulagem obrigatória por registro: "probe leve inspirado em <benchmark> — NÃO é o
benchmark oficial completo". Dashboards: a seção Capacidades renderiza os gráficos
DINAMICAMENTE a partir da lista de categorias presentes nos registros CAP_* (barras por
modelo, rótulos de valor), de modo que novas categorias aparecem sem mudança de UI; ordem
fixa: Intelligence, Coding, Agentic, DeepSearch QA, MCP-Atlas, τ³-Bench, SWE-Bench.
O runner (§14) já cobre o passo `cap` — nenhuma mudança de célula Colab é necessária além
do script.

## 16. Fases finais (5º lote, 2026-08-10) — `FINAL_PHASE_V1`

Fecha a escada das especificações (CASCADE C4–C6; RIFT §38.1 passos 16–19; SPECTRA Fases
2–8; AETHER MVS+) até o marco "compilador e executor LLM". Script único:
`final_phase_auto_batteries.py --technology rift|aether|cascade|spectra` (auto-contido no
estilo do C3; reutiliza o pacote `cascade/`; codec F0 por tecnologia como no §2/§6).

| battery_id | Fase da spec | Conteúdo | PASS sse |
| --- | --- | --- | --- |
| `C4_<TECH>_SECOND_FAMILY` | CASCADE C4 / RIFT 19 | MESMO core em 2 famílias: modelo A (default Qwen/Qwen2.5-0.5B) e modelo B de OUTRA família (default HuggingFaceTB/SmolLM2-360M, arch Llama; `--second-model`); Linear+bloco gated em ambos | cosine gated ≥0.98 nas DUAS famílias |
| `C5_<TECH>_REPR_BLOCKS` | CASCADE C5 / RIFT 18 | 8–10 blocos representativos de modelo maior (default Qwen/Qwen2.5-1.5B; `--large-model`), amostrados no espectro de profundidade (início/meio/fim); qualidade por profundidade + drift acumulado encadeando os blocos amostrados | drift acumulado ≤ 0.12 (budget da spec SPECTRA) ∧ cosine por bloco ≥0.95 |
| `C6_<TECH>_COMPILE_EXECUTE` | CASCADE C6/§25 / RIFT 16 | MARCO FINAL: compila TODAS as Linear dos blocos do modelo pequeno → grava bundles CSCD REAIS em disco (`<out>/bundle/`, codec da tech) → módulos de runtime carregam os stages DO ARQUIVO (mmap) → pesos originais REMOVIDOS do caminho quente (descartados após swap) → `generate` completo | executa ∧ gate ativo (skip>0) ∧ logits cosine ≥0.95 ∧ bundle_bytes < checkpoint_bytes ∧ nenhum peso original denso reconstruído |

- `comparison_role="primary"` SOMENTE em `C6_*` (com `baseline_tok_s` E `candidate_tok_s`
  reais via `generate` — baseline medido ANTES da compilação; `metrics.e2e.measured=true`);
  C4/C5 são gates diagnósticos (`comparison_role=null`).
- `benchmark_protocol="FINAL_PHASE_V1"`; schema v2 completo; RAM VmRSS por fase;
  `candidate_disk_bytes` = soma os.stat dos bundles reais; publish incremental endurecido.
- Guard de recursos: >3e9 params → SKIPPED; OOM → SKIPPED com nota; C5/C4 padrão podem ser
  reduzidos via flags (`--skip-c5`, `--skip-c4`).
- NÍVEL de exibição: prefixos `C4_`/`C5_`/`C6_` → **Nível 5 · Fase final** (atualizar
  `batteryLevel` E `batteryGroupKey` nos DOIS dashboards + fixtures: regra nova ANTES da
  regra 8; grupos: 'C4 · Segunda família', 'C5 · Blocos representativos',
  'C6 · Compilar+Executar' — C6_* NÃO cai na regra 2 `_E2E_TOKS` pois o id não tem o sufixo).
- Rotas: `/final/:tech/:model*` → célula via api/test (`battery=final`, `technology=all`
  suportado); runner (§14) ganha o passo `final` na expansão 'all'; preclean +=
  `/content/final_test_output`.

## 17. UX de comparação estilo openrouter.ai/compare (5º lote)

1. O painel deixa de listar benchmark de TODOS os modelos: existe um estado de MODELOS
   SELECIONADOS (persistido em `localStorage`, chave `observatorio_selected_models`).
2. Botão "+ Adicionar modelo" abre um POPUP/modal: lista dos `model_id` distintos presentes
   no histórico publicado (com busca), cada item com contagem de baterias/techs; ao focar um
   item, painel de PREVIEW no próprio modal (melhor tecnologia e score, tok/s medido quando
   houver, capacidades resumidas, nº de baterias por nível, último timestamp). Botão
   "Adicionar à comparação" inclui o modelo na seleção (múltiplos).
3. A área de benchmark/comparação renderiza SÓ os modelos selecionados, em colunas lado a
   lado (estilo openrouter/compare): capacidades (7 categorias), score por tecnologia,
   tok/s, ganhos RAM/disco, contagens por nível; chips removíveis por modelo.
4. Seleção vazia → estado convidando a adicionar (+ atalho "adicionar todos ≤N").
   Gráficos globais (série temporal, ranking geral) permanecem globais.
5. Vale para dashboard.html (principal) e index.html (seção de cards por modelo).

## 18. GEYSER v0.2.0 + comparação de gerações (6º lote, 2026-08-10)

1. `geyser_launcher.py` do repositório adota a base científica v0.2.0 do usuário (draft
   proxy INT4g32 p/ tau com disclosure condicional H1, probe do draft INT2 quente,
   KV KIVI-classe real com sink/janela/grupos e bits medidos vs assintóticos, tuning
   disclosure, tau_by_k) MANTENDO a camada de pipeline do repo (§7): conversão para
   `{records:[...]}` schema v2, HTTPS+token≥32, Colab Secrets fallback, G1 e G3
   primários, promoção do tok/s medido do G3 (vanilla→baseline, burst→candidato,
   escopo python wall-clock). Regra permanente: atualizações de ciência do launcher
   NUNCA removem a camada de publicação — merge, não substituição.
2. Comparação de gerações (novo artefato `compare_generations_report.json`; formato:
   `{model, target_linear, schemes_tensor{TECH...}, e2e{ORIGINAL+TECH...}, ceilings}`):
   - Conversão para registros: battery_id `CMP_<TECH>_GENERATIONS`, technology=<TECH>,
     `status=PASS` sse `top1_agreement≥0.70 ∧ ppl≤1.5×ppl_original`, senão
     `EXPERIMENTAL_FAIL`; `comparison_role=null`; métricas em `metrics.compare`
     (top1, mean_kl, ppl, ppl_original, gen_sample, bits_eff, seconds, schemes do tensor);
     tetos PROJETADOS ficam só em metrics com rótulo.
   - Publicação: script `batteries/compare_generations_publisher.py` (§20; stdlib-only, hardening
     padrão) converte e faz POST; os dashboards também aceitam o arquivo bruto via
     upload (detecção pela chave `schemes_tensor`) e via fetch best-effort de
     `data/compare_generations_report.json` (cópia real commitada para verificação
     visual).
   - Exibição: seção própria "Comparação de gerações (e2e real)" nos dois dashboards —
     colunas por tecnologia com top1/KL/PPL (vs original), amostra de geração (esc()!),
     bits efetivos e tempo; bloco de tetos rotulado PROJETADO.
   - Níveis/grupos: `/^CMP_/` → nível 2, grupo 'E2E · comparação de gerações'
     (regra nova antes das demais; espelhada + fixtures).

## 19. Regras de UI — 7º lote (2026-08-10): dados e gráficos, não texto

1. TEXTO EXPLICATIVO: NENHUM parágrafo explicativo visível nos dashboards. Toda explicação
   (referências de contrato, escopos, notas de honestidade, instruções) vai para um BADGE
   de informação "ⓘ" ao lado do título da seção/card — clicar abre um popover pequeno com
   o texto (fechar por clique fora/ESC). Implementação única por página (componente
   reutilizável `infoBadge`). Rótulos de honestidade CRÍTICOS (ex.: "PROJETADO") podem
   permanecer como chip curto de 1 palavra; o restante vai para o ⓘ.
2. META-LINGUAGEM PROIBIDA no texto visível: nada de "estilo openrouter", "contrato §17",
   nomes de protocolo/battery_id crus em títulos. Título da seção: "Comparação de modelos".
3. Botões "+ Adicionar modelo" e "adicionar todos (≤6)" lado a lado (mesma linha flex).
4. O card "Comparador auditável" de dashboard.html é REMOVIDO (o destaque ★ IA permanece
   nos rankings/cards).
5. NOMES DE BATERIA: o título visível é SEMPRE 'N<nível> · <nome amigável PT-BR
   não-técnico>' (ex.: P1_CASCADE_C2_E2E_TOKS → "N2 · Velocidade ponta a ponta";
   B0_* → "N1 · Fundação do formato"; C3_*_LINEAR_F0_GATE_F1 → "N3 · Linear inteligente
   (gate)"; CAP_CODING → "N4 · Coding"; C6_* → "N5 · Compilar e executar";
   CMP_* → "N2 · Comparação de gerações"). O battery_id cru aparece SOMENTE em tooltip.
   Mapa `batteryFriendlyName` espelhado nos dois dashboards + fixtures; fallback humanizado
   para ids desconhecidos.
6. CARD POR MODELO (layout canônico, como o design original): dentro de cada grupo de
   bateria, UM card por RECURSO lado a lado — Throughput | RAM necessária | Espaço em
   disco | Ganhos medidos — cada card com a linha "Antes" UMA vez e UMA LINHA POR
   TECNOLOGIA (barra colorida da tech + valor), estilo visual original das barras. É
   PROIBIDO renderizar um card separado por tecnologia para o mesmo grupo e PROIBIDO
   colunas lado a lado por tecnologia/modelo nessa área — tecnologia é LINHA. Modelos
   selecionados aparecem como SEÇÕES empilhadas (um bloco por modelo), não colunas.

## 20. Estrutura de pastas canônica (8º lote, 2026-08-10)

Árvore do repositório (raiz limpa: só web/config/docs de topo):

```
/                     README.md, package.json, vercel.json, index.html, dashboard.html,
                      .env.example, .gitignore, .vercelignore, .gitattributes
/api/                 funções Vercel (inalterado — convenção da plataforma)
/data/                histórico publicado + relatórios (inalterado)
/docs/                documentação; /docs/specs/ ← especificações .txt (movidas da raiz)
/scripts/             smokes, dev server, runner de bancada, validadores (inalterado)
/core/cascade/        PACOTE python compartilhado (ex-`cascade/` da raiz: compiler, kernels,
                      runtime, converter, tests, benchmarks) — nome importável continua
                      `cascade`; consumidores adicionam `<repo>/core` ao sys.path
/engines/rift/        rift_m0_phase1_test_v035_auto_batteries.py
/engines/aether/      aether_m0_phase1_test_v100_auto_batteries.py
/engines/spectra/     SPECTRA_Colab_Test_M0.py
/engines/cascade/     cascade_m0/c0/c1/c2_*_auto_batteries.py
/engines/geyser/      geyser_launcher.py
/engines/winner/      winner_m0_phase1_test_v080_auto_batteries.py; /engines/winner/cpp/ ←
                      ex-winner_cpp/ (o build do winner_m0 usa este caminho no tarball)
/batteries/           baterias multi-motor: c3_methodology, final_phase, capability_eval,
                      gguf_e2e, compare_generations_publisher
```

Regras:
1. `cascade-model-converter/` (duplicata byte-idêntica) foi ELIMINADA — cópia única em
   `core/cascade/converter/`; launchers/README atualizados.
2. LAYOUT NO COLAB NÃO MUDA: as células continuam baixando o pacote para `cascade/...`
   locais e os scripts para o diretório de execução. Todo download de código passa a usar
   pares (caminho_no_repo → caminho_local): ex.
   `core/cascade/compiler/decompose.py → cascade/compiler/decompose.py`.
3. Bootstraps de sys.path dos scripts: probes existentes (/content, /content/*_run, cwd)
   PERMANECEM; adiciona-se probe do layout do repo (`<script>/../../core` a partir de
   engines/*/ ou `<script>/../core` a partir de batteries/) para execução local/clonada.
4. Consumidores a atualizar quando caminhos mudarem: api/test.mjs (mapas de script por tech
   + listas de pacote), api/real-test.mjs, api/runner.mjs, api/geyser.mjs (leitura do
   arquivo + vercel.json functions.includeFiles), dashboards (pkg_files), scripts/
   real_benchmark_runner.py (TECHNOLOGIES), scripts/security_check.mjs (glob de publishers:
   engines/**/*.py + batteries/*.py), smokes (leituras por caminho), README (Diretórios/
   Executáveis). battery_ids, rotas públicas (/rift/... etc.) e artefatos /content NÃO mudam.

## 21. Estrutura de seções do painel + rodadas por série (9º lote, 2026-08-10)

1. SÉRIES DE BATERIA (mapeamento fixo nível→letra): Série A = Nível 1 (M0/fundação, 6
   tecnologias), Série B = Nível 2 (série C do CASCADE + e2e), Série C = Nível 3
   (metodologia C3 ×4 techs), Série D = Nível 4 (capacidades), Série E = Nível 5 (fases
   finais C4–C6). Rótulo visível: "Bateria · Série A" (+ ⓘ com o que a rodada cobre).
2. ORDEM DAS SEÇÕES do painel legado (index.html):
   (1) Ranking Geral — SEM qualquer mensagem textual sobre IA/Gemini (os destaques ★
       permanecem); (2) Ranking WINNER; (3) Baterias por série — UM card por série (A–E)
       com botão único "Rodar Série X (todas as tecnologias)" que gera a célula curta
       (§14.3) com TECHS correspondente: A=[rift,cascade,aether,spectra,winner,geyser],
       B=[c-series], C=[c3], D=[cap], E=[final]; + botão "Rodar todas as séries"
       (TECHS=["all"]) que substitui o antigo "Teste reforçado"; (4) Lista de modelos —
       seções por modelo com os CARDS UNIFICADOS do §19.6 (um card por recurso, linha por
       tecnologia, nomes amigáveis; NUNCA um card por bateria/tecnologia com id cru).
3. Os cards independentes "CASCADE · Série C", "GEYSER" e "Teste reforçado" são REMOVIDOS
   (conteúdo explicativo absorvido nos ⓘ dos cards de série correspondentes).
4. "Comparação de modelos": exatamente UM botão "+ Adicionar modelo" (na linha de ações,
   ao lado de "adicionar todos"); o estado vazio NÃO repete o botão (só texto).
5. Regra geral: NENHUM card/lista pode exibir battery_id cru como título (reincidência do
   §19.5 — vale para TODAS as áreas, incluindo a lista/histórico de modelos).

## 22. MicroLM (10º lote, 2026-08-10) — 7ª tecnologia, tipo MODELO

MicroLM v0.2 é um MODELO de referência (~22M params ativos + tabelas engram; 27 camadas,
mHC lanes/Sinkhorn, Engram hasheado, GQA janela+sinks com RoPE cache-relativo, MLP
Hadamard, init no-op exato), não um otimizador. Arquivos em `engines/microlm/`
(model.py, test_model.py, CHANGES.md, diagram.svg — cópias verbatim do usuário).

1. `technology="MICROLM"` entra nos enums (`api/results.mjs`, `scripts/validate_data.mjs`),
   nas cores/normalizadores dos dashboards (roxo-rosa `#c026d3`/`#e879f9`) e no alias
   `microlm_*`. NUNCA elegível na política do winner (é modelo, não otimizador — como CAP).
   `model_id="microlm/MicroLM-22M-v0.2"`.
2. Bateria: `engines/microlm/microlm_m0_auto_batteries.py` (auto-contida; importa model.py
   do mesmo diretório; sem pytest — checagens embutidas; torch obrigatório):
   - `B0_MICROLM_NOOP_INIT` — propriedade de init no-op exato (‖logits−readout(emb)‖∞
     medido; PASS ≤1e-4) + contagem de params ativos dentro de 20–24M;
   - `P1_MICROLM_DECODE_PARITY` — paridade decode vs forward de treino (config SMALL,
     max abs diff medido; PASS ≤1e-3) + cache limitado em geração longa;
   - `P1_MICROLM_DECODE_TOKS` — tok/s REAL de decode em CPU na config de referência
     (22M; ≥2 warmup + ≥3 medições de 32 tokens, mediana) → `candidate_tok_s` medido
     (baseline null: não há baseline comparável; comparison_role=null);
   - `P1_MICROLM_TRAINS_FROM_INIT` — 30 passos de Adam a partir do init exato (config
     SMALL): loss final < 0.8× inicial (curva registrada; prova de que não há sela
     duplo-zero — FIX 2 do CHANGES.md);
   - `P1_MICROLM_UNIT_CHECKS` — espelho embutido das checagens-chave da suíte
     (FWHT involução/norma, gate do engram vs zona morta legada, matriz Sinkhorn
     duplamente estocástica ~I, sinks visíveis pós-evicção): contagem PASS/FAIL.
   Schema v2 (`benchmark_protocol="MICROLM_M0_V1"`), publisher endurecido padrão,
   bootstrap Colab (launcher baixa model.py junto), RAM VmRSS por fase.
3. Rota `GET /microlm` (sem parâmetro de modelo) → célula Colab via api/test
   (`battery=microlm`); runner (§14): passo `microlm` na expansão 'all'.
4. Exibição: nomes amigáveis N1 ('N1 · Init no-op exato', 'N1 · Paridade de decode',
   'N1 · Velocidade de decode', 'N1 · Treina do init', 'N1 · Checagens de unidade');
   ⓘ da Série A menciona o MicroLM; diagrama vira link/ⓘ (não imagem inline).
5. Futuro documentado (CHANGES.md): KD do Qwen, QAT INT2/INT4 alinhado ao kernel LUT do
   GEYSER, gramática byte-level → function calling — fora do escopo desta integração.

## 23. UI 11º lote (2026-08-10)

1. ROTAS INVERTIDAS: `/` passa a servir o painel completo (index.html — a tela principal);
   dashboard.html vai para `/v2` (e `/legacy` redireciona→`/v2` por compatibilidade).
   Botões de navegação entre as páginas, launcherOrigin e smokes atualizados.
2. ORDEM (index.html): modelos com gráficos primeiro; o "Comparador RIFT × CASCADE × ..."
   desce para logo ANTES do Histórico.
3. LARGURA FLUIDA: containers com max-width fluida (ex.: min(1880px, 96vw)) e grids
   auto-fit — a página ocupa o monitor inteiro, sem margens mortas gigantes.
4. VARIANTES DE BIT: dentro de um grupo, é DESLEAL misturar precisões — um card por
   (recurso × classe de bit), com BADGE da classe ('2-bit', '4-bit', 'ternário', 'INT4',
   'baixo-bit' fallback) derivada do battery_id/metrics (tokens 2BIT/4BIT/TERNARY/INT4/
   INT2/Q4...); máximo UMA linha por tecnologia em cada card. Função
   `batteryBitClass(battery_id, metrics)` espelhada nos dois dashboards + fixtures.
5. O card "WINNER executa"/"Arquitetura do WINNER (seleção dinâmica)" é REMOVIDO do
   index.html (a lógica selectWinnerArchitecture permanece para rankings/battery).
6. VALIDAÇÃO VISUAL OBRIGATÓRIA antes de concluir qualquer onda de UI: abrir as duas
   páginas no navegador com os dados reais e conferir estrutura + console limpo
   (+ screenshot como evidência quando houver mudança de layout).

## 24. Painel único (12º lote, 2026-08-10)

1. EXISTE UMA ÚNICA PÁGINA: index.html em `/`. dashboard.html é REMOVIDO do deploy
   (arquivo deletado; rotas `/v2` e `/legacy` deixam de existir; dev_server/vercel/smokes/
   README atualizados). Os gráficos exclusivos do painel resumido são PORTADOS para uma
   seção "Gráficos" do principal: tok/s por modelo (barras baseline×candidato), speedup
   por bateria (marcador 1.0x), redução de disco/RAM por tecnologia, latência mediana
   baseline×candidato por bateria primária (a série temporal de tok/s já existe no
   principal — manter uma só). Espelhamentos "entre páginas" (§1/§8/§19.5/§23.4) passam a
   valer entre index.html ↔ api/results.mjs ↔ winner_m0 (4 implementações da política do
   winner: analyze.mjs fórmulas, results.mjs, index.html, winner_m0 Python).
2. A seção "Comparação de gerações" é REMOVIDA da UI (dados CMP_* permanecem no histórico
   e caem na regra 3; o publicador compare_generations_publisher.py permanece).
3. REGRA GERAL DE RELEVÂNCIA: nenhum card/gráfico renderiza com apenas UM modelo/uma
   tecnologia — cards de comparação de tecnologias exigem ≥2 linhas de tecnologia; gráficos
   por modelo exigem ≥2 modelos; abaixo disso o card é OMITIDO (sem estado vazio; os dados
   continuam no Histórico). Exceções: o histórico consolidado e os cards de série (A–E),
   que não são comparações.

## 25. Score canônico v2 — objetivo "computador convencional" (13º lote, 2026-08-10)

OBJETIVO DECLARADO do score: identificar a melhor tecnologia para rodar LLM em um
computador convencional — 4 núcleos, 8 GB de RAM livre, SEM GPU. Nesse cenário a RAM é a
restrição dura (o modelo precisa caber), a velocidade em CPU é o gargalo de uso e o disco
é o recurso mais barato. Novos pesos canônicos (`SCORE_WEIGHTS_V2`, substituem os do §1
em TODAS as implementações espelhadas na mesma mudança):

| Métrica | Peso antigo | PESO NOVO |
| --- | --- | --- |
| `output_cosine` | 25 | **25** |
| `output_nrmse` | 15 | **10** |
| `quality_gate_pass` | 5 | **5** |
| `ram_reduction_pct` | 15 | **30** |
| `operation_speedup_x` | 20 | **20** |
| `disk_reduction_pct` | 20 | **10** |

(Qualidade 40% • RAM 30% • latência/velocidade 20% • disco 10% — RAM > disco por
construção.) As fórmulas de NORMALIZAÇÃO do §1 não mudam; apenas os pesos. O fator de
coverage permanece. Implementações a atualizar juntas: api/analyze.mjs (fórmulas +
prompt/enum), api/results.mjs (winnerRecordScore), index.html (SCORE_WEIGHTS + textos
visíveis do ranking, ex.: "Qualidade 40% • RAM 30% • latência 20% • disco 10%"),
winner_m0_phase1_test_v080_auto_batteries.py (_record_score). Fixtures de smoke com
valores exatos devem ser RECALCULADAS. A tabela de pesos do §1 passa a referenciar esta
seção.

## 26. Card do Conversor (14º lote, 2026-08-10)

Fluxo completo: baixar modelo da HF → converter para CASCADE-DIR (pacote-por-pacote) →
ENVIAR o modelo convertido de volta ao Hugging Face Hub (repo do usuário).

1. `GET /converter.py` (rota → `api/converter.mjs`): serve um runner LOCAL auto-contido
   (gerado no servidor com repo/ref resolvidos): CLI `--model <org/modelo>` OU
   `--input <pasta local>`, `--output` (default `<modelo>-cascade`), `--hf-repo <destino>`
   (opcional: upload via huggingface_hub `create_repo(exist_ok)`+`upload_folder`, exige
   HF_TOKEN de ESCRITA no env — nunca logado), `--publish` (repassa ao conversor),
   passthrough dos flags do conversor (defaults pacote-por-pacote: `--disk-budget-gb 75
   --resume`; `--delete-source-shards` só com download próprio). O runner baixa os 4
   arquivos do conversor de `core/cascade/converter/` no ref pinado, faz
   `snapshot_download` quando `--model` (apenas *.safetensors + config/tokenizer),
   converte e (se pedido) sobe. `Content-Disposition: attachment` (é o botão de download).
2. `GET /converter/:model*` (→ api/test `battery=converter`): célula Colab completa —
   Secrets (HF_TOKEN write obrigatório p/ upload; RIFT_INGEST_TOKEN opcional p/ --publish),
   deps pinadas (torch/safetensors/numpy/huggingface_hub), curl do `/converter.py` e
   execução com `--model <modelo> --hf-repo <destino>`; params de query: `hf_repo`
   (validado `org/nome`), `publish=on|off`.
3. UI (index.html): card "Conversor de modelos" com campo de modelo (reutiliza o
   picker/campo existente), campo "repo de destino no HF" (placeholder
   `seu-usuario/<modelo>-cascade`), botões lado a lado "Copiar célula Colab" e
   "Baixar script (rodar no PC)" (link `/converter.py`); ⓘ com requisitos (token de
   escrita, upload torna o repo público/privado conforme a conta, orçamento de disco).
   Card operacional — isento da regra §24.3.
4. Segurança: nenhum token em código/URL; upload é ação do usuário no ambiente dele.

## 27. Conversor: entrada GGUF + escada de codecs (15º lote, 2026-08-10)

Objetivo declarado: rodar modelos MAIORES em PC convencional (4 núcleos, 8 GB
livres, sem GPU) **sem perder inteligência**. A evidência medida do próprio
projeto (data/compare_generations_report.json) mostra que 2-bit PTQ uniforme
destrói o modelo (ternário → PPL 41,5M; INT2 → 22k; INT4 → 55,9 vs 49,1 do
original), então a compressão precisa ser DECIDIDA POR MEDIÇÃO, tensor a
tensor, e nunca imposta uniformemente.

1. `GGUFSource` em `core/cascade/converter/cascade_converter.py`: entrada
   `.gguf` com streaming por blocos de linhas (pico = chunk × colunas × 4 B,
   não o tensor inteiro). Shape lógico = shape do ggml invertido. Passthrough
   exato copia os blocos GGUF originais e rotula `source_dtype=GGUF_<QTYPE>`.
   Dependência OPCIONAL `gguf>=0.10,<1` (só para entrada .gguf) — sujeita à
   homologação de TI/SI como as demais.
2. Escada de codecs por tensor (`--codec-ladder`): `safe` (padrão) =
   int4/g64 → int4/g32 → raw; `compact` = int2/g64 → int4/g64 → int4/g32 → raw;
   `int4` = comportamento anterior. INVARIANTE: o gate de qualidade
   (`--cosine-min`/`--nrmse-max`) é idêntico em TODOS os degraus; um degrau só
   é escolhido se passar, e se nenhum passar o tensor cai em passthrough exato.
   A escada reduz bytes sem afrouxar qualidade — nunca degrada em silêncio.
   O degrau `int4/g32` existe para RESGATAR tensores que hoje caem em raw
   (16 bpw → 4,5 bpw).
3. Codec INT2: groupwise ASSIMÉTRICO min-max, 4 níveis, escala+mínimo FP16 por
   grupo (`INT2_GROUP_ASYMMETRIC_MINMAX`, 2 + 32/group bpw), arquivos
   `f0.int2` + `f0.scales.f16` + `f0.mins.f16`. Todo stage 0 declara `codec`,
   `bits` e `effective_bits_per_weight`; leitores DEVEM despachar por
   `representation` e nunca assumir INT4.
4. `summary.residency`: HOT (F0+raw, residente) vs WARM (F1, paginável), bpw
   médio medido e veredito `fits_resident_in_target` contra `--target-ram-gb`
   (padrão 8,0) descontando 1,5 GiB de reserva para KV-cache/ativações.
   `summary.conversion_peak_rss_bytes` registra o pico de RSS medido.
5. `--ram-budget-mb` (padrão 16) reduz `--chunk-rows` em tensores muito largos.
6. NÃO adotado: o formato WINR-F0 2-bit uniforme (`convert_gguf_to_winr_f0.py`).
   Motivos medidos/verificados: limiar `0.5×absmax` com escala `absmax` zera
   ~99% dos pesos (cosine ≈ 0,26 em simulação com estatística de LLM); sem gate
   de qualidade; grava container versão 0x0200 que o leitor C++ do repo rejeita
   (exige 0x0100); desquantiza o tensor inteiro (pico de GBs). O que foi
   aproveitado da proposta: a ENTRADA GGUF e a disciplina de empacotar por
   blocos de linhas.

## 28. Orçamento de RAM pela máquina + RAM por largura de bits (16º lote)

1. ORÇAMENTO AUTOMÁTICO. `--target-ram-gb 0` (novo padrão) = detecta a RAM
   física e reserva 8 GiB para SO/apps, com piso de 50% do total:
   16 GiB→8 · 24 GiB→16 · 32 GiB→24 · 64 GiB→56 · 8 GiB→4 (piso).
   Sem detecção, cai no alvo canônico de 8 GiB. Detecção sem dependências
   novas: `/proc/meminfo` → `os.sysconf` → `GlobalMemoryStatusEx` (Windows).
   `--ram-budget-mb 0` (novo padrão) escala a fatia de conversão com a
   máquina: `clamp(total/512, 16 MB, 128 MB)`. Ambos os flags aceitam valor
   explícito, que sempre vence o automático (`target_source` registra qual foi).
2. RAM POR LARGURA DE BITS (estilo cartão de modelo do Hugging Face).
   `summary.residency.memory_by_bits` traz, para 1/2/3/4/5/6/8 bits e fp16, os
   bytes residentes e `fits_in_target`. É PROJEÇÃO: recalcula apenas o estágio
   base com cada largura hipotética (mantendo o overhead de escala por grupo) e
   soma o passthrough exato REAL, que não é quantizado. A linha `medido` é a
   residência real da conversão (`resident_hot_bytes` + bpw da escada).
   Rotulagem obrigatória `PROJETADO` + a nota de que 1–2 bits uniformes NÃO
   preservam qualidade neste projeto (medido: PPL 41,5M em ternário e 22k em
   INT2 contra 49,1 do original).
3. CARD "Modelos convertidos" (index.html, `renderConvertedModels`): lê os
   registros `battery_id=CASCADE_MODEL_CONVERSION` e `metrics.converter`
   (bloco novo do `dashboard_battery.json`), mais recente por modelo. Mostra
   residente/paginável/passthrough, bpw médio, degraus escolhidos, veredito
   contra o orçamento da máquina e a tabela por bits com barras verde/vermelho.
   Card operacional/informativo — ISENTO da regra §24.3 (um único modelo
   convertido já é informação útil).

## 29. Conversor: decisões auditáveis por tensor (17º lote)

Origem: bateria real no Muse Glimmer (fonte GGUF IQ2). Achados medidos que
motivam esta seção — INT2/g64 nunca passou o gate (cosine 0,91–0,92); INT4/g64
serve Q e gate (~0,997), K/V (~0,994) e `attn_output` (~0,992); F1 rank 8→32 não
move `attn_output`; a conversão frequentemente AUMENTA bytes contra a fonte IQ2.

1. ESCADA AUTOMÁTICA PELA FONTE (`--codec-ladder auto`, novo padrão).
   `source_is_low_bit(desc)` = `^GGUF_` exceto F32/F16/BF16. Fonte low-bit →
   `safe` (não tenta INT2, que nunca passa e cujo raw custa 2,66 bpw e não 16);
   BF16/F16/F32 → `compact`. Modo explícito sempre vence o auto.
   `ladder.mode` e `ladder.requested_mode` registram os dois.

2. GUARDA DE EXPANSÃO DE BYTES (padrão ligado; `--allow-byte-expansion`
   desliga). Antes de escrever cada F0, `projected_f0_bytes()` compara com
   `desc.nbytes`: se o degrau ficaria ≥ à fonte, o degrau é PULADO sem gravar
   (`attempts[].skipped = "projected_byte_expansion"`) e a escada é interrompida
   (`ladder.stopped_by = "byte_expansion_guard"`), porque a escada é crescente em
   bpw. A checagem se repete no total F0+F1 (`byte_expansion_with_f1`). O
   fallback é o passthrough exato, com `reason` explicando a decisão. Medido em
   `o_proj` 256×512 com fonte a 2,66 bpw: 43 581 B exatos contra 69 632 B com
   perda se a guarda for desligada.

3. PASSTHROUGH SEM CÓPIA (`--keep-source-passthrough`). Tensores não elegíveis
   viram `SOURCE_EXTERNAL`: `files: {}`, `bytes: 0`, `external_bytes` = tamanho
   real, `requires_source_file: true`. Substitui a cópia byte a byte que estourava
   RAM/disco em embedding grande. `residency` soma `external_bytes` no HOT —
   economiza DISCO, não RAM de execução — e marca
   `bundle_requires_source: true`; o bundle deixa de ser autocontido.
   `verify_tensor_outputs` valida a existência do checkpoint em vez do arquivo de
   estágio; `estimate_tensor_output_peak(..., keep_source=True)` projeta 0 bytes.

4. PISO DE ENERGIA DO F1 DERIVADO DO GATE (sem constante mágica).
   `required_capture_fraction(f0_metrics, cosine_min, nrmse_max)` =
   `max(1 − (nrmse_max/nrmse_f0)², 1 − (1−cosine_min)/(1−cosine_f0))`. O F1 é
   abortado quando a energia capturada pelos ranks disponíveis fica abaixo de
   `F1_ENERGY_SAFETY = 0.75` desse mínimo (`trigger: below_gate_requirement`),
   evitando o passe de avaliação e a gravação. A margem existe porque o vínculo
   do cosseno usa `1−cos ≈ nrmse²/2` (2ª ordem) — na dúvida, NÃO abortar.
   `--f1-min-energy` (padrão 0) é piso absoluto adicional
   (`trigger: below_explicit_floor`). Medido: resíduo INT4 entrega
   `captured_fraction ≈ 0,247` com rank ≤ 32, enquanto `attn_output` precisaria
   de 0,375 — o que explica o rank 8→32 não mover a métrica.

5. INVARIANTE MANTIDO: nenhuma das quatro mudanças afrouxa o gate de qualidade.
   Todas atuam sobre CUSTO (bytes, disco, CPU) ou sobre trabalho comprovadamente
   inútil. Quando nada passa, o resultado continua sendo passthrough exato.

6. RELATÓRIO NÃO PODE MENTIR SOBRE A ESCADA. Com `auto`, a escada é decidida
   POR TENSOR: `summary.codec_ladder` guarda o modo PEDIDO e
   `summary.codec_ladder_resolved` conta os modos realmente usados
   (`{"safe": 168}`); `ladder_rungs` segue o modo dominante. O bloco
   `metrics.converter` publicado no dashboard leva os dois, mais
   `bundle_requires_source` e `external_source_bytes`.

7. EXPOSIÇÃO NO COLAB. `--keep-source-passthrough` chega ao caminho que o
   usuário realmente usa: `keep_source=on|off` na rota `/converter/<modelo>`
   (validado com 400 em valor inválido) → `KEEP_SOURCE_MODE` na célula →
   `--keep-source-passthrough` no runner. Checkbox "Não copiar tensores fora do
   CASCADE" no card do conversor, desligado por padrão, com o custo (bundle
   deixa de ser autocontido) no ⓘ. O runner local também repassa
   `--codec-ladder`, `--include-regex` e `--target-ram-gb`.
