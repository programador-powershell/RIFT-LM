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

8. DESPACHO DE FORMATO POR MAGIC BYTES (bug real, reportado em teste com o
   cache do Hugging Face). `convert()` faz `Path(args.input).resolve()`, e
   `resolve()` SEGUE symlink: em `~/.cache/huggingface/.../snapshots/<rev>/
   model.gguf → blobs/<sha256>` o alvo NÃO tem extensão, então o despacho por
   sufixo mandava o GGUF para o `SafeTensorSource` e o erro saía como
   "Header truncado". `sniff_container(path)` passa a decidir pelos magic bytes
   (`GGUF` → gguf, `PK` → npz, ausência de magic → safetensors) e o sufixo fica
   só como FALLBACK, para que um `.gguf` vazio ou truncado ainda falhe COMO
   GGUF. Regra: nenhuma decisão de formato pode depender do NOME do arquivo.
   `maybe_delete_source_shard` mantém de propósito o teste por sufixo
   `.safetensors` — é uma guarda destrutiva que precisa falhar fechada
   (um blob resolvido sem extensão é PRESERVADO, nunca apagado).

9. FOOTPRINT HONESTO — `all_in_ram_bytes` É A MÉTRICA DE HEADLINE (bug real,
   medido no Muse-Glimmer-30B). Com `SOURCE_EXTERNAL` o bundle fica pequeno
   PORQUE os bytes não foram copiados: comparar bundle-vs-fonte publica um ganho
   inexistente. Medido: fonte GGUF 10,73 GB → bundle 1,56 GB + externo 8,48 GB;
   o registro publicava `disk_reduction_pct: 85,51` e `ram_reduction_pct: 85,52`
   quando a redução REAL é 6,5%.

   Regras:
   - `candidate_disk_bytes` / `rift_disk_bytes` = `bundle + external_source_bytes`
     (o disco EXIGIDO para rodar). Igual ao bundle quando não há external.
   - `gains.disk_reduction_pct` e `disk_compression_ratio_x` derivam desse total.
   - `gains.ram_reduction_pct` e `metrics.memory.estimated_candidate_bytes`
     derivam de `all_in_ram_bytes` (F0 + passthrough exato + F1), NUNCA dos
     bytes de estágio do bundle.
   - `summary.required_disk_bytes` / `required_disk_reduction_pct` são os campos
     honestos; `cascade_bundle_directory_bytes` e `bundle_disk_reduction_pct`
     continuam existindo como DETALHE, nunca como headline.
   - `metrics.converter` publica `all_in_ram_bytes`, `bundle_bytes`,
     `required_disk_bytes` e `headline_metric: "all_in_ram_bytes"`.
   - Console: com external > 0 é proibido imprimir "Disk reduction" bundle-vs-
     fonte; imprime-se o disco exigido, a redução REAL e, entre parênteses, o
     número bundle-only com a ressalva de que o bundle depende da fonte.
   - Painel (`renderConvertedModels`): ordena os cards por `all_in_ram_bytes`
     ASCENDENTE (menor primeiro) — nunca por data e nunca por redução de disco;
     primeira métrica do card é "TOTAL em RAM", seguida de "Redução real vs
     fonte"; `bundle_requires_source` vira o selo "depende da fonte" e a linha
     "Fora do bundle (na fonte)". Registros antigos sem `all_in_ram_bytes` caem
     no fallback `resident_hot_bytes + pageable_warm_bytes`.

   Princípio: um bundle que DEPENDE do checkpoint de origem não reduziu o
   footprint — apenas mudou onde os bytes moram.

10. POLÍTICA FASE-1 VALE NOS DOIS ESQUEMAS DE NOME (bug real, reportado no teste
    do Muse-Glimmer-30B). `eligible_matrix` testava por substring com nomes
    SOMENTE de HF (`embed_tokens`, `embedding`, `lm_head`, `expert`, `.moe`).
    Nenhum deles casa o esquema do ggml, então em entrada `.gguf` a política era
    **100% inerte** — `token_embd.weight`, `output.weight` e `ffn_*_exps` entravam
    como elegíveis. Medido: `token_embd` do Muse é 202048 × 6656 = 1,34 G
    elementos; no IQ2 a guarda de expansão o mandou para raw por acidente, mas
    numa fonte BF16 (2,69 GB no tensor contra 0,76 GB de F0 INT4/g32) a guarda
    NÃO dispara e os dois maiores tensores do modelo entram no pipeline F0/F1.

    - `EMBEDDING_NAME_RE`: `embed_tokens|embeddings?|token_embd|tok_embeddings|wte`
    - `OUTPUT_HEAD_NAME_RE`: `lm_head|output_projection` por segmento **mais**
      `^output(\.weight|\.bias)?$` ANCORADO NO INÍCIO. O ancoramento é
      obrigatório: `blk.N.attn_output.weight` contém "output" e é um linear
      legítimo — um teste por substring mataria todas as projeções de atenção.
      `output_norm.weight` também não pode casar (sai por dimensão, sendo 1D).
    - `MOE_NAME_RE`: `experts?|moe` por segmento mais o sufixo `_exps`.
    - `--include-embeddings` cobre embeddings E cabeça de saída (comportamento
      anterior preservado); `--include-moe` cobre só MoE.
    - Razões renomeadas para distinguir a decisão: `embedding_passthrough_phase1`,
      `output_head_passthrough_phase1`, `moe_passthrough_phase1`.
    - `DEFAULT_EXCLUDE` foi REMOVIDO: era código morto com aparência de política
      oficial (definido no topo, referenciado em lugar nenhum).

    Regra: toda política que depende de nome de tensor precisa cobrir HF e ggml,
    e o smoke tem anti-regressão para `attn_output` continuar elegível.

11. CONVERSÃO PARCIAL NUNCA PARECE COMPLETA (teste BF16 do Muse-Glimmer-30B).
    O shard 1 do BF16 tem 49,95 GB — maior que o disco livre da máquina de teste
    —, então a medição foi por AMOSTRA: camadas 0, 25 e 51 lidas por HTTP Range
    (2,90 GB) e convertidas de fato. Regras para registrar isso:

    - Os campos de comparação (`baseline_disk_bytes`, `candidate_disk_bytes`,
      `gains.*`) carregam SOMENTE a amostra medida. A extrapolação do modelo
      inteiro vive em `metrics.converter.projection` com `label: "PROJETADO"` e
      não entra em campo de topo (§28.2). O smoke checa a ordem de grandeza:
      `candidate_disk_bytes < 1 GB` contra os ~20 GB da projeção.
    - `metrics.converter.sample_scope` declara `layers_measured`,
      `layers_total`, o `why` (por que houve amostragem) e a
      `extrapolation_basis`. O card exibe o selo de aviso com o `label`.
    - `measurement_scope` diz em texto que é amostra e NÃO o modelo inteiro.
    - Número não derivável fica `null`, nunca estimado: `f0_effective_bits_per_
      weight` é null porque os 3 raw e os 12 norms entram nos bytes de estágio
      sem separação por tensor. O agregado auditável vai em
      `aggregate_bits_per_weight` (4,7482 bpw na amostra).
    - VEREDITO DE RESIDÊNCIA EXIGE ALVO. Sem `fits_resident_in_target` E
      `target_ram_gb`, o card mostra "alvo não declarado" — antes renderizava
      "não cabe em null GB", afirmando o que o dado não sustenta.
    - Tamanhos de outros formatos entram como `external_reference_bytes`:
      referência publicada de terceiros, nunca medição deste pipeline.

    `LOCAL_WEIGHT_GATE_PASS` foi acrescentado a `KNOWN_STATUSES` de
    `scripts/validate_data.mjs`: o conversor emite esse status desde sempre e o
    validador o rejeitava, então NENHUM registro de conversão publicado passaria
    pelo `npm test`. Mapear para `PASS` superestimaria (lê como certificado) e
    para `EXPERIMENTAL_PASS` perderia a distinção que o protocolo preserva.

    Evidência medida que o registro guarda em `ladder_evidence`, porque decide o
    roadmap do codec: INT2 min-max uniforme NÃO passa o gate nem em fonte BF16
    (melhor cosseno 0,9139 contra gate 0,995) — o degrau int2 da escada compact é
    peso morto sem calibração por ativação. E o F1 não engajou porque o resíduo
    capturou 0,064 contra os 0,190 exigidos pelo gate (§29.4): o early-abort
    derivado disparou corretamente em produção, e o formato efetivo é INT4
    groupwise puro (~4,75 bpw agregado).

## 30. Motor de execução v2 e o bug do Confidence Gate (17º lote)

Origem: auditoria externa do `runtime.zip`, com os três defeitos reproduzidos
nesta árvore antes de qualquer mudança.

1. GATE v0 NÃO FILTRAVA NADA (bug de correção, o mais grave).
   `decide_gate` tirava o threshold de `torch.quantile` sobre o PRÓPRIO batch.
   Em decode autorregressivo o batch é 1 token, então o quantil de um elemento
   É aquele elemento e a máscara virava `score >= score` — sempre True. Medido
   no harness, a taxa era função APENAS do tamanho do batch, idêntica em 8
   sorteios por tamanho:

   | batch | 1 | 2 | 3 | 4 | 8 | 16 |
   | --- | --- | --- | --- | --- | --- | --- |
   | taxa v0 | 1,000 | 0,500 | 0,333 | 0,250 | 0,375 | 0,3125 |

   O "Confidence Gate" media o tamanho do batch, não a confiança da ativação.
   v1: `GateCalibrator.observe/freeze` calibra o threshold offline e congela em
   `GateConfig.fixed_threshold` (para gravar no bundle); precedência
   `argumento explícito → fixed_threshold → percentil do batch (só com
   batch >= min_batch_for_batch_percentile, default 8)`. Sem nenhum dos três, o
   gate escolhe F0_ONLY e emite `warning` na telemetria com
   `gate: "UNCALIBRATED_SMALL_BATCH_F0_ONLY"` e `threshold: None` — nunca uma
   taxa inventada. `features` continua no meta (compat v0); `activation_rate` é
   a única chave que os callers leem.
   NENHUM registro publicado carregava taxa de gate CASCADE, então o histórico
   não foi contaminado; o bug era latente para o próximo run de
   `c3_methodology_auto_batteries.py` e `final_phase_auto_batteries.py`, que
   importam `decide_gate`.

2. CACHE FP32 RESIDENTE ANULAVA O FORMATO. `block_runtime.py` mantinha
   `_w0_cache` com W desquantizado em fp32: 32 bpw residentes num formato de
   4,5 bpw. Medido: 2,01 GB contra 0,28 GB (7,18×). O v2 desquantiza DENTRO do
   produto (`q4k_gemv`) e `w0_cache_bytes` é 0 por construção.

3. "VETORIZAÇÃO" AVX2 QUE ERA GATHER ESCALAR. `cpp/avx2_lowrank.cpp` montava o
   vetor com `_mm256_set_ps` a partir de 8 cargas escalares em stride `rank`
   (V é `(in, rank)` row-major), além de um loop `for (r)` cujo corpo calculava
   `acc` e `Vr` sem usar nenhum dos dois. O arquivo fica como referência do
   caminho C1 com os dois defeitos anotados; o kernel vivo é
   `runtime_v2/kernels/kernels.c`, que transpõe V para `(rank, in)` e lê
   contíguo com FMA.

4. REGISTRO DO MOTOR É SINTÉTICO E NÃO SOBE PARA CAMPO DE TOPO.
   `P1_CASCADE_RUNTIME_V2_KERNEL` (protocolo `RUNTIME_KERNEL_SYNTHETIC_V1`) mede
   um stack sintético de 2,01 GB fp32 / 0,28 GB q4k, NÃO `model.generate`.
   Portanto `baseline_tok_s`/`candidate_tok_s`/`*_ram_bytes` de topo são `null`
   por design e os números medidos ficam em `metrics.runtime`, com
   `scope: "SYNTHETIC_STACK"` e a nota de que aquele tok/s não é comparável com
   as baterias E2E. `eligible_for_primary_ranking: false`. Grupo próprio
   `P1 · Motor de execução` (o fallback `P1 · Codec principal` descreveria a
   coisa errada) e nome amigável "Motor de execução (kernel C)".

5. F1 CONTINUA NÃO SE PAGANDO. Medido neste stack: cosseno 0,996689 com F0
   contra 0,996719 com F0+F1 — **+3e-5** por um GEMV extra. Terceira medição
   independente na mesma direção (§29.4 derivou o piso, §29.11 mediu 0,064 de
   energia capturada em BF16): o resíduo de quantização é ruído de posto alto e
   rank ≤ 32 não o resgata. Enquanto isso valer, o formato efetivo do CASCADE é
   INT4 groupwise puro e o F1 é opcional.

## 31. Conversor v2 (CASCADE-Q4K/2.0) — grava no formato do executor (18º lote)

`core/cascade/runtime_v2/convert.py`. Fonte BF16/F16/F32 apenas: recomprimir
GGUF é beco sem saída medido (§29.9). Ganhos medidos na amostra do
Muse-Glimmer-30B (3/52 camadas + embeddings/lm_head integrais): 2,90 GB → 0,78 GB
(73,1%) em 99,7 s, contra 69,35% em 198,7 s do v1 — 2× mais rápido e 3,7 pp
menor, ao custo de 7,3× no pico de RSS de conversão (0,43 → 3,14 GiB, ainda cabe
em 8 GB; `CHUNK_ROWS` menor reduz).

1. SAÍDA NO FORMATO DO EXECUTOR: blobs de 138 B (g=64) / 144 B (g=32) por
   super-bloco, prontos para mmap + GEMV fundido. Zero repack na carga — era a
   maior perda do v1, que gravava um int4 próprio e obrigava o executor a
   desquantizar para FP32.
2. ESCADA `q4k/g64+clip → q4k/g32+clip`, sem fallback raw para 2D. Na amostra:
   34 tensores em g=64 (4,3125 bpw), 2 em g=32 (4,5), ZERO raw — o `o_proj/L51`
   que o v1 mandava a 16 bpw passou a 4,31. Os degraus q5k/q6k estão PLANEJADOS
   e não implementados (dependem de kernel 5/6-bit): `LADDER` é a fonte da
   verdade, não o docstring.
3. EMBEDDINGS/LM_HEAD QUANTIZADOS com cosseno MEDIDO (0,996976 e 0,996913) em
   vez do raw 16 bpw do v1 — a maior parcela da economia (~4 GB no 30B).

4. MUDANÇA DE POLÍTICA QUE ROMPE O §29.5. Sem fallback raw, quando nenhum degrau
   passa o gate o último é gravado ASSIM MESMO (`RESCUE_LAST_RUNG`) e o tensor
   sai com `quality_flag="abaixo_do_gate_verificar_e2e"`. O invariante do v1 era
   "quando nada passa, o resultado continua sendo passthrough exato"; o v2 troca
   16 bpw exatos por 4,5 bpw com perda DECLARADA. O trade é legítimo, mas o
   bundle deixa de ser aprovado por construção, então o resumo do manifesto
   passa a expor `all_tensors_passed_gate`, `below_gate_tensor_count`,
   `below_gate_tensors` e `gate_policy`, e o console imprime um ATENÇÃO com os
   nomes. Consumidor de bundle DEVE checar `all_tensors_passed_gate`.

5. BUG CORRIGIDO NO RELATÓRIO DE RESIDÊNCIA (subtração dupla). `residency_report`
   recebia `MACHINE_CLASSES_GIB = (8,16,24,40)` — que já eram orçamentos — e
   ainda fazia `budget = cls - 8`, rotulando a linha como `maquina_{cls+8}gb`.
   Resultado: a linha rotulada "maquina_24gb" recebia orçamento de 8 GiB e
   imprimia NÃO CABE para um bundle de 15,54 GiB que CABE em 24 GB pela regra do
   §28. O `pct_final.json` do teste externo afirmava o correto ("24GB/16GiB:
   CABE") e portanto NÃO foi gerado por esta função. Corrigido para
   `MACHINE_TOTAL_GIB = (16,24,32,48)` com `budget = max(total-8, total/2)`, e o
   relatório passa a trazer `folga_gib` e `regra_orcamento` explícitos.

6. MARGEM É PARTE DO VEREDITO. O "cabe em 24 GB" do 30B tem folga de 0,46 GiB
   sobre uma PROJEÇÃO ×52 de ponto único (o registro anterior reportava faixa
   best/central/worst). O ponto de ruptura é **+3,24%** de erro na projeção;
   acima disso o veredito deixa de valer, e o `worst` do intervalo anterior
   (17,98 GB) não cabia. Por isso o card exibe "Folga em 24 GB" com selo
   PROJETADO e a nota do ponto de ruptura ao lado — um "cabe" por 0,46 GiB não é
   a mesma afirmação que um "cabe" por 8,46 GiB.

7. TOK/S DO RELATÓRIO EXTERNO NÃO ENTRA NO REGISTRO. As comparações de tok/s
   (−3,3% vs Q4_K_XL, +96,2% vs IQ2_XXS, >10× na máquina de 24 GB) derivam de
   RAZÃO DE BANDA DE MEMÓRIA e de modelagem de paginação, não de
   `model.generate`. Ficam fora do registro inteiro — nem em `metrics` — para não
   serem confundidas com o tok/s medido das baterias E2E.

## 32. Merge do conversor v2 pós-review e regra de dispersão (19º lote)

O autor do conversor reimplementou as correções do §31 no estilo próprio. Regra
seguida (a mesma do GEYSER, §18.1): atualização de ciência é MERGE, nunca
substituição da camada de publicação.

1. FORMA DO RELATÓRIO DE RESIDÊNCIA MUDOU e a nova é melhor: a chave da classe
   passa a ser `maquina_<total>gb` com `orcamento_gib`, `cabe` (booleano) e
   `folga_gib` como campos — antes o orçamento estava embutido no nome da chave.
   A subtração dos 8 GiB acontece em UM lugar só, com anti-regressão versionada
   em `tests/test_residency.py` (inclui o limiar exato 14,50 GiB CABE / 14,51
   NÃO CABE). Reposto por cima: `kv_runtime_reserve_gib` NUMÉRICO ao lado da
   frase `regra_orcamento` (consumidor de JSON não deve parsear texto para achar
   a reserva) e `gate_policy` no resumo, porque um bundle que rompe o invariante
   §29.5 precisa ser autodescritivo — quem lê o manifesto pode não ter lido o
   MIGRACAO.md.

2. DOCSTRING AINDA TINHA DUAS AFIRMAÇÕES VENCIDAS: listava as classes de máquina
   como `8/16/24/40` (são os ORÇAMENTOS, não os totais — o resíduo exato do bug
   do §31.5) e dizia "144 B/super-bloco" para todo caso, quando g=64 usa 138 B.
   Corrigidas.

3. FAIXA DA PROJEÇÃO, e ela responde à objeção do §31.6 melhor do que a minha
   margem crua: central 15,08 GB (folga +0,46 GiB), worst realista 15,19 (+0,35),
   worst estrutural 15,67 (−0,09). O ponto de ruptura de +3,24% exigiria bpw
   médio de ~4,47, isto é ~96% dos tensores em g32 — medido na amostra: 5,6%. O
   veredito "cabe em 24 GB" sobrevive ao worst realista e cai apenas no worst
   estrutural, por 0,09 GiB. Só a conversão integral (~32 min) fecha o binário.

4. REGRA NOVA — DISPERSÃO ENTRE RUNS. Três execuções do MESMO benchmark
   sintético do kernel, sem mudança de código entre elas:

   | run | v2 tok/s | low_mem | default | speedup vs low_mem |
   | --- | --- | --- | --- | --- |
   | A | 37,88 | 0,869 | 21,75 | 43,6× |
   | B | 38,36 | 0,924 | 22,33 | 41,5× |
   | C | 35,53 | 0,977 | 23,56 | 36,4× |

   Dispersão: 8,0% no tok/s do v2, **19,9% no speedup de manchete**, 15,5% no
   speedup contra o default. A primeira versão do registro publicou 43,6× e
   1,74× — o MELHOR dos três — porque era o único run disponível na época.

   Regra: benchmark com mais de um run publica `<metrica>_range` com
   `{min, max, median}` e `runs`/`runs_detail`; o campo de ponto único carrega a
   **MEDIANA**, nunca o melhor. O smoke falha se o ponto único voltar a ser o
   máximo (`speedup_vs_legacy_low_mem_x < max(runs)`), e `dispersion_note`
   registra a dispersão medida.

   Consequência sobre o §30.5: o ganho de cosseno do F1 (+3e-5) é uma ordem de
   grandeza MENOR que os 8% de dispersão do próprio benchmark — naquele stack o
   F1 é indistinguível de ruído, e a nota do registro passou a dizer isso.

## 33. O gate de cosseno NÃO protege perplexidade (20º lote)

Primeira medição END-TO-END do projeto, no `Qwen/Qwen2.5-0.5B` — modelo pequeno
escolhido porque os dois formatos o rodam (a Muse-Glimmer não carrega no
llama.cpp). Registro `P1_CASCADE_PPL_E2E`, protocolo `PPL_E2E_V1`.

1. ACHADO QUE MUDA O CONTRATO. O gate 0,995 aprovou **todos** os tensores e a
   configuração aprovada degradou a perplexidade em **+29,1%** (18,93 → 24,43).
   Decomposição medida, em pontos de PPL: peso g32 data-free **+2,38** · degrau
   g64 **+1,44** · ativação int8/grupo-256 **+0,59** · cabeça 4,5 bpw **+1,09**
   (soma 5,50 — verificada, reproduz o +29,1% sobre 18,93).

   Consequência: o gate de cosseno/NRMSE é **PRÉ-FILTRO**, nunca veredito.
   Passar o gate significa "a reconstrução do peso é próxima", não "o modelo
   continua bom". Toda afirmação de qualidade anterior baseada em cosseno —
   §29.4, §29.11, §31 — descreve proximidade de PESO, não qualidade de MODELO.
   `convert.py` passa a expor `gate_role: "PREFILTER_NOT_VERDICT"`,
   `end_to_end_validated: false` e `gate_vs_ppl_evidence` no resumo, e imprime o
   aviso no fim de TODA conversão — principalmente quando tudo passou.

2. DUELO DE MESMA CLASSE É EMPATE TÉCNICO, não vitória. CASCADE
   (q4k/g32+clip, ativação int8/256, cabeça FP32) = 21,90 contra q4_0 = 21,97.
   Margem **0,07 PPL**. A variável não pareada é a precisão da cabeça (FP32
   nossa, Q8_0 deles) e o custo conhecido de baixar a cabeça para 4,5 bpw é
   **1,09** — **15,6× a margem**. Com cabeça pareada em 8 bits o resultado é
   empate ou perda marginal. Regra: quando o efeito de uma variável não pareada
   excede a margem, o resultado é EMPATE, e a diferença não vai para manchete.

   O que a medição sustenta de fato: a estrutura assimétrica+clip paga, e o
   formato trata hidden 896 por padding enquanto o llama.cpp teve de abandonar o
   K-quant naquele arquivo. A distância até o q4_k_m (20,02) não é codec, é MIX:
   ~7,9 bpw porque 896 não divide por 256 e a receita M promove v/down/output.

3. A PROJEÇÃO DE 24 GB FICOU CONDICIONAL. O degrau g64 — que sustenta os
   15,08 GB da Muse — custou +1,44 de PPL no 0.5B. Se a Muse exigir piso g32, o
   bundle vai a 15,67 GB e NÃO cabe (folga −0,09). O "worst estrutural" do §32.3
   deixou de ser barra de erro e passou a ter MECANISMO. Modelos de 30B toleram
   quantização melhor que 0.5B, então o g64 pode sobreviver lá — mas isso agora é
   afirmação que só a conversão integral + PPL da Muse pode fazer.
   `verdict_status: "CONDICIONAL"` no registro.

4. ESCOPO DO RUNTIME DECIDIDO POR NÚMERO. llama.cpp completo 40–55 tok/s;
   qualquer caminho PyTorch (BF16 ou CASCADE) 11–13. No 0.5B o **motor** vale
   3–4×, não o kernel: tokenizer, KV-cache, atenção e sampling em C++ dominam
   quando o GEMV não domina — a vantagem do kernel cresce com o tamanho.
   Portanto: "acelerador de PyTorch" serve para bateria e validação; para o
   produto "roda em qualquer PC" é **motor autônomo ou nada**.

5. PPL NÃO É VELOCIDADE — nome e grupo próprios. `batteryFriendlySuffix` e
   `batteryGroupKey` casavam `_E2E` antes de qualquer coisa e rotulavam a bateria
   de perplexidade como "Velocidade ponta a ponta" / "P1 · Codec principal". A
   regra `_PPL` passa a vir ANTES do `_E2E`: nome "Qualidade real
   (perplexidade)", grupo "P1 · Qualidade end-to-end (PPL)". Fixtures travam as
   duas, incluindo que `_E2E_TOKS` continua no grupo de tok/s.

6. BACKLOG COM RETORNO MEDIDO (ordem do relatório, mantida porque é por retorno
   por custo): (1) ativações por grupo-32 estilo Q8_0 recupera **+0,59** a custo
   ~zero em `q4k_prepare_x_i8`; (2) cabeça em classe 8 bits recupera **+1,09** por
   +0,67 GB no 30B — o q4_0 já faz isso; (3) política de degrau por escala (g32
   como piso em modelo pequeno, g64 condicionado a PPL) recupera **+1,44**;
   (4) imatrix, o único item que muda de patamar. Com 1–3: CASCADE data-free
   ≈ 21,3–21,5 (PROJETADO — 21,90 − 0,59 = 21,31), à frente do q4_0 com folga.

   Nota de execução: (1) e (2) são mudanças em `kernels.c` e não foram
   implementadas aqui — esta máquina não tem compilador C nem torch, e editar
   kernel às cegas é pior que não editar.
