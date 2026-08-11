<div align="center">

# Observatório LLM

**RIFT · CASCADE · AETHER · SPECTRA · GEYSER · WINNER**

![App Screenshot](https://rift-lm.vercel.app/screenshot-dashboard.png)

📊

</div>

## :heavy_check_mark: Features

Observatório de benchmarks para executar, publicar e comparar baterias de referência das tecnologias **RIFT-LM v0.3.5**, **CASCADE v0.3**, **AETHER-LM v1.0**, **SPECTRA-LM v0.1**, **GEYSER-LM v0.2.0** e **WINNER.cpp v0.8**, com a série **C3** de 16 passos e a bateria de **Capacidades** (`CAPABILITY_PROBE_V1`) por modelo.

- **Fluxo Colab → Vercel → dashboard**: as baterias rodam na GPU do Google Colab, publicam via `POST /api/results` (Bearer ≥ 32 caracteres, timing-safe, whitelist de campos) e o histórico versionado em `data/rift_test_batteries.json` alimenta o dashboard em tempo quase real.
- **Painel ÚNICO** na rota `/` (index.html) com gráficos desenhados em SVG/CSS inline (sem bibliotecas externas, sem CDN); os gráficos do antigo painel resumido (tok/s por modelo, speedup por bateria com marcador 1,0×, redução de disco/RAM por tecnologia e latência mediana por bateria primária) foram incorporados à seção "Gráficos" do principal (contrato §24.1).
- **Níveis de bateria N1–N5**: o painel agrupa e rotula por nível (`Nível 1 · Fundação (M0)` → `Nível 5 · Fase final`) via `batteryLevel(battery_id)`. Os níveis são SOMENTE camada de exibição — os `battery_id` históricos são imutáveis; os títulos visíveis usam o nome amigável `N<nível> · <nome PT-BR>` (`batteryFriendlyName`, contrato §19.5) e o id bruto aparece somente em tooltip.
- **Seção Capacidades** estilo OpenRouter compare: gráficos de barras por modelo em 7 categorias com ordem fixa (`CAP_INTELLIGENCE` · `CAP_CODING` · `CAP_AGENTIC` · `CAP_DEEPSEARCH_QA` · `CAP_MCP_ATLAS` · `CAP_TAU3_BENCH` · `CAP_SWE_BENCH`, score 0–100) publicados por `capability_eval_auto_batteries.py` com `technology="CAP"` — excluídos dos rankings de tecnologia e da política do WINNER. Rotulagem obrigatória: "probe leve embutido — não é MMLU/HumanEval/SWE-bench completos"; as 4 categorias novas são probes leves inspirados em DeepSearch QA, MCP-Atlas, τ³-Bench e SWE-Bench — NÃO são os benchmarks oficiais completos.
- **GEYSER-LM (6ª tecnologia)**: suíte M0/G0 (`GEYSER_M0_G0_V1`) em `geyser_launcher.py`, servida pela rota `GET /geyser/:model*` (o arquivo Python é servido com o placeholder `__GEYSER_MODEL_ID__` substituído). `G1_GEYSER_ZDC_LUT` é primário e o GEYSER é elegível na seleção do WINNER.
- **Seletor de modelos estilo OpenRouter** no painel `/` (lista + detalhes com parâmetros/downloads) e **histórico recolhível** (últimos 100 registros dentro de `<details>`).
- **Gerador "Modelo e fila"** preservado: gera a célula Colab CURTA (bootstrap de ~14 linhas com variáveis `MODELS`/`TECHS`, contrato §14.3) que executa a fila completa via `GET /runner.py` — o orquestrador servidor cobre todas as tecnologias (incl. GEYSER, o probe CAP e a bateria GGUF) com limpeza prévia e espera de liberação de VRAM entre passos.
- **`GET /api/results`** público como fonte primária do painel (cache `s-maxage=15, stale-while-revalidate=60` + ETag; fallback client-side para `/data/rift_test_batteries.json`).
- **WINNER dinâmico**: a arquitetura executada pelo WINNER é escolhida pela política determinística `selectWinnerArchitecture(records)` (JS) / `select_winner_architecture(records)` (Python) — vence a tecnologia com mais modelos otimizados; empate resolvido por score médio ponderado; fallback fixo `CASCADE, RIFT, AETHER, SPECTRA, GEYSER`. Override por `RIFT_WINNER_ARCH` ou `?arch=`.
- **Série C3 (16 passos)** via `c3_methodology_auto_batteries.py --technology rift|aether|cascade|spectra`: bundle M0 congelado, ABI de Stage Table/Page, CASCADE-IR v3, leitor C++ mmap, Linear e Bloco reais em 4 caminhos, decisão C1, expansão para 4 blocos e modelo completo com tok/s reais via `model.generate` (baseline E candidato).
- **Conversor pacote-por-pacote** (`core/cascade/converter/`) com orçamento de disco: converte checkpoints grandes sem exigir espaço para o modelo inteiro de uma vez. O **card "Conversor de modelos"** do painel (contrato §26) gera a célula Colab (`GET /converter/:model*?hf_repo=...&publish=on|off`) e oferece o download do runner local auto-contido (`GET /converter.py`, `Content-Disposition: attachment`) para rodar o MESMO fluxo no próprio PC: baixar da HF (snapshot só de pesos+config/tokenizer) → converter para CASCADE-DIR → (opcional) enviar o resultado de volta ao Hugging Face Hub (`create_repo(exist_ok)` + `upload_folder`, exige `HF_TOKEN` de ESCRITA no ambiente — nunca logado).
- Launchers Python por URL curta (`/rift/...`, `/cascade/...`, `/aether/...`, `/spectra/...`, `/winner/...`, `/geyser/...`, `/cap/...`, `/gguf/...`, `/microlm` — sem parâmetro de modelo, `/c3/:tech/:model`, `/final/:tech/:model` — `tech` aceita `all`), pinados no commit implantado.
- Ranking auditável + análise Gemini 2.5 Flash (`/api/analyze`) e autocomplete de modelos Hugging Face (`/api/models?q=...`).
- Datas exibidas em `America/Sao_Paulo` (GMT−3) preservando UTC no dado original; aceita históricos JSON e CSV locais.

### Tabela de pesos do score (`SCORE_WEIGHTS`, score canônico v2 — contrato §25)

| Métrica | Peso |
| ------- | ---- |
| Cosine | 25 |
| NRMSE | 10 |
| Redução em disco | 10 |
| Redução de RAM | 30 |
| Speedup | 20 |
| Quality gate | 5 |

Pesos calibrados para o objetivo declarado do score: identificar a melhor tecnologia para rodar LLM em um PC convencional (4 núcleos, 8 GB de RAM livre, sem GPU) — a RAM é a restrição dura, por isso RAM 30% > disco 10%. Composição: Qualidade 40% • RAM 30% • latência da operação 20% • disco 10%. O score é calculado sobre o registro mais recente por par `modelo|battery_id`. Métricas ausentes reduzem a cobertura e penalizam o score. O ranking mede somente estas baterias de referência; não representa inteligência geral do modelo.

### Limites atuais

- Fora da série C3, a latência de uma única operação Linear continua não sendo tok/s; proxies ficam apenas em `metrics.operation.*` com sufixo `_proxy` e nunca nos campos de comparação.
- `baseline_tok_s`/`candidate_tok_s` de nível superior existem SOMENTE quando medidos por `model.generate` do modelo completo (bateria `C3_<TECH>_FULLMODEL_E2E_TOKS`), baseline e candidato sob o mesmo protocolo.
- RAM de nível superior é SOMENTE RSS medido (amostragem de `VmRSS` por fase); estimativas aritméticas vivem em `metrics.memory.estimated_*`. Sem medição → `null`.
- O prefetch CASCADE, P-IO/SRFA do AETHER e o drift-proxy do SPECTRA seguem como simulações/diagnósticos (`implementation.kind = "SIMULATED"`, `eligible_for_primary_ranking = false`).
- No GEYSER, o tok/s de nível superior existe SOMENTE como promoção do wall-clock Python medido do `G3_GEYSER_BURST` (vanilla→baseline, burst→candidato) e agora é adicionalmente condicionado a `greedy_equivalence=true`; sem equivalência greedy permanece `null`. Projeções BPAT são `PROJETADO` e ficam apenas em `metrics`. O probe de capacidades (CAP) mede o modelo baseline com tarefas leves embutidas e nunca entra em ranking de tecnologia.
- Checkpoints `GGUF`, `DeepSeek-V4`, `NVFP4/MTP` e `Kimi-K3` continuam bloqueados antes do download nos launchers Transformers (formato ou porte incompatível com a bateria `AutoModel`); repositórios GGUF agora têm caminho PRÓPRIO via rota `/gguf` (bateria `GGUF_RUNTIME_V1` sobre llama.cpp). Nenhum resultado sintético é publicado em nome deles.
- Comparações de latência exigem `comparison_group_id` compatível (mesmo hardware, camada e forma de ativação).

## :new: Releases Notes

### :up: V.3

### :warning: Latest Changes

- **Score recalibrado para o objetivo real** (contrato §25, 13º lote): pesos `SCORE_WEIGHTS_V2` voltados ao PC convencional (4 núcleos, 8 GB de RAM livre, sem GPU) — RAM 30% > disco 10% (cosine 25 · nrmse 10 · gate 5 · RAM 30 · speedup 20 · disco 10), espelhados nas 4 implementações (`api/analyze.mjs`, `api/results.mjs`, `index.html`, `winner_m0` Python); normalizações e fator de cobertura inalterados.
- **Painel único** (contrato §24, 12º lote): `dashboard.html` foi deletado e as rotas `/v2` e `/legacy` extintas — existe UMA página em `/`; os gráficos do painel resumido foram incorporados à seção "Gráficos" do principal (tok/s por modelo baseline × candidato, speedup por bateria com marcador 1,0×, redução de disco/RAM por tecnologia e latência mediana por bateria primária — a série temporal de tok/s permanece única); a seção **"Comparação de gerações" foi removida da UI** (os registros `CMP_*` permanecem no histórico e nos grupos de bateria; o publicador `compare_generations_publisher.py` permanece); **regra geral de relevância (§24.3)**: cards e gráficos só renderizam com ≥2 modelos/tecnologias — abaixo disso o card é OMITIDO (sem estado vazio; os dados continuam no Histórico consolidado); a política do WINNER passa a ser espelhada em 4 implementações (`api/analyze.mjs` fórmulas, `api/results.mjs`, `index.html`, `winner_m0` Python).
- **Painel completo vira a página principal** (contrato §23, 11º lote): `/` serve o painel completo (index.html); **cards por classe de bit** com badge (`batteryBitClass(battery_id, metrics)` — 2-bit vs 4-bit vs ternário separados, nunca misturados no mesmo card, máx. 1 linha por tecnologia); **largura fluida** (`max-width:min(1880px,96vw)` + grids auto-fit — sem margens mortas em monitores largos); Comparador desce para antes do Histórico e o card "WINNER executa" é removido do painel principal.
- Painel renomeado para **Observatório LLM** (subtítulo de linhagem `RIFT · CASCADE · AETHER · SPECTRA · GEYSER · WINNER`).
- Redesign com **níveis de bateria N1–N4** (`batteryLevel(battery_id)`): Fundação (M0) → Série C → Metodologia C3 → Capacidades. Os `battery_id` permanecem imutáveis — o nível é somente exibição.
- Seção **Capacidades** estilo OpenRouter compare: `CAP_INTELLIGENCE` · `CAP_CODING` · `CAP_AGENTIC` (probe leve embutido — não é MMLU/HumanEval/SWE-bench completos), publicadas por `capability_eval_auto_batteries.py` (`CAPABILITY_PROBE_V1`, `technology="CAP"`).
- **Probes de benchmark agêntico** (DeepSearch QA, MCP-Atlas, τ³-Bench, SWE-Bench) na bateria de capacidades — probes leves inspirados, não os benchmarks oficiais (§15): novos `battery_id` `CAP_DEEPSEARCH_QA` · `CAP_MCP_ATLAS` · `CAP_TAU3_BENCH` · `CAP_SWE_BENCH`, ordem fixa de exibição de 7 categorias e grade renderizada dinamicamente a partir dos registros no painel.
- Nova tecnologia **GEYSER-LM**: rota `GET /geyser/:model*` servindo `geyser_launcher.py` (suíte M0/G0, `GEYSER_M0_G0_V1`), cor teal nos gráficos e **elegível na seleção do WINNER** (`G1_GEYSER_ZDC_LUT` primário).
- **Seletor de modelos estilo OpenRouter** no painel principal `/` (lista, detalhes, filtro de fit e re-teste de TODAS as tecnologias direto do ranking).
- **Histórico recolhível** (`<details>`, últimos 100 registros) no painel.
- Gerador **"Modelo e fila"** preservado e estendido com os passos GEYSER (`curl -fsSL .../geyser/<modelo> -o /content/geyser_launcher.py && python /content/geyser_launcher.py`) e probe CAP (`/cap/:model*`).
- **Muse Glimmer 2-bit via llama.cpp** (`GGUF_RUNTIME_V1`, contrato §11): nova rota `GET /gguf/:model*` servindo a célula da bateria `gguf_e2e_auto_batteries.py` com quant padrão `UD-Q2_K_XL` (`?quant=` opcional), sugestão `unsloth/Muse-Glimmer-30B-GGUF` e cor âmbar no painel. Dependências Colab: binário llama.cpp de release **pinada** (tag + sha256) e `pip gguf>=0.10,<1` — instalação apenas no Colab e **sujeita a homologação TI/SI**.
- **tok/s MEDIDO em todas as tecnologias**: baterias `P1_RIFT_E2E_TOKS`, `P1_AETHER_E2E_TOKS`, `P1_SPECTRA_E2E_TOKS`, `P1_WINNER_E2E_TOKS` e `P1_GGUF_E2E_TOKS` (`model.generate` real, `metrics.e2e.measured=true`), candidato real no C2 (`P1_CASCADE_C2_E2E_TOKS` patcheia todas as `nn.Linear` dos blocos) e promoção do tok/s Python medido do `G3_GEYSER_BURST` do GEYSER para `baseline_tok_s`/`candidate_tok_s` de topo.
- **Cards por modelo agrupados por bateria** (`batteryGroupKey`, 8 regras ordenadas): baseline "Antes" ÚNICO no topo do grupo, uma linha por tecnologia e alternância de visão métricas ⇄ qualidade (`data-quality-toggle`).
- **Execução sempre de TODAS as tecnologias** (fim dos botões por tecnologia, §13.3): `Rodar todas as séries` e os botões por série (A–E) no painel `/`.
- **Painel reorganizado** (§21): rankings → baterias por série (A–E, um botão por rodada — Série A `rift,cascade,aether,spectra,winner,geyser`, B `c-series`, C `c3`, D `cap`, E `final`; `Rodar todas as séries` substitui o "Teste reforçado") → lista de modelos unificada (cards §19.6 com nomes amigáveis — nenhum `battery_id` cru como título); os cards independentes CASCADE·Série C e GEYSER foram absorvidos nos ⓘ das séries e a "Comparação de modelos" tem exatamente UM botão "+ Adicionar modelo".
- **Análise de IA como destaque inline** `★ IA recomenda` na linha da tecnologia recomendada (§13.4) — sem painel/card dedicado.
- **Células Colab curtas via `/runner.py`** (bootstrap de ~14 linhas com variáveis `MODELS`/`TECHS`): o painel gera APENAS o bootstrap (Secrets → `BASE` → `MODELS`/`TECHS` → `exec` de `GET /runner.py`); o orquestrador completo da fila (deps de tokenização pinadas, limpeza prévia, série C, C3, CAP, GEYSER, GGUF, espera de liberação de VRAM) é gerado no servidor com origin/repo/ref já resolvidos (§14.3).
- **Fases finais C4–C6 por tecnologia até o marco compilador+executor** (`FINAL_PHASE_V1`, contrato §16) via `final_phase_auto_batteries.py --technology rift|aether|cascade|spectra`: `C4_<TECH>_SECOND_FAMILY` (mesmo core em 2 famílias de modelo), `C5_<TECH>_REPR_BLOCKS` (blocos representativos de modelo maior, drift acumulado ≤ 0,12) e `C6_<TECH>_COMPILE_EXECUTE` (C6: bundles reais em disco, execução via mmap sem os pesos originais — única bateria primária, `metrics.e2e.measured=true`). Nova rota `/final/:tech/:model` (`tech` aceita `all`), passo `final` na expansão `TECHS=["all"]` do `/runner.py`, preclean `/content/final_test_output` e exibição como **Nível 5 · Fase final** (grupos `C4 · Segunda família` · `C5 · Blocos representativos` · `C6 · Compilar+Executar`) no painel.
- **Comparação de modelos estilo OpenRouter** (openrouter.ai/compare, contrato §17) no painel: popup "+ Adicionar modelo" com busca e preview (melhor tecnologia/score, tok/s medido, capacidades, baterias por nível), seções empilhadas só dos modelos selecionados (um bloco por modelo — §19.6), chips removíveis, atalho "adicionar todos (≤6)" e seleção persistida em `localStorage` (chave `observatorio_selected_models`). Gráficos globais permanecem globais.
- **Painel e launchers agora independentes do nome do repositório GitHub** (renomeie o repo sem quebrar nada — resolução automática via env da Vercel): cadeia única `GITHUB_REPO → RIFT_GITHUB_REPOSITORY → VERCEL_GIT_REPO_OWNER/VERCEL_GIT_REPO_SLUG → fallback legado` em `api/_lib/repo.mjs` (ref: `RIFT_GITHUB_BRANCH → VERCEL_GIT_COMMIT_SHA → main`), espelhada nos scripts Python pelas envs exportadas `RIFT_GITHUB_REPOSITORY`/`RIFT_SOURCE_REF` (§14.1/§14.2).
- **GEYSER v0.2.0** (contrato §18.1): base científica atualizada — draft proxy INT4g32 com disclosure condicional H1, probe do draft INT2 quente e KV KIVI-classe real (sink/janela/grupos, bits medidos vs assintóticos) — **mantendo a camada de publicação schema v2** (`{records:[...]}`, HTTPS + Bearer ≥ 32, G1/G3 primários; promoção do tok/s do G3 agora exige `greedy_equivalence=true`). Regra permanente: atualização de ciência do launcher é merge, nunca substituição da camada de publicação.
- **Comparação de gerações** (contrato §18.2, atualizado pelo §24.2): registros `CMP_<TECH>_GENERATIONS` (`COMPARE_GENERATIONS_V1`, nível 2, grupo `E2E · comparação de gerações`, `comparison_role=null`) publicados por `compare_generations_publisher.py` — entram no histórico e nos grupos de bateria do painel (a seção própria da UI foi removida pelo §24.2).
- **Painéis focados em dados** (contrato §19, 7º lote): explicações movidas para badges ⓘ com popover (componente `infoBadge` único por página — clique abre, ESC/clique fora fecha), nomes de bateria amigáveis por nível **N1–N5** (`batteryFriendlyName`, ex.: `P1_CASCADE_C2_E2E_TOKS` → "N2 · Velocidade ponta a ponta"; o `battery_id` cru aparece somente em tooltip), cards por recurso (Throughput | RAM necessária | Espaço em disco | Ganhos medidos) com "Antes" único e **uma linha por tecnologia** (barra colorida da tech), modelos selecionados como seções empilhadas, título "Comparação de modelos" sem meta-linguagem e botões "+ Adicionar modelo" / "adicionar todos (≤6)" lado a lado; card **Comparador auditável removido** (o destaque ★ IA permanece inline).
- **card Conversor** (contrato §26, 14º lote): célula Colab (HF → CASCADE-DIR → upload de volta ao HF) + script local para rodar no próprio PC (`/converter.py`). Novas rotas `GET /converter.py` (runner local auto-contido servido por `api/converter.mjs` com repo/ref pinados — botão "Baixar script (rodar no PC)") e `GET /converter/:model*` (`battery=converter` em `api/test.mjs`: célula com Secrets, deps pinadas e execução `--model <modelo> [--hf-repo <org/nome>] [--publish on]`); params `hf_repo` (validado org/nome, 400 se inválido) e `publish=on|off` (default off — não confundir com o `auto|required|off` das baterias). Defaults pacote-por-pacote: `--disk-budget-gb 75 --resume`; `--delete-source-shards` só com download do próprio runner. A saída `/content/<nome>-cascade` é o PRODUTO (fora da limpeza prévia); card operacional sempre visível (isento da §24.3); nenhum token em código/URL/log (§26.4).
- **MicroLM (modelo de referência 22M com init no-op exato) como 7ª tecnologia** (`technology="MICROLM"`, contrato §22 — tipo MODELO, nunca elegível na política do WINNER): bateria própria `MICROLM_M0_V1` na rota `GET /microlm` (sem parâmetro de modelo — o MicroLM É o modelo, `microlm/MicroLM-22M-v0.2`), arquivos em `engines/microlm/` (`model.py`, `test_model.py`, `CHANGES.md` e `diagram.svg` verbatim + `microlm_m0_auto_batteries.py`); 5 baterias medidas em CPU (`B0_MICROLM_NOOP_INIT`, `P1_MICROLM_DECODE_PARITY`, `P1_MICROLM_DECODE_TOKS`, `P1_MICROLM_TRAINS_FROM_INIT`, `P1_MICROLM_UNIT_CHECKS`), passo `microlm` na Série A e na expansão `TECHS=["all"]` do `/runner.py`, cor roxo-rosa e nomes amigáveis N1 no painel (diagrama como link no ⓘ, não imagem inline).

### :pushpin: Fixes

- Aliases `geyser_*` aceitos na whitelist do ingest `POST /api/results` (campos numéricos `geyser_tok_s`/`geyser_ram_bytes`/`geyser_disk_bytes` validados).
- Enum de `technology` do validador (`scripts/validate_data.mjs`) e do ingest atualizado com `GEYSER` e `CAP` (timestamps ISO-8601 com offset `+00:00` já eram aceitos).
- Enum `GGUF` no ingest (`api/results.mjs`) e no validador (`scripts/validate_data.mjs`), com aliases numéricos `gguf_tok_s`/`gguf_ram_bytes`/`gguf_disk_bytes` na whitelist.
- `model_id` de registro aceita o sufixo opcional `:quant` (ex.: `unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL`, CAP-sobre-GGUF) no ingest e no validador.
- Corrigido 404 no Colab após renomear o repositório: dashboard e launchers não hardcodam mais `programador-powershell/RIFT-LM` — o único resquício é o fallback legado documentado (`api/_lib/repo.mjs` e a constante `LEGACY_REPOSITORY` dos scripts Python).
- O publish do GEYSER v0.2.0 enviado pelo usuário postava o payload legado `{batteries, gain_report}`, rejeitado pelo ingest `POST /api/results` — a camada de conversão para `{records:[...]}` (schema v2) foi restaurada no merge (§18.1).

### :construction_worker: Refactors

- **Árvore de pastas canônica** (contrato §20): reorganização em `engines/` (uma pasta por tecnologia, incl. `engines/winner/cpp/`), `core/cascade/` (pacote python compartilhado), `batteries/` (baterias multi-motor) e `docs/specs/` (código e launchers atualizados para pares `repo_path → local_path` — o layout local do Colab não muda; duplicata do conversor `cascade-model-converter/` eliminada, cópia única em `core/cascade/converter/`).
- Política do WINNER com a 6ª tecnologia espelhada nas 4 implementações (`api/analyze.mjs` fórmulas, `api/results.mjs`, `index.html`, `winner_m0_phase1_test_v080_auto_batteries.py`): elegíveis `[RIFT, AETHER, CASCADE, SPECTRA, GEYSER]`, desempate `[CASCADE, RIFT, AETHER, SPECTRA, GEYSER]`; `CAP` nunca é elegível.
- Níveis de bateria implementados como camada de exibição (`batteryLevel` + rótulos amigáveis), sem tocar em nenhum `battery_id` histórico.
- `batteryGroupKey(battery_id)` com implementação única no painel (`index.html`), fixada por fixtures (`scripts/battery_group_key_fixtures.mjs` + smoke).
- `scripts/real_benchmark_runner.py` preserva o tok/s medido das baterias `*_E2E_TOKS` com `metrics.e2e.measured=true` (Adendo `E2E_TOKS_V1`) — passthrough sem anulação, aliases derivados normalizados.

## :wrench: Instalação

Clona o repositório.

```bash
git clone https://github.com/programador-powershell/RIFT-LM.git
cd RIFT-LM
```

Roda os smoke tests (Node ≥ 18, sem dependências externas).

```bash
npm test
```

Inicia o servidor local.

```bash
npm run dev
```

Se o PowerShell bloquear `npm.ps1`, use `npm.cmd test` e `npm.cmd run dev` sem alterar a política global.

Para o deploy, conecte o repositório a um projeto Vercel e configure em **Project Settings → Environment Variables**: `RIFT_GITHUB_TOKEN`, `RIFT_GITHUB_REPOSITORY`, `RIFT_GITHUB_BRANCH`, `RIFT_GITHUB_DATA_PATH`, `RIFT_INGEST_TOKEN` (≥ 32 caracteres) e `API_GOOGLE`. Segredos ficam somente em env vars/Colab Secrets/Vercel — nunca em arquivos versionados.

## :file_folder: Diretórios

```
├── RIFT-LM
│   ├── api                       # Functions Vercel: results (POST/GET), analyze, models, test, real-test, geyser, runner, converter
│   ├── batteries                 # Baterias multi-motor: c3_methodology, final_phase, capability_eval, gguf_e2e, compare_generations_publisher
│   ├── core
│   │   └── cascade               # Pacote python compartilhado (importável como `cascade`): compiler (IR), kernels, runtime, converter, tests, benchmarks
│   ├── data                      # Histórico publicado (rift_test_batteries.json) + relatórios + exemplo de schema
│   ├── docs                      # C3_CONTRACTS_V1, C3_METHODOLOGY, REAL_BENCHMARK_PROTOCOL_V3
│   │   └── specs                 # Especificações técnicas .txt (RIFT-LM, SPECTRA)
│   ├── engines                   # Um diretório por tecnologia
│   │   ├── rift                  # rift_m0_phase1_test_v035_auto_batteries.py
│   │   ├── aether                # aether_m0_phase1_test_v100_auto_batteries.py
│   │   ├── spectra               # SPECTRA_Colab_Test_M0.py
│   │   ├── cascade               # cascade_m0/c0/c1/c2_*_auto_batteries.py
│   │   ├── geyser                # geyser_launcher.py (servido por GET /geyser/:model*)
│   │   ├── microlm               # MicroLM (7ª tecnologia, tipo MODELO): model.py/test_model.py/CHANGES.md/diagram.svg (verbatim) + microlm_m0_auto_batteries.py (rota GET /microlm)
│   │   └── winner                # winner_m0_*.py + cpp/ (runtime C++ CMake: include/, src/, bench/)
│   ├── scripts                   # dev server, smoke tests (.mjs) e utilitários Python de benchmark/publicação
│   └── index.html                # Painel único (rota /)
```

## :rocket: Executáveis

| Nome | Descrição |
| ---- | --------- |
| engines/rift/rift_m0_phase1_test_v035_auto_batteries.py | Bateria de referência RIFT-LM (codec int2 experimental) sobre camada Linear real |
| engines/cascade/cascade_m0_phase1_test_v030_auto_batteries.py | Bateria de referência CASCADE (INT4 + low-rank + Gate) |
| engines/aether/aether_m0_phase1_test_v100_auto_batteries.py | Bateria de referência AETHER-LM (base ternária 2 bits + TADDS) |
| engines/spectra/SPECTRA_Colab_Test_M0.py | Bateria de referência SPECTRA-LM (ternário + métrica de drift) |
| engines/winner/winner_m0_phase1_test_v080_auto_batteries.py | Bateria WINNER: compila o runtime C++ (`engines/winner/cpp/`), roda self-test e executa a arquitetura vencedora escolhida dinamicamente (incl. GEYSER) |
| engines/geyser/geyser_launcher.py | Suíte GEYSER-LM v0.2.0 M0/G0 (`GEYSER_M0_G0_V1`): baterias B0/G1–G5 (draft proxy INT4g32 + disclosure H1, probe INT2 quente, KV KIVI-classe medido), publica schema v2 com `technology="GEYSER"`; servido pela rota `/geyser/:model*` com `__GEYSER_MODEL_ID__` substituído |
| engines/microlm/microlm_m0_auto_batteries.py | Bateria MicroLM (`MICROLM_M0_V1`, contrato §22): 5 baterias medidas em CPU sobre o modelo de referência `microlm/MicroLM-22M-v0.2` — init no-op exato, paridade de decode + cache limitado, tok/s real de decode (caches novos por run, prefill fora do cronômetro, mediana de 3 runs de 32 tokens), treino do init e checagens de unidade; `technology="MICROLM"` nunca elegível no WINNER; célula Colab via `GET /microlm` (baixa também `engines/microlm/model.py` — import do mesmo diretório) |
| batteries/capability_eval_auto_batteries.py | Bateria de capacidades (`CAPABILITY_PROBE_V1`): 7 categorias por modelo — CAP_INTELLIGENCE, CAP_CODING, CAP_AGENTIC, CAP_DEEPSEARCH_QA, CAP_MCP_ATLAS, CAP_TAU3_BENCH e CAP_SWE_BENCH (probe leve embutido, score 0–100, fora da política do WINNER; as 4 novas são probes leves inspirados nos benchmarks homônimos — não os oficiais completos); backend opcional `--backend llamacpp --server-url` para CAP-sobre-GGUF |
| batteries/gguf_e2e_auto_batteries.py | Bateria GGUF (`GGUF_RUNTIME_V1`, §11): `B0_GGUF_RUNTIME_SETUP` baixa o binário llama.cpp de release pinada (tag + sha256) e os arquivos `.gguf` do quant (padrão `UD-Q2_K_XL`); `P1_GGUF_E2E_TOKS` mede tok/s E2E reais; servida pela rota `/gguf/:model*`. Dependências (llama.cpp pinado + `pip gguf`) instaladas só no Colab e sujeitas a homologação TI/SI |
| engines/cascade/cascade_c0_phase1_auto_batteries.py | Série C0: fase 1 do pipeline CASCADE |
| engines/cascade/cascade_c1_block_auto_batteries.py | Série C1: bloco Transformer real com critérios de aprovação |
| engines/cascade/cascade_c2_e2e_auto_batteries.py | Série C2: caminho end-to-end |
| batteries/c3_methodology_auto_batteries.py | Série C3: metodologia de 16 passos para rift/aether/cascade/spectra (bundle, ABI, IR, Linear/Bloco 4 caminhos, decisão C1, 4 blocos, modelo completo com tok/s reais) |
| batteries/final_phase_auto_batteries.py | Fases finais C4–C6 (`FINAL_PHASE_V1`, contrato §16) para rift/aether/cascade/spectra: C4 segunda família, C5 blocos representativos e C6 compilar+executar (bundles CSCD reais em disco, `generate` via mmap sem os pesos originais — única primária, tok/s e2e medido); servido pela rota `/final/:tech/:model` e pelo passo `final` da fila `TECHS=["all"]` |
| batteries/compare_generations_publisher.py | Publicador da comparação de gerações (`COMPARE_GENERATIONS_V1`, §18.2, stdlib-only): converte `compare_generations_report.json` em registros `CMP_<TECH>_GENERATIONS` (schema v2, gate PASS sse top1≥0.70 ∧ ppl≤1.5×ppl_original, tetos apenas em `metrics` com rótulo PROJETADO) e faz POST com HTTPS + Bearer ≥ 32; `--selftest` embutido |
| core/cascade/converter/cascade_converter.py | Conversor de checkpoints pacote-por-pacote com orçamento de disco (cópia única; no Colab é baixado como `cascade/converter/`): subcomandos `convert` (`--input --output --model-id --group-size --ranks --disk-budget-gb --resume --delete-source-shards --publish`), `inspect` e `self-test` |
| GET /converter.py (api/converter.mjs) | Runner LOCAL auto-contido do conversor (contrato §26.1, gerado no servidor com repo/ref pinados; download via botão "Baixar script (rodar no PC)"): `python cascade-converter-runner.py --model org/modelo` OU `--input pasta-local`, `--hf-repo org/nome` (upload ao HF Hub, `HF_TOKEN` de ESCRITA via env), `--publish on\|off`, defaults `--disk-budget-gb 75` + retomada |
| GET /converter/:model* (api/test.mjs, battery=converter) | Célula Colab completa do conversor (contrato §26.2, `CONVERTER_STATIC_V1`): Secrets → deps pinadas → baixa `/converter.py` → executa com `--model <modelo> [--hf-repo <destino>] [--publish on]`; params `hf_repo` e `publish=on\|off` |
| engines/winner/cpp (binário `winner`) | Runtime C++ compilado via CMake/g++; `winner --self-test` e leitor de bundle com validação de offsets/CRC |
| scripts/dev_server.mjs | Servidor local de desenvolvimento (`npm run dev`, porta 3000) |
| scripts/dashboard_smoke.mjs | Smoke test do dashboard (`npm test`) |
| scripts/real_benchmark_smoke.mjs | Smoke test do protocolo de benchmark real (`npm test`) |
| scripts/real_benchmark_runner.py | Runner do protocolo `LINEAR_REAL_MEASURED_V3` |
| scripts/publish_batteries.py | Publicador de históricos de baterias |

## :computer: Acesso

Para o Observatório LLM em produção acesse https://rift-lm.vercel.app (painel único em `/` — uma página só).

Para o ambiente local acesse http://localhost:3000 após `npm run dev`.

Não há login: os dados publicados são públicos. A escrita (`POST /api/results`) exige o Bearer `RIFT_INGEST_TOKEN` (≥ 32 caracteres), cadastrado na Vercel e como Secret do Colab (com `HF_TOKEN` opcional para modelos gated).

## :book: Documentação

### :link: [Wiki](https://github.com/programador-powershell/RIFT-LM/wiki)

- [docs/C3_CONTRACTS_V1.md](docs/C3_CONTRACTS_V1.md) — contratos compartilhados: política do WINNER dinâmico, battery ids da série C3, schema v2, endpoints e segurança.
- [docs/C3_METHODOLOGY.md](docs/C3_METHODOLOGY.md) — metodologia de 16 passos da série C3, critérios de decisão C1 e como tok/s se torna mensurável end-to-end.
- [docs/REAL_BENCHMARK_PROTOCOL_V3.md](docs/REAL_BENCHMARK_PROTOCOL_V3.md) — protocolo de honestidade de medição (latência, RAM, disco, tok/s) + Adendo C3_METHODOLOGY_V1.

Execução no Colab — célula CURTA (contrato §14.3): o painel gera apenas o bootstrap abaixo; edite as variáveis `MODELS`/`TECHS` à vontade. Todo o resto (deps de tokenização pinadas, limpeza prévia, fila serial, série C, espera de VRAM e publicação) vive no orquestrador `GET /runner.py`, gerado no servidor com repo/ref/origin já resolvidos:

```python
# Observatório LLM — fila de baterias (célula curta)
from google.colab import userdata
import os, urllib.request
for k in ("RIFT_INGEST_TOKEN", "HF_TOKEN"):
    try:
        v = str(userdata.get(k) or "").strip()
        if v: os.environ[k] = v
    except Exception: pass
if len(os.environ.get("RIFT_INGEST_TOKEN", "")) < 32:
    raise SystemExit("Configure o Secret RIFT_INGEST_TOKEN (>=32 chars) no Colab")
BASE = "https://rift-lm.vercel.app"
MODELS = ["Qwen/Qwen2.5-0.5B"]   # variavel — edite a vontade
TECHS  = ["all"]              # ou lista: rift,cascade,aether,spectra,winner,geyser,microlm,c3,final,cap,gguf
exec(urllib.request.urlopen(BASE + "/runner.py").read().decode("utf-8"))
```

As rotas clássicas (`/rift/...`, `/cascade/...`, `/aether/...`, `/spectra/...`, `/winner/...`) continuam disponíveis, além de `/final/:tech/:model` (célula Colab das fases finais C4–C6 `FINAL_PHASE_V1`; `tech` aceita `rift|aether|cascade|spectra|all`), `/cap/...` (célula Colab do probe de capacidades), `/gguf/...` (célula Colab da bateria GGUF `GGUF_RUNTIME_V1`, quant padrão `UD-Q2_K_XL` via `?quant=`; o download do binário llama.cpp pinado e do `pip gguf` acontece dentro da bateria, apenas no Colab, e essas dependências estão sujeitas a homologação TI/SI) `/microlm` (célula Colab da bateria MicroLM `MICROLM_M0_V1` — SEM parâmetro de modelo: a bateria avalia o próprio modelo de referência `microlm/MicroLM-22M-v0.2` em CPU), `/converter/:model*` (célula Colab do conversor CASCADE `CONVERTER_STATIC_V1`, contrato §26 — params `hf_repo` e `publish=on|off`), `/converter.py` (runner LOCAL auto-contido do conversor, para rodar no próprio PC — download com `Content-Disposition: attachment`) e `/geyser/...` — esta última serve o PRÓPRIO `geyser_launcher.py` (uso: `curl -fsSL "https://rift-lm.vercel.app/geyser/<org/modelo>" -o /content/geyser_launcher.py && python /content/geyser_launcher.py`). O ingest é append-only por execução, chaveado por `run_id + technology + battery_id` (reenviar o mesmo `run_id` é um upsert idempotente; uma execução nova nunca apaga a anterior); a deduplicação por `model_id + technology + battery_id` acontece no cliente, nos dashboards, que exibem apenas o snapshot mais recente de cada combinação. Tecnologias e modelos diferentes permanecem independentes. A fila serial do dashboard executa cada benchmark em subprocesso isolado, aguarda a publicação e três leituras estáveis de VRAM (`nvidia-smi`, tolerância 128 MB, timeout 180 s) antes do próximo item.
