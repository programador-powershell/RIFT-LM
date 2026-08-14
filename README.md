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

### :up: V.3.8

### :warning: Latest Changes

- **PPL da Muse-Glimmer-30B medida: 10,0118** (contrato §39) — a medição pendente desde o §33. WikiText-2, 12 janelas × 512 (6.132 tokens = 12 × 511, NLL em float64 com labels deslocados), arquitetura `muse_glimmer` **nativa** do transformers 5.15, não reimplementação. **O bundle de 16,30 GB deixa de ser qualidade desconhecida**: lixo de wiring apareceria em ~1,3 M (foi o que os autores viram antes de caçar os bugs) e dano severo de quantização em 20+ — 10,01 exclui os dois, um por cinco ordens de grandeza, o outro por 2×. Nove rodadas de incerteza encerradas.
- **Dois bugs de integração caçados no percurso**, e eles importam mais que o número (§39.4): **layout q8r intercalado** (saída incorreta antes da correção) e **o embedding da Muse é `TextNormedEmbedding`** — RMSNorm sem escala por linha, que a substituição usada nos testes anteriores dropava; só o código nativo do release revelou. Até este lote **nenhuma medição provava que o executor produzia saída correta**: os cossenos mediam reconstrução de peso, as bandas mediam bytes/tempo, e os dois bugs viviam exatamente nesse ponto cego.

### :pushpin: Fixes

- **Correção ao §36.1**: o embedding continua fora do caminho de **banda** (uma linha por token, pesa em RAM e não em banda), mas **não é lookup puro** — o executor tem de aplicar RMSNorm na linha lida. A conclusão sobre banda se mantém; a caracterização como "lookup" estava incompleta.
- **Minha asserção do §35.6 travava "PPL da Muse não medida"** e quebrou quando a realidade mudou — o tipo bom de falha. Passou a exigir `end_to_end_measured: true` com `certified: false`, mais a declaração do que a PPL não estabelece.
- **Minha asserção de tokens estava errada**: usei `janelas × ctx` e o correto é `janelas × (ctx − 1)`, porque o primeiro token de cada janela não tem alvo de predição. O cross-check fecha nos dois testes (12 × 511 = 6.132 na Muse; 24 × 511 = 12.264 no Qwen), confirmando a mesma convenção de NLL com labels deslocados.

### :construction_worker: Refactors

- **O que 10,01 ainda não estabelece, registrado junto com o número** (§39.2): **quanto a quantização custou**. Não existe PPL do BF16 da própria Muse, então 10,01 é absoluto sem comparação. Inferindo do proxy 0.5B, o BF16 estaria entre **~7,8 e ~9,5** — e 10,01 é compatível com toda a faixa. Logo `end_to_end_measured: true` mas `end_to_end_certified: false`: o contrato exige baseline **e** candidato no mesmo protocolo. Duas ressalvas de protocolo obrigatórias: metade do corpus do teste do 0.5B (12 janelas contra 24) e **weight-only** com ativações fp32, enquanto o executor usa int8/g32 (+0,1 a +0,6 pelo proxy).
- **"Vence o melhor GGUF em qualidade-por-byte" segue sem artefato** (§39.6): com os dados que este projeto tem, a melhor config CASCADE no 0.5B é 21,3122 contra 20,0172 do q4_k_m — o CASCADE **perde 6,5%**. A inversão depende do 19,89, que nunca veio como artefato. O que está medido e sustenta a tese é cobertura de arquitetura, não qualidade-por-byte.

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
