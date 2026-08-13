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

### :up: V.3.4.1

### :pushpin: Fixes

- **Publiquei o melhor de três runs como se fosse o número** (contrato §32.4): o registro do kernel v2 trazia speedup de **43,6×** contra o caminho `low_mem` e **1,74×** contra o `default` — os valores do único run que eu tinha na época. Com três execuções do MESMO benchmark sintético, sem mudança de código entre elas, a dispersão é de **8,0%** no tok/s do v2 e **19,9% no speedup de manchete** (36,4× · 41,5× · 43,6×). Regra nova: benchmark com mais de um run publica `<metrica>_range` com `{min,max,median}` mais `runs`/`runs_detail`, e o campo de ponto único carrega a **mediana**, nunca o melhor — o smoke falha se voltar a ser o máximo. Consequência sobre o §30.5: o ganho de cosseno do F1 (+3e-5) é uma ordem de grandeza menor que os 8% de dispersão do próprio benchmark, então naquele stack o F1 é indistinguível de ruído.
- **Faixa da projeção de residência** (§32.3): o "cabe na máquina de 24 GB" passa a vir com os três cenários — central 15,08 GB (folga +0,46 GiB), worst realista 15,19 (+0,35) e worst estrutural 15,67 (−0,09). O ponto de ruptura de +3,24% exigiria bpw médio de ~4,47, ou seja **~96% dos tensores em g32 quando o medido é 5,6%** — o veredito sobrevive ao worst realista e cai só no estrutural, por 0,09 GiB.
- **Docstring do conversor tinha dois resíduos vencidos** (§32.2): listava as classes de máquina como `8/16/24/40`, que são os ORÇAMENTOS e não os totais (resto exato do bug de subtração dupla do §31.5), e afirmava "144 B/super-bloco" para todo caso quando g=64 usa 138 B.

### :construction_worker: Refactors

- **Merge do conversor v2 pós-review** (§32.1), pela mesma regra do GEYSER: atualização de ciência é merge, nunca substituição da camada de publicação. A forma nova do relatório de residência é melhor e foi adotada — chave `maquina_<total>gb` com `orcamento_gib`/`cabe`/`folga_gib` como campos, em vez do orçamento embutido no nome da chave — com anti-regressão versionada em `tests/test_residency.py` (inclui o limiar exato 14,50 CABE / 14,51 NÃO CABE). Repostos por cima: `kv_runtime_reserve_gib` numérico ao lado da frase `regra_orcamento`, para consumidor de JSON não precisar parsear texto, e `gate_policy` no resumo, porque um bundle que rompe o invariante §29.5 precisa ser autodescritivo.

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
