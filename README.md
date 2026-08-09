# RIFT-LM, CASCADE, AETHER & SPECTRA Test Observatory

Dashboard estático para executar, publicar e comparar baterias de referência das tecnologias RIFT-LM v0.3.5, CASCADE v0.3, AETHER-LM v1.0 e SPECTRA-LM v0.1.

## O que o dashboard faz

- carrega automaticamente `data/rift_test_batteries.json`;
- exibe datas em `America/Sao_Paulo` (GMT−3), preservando UTC no dado original;
- aceita históricos JSON e CSV locais;
- gera uma célula pronta para testar qualquer model ID ou URL do Hugging Face no Google Colab;
- seleciona automaticamente uma camada `torch.nn.Linear` compatível entre famílias;
- compara as baterias principais de RIFT, CASCADE, AETHER e SPECTRA para o mesmo modelo;
- calcula um ranking auditável de modelos e tecnologias a partir das métricas publicadas;
- usa o Gemini 2.5 Flash para explicar qual tecnologia é mais adequada para cada modelo;
- mantém métricas de operação Linear separadas de Tok/s do modelo;
- entrega launchers Python por URLs curtas, sem duplicar os scripts dentro do notebook.
- pesquisa modelos públicos no Hugging Face e monta uma fila Colab com vários modelos e tecnologias.

## Executar localmente

```bash
npm test
npm run dev
```

Abra `http://localhost:3000`.

Se o PowerShell bloquear `npm.ps1`, use os executáveis `.cmd` sem alterar a política global:

```powershell
npm.cmd test
npm.cmd run dev
```

## Deploy automático na Vercel

Conecte este repositório a um projeto Vercel. Cada atualização da branch `main`, inclusive commits de dados feitos pela Function, dispara um novo deploy.

Configure as variáveis abaixo em **Vercel → Project Settings → Environment Variables**:

```dotenv
RIFT_GITHUB_TOKEN=github_pat_SEU_FINE_GRAINED_PAT
RIFT_GITHUB_REPOSITORY=programador-powershell/RIFT-LM
RIFT_GITHUB_BRANCH=main
RIFT_GITHUB_DATA_PATH=data/rift_test_batteries.json
RIFT_INGEST_TOKEN=UMA_CHAVE_ALEATORIA_COM_PELO_MENOS_32_CARACTERES
API_GOOGLE=SUA_CHAVE_DA_GOOGLE_AI_STUDIO
```

O PAT e `API_GOOGLE` devem ficar somente na Vercel. O PAT precisa ter `Contents: Read and write` apenas neste repositório. Nunca coloque esses segredos no Colab, no HTML ou em um arquivo versionado.

A Function `/api/analyze` envia ao Gemini somente identificadores técnicos, o estado da bateria e métricas numéricas já publicadas. Ela valida e sanitiza a entrada, valida a resposta estruturada, limita tamanho e frequência das requisições e mantém um cache curto em memória. Em produção, o dashboard solicita a análise automaticamente; localmente, ele preserva apenas o ranking determinístico para não consumir a API.

A Function pública `/api/models?q=...` faz o autocomplete de modelos públicos `text-generation` do Hugging Face, limita e sanitiza a busca e devolve apenas model ID, pipeline, biblioteca, downloads, likes e estado gated. A resposta usa cache da Vercel por cinco minutos.

## Secrets do Google Colab

Cadastre e autorize:

- `RIFT_RESULTS_ENDPOINT`: `https://SEU-DOMINIO.vercel.app/api/results`;
- `RIFT_INGEST_TOKEN`: o mesmo segredo de ingestão configurado na Vercel;
- `HF_TOKEN`: opcional para limites maiores ou modelos gated.

Os launchers por URL configuram `RIFT_RESULTS_ENDPOINT` automaticamente. Mesmo assim, `RIFT_INGEST_TOKEN` continua obrigatório no Colab e nunca é incluído na resposta pública da API.

## Executar por URL no Colab

As rotas amigáveis retornam um pequeno programa Python que baixa a bateria do mesmo commit implantado e a executa na GPU do Colab:

```python
!curl -fsSL "https://rift-lm.vercel.app/rift/Qwen/Qwen2.5-0.5B" -o /content/rift_launcher.py && python /content/rift_launcher.py
!curl -fsSL "https://rift-lm.vercel.app/cascade/Qwen/Qwen2.5-0.5B" -o /content/cascade_launcher.py && python /content/cascade_launcher.py
!curl -fsSL "https://rift-lm.vercel.app/aether/Qwen/Qwen2.5-0.5B" -o /content/aether_launcher.py && python /content/aether_launcher.py
!curl -fsSL "https://rift-lm.vercel.app/spectra/Qwen/Qwen2.5-0.5B" -o /content/spectra_launcher.py && python /content/spectra_launcher.py
```

### Limite explícito para Kimi-K3

`Kimi-K3` resolve para `moonshotai/Kimi-K3`, mas não é colocado na fila do Colab. O modelo possui 2,8 trilhões de parâmetros; mesmo no limite teórico de 4 bits, somente os pesos exigiriam pelo menos 1,4 TB. Além disso, o código remoto da revisão atual declara `transformers==4.56.2`, enquanto versões mais novas podem falhar ao importar `OutputRecorder` e `check_model_inputs`.

As rotas antigas com Kimi continuam respondendo, porém o launcher encerra antes de baixar os pesos, exibe a versão compatível do Transformers e informa os recursos detectados. Nenhum resultado sintético ou de um modelo substituto é publicado em nome do Kimi-K3. Para testar o checkpoint real, é necessário adaptar as baterias para carregamento distribuído e executá-las em infraestrutura compatível com os engines recomendados pelo projeto, como vLLM, SGLang ou TokenSpeed.

## Fila serial de benchmarks

No card **Fila serial de baterias**, digite pelo menos dois caracteres, selecione um resultado público do Hugging Face e repita para todos os modelos desejados. A seleção de tecnologia, camada e `trust_remote_code` é capturada individualmente quando o modelo entra na fila.

O botão **Copiar lista serial** gera uma única célula Python para o Colab. Ela:

1. expande cada modelo para as tecnologias escolhidas;
2. baixa um launcher por vez;
3. executa o benchmark em um subprocesso isolado e aguarda sua publicação;
4. encerra a fila imediatamente se um benchmark falhar;
5. remove o launcher temporário, força coleta de lixo e aguarda três leituras estáveis de VRAM via `nvidia-smi` antes de iniciar o próximo.

O encerramento do subprocesso é o limite de isolamento de RAM/VRAM. A tolerância de VRAM é de 128 MB sobre o nível medido antes do benchmark e o timeout de liberação é de 180 segundos.

## Testar um modelo

O campo do dashboard aceita tanto `org/modelo` quanto uma URL, por exemplo:

```text
microsoft/Phi-3.5-mini-instruct
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

### RIFT

```bash
python rift_m0_phase1_test_v035_auto_batteries.py \
  --mode phase1 \
  --model microsoft/Phi-3.5-mini-instruct \
  --target-layer auto \
  --device cuda \
  --publish required
```

### CASCADE

```bash
python cascade_m0_phase1_test_v030_auto_batteries.py \
  --model microsoft/Phi-3.5-mini-instruct \
  --target-layer auto \
  --device cuda \
  --publish required
```

### AETHER

```bash
python aether_m0_phase1_test_v100_auto_batteries.py \
  --mode phase1 \
  --model microsoft/Phi-3.5-mini-instruct \
  --target-layer auto \
  --device cuda \
  --publish required
```

### SPECTRA

```bash
python SPECTRA_Colab_Test_M0.py \
  --mode phase1 \
  --model microsoft/Phi-3.5-mini-instruct \
  --target-layer auto \
  --device cuda \
  --publish required
```

Use `--trust-remote-code` somente quando o modelo realmente exigir e você confiar no código do repositório.

## Contrato das métricas

Registros novos declaram:

- `technology`: `RIFT`, `CASCADE`, `AETHER` ou `SPECTRA`;
- `candidate_tok_s`, `candidate_ram_bytes` e `candidate_disk_bytes`;
- `comparison_role: primary` na bateria indicada ao comparador;
- `quality`, `metrics`, `measurement_scope` e `notes`.

O dashboard continua compatível com registros RIFT anteriores que usam `rift_*`.

## Ranking e análise de IA

O score composto usa qualidade de saída (40%), quality gate (5%), redução em disco (20%), redução de RAM (15%) e speedup da operação Linear (20%). Métricas ausentes reduzem a cobertura e aplicam uma penalização explícita ao score. O ranking mede somente esta bateria de referência; não representa inteligência geral do modelo.

O Gemini recebe as mesmas métricas e o ranking calculado pelo servidor. Ele pode recomendar `RIFT`, `CASCADE`, `AETHER`, `SPECTRA` ou `INCONCLUSIVO`, com resumo, confiança, métricas decisivas e ressalvas. Com menos de duas tecnologias medidas, ou quando a resposta recomenda uma tecnologia ausente, o servidor força `INCONCLUSIVO`.

## Limites atuais

O teste RIFT usa o codec experimental `Q4_LINEAR_TEST`; ele não é MXFP4. O teste CASCADE usa decomposição low-rank, Gate heurístico e simulação lag-one do prefetch. O AETHER usa base ternária realmente empacotada em 2 bits e TADDS low-rank por entropia, mas não implementa HQR-ANS 0,85 bit, P-IO assíncrono nem o kernel SRFA. O SPECTRA mede o mesmo tipo de base física, Gate/TADDS e um proxy de drift de uma única operação Linear; prefetch assíncrono, kernel fused, compensação de drift e speculative path permanecem simulados. Nenhum dos scripts contém o kernel low-bit/fused nativo de produção.

Por isso:

- latência de uma única operação Linear não deve ser chamada de Tok/s;
- o prefetch CASCADE não representa I/O assíncrono real;
- P-IO e SRFA do AETHER permanecem simulações do caminho Python;
- o proxy de drift SPECTRA não equivale a certificação de qualidade end-to-end;
- speedups nativos de inferência não devem ser reivindicados a partir dessas baterias de referência;
- comparações de latência exigem o mesmo hardware, camada e forma de ativação.

## Arquivos principais

```text
index.html
api/results.mjs
api/analyze.mjs
api/models.mjs
api/test.mjs
rift_m0_phase1_test_v035_auto_batteries.py
cascade_m0_phase1_test_v030_auto_batteries.py
aether_m0_phase1_test_v100_auto_batteries.py
SPECTRA_Colab_Test_M0.py
SPECTRA_Especificacao_Tecnica_v0.1.txt
data/rift_test_batteries.json
data/record-schema-example.json
```
