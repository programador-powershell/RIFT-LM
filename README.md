# RIFT-LM & CASCADE Test Observatory

Dashboard estático para executar, publicar e comparar baterias de referência das tecnologias RIFT-LM v0.3.5 e CASCADE v0.3.

## O que o dashboard faz

- carrega automaticamente `data/rift_test_batteries.json`;
- exibe datas em `America/Sao_Paulo` (GMT−3), preservando UTC no dado original;
- aceita históricos JSON e CSV locais;
- gera uma célula pronta para testar qualquer model ID ou URL do Hugging Face no Google Colab;
- seleciona automaticamente uma camada `torch.nn.Linear` compatível entre famílias;
- compara as baterias principais de RIFT e CASCADE para o mesmo modelo;
- calcula um ranking auditável de modelos e tecnologias a partir das métricas publicadas;
- usa o Gemini 2.5 Flash para explicar qual tecnologia é mais adequada para cada modelo;
- mantém métricas de operação Linear separadas de Tok/s do modelo.

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

## Secrets do Google Colab

Cadastre e autorize:

- `RIFT_RESULTS_ENDPOINT`: `https://SEU-DOMINIO.vercel.app/api/results`;
- `RIFT_INGEST_TOKEN`: o mesmo segredo de ingestão configurado na Vercel;
- `HF_TOKEN`: opcional para limites maiores ou modelos gated.

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

Use `--trust-remote-code` somente quando o modelo realmente exigir e você confiar no código do repositório.

## Contrato das métricas

Registros novos declaram:

- `technology`: `RIFT` ou `CASCADE`;
- `candidate_tok_s`, `candidate_ram_bytes` e `candidate_disk_bytes`;
- `comparison_role: primary` na bateria indicada ao comparador;
- `quality`, `metrics`, `measurement_scope` e `notes`.

O dashboard continua compatível com registros RIFT anteriores que usam `rift_*`.

## Ranking e análise de IA

O score composto usa qualidade de saída (40%), quality gate (5%), redução em disco (20%), redução de RAM (15%) e speedup da operação Linear (20%). Métricas ausentes reduzem a cobertura e aplicam uma penalização explícita ao score. O ranking mede somente esta bateria de referência; não representa inteligência geral do modelo.

O Gemini recebe as mesmas métricas e o ranking calculado pelo servidor. Ele pode recomendar `RIFT`, `CASCADE` ou `INCONCLUSIVO`, com resumo, confiança, métricas decisivas e ressalvas. Quando uma das tecnologias não possui bateria principal para o modelo, o servidor força o resultado para `INCONCLUSIVO`, independentemente da resposta da IA.

## Limites atuais

O teste RIFT usa o codec experimental `Q4_LINEAR_TEST`; ele não é MXFP4. O teste CASCADE usa decomposição low-rank, Gate heurístico e simulação lag-one do prefetch. Nenhum dos scripts contém o kernel low-bit/fused nativo de produção.

Por isso:

- latência de uma única operação Linear não deve ser chamada de Tok/s;
- o prefetch CASCADE não representa I/O assíncrono real;
- speedups nativos de inferência não devem ser reivindicados a partir dessas baterias de referência;
- comparações de latência exigem o mesmo hardware, camada e forma de ativação.

## Arquivos principais

```text
index.html
api/results.mjs
api/analyze.mjs
rift_m0_phase1_test_v035_auto_batteries.py
cascade_m0_phase1_test_v030_auto_batteries.py
data/rift_test_batteries.json
data/record-schema-example.json
```
