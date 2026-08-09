# RIFT-LM Test Observatory — Vercel

Dashboard estático para acompanhar baterias de benchmark RIFT-LM.

## Estrutura

```text
rift-dashboard-vercel/
├── index.html
├── vercel.json
├── .vercelignore
├── data/
│   ├── rift_test_batteries.json
│   └── record-schema-example.json
└── scripts/
    └── publish_batteries.py
```

## Deploy com Vercel CLI

Na raiz deste projeto:

```bash
npm i -g vercel
vercel
```

Para produção:

```bash
vercel --prod
```

A CLI pode solicitar login e associação/criação do projeto na primeira execução.

## Deploy pelo Dashboard/Drop

Também é possível enviar esta pasta como projeto estático para a Vercel.

## Atualizar os testes publicados

O dashboard carrega automaticamente:

```text
/data/rift_test_batteries.json
```

Copie o histórico produzido pelo script de testes RIFT:

```bash
python scripts/publish_batteries.py /caminho/para/rift_test_batteries.json
```

Depois faça novo deploy/commit.

## Publicação automática pelo Google Colab

O script `rift_m0_phase1_test_v035_auto_batteries.py` pode mesclar os resultados
no arquivo `data/rift_test_batteries.json` por meio da Function `/api/results`.
O PAT do GitHub fica somente no Vercel; o Colab recebe apenas uma chave de
ingestão independente. O commit de dados dispara automaticamente um novo deploy.

1. Crie um Fine-grained Personal Access Token no GitHub limitado a este
   repositório, com permissão **Contents: Read and write**.
2. Abra o `.env` local, substitua `RIFT_GITHUB_TOKEN` pelo PAT e importe o
   arquivo em **Vercel > Project Settings > Environment Variables** para
   Production, Preview e Development. O `.env` é ignorado pelo Git.
3. Faça um novo deploy para aplicar as variáveis.
4. No painel **Secrets** do Colab, cadastre:
   - `RIFT_RESULTS_ENDPOINT`: `https://SEU-DOMINIO.vercel.app/api/results`;
   - `RIFT_INGEST_TOKEN`: exatamente o mesmo valor presente no `.env` importado.
5. Autorize o notebook a acessar esses dois secrets e execute:

```bash
python rift_m0_phase1_test_v035_auto_batteries.py \
  --mode phase1 \
  --device cuda \
  --publish required
```

Também é possível informar o endpoint sem salvá-lo como secret:

```bash
python rift_m0_phase1_test_v035_auto_batteries.py \
  --mode phase1 \
  --device cuda \
  --publish required \
  --results-endpoint https://SEU-DOMINIO.vercel.app/api/results
```

O PAT do GitHub nunca deve ser colocado no Colab, passado como argumento ou
gravado no notebook. O modo `auto` é o padrão e, dentro do Colab, falha
explicitamente quando endpoint ou chave não estão configurados. Use
`--publish off` somente para execuções locais.

## Métricas

O dashboard aceita, por bateria:

- `baseline_tok_s` / `rift_tok_s`
- `baseline_ram_bytes` / `rift_ram_bytes`
- `baseline_disk_bytes` / `rift_disk_bytes`
- `quality`
- `gains`
- `metrics.operation`

Quando Tok/s real de modelo não foi medido, mantenha os campos como `null`.
Não substitua Tok/s por throughput de uma única operação Linear.

## Dados persistentes

Esta versão publica os resultados como arquivo estático versionado junto com o deploy.
Isso é deliberado para a Fase 1: cada deploy preserva exatamente o conjunto de
resultados usado naquele build. A API de ingestão não grava no filesystem efêmero
da Function: ela atualiza o arquivo no GitHub, preservando o histórico e acionando
o redeploy da branch de produção.
