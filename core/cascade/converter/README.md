# CASCADE Model Converter v0.1

Conversor de checkpoints open-weight para uma representação física menor baseada
na arquitetura CASCADE.

A base técnica é:

```text
W ~= W0 + R1

W0 = F0 INT4 groupwise
R1 = U @ V^T  (low-rank)
```

A saída é um diretório de desenvolvimento `CASCADE-DIR/0.1`.

## O que esta versão faz

- lê checkpoints locais Safetensors, inclusive shards;
- lê um tensor por vez e matrizes em chunks de linhas;
- mantém arquitetura/config/tokenizer como sidecars;
- converte matrizes elegíveis para INT4 empacotado;
- mede cosine e NRMSE locais;
- se INT4 sozinho não passar, calcula residual low-rank por randomized SVD streaming;
- testa ranks 8/16/32 por padrão;
- usa o menor rank que passa o quality gate local;
- se nenhum candidato passar, faz fallback exato para os bytes originais;
- deixa embeddings/lm_head e MoE em passthrough por padrão na Fase 1;
- grava relatório de compressão;
- gera um registro compatível com o dashboard (schema v2, `CONVERTER_STATIC_V1`);
- nunca declara qualidade end-to-end apenas pelos pesos;
- respeita um orçamento de disco (`--disk-budget-gb`, padrão 75 GB) e aborta
  com estado retomável antes de esgotar o disco;
- grava `cascade_manifest.json` incrementalmente (tmp + rename atômico) após
  cada tensor, permitindo `--resume`;
- opcionalmente apaga cada shard `.safetensors` de origem após todos os seus
  tensores serem convertidos e verificados (`--delete-source-shards`);
- opcionalmente publica o registro no dashboard (`--publish`, HTTPS + token).

## Importante

Este conversor **não reduz a quantidade lógica de parâmetros/camadas**. Ele cria
uma representação física menor dos mesmos pesos:

```text
modelo original
      ↓
CASCADE converter
      ↓
F0 compacto + F1 residual seletivo
```

Para obter ganho de Tok/s e RAM em execução é necessário o runtime CASCADE,
que deve consumir F0 diretamente sem reconstruir a matriz original inteira.

## Dependências

Para NPZ/self-test:

```bash
pip install numpy
```

Para Safetensors:

```bash
pip install numpy torch safetensors
```

## Self-test

```bash
python cascade_converter.py self-test
```

## Converter um modelo Hugging Face já baixado

```bash
python cascade_converter.py convert \
  --input /models/Qwen2.5-7B-Instruct \
  --output /models/Qwen2.5-7B-Instruct.cascade \
  --ranks 8,16,32 \
  --group-size 64
```

## Converter um único arquivo

```bash
python cascade_converter.py convert \
  --input model.safetensors \
  --output model.cascade
```

## Inspecionar resultado

```bash
python cascade_converter.py inspect model.cascade
```

## Limite de disco, pacote-por-pacote e retomada (Colab ~120 GB)

Checkpoints grandes não cabem duas vezes no disco do Colab (fonte + saída).
O fluxo recomendado é converter shard por shard, liberando cada shard de
origem assim que ele termina:

```bash
python cascade_converter.py convert \
  --input /content/modelo \
  --output /content/modelo.cascade \
  --disk-budget-gb 75 \
  --delete-source-shards \
  --resume
```

Como funciona:

1. `--disk-budget-gb 75` (padrão): antes de cada tensor o conversor consulta
   `shutil.disk_usage` e projeta o pico de escrita do tensor. Se a projeção
   estourar o orçamento de saída ou derrubar o disco livre abaixo da margem
   mínima (1 GiB), ele aborta LIMPO: o manifesto incremental já está no disco
   e a mensagem instrui a re-executar com `--resume`. Valores `<= 0` desativam
   o orçamento (a margem mínima de disco livre continua valendo).
2. `--delete-source-shards` (opt-in): ordena os tensores por shard de origem
   (modo pacote-por-pacote) e, quando TODOS os tensores de um shard foram
   convertidos e verificados (existência + tamanho dos artefatos), apaga o
   shard e imprime os bytes liberados. Recusado quando o diretório de entrada
   é o mesmo da saída. Fora do Colab (caminhos fora de `/content` e `/tmp`) a
   deleção exige `RIFT_ALLOW_LOCAL_CLEANUP=1`.
3. `--resume`: relê `cascade_manifest.json` (gravado de forma atômica após
   cada tensor) e pula tensores já completos e verificados. Registros cujos
   shards de origem já foram apagados em execuções anteriores são preservados
   no novo manifesto. Use os MESMOS flags da execução original.
4. Progresso: uma linha por tensor e, ao final de cada shard, uma linha
   `[shard k/n]` com o disco livre atual.

O mesmo `--force` continua existindo para recomeçar do zero, mas a remoção
do diretório de saída é destrutiva e, fora do Colab, exige
`RIFT_ALLOW_LOCAL_CLEANUP=1`.

## Publicar no dashboard (opcional)

```bash
export RIFT_RESULTS_ENDPOINT="https://rift-lm.vercel.app/api/results"
export RIFT_INGEST_TOKEN="<token com 32+ caracteres>"
python cascade_converter.py convert ... --publish
```

Desligado por padrão. O publisher recusa endpoints que não sejam HTTPS e
tokens com menos de 32 caracteres (no Colab, `RIFT_INGEST_TOKEN` também é
buscado em `google.colab.userdata`). Nunca versione o token; use Colab
Secrets ou variáveis de ambiente.

O `dashboard_battery.json` gerado segue o schema v2: `schema_version: 2`,
`benchmark_protocol: "CONVERTER_STATIC_V1"`, `comparison_group_id`/
`comparison_context`, `implementation {kind: REFERENCE_MEASURED}` e campos
`candidate_*` ao lado dos aliases legados `rift_*`. Os campos `*_ram_bytes`
de nível superior são `null` (este conversor não mede RSS de runtime); as
estimativas aritméticas de representação ficam em
`metrics.memory.estimated_*`. Tok/s permanece `null`.

## Estrutura gerada

```text
model.cascade/
├── cascade_manifest.json
├── cascade_ir.json
├── gate_config.json
├── dashboard_battery.json
├── source_config/
└── tensors/
    ├── 000000_.../
    │   ├── f0.int4
    │   ├── f0.scales.f16
    │   ├── f1.u.f16
    │   └── f1.v.f16
    └── ...
```

Nem todo tensor terá F1. Tensores não elegíveis ou que falharem no gate local
podem aparecer como:

```text
f0.raw
```

Isso é intencional: o conversor prefere preservar o tensor a fingir sucesso.

## Gate

O conversor de pesos não possui ativações reais suficientes para calibrar um
Confidence Gate dinâmico por token. Por isso:

```text
gate_status = CALIBRATION_REQUIRED
safe_runtime_default = F1_ALWAYS_WHEN_PRESENT
```

O próximo componente deve executar prompts de calibração e aprender o threshold
para:

```text
F0_ONLY
versus
F0 + F1
```

## Grandes modelos

O converter não carrega o checkpoint completo. Ele:

1. lê o header Safetensors;
2. processa um tensor por vez;
3. lê matrizes por chunks de linhas;
4. grava F0 imediatamente;
5. calcula o residual low-rank por passes streaming.

Ainda assim, a fatoração low-rank pode ser computacionalmente cara para matrizes
gigantes. Use ranks menores e `--power-iters 0` para uma primeira exploração.

Para o limite de disco (Colab ~120 GB), retomada e deleção de shards de origem,
veja a seção "Limite de disco, pacote-por-pacote e retomada".

## Política conservadora da Fase 1

Por padrão não converte:

- embeddings;
- lm_head;
- tensors MoE/expert;
- tensores 1D;
- matrizes muito pequenas.

Habilite explicitamente quando o runtime correspondente existir:

```bash
--include-embeddings
--include-moe
```

## Status

O resultado é um **modelo CASCADE compilado em formato de desenvolvimento**,
não um modelo diretamente carregável por `AutoModelForCausalLM`.

Para torná-lo executável como LLM é necessário o CASCADE Runtime C++/CPU/GPU.
