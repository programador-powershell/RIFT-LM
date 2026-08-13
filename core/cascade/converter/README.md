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

## Entrada GGUF (llama.cpp)

```bash
python core/cascade/converter/cascade_converter.py convert \
  --input modelo.gguf --output modelo-cascade --codec-ladder compact
```

Requer `pip install 'gguf>=0.10,<1'`. A leitura é **streaming por blocos de
linhas**: o conversor desquantiza apenas a fatia pedida (o ggml não deixa um
bloco cruzar a fronteira de uma linha), então o pico de RAM fica em
`chunk_rows × colunas × 4 bytes` em vez do tensor inteiro — a diferença entre
alguns MB e vários GB numa embedding grande. Um GGUF já quantizado (IQ2/Q4_K…)
é desquantizado para float32 por fatia antes de entrar no codec, então
converter de IQ2 quantiza duas vezes: prefira a origem de maior precisão
disponível quando ela existir.

GGUF multi-parte não é suportado (aponte `--input` para o arquivo).

O formato é decidido pelos **magic bytes**, não pelo nome: `--input` pode apontar
direto para um caminho do cache do Hugging Face
(`.../snapshots/<rev>/model.gguf`), cujo alvo real é um blob sem extensão.
Nenhum hardlink ou renomeação é necessário.

### Política Fase-1 (o que NÃO é convertido)

Embeddings, cabeça de saída e MoE ficam em passthrough. Os nomes mudam por
formato e a política cobre os dois:

| papel | HF | ggml/GGUF | liberado por |
| --- | --- | --- | --- |
| embeddings | `embed_tokens`, `wte` | `token_embd` | `--include-embeddings` |
| cabeça de saída | `lm_head`, `output_projection` | `output.weight` | `--include-embeddings` |
| MoE | `experts`, `moe` | `ffn_*_exps` | `--include-moe` |

O padrão da cabeça de saída é **ancorado no início**: `blk.N.attn_output.weight`
contém "output" e é um linear legítimo — precisa continuar elegível.

São os maiores tensores do modelo (no Muse-Glimmer-30B o `token_embd` tem 1,34 G
elementos), então incluí-los é o principal risco de pico de RSS e de disco. Em
modelo grande, combine com `--keep-source-passthrough` para não copiá-los.

## Escada de codecs (`--codec-ladder`)

Por tensor, tenta do mais barato ao mais caro e fica no **primeiro degrau que
passa o gate de qualidade**; se nenhum passa, o tensor vai para passthrough
exato. O gate é o mesmo em todos os degraus, então a escada reduz bytes sem
afrouxar qualidade.

| modo | degraus | quando usar |
| --- | --- | --- |
| `auto` (padrão) | escolhe pela fonte | fonte já low-bit (IQ2/Q4_K/…) → `safe`; BF16/F16/F32 → `compact` |
| `safe` | int4/g64 → int4/g32 → raw | mesmo primeiro degrau de sempre, com um resgate mais fino antes do raw (troca 16 bpw por 4.5 bpw nos tensores que hoje caem em passthrough) |
| `compact` | int2/g64 → int4/g64 → int4/g32 → raw | alvo 8 GB: tenta 2.5 bpw primeiro |
| `int4` | int4/g64 → raw | compatibilidade estrita com versões anteriores |

`auto` existe porque o INT2 só compensa quando o raw é caro. Medido no Muse
Glimmer (fonte IQ2): INT2/g64 nunca passou o gate (cosine 0.91–0.92) e o raw
custa 2.66 bpw, não 16 — tentar INT2 ali é só custo de CPU. Numa fonte BF16 o
mesmo degrau vale a tentativa.

`--ladder-f0-min-cosine` (padrão 0.98) é uma heurística de custo: num degrau
intermediário cujo F0 fique muito longe do gate, o SVD do residual é pulado.
Use 0 para sempre tentar o F1.

## Guarda de expansão de bytes

Um degrau que ficaria **maior ou igual à fonte** não interessa nem passando o
gate: o passthrough exato é menor *e* sem perda. Antes de escrever cada F0 o
conversor projeta os bytes e, se houver expansão, pula direto para o raw
(`ladder.stopped_by = "byte_expansion_guard"`); a checagem se repete no total
F0+F1. Medido em `o_proj` 256×512 com fonte IQ2 ~2.66 bpw:

| decisão | bytes | perda |
| --- | --- | --- |
| guarda ativa → passthrough | 43 581 | nenhuma (bit a bit) |
| `--allow-byte-expansion` → INT4/g64 | 69 632 | quantização |

`--allow-byte-expansion` desliga a guarda (útil para medir a alternativa).

## Passthrough sem cópia (`--keep-source-passthrough`)

Tensores que não entram no CASCADE (embeddings, `lm_head`, MoE, fora do
`--include-regex`) são copiados byte a byte por padrão. Em modelo grande essa
cópia é o pico de disco e de RAM da conversão. Com a flag eles viram
`SOURCE_EXTERNAL`: ficam **apenas** no checkpoint de origem, com `bytes: 0` no
bundle e `external_bytes` registrando o tamanho real.

O manifesto marca `residency.bundle_requires_source = true` — o bundle passa a
depender do arquivo de origem, e `verify` recusa o tensor se o checkpoint
desapareceu. **Economiza disco, não RAM de execução**: `external_bytes` continua
somando no residente (HOT), porque o peso ainda precisa estar em memória para
rodar.

### O bundle pequeno NÃO é o ganho

Com external, o bundle encolhe porque os bytes não foram copiados — comparar
bundle-vs-fonte publica um ganho que não existe. Medido no Muse-Glimmer-30B
(IQ2_XXS, 731 tensores, 688 externos):

| número | valor | serve como headline? |
| --- | --- | --- |
| fonte GGUF | 10,73 GB | baseline |
| bundle | 1,56 GB | **não** — depende da fonte |
| externo (na fonte) | 8,48 GB | — |
| disco exigido (bundle + externo) | 10,03 GB | sim |
| **TOTAL em RAM** (`all_in_ram_bytes`) | **10,03 GB** | **sim — é o headline** |
| redução real | **6,5%** | sim (bundle-vs-fonte daria 85,5%) |

Por isso `candidate_disk_bytes` é o disco EXIGIDO, `ram_reduction_pct` sai de
`all_in_ram_bytes` e o painel ordena os modelos convertidos por
`all_in_ram_bytes` ascendente. `cascade_bundle_directory_bytes` e
`bundle_disk_reduction_pct` seguem no manifesto como detalhe.

## Piso de energia do F1

Antes de avaliar os ranks, o conversor mede que fração da energia do resíduo os
ranks disponíveis capturam e compara com o **mínimo que o gate exigiria**:

```
nrmse   : 1 - (nrmse_max / nrmse_f0)²                 (exato)
cosseno : 1 - (1 - cosine_min) / (1 - cosine_f0)      (1-cos ~ nrmse²/2)
```

Vale o maior dos dois. Se a energia capturada fica abaixo de 75% desse mínimo, o
F1 é abortado (`f1_spectrum.trigger = "below_gate_requirement"`) e o tensor segue
para o próximo degrau ou para o raw — o SVD de avaliação e a gravação do F1 não
acontecem. Medido: o resíduo INT4 entrega ~0.25 com rank ≤ 32, enquanto
`attn_output` precisaria de ~0.38 para sair de cosine 0.992 e fechar 0.995 — daí
o F1 rank 8→32 não mover a métrica.

`--f1-min-energy` (padrão 0, desligado) adiciona um piso absoluto por cima.

## Relatório de residência

O fim da conversão e `summary.residency` no manifesto respondem à pergunta
prática: **cabe na máquina alvo?**

```
Escada de codecs    : compact (int2/g64 -> int4/g64 -> int4/g32 -> raw)
Degraus escolhidos  : {'int4/g64': 168, 'int2/g64': 14, 'raw': 2}
F0 médio (bpw)      : 4.108
Residente (HOT)     : 3.612 GiB (raw exato: 0.180 GiB)
Paginável (WARM F1) : 0.421 GiB
Alvo 8.0 GB livres  : CABE residente (reserva de 1.5 GiB p/ KV+runtime)
Pico de RSS medido  : 0.740 GiB (conversão)
```

`--target-ram-gb 0` (padrão) deriva o alvo da máquina: `total − 8 GiB`, com piso
de metade da RAM total (16 GB → 8, 24 GB → 16, 32 GB → 24). Um valor explícito
sobrescreve. `--ram-budget-mb 0` (padrão) escala a fatia de linhas junto com a
máquina (16 MB em 8 GiB, teto 128 MB).
