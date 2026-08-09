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
- gera um registro compatível com o dashboard;
- nunca declara qualidade end-to-end apenas pelos pesos.

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
