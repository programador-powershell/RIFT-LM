# WINNER.cpp v0.8

Runtime C++17 experimental para inferência progressiva com base ternária F0, residual low-rank, planejamento híbrido, paginação de experts e caminhos de referência para atenção e quantização.

## Estado real da implementação

O projeto é uma referência de pesquisa, não um runtime de produção. O benchmark nativo em modo sintético mede kernels do próprio WINNER; ele não equivale a throughput end-to-end de um LLM. CUDA/HIP, io_uring verdadeiro, kernels AMX/AVX-512, servidor OpenAI completo e execução de um modelo completo ainda não estão implementados.

O que JÁ é real: o loader parseia o container CASCADE `CSCD` v0x0003 (o mesmo escrito por `cascade/compiler/bundle_writer.py`), valida todos os offsets/tamanhos contra o tamanho do arquivo antes de qualquer acesso, verifica o CRC32 e carrega UMA camada real (F0 INT4 groupwise + residual low-rank F1) executada por kernels nativos (`gemv_int4_group` + `gemv_lowrank_add`) com o gate de energia decidindo o residual. Isso é uma camada, não um modelo — o tok/s reportado continua não sendo throughput de modelo.

O dashboard usa uma bateria separada para carregar um tensor real do Hugging Face, aplicar F0 + residual LS e medir qualidade/armazenamento. Ela compila e executa também o `--self-test` deste projeto, mas identifica explicitamente as latências Python e C++ para não misturar escopos.

## Correções da revisão v0.8

- remove o estouro de buffer do payload Q4;
- impede GEMV in-place de corromper a ativação usada pelo residual;
- valida cabeçalho, offsets, versão e limites do Bundle antes de mapear estágios;
- corrige consultas CPUID e não anuncia ISA sem kernel compilado;
- estabiliza ponteiros de KV pages, batches e experts concorrentes;
- corrige invalidação de referência na árvore especulativa;
- sincroniza o prefetch fallback e respeita timeout/conclusões;
- valida argumentos CLI e dimensões antes de alocar memória;
- grava JSON no caminho informado por `--output`, sem caminho absoluto;
- torna o build portátil por padrão e adiciona testes CTest.

## Build

Linux/macOS:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

No ambiente de execução mostrado na interface, o script de configuração correto
(a partir da raiz clonada `/workspace/RIFT-LM`) é simplesmente:

```bash
bash engines/winner/cpp/setup_test.sh
```

Não use `cd winner.cpp`: o diretório do repositório chama-se `engines/winner/cpp`
(ex-`winner_cpp/`, árvore canônica do §20). Para também compilar uma cópia já
clonada do llama.cpp, informe-a explicitamente:

```bash
LLAMA_CPP_DIR=/workspace/llama.cpp GGUF_MODEL=/models/model.gguf \
  bash engines/winner/cpp/setup_test.sh
```

Para um benchmark ligado à CPU da máquina, ative explicitamente:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DWINNER_NATIVE=ON
cmake --build build-native --parallel
```

O build portátil não usa `-march=native` nem `fast-math` por padrão, evitando binários que falham em outra CPU e mudanças silenciosas de semântica numérica.

## CLI

```bash
./winner --self-test
./winner --self-test --bundle caminho/model.cascade   # inclui load+decode sobre bundle real
./winner --devices
./winner --vcpus
./winner --quants
./winner --bench-kernels --dim 256 --layers 8 --tokens 16 --output winner_profile_bench.json
./winner --bundle caminho/model.cascade --tokens 64   # bench de perfis sobre a camada real
./winner --model arquivo.winr
```

Em Unix, o servidor experimental pode ser iniciado com `--serve`. Ele liga em
`127.0.0.1` por padrão e oferece `GET /health`, `GET /v1/models` e
`POST /v1/chat/completions` (incluindo SSE). Ele **ainda retorna respostas
sintéticas**: carregar um Bundle valida o runtime, mas o caminho HTTP ainda não
executa os tensores do modelo. Não o exponha à internet.

Os nomes comerciais de uma família não são arquivos testáveis. Um ensaio
reproduzível precisa registrar o identificador exato do checkpoint, revisão,
formato/quantização, licença, prompt, hardware e comando. WINNER usa `.winr`,
enquanto llama.cpp usa GGUF; portanto, comparar ambos exige conversões verificadas
do mesmo checkpoint. Resultados do `--bench-kernels` são microbenchmarks sintéticos
e não demonstram superioridade end-to-end sobre llama.cpp ou BitNet.

## Bundles CSCD (v0x0003) e labels de workload

O loader aceita, além dos containers legados (`WINR`/`SPCT`/`AETH`/`CASC`/`RIFT`,
versão 0x0100), o container real do CASCADE:

- magic `CSCD`, versão `0x0003`, header de 128 bytes;
- stage table com entradas de 24 bytes (`offset u64, size u64, stage_id u32, flags u32`);
- meta JSON por estágio (`BASE_STAGE`/`INT4_GROUP` e `RESIDUAL_LOWRANK`/`FP32_LOWRANK`);
- checksum CRC32 (zlib) sobre tudo após o header, armazenado em u64 — bundles com
  CRC divergente, offsets fora dos limites ou tensores truncados são REJEITADOS
  com erro claro (nenhum acesso fora dos limites).

Com `--bundle`, o runtime constrói uma camada real a partir dos tensores F0/F1 do
bundle e o `winner_profile_bench.json` marca `"workload": "real_bundle"`. Nesse
modo o residual é decidido SOMENTE pelo gate de energia real (nunca pelo fallback
simulado por hash). Sem bundle (ou com bundle legado sem tensores), o bench roda
camadas sintéticas e grava `"workload": "synthetic"`; quando a taxa de residual
foi dirigida pelo fallback por hash de perfil (e não pelo gate), o registro do
perfil traz `"residual_simulated": true`.

Importante: tok/s de workload `synthetic` NÃO é throughput de modelo — mede
apenas os kernels do WINNER sobre pesos aleatórios. Mesmo em `real_bundle`, é
uma única camada real, não um modelo completo.

## Perfis

| Perfil | Rank residual alvo |
|---|---:|
| MINMEM | 0 |
| FAST | 16 |
| BALANCED | 64 |
| SAFE | 128 |

O rank é limitado pela menor dimensão do tensor.
