# WINNER.cpp v0.8

Runtime C++17 experimental para inferência progressiva com base ternária F0, residual low-rank, planejamento híbrido, paginação de experts e caminhos de referência para atenção e quantização.

## Estado real da implementação

O projeto é uma referência de pesquisa, não um runtime de produção. O benchmark nativo mede kernels sintéticos do próprio WINNER; ele não equivale a throughput end-to-end de um LLM. CUDA/HIP, io_uring verdadeiro, kernels AMX/AVX-512, parsing completo de Bundle, servidor OpenAI completo e carregamento de tensores reais ainda não estão implementados.

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
bash winner_cpp/setup_test.sh
```

Não use `cd winner.cpp`: o diretório do repositório chama-se `winner_cpp`. Para
também compilar uma cópia já clonada do llama.cpp, informe-a explicitamente:

```bash
LLAMA_CPP_DIR=/workspace/llama.cpp GGUF_MODEL=/models/model.gguf \
  bash winner_cpp/setup_test.sh
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
./winner --devices
./winner --vcpus
./winner --quants
./winner --bench-kernels --dim 256 --layers 8 --tokens 16 --output winner_profile_bench.json
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

## Perfis

| Perfil | Rank residual alvo |
|---|---:|
| MINMEM | 0 |
| FAST | 16 |
| BALANCED | 64 |
| SAFE | 128 |

O rank é limitado pela menor dimensão do tensor.
