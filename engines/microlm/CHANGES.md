# MicroLM — correções aplicadas sobre o diagrama original

Implementação de referência em `microlm/model.py`; cada correção é provada por
teste em `microlm/test_model.py`. Config de referência: d=512, L=27, V=8192,
GQA 8q/2kv (d_h=64), 4 lanes, janela 256 + sinks. Parâmetros ativos ≈ 22M
(fora tabelas engram) — verificado por teste.

## Correções (before → after → por quê)

### 1. Gate do engram estava morto ★ crítico
- **Antes:** `x_t ← x_t + σ(⟨x̂_t, k̂_t⟩/√d)·v_t`
- **Problema:** com x̂ e k̂ normalizados, ⟨x̂,k̂⟩ ∈ [−1,1]; dividido por √512
  o argumento fica em ±0.044 e o gate preso em **[0.489, 0.511]** — não gateia.
- **Depois:** `σ(τ_o·⟨x̂,k̂⟩ + β_o)` com τ (init 4.0) e β (init −2.0)
  aprendidos por ordem de n-grama; span do gate ≈ [0.0025, 0.88].
- **Extras:** K=4 hashes por ordem (bigrama + trigrama, tabelas separadas) —
  corrupção por colisão cai de ~38% para ~2% nos n-gramas frequentes;
  metade-valor da tabela zero-init (fusão nasce no-op).
- **Refinado pelo review adversarial:** o hash original era **afim no salt**
  (salt vira deslocamento aditivo constante mod 2^15 — colisão em 1 hash
  implicava colisão nos 4, anulando o multi-hash). Fix: multiplicador ímpar
  dependente do salt + mistura xor-shift; colisões entre salts agora são
  independentes (taxa ≈ 1/buckets).
- Testes: `test_engram_gate_spans_wide_range_vs_legacy_dead_zone`,
  `test_engram_gate_modulates_output_through_real_module` (exercita o módulo
  real; mata a mutação que reintroduz o /√d),
  `test_hash_salts_produce_independent_collisions`.

### 2. Init do mHC write: "A=0 ⇒ P=I" é falso ★ crítico
- **Antes (receita intuitiva):** A=0, r=c=1 ⇒ P=I.
- **Problema:** P = Sinkhorn(exp∘A) com exponencial **elemento a elemento**:
  A=0 ⇒ exp(A)=11ᵀ ⇒ Sinkhorn converge para **J/n (média uniforme das
  lanes)**, não identidade. P=I é vértice inatingível do politopo de Birkhoff
  com entradas positivas.
- **Depois:** `A = α·I` com α=8 ⇒ P com diagonal > 0.999 (quase-identidade
  atingível). Sinkhorn com 15 iterações, diferenciável.
- **Refinado pelo review adversarial:** `g` zero-init criava um **ponto de
  sela duplo-zero** — com g=0 E y=0 (W_O=0, D₃=0, v=0), dL/dg ∝ y = 0 e
  dL/dy ∝ g = 0: gradiente exatamente zero para TODOS os parâmetros de
  camada, para sempre (verificado: 200–500 passos de Adam sem sair do
  lugar). Fix: **g=1** — o no-op exato do init é preservado (y=0 garante
  g·y=0) e o caminho de gradiente fica aberto; a cascata destrava em ~5
  passos e a loss cai de 6.28 para 0.012 em 200 passos.
- Testes: `test_mixing_matrix_is_doubly_stochastic_and_near_identity`,
  `test_model_trains_from_exact_init_without_perturbation` (mata a mutação
  g=0).

### 3. Cadeia de estabilidade deep-narrow (27×512)
- **Adicionado:** QK-RMSNorm por head antes do RoPE (sinks concentram massa de
  atenção e explodem logits com d_h=64); zero-init em W_O e em D₃ do MLP;
  lane read iniciando como média exata (a=0, σ(b)=1/K).
- **Propriedade resultante:** no init o modelo inteiro é um **no-op exato**
  (logits = readout(embedding)) — cada camada só escreve quando aprende algo.
- Testes: `test_model_init_is_exact_noop_pipeline`,
  `test_residual_stream_norm_stays_bounded_across_layers`,
  `test_lane_read_at_init_equals_mean_of_lanes`.

### 4. RoPE sob evicção: posições cache-relativas no decode
- **Antes:** posições absolutas; com evicção da janela, a distância
  query↔sinks cresce sem limite e sai da distribuição de treino.
- **Depois (estilo StreamingLLM):** o cache guarda K **pré-RoPE**; a rotação é
  aplicada na hora da atenção com posições = índice no cache (sinks 0..s−1,
  janela s..s+w−1). Distâncias intra-janela ficam idênticas às absolutas
  (RoPE é relativo); a distância aos sinks fica **constante**.
- **Limite documentado:** o caminho de treino paralelo suporta T ≤ sinks +
  janela (regime sem evicção, onde absoluto ≡ cache-relativo); sequências
  maiores usam o caminho de decode. Levantar esse limite exige atenção
  chunked — fora do escopo desta referência.
- **Refinado pelo review adversarial:** o caminho de decode agora rejeita
  entrada multi-token com `ValueError` (antes aceitava T>1 em silêncio com
  broadcast de posição única e sem máscara causal — resultado errado); e o
  regime de evicção ganhou teste discriminante de **estacionariedade**
  (entrada constante ⇒ cache estável ⇒ saída constante; posições absolutas
  derivariam a cada passo).
- Testes: `test_decode_matches_training_forward_for_short_sequences`,
  `test_decode_cache_stays_bounded_for_long_generation`,
  `test_decode_attention_is_stationary_after_eviction_with_constant_input`,
  `test_decode_rejects_multi_token_input`,
  `test_training_path_rejects_sequences_beyond_window_plus_sinks`,
  `test_attention_sink_tokens_remain_visible_beyond_window`.

### 5. MLP Hadamard: gate de 2 ramos + bloco-diagonal opcional
- **Antes:** `y = D₃H σ(D₂H D₁x)` — 1.536 graus de liberdade/camada, sem
  interação multiplicativa entre features.
- **Depois:** `y = D₃·H·(σ(D₂ₐH D₁ₐx) ⊙ (D₂ᵦH D₁ᵦx))` (gating estilo SwiGLU,
  +2d params/camada) com D₂ₐ opcionalmente bloco-diagonal b=32
  (`mlp_block_size=32`, +~0.44M no modelo de referência) — **ligar só se a
  ablação memorizável×composicional justificar**; init identidade reproduz a
  variante diagonal exatamente.
- Teste: `test_hadamard_mlp_blockdiag_init_matches_diagonal_variant`.

### 6. Readout: `mean lanes` → mistura aprendida
- **Antes:** média fixa das lanes.
- **Depois:** `Σ softmax(θ)_k·X_k` com θ=0 no init (= média exata no passo 0;
  o gradiente decide). Escala de logits aprendida no unembed tied
  (`logit_scale`, relevante para muP).

### 7. Sites do engram: interpretação fixada e configurável
- "Two sites" foi implementado como **duas profundidades** (camadas 4 e 14,
  `engram_layers`), tabelas próprias por ordem — consistente com o Engram
  original (memória em camadas específicas) e com o orçamento de latência de
  acessos aleatórios em CPU. Se a intenção era 2 fusões por camada (54
  lookups/token), basta mudar o config — mas o custo de latência DRAM
  precisa ser re-orçado.
- Teste: `test_engram_lookup_is_deterministic_and_position_consistent`
  (paridade exata entre o caminho de treino e o de decode token-a-token).

## O que NÃO está nesta referência (decisões que continuam com o time)

- Tokenizer 8192 (dígito-a-dígito para IDs/valores) e o mapeamento
  token→bytes da máscara do PDA da gramática.
- Compilador JSON Schema → PDA, jump-forward, unembed mascarado, pointer
  head e argument sinks dinâmicos (propostas P0/P1 do review — mudanças de
  produto, não correções do diagrama).
- Treino (KD de sequência do Qwen, loss renormalizada pela gramática, QAT
  INT2/INT4 alinhado ao kernel LUT do GEYSER).
- Segurança (injeção via cópia verbatim), LGPD nas tabelas engram,
  multi-turn, versionamento atômico dos artefatos.
