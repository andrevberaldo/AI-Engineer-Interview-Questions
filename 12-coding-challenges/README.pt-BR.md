# 🧑‍💻 Coding Challenges

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Quase todo loop de AI engineer hoje inclui uma rodada de "implemente do zero". Os entrevistadores usam estes exercícios para separar quem *usou* transformers, RAG e agents de quem os *entende* - você não consegue enrolar uma causal mask ou um softmax numericamente estável só com conhecimento de API. Frontier labs e startups de IA sérias rotineiramente pedem attention, tokenizers, sampling e KV caches em numpy puro; times voltados a produto se inclinam para as variantes de RAG, agent loop e streaming parser.

## Regras do jogo

- **Dependências:** apenas `numpy` + biblioteca padrão do Python. Sem PyTorch, sem transformers, sem muletas de framework. Esse é o ponto.
- **Autocontido:** todo arquivo roda isoladamente com `python3 <file>.py` e termina com `All tests passed.`
- **Estrutura:** cada arquivo tem um enunciado `PROBLEM` (o que lhe seria pedido, com assinaturas exatas), `INTERVIEW NOTES` (o que soluções fortes demonstram, erros comuns, variações de follow-up), uma solução de referência limpa e testes baseados em assert.

**Como praticar:** leia apenas a docstring, feche o arquivo, implemente você mesmo sob pressão de tempo (30-45 min é o típico), depois rode os testes e compare com a referência. Conseguir *falar enquanto programa* - nomeando o truque de estabilidade, a complexidade, o modo de falha - é o que de fato é avaliado.

## Os desafios

| # | Arquivo | Dificuldade | Conceitos testados | O que os entrevistadores procuram |
|---|------|------------|-----------------|----------------------------|
| 01 | [01_attention.py](01_attention.py) | Médio | Scaled dot-product attention, multi-head, causal masking | Softmax estável, máscara como −inf *antes* do softmax, escala √d_k, split de heads via reshape/transpose e não com loops |
| 02 | [02_bpe_tokenizer.py](02_bpe_tokenizer.py) | Médio | BPE em nível de byte: train, encode, decode | Ordem de merge-rank no encode, round-trip em nível de byte para qualquer unicode, desempate determinístico |
| 03 | [03_sampling.py](03_sampling.py) | Fácil | Temperature, top-k, top-p, min-p, repetition penalty | Ordem correta das operações, filtrar com −inf e depois renormalizar, edge cases (T→0, p=1, k=1) |
| 04 | [04_positional_encodings.py](04_positional_encodings.py) | Médio | Sinusoidal PE, RoPE | Por que o RoPE codifica posição *relativa*, rotação preservando normas, intuição do espectro de frequências |
| 05 | [05_layernorm_and_softmax.py](05_layernorm_and_softmax.py) | Fácil | Softmax/log-softmax estáveis, LayerNorm, RMSNorm | Truque da subtração do máximo, posição do eps, o que o RMSNorm descarta e por que é mais barato |
| 06 | [06_kv_cache.py](06_kv_cache.py) | Médio | Loop de decode autorregressivo com vs. sem KV cache | Custo de attention por passo de O(T²)→O(T), equivalência exata de saída, contabilidade de memória do cache |
| 07 | [07_mini_gpt_forward.py](07_mini_gpt_forward.py) | Difícil | Forward pass completo de um bloco GPT: embeddings → attention → MLP → logits | Ligação residual pre-norm, shapes corretos de ponta a ponta, weight tying, onde vivem os FLOPs |
| 08 | [08_semantic_search_rag.py](08_semantic_search_rag.py) | Médio | Embed, index, retrieval por cosseno, montagem de contexto (mini-RAG) | Normalização de vetores, corretude do retrieval top-k, empacotamento de contexto com token budget |
| 09 | [09_text_chunking.py](09_text_chunking.py) | Fácil | Chunkers fixo, sliding-window, recursivo e por sentença | Tratamento de overlap, preservação de fronteiras, nenhum texto perdido/duplicado nas bordas |
| 10 | [10_agent_loop.py](10_agent_loop.py) | Médio | Agent loop com tool calling usando um LLM mock | State machine limpa, feedback do resultado da tool no contexto, tratamento de máximo de iterações e de erros |
| 11 | [11_rate_limiter_and_retry.py](11_rate_limiter_and_retry.py) | Médio | Token bucket, exponential backoff com jitter | Raciocínio sobre burst vs. taxa sustentada, por que o jitter evita thundering herd, limites de retry budget |
| 12 | [12_eval_metrics.py](12_eval_metrics.py) | Médio | Estimador não enviesado de pass@k, QA F1/EM, esqueleto de judge harness | Por que o estimador ingênuo de pass@k é enviesado, detalhes do F1 em nível de token, hooks de calibração do judge |
| 13 | [13_streaming_parser.py](13_streaming_parser.py) | Difícil | Parser de SSE, montagem incremental de argumentos de tool call | Lidar com chunks cortados no meio de um evento/JSON, disciplina de buffering, recuperação de parse parcial |
| 14 | [14_lora_adapter.py](14_lora_adapter.py) | Médio | Forward de LoRA, merge e serving batched multi-adapter | Por que B é inicializado com zero, o que alpha/r desacopla, por que o merge custa a troca a quente (hot-swapping) |
| 15 | [15_beam_search.py](15_beam_search.py) | Médio | Beam search com normalização de comprimento | Acumular log-probs em vez de multiplicar, estacionar beams finalizados, o que a length penalty compra |
| 16 | [16_semantic_cache.py](16_semantic_cache.py) | Médio | Tiers de cache exato e semântico, TTL, LRU, invalidação por tag | Threshold como decisão de negócio de precision/recall, por que as chaves de cache devem incluir o prompt e a versão do modelo |
| 17 | [17_hybrid_search_and_rerank.py](17_hybrid_search_and_rerank.py) | Médio | BM25, reciprocal rank fusion, re-ranking de diversidade MMR | Onde o lexical vence o dense e vice-versa, por que o RRF funde ranks em vez de scores |
| 18 | [18_constrained_json_decoding.py](18_constrained_json_decoding.py) | Difícil | Mascarar tokens inválidos contra um schema durante o decode | Mascarar antes do softmax, por que constrained decoding desloca a distribuição, DFA vs pushdown para JSON aninhado |
| 19 | [19_speculative_decoding.py](19_speculative_decoding.py) | Difícil | Draft-and-verify com a regra exata de rejection-sampling | Por que é lossless, por que aceitar-se-o-argmax-coincide não é, por que é um ganho de bandwidth e não de FLOPs |

## Executando

```bash
python3 12-coding-challenges/01_attention.py
# ...
# All tests passed.
```

A única dependência é o numpy (`pip install numpy`).
