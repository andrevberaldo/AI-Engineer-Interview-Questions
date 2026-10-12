# 🧑‍💻 Coding Challenges

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Quase todo loop de AI engineer hoje inclui uma rodada de "implemente do zero". Os entrevistadores usam esses exercícios para separar quem *usou* transformers, RAG e agents de quem os *entende*: só com conhecimento de API, você não consegue enrolar na hora de escrever uma causal mask ou um softmax numericamente estável. Frontier labs e startups de IA sérias costumam pedir attention, tokenizers, sampling e KV caches em numpy puro. Já os times voltados a produto preferem as variantes de RAG, agent loop e streaming parser.

## Regras do jogo

- **Dependências:** apenas `numpy` e a biblioteca padrão do Python. Nada de PyTorch, transformers ou muletas de framework. Esse é o ponto.
- **Autocontido:** cada arquivo roda sozinho com `python3 <file>.py` e termina com `All tests passed.`
- **Estrutura:** cada arquivo tem um enunciado `PROBLEM` (o que seria pedido a você, com as assinaturas exatas), `INTERVIEW NOTES` (o que as boas soluções demonstram, erros comuns, variações de follow-up), uma solução de referência limpa e testes com assert.

**Como praticar:** leia só a docstring, feche o arquivo e implemente você mesmo, com pressão de tempo (30 a 45 min é o normal). Depois rode os testes e compare com a referência. O que de fato pesa na avaliação é conseguir *falar enquanto programa*: citar o truque de estabilidade, a complexidade e o modo de falha.

## Os desafios

| # | Arquivo | Dificuldade | Conceitos testados | O que os entrevistadores procuram |
|---|------|------------|-----------------|----------------------------|
| 01 | [01_attention.py](01_attention.py) | Médio | Scaled dot-product attention, multi-head, causal masking | Softmax estável, máscara como −inf *antes* do softmax, escala √d_k, split de heads via reshape/transpose e não com loops |
| 02 | [02_bpe_tokenizer.py](02_bpe_tokenizer.py) | Médio | BPE em nível de byte: train, encode, decode | Ordem de merge-rank no encode, round-trip em nível de byte para qualquer unicode, desempate determinístico |
| 03 | [03_sampling.py](03_sampling.py) | Fácil | Temperature, top-k, top-p, min-p, repetition penalty | Ordem correta das operações, filtrar com −inf e depois renormalizar, edge cases (T→0, p=1, k=1) |
| 04 | [04_positional_encodings.py](04_positional_encodings.py) | Médio | Sinusoidal PE, RoPE | Por que o RoPE codifica a posição *relativa*, rotação que preserva a norma, intuição sobre o espectro de frequências |
| 05 | [05_layernorm_and_softmax.py](05_layernorm_and_softmax.py) | Fácil | Softmax/log-softmax estáveis, LayerNorm, RMSNorm | Truque da subtração do máximo, onde entra o eps, o que o RMSNorm descarta e por que isso o torna mais barato |
| 06 | [06_kv_cache.py](06_kv_cache.py) | Médio | Loop de decode autorregressivo com vs. sem KV cache | Custo de attention por passo caindo de O(T²) para O(T), saída exatamente equivalente, conta de memória do cache |
| 07 | [07_mini_gpt_forward.py](07_mini_gpt_forward.py) | Difícil | Forward pass completo de um bloco GPT: embeddings → attention → MLP → logits | Conexão residual pre-norm, shapes corretos de ponta a ponta, weight tying, onde estão os FLOPs |
| 08 | [08_semantic_search_rag.py](08_semantic_search_rag.py) | Médio | Embed, index, retrieval por cosseno, montagem de contexto (mini-RAG) | Normalização de vetores, retrieval top-k correto, empacotamento do contexto dentro de um token budget |
| 09 | [09_text_chunking.py](09_text_chunking.py) | Fácil | Chunkers fixo, sliding-window, recursivo e por sentença | Tratamento de overlap, preservação de fronteiras, nenhum texto perdido ou duplicado nas bordas |
| 10 | [10_agent_loop.py](10_agent_loop.py) | Médio | Agent loop com tool calling usando um LLM mock | State machine limpa, resultado da tool voltando para o contexto, tratamento de erros e de máximo de iterações |
| 11 | [11_rate_limiter_and_retry.py](11_rate_limiter_and_retry.py) | Médio | Token bucket, exponential backoff com jitter | Raciocínio sobre burst vs. taxa sustentada, por que o jitter evita o thundering herd, limites de retry budget |
| 12 | [12_eval_metrics.py](12_eval_metrics.py) | Médio | Estimador não enviesado de pass@k, QA F1/EM, esqueleto de judge harness | Por que o estimador ingênuo de pass@k é enviesado, detalhes do F1 em nível de token, hooks de calibração do judge |
| 13 | [13_streaming_parser.py](13_streaming_parser.py) | Difícil | Parser de SSE, montagem incremental dos argumentos de tool call | Lidar com chunks cortados no meio de um evento ou de um JSON, disciplina de buffering, recuperação de parse parcial |
| 14 | [14_lora_adapter.py](14_lora_adapter.py) | Médio | Forward de LoRA, merge e serving batched multi-adapter | Por que B é inicializado com zero, o que alpha/r desacopla, por que o merge faz você perder o hot-swapping |
| 15 | [15_beam_search.py](15_beam_search.py) | Médio | Beam search com normalização de comprimento | Somar log-probs em vez de multiplicar probabilidades, guardar à parte os beams finalizados, o que a length penalty traz |
| 16 | [16_semantic_cache.py](16_semantic_cache.py) | Médio | Tiers de cache exato e semântico, TTL, LRU, invalidação por tag | Threshold como decisão de negócio entre precision e recall, por que as chaves de cache devem incluir o prompt e a versão do modelo |
| 17 | [17_hybrid_search_and_rerank.py](17_hybrid_search_and_rerank.py) | Médio | BM25, reciprocal rank fusion, re-ranking de diversidade com MMR | Onde o lexical vence o dense e vice-versa, por que o RRF combina ranks em vez de scores |
| 18 | [18_constrained_json_decoding.py](18_constrained_json_decoding.py) | Difícil | Mascarar tokens inválidos conforme um schema durante o decode | Mascarar antes do softmax, por que o constrained decoding desloca a distribuição, DFA vs pushdown para JSON aninhado |
| 19 | [19_speculative_decoding.py](19_speculative_decoding.py) | Difícil | Draft-and-verify com a regra exata de rejection-sampling | Por que é lossless, por que aceitar quando o argmax coincide não é, por que o ganho é de bandwidth e não de FLOPs |

## Executando

```bash
python3 12-coding-challenges/01_attention.py
# ...
# All tests passed.
```

A única dependência é o numpy (`pip install numpy`).
