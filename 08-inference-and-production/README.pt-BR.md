# ⚡ Inference, Serving & Production LLM Systems

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Este é o tópico que separa "eu chamo a API" de "eu consigo ser dono de uma feature de LLM em produção" - e ele domina as rodadas de system design para vagas de AI Engineer em labs, big techs e startups. Times com perfil de infra vão aprofundar em contas de KV cache e schedulers de batching; times com perfil de produto vão aprofundar em SLOs, cost engineering e reliability. De qualquer forma, espera-se que você raciocine quantitativamente: bytes, tokens/sec, dólares.

## Crash course

### As duas fases da inference

Toda requisição a um LLM tem duas fases radicalmente diferentes:

- **Prefill**: processa o prompt inteiro em uma única passada. Todos os tokens do prompt são computados em paralelo → grandes multiplicações matriz-matriz → **compute-bound**. O custo cresce com o tamanho do prompt (attention é quadrática, GEMMs são lineares). O prefill determina o **time-to-first-token (TTFT)**.
- **Decode**: gera os tokens de saída um de cada vez. Cada passo é uma multiplicação matriz-*vetor* que precisa transmitir **todos os pesos do modelo (mais o KV cache) da HBM da GPU** para produzir um único token → **memory-bandwidth-bound**. O decode determina o **time-per-output-token (TPOT)**.

A **arithmetic intensity** (FLOPs por byte movido) explica isso. GPUs têm um "ridge point" - aproximadamente o pico de FLOPs ÷ largura de banda de memória (~300 FLOPs/byte para uma H100 em BF16). O prefill fica muito acima dele (GEMMs totalmente paralelos); o decode com batch 1 fica em ~1-2 FLOPs/byte, então a GPU fica ociosa esperando a memória. Consequências:

- A velocidade de decode com batch 1 tem um teto rígido: `bytes dos pesos ÷ largura de banda`. Um modelo de 70B em FP16 (~140 GB) em HBM de ~3,35 TB/s ⇒ ~24 tok/s no máximo, não importa quão rápido seja o compute.
- **Fazer batching do decode é quase de graça**: a mesma leitura de pesos serve a todas as sequências do batch, então o throughput escala quase linearmente com o batch size até você atingir os limites de compute ou de memória. Quase toda otimização de serving é um esquema para manter os batches de decode grandes.

### KV cache: a coisa que devora sua GPU

Durante o decode, cada novo token faz attention sobre as keys/values de todos os tokens anteriores. Fazer cache delas evita o recompute O(n²) - mas o cache é enorme:

```
KV bytes/token = 2 (K,V) × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

Exemplo resolvido - modelo no formato do Llama-3-70B (80 layers, 8 KV heads via GQA, head_dim 128, FP16):

```
2 × 80 × 8 × 128 × 2 bytes ≈ 320 KB/token
  contexto de 8K tokens   → ~2,6 GB por sequência
  contexto de 128K tokens → ~41 GB por sequência (!)
```

Em uma GPU de 80 GB que já guarda ~40 GB de pesos INT4, o KV cache é o que limita as sequências concorrentes - **o batch size, e portanto o throughput, é limitado pela memória de KV**, não pelo compute. Daí GQA (menos KV heads), MLA (compressão low-rank do KV), quantização do KV cache (FP8/INT8 KV), alocação paginada e arquiteturas híbridas (p. ex. Jamba, Nemotron-H) que substituem a maioria das camadas de attention por camadas state-space ou linear-attention que mantêm um estado de tamanho fixo em vez de um KV cache crescente.

### Tornando o decode rápido: o playbook de serving

- **Continuous (in-flight) batching** (Orca, hoje universal): compõe o batch *a cada passo de decode*, não por requisição. Sequências terminadas saem imediatamente; as enfileiradas entram no meio do caminho. Resolve o problema de convoy do static batching (o batch inteiro espera a saída mais longa) e tipicamente multiplica o throughput várias vezes com latência similar.
- **PagedAttention** (vLLM): aloca o KV cache em blocos de tamanho fixo (p. ex. 16 tokens) mapeados por uma block table - memória virtual para o KV. Elimina a fragmentação interna/externa das alocações contíguas por sequência (que desperdiçavam a maior parte da memória de KV), permite compartilhamento de prefixo quase sem cópia e deixa você rodar batches muito maiores com segurança.
- **Chunked prefill** (Sarathi-Serve): divide prefills longos em chunks e os mistura nos batches de decode, para que um prompt de 50K tokens não trave o stream de tokens de todos os outros usuários. Troca um TTFT ligeiramente pior para esse prompt por uma inter-token latency estável para todos.
- **Prefix / prompt caching**: reutiliza o KV cache de um prefixo compartilhado (system prompt, exemplos few-shot, documentos longos, histórico da conversa) entre requisições. Self-hosted: roteamento ciente de prefixo + radix trees (SGLang) ou reuso de blocos baseado em hash (vLLM). Provedores de API cobram tokens de input em cache com um desconto acentuado: a Anthropic cobra ~0,1× o input base para leituras de cache, com um prêmio de escrita de 1,25× no TTL padrão de 5 minutos; a OpenAI faz cache automaticamente acima de ~1K tokens, com ~90% de desconto em seus modelos recentes (gerações anteriores ~50-75%); o Gemini oferece caching implícito e explícito. Os multiplicadores mudam entre gerações de modelos, então confira a página de preços atual. Estruturar prompts como *prefixo estável primeiro, conteúdo variável por último* é uma das otimizações de custo de maior ROI disponíveis.
- **Speculative decoding**: um drafter barato propõe k tokens; o modelo alvo os verifica em uma única passada paralela; os tokens aceitos são emitidos, e a primeira rejeição é substituída por uma amostra do modelo alvo. Com a regra de accept/reject correta, a distribuição de saída é *exatamente* a do modelo alvo - é uma otimização de latência sem tradeoff de qualidade. Ganha (~2-3× em ITL) quando a taxa de aceitação é alta e há compute sobrando; perde com batch sizes altos ou com drafters mal casados. Os drafters vão de um modelo pequeno com o mesmo tokenizer, a heads de self-drafting (Medusa, EAGLE), a heads de multi-token-prediction treinadas dentro do próprio modelo (DeepSeek-V3), até drafting de n-gram/prompt-lookup gratuito para workloads de extração e edição.

### Quantization

- **Weights-only** (W4A16, W8A16): reduz os pesos para 4/8 bits, computa em FP16. Como o decode é limitado pela largura de banda dos pesos, pesos de 4 bits ≈ até ~4× mais rápido no decode com batch 1 *e* 4× menos memória. GPTQ (arredondamento baseado em Hessian) e AWQ (escalonamento activation-aware de canais salientes) são os métodos de PTQ de 4 bits padrão; os k-quants do GGUF são o equivalente no ecossistema llama.cpp/Ollama.
- **Weights + activations** (W8A8 INT8/FP8): também acelera os próprios *matmuls* via tensor cores INT8/FP8 - é o que ajuda em batch alto/prefill (compute-bound). FP8 é o cavalo de batalha padrão em Hopper e Blackwell. A Blackwell adiciona tensor cores FP4 nativos (NVFP4/MXFP4), e lançamentos open-weight como o gpt-oss já distribuem seus pesos MoE em MXFP4.
- **Custo típico de qualidade**: 8 bits ≈ desprezível; bom 4 bits ≈ pequeno mas mensurável (avalie na *sua* tarefa, não apenas em perplexity); abaixo de 4 bits fica difícil. A **quantização do KV cache** (FP8 KV) é uma alavanca separada que compra batch size diretamente.

### Métricas, SLOs e a curva throughput - latency

| Métrica | Significado | Determinada por |
|---|---|---|
| **TTFT** | time to first token | queueing + prefill (tamanho do prompt) |
| **TPOT / ITL** | tempo por token de saída após o primeiro | largura de banda de memória, contenção no batch |
| **tokens/sec** | throughput por requisição ou agregado | batch size, hardware |
| **Goodput** | throughput *que atende aos SLOs* | qualidade do scheduler, carga |
| **P50 vs P99** | mediana vs cauda | cauda = picos de fila, prompts longos, preemptions |

Latência total ≈ `TTFT + TPOT × output_tokens`. Throughput e latency trocam entre si ao longo de uma curva: aumentar o batch size eleva tokens/sec, mas infla o TPOT e o atraso de fila; após a saturação, o throughput estabiliza enquanto a latency explode. Você escolhe um ponto de operação a partir do seu SLO (p. ex. "P99 TTFT < 800 ms, P99 ITL < 60 ms para chat interativo") e compra capacidade suficiente para ficar à esquerda do joelho da curva. Reporte **goodput**, não throughput bruto - 10K tok/s com 30 s de TTFT não vale nada para chat.

O **Streaming** é a metade de UX da latency: SSE (`text/event-stream`) entrega deltas conforme são gerados, então a latency percebida ≈ TTFT (centenas de ms) em vez da conclusão completa (dezenas de segundos). Tool calls e JSON chegam como fragmentos parciais - faça buffer por bloco, ou use parsing incremental/parcial de JSON para UI progressiva.

### Contas de memória da GPU e one-liners de paralelismo

```
Memória da GPU ≈ pesos (params × bytes/param)
               + KV cache (bytes por token × total de tokens em cache)
               + activations e buffers (~alguns GB, depende de batch/seq)
```

- Modelo de 8B: FP16 ≈ 16 GB, INT4 ≈ ~4,5 GB (roda em um laptop).
- Modelo de 70B: FP16 ≈ 140 GB (precisa de 2× GPUs de 80 GB, ou uma GPU classe Blackwell de ~180 GB), INT4 ≈ ~40 GB (cabe em uma GPU de 80 GB com espaço para o KV).
- **Tensor parallelism (TP)**: divide as matrizes de cada layer entre GPUs, com all-reduce a cada layer - reduz a latency por token, exige interconexão classe NVLink, padrão dentro de um node.
- **Pipeline parallelism (PP)**: divide por faixas de layers entre GPUs/nodes - tolera interconexão mais lenta, adiciona latency e pipeline bubbles; usado quando o modelo não cabe em um node.
- **Expert parallelism (EP)**: para modelos mixture-of-experts, espalha os experts entre as GPUs e despacha tokens para eles com um all-to-all por layer MoE. Os parâmetros totais definem a conta de memória, os parâmetros ativos definem o compute e a largura de banda por token; o desbalanceamento de roteamento e a latency do all-to-all viram os novos gargalos.

### Escalando além de uma réplica

- **Prefill/decode disaggregation** (DistServe, Splitwise; produtizada em NVIDIA Dynamo, llm-d, vLLM e SGLang): roda prefill e decode em pools de GPU separados e envia o KV cache entre eles por NVLink ou RDMA. Cada pool recebe seu próprio paralelismo e tamanho, e prefills longos deixam de perturbar o ITL dos demais usuários. Compensa em escala com prompts longos; para frotas pequenas, o custo de transferência do KV e o custo operacional geralmente superam o ganho.
- **KV cache tiers**: faz offload de blocos de KV frios para DRAM da CPU, NVMe local ou um store compartilhado (p. ex. LMCache) para que os prefixos sobrevivam à eviction. Recarregar vence recomputar quando o prefixo é longo e o caminho de transferência é rápido.
- **Cache-aware routing**: round-robin entre réplicas dispersa os prefixos compartilhados e destrói a hit rate. Roteie por hash de prefixo ou pelo estado de KV reportado, com um limite de carga para que prefixos quentes não derretam uma única réplica.
- **Multi-LoRA serving**: um modelo base na memória, muitos adapters pequenos trocados por requisição dentro do mesmo batch (kernels estilo S-LoRA/Punica, embutidos no vLLM e no SGLang). Centenas de fine-tunes de clientes podem compartilhar um punhado de GPUs.
- **Constrained decoding** para structured output: o engine compila um JSON schema ou gramática em uma máscara de tokens por passo (XGrammar, llguidance, Outlines). Isso garante validade sintática, não correção semântica, e os engines modernos sobrepõem o cálculo da máscara ao trabalho da GPU, então o overhead é pequeno.

### Escolhendo um serving stack

| Stack | Sweet spot |
|---|---|
| **vLLM** | escolha open-source padrão; PagedAttention, continuous batching, enorme cobertura de modelos/hardware |
| **SGLang** | workloads com muito prefixo compartilhado e agentic (reuso de prefixo com RadixAttention), structured output rápido |
| **TensorRT-LLM** | performance máxima em NVIDIA quando você vai investir em tuning; runtime baseado em PyTorch por padrão desde a 1.0, FP8/FP4 de primeira classe |
| **TGI** | em modo de manutenção desde o fim de 2025, com a Hugging Face apontando usuários para vLLM e SGLang: item de migração, não escolha para novos deployments |
| **llama.cpp / Ollama / MLX** | CPU/edge/laptop e Apple Silicon; quants GGUF; serving local amigável para dev |
| **NVIDIA Dynamo / llm-d** | não são engines, mas a camada de cluster acima deles: prefill/decode disaggregated, roteamento ciente de KV, offload de KV em toda a frota |

### Padrões de produção (a outra metade da entrevista)

- **Self-host vs API**: decida com base em utilização (volume alto e estável favorece self-host; volume em picos/baixo favorece API), privacidade/residência de dados, necessidades de modelo (qualidade de fronteira → API; fine-tuned/custom → self-host), time (infra de inference é um compromisso de oncall) e unit economics (US$/hora de GPU ÷ tokens/hora alcançáveis vs preço por token da API - a eficiência de batching domina essa conta).
- **Reliability**: retries com **exponential backoff + full jitter** em 429/5xx (respeite `Retry-After`), limites de concorrência no client, **circuit breakers** por provedor, timeouts em camadas (connect, TTFT, inatividade entre chunks, total), idempotency keys para pipelines com efeitos colaterais, cadeias de fallback de provedor/modelo.
- **Cost engineering**: tokens de saída custam ~4-8× tokens de input nas listas de preços atuais de modelos de fronteira, e reasoning tokens são cobrados como saída - controle `max_tokens` e o reasoning effort. Alavancas, em ordem aproximada de ROI: prompt/prefix caching, model tiering/cascades (modelo barato + escalonamento), truncamento/sumarização de contexto, batch APIs (~50% de desconto para trabalho assíncrono), semantic caching. Atribua custo por requisição/feature/usuário desde o primeiro dia.
- **Capacity & autoscaling**: cold starts de GPU levam minutos (provisionamento do node + pull da imagem + carga dos pesos), então faça autoscale por profundidade da fila/utilização de KV com warm pools e escalonamento preditivo - não por CPU%.
- **Observability**: faça dashboards de percentis de TTFT/ITL, profundidade da fila, batch size, utilização do KV cache, preemptions, tokens in/out, taxas de erro/429, custo por requisição. A "utilização" de GPU sozinha engana - um decode memory-bound pode mostrar 100% enquanto faz pouco compute.
- **Graceful degradation**: sob sobrecarga, descarte ou enfileire tráfego de baixa prioridade, faça fallback para modelos menores, trunque o histórico, limite o tamanho da saída, sirva respostas em cache - degrade a qualidade antes da disponibilidade.

## Perguntas de entrevista

Veja [questions.pt-BR.md](questions.pt-BR.md) - 56 perguntas entre básicas, intermediárias e avançadas.

## Red flags que os entrevistadores observam

- Não consegue explicar *por que* prefill e decode têm gargalos diferentes - trata "GPUs são rápidas" como a história toda, ou não consegue conectar largura de banda de memória a tokens/sec.
- Nenhuma noção numérica de KV cache: não consegue estimar bytes/token nem explicar por que contextos longos e batches grandes disputam a mesma memória.
- Resolve com um "vamos usar vLLM" em system design sem conseguir dizer o que PagedAttention ou continuous batching realmente fazem.
- Acredita que speculative decoding muda a qualidade da saída, ou que quantization é de graça - sem instinto de avaliar na tarefa real.
- Desenha SLOs apenas em torno da latency média; nunca menciona P99, TTFT vs ITL ou goodput; não consegue esboçar o tradeoff throughput - latency.
- Cegueira de custo: não sabe que tokens de saída custam mais que os de input, nunca menciona prompt caching, batch APIs ou model tiering quando perguntado como reduzir uma fatura.
- Reliability ingênua: retries sem jitter ou sem budget (amplificando outages), um único timeout global para uma chamada com streaming, nenhum plano de fallback para incidentes do provedor.
- Escolhe self-hosting "por custo" sem nenhuma conta de utilização, ou uma API "por simplicidade" onde restrições de privacidade/latency claramente a proíbem.

## Leituras complementares

- [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102) - o tratamento canônico dos tradeoffs de prefill/decode, batching e paralelismo.
- [Transformer Inference Arithmetic (kipply)](https://kipp.ly/transformer-inference-arithmetic/) - o guia clássico de estimativas de KV cache, largura de banda e latency.
- [Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) - o paper do vLLM.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) - de onde vem o continuous batching.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) - speculative decoding com a prova de corretude.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al., 2024)](https://arxiv.org/abs/2403.02310) - chunked prefill e scheduling stall-free.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670) - o argumento por pools separados de prefill e decode, e goodput como métrica-alvo.
- [vLLM documentation](https://docs.vllm.ai/) - a referência prática para um serving stack moderno.
- [Exponential Backoff and Jitter (AWS Architecture Blog)](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) - a referência padrão para design de retries.
