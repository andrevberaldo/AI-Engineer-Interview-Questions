# ⚡ Inference, Serving & Production LLM Systems

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Este é o tópico que separa quem "chama a API" de quem "consegue ser dono de uma feature de LLM em produção". Ele domina as rodadas de system design para vagas de AI Engineer em labs, big techs e startups. Times com perfil de infra vão fundo em contas de KV cache e schedulers de batching. Times com perfil de produto vão fundo em SLOs, cost engineering e reliability. Em qualquer caso, esperam que você raciocine com números: bytes, tokens/sec, dólares.

## Crash course

### As duas fases da inference

Toda requisição a um LLM passa por duas fases radicalmente diferentes:

- **Prefill**: processa o prompt inteiro de uma só vez. Todos os tokens do prompt são calculados em paralelo → grandes multiplicações matriz-matriz → **compute-bound**. O custo cresce com o tamanho do prompt (attention é quadrática, GEMMs são lineares). O prefill determina o **time-to-first-token (TTFT)**.
- **Decode**: gera os tokens de saída um de cada vez. Cada passo é uma multiplicação matriz-*vetor* que precisa ler **todos os pesos do modelo (mais o KV cache) da HBM da GPU** para produzir um único token → **memory-bandwidth-bound**. O decode determina o **time-per-output-token (TPOT)**.

A **arithmetic intensity** (FLOPs por byte movido) explica isso. As GPUs têm um "ridge point", que é mais ou menos o pico de FLOPs ÷ largura de banda de memória (~300 FLOPs/byte para uma H100 em BF16). O prefill fica muito acima dele (GEMMs totalmente paralelos). O decode com batch 1 fica em ~1-2 FLOPs/byte, então a GPU passa o tempo ociosa, esperando a memória. Consequências:

- A velocidade de decode com batch 1 tem um teto rígido: `bytes dos pesos ÷ largura de banda`. Um modelo de 70B em FP16 (~140 GB) com HBM de ~3,35 TB/s chega a ~24 tok/s no máximo, por mais rápido que seja o compute.
- **Fazer batching do decode é quase de graça**: a mesma leitura de pesos atende todas as sequências do batch, então o throughput escala quase linearmente com o batch size até você bater no limite de compute ou de memória. Quase toda otimização de serving é um jeito de manter os batches de decode grandes.

### KV cache: o que devora a sua GPU

Durante o decode, cada novo token faz attention sobre as keys/values de todos os tokens anteriores. Guardá-las em cache evita o recompute O(n²), mas o cache é enorme:

```
KV bytes/token = 2 (K,V) × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

Exemplo resolvido, com um modelo no formato do Llama-3-70B (80 layers, 8 KV heads via GQA, head_dim 128, FP16):

```
2 × 80 × 8 × 128 × 2 bytes ≈ 320 KB/token
  contexto de 8K tokens   → ~2,6 GB por sequência
  contexto de 128K tokens → ~41 GB por sequência (!)
```

Numa GPU de 80 GB que já guarda ~40 GB de pesos INT4, é o KV cache que limita o número de sequências concorrentes: **o batch size, e portanto o throughput, é limitado pela memória de KV**, não pelo compute. Daí vêm GQA (menos KV heads), MLA (compressão low-rank do KV), quantização do KV cache (FP8/INT8 KV), alocação paginada e arquiteturas híbridas (p. ex. Jamba, Nemotron-H), que trocam a maioria das camadas de attention por camadas state-space ou linear-attention, com um estado de tamanho fixo no lugar de um KV cache que só cresce.

### Deixando o decode rápido: o playbook de serving

- **Continuous (in-flight) batching** (Orca, hoje universal): monta o batch *a cada passo de decode*, e não por requisição. As sequências que terminam saem na hora, e as que estavam na fila entram no meio do caminho. Isso resolve o problema de convoy do static batching (o batch inteiro espera pela saída mais longa) e, em geral, multiplica o throughput várias vezes com latency parecida.
- **PagedAttention** (vLLM): aloca o KV cache em blocos de tamanho fixo (p. ex. 16 tokens), mapeados por uma block table, como se fosse memória virtual para o KV. Elimina a fragmentação interna e externa das alocações contíguas por sequência (que desperdiçavam a maior parte da memória de KV), permite compartilhar prefixos quase sem cópia e deixa você rodar batches bem maiores com segurança.
- **Chunked prefill** (Sarathi-Serve): divide prefills longos em chunks e os mistura nos batches de decode, para que um prompt de 50K tokens não trave o stream de tokens de todos os outros usuários. Você aceita um TTFT um pouco pior para esse prompt em troca de uma inter-token latency estável para todo mundo.
- **Prefix / prompt caching**: reaproveita o KV cache de um prefixo compartilhado (system prompt, exemplos few-shot, documentos longos, histórico da conversa) entre requisições. Em self-hosting: roteamento ciente de prefixo + radix trees (SGLang) ou reuso de blocos por hash (vLLM). Os provedores de API cobram bem menos pelos tokens de input em cache: a Anthropic cobra ~0,1× o input base nas leituras de cache, com um acréscimo de 1,25× na escrita no TTL padrão de 5 minutos; a OpenAI faz cache automático acima de ~1K tokens, com ~90% de desconto nos modelos recentes (gerações anteriores ~50-75%); o Gemini oferece caching implícito e explícito. Os multiplicadores mudam de uma geração de modelos para outra, então confira a página de preços atual. Estruturar o prompt com *o prefixo estável primeiro e o conteúdo variável por último* é uma das otimizações de custo com maior ROI.
- **Speculative decoding**: um drafter barato propõe k tokens, e o modelo alvo os verifica numa única passada paralela. Os tokens aceitos são emitidos, e a primeira rejeição é substituída por uma amostra do modelo alvo. Com a regra de accept/reject correta, a distribuição de saída é *exatamente* a do modelo alvo: é uma otimização de latency, sem trade-off de qualidade. Ganha (~2-3× em ITL) quando a taxa de aceitação é alta e há compute sobrando, e perde com batch sizes altos ou com drafters mal ajustados ao modelo. Os drafters vão de um modelo pequeno com o mesmo tokenizer, passando por heads de self-drafting (Medusa, EAGLE) e por heads de multi-token-prediction treinadas dentro do próprio modelo (DeepSeek-V3), até o drafting de n-gram/prompt-lookup, que sai de graça e serve bem para workloads de extração e edição.

### Quantization

- **Weights-only** (W4A16, W8A16): reduz os pesos para 4/8 bits e calcula em FP16. Como o decode é limitado pela largura de banda dos pesos, pesos de 4 bits dão até ~4× de velocidade no decode com batch 1 *e* 4× menos memória. GPTQ (arredondamento baseado em Hessian) e AWQ (escalonamento activation-aware dos canais mais salientes) são os métodos padrão de PTQ de 4 bits; os k-quants do GGUF são o equivalente no ecossistema llama.cpp/Ollama.
- **Weights + activations** (W8A8 INT8/FP8): acelera também os próprios *matmuls*, via tensor cores INT8/FP8. É o que ajuda com batch alto e no prefill (compute-bound). FP8 é o carro-chefe em Hopper e Blackwell. A Blackwell traz tensor cores FP4 nativos (NVFP4/MXFP4), e lançamentos open-weight como o gpt-oss já distribuem seus pesos MoE em MXFP4.
- **Custo típico de qualidade**: 8 bits é desprezível; um bom 4 bits custa pouco, mas dá para medir (avalie na *sua* tarefa, não só em perplexity); abaixo de 4 bits a coisa fica complicada. A **quantização do KV cache** (FP8 KV) é uma alavanca à parte, que compra batch size diretamente.

### Métricas, SLOs e a curva throughput × latency

| Métrica | Significado | Determinada por |
|---|---|---|
| **TTFT** | time to first token | queueing + prefill (tamanho do prompt) |
| **TPOT / ITL** | tempo por token de saída depois do primeiro | largura de banda de memória, contenção no batch |
| **tokens/sec** | throughput por requisição ou agregado | batch size, hardware |
| **Goodput** | throughput *que cumpre os SLOs* | qualidade do scheduler, carga |
| **P50 vs P99** | mediana vs cauda | cauda = picos de fila, prompts longos, preemptions |

Latency total ≈ `TTFT + TPOT × output_tokens`. Throughput e latency se trocam ao longo de uma curva: aumentar o batch size eleva os tokens/sec, mas infla o TPOT e o tempo de fila. Depois da saturação, o throughput se estabiliza e a latency explode. Você escolhe um ponto de operação a partir do seu SLO (p. ex. "P99 TTFT < 800 ms, P99 ITL < 60 ms para chat interativo") e compra capacidade suficiente para ficar à esquerda do joelho da curva. Reporte o **goodput**, não o throughput bruto: 10K tok/s com 30 s de TTFT não servem para nada num chat.

O **streaming** é a metade da latency que toca a UX: SSE (`text/event-stream`) entrega deltas conforme são gerados, então a latency percebida fica perto do TTFT (centenas de ms) em vez do tempo da resposta completa (dezenas de segundos). Tool calls e JSON chegam em fragmentos parciais: faça buffer por bloco ou use parsing incremental/parcial de JSON para montar uma UI progressiva.

### Contas de memória da GPU e paralelismo em poucas linhas

```
Memória da GPU ≈ pesos (params × bytes/param)
               + KV cache (bytes por token × total de tokens em cache)
               + activations e buffers (~alguns GB, depende de batch/seq)
```

- Modelo de 8B: FP16 ≈ 16 GB, INT4 ≈ ~4,5 GB (roda num laptop).
- Modelo de 70B: FP16 ≈ 140 GB (precisa de 2× GPUs de 80 GB, ou de uma GPU classe Blackwell de ~180 GB), INT4 ≈ ~40 GB (cabe numa GPU de 80 GB, com espaço para o KV).
- **Tensor parallelism (TP)**: divide as matrizes de cada layer entre as GPUs, com um all-reduce por layer. Reduz a latency por token e exige interconexão classe NVLink; é o padrão dentro de um node.
- **Pipeline parallelism (PP)**: divide por faixas de layers entre GPUs/nodes. Tolera uma interconexão mais lenta, mas adiciona latency e pipeline bubbles; é usado quando o modelo não cabe em um node.
- **Expert parallelism (EP)**: em modelos mixture-of-experts, espalha os experts pelas GPUs e despacha os tokens até eles com um all-to-all por layer MoE. Os parâmetros totais definem a conta de memória, e os parâmetros ativos definem o compute e a largura de banda por token. Os novos gargalos passam a ser o desbalanceamento do roteamento e a latency do all-to-all.

### Escalando além de uma réplica

- **Prefill/decode disaggregation** (DistServe, Splitwise; virou produto em NVIDIA Dynamo, llm-d, vLLM e SGLang): roda prefill e decode em pools de GPU separados e envia o KV cache de um para o outro por NVLink ou RDMA. Cada pool ganha seu próprio paralelismo e tamanho, e prefills longos deixam de atrapalhar o ITL dos outros usuários. Compensa em escala e com prompts longos; em frotas pequenas, o custo de transferir o KV e o custo operacional costumam superar o ganho.
- **KV cache tiers**: faz offload dos blocos de KV frios para a DRAM da CPU, para NVMe local ou para um store compartilhado (p. ex. LMCache), de modo que os prefixos sobrevivam à eviction. Recarregar vence recomputar quando o prefixo é longo e o caminho de transferência é rápido.
- **Cache-aware routing**: round-robin entre réplicas espalha os prefixos compartilhados e destrói a hit rate. Roteie por hash de prefixo ou pelo estado de KV reportado, com um limite de carga para que prefixos muito quentes não derretam uma única réplica.
- **Multi-LoRA serving**: um modelo base na memória e vários adapters pequenos, trocados por requisição dentro do mesmo batch (kernels no estilo S-LoRA/Punica, já embutidos no vLLM e no SGLang). Centenas de fine-tunes de clientes podem dividir um punhado de GPUs.
- **Constrained decoding** para structured output: o engine compila um JSON schema ou uma gramática numa máscara de tokens por passo (XGrammar, llguidance, Outlines). Isso garante validade sintática, não correção semântica. Os engines modernos sobrepõem o cálculo da máscara ao trabalho da GPU, então o overhead é pequeno.

### Escolhendo um serving stack

| Stack | Sweet spot |
|---|---|
| **vLLM** | a escolha open-source padrão; PagedAttention, continuous batching, enorme cobertura de modelos/hardware |
| **SGLang** | workloads com muito prefixo compartilhado e agentic (reuso de prefixo com RadixAttention), structured output rápido |
| **TensorRT-LLM** | performance máxima em NVIDIA, se você topar investir em tuning; runtime baseado em PyTorch por padrão desde a 1.0, FP8/FP4 de primeira classe |
| **TGI** | em modo de manutenção desde o fim de 2025, com a Hugging Face apontando os usuários para vLLM e SGLang: é item de migração, não escolha para novos deployments |
| **llama.cpp / Ollama / MLX** | CPU/edge/laptop e Apple Silicon; quants GGUF; serving local amigável para quem desenvolve |
| **NVIDIA Dynamo / llm-d** | não são engines, e sim a camada de cluster acima deles: prefill/decode disaggregated, roteamento ciente de KV, offload de KV por toda a frota |

### Padrões de produção (a outra metade da entrevista)

- **Self-host vs API**: decida com base em utilização (volume alto e estável favorece self-host; volume baixo ou em picos favorece API), privacidade/residência de dados, necessidades de modelo (qualidade de fronteira → API; fine-tuned/custom → self-host), time (infra de inference é compromisso de oncall) e unit economics (US$/hora de GPU ÷ tokens/hora que você consegue alcançar vs preço por token da API; a eficiência de batching domina essa conta).
- **Reliability**: retries com **exponential backoff + full jitter** em 429/5xx (respeite o `Retry-After`), limites de concorrência no client, **circuit breakers** por provedor, timeouts em camadas (connect, TTFT, inatividade entre chunks, total), idempotency keys para pipelines com efeitos colaterais e cadeias de fallback de provedor/modelo.
- **Cost engineering**: nas listas de preços atuais dos modelos de fronteira, tokens de saída custam ~4-8× os de input, e reasoning tokens são cobrados como saída; por isso, controle `max_tokens` e o reasoning effort. As alavancas, em ordem aproximada de ROI: prompt/prefix caching, model tiering/cascades (modelo barato + escalonamento), truncamento/sumarização de contexto, batch APIs (~50% de desconto para trabalho assíncrono) e semantic caching. Atribua custo por requisição/feature/usuário desde o primeiro dia.
- **Capacity & autoscaling**: o cold start de uma GPU leva minutos (provisionar o node + pull da imagem + carregar os pesos). Por isso, faça autoscale por profundidade da fila/utilização de KV, com warm pools e escalonamento preditivo, e não por CPU%.
- **Observability**: monte dashboards de percentis de TTFT/ITL, profundidade da fila, batch size, utilização do KV cache, preemptions, tokens in/out, taxas de erro/429 e custo por requisição. A "utilização" da GPU, sozinha, engana: um decode memory-bound pode mostrar 100% fazendo pouquíssimo compute.
- **Graceful degradation**: sob sobrecarga, descarte ou enfileire o tráfego de baixa prioridade, faça fallback para modelos menores, trunque o histórico, limite o tamanho da saída e sirva respostas em cache. Degrade a qualidade antes da disponibilidade.

## Perguntas de entrevista

Veja [questions.pt-BR.md](questions.pt-BR.md): 56 perguntas entre básicas, intermediárias e avançadas.

## Red flags que os entrevistadores observam

- Não explica *por que* prefill e decode têm gargalos diferentes: acha que "GPUs são rápidas" resolve a história toda, ou não conecta largura de banda de memória a tokens/sec.
- Não tem noção numérica de KV cache: não estima bytes/token nem explica por que contextos longos e batches grandes disputam a mesma memória.
- Resolve tudo com um "vamos usar vLLM" em system design, sem saber dizer o que PagedAttention ou continuous batching realmente fazem.
- Acredita que speculative decoding muda a qualidade da saída, ou que quantization é de graça, sem o instinto de avaliar na tarefa real.
- Desenha SLOs só em torno da latency média; nunca menciona P99, TTFT vs ITL ou goodput; não consegue esboçar o trade-off throughput × latency.
- Cegueira de custo: não sabe que tokens de saída custam mais que os de input e, quando pedem para reduzir uma fatura, nem cita prompt caching, batch APIs ou model tiering.
- Reliability ingênua: retries sem jitter ou sem budget (que amplificam outages), um único timeout global para uma chamada com streaming, nenhum plano de fallback para incidentes do provedor.
- Escolhe self-hosting "por custo" sem nenhuma conta de utilização, ou uma API "pela simplicidade" onde as restrições de privacidade/latency claramente a proíbem.

## Leituras complementares

- [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102) - o tratamento canônico dos trade-offs de prefill/decode, batching e paralelismo.
- [Transformer Inference Arithmetic (kipply)](https://kipp.ly/transformer-inference-arithmetic/) - o guia clássico de estimativas de KV cache, largura de banda e latency.
- [Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) - o paper do vLLM.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) - de onde vem o continuous batching.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) - speculative decoding com a prova de corretude.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al., 2024)](https://arxiv.org/abs/2403.02310) - chunked prefill e scheduling stall-free.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Zhong et al., 2024)](https://arxiv.org/abs/2401.09670) - o argumento a favor de pools separados de prefill e decode, e do goodput como métrica-alvo.
- [vLLM documentation](https://docs.vllm.ai/) - a referência prática para um serving stack moderno.
- [Exponential Backoff and Jitter (AWS Architecture Blog)](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) - a referência padrão para design de retries.
