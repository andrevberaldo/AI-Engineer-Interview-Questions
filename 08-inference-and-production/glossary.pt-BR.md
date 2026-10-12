# 📚 Glossário de estudo — Inference & Production

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Fundamentos: como uma requisição vira tokens](#1-fundamentos-como-uma-requisição-vira-tokens)
- [2. KV cache e memória da GPU](#2-kv-cache-e-memória-da-gpu)
- [3. Batching, scheduling e kernels de attention](#3-batching-scheduling-e-kernels-de-attention)
- [4. Speculative decoding](#4-speculative-decoding)
- [5. Quantização](#5-quantização)
- [6. Métricas, SLOs e latência](#6-métricas-slos-e-latência)
- [7. Paralelismo e hardware](#7-paralelismo-e-hardware)
- [8. Escalando além de uma réplica](#8-escalando-além-de-uma-réplica)
- [9. Stacks de serving](#9-stacks-de-serving)
- [10. APIs, streaming e confiabilidade](#10-apis-streaming-e-confiabilidade)
- [11. Custo e decisões de produção](#11-custo-e-decisões-de-produção)

## 1. Fundamentos: como uma requisição vira tokens

### Prefill

A primeira fase de uma requisição: o modelo lê o prompt inteiro de uma vez, em paralelo, e produz o primeiro token. Gasta muito cálculo e define o TTFT.

> 💡 **Pense assim:** Ler a pergunta toda antes de começar a responder.

*Aparece em:* Q1, Q18, Q47, Q53, Q55

### Decode

A segunda fase: gerar a resposta um token por vez. Cada passo precisa ler todos os pesos do modelo da memória da GPU para produzir um único token, por isso é limitado pela memória, não pelo cálculo.

> 💡 **Pense assim:** Escrever a resposta palavra por palavra, relendo o caderno inteiro a cada palavra.

*Aparece em:* Q1, Q16, Q18, Q44, Q47

### HBM (memória da GPU)

A memória de alta largura de banda da GPU, onde ficam os pesos do modelo e o KV cache. Sua capacidade (GB) limita o que cabe, e sua velocidade (TB/s) limita a velocidade do decode.

*Aparece em:* Q1, Q12, Q32, Q33, Q56

### Memory-bound vs compute-bound

Compute-bound: o gargalo é a quantidade de contas (a GPU está calculando sem parar). Memory-bound: o gargalo é buscar dados na memória (a GPU espera chegar o dado). Prefill costuma ser compute-bound; decode, memory-bound.

*Aparece em:* Q1, Q7, Q16, Q20, Q53

### Arithmetic intensity

Quantas operações (FLOPs) você faz para cada byte lido da memória. Se for baixa, a GPU fica esperando dados; se for alta, ela fica ocupada calculando.

> 💡 **Pense assim:** Um cozinheiro que busca um ingrediente na despensa e só mexe a panela uma vez: passa o tempo todo andando.

*Aparece em:* Q1, Q16

### FLOPs

Número de operações de ponto flutuante: a medida do custo de cálculo. A GPU tem um teto de FLOPs por segundo.

*Aparece em:* Q16, Q32, Q33, Q50, Q55

### Roofline model e ridge point

Modelo que diz que a performance é o menor entre dois tetos: o de cálculo (peak FLOPs) e o de memória (intensidade × largura de banda). O ridge point é a intensidade em que um teto cruza o outro: abaixo dele você está memory-bound, acima, compute-bound.

*Aparece em:* Q16, Q22, Q39

### Logits, argmax e sampling

Logits são os scores brutos que o modelo dá a cada token possível. Argmax escolhe sempre o de maior score (greedy). Sampling sorteia entre eles conforme as probabilidades.

*Aparece em:* Q11, Q17, Q19, Q44, Q46

## 2. KV cache e memória da GPU

### KV cache

Memória que guarda as Keys e Values de todos os tokens já processados, para não recalculá-los a cada novo token. Acelera muito o decode, mas é grande e cresce com contexto × número de sequências. Na prática é ele, não o cálculo, que limita quantas requisições cabem juntas na GPU.

> 💡 **Pense assim:** Anotações de reunião: em vez de reouvir a gravação inteira a cada pergunta, você consulta as anotações. O preço é o espaço que elas ocupam.

*Aparece em:* Q2, Q5, Q17, Q37, Q55

### Fórmula de bytes por token do KV cache

bytes/token = 2 (K e V) × camadas × KV heads × head_dim × bytes por elemento. Para um modelo de 70B com GQA em FP16 dá cerca de 320 KB por token, ou seja, ~41 GB para um contexto de 128K em uma única sequência.

*Aparece em:* Q2, Q21, Q37, Q47, Q55

### GQA, MQA e MLA

Técnicas de arquitetura que encolhem o KV cache. GQA e MQA fazem várias attention heads compartilharem os mesmos K/V (menos KV heads). MLA guarda uma versão comprimida de K e V. Todas reduzem os bytes por token sem mudar muito o tamanho do modelo.

*Aparece em:* Q2, Q37, Q47, Q50, Q55

### Memory budget (orçamento de memória)

A conta de quanto cabe na GPU: pesos (parâmetros × bytes por parâmetro) + KV cache + ativações e buffers + overhead do runtime. O que sobra depois dos pesos é o KV pool, e ele define o tamanho máximo do batch.

*Aparece em:* Q5, Q24, Q37, Q50, Q54

### Bytes por parâmetro (FP32, FP16, INT8, INT4)

O tamanho de cada peso depende do formato: FP32 usa 4 bytes, FP16/BF16 usa 2, INT8/FP8 usa 1, INT4 usa cerca de 0,5 (um pouco mais, por causa das escalas). Um modelo de 70B em FP16 ocupa ~140 GB; em INT4, ~40 GB.

*Aparece em:* Q5, Q37

### Fragmentation (fragmentação)

Desperdício de memória por alocar em blocos grandes e contíguos: a sequência reservou espaço que não usa (interna) ou sobraram buracos inutilizáveis entre alocações (externa). O PagedAttention foi criado para acabar com isso.

*Aparece em:* Q2, Q12, Q17, Q30, Q48

### KV cache offloading e tiers

Mover blocos de KV que não estão em uso da GPU para a RAM do servidor (CPU), para um SSD NVMe ou para um armazenamento compartilhado, e trazê-los de volta quando necessário. Vale a pena quando carregar o KV é mais rápido do que recalcular o prefill.

> 💡 **Pense assim:** Guardar roupa de inverno no armário do fundo: ocupa a gaveta só quando precisa.

*Aparece em:* Q32, Q49, Q54, Q55, Q56

### Preemption

Quando a memória de KV acaba, o engine tira uma sequência em andamento do batch (descartando ou movendo seu KV) para dar lugar a outras, e a retoma depois. Causa picos de latência, por isso a taxa de preemption é uma métrica importante.

*Aparece em:* Q37, Q41, Q51, Q53, Q55

### Prompt / prefix caching

Reaproveitar o KV cache de um início de prompt que se repete entre requisições (system prompt, ferramentas, histórico). O prefill dessa parte é pulado, o que reduz TTFT e custo. Qualquer byte alterado no começo invalida tudo que vem depois, então a regra é: parte estável primeiro, parte variável por último.

> 💡 **Pense assim:** Marcar a página do livro onde você parou: na volta você só lê o que é novo.

*Aparece em:* Q6, Q9, Q27, Q53, Q54

### Radix tree / RadixAttention

Estrutura em árvore que o SGLang usa para lembrar quais prefixos já estão no cache e reaproveitá-los, mesmo que só parte do prompt coincida. Também serve para um roteador saber qual réplica tem o maior prefixo em comum.

*Aparece em:* Q9, Q17, Q23, Q49

## 3. Batching, scheduling e kernels de attention

### Batching

Processar várias requisições juntas no mesmo passo. No decode isso é quase de graça: a mesma leitura dos pesos serve a todas as sequências do batch, então a vazão cresce quase linearmente com o tamanho do batch.

> 💡 **Pense assim:** Um ônibus: custa quase o mesmo rodar com 50 passageiros ou com 1.

*Aparece em:* Q8, Q16, Q26, Q29, Q35

### Static vs continuous batching

Static: junta N requisições e só devolve quando todas terminam (as curtas ficam esperando a mais longa). Continuous (in-flight): o batch é remontado a cada passo de decode, as sequências terminadas saem na hora e novas entram. Virou padrão por aumentar muito a vazão sem piorar a latência.

*Aparece em:* Q8, Q23, Q24, Q29, Q51

### Convoy effect / head-of-line blocking

Quando um item lento segura a fila inteira: no batching estático, uma resposta longa prende as curtas; num prefill longo, ele trava o decode de todo mundo.

> 💡 **Pense assim:** Fila de supermercado em que o primeiro cliente está pagando em moedas.

*Aparece em:* Q8, Q15, Q31

### Scheduler

O componente do engine que decide, a cada passo, quais requisições entram no batch, quantas ficam esperando e quando preemptar alguma. A qualidade do scheduler afeta diretamente latência e goodput.

*Aparece em:* Q8, Q22, Q35, Q51, Q53

### PagedAttention

Guarda o KV cache em blocos pequenos de tamanho fixo (por exemplo, 16 tokens), espalhados pela memória e acessados por uma tabela de blocos, como a memória virtual de um sistema operacional. Elimina a fragmentação, permite batches bem maiores e torna o compartilhamento de prefixos quase gratuito. É uma técnica de gestão de memória, não de cálculo.

> 💡 **Pense assim:** Livro com páginas soltas e um índice, em vez de um bloco de papel contíguo reservado para o tamanho máximo possível.

*Aparece em:* Q2, Q12, Q17, Q23, Q37

### FlashAttention

Implementação da attention que calcula exatamente o mesmo resultado, só que muito mais rápido, porque processa em pedaços (tiles) que cabem na memória rápida do chip (SRAM) e nunca monta a matriz n×n inteira na HBM. Reduz o tráfego de memória, não o número de contas. Complementa o PagedAttention: um corta o tráfego, o outro corta o desperdício de capacidade.

*Aparece em:* Q12

### Chunked prefill

Quebrar um prompt longo em pedaços (chunks) e processar um pedaço por passo, misturado com os decodes em andamento. Assim um prompt de 100K tokens não congela o stream de todos os outros usuários. Em troca, o TTFT desse prompt piora um pouco.

> 💡 **Pense assim:** Em vez de uma pessoa ocupar o caixa por uma hora, ela é atendida em fatias entre os outros clientes.

*Aparece em:* Q15, Q18, Q23, Q47, Q53

### CUDA graphs

Gravar o passo de decode inteiro (centenas de lançamentos de kernel) uma vez e reexecutá-lo com uma única chamada. Reduz o overhead da CPU, mas exige formas fixas, então o engine captura alguns tamanhos de batch e preenche (padding) até o mais próximo.

*Aparece em:* Q15, Q30, Q35, Q40, Q54

## 4. Speculative decoding

### Speculative decoding

Um modelo pequeno e barato (o drafter) propõe vários tokens; o modelo grande verifica todos de uma vez, em uma única passada paralela. Os aceitos são emitidos de graça e, na primeira rejeição, o modelo grande corrige. Com a regra de aceitação certa a distribuição de saída é exatamente a do modelo grande, então reduz a latência sem perder qualidade.

> 💡 **Pense assim:** Um estagiário rascunha a resposta e o chefe só corrige onde errou, em vez de escrever tudo do zero.

*Aparece em:* Q19, Q23, Q35, Q43, Q46

### Acceptance rate (α)

A fração dos tokens propostos que o target aceita. Quanto maior, mais tokens saem por passada do target. É alta em código e texto previsível e baixa em texto criativo.

*Aparece em:* Q19, Q46

### Rejection sampling

A regra que garante a fidelidade: aceita o token proposto com probabilidade min(1, p_target/p_draft); se rejeitar, sorteia de uma distribuição residual corrigida. O resultado tem exatamente a distribuição do target.

*Aparece em:* Q19, Q46

### Medusa e EAGLE

Drafters que ficam dentro do próprio modelo: Medusa acrescenta cabeças extras que prevêem vários tokens futuros de uma vez; EAGLE prevê no espaço de features da última camada do modelo, com aceitação melhor, e verifica uma árvore de continuações.

*Aparece em:* Q19, Q46

### N-gram / prompt-lookup drafting

Em vez de um modelo, o rascunho é copiado de trechos do próprio prompt. Praticamente grátis e muito eficaz quando a saída repete a entrada (edição, extração, citações).

*Aparece em:* Q19, Q46

## 5. Quantização

### Quantization

Guardar os números em menos bits (por exemplo, FP16 para INT8, FP8 ou INT4) para economizar memória e, dependendo do esquema, acelerar. Tem um custo de qualidade que deve ser medido na sua própria tarefa.

*Aparece em:* Q7, Q20, Q21, Q23, Q53

### Weights-only (W4A16, W8A16)

Só os pesos ficam em 4 ou 8 bits; as contas continuam em FP16. Como o decode é limitado pela leitura dos pesos, isso acelera o decode de batch pequeno e economiza memória, mas ajuda pouco em prefill e em batches grandes.

*Aparece em:* Q7, Q20

### W8A8 (weights + activations)

Pesos e ativações em 8 bits, o que permite usar tensor cores de INT8/FP8 e acelerar as próprias multiplicações. Ajuda nos regimes compute-bound: prefill e alto volume.

*Aparece em:* Q7, Q16, Q20

### Activation outliers e SmoothQuant

Algumas ativações têm valores extremos em certos canais, o que atrapalha a quantização para 8 bits. O SmoothQuant move parte dessa dificuldade para os pesos, que são mais fáceis de quantizar.

*Aparece em:* Q7, Q20

### GPTQ

Método de PTQ de 4 bits só para pesos. Quantiza camada por camada e compensa o erro de arredondamento nos pesos seguintes, usando informação aproximada de segunda ordem (Hessiana).

*Aparece em:* Q7, Q20

### AWQ

Método de PTQ de 4 bits só para pesos. Percebe que uma pequena fração dos canais é muito importante (medindo pelas ativações) e reescala esses canais antes de quantizar para protegê-los.

*Aparece em:* Q7, Q20

### GGUF e k-quants

GGUF é o formato de arquivo do llama.cpp (não é um algoritmo). Os k-quants (Q4_K_M etc.) são seus esquemas de quantização por blocos. É o padrão para CPU, Apple Silicon e edge.

*Aparece em:* Q7, Q20, Q23, Q34

### FP8, FP4 (NVFP4 / MXFP4)

Formatos de ponto flutuante de 8 e 4 bits. FP8 é o carro-chefe em Hopper e Blackwell, quase sem perda. FP4 é suportado nativamente pela Blackwell, mas é mais sensível e exige avaliação cuidadosa.

*Aparece em:* Q5, Q7, Q20, Q21, Q55

### KV-cache quantization

Guardar o KV cache em FP8 ou INT8 em vez de FP16. Metade dos bytes por token significa o dobro de sequências ou de contexto na mesma memória. As keys são mais sensíveis do que os values.

*Aparece em:* Q2, Q7, Q21, Q37, Q55

## 6. Métricas, SLOs e latência

### TTFT (time to first token)

Tempo entre a chegada da requisição e o primeiro token gerado. É fila + prefill, e cresce com o tamanho do prompt e a carga. É a métrica de "parece responsivo?".

*Aparece em:* Q1, Q3, Q18, Q22, Q53

### TPOT / ITL

Tempo médio entre um token e o seguinte durante o decode. É a métrica de "o stream é fluido?". Latência total ≈ TTFT + TPOT × tokens gerados.

*Aparece em:* Q1, Q3, Q18, Q22, Q47

### Tokens/sec (por requisição vs agregado)

Pode ser a velocidade de uma requisição (≈ 1/TPOT) ou a vazão do sistema inteiro (soma de todas as requisições). Um sistema pode ter ótima vazão agregada e experiência individual ruim.

*Aparece em:* Q3, Q22, Q32, Q39, Q51

### Goodput

A vazão que de fato cumpre os SLOs (por exemplo, requisições/s com P99 de TTFT abaixo de 1 s). Raw throughput pode enganar em saturação, onde todas as requisições estouram o prazo; o goodput cai a zero.

*Aparece em:* Q22, Q39, Q40, Q46, Q47

### Percentis (P50, P95, P99)

P99 é o valor abaixo do qual ficam 99% das requisições: mostra os piores casos, que a média esconde. Em fluxos com várias chamadas em sequência, as caudas se acumulam.

*Aparece em:* Q3, Q15, Q30, Q38, Q53

### SLO, SLI e error budget

SLO: a meta (ex.: P99 TTFT < 1,5 s). SLI: a métrica que mede isso. Error budget: quanto você pode falhar dentro do SLO. Burn rate: a velocidade com que esse orçamento está sendo consumido; alertar nele evita falsos alarmes de picos curtos.

*Aparece em:* Q22, Q38, Q41, Q47, Q51

### Knee (joelho da curva)

O ponto da curva vazão × latência em que a latência começa a explodir enquanto a vazão para de subir. A operação normal deve ficar bem à esquerda dele (≈ 60 a 70% da capacidade).

*Aparece em:* Q15, Q22, Q30

### Queueing / queue depth

Fila de espera antes de a requisição entrar no batch. Perto da saturação, o tempo de fila cresce muito rápido. Queue depth é um bom sinal antecipado para alertas e autoscaling.

*Aparece em:* Q15, Q22, Q40, Q41, Q53

### Load testing: open loop vs closed loop

Closed loop: N clientes que só enviam a próxima requisição quando a anterior termina, então o teste se alivia sozinho quando o servidor desacelera (coordinated omission). Open loop: requisições chegam num ritmo fixo (por exemplo, processo de Poisson) independente das respostas, o que revela a fila crescendo.

*Aparece em:* Q30, Q52

## 7. Paralelismo e hardware

### Tensor parallelism (TP)

Dividir as matrizes de cada camada entre várias GPUs. Todas participam de cada token, o que reduz a latência por token, mas exige um all-reduce por camada e portanto NVLink. Padrão dentro de um nó.

*Aparece em:* Q24, Q39, Q41, Q50, Q56

### Pipeline parallelism (PP)

Dividir o modelo por faixas de camadas (GPU 0 tem as camadas 1 a 20 etc.). Comunica pouco e tolera rede mais lenta, mas não reduz a latência por token e deixa estágios ociosos (bubbles) se não houver trabalho suficiente.

*Aparece em:* Q23, Q24, Q47, Q56

### Expert parallelism (EP)

Para modelos MoE: cada GPU guarda experts inteiros e os tokens são enviados à GPU certa. Mantém as multiplicações de cada expert grandes, mas põe dois all-to-all por camada MoE no caminho crítico.

*Aparece em:* Q24, Q50, Q56

### Context parallelism e ring attention

Dividir a sequência (não as camadas) entre GPUs, cada uma com uma fatia de Q, K e V, passando blocos de KV em anel. Acelera muito o prefill de contextos enormes e deixa o KV cache distribuído.

*Aparece em:* Q55

### All-reduce e all-to-all

Operações de comunicação entre GPUs. All-reduce: todas somam e compartilham um resultado (usado no TP). All-to-all: cada GPU envia dados diferentes a cada outra (usado para despachar tokens aos experts no MoE).

*Aparece em:* Q24, Q33, Q50, Q56

### NVLink, InfiniBand e RDMA

NVLink é a interconexão muito rápida entre GPUs de um mesmo nó (ou rack). InfiniBand e Ethernet ligam nós diferentes, cerca de uma ordem de grandeza mais lentos. RDMA permite copiar memória entre máquinas sem passar pela CPU.

*Aparece em:* Q24, Q41, Q47, Q56

### NVL72 (domínio NVLink de rack)

Sistemas em que 72 GPUs formam um único domínio NVLink dentro de um rack, empurrando a fronteira entre "rede rápida" e "rede lenta" do servidor para o rack. Viabiliza expert parallelism bem amplo, ao custo de um raio de falha maior.

*Aparece em:* Q56

### Hopper, Blackwell, H100, H200, MI300X

Gerações e modelos de GPU. Hopper (H100/H200) tem FP8 nativo; Blackwell adiciona FP4 nativo; MI300X é a GPU da AMD, com 192 GB de HBM. Importa pela capacidade de memória, largura de banda e formatos suportados.

*Aparece em:* Q16, Q20, Q33, Q37, Q56

### TPU, Inferentia e Trainium

Alternativas às GPUs NVIDIA: TPU (Google), Inferentia e Trainium (AWS, via compilador Neuron) e o ecossistema ROCm da AMD. Podem ser mais baratas por token, mas com menor cobertura de modelos e recursos.

*Aparece em:* Q23, Q33

### Mixture of Experts (MoE)

Arquitetura em que o modelo tem muitos "experts" mas só alguns são ativados por token. Total de parâmetros define a memória; parâmetros ativos definem o cálculo por token. Em serving, o desafio é balancear a carga entre experts e a comunicação all-to-all.

> 💡 **Pense assim:** Hospital com dezenas de especialistas: cada paciente só passa por dois ou três, mas o prédio inteiro precisa existir.

*Aparece em:* Q16, Q24, Q35, Q50, Q56

### Hybrid Mamba-attention (SSM)

Arquitetura que mistura poucas camadas de attention com muitas camadas de Mamba/linear attention, que mantêm um estado de tamanho fixo em vez de um KV cache crescente. Barata em contexto longo, mas complica o gerenciamento de cache, pois o estado recorrente não pode ser fatiado token a token.

*Aparece em:* Q25, Q34, Q54, Q55

## 8. Escalando além de uma réplica

### Prefill/decode disaggregation

Separar o prefill e o decode em pools de GPUs diferentes e transferir o KV cache entre eles. Cada pool é ajustado para seu próprio SLO (TTFT ou ITL) e um prompt gigante não congela os streams dos outros. Em troca há o custo de transferir o KV e mais complexidade; só compensa em escala. Chunked prefill divide o tempo de um pool; a disaggregation divide o espaço.

*Aparece em:* Q32, Q47, Q50, Q54, Q56

### NVIDIA Dynamo e llm-d

Camada de orquestração acima dos engines (não são engines): cuidam de disaggregation, roteamento ciente de KV e offload de KV numa frota.

*Aparece em:* Q23, Q47, Q56

### Cache-aware routing / session affinity

Rotear requisições para a réplica que já tem o prefixo no cache, em vez de round-robin, que espalha os prefixos e destrói o hit rate. A afinidade por sessão (conversa na mesma réplica) resolve a maior parte do problema; não se deve rotear só por cache, sem olhar a carga.

*Aparece em:* Q29, Q47, Q49, Q52, Q53

### Multi-LoRA serving

Carregar o modelo base uma vez e trocar pequenos adapters LoRA (as fine-tunes de cada cliente) por requisição, no mesmo batch. Centenas de fine-tunes cabem em poucas GPUs. O prefix cache não é compartilhado entre adapters diferentes.

> 💡 **Pense assim:** Uma máquina de café com uma cápsula diferente por cliente, em vez de uma máquina inteira para cada cliente.

*Aparece em:* Q29, Q33, Q48, Q49, Q53

### Constrained / guided decoding

Compilar um JSON Schema, regex ou gramática em um autômato e, a cada passo, mascarar os tokens que quebrariam o formato. A saída é sintaticamente válida por construção. Garante sintaxe, não que os valores estejam certos.

*Aparece em:* Q19, Q23, Q33, Q36, Q44

### Cold start

O tempo entre decidir subir uma nova réplica e ela atender de fato: obter o nó de GPU, baixar a imagem, carregar os pesos (centenas de GB) e fazer warm-up. Pode levar minutos, por isso se usam warm pools e escala preditiva.

*Aparece em:* Q40

### Autoscaling

Aumentar ou diminuir o número de réplicas conforme a demanda. Em LLMs deve se basear em sinais antecipados (fila, utilização de KV), não em CPU ou na utilização da GPU, que engana.

*Aparece em:* Q39, Q40, Q49, Q53, Q56

### GPU utilization (por que engana)

Mede quanto tempo a GPU esteve ocupada, não quão bem ela está sendo usada. Um decode limitado pela memória pode mostrar 100% enquanto faz pouco cálculo útil.

*Aparece em:* Q22, Q40, Q41, Q53

### Admission control e load shedding

Admission control decide quais requisições aceitar. Load shedding descarta ou adia as menos importantes quando há sobrecarga, para proteger o tráfego prioritário. Backpressure é a pressão de volta para quem envia desacelerar.

*Aparece em:* Q38, Q43, Q51, Q55, Q56

### Graceful degradation

Perder qualidade ou funcionalidade de forma planejada antes de perder disponibilidade: modelo menor, contexto menor, saídas mais curtas, desligar recursos opcionais.

*Aparece em:* Q34, Q38, Q40, Q43, Q55

### Noisy neighbor / multi-tenancy

Tenants são os diferentes clientes ou times que dividem a mesma frota. Noisy neighbor é um tenant que, ao consumir muito (por exemplo, KV com contextos longos), degrada os demais.

*Aparece em:* Q27, Q29, Q45, Q51, Q53

### Fair queue (virtual token counter)

Escalonamento que prioriza o tenant que consumiu menos serviço recentemente, em vez de pura ordem de chegada. Token bucket limita o consumo por tokens por segundo, uma unidade mais justa do que requisições por segundo.

*Aparece em:* Q8, Q18, Q51

## 9. Stacks de serving

### vLLM

O engine open source padrão: PagedAttention, continuous batching, chunked prefill, prefix caching, speculative decoding, quantização e ampla cobertura de modelos e hardware, com servidor compatível com a API da OpenAI.

*Aparece em:* Q17, Q23, Q29, Q47, Q54

### SGLang

Engine forte em cargas com muitos prefixos compartilhados (agentes, chat multi-turno) graças ao RadixAttention, além de saída estruturada rápida.

*Aparece em:* Q23, Q35, Q44, Q46, Q49

### TensorRT-LLM

Caminho de máxima performance da NVIDIA, com kernels muito afinados e FP8/FP4. Dá os melhores números em hardware NVIDIA, em troca de mais complexidade e dependência do fornecedor.

*Aparece em:* Q21, Q23, Q33, Q46, Q54

### TGI

Servidor da Hugging Face, hoje em modo de manutenção (segundo o material, desde o fim de 2025), com migração indicada para vLLM e SGLang. Não é uma escolha para novos projetos.

*Aparece em:* Q8, Q23

### llama.cpp, Ollama e MLX

Stack de edge e local: CPU, Apple Silicon e GPUs de consumo, com modelos GGUF. Ótimo para desenvolvimento e on-device, não para serving de alto volume em datacenter.

*Aparece em:* Q7, Q20, Q21, Q23, Q34

### On-device / edge inference

Rodar o modelo no dispositivo do usuário (celular, laptop). Limitado por memória, banda (batch 1 sempre), temperatura e bateria. Costuma ser híbrido: o modelo local faz o simples e escala os casos difíceis para a nuvem.

*Aparece em:* Q20, Q23, Q34, Q41, Q47

## 10. APIs, streaming e confiabilidade

### SSE (Server-Sent Events)

Protocolo simples sobre HTTP em que o servidor mantém a resposta aberta e envia eventos aos poucos. É como as APIs de LLM fazem streaming de tokens.

*Aparece em:* Q4, Q13

### Streaming

Entregar os tokens conforme são gerados, em vez de esperar a resposta completa. A latência percebida passa a ser aproximadamente o TTFT.

*Aparece em:* Q4, Q13, Q28, Q42, Q48

### Partial JSON parsing e accumulator

Em tool calls com streaming, os argumentos chegam em pedaços. O accumulator junta os fragmentos por bloco; só quando o bloco termina se valida e executa. O parser de JSON parcial serve só para mostrar progresso na interface.

*Aparece em:* Q28

### finish_reason / stop_reason

Campo da resposta que diz por que a geração parou: terminou naturalmente, bateu o limite de tokens (length), achou uma stop sequence, quer chamar uma ferramenta ou foi bloqueada por segurança. Ignorá-lo faz você tratar saída truncada como completa.

*Aparece em:* Q14

### max_tokens

Limite de tokens que o modelo pode gerar. Controla custo e latência, mas se for atingido a resposta é cortada no meio (inclusive um JSON).

*Aparece em:* Q13, Q15, Q45, Q51, Q53

### Rate limits: 429, RPM, TPM

O provedor limita requisições por minuto (RPM) e tokens por minuto (TPM). Passar do limite devolve o erro 429. A solução é controlar o consumo do lado do cliente, usar retry com backoff e mover tráfego não urgente para batch.

*Aparece em:* Q10, Q26, Q36, Q42, Q48

### Exponential backoff + jitter

Ao repetir uma chamada que falhou, esperar um tempo que dobra a cada tentativa e é sorteado (jitter), para que os clientes não tentem de novo todos ao mesmo tempo (thundering herd). Respeite o cabeçalho Retry-After quando existir.

*Aparece em:* Q10, Q42

### Retry budget

Limite global para a fração de tráfego que pode estar em retry. Acima dele, o sistema para de repetir e descarta, para não amplificar uma sobrecarga.

*Aparece em:* Q10, Q42

### Circuit breaker

Disjuntor por provedor e modelo: se a taxa de falhas sobe, ele "abre" e as chamadas falham rápido (ou vão para o fallback); depois de um tempo, deixa passar poucas tentativas de teste (half-open) e fecha de novo se voltou ao normal.

> 💡 **Pense assim:** O disjuntor de casa: desarma para a fiação não pegar fogo e você religa quando o problema passou.

*Aparece em:* Q10, Q42, Q48, Q51

### Timeouts em camadas

Em vez de um único timeout, usar vários: de conexão, de primeiro token (TTFT), de inatividade entre chunks e total. Assim se distingue "lento mas progredindo" de "travado".

*Aparece em:* Q36, Q38, Q41, Q42, Q43

### Idempotency key

Identificador que garante que repetir uma operação não a executa duas vezes (por exemplo, não mandar o mesmo e-mail em dobro quando um agente refaz um passo).

*Aparece em:* Q13, Q15, Q26, Q28, Q42

### LLM gateway

Camada única entre a aplicação e os provedores: normaliza formatos de API, centraliza credenciais, quotas, medição de custo e redação de dados sensíveis, e decide o roteamento.

*Aparece em:* Q38, Q39, Q41, Q48, Q51

### Reasoning tokens / extended thinking

Tokens de "raciocínio" que modelos de reasoning geram antes da resposta. São cobrados como saída e ocupam KV e passos de decode, então a quantidade de tokens por requisição sobe muito e a cauda fica pesada.

*Aparece em:* Q6, Q14, Q31, Q42, Q45

### Floating-point e determinismo

Somas em ponto flutuante dependem da ordem, e os kernels escolhem a ordem conforme o formato do batch. Como o batch muda com o tráfego dos outros usuários, os logits variam nos últimos bits e podem trocar o token escolhido, mesmo com temperature 0.

*Aparece em:* Q9, Q11

## 11. Custo e decisões de produção

### Input vs output tokens (preço)

Tokens de saída custam cerca de 4 a 8 vezes mais que os de entrada, pois cada um exige uma passada sequencial inteira, enquanto a entrada é processada em paralelo. Por isso: prompt generoso, resposta enxuta.

*Aparece em:* Q6, Q9, Q45, Q51, Q52

### Cache read / write (preço de caching)

Nas APIs com prompt caching, ler do cache custa uma fração do preço de entrada (cerca de 0,1× na Anthropic, segundo o material), e gravar pode ter um adicional. O TTL é o tempo que a entrada fica no cache. Os valores mudam; consulte a tabela de preços atual.

*Aparece em:* Q9, Q16, Q43

### Batch API

API assíncrona com janela de até ~24 h e cerca de 50% de desconto, ideal para trabalho sem usuário esperando (evals, backfills, enriquecimento). Cada requisição leva um custom_id para reconciliar os resultados, que voltam fora de ordem.

*Aparece em:* Q26, Q31, Q42, Q43, Q45

### Semantic caching

Guarda pares pergunta/resposta e devolve a resposta salva quando uma nova pergunta é parecida (por embedding). Economiza 100% em um acerto, mas é com perda: pode servir resposta errada, vazar dados entre usuários ou ficar defasado. É diferente do prefix caching, que reaproveita cálculo e não a resposta.

*Aparece em:* Q9, Q27

### Model tiering / cascade

Mandar o tráfego fácil para um modelo pequeno e barato e escalar para o grande só quando necessário (verificando a resposta). Se escalar demais, sai mais caro que ir direto ao grande.

*Aparece em:* Q31, Q43, Q45, Q48, Q51

### Service tiers: standard, priority, flex, provisioned

O mesmo modelo vendido com garantias diferentes: standard (sob demanda), priority (premium para ser atendido primeiro), flex/batch (desconto em troca de lentidão) e provisioned throughput (capacidade reservada e paga mesmo sem uso, que só compensa com uso alto e estável).

*Aparece em:* Q10, Q25, Q36, Q39

### Self-host vs API

Rodar o modelo na sua infraestrutura versus pagar por token a um provedor. Self-hosting só compensa com utilização alta e constante, ou por requisitos de privacidade e customização, e exige um time de infra de plantão.

*Aparece em:* Q9, Q13, Q25, Q36, Q55

### Capacity planning

Estimar quantas GPUs são necessárias: converter o pico de demanda em tokens/s, dividir pelo goodput medido por réplica sob o SLO e acrescentar folga (headroom) de cerca de 1,4 a 1,7× mais redundância N+1.

*Aparece em:* Q38, Q39, Q41, Q51, Q56
