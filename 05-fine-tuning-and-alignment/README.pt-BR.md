# 🎛️ Fine-tuning, RLHF & Alignment

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Este tópico aparece em quase todo processo seletivo de AI Engineer porque separa quem chama APIs de quem consegue mudar o comportamento de um modelo. Espere vê-lo se você está entrevistando para vagas de GenAI/LLM engineer em frontier labs, times de applied ML em big techs ou qualquer startup que entrega modelos customizados - os entrevistadores o usam para sondar se você sabe *quando* fazer fine-tuning, quanto custa e como o post-training realmente funciona por baixo dos panos.

## Crash course

### O framework de decisão: prompt → RAG → fine-tune

**Fine-tuning muda a forma; RAG muda o conhecimento.** O caminho padrão de escalonamento é: prompt engineering (horas, grátis) → RAG (dias, adiciona conhecimento atualizado/privado) → fine-tuning (semanas, muda o comportamento). Faça fine-tuning quando você precisa de **estilo/formato/tom** consistentes, **structured output** confiável, **ganhos de latency/custo** (um modelo de 8B ajustado substituindo um modelo frontier, ou eliminando um system prompt de 2.000 tokens) ou **habilidade em tarefa estreita** (classificação, extração, um dialeto SQL da casa). Não faça fine-tuning para injetar fatos - injeção de conhecimento via SFT é pouco confiável e fica desatualizada; esse é o trabalho do RAG (ou do continued pretraining, se a *própria linguagem do domínio* for estranha ao modelo, por exemplo corpora jurídicos ou de genômica). A nuance: sistemas reais combinam os dois - RAG para fatos, e um modelo com fine-tuning que é melhor em *usar* o contexto recuperado.

### Mecânica do SFT

**Supervised fine-tuning (SFT)** = previsão do próximo token sobre pares (prompt, resposta) curados, renderizados pelo **chat template** do modelo - a markup exata de special tokens (por exemplo `<|im_start|>`, `[INST]`, header tokens) com a qual o modelo foi pós-treinado. Incompatibilidade de template entre treino e inferência é a falha silenciosa mais comum em fine-tuning: a loss parece normal, as saídas degradam.

**Loss masking:** calcule a cross-entropy apenas sobre os tokens da resposta; defina os labels dos tokens do prompt como `-100` para que o modelo aprenda a *responder*, não a *reproduzir prompts*:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100   # ignorado por F.cross_entropy
```

**Qualidade >> quantidade.** O LIMA mostrou que ~1.000 exemplos meticulosamente curados podem alinhar um modelo base de 65B. Algumas centenas a alguns milhares de exemplos excelentes, deduplicados e diversos superam 100 mil exemplos raspados da web. Faça dedup (exata + quase-duplicatas via MinHash/embeddings), descontamine contra seus eval sets e leia manualmente uma amostra aleatória - sempre.

### PEFT: LoRA e QLoRA

**LoRA** congela os pesos base e aprende uma atualização de baixo rank por matriz-alvo:

$$W' = W + \frac{\alpha}{r} BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll d$$

`B` começa em zero, então o treino começa no modelo base. `r` típico = 8-64, `alpha` ≈ r ou 2r, mirando todas as projeções de attention (q/k/v/o) e frequentemente as matrizes up/down/gate do MLP. Os parâmetros treináveis caem para ~0,1-1% do modelo e - o ganho real - **sem gradientes nem estados do Adam para os pesos congelados**, reduzindo a memória de treino várias vezes. Os adapters podem ser mesclados em `W` para zero overhead de inferência, ou mantidos separados para **multi-LoRA serving** (um modelo base, muitos adapters trocáveis a quente, como no vLLM/S-LoRA).

**QLoRA** = base congelada quantizada em 4-bit **NF4** (um tipo de dado com motivação teórico-informacional para pesos normalmente distribuídos) + **double quantization** (quantizar as constantes de quantização) + **paged optimizers** (descarregar estados do otimizador para a RAM da CPU em picos), com LoRA treinado em bf16 por cima. Faz fine-tuning de um modelo de 65-70B em uma única GPU de 48 GB com pouca perda de qualidade medida em benchmarks - embora você esteja treinando sobre uma base levemente ruidosa.

### Variantes de LoRA e model merging

Cada variante corrige uma fraqueza conhecida do LoRA vanilla. **rsLoRA** escala a atualização por $\alpha/\sqrt{r}$ em vez de $\alpha/r$, para que ranks altos não sejam amortecidos até a inutilidade. **LoRA+** dá a `B` um learning rate muito maior que o de `A`, já que os dois têm papéis assimétricos. **DoRA** divide cada peso em magnitude e direção, aplica LoRA à direção e treina a magnitude separadamente, fechando boa parte da diferença para o fine-tuning completo em rank baixo, ao custo de algum compute extra de treino. Todas continuam mescláveis, e nenhuma delas importa tanto quanto a qualidade dos dados e a escolha dos módulos-alvo.

**Model merging** combina fine-tunes de uma *base compartilhada* sem treino. Task arithmetic soma task vectors $\tau = \theta_{ft} - \theta_{base}$ com pesos; **TIES** apara deltas pequenos e resolve conflitos de sinal antes de fazer a média; **DARE** descarta aleatoriamente a maioria das entradas do delta e reescala o restante; **SLERP** interpola esfericamente entre exatamente dois modelos. Funciona dentro de uma mesma bacia de loss (fine-tunes irmãos, checkpoint soups, LoRAs sobre uma mesma base, puxar um fine-tune estreito de volta em direção à base para recuperar capacidade geral) e falha entre modelos pré-treinados de forma independente. Os pesos são encontrados varrendo contra um eval real, não por teoria.

### Matemática de memória de treino (por que você não consegue fazer full fine-tuning de um 7B em uma placa de 24 GB)

Full fine-tuning de precisão mista com Adam custa aproximadamente **16 bytes/param**: 2 (pesos bf16) + 2 (gradientes) + 4 (pesos master em fp32) + 8 (m e v do Adam em fp32). Um modelo de 7B ⇒ ~112 GB antes das activations. Alavancas: **gradient checkpointing** (recalcular activations, ~30% mais lento, grande ganho de memória), **gradient accumulation** (simular batches grandes), **ZeRO/FSDP** (shard de estados do otimizador/gradientes/parâmetros entre GPUs) e PEFT (estados do otimizador apenas para os parâmetros do adapter).

### Hiperparâmetros principais

| Botão | Full FT | LoRA |
|---|---|---|
| LR | ~1e-5 a 5e-5 | ~1e-4 a 2e-4 |
| Epochs | 1-3 | 1-5 |
| Schedule | cosine + ~3% de warmup | igual |

Atenção ao **catastrophic forgetting** (regressões na capacidade geral): mitigue com LR menor, menos epochs, LoRA em vez de full FT e misturando ~5-25% de dados gerais de instruções nos dados da sua tarefa. **Sinais de overfitting:** eval loss subindo enquanto a train loss cai, regurgitação literal, diversidade de saída colapsando, regressões em benchmarks. Sempre faça eval **antes e depois** em (1) um conjunto de tarefas held-out e (2) uma suíte de capacidade geral (por exemplo estilo MMLU, instruction following) para detectar regressões.

### Otimização de preferências: RLHF → DPO → GRPO

**Pipeline de RLHF** (receita do InstructGPT): SFT → coletar pares de preferência humana → treinar um **reward model** com uma loss de Bradley-Terry, $-\log\sigma(r(y_w) - r(y_l))$ → otimizar a policy com **PPO** contra essa recompensa, com uma **penalidade de KL por token em relação ao reference model congelado**, para que a policy não derive para textos degenerados de alta recompensa. Remova o termo de KL e você obtém **reward hacking**: sycophancy, respostas inchadas e de tom confiante, gibberish que engana o RM.

O **DPO** colapsa isso: o objetivo do RLHF com restrição de KL tem forma fechada, então a própria policy define uma *recompensa implícita* $\beta\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$ e você pode treinar diretamente sobre pares de preferência com uma loss de classificação - sem reward model, sem loop de RL, sem sampling durante o treino. Mais barato e mais estável; o RL online (PPO ou métodos da família GRPO) ainda vence quando você consegue gerar e pontuar amostras novas, e é a norma no post-training de fronteira. Resumos de uma linha: **IPO** (objetivo limitado, menos overfitting às preferências), **KTO** (funciona com labels binários bom/ruim, sem precisar de pares), **ORPO** (incorpora a loss de preferência ao SFT, sem reference model).

O **GRPO** (introduzido no DeepSeekMath, escalado no DeepSeek-R1): amostre um *grupo* de respostas por prompt e use a recompensa de cada resposta, normalizada pela média e pelo desvio padrão do grupo, como sua advantage em vez de uma value network aprendida - PPO sem o critic, mantendo o termo de KL-to-reference na loss. Combinado com **verifiable rewards** (RLVR: unit tests, verificadores de resposta matemática), ele impulsiona o RL para reasoning: matemática/código são amigáveis ao RL porque a recompensa é programática, então não há reward model aprendido para sofrer hacking (um verificador fraco ainda pode ser burlado). Variantes posteriores corrigem fraquezas específicas: o **Dr. GRPO** remove os vieses de comprimento e de normalização por desvio padrão, o **DAPO** desacopla o clip range e filtra grupos em que todas as amostras recebem a mesma pontuação, o **GSPO** faz clipping em nível de sequência para estabilidade em modelos grandes e MoE.

Quando a tarefa não é verificável, você precisa de uma recompensa aprendida: um **outcome reward model (ORM)** pontua a resposta inteira (labels baratos, credit assignment cego), um **process reward model (PRM)** pontua cada passo de raciocínio (sinal denso, labels caros ou derivados de rollouts, e mais útil para reranking e search em tempo de inferência). **RLAIF/Constitutional AI** substitui labels humanos por feedback de IA guiado por um conjunto explícito de princípios.

### Distillation & synthetic data

**Black-box distillation** = gerar dados de treino sintéticos a partir de um modelo mais forte (o mais comum; verifique os ToS do teacher - muitos provedores de API restringem treinar modelos concorrentes com as saídas). **Logit distillation** = casar as distribuições completas de tokens do teacher (precisa dos logits do teacher, então na prática pesos abertos; mais sinal por exemplo). Armadilhas de synthetic data: **mode collapse** (tiques estilísticos do teacher amplificados), **error amplification** (erros do teacher viram ground truth) e contaminação de eval. Filtre com verificadores/judges e mantenha dados humanos na mistura.

## Perguntas de entrevista

Todas as 56 perguntas com respostas detalhadas: [questions.pt-BR.md](questions.pt-BR.md)

## Red flags que os entrevistadores observam

- Recorrer ao fine-tuning para "ensinar fatos ao modelo" - não saber que conhecimento pertence ao RAG/continued pretraining e que fine-tuning é para comportamento/forma.
- Não consegue escrever a equação do LoRA nem explicar o que `r` e `alpha` fazem; acha que o LoRA acelera a inferência de um adapter não mesclado.
- Sem matemática de memória: não consegue estimar por que o full fine-tuning de um modelo de 7B precisa de ~112 GB com Adam, ou de onde vêm as economias do QLoRA.
- Descreve o RLHF como "treinar com thumbs up/down" - não consegue articular o reward model, o PPO ou por que existe a penalidade de KL contra o reference model.
- Acha que o DPO é estritamente melhor que o PPO (ou vice-versa) em vez de conhecer o tradeoff offline vs. online.
- Nunca menciona avaliação: sem held-out set, sem checagens de regressão antes/depois, sem consciência de contaminação.
- Ignora chat templates e loss masking - quem realmente já fez fine-tuning de um modelo se queimou com ambos.
- Trata synthetic data como almoço grátis - sem menção a mode collapse, error amplification ou restrições de ToS/licenciamento.

## Leituras complementares

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - o paper do InstructGPT (receita canônica de RLHF)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - Zhou et al., 2023
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Anthropic, 2022
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - onde o GRPO foi introduzido
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - GRPO + verifiable rewards em escala
- [Hugging Face TRL documentation](https://huggingface.co/docs/trl) e [PEFT documentation](https://huggingface.co/docs/peft) - a stack open-source de referência para post-training
