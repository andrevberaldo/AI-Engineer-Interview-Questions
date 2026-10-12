# 🎛️ Fine-tuning, RLHF & Alignment

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Este tópico aparece em quase todo processo seletivo de AI Engineer, porque separa quem apenas chama APIs de quem sabe mudar o comportamento de um modelo. Conte com ele se você está se candidatando a vagas de GenAI/LLM engineer em frontier labs, em times de applied ML de big techs ou em qualquer startup que entrega modelos customizados. Os entrevistadores usam o tema para checar se você sabe *quando* fazer fine-tuning, quanto isso custa e como o post-training funciona por baixo dos panos.

## Crash course

### O framework de decisão: prompt → RAG → fine-tune

**Fine-tuning muda a forma; RAG muda o conhecimento.** O caminho de escalada padrão é: prompt engineering (horas, de graça) → RAG (dias, traz conhecimento atualizado ou privado) → fine-tuning (semanas, muda o comportamento). Faça fine-tuning quando você precisar de **estilo/formato/tom** consistentes, de **structured output** confiável, de **ganhos de latency/custo** (um modelo de 8B ajustado no lugar de um modelo frontier, ou o corte de um system prompt de 2.000 tokens) ou de **habilidade em tarefa estreita** (classificação, extração, um dialeto SQL próprio da empresa). Não faça fine-tuning para injetar fatos: a injeção de conhecimento via SFT é pouco confiável e fica desatualizada. Esse é o papel do RAG (ou do continued pretraining, se a *própria linguagem do domínio* for estranha ao modelo, como corpora jurídicos ou de genômica). A nuance é que sistemas reais combinam os dois: RAG para os fatos, e um modelo com fine-tuning que seja melhor em *usar* o contexto recuperado.

### Mecânica do SFT

**Supervised fine-tuning (SFT)** é a previsão do próximo token sobre pares (prompt, resposta) curados, renderizados pelo **chat template** do modelo. Esse template é a markup exata de special tokens (por exemplo `<|im_start|>`, `[INST]`, header tokens) com a qual o modelo passou pelo post-training. Template diferente entre treino e inferência é a falha silenciosa mais comum em fine-tuning: a loss parece normal, mas as saídas pioram.

**Loss masking:** calcule a cross-entropy apenas sobre os tokens da resposta e defina os labels dos tokens do prompt como `-100`. Assim o modelo aprende a *responder*, não a *reproduzir prompts*:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100   # ignorado por F.cross_entropy
```

**Qualidade >> quantidade.** O LIMA mostrou que ~1.000 exemplos meticulosamente curados bastam para alinhar um modelo base de 65B. Algumas centenas a alguns milhares de exemplos excelentes, deduplicados e diversos superam 100 mil exemplos coletados da web sem curadoria. Faça dedup (exata, mais quase-duplicatas via MinHash/embeddings), descontamine contra seus eval sets e leia manualmente uma amostra aleatória. Sempre.

### PEFT: LoRA e QLoRA

O **LoRA** congela os pesos base e aprende uma atualização de baixo rank para cada matriz-alvo:

$$W' = W + \frac{\alpha}{r} BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll d$$

`B` começa em zero, então o treino parte do modelo base. Valores típicos: `r` entre 8 e 64, `alpha` ≈ r ou 2r, aplicado a todas as projeções de attention (q/k/v/o) e, com frequência, às matrizes up/down/gate do MLP. Os parâmetros treináveis caem para ~0,1-1% do modelo. O ganho real, porém, é outro: **não há gradientes nem estados do Adam para os pesos congelados**, o que reduz a memória de treino em várias vezes. Os adapters podem ser mesclados em `W`, sem nenhum overhead na inferência, ou mantidos separados para **multi-LoRA serving** (um modelo base e muitos adapters que se trocam a quente, como no vLLM/S-LoRA).

O **QLoRA** combina três coisas: a base congelada quantizada em 4-bit **NF4** (um tipo de dado projetado, com base em teoria da informação, para pesos com distribuição normal), a **double quantization** (quantizar as próprias constantes de quantização) e os **paged optimizers** (que descarregam os estados do otimizador para a RAM da CPU nos picos de memória). O LoRA é treinado em bf16 por cima disso. O resultado é fazer fine-tuning de um modelo de 65-70B em uma única GPU de 48 GB, com pouca perda de qualidade nos benchmarks, embora você esteja treinando sobre uma base levemente ruidosa.

### Variantes de LoRA e model merging

Cada variante corrige uma fraqueza conhecida do LoRA vanilla. O **rsLoRA** escala a atualização por $\alpha/\sqrt{r}$ em vez de $\alpha/r$, para que ranks altos não sejam amortecidos até perderem a utilidade. O **LoRA+** dá a `B` um learning rate bem maior que o de `A`, já que os dois têm papéis assimétricos. O **DoRA** divide cada peso em magnitude e direção: aplica o LoRA à direção e treina a magnitude separadamente. Com isso, fecha boa parte da distância para o fine-tuning completo em rank baixo, ao custo de um pouco mais de compute no treino. Todas continuam mescláveis, e nenhuma delas pesa tanto quanto a qualidade dos dados e a escolha dos módulos-alvo.

O **model merging** combina fine-tunes de uma *mesma base* sem nenhum treino. A task arithmetic soma task vectors $\tau = \theta_{ft} - \theta_{base}$ com pesos. O **TIES** apara os deltas pequenos e resolve conflitos de sinal antes de tirar a média. O **DARE** descarta aleatoriamente a maioria das entradas do delta e reescala o restante. O **SLERP** interpola esfericamente entre exatamente dois modelos. O merging funciona dentro de uma mesma bacia de loss (fine-tunes irmãos, checkpoint soups, LoRAs sobre a mesma base, ou puxar um fine-tune estreito de volta para a base a fim de recuperar capacidade geral) e falha entre modelos pré-treinados de forma independente. Os pesos se encontram varrendo valores contra um eval real, não por teoria.

### Conta de memória do treino (por que você não faz full fine-tuning de um 7B em uma placa de 24 GB)

O full fine-tuning em precisão mista com Adam custa cerca de **16 bytes/param**: 2 (pesos em bf16) + 2 (gradientes) + 4 (pesos master em fp32) + 8 (m e v do Adam em fp32). Um modelo de 7B dá ~112 GB, antes das activations. As alavancas são: **gradient checkpointing** (recalcular as activations; fica ~30% mais lento, mas economiza muita memória), **gradient accumulation** (simular batches grandes), **ZeRO/FSDP** (shard dos estados do otimizador, dos gradientes e dos parâmetros entre as GPUs) e PEFT (estados do otimizador só para os parâmetros do adapter).

### Hiperparâmetros principais

| Parâmetro | Full FT | LoRA |
|---|---|---|
| LR | ~1e-5 a 5e-5 | ~1e-4 a 2e-4 |
| Epochs | 1-3 | 1-5 |
| Schedule | cosine + ~3% de warmup | igual |

Fique de olho no **catastrophic forgetting** (regressões na capacidade geral). Para mitigar, use LR menor, menos epochs, LoRA no lugar de full FT e misture ~5-25% de dados gerais de instruções aos dados da sua tarefa. **Sinais de overfitting:** eval loss subindo enquanto a train loss cai, regurgitação literal do treino, diversidade de saída em colapso e regressões em benchmarks. Sempre rode o eval **antes e depois**, em (1) um conjunto de tarefas held-out e (2) uma suíte de capacidade geral (por exemplo, no estilo MMLU, ou de instruction following), para pegar as regressões.

### Otimização de preferências: RLHF → DPO → GRPO

**Pipeline de RLHF** (receita do InstructGPT): SFT → coleta de pares de preferência humana → treino de um **reward model** com a loss de Bradley-Terry, $-\log\sigma(r(y_w) - r(y_l))$ → otimização da policy com **PPO** contra essa recompensa, com uma **penalidade de KL por token em relação ao reference model congelado**. A penalidade impede que a policy derive para textos degenerados que rendem recompensa alta. Sem o termo de KL, você tem **reward hacking**: sycophancy, respostas inchadas que soam confiantes, gibberish que engana o RM.

O **DPO** simplifica tudo isso. O objetivo do RLHF com restrição de KL tem forma fechada, então a própria policy define uma *recompensa implícita* $\beta\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$, e você treina direto sobre os pares de preferência com uma loss de classificação. Não há reward model, nem loop de RL, nem sampling durante o treino. O método é mais barato e mais estável. Já o RL online (PPO ou métodos da família GRPO) ainda vence quando dá para gerar e pontuar amostras novas, e é a norma no post-training de fronteira. Em uma linha cada: **IPO** (objetivo limitado, com menos overfitting às preferências), **KTO** (funciona com labels binários bom/ruim, sem precisar de pares), **ORPO** (incorpora a loss de preferência ao SFT, sem reference model).

O **GRPO** (introduzido no DeepSeekMath e escalado no DeepSeek-R1) amostra um *grupo* de respostas por prompt e usa a recompensa de cada uma, normalizada pela média e pelo desvio padrão do grupo, como advantage, no lugar de uma value network aprendida. É o PPO sem o critic, mantendo o termo de KL-to-reference na loss. Combinado com **verifiable rewards** (RLVR: unit tests, verificadores de resposta matemática), é o que move o RL para reasoning. Matemática e código se prestam bem ao RL porque a recompensa é programática, então não há reward model aprendido para sofrer hacking (um verificador fraco ainda pode ser burlado). Variantes posteriores corrigem fraquezas específicas: o **Dr. GRPO** remove os vieses de comprimento e de normalização por desvio padrão, o **DAPO** desacopla o clip range e filtra os grupos em que todas as amostras recebem a mesma pontuação, e o **GSPO** faz clipping no nível da sequência, para dar estabilidade em modelos grandes e MoE.

Quando a tarefa não é verificável, você precisa de uma recompensa aprendida. Um **outcome reward model (ORM)** pontua a resposta inteira (labels baratos, mas credit assignment às cegas). Um **process reward model (PRM)** pontua cada passo do raciocínio: o sinal é denso, os labels são caros ou derivados de rollouts, e ele serve mais para reranking e search em tempo de inferência. O **RLAIF/Constitutional AI** troca os labels humanos por feedback de IA, guiado por um conjunto explícito de princípios.

### Distillation & synthetic data

A **black-box distillation** gera dados de treino sintéticos a partir de um modelo mais forte. É a forma mais comum, mas confira os ToS do teacher: muitos provedores de API proíbem usar as saídas para treinar modelos concorrentes. A **logit distillation** faz o modelo imitar as distribuições completas de tokens do teacher. Ela precisa dos logits do teacher, então na prática exige pesos abertos, e em troca dá mais sinal por exemplo. As armadilhas de synthetic data são o **mode collapse** (os tiques de estilo do teacher ficam amplificados), a **error amplification** (os erros do teacher viram ground truth) e a contaminação de eval. Filtre com verificadores/judges e mantenha dados humanos na mistura.

## Perguntas de entrevista

Todas as 56 perguntas com respostas detalhadas: [questions.pt-BR.md](questions.pt-BR.md)

## Red flags que os entrevistadores observam

- Recorrer ao fine-tuning para "ensinar fatos ao modelo", sem saber que conhecimento é assunto de RAG/continued pretraining e que fine-tuning serve para comportamento e forma.
- Não saber escrever a equação do LoRA nem explicar o que `r` e `alpha` fazem, ou achar que o LoRA acelera a inferência de um adapter não mesclado.
- Não ter conta de memória: não conseguir estimar por que o full fine-tuning de um modelo de 7B com Adam exige ~112 GB, nem de onde vem a economia do QLoRA.
- Descrever o RLHF como "treinar com thumbs up/down", sem conseguir explicar o reward model, o PPO ou por que existe a penalidade de KL contra o reference model.
- Achar que o DPO é estritamente melhor que o PPO (ou vice-versa), em vez de conhecer o tradeoff offline vs. online.
- Nunca mencionar avaliação: nada de held-out set, de checagens de regressão antes/depois, de atenção à contaminação.
- Ignorar chat templates e loss masking. Quem já fez fine-tuning de verdade se queimou com os dois.
- Tratar synthetic data como almoço grátis, sem falar de mode collapse, error amplification ou restrições de ToS/licenciamento.

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
