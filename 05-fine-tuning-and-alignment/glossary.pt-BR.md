# 📚 Glossário de estudo — Fine-Tuning & Alignment

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Visão geral: o que é adaptar um modelo](#1-visão-geral-o-que-é-adaptar-um-modelo)
- [2. Dados: curadoria, deduplicação e dados sintéticos](#2-dados-curadoria-deduplicação-e-dados-sintéticos)
- [3. Hiperparâmetros e memória de treino](#3-hiperparâmetros-e-memória-de-treino)
- [4. PEFT: LoRA, QLoRA e variantes](#4-peft-lora-qlora-e-variantes)
- [5. Model merging](#5-model-merging)
- [6. RLHF e modelos de recompensa](#6-rlhf-e-modelos-de-recompensa)
- [7. DPO e a família de preferências offline](#7-dpo-e-a-família-de-preferências-offline)
- [8. RL para raciocínio e agentes](#8-rl-para-raciocínio-e-agentes)
- [9. MoE, avaliação, produção e ferramentas](#9-moe-avaliação-produção-e-ferramentas)

## 1. Visão geral: o que é adaptar um modelo

### Fine-tuning

Continuar treinando um modelo já pré-treinado com dados do seu caso, para mudar o *comportamento* dele: estilo, formato, tom, uma tarefa estreita. Serve pouco para ensinar fatos novos.

> 💡 **Pense assim:** Contratar alguém já formado e dar um treinamento interno: ele não aprende a pensar de novo, aprende o jeito da casa.

*Aparece em:* Q1, Q5, Q41, Q42, Q55

### Prompting → RAG → fine-tuning

A escada de escalada: primeiro prompt engineering (rápido e quase de graça), depois RAG (traz conhecimento atualizado ou privado ao prompt) e só então fine-tuning (muda o comportamento). RAG é *retrieval-augmented generation*: buscar trechos de documentos e colocá-los no contexto antes de o modelo responder. Regra de bolso: fine-tuning muda a forma, RAG muda o conhecimento. Antes de tudo, meça um *prompted baseline* (bom prompt, sem treino, no seu eval set): é a régua que o fine-tune precisa vencer.

*Aparece em:* Q1, Q3, Q24, Q41, Q44

### Base model vs instruct model

*Base*: só prevê o próximo token, não segue instruções nem sabe parar sozinho. *Instruct*: já passou por post-training e conversa, segue instruções, tem hábito de terminar a resposta e uma base de segurança. Para trabalho aplicado, a recomendação é começar do instruct.

*Aparece em:* Q2, Q6, Q11, Q47, Q49

### Instruction tuning / SFT

Supervised fine-tuning: treinar a previsão do próximo token em pares curados (prompt, resposta). É o mesmo objetivo do pré-treino, só que com dados diferentes. Quando os exemplos seguem o formato de conversa do modelo, vira o *instruction tuning*.

*Aparece em:* Q10, Q22, Q24, Q35, Q38

### Continued pretraining (CPT)

Continuar o pré-treino em texto bruto do domínio (artigos, contratos, código), sem template de chat e sem mascarar nada. Serve quando o modelo não conhece a *linguagem* da área. Precisa de muito texto (centenas de milhões a bilhões de tokens) e costuma ser seguido de SFT para recuperar o comportamento de chat.

*Aparece em:* Q1, Q5, Q24, Q41, Q45

### Chat template

O formato exato, com marcadores especiais, que separa papéis e turnos da conversa (system, user, assistant). Cada modelo foi treinado com o seu. Usar um template no treino e outro na inferência é a falha silenciosa mais comum do fine-tuning: a loss parece normal e as respostas pioram.

*Aparece em:* Q2, Q34, Q41, Q43, Q49

### Special tokens e EOS

Tokens reservados que funcionam como pontuação estrutural, como o que marca o início de um turno ou o fim da resposta (*EOS*, end of sequence). Se o EOS não estiver nos alvos de treino, o modelo não aprende a parar. Cuidado também com `pad_token = eos_token`: sem mascarar o padding nos labels, o modelo aprende a parar cedo ou a nunca parar.

*Aparece em:* Q2, Q28, Q41, Q43, Q46

### Loss masking (-100)

Calcular a loss só sobre os tokens da resposta, marcando o prompt com o label `-100` (que o PyTorch ignora). Assim o modelo aprende a *responder*, não a reproduzir prompts. O prompt continua sendo lido pela attention; só não gera loss.

*Aparece em:* Q3, Q24, Q43, Q46, Q52

### Sequence packing

Concatenar vários exemplos curtos em uma sequência do tamanho máximo, para não desperdiçar conta com padding. Dá ganho grande de throughput, mas exige isolar os exemplos entre si (ver block-diagonal attention).

> 💡 **Pense assim:** Em vez de mandar uma caixa grande quase vazia por pedido, encher cada caixa com vários pedidos pequenos.

*Aparece em:* Q3, Q8, Q46

### Block-diagonal attention e position IDs

Na sequência empacotada, a máscara faz cada token olhar só para o próprio exemplo, e os position IDs voltam a 0 a cada exemplo. Sem isso, há *contaminação de attention*: o exemplo 3 enxerga os exemplos 1 e 2. As FlashAttention varlen kernels fazem isso sem montar a máscara gigante.

*Aparece em:* Q8, Q46

### Catastrophic forgetting

Ao otimizar numa distribuição estreita, o modelo perde capacidades que já tinha (raciocínio geral, outros idiomas, segurança). O gradiente não tem lealdade a habilidades antigas. Só é visível se você medir.

*Aparece em:* Q7, Q43, Q44, Q45, Q56

### Data mixing / replay

Misturar dados gerais (tipicamente 5 a 25% em SFT) aos dados da tarefa, para que os comportamentos antigos continuem recebendo gradiente. É a alavanca mais eficaz contra forgetting. A mistura deve ser intercalada, não em fases sequenciais.

*Aparece em:* Q7, Q24, Q45, Q55, Q56

### Overfitting em LLMs

Sinais: eval loss sobe enquanto a train loss cai, o modelo recita exemplos de treino (regurgitation), as saídas ficam todas parecidas (diversity collapse) e tudo vira o formato do fine-tune. Meça sempre num *held-out set* (nunca visto no treino, refletindo o tráfego real). Costuma vir de épocas demais, duplicatas, LR alto ou dados pouco diversos.

*Aparece em:* Q9, Q41, Q43, Q50, Q53

## 2. Dados: curadoria, deduplicação e dados sintéticos

### LIMA

Resultado clássico mostrando que cerca de 1.000 exemplos meticulosamente curados bastaram para alinhar um modelo base de 65B. A lição: qualidade e diversidade pesam mais que volume, porque o SFT só ensina qual parte do conhecimento mostrar e em que formato.

*Aparece em:* Q4

### Deduplicação (exata, MinHash, semântica)

Remover exemplos repetidos. Cadeia usual: hash do texto normalizado (duplicatas exatas), MinHash com LSH (quase-duplicatas) e embeddings (paráfrases). Duplicatas viram épocas extras escondidas e viciam a métrica. Deduplique antes de dividir treino e teste.

*Aparece em:* Q4, Q12, Q25, Q44, Q49

### MinHash, LSH e Jaccard

*Jaccard*: o quanto dois conjuntos se sobrepõem (interseção dividida pela união). *MinHash*: resumo compacto de um texto (feito de shingles, trechos de n palavras) que estima o Jaccard rápido. *LSH*: agrupa resumos parecidos em baldes, evitando comparar todos os pares.

*Aparece em:* Q4, Q12

### Decontamination e contamination

Contaminação é quando exemplos do eval vazam para o treino (direto, ou via um teacher que viu o benchmark), tornando as métricas falsas. Descontaminar é checar a sobreposição (n-gram, embeddings) contra todos os evals e removê-la. O *n-gram overlap* (sequências de n tokens iguais; 13-gram é convenção) acha cópias literais, mas não paráfrases, por isso se usa também busca por embeddings.

*Aparece em:* Q12, Q23, Q25, Q46, Q49

### LLM-as-judge

Usar um LLM com uma rubrica explícita para dar nota ou comparar respostas. Barato e escalável, mas tem vieses (prefere respostas longas, a primeira posição, o próprio estilo). Por isso se calibra contra humanos.

*Aparece em:* Q23, Q25, Q31, Q32, Q39

### Seleção por qualidade, dificuldade e diversidade (IFD)

Para escolher 20k de 500k exemplos: filtros duros, nota de qualidade, dificuldade (IFD mede o quanto ter a instrução na frente reduz a loss da resposta; exemplos que o modelo já acerta pouco ensinam) e diversidade (clusterizar e limitar por cluster). Sempre compare contra uma amostra aleatória.

*Aparece em:* Q25, Q31, Q50, Q52

### Synthetic data (self-instruct, evol-instruct)

Dados de treino gerados por um LLM. *Self-instruct*: a partir de tarefas-semente, o LLM cria variações de instruções e respostas. *Evol-instruct*: reescrever prompts para ficarem mais difíceis. Riscos: pouca diversidade, erros do teacher virando verdade, viés herdado e restrições de licença.

*Aparece em:* Q25, Q26, Q49, Q51

### Rejection sampling

Gerar k respostas por prompt e guardar só as que passam num verificador (testes, checagem de resposta) ou num judge. É o cavalo de batalha para dados de raciocínio e código. Também chamado de rejection fine-tuning quando você treina nelas.

*Aparece em:* Q25, Q34, Q38, Q51

### Mode collapse e error amplification

Em dados sintéticos: *mode collapse* é a diversidade encolher (mesmas aberturas, mesma estrutura, os tiques do teacher amplificados). *Error amplification*: os erros do teacher viram verdade absoluta para o aluno, e em ciclos de auto-treino se acumulam.

*Aparece em:* Q9, Q25, Q40

### Distillation (black-box vs logit)

Passar a capacidade de um modelo forte (teacher) para um menor (student). *Black-box*: só se usa o texto gerado pelo teacher e faz SFT nele (o caso mais comum, funciona via API). *Logit (white-box)*: o aluno imita as distribuições completas de probabilidade do teacher, com mais sinal por token, mas exige pesos ou logprobs e tokenizers compatíveis. O *teacher* é o modelo forte e o *student* o menor que aprende. A logit distillation rende mais sinal por exemplo por transmitir o *dark knowledge*: a informação nas probabilidades dos tokens errados mas plausíveis.

*Aparece em:* Q25, Q26, Q34, Q49, Q51

### On-policy distillation e exposure bias

No SFT offline o aluno só vê a distribuição do teacher, então seus próprios erros típicos nunca são corrigidos (*exposure bias*). Na on-policy distillation o aluno gera as respostas e o teacher as avalia token a token (por exemplo, com um reverse KL por token) ou como judge.

*Aparece em:* Q51

### Licenciamento e ToS

Muitos provedores de API proíbem usar as saídas deles para treinar modelos concorrentes, e licenças de pesos abertos (como a do Llama) impõem condições a derivados. Isso é uma barreira jurídica, não um detalhe de engenharia: revise antes de montar o pipeline.

*Aparece em:* Q25, Q26, Q51

## 3. Hiperparâmetros e memória de treino

### Learning rate, warmup e schedule

O learning rate é o tamanho do passo de cada atualização, o botão mais sensível. Típico: 1e-5 a 5e-5 em full FT e 1e-4 a 2e-4 em LoRA. *Warmup* sobe o LR devagar no começo (as estatísticas do Adam ainda são ruins) e *cosine* o reduz suavemente depois.

*Aparece em:* Q8, Q12, Q24, Q35, Q44

### Gradient accumulation

Somar os gradientes de vários micro-batches pequenos antes de dar um passo do otimizador, simulando um batch grande sem estourar a memória das activations (os valores intermediários do forward, que escalam com batch e comprimento). O *effective batch size* é micro-batch × acumulação × GPUs, tipicamente 16 a 128.

*Aparece em:* Q8, Q27, Q36

### Gradient checkpointing

Em vez de guardar todas as activations para o backward, guardar só algumas e recalcular o resto. Custa cerca de 30% mais compute e economiza muita memória.

> 💡 **Pense assim:** Anotar só os pontos-chave de uma conta e refazer o resto quando precisar, em vez de guardar o rascunho inteiro.

*Aparece em:* Q14, Q27, Q36

### Mixed precision (bf16) e master weights

Treinar com números de 16 bits (bf16) para economizar memória e tempo, mantendo uma cópia dos pesos em 32 bits (fp32, os *master weights*) para que atualizações pequenas não se percam.

*Aparece em:* Q14, Q16, Q27, Q36, Q54

### Adam e seus estados (m, v)

O otimizador mais usado. Para cada parâmetro guarda duas médias móveis em fp32 (m, a do gradiente, e v, a do gradiente ao quadrado). É o principal consumidor de memória do treino: mais que o próprio modelo.

*Aparece em:* Q6, Q8, Q14, Q27, Q36

### A conta dos 16 bytes por parâmetro

Full fine-tuning com Adam em precisão mista custa: 2 (pesos bf16) + 2 (gradientes) + 4 (master fp32) + 4 (m) + 4 (v) = 16 bytes/param. Um modelo de 7B dá ~112 GB só de estado estático, antes das activations; por isso não cabe numa placa de 24 GB.

*Aparece em:* Q5, Q14, Q27, Q36

### DDP, ZeRO e FSDP

Na DDP cada GPU guarda uma cópia completa de pesos, gradientes e estados. O ZeRO divide (shard) isso entre as GPUs: estágio 1 os estados do otimizador, 2 também os gradientes, 3 também os parâmetros. O FSDP é a versão nativa do PyTorch do estilo ZeRO-3 (DeepSpeed é a outra stack). Quanto mais estágios, mais comunicação.

*Aparece em:* Q6, Q36, Q37, Q38, Q50

### 8-bit optimizers e offload

Quantizar os estados m e v do Adam para 8 bits (cortando de 8 para ~2 bytes/param) ou jogar estados para a RAM da CPU ou NVMe, trocando velocidade por memória.

*Aparece em:* Q27, Q36

## 4. PEFT: LoRA, QLoRA e variantes

### PEFT (parameter-efficient fine-tuning)

Famílias de técnicas que congelam o modelo base e treinam só um pequeno conjunto de parâmetros adicionais (cerca de 0,1 a 1%). É o padrão prático: menos memória, menos forgetting e adapters pequenos de guardar e versionar.

*Aparece em:* Q5, Q7, Q15

### LoRA

Congela o peso W e aprende uma atualização de baixo rank: W' = W + (alpha/r)·B·A, com A (r×k) e B (d×r) pequenas. A começa aleatória e B em zero, então o treino parte exatamente do modelo base. A hipótese por trás é que a mudança necessária para uma tarefa tem *intrinsic rank* baixo (vive num subespaço pequeno). A economia real de memória vem de não precisar de gradientes nem estados do Adam para os pesos congelados.

> 💡 **Pense assim:** Em vez de reescrever o livro todo, colar um pequeno adesivo de correção em cada página.

*Aparece em:* Q5, Q6, Q29, Q35, Q56

### Rank (r) e alpha

`r` é a largura do gargalo, ou seja, a capacidade da atualização (típico 8 a 64). `alpha` é uma constante de escala; a atualização é multiplicada por alpha/r, o que desacopla a magnitude do rank. Heurística comum: alpha = r ou 2r.

*Aparece em:* Q5, Q6, Q14, Q29, Q35

### Target modules e modules_to_save

Quais matrizes recebem o adapter. A prática atual é mirar todas as projeções de attention (q, k, v, o) e as matrizes do MLP (gate, up, down). Embeddings e LM head ficam congelados, a não ser que você os liste em `modules_to_save` (necessário ao adicionar tokens novos).

*Aparece em:* Q14, Q28, Q29, Q35, Q44

### Merge de adapter (merge_and_unload)

Somar B·A (com a escala) em W offline, gerando um checkpoint comum sem overhead na inferência. Custo: um artefato do tamanho do modelo por variante, e não dá para destacar o adapter. Melhor mesclar no base em precisão cheia e só então quantizar.

*Aparece em:* Q15, Q30, Q44, Q47, Q56

### Multi-LoRA serving e hot-swapping

Manter *um* modelo base na GPU e vários adapters (dezenas de MB cada) anexados por requisição, com kernels que processam adapters diferentes no mesmo batch (S-LoRA, vLLM). *Hot-swapping*: carregar ou trocar adapters sem reiniciar o servidor; o rollback vira apontar para o ID anterior.

> 💡 **Pense assim:** Uma só impressora com várias fichas de configuração, trocadas a cada pedido.

*Aparece em:* Q5, Q15, Q42, Q47, Q56

### Quantization

Armazenar números com menos bits (por exemplo 4 bits em vez de 16) para economizar memória, aceitando uma pequena perda de precisão.

*Aparece em:* Q15, Q16, Q43, Q47, Q54

### QLoRA

LoRA treinado em bf16 por cima de uma base congelada e quantizada em 4 bits. Os pesos 4-bit são dequantizados bloco a bloco a cada multiplicação e nunca atualizados. Permitiu ajustar um modelo de 65B em uma GPU de 48 GB. Contrapartida: você adapta uma versão levemente degradada da base.

*Aparece em:* Q5, Q14, Q16, Q27, Q47

### NF4 (4-bit NormalFloat)

Um tipo de dado de 4 bits cujos 16 níveis ficam nos quantis de uma distribuição normal, porque os pesos pré-treinados são aproximadamente normais. Assim cada nível é usado mais ou menos igualmente, e ele supera inteiros ou floats de 4 bits.

*Aparece em:* Q14, Q16, Q27, Q47

### Double quantization e paged optimizers

*Double quantization*: quantizar também as constantes de quantização (de fp32 para 8 bits), poupando ~0,37 bit/param. *Paged optimizers*: usar memória unificada da NVIDIA para jogar estados do otimizador na RAM da CPU nos picos, transformando um OOM em lentidão.

*Aparece em:* Q16

### rsLoRA e LoRA+

*rsLoRA* escala por alpha/√r em vez de alpha/r, para ranks altos não serem amortecidos (vale a partir de r ≥ 64). *LoRA+* dá um LR bem maior à matriz B que à A (a sugestão inicial é uma razão em torno de 16×), pois os papéis delas são assimétricos.

*Aparece em:* Q6, Q29

### DoRA, PiSSA e OLoRA

*DoRA* separa o peso em magnitude e direção, aplica LoRA à direção e treina a magnitude à parte, fechando parte da distância para o full FT em rank baixo (cerca de 20 a 30% mais lento). *PiSSA* e *OLoRA* mudam só a inicialização do adapter (componentes principais de W, ou base ortonormal por QR) para convergir mais rápido.

*Aparece em:* Q29

### Embedding resize e tied embeddings

Ao adicionar tokens novos é preciso redimensionar a matriz de embeddings; as linhas novas nascem aleatórias, e o ideal é inicializá-las com a média dos embeddings dos pedaços do token. Se entrada e saída são *tied* (compartilham pesos), você inicializa as duas ao mesmo tempo. Matrizes costumam ter linhas extras de padding (múltiplo de 64 ou 128), que devem ser mascaradas.

*Aparece em:* Q2, Q28

## 5. Model merging

### Task vector e task arithmetic

O *task vector* é a diferença entre o fine-tune e a base (τ = θ_ft − θ_base). Na *task arithmetic* você soma vetores com pesos para somar comportamentos; negar um vetor serve como um botão tosco de esquecimento.

*Aparece em:* Q30

### TIES

Método de merging que primeiro apara os deltas pequenos (fica com ~20% maiores), elege um sinal por parâmetro entre os modelos e faz a média só das entradas que concordam com o sinal eleito. Ataca o conflito de sinais, que é o que mais estraga a média ingênua.

*Aparece em:* Q30

### DARE

Descarta aleatoriamente 80 a 99% das entradas de cada delta e reescala o restante por 1/(1−p). Como os deltas são muito redundantes, quase não machuca e reduz a sobreposição entre modelos. Combina bem com TIES.

*Aparece em:* Q30

### SLERP

Interpolação *esférica* entre exatamente dois modelos, que preserva a norma dos pesos em vez de encolhê-la como a média linear.

*Aparece em:* Q30

### Loss basin, model soups e mergekit

Merging só funciona entre modelos que partem do mesmo ancestral e ficam na mesma bacia de loss (fine-tunes irmãos, checkpoints de um run, LoRAs na mesma base). Entre modelos pré-treinados separadamente falha (os neurônios não são comparáveis, por simetria de permutação). O peso do merge (lambda) se escolhe varrendo contra um eval; `mergekit` é a ferramenta padrão.

*Aparece em:* Q30

## 6. RLHF e modelos de recompensa

### Alignment

Ajustar o modelo para se comportar como as pessoas querem: ser útil, honesto e seguro, seguindo preferências humanas e políticas. É o objetivo do post-training depois do SFT.

*Aparece em:* Q26, Q30, Q39, Q40, Q55

### RLHF

Reinforcement Learning from Human Feedback. A receita do InstructGPT: (1) SFT em demonstrações, (2) treinar um reward model a partir de comparações humanas, (3) otimizar o modelo com PPO contra esse reward, com uma penalidade de KL.

*Aparece em:* Q10, Q18, Q19, Q20, Q33

### Preference data (pairwise)

Dados em que, para um prompt, humanos (ou uma IA) dizem qual de duas respostas é melhor (*chosen* vs *rejected*). Comparar pares é mais consistente do que dar nota absoluta, porque as pessoas não são calibradas igual.

*Aparece em:* Q10, Q17, Q20, Q22, Q48

### Reward model (RM)

Um modelo que dá uma nota (um escalar) a um par (prompt, resposta), em geral o próprio SFT model com uma cabeça escalar no lugar do LM head. Funciona como um *proxy* de preferência humana.

*Aparece em:* Q17, Q18, Q19, Q21, Q33

### Bradley-Terry

Modelo estatístico que diz que a probabilidade de a resposta A vencer B é uma sigmoide da diferença entre suas notas. A loss é −log σ(r_vencedora − r_perdedora): uma regressão logística sobre diferenças. Só as diferenças são identificadas, então a nota absoluta tem um deslocamento arbitrário.

*Aparece em:* Q10, Q17, Q20, Q22, Q33

### Policy e reference model

A *policy* é o próprio LLM visto como quem decide (a ação é emitir um token). O *reference model* é uma cópia congelada (em geral o SFT model) usada como âncora para medir o quanto a policy se afastou.

*Aparece em:* Q18, Q19, Q20, Q21, Q33

### PPO

Proximal Policy Optimization: um método de policy gradient cuja marca é fazer atualizações *conservadoras*, limitando (clipping) o quanto a probabilidade de cada token pode mudar por passo. Assim dá para usar o mesmo lote de gerações caras em vários passos. Em RLHF costuma vir com um critic e GAE.

*Aparece em:* Q10, Q18, Q20, Q21, Q37

### Critic (value network) e GAE

O *critic* é um segundo modelo, do tamanho da policy, que estima o valor esperado de cada ponto da geração. Com o *GAE* (generalized advantage estimation) vira a vantagem de cada token, reduzindo a variância do gradiente. É caro e difícil de manter preciso, o que motivou o GRPO.

*Aparece em:* Q18, Q21, Q37, Q45, Q52

### Penalidade de KL

Divergência de Kullback-Leibler: mede o quanto duas distribuições diferem. Na RLHF, subtrai-se da recompensa β·KL(policy ‖ reference) (implementada por token), para a policy não derivar para textos degenerados que enganam o RM. β alto: o modelo quase não muda. β baixo: risco de hacking. No DPO, o β tem o mesmo papel (valores usuais em varreduras: 0,05 a 0,3).

> 💡 **Pense assim:** Uma coleira: deixa o modelo explorar, mas não o deixa fugir longe demais.

*Aparece em:* Q10, Q18, Q19, Q37, Q50

### Reward hacking

Quando a policy acha comportamentos que aumentam a *recompensa-proxy* sem melhorar (ou piorando) o objetivo real. É a lei de Goodhart: toda medida usada como alvo deixa de ser boa medida. Exemplos: respostas inchadas, bajulação, tom confiante e formatação enfeitada, recusar demais, burlar testes.

*Aparece em:* Q10, Q18, Q19, Q38, Q39

### Length bias e sycophancy

*Length bias*: avaliadores (e portanto o RM) preferem respostas longas, e a policy passa a inflar. *Sycophancy*: concordar e elogiar o usuário, mesmo endossando premissas falsas, porque isso pontua melhor.

*Aparece em:* Q10, Q17, Q19, Q39, Q48

### Over-refusal e alignment tax

*Alignment tax*: capacidade ou utilidade perdida como efeito colateral do alinhamento. A forma mais visível é o *over-refusal*: recusar pedidos legítimos que apenas lembram temas sensíveis (como 'how do I kill a Python process'), porque recusar é uma ação segura para o reward.

*Aparece em:* Q30, Q40, Q41, Q44, Q45

### ORM e PRM (outcome vs process reward model)

O *ORM* dá uma nota à resposta inteira (rótulo barato, mas o crédito é cego: acerto por sorte rende nota máxima). O *PRM* dá uma nota a cada passo do raciocínio (sinal denso, rótulo caro; pode ser gerado por rollouts, como no Math-Shepherd, e brilha em reranking e tree search na inferência).

*Aparece em:* Q13, Q52

### RLAIF e Constitutional AI

*RLAIF*: trocar o anotador humano por um LLM que gera os rótulos de preferência, mais barato e rápido, mas com os vieses do próprio labeller. *Constitutional AI* (Anthropic): o modelo critica e revisa as próprias respostas seguindo uma constituição escrita de princípios e depois aplica RLAIF. Os critérios ficam explícitos e editáveis.

*Aparece em:* Q39

### Inter-annotator agreement (Krippendorff's alpha)

Medida do quanto anotadores diferentes concordam nos mesmos itens. Em preferência subjetiva, alpha em torno de 0,6 a 0,75 é esperado; valores muito baixos pedem rubrica melhor ou mais calibração.

*Aparece em:* Q17, Q23, Q32, Q33

### Position bias e rubrica

*Position bias*: preferir a resposta que aparece primeiro; por isso se randomiza a ordem. *Rubrica*: critérios escritos, por dimensão (correção, instruções, segurança, tom), com prioridade de desempate. Sem ela, anotadores caem no comprimento e no tom confiante.

*Aparece em:* Q17, Q31, Q32, Q38, Q39

### Best-of-N

Amostrar N respostas e ficar com a de maior nota do RM. Serve como baseline sem treino, e a curva de qualidade real × N é um diagnóstico de RM: um RM hackeável melhora e depois piora quando N cresce.

*Aparece em:* Q13, Q21, Q33

## 7. DPO e a família de preferências offline

### DPO

Direct Preference Optimization: treina direto nos pares de preferência com uma loss do tipo classificação, sem reward model, sem RL e sem gerar amostras durante o treino. A ideia-chave é que o objetivo da RLHF com KL tem forma fechada, então a própria policy define uma recompensa implícita, β·log(π/π_ref): quanto a policy aumentou a probabilidade da resposta em relação à referência. A constante que depende só do prompt some na diferença entre chosen e rejected.

*Aparece em:* Q20, Q21, Q22, Q44, Q48

### Offline vs online (on-policy vs off-policy)

*Offline* (DPO): treina num dataset fixo. *Online* (PPO, GRPO): gera amostras novas da policy atual a cada passo e as avalia. O online explora comportamentos que nenhum anotador demonstrou; o offline só repondera o que está nos pares. Dados *off-policy* são aqueles gerados por outro modelo (por exemplo, o SFT) e não pelo atual. O *iterative DPO* amostra da policy atual, rotula com RM ou judge, roda DPO nos pares novos e repete, recuperando boa parte do benefício online.

*Aparece em:* Q21, Q43, Q51, Q53, Q54

### IPO

Variante do DPO que troca a log-sigmoide por uma perda quadrada em direção a uma margem fixa. Evita o overfitting em que o DPO empurra a margem para o infinito, útil quando os rótulos são ruidosos ou quase determinísticos.

*Aparece em:* Q20, Q22

### KTO

Treina com rótulos binários bom/ruim independentes, sem precisar de pares (inspirado na prospect theory). Ideal para feedback de produção como polegares para cima e para baixo.

*Aparece em:* Q22

### ORPO

Funde SFT e preferência em uma única loss: NLL da resposta escolhida mais uma penalidade de razão de chances (odds ratio) para a rejeitada, sem reference model. É a receita mais simples, ao custo de perder a âncora que limita a deriva.

*Aparece em:* Q22, Q48

### SimPO e length-normalized objectives

Objetivos que normalizam por comprimento (SimPO) ou avaliações que controlam o tamanho (*length-controlled win rate*), para não confundir 'mais longo' com 'melhor'.

*Aparece em:* Q19, Q48

### Likelihood displacement

No DPO só a margem importa, então a probabilidade absoluta da resposta chosen também pode cair, e a massa vai para tokens não relacionados (hedging genérico). Correções: aumentar β, fazer SFT antes nas chosen, ou adicionar um termo de SFT auxiliar.

*Aparece em:* Q26, Q48

## 8. RL para raciocínio e agentes

### GRPO

Group Relative Policy Optimization: o PPO *sem o critic*. Para cada prompt amostra um grupo de G respostas e usa a recompensa de cada uma, normalizada pela média e pelo desvio padrão do grupo, como vantagem. A média do grupo faz o papel do baseline do critic. Foi introduzido no DeepSeekMath e escalado no DeepSeek-R1.

> 💡 **Pense assim:** Uma prova em que sua nota é medida em relação à média da turma, não em relação a uma régua externa.

*Aparece em:* Q37, Q42, Q50, Q52, Q54

### Advantage

Quanto uma ação (ou resposta) foi melhor ou pior do que o esperado. Vantagem positiva reforça a resposta; negativa a desencoraja. No GRPO, vantagem zero significa que a resposta não ensina nada.

*Aparece em:* Q18, Q37, Q50, Q52, Q54

### Degenerate groups

Grupos em que todas as respostas recebem a mesma nota (todas passam ou todas falham): a vantagem é zero e o gradiente some. Filtrar esses prompts evita gastar compute à toa.

*Aparece em:* Q28, Q37, Q50, Q52

### RLVR (verifiable rewards)

RL com recompensas verificáveis: um programa (testes unitários, checagem de resposta de matemática) decide o sucesso, sem reward model aprendido. Daí menos reward hacking, rótulos baratos e sinal limpo, embora um verificador fraco ainda possa ser burlado.

*Aparece em:* Q13, Q25, Q38, Q50, Q53

### R1-Zero e cold-start SFT

O *R1-Zero* aplicou GRPO com recompensas por regras direto num modelo base, sem SFT, e o raciocínio longo *emergiu*, mas com problemas de legibilidade e mistura de idiomas. O R1 completo adiciona um *cold-start SFT* curto em chain-of-thought curado, alterna RL e rejection sampling, e depois foi destilado em modelos menores.

*Aparece em:* Q15, Q21, Q26, Q37, Q38

### Dr. GRPO, DAPO e GSPO

Variantes que corrigem fraquezas do GRPO. *Dr. GRPO* remove os vieses de comprimento e de normalização por desvio padrão. *DAPO* desacopla o clip range e filtra grupos degenerados (e, em algumas receitas, tira o KL). *GSPO* faz o clipping no nível da sequência, para estabilidade em modelos grandes e MoE.

*Aparece em:* Q37, Q50, Q56

### Entropy collapse

A entropia (diversidade) das respostas da policy despenca: todas as amostras viram quase iguais. No GRPO isso zera as vantagens e o treino só afia um modo. Diagnóstico: plotar a entropia por passo; remédios: temperatura maior, bônus de entropia, filtrar grupos degenerados.

*Aparece em:* Q3, Q26, Q40, Q50, Q52

### Chain-of-thought e thinking block

O *chain-of-thought* é o raciocínio passo a passo escrito antes da resposta. Em modelos de raciocínio ele fica num *thinking block*. Treinar só em (prompt, resposta final) faz o modelo parar de pensar; os traces devem entrar nos alvos, no mesmo formato do template.

*Aparece em:* Q26, Q34, Q38, Q42, Q54

### Test-time compute

Gastar mais computação na inferência (mais tokens de raciocínio, várias amostras, busca) para melhorar a resposta. O RL ensina o modelo a usar esse tempo extra de forma produtiva.

*Aparece em:* Q13, Q38

### Rollout, episódio e trajetória

Um *rollout* (ou episódio, trajetória) é uma tentativa completa: o modelo gera a resposta (ou conduz um agente em vários turnos) e recebe a recompensa no fim. Em agentes, isso acontece num *sandbox* (ambiente simulado com ferramentas e APIs mockadas), que precisa ser resetável, paralelo e determinístico: ruído do ambiente é indistinguível de uma ação ruim.

*Aparece em:* Q13, Q35, Q50, Q52, Q54

### Credit assignment e sparse reward

Descobrir *qual* parte da trajetória causou o sucesso ou a falha. Com recompensa esparsa (só no fim), é difícil: uma nota única é compartilhada por todos os passos. Soluções: grupos grandes, PRM ou marcos programáticos pequenos (*reward shaping*, que ajuda mas é fácil de burlar e pode dominar; registre cada componente separadamente).

*Aparece em:* Q13, Q37, Q38, Q50, Q52

### Curriculum (filtro de dificuldade)

Treinar na faixa de tarefas que a policy às vezes resolve: as sempre resolvidas e as nunca resolvidas não dão gradiente. Refiltrar conforme o modelo melhora.

*Aparece em:* Q37, Q50, Q52

### Training-inference mismatch

O motor de inferência que amostra (probabilidade μ) e o trainer (probabilidade π) usam kernels e precisões diferentes, então dão probabilidades diferentes para o mesmo token mesmo com pesos idênticos. O RL 'on-policy' fica silenciosamente off-policy. Mede-se a diferença por token e corrige-se com pesos de importância truncados π_old/μ, precisão alinhada e limite de defasagem em pipelines assíncronos.

*Aparece em:* Q35, Q41, Q43, Q54, Q56

### Emergent misalignment e safety erosion

Mesmo fine-tunes em dados benignos enfraquecem as recusas do modelo (a segurança é um comportamento raso, concentrado nos primeiros tokens). No *emergent misalignment*, treinar em uma tarefa estreita (código inseguro) gerou respostas desalinhadas em prompts não relacionados. Defesa: evals de segurança a cada fine-tune e mistura de dados de segurança.

*Aparece em:* Q23, Q40, Q44, Q45, Q55

## 9. MoE, avaliação, produção e ferramentas

### MoE (Mixture of Experts) e router

Modelo com vários 'especialistas' (MLPs) em que um *router* aprendido escolhe quais atendem cada token. A sparsity poupa FLOPs, não memória: todos os parâmetros precisam caber. No fine-tuning, costuma-se congelar o router ou manter a loss de balanceamento e monitorar a carga por expert.

*Aparece em:* Q34, Q35, Q37, Q54, Q56

### Expert parallelism e routing replay

*Expert parallelism*: distribuir os experts entre GPUs, com comunicação all-to-all. *Routing replay*: reaproveitar no trainer as decisões de roteamento do sampler, para reduzir o ruído na razão de probabilidades em RL.

*Aparece em:* Q56

### Eval antes/depois e regression suite

Avaliar sempre em duas direções: ganhos na tarefa-alvo (com fatias por tipo de caso) e uma suíte de regressão de capacidade geral, segurança, formatação e contexto longo. Fine-tuning é onde forgetting e over-refusal entram em silêncio.

*Aparece em:* Q7, Q23, Q40, Q45, Q55

### Benchmarks citados (MMLU, XSTest, LiveBench)

*MMLU*: perguntas de conhecimento geral de múltipla escolha. *XSTest*: suíte de over-refusal. *LiveBench* e *LiveCodeBench*: benchmarks renovados continuamente, que ajudam a evitar contaminação. *RewardBench* mede a acurácia de reward models por categoria (chat, segurança, raciocínio): serve de triagem, não de critério final.

*Aparece em:* Q23, Q33, Q40, Q45, Q49

### Shadow deployment, canary e A/B

Formas de validar em produção: *shadow* (o modelo novo roda em paralelo sem afetar o usuário), *canary* (uma fatia pequena do tráfego, que vai crescendo) e *A/B* (comparação com métricas online). Contaminação não consegue falsificar um A/B real.

*Aparece em:* Q23, Q44, Q47, Q49, Q53

### Hosted fine-tuning vs DIY

*Hosted*: o provedor treina e serve (rápido, sem infra, mas seus dados saem, poucos botões e sem posse dos pesos de bases proprietárias). *DIY*: pesos abertos na sua infra, liberdade total de método e dados na sua rede, mas você assume tudo. O *RFT* é fine-tuning por reforço contra um grader que você define. Há também APIs gerenciadas (como o Tinker) em que você escreve o loop de treino.

*Aparece em:* Q11, Q26, Q42, Q55

### Ferramentas citadas (TRL, PEFT, Axolotl, vLLM, bitsandbytes)

*TRL* e *PEFT* (Hugging Face) e *Axolotl* são stacks de post-training open source. *vLLM* e *SGLang* são motores de inferência. *bitsandbytes* oferece quantização e otimizadores de 8 bits.

*Aparece em:* Q8, Q42, Q44, Q46, Q47
