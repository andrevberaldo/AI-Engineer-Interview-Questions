# 📚 Glossário de estudo — ML & Deep Learning Foundations

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Generalização e regularização](#1-generalização-e-regularização)
- [2. Dados, splits e vazamento](#2-dados-splits-e-vazamento)
- [3. Métricas, calibração e comparação de modelos](#3-métricas-calibração-e-comparação-de-modelos)
- [4. Desbalanceamento, ruído e rótulos](#4-desbalanceamento-ruído-e-rótulos)
- [5. Modelos clássicos, features e dados tabulares](#5-modelos-clássicos-features-e-dados-tabulares)
- [6. Otimização e treino](#6-otimização-e-treino)
- [7. Redes neurais por dentro: backprop, ativações, inicialização e normalização](#7-redes-neurais-por-dentro-backprop-ativações-inicialização-e-normalização)
- [8. Losses e a visão probabilística](#8-losses-e-a-visão-probabilística)
- [9. Embeddings, contrastive learning e redução de dimensão](#9-embeddings-contrastive-learning-e-redução-de-dimensão)
- [10. Clustering, ensembles e explicabilidade](#10-clustering-ensembles-e-explicabilidade)
- [11. Arquiteturas: RNN, LSTM, transformer e attention](#11-arquiteturas-rnn-lstm-transformer-e-attention)
- [12. Produção: skew, drift, experimentos e eficiência](#12-produção-skew-drift-experimentos-e-eficiência)

## 1. Generalização e regularização

### Bias e variance

Duas fontes de erro. *Bias* é o erro de um modelo simples demais, que não consegue representar o padrão real. *Variance* é o erro de um modelo sensível demais à amostra de treino, que acaba aprendendo o ruído. O erro esperado no teste soma os dois mais o ruído irredutível dos próprios dados.

> 💡 **Pense assim:** Bias é um arqueiro que sempre erra para o mesmo lado; variance é um arqueiro cujas flechas se espalham em todas as direções.

*Aparece em:* Q1, Q11, Q25, Q45, Q53

### Underfitting e overfitting

*Underfitting*: o modelo não aprendeu nem o treino (erro alto em tudo). *Overfitting*: o modelo vai muito bem no treino e mal na validação, porque decorou exemplos e ruído em vez do padrão. O diagnóstico vem de comparar o erro de treino com o de validação.

> 💡 **Pense assim:** Underfitting é estudar pouco e não saber nada. Overfitting é decorar as respostas da prova do ano passado e travar quando as perguntas mudam.

*Aparece em:* Q1, Q2, Q18, Q41, Q45

### Regularization

Qualquer técnica que reduz o erro de generalização sem (principalmente) reduzir o erro de treino. Na prática, é um "freio" que impede o modelo de se ajustar demais aos detalhes do treino.

*Aparece em:* Q1, Q2, Q21, Q41, Q44

### L1 e L2 regularization

Penalidades somadas à loss para manter os pesos pequenos. *L2* penaliza o quadrado dos pesos e os encolhe de forma proporcional, quase nunca até zero. *L1* penaliza o valor absoluto e empurra muitos pesos exatamente para zero, o que dá um modelo esparso (sparse) e funciona como seleção de features.

*Aparece em:* Q3, Q12, Q21, Q41, Q44

### Weight decay

Encolher os pesos um pouco a cada passo de treino. Com SGD simples, é o mesmo que L2. Com otimizadores adaptativos como o Adam, os dois deixam de ser equivalentes, e é por isso que existe o AdamW.

*Aparece em:* Q1, Q2, Q3, Q21, Q54

### Dropout

No treino, desliga aleatoriamente uma fração das ativações a cada passo, forçando a rede a não depender de nenhuma unidade isolada. Na inferência, o dropout fica desligado, sem reescalar nada (os frameworks usam *inverted dropout*, que já divide por 1−p no treino). Esquecer `model.eval()` em PyTorch deixa o dropout ligado em produção, um bug clássico. Funciona como treinar um "comitê" de sub-redes que compartilham pesos.

> 💡 **Pense assim:** Um time em que, a cada treino, alguns jogadores faltam por sorteio: todos aprendem a cobrir os outros.

*Aparece em:* Q1, Q4, Q7, Q41, Q53

### Early stopping

Parar o treino quando a loss de validação para de melhorar e ficar com o melhor checkpoint. Barato e muito usado. Limita o quanto os pesos se afastam da inicialização, o que tem efeito parecido com o de L2.

*Aparece em:* Q1, Q11, Q33, Q41, Q45

### Data augmentation

Criar variações dos dados de treino que não mudam o rótulo (recortes e espelhamentos em imagens, paráfrases em texto). Ensina ao modelo invariâncias e costuma ser um dos regularizadores mais fortes.

*Aparece em:* Q1, Q2, Q40, Q41, Q49

### Label smoothing

Em vez de um alvo "100% na classe certa", usa algo como 90% na certa e o resto dividido entre as outras. Evita que o modelo busque confiança infinita e costuma melhorar a calibração, mas atrapalha distillation e normalmente não é usado no pré-treino de LLMs.

*Aparece em:* Q2, Q27, Q33, Q41

### Double descent

Fenômeno em que o erro de teste sobe perto do ponto em que o modelo mal consegue decorar o treino (interpolation threshold) e depois *volta a cair* quando o modelo cresce muito além disso. O ponto crítico é o *interpolation threshold*, quando o modelo tem capacidade de ajustar todo o treino, inclusive o ruído. Quebra a ideia clássica de que modelo grande sempre overfita e é coerente com o fato de que scaling funciona.

*Aparece em:* Q1, Q45

## 2. Dados, splits e vazamento

### Train, validation e test

Três conjuntos com funções diferentes. *Train* ajusta os parâmetros. *Validation* guia todas as escolhas (hiperparâmetros, checkpoint, quando parar). *Test* mede o resultado final, uma única vez. Cada decisão tomada olhando um conjunto "gasta" esse conjunto.

*Aparece em:* Q5, Q30, Q38, Q41, Q47

### Cross-validation (k-fold)

Divide os dados em k partes, treina k vezes usando uma parte diferente como validação a cada vez e tira a média. Dá uma estimativa mais estável quando há poucos dados, mas é caro demais para deep learning em escala e errado para séries temporais.

*Aparece em:* Q5, Q16, Q33, Q35

### Data leakage

Quando informação que não existiria na hora da previsão contamina o treino ou a avaliação. Infla as métricas offline, que desabam em produção. As formas sutis são as mais perigosas: pré-processamento ajustado antes de dividir, mesmo usuário em treino e teste, features que "enxergam o futuro".

> 💡 **Pense assim:** Descobrir o gabarito antes da prova: a nota é ótima, mas não prova que você sabe a matéria.

*Aparece em:* Q16, Q35, Q38, Q47, Q53

### Temporal split e walk-forward validation

Para dados com tempo, treina no passado e valida no futuro, e vai "andando" a janela para a frente, repetindo. Dobras aleatórias deixariam o modelo treinar com o futuro.

*Aparece em:* Q5, Q35, Q47

### Embargo e point-in-time join

*Embargo*: deixar um intervalo entre treino e validação igual ao horizonte do rótulo, para linhas vizinhas não serem quase duplicadas. *Point-in-time join*: ao montar o treino, cada linha só enxerga o que já era conhecido naquele momento (nada de dados corrigidos ou chegados depois).

*Aparece em:* Q31, Q35

### Benchmark contamination

Quando as perguntas de um benchmark já estavam no corpus de pré-treino do modelo, então a nota mede memória e não capacidade. É o data leakage da era dos LLMs. Detecta-se com buscas por trechos repetidos (n-grams) e canary strings.

*Aparece em:* Q5, Q16, Q45

## 3. Métricas, calibração e comparação de modelos

### Precision, recall e F1

*Precision*: de tudo que o modelo marcou como positivo, quanto era positivo de fato. *Recall*: de todos os positivos reais, quantos o modelo achou. *F1*: média harmônica das duas, que pune quando uma é boa e a outra péssima.

> 💡 **Pense assim:** Uma rede de pesca: recall é quantos peixes você pegou; precision é quanto da rede é peixe e não lixo.

*Aparece em:* Q6, Q17, Q19, Q46, Q47

### ROC-AUC

Área sob a curva que cruza taxa de verdadeiros positivos (recall) com taxa de falsos positivos para todos os thresholds. Equivale à probabilidade de um positivo aleatório receber score maior que um negativo aleatório: mede *ranking*. O baseline é 0.5 mesmo com classes desbalanceadas.

*Aparece em:* Q16, Q17, Q19, Q31, Q47

### PR-AUC e average precision

Área sob a curva precision × recall. O baseline é a prevalência da classe positiva. É a métrica honesta quando os positivos são raros, porque os falsos positivos entram direto na precision, sem serem diluídos pela montanha de negativos fáceis. *Precision@k* mede a precision nos k itens mais bem pontuados.

*Aparece em:* Q17, Q19, Q47

### Calibration

Um modelo é calibrado quando as probabilidades batem com a frequência real: entre os casos em que ele diz 80%, cerca de 80% são positivos. É independente de ranking: você pode ordenar perfeitamente e ainda assim ter probabilidades erradas.

*Aparece em:* Q18, Q19, Q37, Q47, Q53

### Reliability diagram e ECE

O *reliability diagram* agrupa as previsões por confiança e compara com a frequência observada em cada grupo (o ideal é a diagonal). O *ECE* resume o desvio numa média. O *Brier score* é o erro quadrático das probabilidades e mistura calibração e discriminação.

*Aparece em:* Q18, Q39

### Temperature scaling, Platt e isotonic

Formas de corrigir calibração depois do treino, ajustadas na validação. *Temperature scaling* divide os logits por um único T (não muda o ranking, difícil de overfitar). *Platt* é uma regressão logística sobre os scores. *Isotonic* é uma curva monótona sem forma fixa, que precisa de mais dados.

*Aparece em:* Q18, Q28, Q47

### Bootstrap e paired bootstrap

Reamostrar os dados com reposição milhares de vezes para estimar a incerteza de uma métrica. No *paired bootstrap*, os dois modelos são avaliados nas mesmas amostras, o que cancela a dificuldade do conjunto e dá um intervalo bem mais apertado para a diferença.

*Aparece em:* Q11, Q30, Q32, Q36, Q53

## 4. Desbalanceamento, ruído e rótulos

### Class imbalance

Quando uma classe é muito mais rara que a outra (por exemplo, 0,5% de fraude). Quebra métricas como accuracy e ROC-AUC e o threshold padrão de 0.5, mas não impede o modelo de aprender um bom ranking.

*Aparece em:* Q6, Q18, Q19, Q47, Q50

### Oversampling, undersampling e SMOTE

*Undersampling* descarta exemplos da classe comum; *oversampling* repete (ou sintetiza) os da classe rara. *SMOTE* cria positivos sintéticos interpolando entre vizinhos no espaço de features. Nunca aplique antes de dividir os dados, ou cópias sintéticas vazam para a validação.

*Aparece em:* Q18, Q19, Q30, Q36, Q47

### Label noise e confident learning

*Label noise* são rótulos errados nos dados. Ruído aleatório é tolerado razoavelmente, mas o sistemático (anotadores errando sempre do mesmo jeito) é aprendido como se fosse sinal. *Confident learning* (ex.: cleanlab) usa a confiança do modelo para apontar os exemplos mais prováveis de estarem mal rotulados.

*Aparece em:* Q33, Q45, Q52, Q53

### Active learning e uncertainty sampling

Estratégia para gastar um orçamento limitado de rótulos: o modelo escolhe quais exemplos pedir. *Uncertainty sampling* pede os que ele considera mais incertos. Na prática, combina-se incerteza com diversidade, e mantém-se à parte um conjunto aleatório para avaliar.

*Aparece em:* Q52

## 5. Modelos clássicos, features e dados tabulares

### Random forest e bagging

*Bagging* treina vários modelos em paralelo, cada um numa amostra bootstrap dos dados, e faz a média. O *random forest* é bagging de árvores profundas, em que cada corte considera só um subconjunto aleatório de features. Reduz a variance e dificilmente overfita ao adicionar árvores.

> 💡 **Pense assim:** Perguntar a opinião de muitos especialistas independentes e tirar a média.

*Aparece em:* Q11, Q53

### Gradient boosting (GBDT)

Treina árvores rasas em sequência, cada uma corrigindo o erro (o gradiente da loss) do conjunto até ali. Reduz o bias, tem melhor accuracy que o random forest em geral, mas overfita se você continuar adicionando rodadas: por isso learning rate e early stopping importam.

> 💡 **Pense assim:** Um aluno que refaz a lista de exercícios focando só nas questões que errou da última vez.

*Aparece em:* Q11, Q14, Q38, Q53

### Feature scaling (standardization e min-max)

*Standardization* subtrai a média e divide pelo desvio padrão. *Min-max* leva os valores para um intervalo fixo, em geral [0, 1], e é sensível a outliers. Modelos baseados em distância, regularizados, PCA e redes neurais precisam; árvores não.

*Aparece em:* Q12, Q29

### Feature engineering

Você mesmo escrever a função que transforma dados brutos em entradas do modelo (contagens, razões, janelas de tempo como "transações nas últimas 24h"). Continua valioso em dados tabulares, onde essas features dão ganhos que capacidade de modelo não inventa.

*Aparece em:* Q14

### Feature store

Sistema que guarda uma única definição de cada feature e a serve tanto ao treino quanto à produção. Existe sobretudo para evitar que duas implementações diferentes da mesma feature divirjam (training-serving skew).

*Aparece em:* Q31

### Grid, random search e Bayesian optimization

Maneiras de procurar hiperparâmetros. *Grid* testa todas as combinações de uma grade (desperdiça em dimensões altas). *Random* sorteia e costuma ser melhor, pois poucos hiperparâmetros realmente importam. *Bayesian optimization* e *successive halving* aprendem com as tentativas ou matam cedo as ruins.

*Aparece em:* Q7

### μP (μTransfer)

Forma de parametrizar o modelo para que os melhores hiperparâmetros encontrados num modelo pequeno valham também para um modelo muito maior, evitando fazer a busca no modelo grande.

*Aparece em:* Q7, Q25

## 6. Otimização e treino

### Batch, mini-batch e SGD

*Batch GD* usa o dataset inteiro por passo (preciso, lento). *SGD* puro usa um exemplo (ruidoso). *Mini-batch* usa um lote: é o que todos fazem, mesmo chamando de "SGD". A variância do gradiente cai com o tamanho do lote.

*Aparece em:* Q8, Q20, Q26, Q40, Q51

### Momentum

Em vez de seguir só o gradiente atual, segue uma média móvel dos gradientes recentes. Amortece as oscilações em vales estreitos e acelera nas direções consistentes.

> 💡 **Pense assim:** Uma bola pesada rolando morro abaixo: ela não muda de rumo a cada pedrinha.

*Aparece em:* Q20, Q21, Q54

### Adam e bias correction

Otimizador que combina momentum com um passo *adaptativo por parâmetro*: divide cada atualização pela raiz da média móvel dos gradientes ao quadrado. Parâmetros com gradientes pequenos ou raros ganham passos relativamente maiores. A *bias correction* compensa o fato de as médias começarem em zero e subestimarem os valores nos primeiros passos.

*Aparece em:* Q3, Q20, Q21, Q22, Q54

### AdamW e decoupled weight decay

Variante do Adam em que o weight decay é aplicado direto nos pesos, fora da conta adaptativa. No Adam com L2, a penalidade é dividida pela raiz do segundo momento, então os pesos com mais gradiente são os menos regularizados. É o otimizador padrão de transformers.

*Aparece em:* Q3, Q7, Q21, Q54

### Muon e Newton-Schulz

Otimizador que trata cada matriz de pesos como matriz: pega o momentum e o "ortogonaliza" (deixa os valores singulares perto de 1) com poucas iterações de *Newton-Schulz*, que usam só multiplicações de matrizes. Em geral só vale para matrizes 2D internas; embeddings, a LM head e normalizações continuam no AdamW. *QK-clip* (no MuonClip) reescala as matrizes de query e key quando os logits de attention crescem demais.

*Aparece em:* Q21, Q54

### Warmup, cosine decay e WSD

*Warmup*: começar com learning rate quase zero e subir até o pico em alguns milhares de passos, porque as estimativas do Adam são instáveis no início. Depois, *cosine decay* reduz o LR suavemente até ~10% do pico. *WSD* mantém o LR constante e só desce no final, o que permite tirar um checkpoint "final" de qualquer ponto do treino.

*Aparece em:* Q7, Q9, Q22, Q40, Q43

### Gradient clipping

Se a norma global do gradiente passa de um limite (tipicamente 1.0), reescala o vetor inteiro para caber nele. Preserva a direção e limita o tamanho do passo; é uma rede de segurança contra explosões.

*Aparece em:* Q24, Q46

### Mixed precision (bf16, fp16, FP8)

Treinar com números de menos bits para ganhar velocidade e memória. *bf16* mantém a mesma faixa de valores do fp32 e dispensa loss scaling. *fp16* tem faixa pequena e precisa de *loss scaling* (multiplicar a loss para os gradientes pequenos não virarem zero). *FP8* vai ainda mais longe, com escalas por bloco. Sempre se guardam os pesos mestres e o estado do otimizador em fp32.

*Aparece em:* Q46, Q51, Q54

### Loss spike e z-loss

*Loss spike* é um salto repentino da loss durante o treino, comum em LLMs grandes. As causas típicas são um batch de dados ruim, otimização instável ou hardware com defeito. O *z-loss* é um termo extra que impede o normalizador do softmax de derivar. O remédio clássico é voltar a um checkpoint anterior e pular o batch problemático.

*Aparece em:* Q24, Q46

### Fine-tuning, linear probe e freezing

*Fine-tuning* é continuar treinando um modelo pré-treinado na sua tarefa. *Freezing* é travar camadas para que não mudem. Um *linear probe* treina só uma camada linear em cima do backbone congelado, e mede o quão boa já é a representação. Usa-se learning rate bem menor que o do pré-treino.

*Aparece em:* Q34, Q37, Q43, Q54, Q55

### PEFT e LoRA

Fine-tuning eficiente: em vez de atualizar todos os pesos, treina um pequeno número de parâmetros extras (adapters), em torno de 1% do modelo. LoRA é a técnica mais popular. Mantém um único modelo base e troca adapters por tarefa, e esquece menos o que já sabia.

*Aparece em:* Q4, Q7, Q34, Q47, Q55

## 7. Redes neurais por dentro: backprop, ativações, inicialização e normalização

### Backpropagation e chain rule

Algoritmo que calcula o gradiente da loss em relação a *todos* os pesos numa única passada para trás, aplicando a regra da cadeia e reaproveitando resultados intermediários (guardados na passada para a frente). Custa cerca de 2× o forward, em vez de uma passada por peso.

> 💡 **Pense assim:** Descobrir de quem foi a culpa por um erro numa fila de pessoas: cada uma só precisa saber a contribuição da anterior, e não refazer a conta toda.

*Aparece em:* Q13, Q23, Q24

### Activation function (sigmoid, ReLU, GELU)

A função não linear aplicada após cada camada. Sem ela, empilhar camadas equivale a uma única conta linear. *Sigmoid* e *tanh* saturam e causam gradientes minúsculos. *ReLU* (max(0, z)) treina bem, mas pode "morrer". *GELU* e *SiLU* são versões suaves, comuns em transformers.

*Aparece em:* Q15, Q24, Q25, Q39

### Vanishing e exploding gradients

O backprop multiplica muitos fatores, um por camada. Se eles são menores que 1, o produto encolhe até zero e as primeiras camadas deixam de aprender (*vanishing*). Se são maiores que 1, ele explode e causa NaNs (*exploding*). Mesma causa, sinais opostos.

*Aparece em:* Q13, Q15, Q22, Q24, Q25

### Residual connection

Em vez de y = f(x), faz y = x + f(x). O termo identidade cria uma "via expressa" para o gradiente passar sem encolher, e permite treinar redes com centenas de camadas. Todo transformer é uma rede residual.

> 💡 **Pense assim:** Um atalho de elevador ao lado da escada: o sinal sempre tem um caminho direto para subir.

*Aparece em:* Q13, Q24, Q25, Q26, Q55

### Weight initialization (Xavier, He)

Escolha dos pesos iniciais. Zero não funciona (simetria: todos os neurônios aprendem igual). A escala também importa: *Xavier* usa variância 2/(fan_in+fan_out), pensada para tanh; *He* usa 2/fan_in, que compensa o ReLU zerar metade das entradas. Transformers costumam usar uma normal pequena (~0.02).

*Aparece em:* Q15, Q24, Q25, Q40, Q41

### BatchNorm

Normaliza cada feature usando estatísticas calculadas *ao longo do batch* e depois aplica uma escala e um deslocamento aprendidos. Funciona bem em convnets, mas depende do tamanho do batch, sofre com sequências de tamanhos variados e exige estatísticas acumuladas na inferência.

*Aparece em:* Q4, Q8, Q26, Q34

### LayerNorm e RMSNorm

A *LayerNorm* normaliza as features *dentro de cada token*, sem depender de outros exemplos, então funciona igual no treino e na inferência. A *RMSNorm* é a versão mais barata: só divide pela raiz da média dos quadrados, sem subtrair a média.

*Aparece em:* Q3, Q21, Q24, Q26, Q46

### Pre-norm vs post-norm

Onde fica a normalização: *antes* de cada sublayer (pre-norm) ou *depois* da soma residual (post-norm). Pre-norm mantém o caminho residual limpo e é bem mais estável, por isso é o padrão. "Sandwich" norm adiciona uma segunda norma na saída do sublayer.

*Aparece em:* Q9, Q22, Q24, Q26, Q46

### QK-norm

Aplicar uma RMSNorm às queries e keys antes do produto escalar da attention, para os logits de attention não crescerem sem limite durante o treino.

*Aparece em:* Q26, Q46, Q54

## 8. Losses e a visão probabilística

### Likelihood e MLE

*Likelihood* é a probabilidade que o modelo atribui aos dados observados, vista como função dos parâmetros. O *MLE* escolhe os parâmetros que a tornam máxima. Na prática, maximiza-se o log da likelihood, que vira uma soma e evita underflow.

*Aparece em:* Q27, Q39, Q44

### Cross-entropy

A loss padrão de classificação e de modelos de linguagem: −log da probabilidade que o modelo deu à resposta certa. Pune muito as respostas erradas dadas com confiança, e o gradiente através do softmax é simplesmente p − y.

*Aparece em:* Q27, Q33, Q39, Q40, Q44

### Perplexity

A exponencial da cross-entropy média por token. Pode ser lida como "entre quantos tokens igualmente prováveis o modelo está escolhendo": quanto menor, melhor.

*Aparece em:* Q27, Q29

### KL divergence

Mede o quanto uma distribuição de probabilidade difere de outra. Minimizar a cross-entropy em relação aos dados equivale a minimizar a KL entre os dados e o modelo. Também aparece como a "coleira" que mantém um modelo de RLHF perto do modelo de referência.

*Aparece em:* Q27, Q31, Q42, Q44

### MAP e priors

*MAP* (maximum a posteriori) é o MLE com uma crença prévia (*prior*) sobre os parâmetros. Um prior Gaussiano nos pesos dá a regularização L2; um prior Laplace dá a L1. Regularizar é, literalmente, acreditar que os pesos devem ser pequenos.

*Aparece em:* Q3, Q41, Q42, Q44, Q49

### MSE, MAE e Huber

*MSE* (erro quadrático médio) é o MLE sob ruído Gaussiano: ideal para regressão. *MAE* (erro absoluto) corresponde a ruído Laplace e é mais robusto a outliers. *Huber* mistura os dois. Para classificação, a cross-entropy é a certa.

*Aparece em:* Q27, Q29, Q39, Q44

### Softmax e temperature

O *softmax* transforma scores brutos (logits) em probabilidades que somam 1. A *temperature* T divide os logits antes: T baixo deixa a distribuição pontuda (tende ao argmax), T alto a deixa achatada (tende ao uniforme). É o mesmo botão em sampling de LLMs, distillation, losses contrastivas e calibração. Até a divisão por √d_k na attention é uma temperatura fixa.

*Aparece em:* Q18, Q27, Q28, Q40, Q47

### Logits e log-sum-exp trick

*Logits* são os scores brutos de saída, antes do softmax. O *log-sum-exp trick* subtrai o maior logit antes de exponenciar: não muda o resultado, mas evita overflow numérico. Em código, use `log_softmax` direto, nunca `log(softmax(z))`.

*Aparece em:* Q18, Q28, Q39, Q46, Q54

### Knowledge distillation

Treinar um modelo pequeno (student) para imitar as saídas de um grande (teacher). Usar a distribuição suavizada do teacher, com T > 1, passa mais informação (a "dark knowledge") do que o rótulo duro.

*Aparece em:* Q40, Q41, Q49, Q51, Q53

## 9. Embeddings, contrastive learning e redução de dimensão

### Embedding

Um vetor denso de números que representa um item (palavra, frase, imagem, usuário) de forma que itens parecidos fiquem próximos no espaço. É produzido por um encoder treinado para isso.

> 💡 **Pense assim:** Coordenadas num mapa: coisas parecidas ficam em bairros vizinhos.

*Aparece em:* Q9, Q29, Q40, Q52, Q54

### Cosine, dot product e Euclidean

Três formas de medir similaridade. *Dot product* recompensa direção e tamanho do vetor. *Cosine* olha só a direção. *Euclidean* é a distância direta. Com vetores normalizados (norma 1), os três dão o mesmo ranking. Use a que o modelo usou no treino.

*Aparece em:* Q9, Q22, Q36, Q40, Q43

### ANN index (FAISS, HNSW)

Índices de busca aproximada de vizinhos mais próximos, que encontram rapidamente os vetores mais parecidos em milhões deles. A métrica configurada no índice (inner product, L2 ou cosine) precisa bater com a do treino.

*Aparece em:* Q9, Q43

### Matryoshka embeddings (MRL)

Embeddings treinados para que cortar só as primeiras k dimensões continue funcionando. Permite trocar um pouco de qualidade por memória e latência.

> 💡 **Pense assim:** Bonecas russas: cada tamanho menor cabe dentro do maior e ainda é uma boneca completa.

*Aparece em:* Q9

### Anisotropy e hubness

*Anisotropy*: os vetores de um modelo de linguagem cru ocupam um cone estreito, então até frases sem relação têm cosine alto. *Hubness*: em dimensões altas, alguns pontos viram "hubs" que aparecem entre os vizinhos de quase toda consulta. Ambos distorcem a busca por similaridade.

*Aparece em:* Q9, Q43

### Contrastive learning e InfoNCE

Treinar um encoder para que pares que combinam (imagem e legenda, pergunta e passagem) tenham similaridade alta e pares que não combinam, baixa, sem precisar de rótulos por exemplo. A *InfoNCE* faz isso como uma cross-entropy: o par certo precisa "ganhar" entre vários negativos.

*Aparece em:* Q9, Q10, Q28, Q40, Q43

### In-batch e hard negatives

*In-batch negatives*: os outros exemplos do mesmo batch servem de exemplos negativos de graça. *Hard negatives*: negativos difíceis, parecidos com o positivo mas errados (achados por BM25 ou por outro modelo), que ensinam distinções finas. *False negatives*: "negativos" que na verdade eram relevantes e atrapalham o treino.

*Aparece em:* Q6, Q40

### Bi-encoder e cross-encoder

O *bi-encoder* codifica consulta e documento separadamente (rápido, serve para a busca inicial). O *cross-encoder* lê os dois juntos e é mais preciso, mas lento demais para toda a base, então reordena (rerank) só os melhores candidatos.

*Aparece em:* Q40, Q43

### Self-supervised, supervised e unsupervised

*Supervised*: aprende de pares (entrada, rótulo) vindos de fora. *Unsupervised*: acha estrutura sem rótulos (clusters). *Self-supervised*: o rótulo é fabricado a partir do próprio dado (prever o próximo token), o que dá dados quase ilimitados.

*Aparece em:* Q10, Q15, Q40, Q41

### SFT, RLHF, GRPO e DPO

Etapas de pós-treino de LLMs. *SFT*: treino supervisionado em pares (prompt, resposta). *RLHF*: reinforcement learning com recompensa de um modelo treinado em preferências humanas (usa PPO). *GRPO*: RL com recompensas verificáveis (matemática, código). *DPO*: aprende de pares de preferência com uma loss direta, sem laço de RL.

*Aparece em:* Q2, Q10, Q18, Q41, Q44

### PCA

Técnica linear de redução de dimensão: acha as direções de maior variância dos dados. É determinística, barata e reversível, por isso entra em pipelines (ex.: encolher embeddings de 1536 para 256). Falha quando uma direção de baixa variância é a que discrimina.

*Aparece em:* Q12, Q16, Q29, Q36

### t-SNE e UMAP

Técnicas não lineares para *visualizar* dados em 2D, preservando vizinhanças. Tamanhos de clusters e distâncias entre clusters não significam muito, e o gráfico muda com os parâmetros e a seed. Use como ferramenta de depuração, não como evidência.

*Aparece em:* Q29, Q36

## 10. Clustering, ensembles e explicabilidade

### k-means e inertia

Algoritmo de clustering que agrupa pontos em k grupos em torno de centros. Assume clusters esféricos e de tamanho parecido, usa distância Euclidiana (logo, precisa de scaling). O "elbow" na *inertia* (soma das distâncias aos centros) é um método fraco para escolher k.

*Aparece em:* Q12, Q36

### DBSCAN, HDBSCAN e agglomerative

Alternativas ao k-means. *DBSCAN/HDBSCAN* descobrem o número de clusters, aceitam formas arbitrárias e marcam outliers como ruído. *Agglomerative* funde pontos passo a passo e gera um dendrograma, adiando a escolha de k.

*Aparece em:* Q36

### Ensemble

Combinar vários modelos para obter um resultado melhor que o de cada um. O ganho depende de os erros dos membros serem pouco correlacionados: dez cópias quase iguais ajudam pouco.

> 💡 **Pense assim:** Um júri: vale mais pela diversidade de pontos de vista do que pelo número de jurados.

*Aparece em:* Q11, Q12, Q38, Q52, Q53

### Self-consistency e best-of-n

Versões de ensemble em LLMs. *Self-consistency* gera várias respostas e escolhe a mais votada. *Best-of-n* gera n e escolhe a melhor segundo um verifier ou reward model. Gastam mais tokens para ganhar qualidade.

*Aparece em:* Q52, Q53

### Feature importance (built-in, permutation)

Maneiras de saber em que features o modelo se apoia. A importância *built-in* (contagem de cortes ou ganho) é rápida e a menos confiável. A *permutation importance* embaralha uma coluna na validação e mede a queda da métrica. Nenhuma delas diz o que *causa* o resultado.

*Aparece em:* Q38, Q47

### SHAP

Divide cada previsão individual em contribuições aditivas por feature, com base em valores de Shapley (teoria dos jogos). Serve para explicações locais, como os "motivos" de uma decisão de crédito. Features correlacionadas dividem o crédito de forma que parece arbitrária.

*Aparece em:* Q38

## 11. Arquiteturas: RNN, LSTM, transformer e attention

### RNN e hidden state

Rede recorrente que lê a sequência passo a passo, carregando um estado oculto de um passo para o próximo. Como multiplica o mesmo fator a cada passo, o gradiente some ou explode em sequências longas. E por ser sequencial, não paraleliza o treino.

*Aparece em:* Q9, Q13, Q24, Q43

### LSTM e GRU

Tipos de RNN com "portões" (forget, input, output) e um estado de célula atualizado por *soma*. Esse caminho aditivo, parecido com uma conexão residual, preserva o gradiente por centenas de passos. A GRU é uma versão mais simples, com dois portões.

*Aparece em:* Q13, Q15, Q24

### Attention e transformer

*Attention* deixa cada posição consultar todas as outras e misturar a informação delas com pesos calculados a partir do conteúdo. O *transformer* é a arquitetura construída em cima disso, sem recorrência: o caminho entre quaisquer dois tokens é curto e o treino vira grandes multiplicações de matrizes que ocupam bem a GPU.

*Aparece em:* Q13, Q20, Q22, Q26, Q49

### Inductive bias

Suposições embutidas na arquitetura sobre como os dados são. A convolução assume localidade; a attention assume pouco e aprende o resto. Um viés forte ajuda quando há poucos dados e atrapalha quando há muitos.

*Aparece em:* Q11, Q49

### O(n²) e sliding-window attention

A attention completa compara todos os pares de tokens, então o custo cresce com o quadrado do tamanho da sequência. *Sliding-window attention* limita cada token a uma janela local para reduzir o custo, reintroduzindo o viés de localidade.

*Aparece em:* Q13, Q35, Q41, Q49

### Interpretability: superposition e SAE

*Superposition* é a hipótese de que a rede guarda mais conceitos (features) do que tem neurônios, usando direções quase ortogonais, o que torna os neurônios *polissemânticos*. Um *sparse autoencoder* (SAE) tenta desfazer isso, aprendendo um dicionário largo e esparso de features candidatas. É útil para gerar hipóteses, mas a reconstrução é imperfeita e, em alguns testes, perde para um simples linear probe.

*Aparece em:* Q55

## 12. Produção: skew, drift, experimentos e eficiência

### Training-serving skew

As features que o modelo vê em produção são calculadas de forma diferente das do treino (código distinto, joins com dados futuros, tratamento diferente de nulos). As métricas offline ficam boas, a produção piora, e nada dá erro. Resolve-se com uma única definição de feature, point-in-time joins e logando as features servidas.

*Aparece em:* Q12, Q31, Q48

### Covariate, label e concept drift

*Covariate shift*: mudam as entradas P(x), a relação entrada→rótulo continua. *Label/prior shift*: muda a frequência das classes P(y). *Concept drift*: muda a própria relação P(y|x), a mesma entrada passa a significar outra coisa.

*Aparece em:* Q42

### PSI, KS e drift monitoring

Estatísticas para comparar a distribuição atual de uma feature com uma de referência (*PSI*, divergência KL, teste *KS*). Fazem parte do monitoramento junto a drift de predição, métricas de proxy, métricas com atraso e golden sets.

*Aparece em:* Q31, Q42, Q46, Q47, Q51

### Label latency e selective labels

Em fraude, os rótulos chegam semanas depois (chargebacks) e só existem para o que foi investigado. Para medir o recall real, deixa-se uma pequena amostra aleatória passar sem ação e avalia-se só em janelas em que os rótulos já "assentaram".

*Aparece em:* Q47

### A/B test, power e MDE

Experimento online que compara dois grupos. Antes de lançar, fixam-se a métrica, o menor efeito que vale detectar (*MDE*), o nível de significância e o *power* (chance de detectar um efeito real), e daí vem o tamanho de amostra necessário.

*Aparece em:* Q15, Q32, Q47, Q48

### Peeking e sequential tests

*Peeking* é olhar o teste repetidas vezes e parar quando p < 0.05, o que infla os falsos positivos. Alternativas corretas: horizonte fixo, limites *group sequential* ou p-valores *always-valid*.

*Aparece em:* Q11, Q13, Q32, Q53

### Sample ratio mismatch (SRM)

Quando a proporção real de usuários em cada grupo difere da configurada. É o primeiro teste de sanidade: indica problema na atribuição e invalida o resto.

*Aparece em:* Q32, Q48

### CUPED

Técnica de redução de variância que remove da métrica a parte explicada por um dado pré-experimento do mesmo usuário. Encurta o teste sem precisar de mais tráfego.

*Aparece em:* Q32

### Uplift e causal inference

Prever quem vai sair (churn) não diz quem pode ser *salvo* por uma ação. *Uplift* mede o efeito do tratamento por pessoa: P(fica | desconto) − P(fica | sem desconto). Dados observacionais têm *confounding* (quem recebeu desconto não foi sorteado), então o ideal é randomizar um grupo de controle.

*Aparece em:* Q38, Q39, Q48, Q50, Q52

### Conformal prediction

Embrulha qualquer modelo e entrega *conjuntos de predição* com cobertura garantida: a resposta certa está dentro do conjunto em pelo menos 1−α dos casos. Exige só que os dados de calibração e de teste sejam parecidos (exchangeable). O tamanho do conjunto indica a dificuldade: singleton, age automático; conjunto grande, manda para um humano.

*Aparece em:* Q37

### Compute-bound vs memory-bound

*Compute-bound*: o gargalo são as contas. *Memory-bound*: o gargalo é ler dados da memória. A geração token a token com batch pequeno é limitada pela memória, e por isso reduzir o tamanho dos pesos (quantização) acelera. O prefill é limitado por cálculo.

*Aparece em:* Q51

### Quantization (PTQ, QAT, GPTQ, AWQ)

Armazenar e calcular com menos bits (ex.: 8 ou 4 em vez de 16). *PTQ* faz isso depois do treino, com um conjunto de calibração. *QAT* inclui a quantização no treino e recupera qualidade em bits muito baixos. GPTQ e AWQ são métodos de PTQ de 4 bits.

*Aparece em:* Q51

### KV cache, continuous batching e paged KV

Técnicas de servir LLMs. O *KV cache* guarda as keys e values já calculadas para não recalcular. *Continuous batching* mistura requisições no mesmo batch à medida que chegam. *Paged KV* guarda o cache em páginas pequenas para desperdiçar menos memória. *Prefix caching* reaproveita um início de prompt repetido.

*Aparece em:* Q13, Q51

### Speculative decoding

Um modelo pequeno (draft) propõe vários tokens e o modelo grande os verifica de uma vez. A regra de aceitar/rejeitar preserva exatamente a distribuição do modelo grande, então não há perda de qualidade.

*Aparece em:* Q51
