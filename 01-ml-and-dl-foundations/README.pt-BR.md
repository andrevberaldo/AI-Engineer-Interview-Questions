# 🧠 ML & Deep Learning Foundations

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

> 📚 Material de apoio: [Glossário de estudo](glossary.pt-BR.md), com explicações simples dos termos técnicos deste módulo.

Toda entrevista de AI Engineer, seja em frontier lab, big tech ou startup, ainda começa pelos fundamentos. Os entrevistadores usam essas perguntas para separar quem *opera* modelos de quem os *entende*. Elas costumam aparecer em phone screens, como aquecimento antes de system design e como teste de profundidade quando você diz ter experiência com LLMs ("você fez fine-tuning de um modelo: por que AdamW? por que warmup?"). Você não precisa de matemática de nível de pesquisa, mas precisa explicar esses conceitos com clareza e ligá-los ao treinamento moderno de transformers.

## Crash course

### Generalização: bias, variance e a ressalva moderna

**Bias** é o erro de um modelo simples demais para capturar o sinal. **Variance** é o erro de se ajustar ao ruído de um conjunto de treino específico. Com squared-error loss, o erro esperado no teste se decompõe exatamente em `bias² + variance + ruído irredutível` (para outras losses, a decomposição é só uma analogia). Para diagnosticar, olhe o **gap train/val**: erro de treino alto indica underfitting (bias); erro de treino baixo com erro de val alto indica overfitting (variance). A ressalva moderna é que redes muito superparametrizadas fogem da clássica curva em U (**double descent**): depois do limiar de interpolação, modelos maiores muitas vezes generalizam *melhor*, e isso explica em parte por que o scaling funciona.

**Kit de regularização** (saiba o que cada técnica sacrifica):
- **L2 / weight decay**: encolhe os pesos em direção a zero; equivale à estimação MAP com um prior Gaussiano.
- **L1**: leva os pesos exatamente a zero (sparsity); equivale a um prior de Laplace.
- **Dropout**: zera ativações aleatórias durante o treino (com reescala, para a inferência não precisar de ajuste); funciona como um ensemble implícito. No pretraining de LLMs em larga escala, ele costuma ser 0, porque os dados são praticamente ilimitados.
- **Early stopping**: pare quando a val loss deixa de melhorar. É barato e, na prática, tem efeito surpreendentemente parecido com o do L2.
- **Data augmentation**: embute invariâncias no modelo (crops e flips em visão; paráfrase e back-translation em NLP).

### Higiene de dados: splits, cross-validation, leakage

O train ajusta os parâmetros, o **validation** escolhe hiperparâmetros e checkpoints, e o **test** é usado uma única vez. Use **k-fold CV** quando houver poucos dados e **temporal splits** para qualquer problema que dependa do tempo. **Leakage** é o grande vilão silencioso. As formas sutis são: ajustar scalers, encoders ou vocabulários no dataset inteiro antes de dividir; near-duplicates entre os splits; o mesmo usuário ou paciente em train e test (group leakage); e features que codificam o futuro do rótulo. Na era dos LLMs, a versão disso é a **benchmark contamination**: dados de eval que já estavam nos corpora de pretraining.

### Métricas: quando a accuracy engana

Com 1% de positivos, prever sempre "negativo" dá 99% de accuracy. Saiba na ponta da língua o vocabulário da confusion matrix: **precision** = TP/(TP+FP), **recall** = TP/(TP+FN), **F1** = média harmônica das duas. O **ROC-AUC** mede a qualidade do ranking (a probabilidade de um positivo aleatório receber score maior que um negativo aleatório), mas quase não percebe o desbalanceamento de classes, porque a FPR é normalizada pela enorme classe negativa. Para positivos raros, a métrica honesta é o **PR-AUC** (cujo baseline é a prevalência, não 0,5). **Calibration**, ou seja, se uma previsão de 0,8 realmente significa 80%, é outra coisa que discriminação. Meça com reliability diagrams e ECE, e corrija depois do treino com **temperature scaling**.

### A melhoria é real?

Um ganho de 0,7 ponto em alguns milhares de exemplos de teste muitas vezes é só ruído. Compare os modelos de forma **pareada**, nos mesmos exemplos (paired bootstrap para o intervalo de confiança da diferença; teste de McNemar para dois classificadores). Verifique também a variância entre random seeds e confirme que o test set não foi reaproveitado, sem ninguém perceber, para escolher o modelo. Em produção, um A/B test precisa de uma métrica primária registrada de antemão, de um cálculo de poder para definir o tamanho da amostra, de métricas de guardrail e de nenhum peek-and-stop (olhar os resultados e parar o teste antes da hora), a menos que o design seja um teste sequencial apropriado.

### Tree ensembles e dados tabulares

**Random forests** tiram a média de muitas árvores profundas e pouco correlacionadas (bootstrap samples mais subconjuntos aleatórios de features). O bagging reduz a variance. **Gradient boosting** (XGBoost, LightGBM, CatBoost) ajusta árvores rasas, em sequência, aos gradientes da loss do ensemble atual. Ele reduz o bias, precisa de learning rate e early stopping, e costuma vencer em accuracy. Em problemas tabulares típicos, os GBDTs continuam sendo o padrão a ser batido: lidam de forma nativa com features heterogêneas, sem escala e com distribuição assimétrica, tratam valores faltantes, ignoram colunas sem informação e são baratos de ajustar (foundation models tabulares, como o TabPFN, já são competitivos em datasets pequenos). Árvores não precisam de feature scaling; modelos baseados em distância ou gradiente (k-NN, SVMs, modelos lineares, redes neurais) precisam.

### Otimização: SGD → momentum → Adam → AdamW

- **SGD** segue gradientes ruidosos de mini-batch. O ruído também funciona como regularização.
- **Momentum** mantém uma EMA dos gradientes: amortece a oscilação em ravinas íngremes e acelera nas direções consistentes.
- **Adam** acrescenta um tamanho de passo adaptativo por parâmetro, calculado a partir de uma EMA dos gradientes ao quadrado (mais bias correction). Isso dá conta das escalas de gradiente muito diferentes entre as camadas de um transformer.
- **AdamW** separa o weight decay da atualização adaptativa. No Adam puro, o gradiente da penalidade L2 é dividido pela raiz quadrada da estimativa do segundo momento, então pesos com gradientes históricos grandes quase não são regularizados. O AdamW aplica o decay diretamente (`w ← w − lr·λ·w`). Ele é *o* padrão para transformers (λ típico ≈ 0,1; β₂ ≈ 0,95 para LLMs; sem decay em biases e parâmetros de norm).
- **Muon** é o principal desafiante recente. Ele ortogonaliza a atualização de momentum de cada matriz de pesos 2D (com algumas iterações de Newton-Schulz) e já foi usado em escala de trilhões de parâmetros (Kimi K2, da Moonshot, com o MuonClip). Mesmo nos runs com Muon, embeddings, LM head e ganhos de norm geralmente continuam no AdamW. Por isso, domine o AdamW primeiro.

**Schedules:** comece com **warmup** linear (a estimativa do segundo momento do Adam não vale nada nos primeiros passos, e transformers profundos podem divergir sem warmup) e siga com **cosine decay** até cerca de 10% do pico. Outra opção é o schedule **warmup-stable-decay (WSD)**, útil quando você quer continuar o treinamento a partir de checkpoints intermediários.

### Backprop e patologias de gradiente

Backprop é a regra da cadeia com programação dinâmica: você guarda em cache as ativações do forward pass e as reutiliza para calcular produtos vetor-Jacobiano no backward (cerca de 2× os FLOPs do forward, daí a regra prática de ~6·N·D FLOPs de treinamento). Pilhas profundas multiplicam muitos Jacobianos, então os gradientes sofrem **vanishing** (ativações que saturam, pesos pequenos) ou **exploding**. As correções são:
- conexões residuais (o gradiente flui pelo caminho de identidade);
- camadas de normalização;
- ReLU/GELU no lugar de sigmoid/tanh;
- inicialização que preserva a variância (**Xavier** para tanh; **He**, com var 2/fan_in, para ReLU; transformers costumam usar init normal pequeno, ~0,02, com projeções residuais reduzidas);
- **gradient clipping** (norma global de 1,0 é o padrão).

### Mixed precision e loss spikes

Treine em **bf16**: ele mantém os 8 bits de expoente do fp32, então não precisa de loss scaling. O **fp16** tem dynamic range estreito e exige dynamic loss scaling para que gradientes pequenos não sofram underflow. Em qualquer caso, mantenha master weights e optimizer states em fp32. Runs grandes vão além e levam os matmuls para **FP8**, com escalonamento fino por bloco (o DeepSeek-V3 reportou isso em escala), deixando as operações sensíveis em precisão maior. Quando a loss dá spike, suspeite primeiro do batch de dados. Depois, tente as alavancas usuais: peak LR menor ou warmup mais longo, clipping mais apertado, β₂ menor, QK-norm ou uma z-loss nos logits, e voltar a um checkpoint anterior ao spike, pulando os batches problemáticos.

### Batch norm vs layer norm

**BatchNorm** normaliza cada feature ao longo do batch. É ótimo para convnets e ruim para transformers: quebra com sequências de comprimento variável, com batches pequenos ou em streaming e na decodificação autoregressiva, e precisa de estatísticas sincronizadas entre dispositivos. **LayerNorm** normaliza ao longo das features *dentro de cada token*: não depende do batch e funciona com batch size 1. Os LLMs modernos usam principalmente **RMSNorm** (LayerNorm sem a centralização pela média), posicionada como **pre-norm** (antes de cada sublayer), que é muito mais estável no treino do que o post-norm original. Vários modelos abertos recentes (OLMo 2, Qwen3, Gemma 3) também adicionam **QK-norm**, uma RMSNorm aplicada às queries e keys antes do produto escalar, para impedir que os logits de attention cresçam sem controle no fim do treinamento.

### Losses, softmax, temperature

**Cross-entropy é MLE**: maximizar a verossimilhança de uma distribuição categórica equivale a minimizar a negative log-likelihood, que é a cross-entropy. O gradiente dela através do softmax é surpreendentemente simples, `p − y`, e não satura em previsões confiantes e erradas (ao contrário de MSE + sigmoid). **MSE** é MLE sob ruído Gaussiano: serve para regressão, não para classificação. **InfoNCE** treina modelos de embedding (CLIP, sentence embedders modernos): é uma cross-entropy sobre scores de similaridade, em que o par positivo precisa vencer os negativos do próprio batch, com a nitidez controlada por uma temperature τ.

```python
import numpy as np

def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max()          # truque log-sum-exp: invariante a deslocamento, evita overflow
    e = np.exp(z)
    return e / e.sum()
# T < 1 afunila (→ argmax quando T→0); T > 1 achata (→ uniforme). Mesmo botão da temperature de sampling de LLMs.
```

### Embeddings e similaridade

**Cosine** compara só a direção; o **dot product** também premia a magnitude; a distância **Euclidean** entre vetores normalizados para norma unitária é monotonicamente equivalente ao cosine (`‖a−b‖² = 2 − 2·cos`). Use a similaridade com a qual o modelo de embedding foi *treinado* (na maioria dos text embedders contrastivos, é o cosine). A métrica do seu índice ANN (inner product, L2 ou cosine no FAISS/HNSW) precisa ser a mesma.

### Distribution shift

**Covariate shift**: os inputs mudam. **Label/prior shift**: mudam as frequências das classes. **Concept drift**: muda a própria relação input→label. Para monitorar em produção, acompanhe estatísticas das features de entrada (PSI/KL contra uma janela de referência), detectores de embedding drift, o drift da distribuição das predições e métricas de delayed label. Em apps de LLM, some a isso conjuntos de eval golden/canary fixos, porque uma troca de versão do modelo upstream já é, por si só, um distribution shift. **Training-serving skew** é o primo autoinfligido: a mesma feature calculada de forma diferente offline e online (code paths separados, joins desatualizados, lookups que vazam valores futuros no treino). Evite o problema já no design, com um único feature pipeline compartilhado ou uma feature store, e registre em log as features servidas para poder compará-las com as do treino.

### Como o treinamento moderno de LLMs se encaixa nos enquadramentos clássicos

O pretraining é aprendizado **self-supervised** (os rótulos de next-token são fabricados a partir dos próprios dados). O SFT é aprendizado **supervised** simples. O RLHF (PPO contra um reward model aprendido) e o RL com recompensas verificáveis (muitas vezes com GRPO) são **reinforcement learning**. A otimização de preferências no estilo DPO é uma loss offline derivada do objetivo de RL, e o treinamento contrastivo de embeddings também é self-supervised. Agrupar as queries dos usuários em clusters para descobrir intents? Esse é o raro passo realmente **unsupervised** de uma stack moderna.

## Perguntas de entrevista

Todas as 55 perguntas com respostas: [questions.pt-BR.md](questions.pt-BR.md)

## Red flags que os entrevistadores observam

- Recita "bias-variance tradeoff" como definição, mas não sabe *diagnosticar* qual dos dois aparece numa curva train/val, nem o que mudar em seguida.
- Responde "accuracy", ou usa ROC-AUC por padrão, num problema com 0,1% de positivos, sem citar PR-AUC, precision/recall em um threshold ou custos.
- Diz que regularização L2 e weight decay são sempre idênticos, sem perceber por que o AdamW existe.
- Afirma que "transformers usam layer norm porque funciona melhor", sem nenhum mecanismo por trás (dependência do batch, sequências de comprimento variável, batch de 1 na hora do decode).
- Explica backprop como "a rede aprende com seus erros", sem regra da cadeia, sem ativações em cache, sem noção de custo.
- Não sabe que o dropout fica desligado na inferência (e que, com inverted dropout, não é preciso reescalar), nem que bugs de modo eval vs modo train são uma falha clássica em produção.
- Só conhece data leakage como "não treinar no test", e não consegue dar um exemplo sutil, como ajustar o pré-processamento no dataset inteiro ou group leakage.
- Vê a temperature como um truque de sampling exclusivo de LLMs, sem ligá-la a softmax, distillation e losses contrastivas.

## Leituras complementares

- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) - a referência canônica para tudo nesta página.
- [A Recipe for Training Neural Networks - Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) - a mentalidade de debugging/overfitting que os entrevistadores querem ouvir.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter; por que transformers usam AdamW.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) - Moonshot AI; o que é preciso para rodar Muon em escala de LLM.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Ba, Kiros, Hinton.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) - Guo et al.; temperature scaling.
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) - a loss InfoNCE.
- [Deep Double Descent](https://arxiv.org/abs/1912.02292) - Nakkiran et al.; a visão moderna de overfitting.
