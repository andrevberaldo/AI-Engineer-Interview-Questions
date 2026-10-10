# 🧠 ML & Deep Learning Foundations

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Toda entrevista de AI Engineer - em frontier lab, big tech ou startup - ainda começa pelos fundamentos: os entrevistadores os usam para separar quem *opera* modelos de quem os *entende*. Espere essas perguntas em phone screens, como aquecimento antes de system design e como sondagem de profundidade quando você afirma ter experiência com LLMs ("você fez fine-tuning de um modelo - por que AdamW? por que warmup?"). Você não precisa de matemática de nível de pesquisa, mas precisa explicar esses conceitos com clareza e conectá-los ao treinamento moderno de transformers.

## Crash course

### Generalização: bias, variance e a ressalva moderna

**Bias** é o erro de um modelo simples demais para capturar o sinal; **variance** é o erro de ajustar ruído de um conjunto de treino específico. Sob squared-error loss, o erro esperado de teste se decompõe exatamente como `bias² + variance + ruído irredutível` (para outras losses a decomposição é apenas uma analogia). Diagnostique com o **gap train/val**: erro de treino alto = underfitting (bias); erro de treino baixo mas erro de val alto = overfitting (variance). A ressalva moderna: redes fortemente superparametrizadas violam a clássica curva em U (**double descent**) - além do limiar de interpolação, modelos maiores frequentemente generalizam *melhor*, o que é parte do motivo pelo qual o scaling funciona.

**Kit de regularização** (saiba o que cada um sacrifica):
- **L2 / weight decay** - encolhe os pesos em direção a zero; estimação MAP com um prior Gaussiano.
- **L1** - leva os pesos exatamente a zero (sparsity); prior de Laplace.
- **Dropout** - zera ativações aleatoriamente em tempo de treino (reescalado para que a inferência não precise de mudança); um ensemble implícito. O pretraining de LLMs em larga escala frequentemente o define como 0 porque os dados são efetivamente ilimitados.
- **Early stopping** - pare quando a val loss deixa de melhorar; barato e surpreendentemente próximo do L2 em efeito.
- **Data augmentation** - codifica invariâncias (crops/flips em visão, paráfrase/back-translation em NLP).

### Higiene de dados: splits, cross-validation, leakage

Train ajusta parâmetros, **validation** escolhe hiperparâmetros/checkpoints, **test** é tocado uma única vez. Use **k-fold CV** quando os dados são poucos; use **temporal splits** para qualquer coisa dependente do tempo. **Leakage** é o clássico assassino silencioso - as formas sutis são: ajustar scalers/encoders/vocabulários no dataset inteiro antes de dividir, near-duplicates entre os splits, o mesmo usuário/paciente em train e test (group leakage) e features que codificam o futuro do rótulo. A versão da era dos LLMs é a **benchmark contamination**: dados de eval presentes nos corpora de pretraining.

### Métricas: quando a accuracy mente

Com 1% de positivos, prever sempre "negativo" dá 99% de accuracy. Conheça o vocabulário da matriz de confusão de cor: **precision** = TP/(TP+FP), **recall** = TP/(TP+FN), **F1** = média harmônica. **ROC-AUC** mede a qualidade do ranking (a probabilidade de um positivo aleatório pontuar acima de um negativo aleatório), mas é insensível ao desbalanceamento de classes porque a FPR é normalizada pela enorme classe negativa; **PR-AUC** é a métrica honesta para positivos raros (seu baseline é a prevalência, não 0,5). **Calibration** - se um 0,8 previsto significa 80% - é separada de discriminação; meça com reliability diagrams/ECE e corrija post-hoc com **temperature scaling**.

### A melhoria é real?

Um ganho de 0,7 ponto em alguns milhares de exemplos de teste frequentemente é ruído. Compare modelos **pareados** nos mesmos exemplos (paired bootstrap para um intervalo de confiança da diferença, teste de McNemar para dois classificadores), verifique a variância entre random seeds e confirme que o test set não foi reutilizado silenciosamente para seleção de modelo. Online, um A/B test precisa de uma métrica primária pré-registrada, um cálculo de poder para o tamanho da amostra, métricas de guardrail e nenhum peek-and-stop, a menos que o design seja um teste sequencial apropriado.

### Tree ensembles e dados tabulares

**Random forests** fazem a média de muitas árvores profundas e decorrelacionadas (bootstrap samples mais subconjuntos aleatórios de features): o bagging reduz a variance. **Gradient boosting** (XGBoost, LightGBM, CatBoost) ajusta árvores rasas sequencialmente aos gradientes da loss do ensemble atual: reduz o bias, precisa de learning rate mais early stopping e geralmente vence em accuracy. Em problemas tabulares típicos, os GBDTs continuam sendo o padrão a ser batido: lidam nativamente com features heterogêneas, sem escala e assimétricas e com valores faltantes, ignoram colunas não informativas e são baratos de ajustar (foundation models tabulares como o TabPFN agora são competitivos em datasets pequenos). Árvores não precisam de feature scaling; modelos baseados em distância e gradiente (k-NN, SVMs, modelos lineares, redes neurais) precisam.

### Otimização: SGD → momentum → Adam → AdamW

- **SGD** segue gradientes ruidosos de mini-batch; o ruído serve também como regularização.
- **Momentum** mantém uma EMA dos gradientes - amortece a oscilação em ravinas íngremes, acelera ao longo de direções consistentes.
- **Adam** adiciona um tamanho de passo adaptativo por parâmetro a partir de uma EMA dos gradientes ao quadrado (mais bias correction). Lida com as escalas de gradiente enormemente diferentes entre as camadas de um transformer.
- **AdamW** desacopla o weight decay da atualização adaptativa. No Adam puro, o gradiente de uma penalidade L2 é dividido pela raiz quadrada da estimativa de segundo momento, então pesos com grandes gradientes históricos mal são regularizados; o AdamW aplica o decay diretamente (`w ← w − lr·λ·w`). Este é *o* padrão para transformers (λ típico ≈ 0,1, β₂ ≈ 0,95 para LLMs, sem decay em biases/parâmetros de norm).
- **Muon** é o principal desafiante recente: ele ortogonaliza a atualização de momentum de cada matriz de pesos 2D (via algumas iterações de Newton-Schulz) e já rodou em escala de trilhões de parâmetros (Kimi K2 da Moonshot, como MuonClip). Mesmo em runs com Muon, embeddings, o LM head e os ganhos de norm geralmente ficam no AdamW, então domine o AdamW primeiro.

**Schedules:** **warmup** linear (a estimativa de segundo momento do Adam é lixo nos primeiros passos; transformers profundos podem divergir sem ele), depois **cosine decay** até ~10% do pico - ou um schedule **warmup-stable-decay (WSD)** quando você quer continuar o treinamento a partir de checkpoints intermediários.

### Backprop e patologias de gradiente

Backprop é a regra da cadeia mais programação dinâmica: faça cache das ativações no forward pass e reutilize-as para calcular produtos vetor-Jacobiano no backward (~2× os FLOPs do forward - daí a regra prática de ~6·N·D FLOPs de treinamento). Pilhas profundas multiplicam muitos Jacobianos, então os gradientes **vanish** (ativações saturantes, pesos pequenos) ou **explode**. Correções: conexões residuais (o gradiente flui pelo caminho de identidade), camadas de normalização, ReLU/GELU em vez de sigmoid/tanh, init que preserva a variância (**Xavier** para tanh, **He** = var 2/fan_in para ReLU; transformers tipicamente usam init normal pequeno ~0,02 com projeções residuais reduzidas) e **gradient clipping** (norma global 1,0 é o padrão).

### Mixed precision e loss spikes

Treine em **bf16**: ele mantém os 8 bits de expoente do fp32, então não é necessário loss scaling. **fp16** tem um dynamic range estreito e precisa de dynamic loss scaling para que gradientes pequenos não sofram underflow. De qualquer forma, mantenha master weights e optimizer states em fp32. Runs grandes levam os matmuls ainda mais longe, para **FP8** com escalonamento fino por bloco (o DeepSeek-V3 reportou isso em escala), mantendo operações sensíveis em maior precisão. Quando a loss dá spike, suspeite primeiro do batch de dados e depois das alavancas usuais: peak LR menor ou warmup mais longo, clipping mais apertado, β₂ menor, QK-norm ou uma z-loss nos logits, e voltar a um checkpoint anterior ao spike pulando os batches problemáticos.

### Batch norm vs layer norm

**BatchNorm** normaliza cada feature ao longo do batch - ótimo para convnets, péssimo para transformers: quebra com sequências de comprimento variável, batches pequenos/streaming, decodificação autoregressiva, e precisa de estatísticas sincronizadas entre dispositivos. **LayerNorm** normaliza ao longo das features *dentro de cada token* - independente do batch, funciona com batch size 1. LLMs modernos usam majoritariamente **RMSNorm** (LayerNorm sem a centralização pela média) posicionada como **pre-norm** (antes de cada sublayer), que é marcadamente mais estável de treinar do que o posicionamento post-norm original. Muitos modelos abertos recentes (OLMo 2, Qwen3, Gemma 3) também adicionam **QK-norm**, uma RMSNorm sobre queries e keys antes do produto escalar, para impedir que os logits de attention cresçam sem limite no final do treinamento.

### Losses, softmax, temperature

**Cross-entropy é MLE**: maximizar a verossimilhança de uma distribuição categórica = minimizar a negative log-likelihood = cross-entropy. Seu gradiente através do softmax é lindamente simples - `p − y` - e nunca satura em previsões confiantes e erradas (ao contrário de MSE + sigmoid). **MSE** é MLE sob ruído Gaussiano: certo para regressão, errado para classificação. **InfoNCE** treina modelos de embedding (CLIP, sentence embedders modernos): cross-entropy sobre scores de similaridade em que o par positivo precisa vencer os negativos in-batch, afiada por uma temperature τ.

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

**Cosine** compara apenas a direção; **dot product** também recompensa a magnitude; **Euclidean** em vetores normalizados para norma unitária é monotonicamente equivalente ao cosine (`‖a−b‖² = 2 − 2·cos`). Use a similaridade com a qual o modelo de embedding foi *treinado* (a maioria dos text embedders contrastivos: cosine). A métrica do seu índice ANN (inner product vs L2 vs cosine em FAISS/HNSW) deve corresponder.

### Distribution shift

**Covariate shift** (os inputs mudam), **label/prior shift** (as frequências das classes mudam), **concept drift** (a própria relação input→label muda). Monitoramento em produção: estatísticas das features de input (PSI/KL contra uma janela de referência), detectores de embedding-drift, drift da distribuição de predições e métricas de delayed-label - mais conjuntos de eval golden/canary fixos para apps de LLM, onde uma troca de versão do modelo upstream é, em si, um distribution shift. **Training-serving skew** é o primo autoinfligido: a mesma feature calculada de forma diferente offline e online (code paths separados, joins desatualizados, lookups que vazam valores futuros no treino). Elimine-o por design com um único feature pipeline compartilhado ou feature store, e registre em log as features servidas para poder compará-las com as do treino.

### Como o treinamento moderno de LLMs se mapeia nos enquadramentos clássicos

Pretraining é aprendizado **self-supervised** (rótulos de next-token fabricados a partir dos próprios dados), SFT é aprendizado **supervised** simples, RLHF (PPO contra um reward model aprendido) e RL com recompensas verificáveis (frequentemente com GRPO) são **reinforcement learning**, a otimização de preferências no estilo DPO é uma loss offline derivada do objetivo de RL, e o treinamento contrastivo de embeddings também é self-supervised. Agrupar suas queries de usuários em clusters para encontrar intents? Esse é o raro passo genuinamente **unsupervised** em uma stack moderna.

## Perguntas de entrevista

Todas as 55 perguntas com respostas: [questions.pt-BR.md](questions.pt-BR.md)

## Red flags que os entrevistadores observam

- Recita "bias-variance tradeoff" como definição, mas não consegue *diagnosticar* qual dos dois uma dada curva train/val mostra, nem o que mudar em seguida.
- Diz accuracy, ou usa ROC-AUC por padrão, para um problema com 0,1% de positivos, sem mencionar PR-AUC, precision/recall em um threshold ou custos.
- Afirma que regularização L2 e weight decay são sempre idênticos - sem perceber por que o AdamW existe.
- "Transformers usam layer norm porque funciona melhor" sem nenhum mecanismo (dependência do batch, sequências de comprimento variável, batch de 1 na hora do decode).
- Explica backprop como "a rede aprende com seus erros" - sem regra da cadeia, sem ativações em cache, sem intuição de custo.
- Não sabe que o dropout é desabilitado (e não precisa de reescalonamento com inverted dropout) na inferência, ou que bugs de modo eval vs modo train são uma falha clássica em produção.
- Nunca ouviu falar de data leakage além de "não treine no test" - não consegue citar um exemplo sutil como ajustar o pré-processamento no dataset inteiro ou group leakage.
- Trata temperature como um truque de sampling exclusivo de LLMs, sem conectá-la a softmax, distillation e losses contrastivas.

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
