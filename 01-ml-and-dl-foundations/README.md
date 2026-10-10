# 🧠 ML & Deep Learning Foundations

> 🇧🇷 Disponível em português do Brasil: [README.pt-BR.md](README.pt-BR.md) · [questions.pt-BR.md](questions.pt-BR.md)

Every AI Engineer loop - frontier lab, big tech, or startup - still opens with fundamentals: interviewers use them to separate people who *operate* models from people who *understand* them. Expect these questions in phone screens, as warm-ups before system design, and as depth probes when you claim LLM experience ("you fine-tuned a model - why AdamW? why warmup?"). You don't need research-level math, but you must explain these concepts crisply and connect them to modern transformer training.

## Crash course

### Generalization: bias, variance, and the modern caveat

**Bias** is error from a model too simple to capture the signal; **variance** is error from fitting noise in a particular training set. Under squared-error loss, expected test error decomposes exactly as `bias² + variance + irreducible noise` (for other losses the decomposition is only an analogy). Diagnose with the **train/val gap**: high train error = underfitting (bias); low train error but high val error = overfitting (variance). The modern caveat: heavily overparameterized networks violate the classical U-shaped curve (**double descent**) - past the interpolation threshold, bigger models often generalise *better*, which is part of why scaling works.

**Regularization toolkit** (know what each one trades away):
- **L2 / weight decay** - shrinks weights toward zero; MAP estimation with a Gaussian prior.
- **L1** - drives weights exactly to zero (sparsity); Laplace prior.
- **Dropout** - randomly zeroes activations at train time (rescaled so inference needs no change); an implicit ensemble. Large-scale LLM pretraining often sets it to 0 because data is effectively unlimited.
- **Early stopping** - halt when val loss stops improving; cheap and surprisingly close to L2 in effect.
- **Data augmentation** - encodes invariances (crops/flips in vision, paraphrase/back-translation in NLP).

### Data hygiene: splits, cross-validation, leakage

Train fits parameters, **validation** picks hyperparameters/checkpoints, **test** is touched once. Use **k-fold CV** when data is small; use **temporal splits** for anything time-dependent. **Leakage** is the classic silent killer - the subtle forms are: fitting scalers/encoders/vocabularies on the full dataset before splitting, near-duplicates across splits, the same user/patient in train and test (group leakage), and features that encode the label's future. The LLM-era version is **benchmark contamination**: eval data present in pretraining corpora.

### Metrics: when accuracy lies

With 1% positives, predicting "negative" always gives 99% accuracy. Know the confusion-matrix vocabulary cold: **precision** = TP/(TP+FP), **recall** = TP/(TP+FN), **F1** = harmonic mean. **ROC-AUC** measures ranking quality (probability a random positive scores above a random negative) but is insensitive to class imbalance because FPR is normalized by the huge negative class; **PR-AUC** is the honest metric for rare positives (its baseline is the prevalence, not 0.5). **Calibration** - whether a predicted 0.8 means 80% - is separate from discrimination; measure with reliability diagrams/ECE, fix post-hoc with **temperature scaling**.

### Is the improvement real?

A 0.7-point gain on a few thousand test examples is often noise. Compare models **paired** on the same examples (paired bootstrap for a confidence interval on the difference, McNemar's test for two classifiers), check variance across random seeds, and confirm the test set was not quietly reused for model selection. Online, an A/B test needs a pre-registered primary metric, a power calculation for sample size, guardrail metrics, and no peek-and-stop unless the design is a proper sequential test.

### Tree ensembles and tabular data

**Random forests** average many deep, decorrelated trees (bootstrap samples plus random feature subsets): bagging cuts variance. **Gradient boosting** (XGBoost, LightGBM, CatBoost) fits shallow trees sequentially to the gradients of the current ensemble's loss: it cuts bias, needs a learning rate plus early stopping, and usually wins on accuracy. On typical tabular problems GBDTs remain the default to beat: they handle heterogeneous, unscaled and skewed features and missing values natively, shrug off uninformative columns, and tune cheaply (tabular foundation models such as TabPFN are now competitive on small datasets). Trees need no feature scaling; distance- and gradient-based models (k-NN, SVMs, linear models, neural nets) do.

### Optimization: SGD → momentum → Adam → AdamW

- **SGD** follows noisy mini-batch gradients; noise doubles as regularization.
- **Momentum** keeps an EMA of gradients - damps oscillation across steep ravines, accelerates along consistent directions.
- **Adam** adds a per-parameter adaptive step size from an EMA of squared gradients (plus bias correction). Handles the wildly different gradient scales across transformer layers.
- **AdamW** decouples weight decay from the adaptive update. In plain Adam, an L2 penalty's gradient gets divided by the square root of the second-moment estimate, so weights with large historical gradients are barely regularized; AdamW applies decay directly (`w ← w − lr·λ·w`). This is *the* transformer default (typical λ ≈ 0.1, β₂ ≈ 0.95 for LLMs, no decay on biases/norm params).
- **Muon** is the main recent challenger: it orthogonalises the momentum update of each 2D weight matrix (via a few Newton-Schulz iterations) and has been run at trillion-parameter scale (Moonshot's Kimi K2, as MuonClip). Even in Muon runs, embeddings, the LM head and norm gains usually stay on AdamW, so know AdamW cold first.

**Schedules:** linear **warmup** (Adam's second-moment estimate is garbage for the first steps; deep transformers can diverge without it), then **cosine decay** to ~10% of peak - or a **warmup-stable-decay (WSD)** schedule when you want to keep training from intermediate checkpoints.

### Backprop and gradient pathologies

Backprop is the chain rule plus dynamic programming: cache activations on the forward pass, reuse them to compute vector-Jacobian products backward (~2× the forward FLOPs - hence the ~6·N·D training-FLOPs rule of thumb). Deep stacks multiply many Jacobians, so gradients **vanish** (saturating activations, small weights) or **explode**. Fixes: residual connections (gradient flows through the identity path), normalization layers, ReLU/GELU instead of sigmoid/tanh, variance-preserving init (**Xavier** for tanh, **He** = var 2/fan_in for ReLU; transformers typically use small normal init ~0.02 with scaled-down residual projections), and **gradient clipping** (global norm 1.0 is the standard).

### Mixed precision and loss spikes

Train in **bf16**: it keeps fp32's 8 exponent bits, so no loss scaling is needed. **fp16** has a narrow dynamic range and needs dynamic loss scaling so small gradients don't underflow. Either way, keep fp32 master weights and optimizer states. Large runs push matmuls further to **FP8** with fine-grained per-block scaling (DeepSeek-V3 reported this at scale) while keeping sensitive ops in higher precision. When the loss spikes, suspect the data batch first, then the usual levers: lower peak LR or longer warmup, tighter clipping, lower β₂, QK-norm or a z-loss on logits, and rewinding to a pre-spike checkpoint while skipping the offending batches.

### Batch norm vs layer norm

**BatchNorm** normalizes each feature across the batch - great for convnets, terrible for transformers: it breaks with variable-length sequences, small/streaming batches, autoregressive decoding, and needs synced statistics across devices. **LayerNorm** normalizes across features *within each token* - batch-independent, works at batch size 1. Modern LLMs mostly use **RMSNorm** (LayerNorm minus mean-centring) placed **pre-norm** (before each sublayer), which is markedly more stable to train than the original post-norm placement. Many recent open models (OLMo 2, Qwen3, Gemma 3) also add **QK-norm**, an RMSNorm on queries and keys before the dot product, to stop attention logits growing without bound late in training.

### Losses, softmax, temperature

**Cross-entropy is MLE**: maximising the likelihood of a categorical distribution = minimising negative log-likelihood = cross-entropy. Its gradient through softmax is beautifully simple - `p − y` - and never saturates on confident-wrong predictions (unlike MSE + sigmoid). **MSE** is MLE under Gaussian noise: right for regression, wrong for classification. **InfoNCE** trains embedding models (CLIP, modern sentence embedders): cross-entropy over similarity scores where the positive pair must beat in-batch negatives, sharpened by a temperature τ.

```python
import numpy as np

def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max()          # log-sum-exp trick: shift-invariant, avoids overflow
    e = np.exp(z)
    return e / e.sum()
# T < 1 sharpens (→ argmax as T→0); T > 1 flattens (→ uniform). Same knob as LLM sampling temperature.
```

### Embeddings and similarity

**Cosine** compares direction only; **dot product** also rewards magnitude; **Euclidean** on unit-normalized vectors is monotonically equivalent to cosine (`‖a−b‖² = 2 − 2·cos`). Use whatever similarity the embedding model was *trained* with (most contrastive text embedders: cosine). Your ANN index metric (inner product vs L2 vs cosine in FAISS/HNSW) must match.

### Distribution shift

**Covariate shift** (inputs change), **label/prior shift** (class frequencies change), **concept drift** (the input→label relationship itself changes). Production monitoring: input-feature stats (PSI/KL vs a reference window), embedding-drift detectors, prediction-distribution drift, and delayed-label metrics - plus fixed golden/canary eval sets for LLM apps, where an upstream model version bump is itself a distribution shift. **Training-serving skew** is the self-inflicted cousin: the same feature computed differently offline and online (separate code paths, stale joins, lookups that leak future values in training). Design it out with one shared feature pipeline or feature store, and log served features so you can diff them against training.

### How modern LLM training maps onto classic framings

Pretraining is **self-supervised** learning (next-token labels manufactured from the data itself), SFT is plain **supervised** learning, RLHF (PPO against a learned reward model) and RL on verifiable rewards (often with GRPO) are **reinforcement learning**, DPO-style preference optimisation is an offline loss derived from the RL objective, and contrastive embedding training is self-supervised too. Clustering your user queries to find intents? That's the rare genuinely **unsupervised** step in a modern stack.

## Interview questions

See [questions.md](questions.md) - 55 questions with detailed answers, from basic to advanced.

## Red flags interviewers watch for

- Reciting "bias-variance tradeoff" as a definition but unable to *diagnose* which one a given train/val curve shows, or what to change next.
- Saying accuracy, or defaulting to ROC-AUC, for a 0.1%-positive problem without mentioning PR-AUC, precision/recall at a threshold, or costs.
- Claiming L2 regularization and weight decay are always identical - missing why AdamW exists.
- "Transformers use layer norm because it works better" with no mechanism (batch dependence, variable-length sequences, decode-time batch of 1).
- Explaining backprop as "the network learns from its errors" - no chain rule, no cached activations, no cost intuition.
- Not knowing that dropout is disabled (and needs no rescaling with inverted dropout) at inference, or that eval-mode vs train-mode bugs are a classic production failure.
- Never having heard of data leakage beyond "don't train on test" - can't name a subtle example like fitting preprocessing on the full dataset or group leakage.
- Treating temperature as an LLM-only sampling trick, without connecting it to softmax, distillation, and contrastive losses.

## Further reading

- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) - the canonical reference for everything in this page.
- [A Recipe for Training Neural Networks - Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) - the debugging/overfitting mindset interviewers want to hear.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter; why transformers use AdamW.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) - Moonshot AI; what it takes to run Muon at LLM scale.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Ba, Kiros, Hinton.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) - Guo et al.; temperature scaling.
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) - the InfoNCE loss.
- [Deep Double Descent](https://arxiv.org/abs/1912.02292) - Nakkiran et al.; the modern view of overfitting.
