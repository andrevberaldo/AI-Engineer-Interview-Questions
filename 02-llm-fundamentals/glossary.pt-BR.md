# 📚 Glossário de estudo — LLM Fundamentals

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Blocos básicos: do texto ao próximo token](#1-blocos-básicos-do-texto-ao-próximo-token)
- [2. Attention e suas variantes](#2-attention-e-suas-variantes)
- [3. Posição e contexto longo](#3-posição-e-contexto-longo)
- [4. Tokenização](#4-tokenização)
- [5. Como o modelo escolhe o próximo token](#5-como-o-modelo-escolhe-o-próximo-token)
- [6. Treinamento e pós-treinamento](#6-treinamento-e-pós-treinamento)
- [7. Mixture-of-Experts (MoE) e eficiência](#7-mixture-of-experts-moe-e-eficiência)

## 1. Blocos básicos: do texto ao próximo token

### Token

Pedaço de texto que o modelo realmente lê: pode ser uma palavra, um pedaço de palavra, uma pontuação ou um espaço. O modelo nunca vê letras soltas, só uma sequência de números (os IDs dos tokens).

> 💡 **Pense assim:** Peças de Lego: o texto é desmontado em peças de tamanhos variados, e o modelo só enxerga as peças.

*Aparece em:* Q12, Q13, Q18, Q32, Q35

### Vocabulary (vocab)

A lista fechada de todos os tokens que o modelo conhece (tipicamente de 32 mil a 256 mil). Cada token tem um ID.

*Aparece em:* Q9, Q15, Q20, Q22, Q54

### Embedding

Um vetor (lista de números) que representa um token. Tokens com significado parecido acabam com vetores parecidos. É a forma como o modelo "enxerga" o token por dentro.

> 💡 **Pense assim:** Coordenadas num mapa: palavras parecidas ficam em bairros próximos.

*Aparece em:* Q15, Q22, Q26, Q27, Q54

### Embedding matrix

A tabela de lookup que guarda um embedding para cada token do vocabulário. Entrou o ID do token, sai o vetor.

> 💡 **Pense assim:** Um dicionário em que a chave é o ID e o valor é o vetor.

*Aparece em:* Q9, Q15, Q61

### d_model

O tamanho do vetor que atravessa o modelo inteiro (por exemplo, 4096 números por token). É a "largura" do transformer.

*Aparece em:* Q3, Q6, Q14, Q15, Q54

### Residual stream

O vetor de cada token que passa por todas as camadas. Cada bloco lê esse vetor, calcula algo e *soma* o resultado de volta (conexão residual), em vez de substituí-lo. Isso ajuda o treino de redes profundas e dá um "canal central" por onde a informação circula.

> 💡 **Pense assim:** Uma esteira de fábrica: cada estação (bloco) pode acrescentar peças, mas a esteira continua carregando o que já estava lá.

*Aparece em:* Q2, Q14, Q26, Q48, Q63

### Logits

Os scores brutos que o modelo dá a cada token do vocabulário para ser o próximo. Ainda não são probabilidades: podem ser negativos ou maiores que 1.

*Aparece em:* Q4, Q11, Q49, Q52, Q55

### Softmax

Função que transforma uma lista de scores (logits) em probabilidades que somam 1. Scores maiores ficam com fatias bem maiores.

*Aparece em:* Q4, Q5, Q25, Q35, Q38

### LM head

A última camada do modelo: pega o vetor final de um token e o converte em logits sobre todo o vocabulário ("qual token vem agora?").

*Aparece em:* Q2, Q15, Q22, Q54

### Weight tying

Usar *a mesma matriz* de pesos na entrada (embedding: token → vetor) e na saída (LM head: vetor → token), só que "de cabeça para baixo". As duas matrizes fazem o mesmo tipo de tradução entre tokens e o espaço interno do modelo, então a ideia é aprender uma só. Economiza muitos parâmetros, o que importa bastante em modelos pequenos, onde o vocabulário ocupa uma fatia grande do total.

> 💡 **Pense assim:** Um dicionário Português↔Inglês e um Inglês↔Português: em vez de manter dois livros, você mantém um só e o lê nos dois sentidos.

*Aparece em:* Q15, Q22, Q54

### Autoregressive

Gerar um token por vez, em que cada novo token depende de todos os anteriores (incluindo os que o próprio modelo acabou de gerar).

> 💡 **Pense assim:** Completar uma frase palavra por palavra, sempre relendo o que já escreveu.

*Aparece em:* Q16, Q22, Q32, Q50, Q64

### Next-token prediction

A tarefa de treino dos LLMs: dado o texto até aqui, prever o próximo token. Repetida bilhões de vezes, vira a capacidade de escrever, resumir, programar etc.

*Aparece em:* Q7, Q8, Q34, Q50, Q53

### Decoder block

A unidade que se repete dezenas de vezes num LLM: uma camada de attention (tokens trocam informação) seguida de uma feed-forward (cada token "pensa" sozinho), com normalização e somas residuais.

*Aparece em:* Q2, Q54

### Feed-forward network (FFN / MLP)

A parte do bloco que processa cada token *individualmente*, sem olhar os outros. Concentra a maior parte dos parâmetros do modelo e é onde se acredita que boa parte do "conhecimento" fica guardada.

> 💡 **Pense assim:** A attention é a conversa em grupo; a FFN é cada pessoa pensando sozinha no que ouviu.

*Aparece em:* Q2, Q14, Q48, Q54, Q60

### SwiGLU

Um tipo de FFN com "portão" (gate): uma parte do cálculo decide, número por número, quanto da outra parte passa adiante. Virou padrão porque rende melhor do que a FFN simples com o mesmo custo.

*Aparece em:* Q2, Q14, Q54

### Parâmetros vs FLOPs

Parâmetros são os números aprendidos (o "tamanho" do modelo e quanta memória ele ocupa). FLOPs são o número de operações matemáticas necessárias para processar um token (o "custo de cálculo"). Um modelo pode ter muitos parâmetros e gastar poucos FLOPs (como os MoE).

*Aparece em:* Q24, Q25, Q46, Q48, Q54

### Normalization (LayerNorm / RMSNorm)

Reescala o vetor de cada token para uma magnitude previsível, para que os números não explodam nem sumam ao longo de dezenas de camadas. O RMSNorm é a versão mais simples e barata, que só divide pela magnitude do vetor, sem subtrair a média.

*Aparece em:* Q2, Q4, Q19

### Pre-norm vs post-norm

Onde colocar a normalização: *antes* de cada sublayer (pre-norm) ou *depois* da soma residual (post-norm). Pre-norm deixa o gradiente fluir pelo caminho residual sem obstáculos e treina de forma bem mais estável, por isso virou o padrão.

*Aparece em:* Q2, Q19

### QK-norm

Normalizar Q e K antes de calcular os scores de attention, para os scores não crescerem sem limite durante o treino (o que causaria instabilidade).

*Aparece em:* Q4, Q19, Q38

## 2. Attention e suas variantes

### Self-attention

O mecanismo em que cada token olha para os outros tokens da sequência e decide *quanto de atenção dar a cada um* para montar sua nova representação. É o que permite ligar "ele" ao substantivo certo, mesmo longe.

> 💡 **Pense assim:** Numa reunião, cada pessoa decide a quem prestar mais atenção antes de formar sua opinião.

*Aparece em:* Q1, Q2, Q3, Q7, Q26

### Query, Key, Value (Q, K, V)

Cada token gera três vetores. *Query*: "o que estou procurando?". *Key*: "o que eu tenho a oferecer?". *Value*: "o conteúdo que eu entrego se me escolherem". A attention compara cada Query com todas as Keys (produto escalar) e usa o resultado como peso para misturar os Values.

> 💡 **Pense assim:** Busca numa biblioteca: a query é o que você digita, as keys são os títulos dos livros e os values são o conteúdo dos livros.

*Aparece em:* Q3, Q27, Q30, Q31, Q65

### Scaled dot-product (√d_k)

Os scores de attention são divididos por √d_k. Sem isso, os produtos escalares ficam grandes demais quando os vetores são longos, o softmax satura (um token leva tudo) e o gradiente some.

*Aparece em:* Q2, Q3, Q4

### Attention head / multi-head

Em vez de uma attention só, o modelo roda várias em paralelo (as heads), cada uma com seu próprio Q, K e V em um subespaço menor. Cada head pode se especializar em um tipo de relação (sintaxe, referência, posição...).

> 💡 **Pense assim:** Vários especialistas lendo o mesmo texto, cada um atento a uma coisa.

*Aparece em:* Q6, Q30, Q31, Q37, Q40

### Causal mask

Uma máscara que bloqueia cada token de olhar tokens *futuros*. Necessária no treino de LLMs para o modelo não "colar" a resposta, já que na geração o futuro ainda não existe.

*Aparece em:* Q1, Q3, Q5, Q35, Q44

### Encoder-only / decoder-only / encoder-decoder

Três arquiteturas. *Encoder-only* (BERT): lê o texto inteiro nos dois sentidos, bom para entender/classificar. *Decoder-only* (GPT, Llama, Claude): só olha para o passado e gera texto; é o padrão dos LLMs. *Encoder-decoder* (T5): um encoder lê a entrada e um decoder gera a saída, comum em tradução.

*Aparece em:* Q7, Q8, Q35

### Causal LM vs Masked LM

Dois objetivos de pré-treino. *Causal LM*: prever o próximo token olhando só o passado (GPT). *Masked LM*: esconder tokens no meio do texto e prever os que faltam olhando dos dois lados (BERT).

*Aparece em:* Q5, Q7, Q8

### Prefill vs decode

As duas fases de uma resposta. *Prefill*: o modelo lê o prompt inteiro de uma vez, em paralelo (limitado por cálculo). *Decode*: gera a resposta um token por vez (limitado pela velocidade de ler a memória).

> 💡 **Pense assim:** Prefill é ler a pergunta toda; decode é escrever a resposta palavra por palavra.

*Aparece em:* Q13, Q24, Q30, Q44, Q54

### KV cache

Memória que guarda os vetores K e V de todos os tokens já processados, para não recalculá-los a cada novo token. Acelera muito a geração, mas ocupa bastante memória de GPU, e cresce com o tamanho do contexto.

> 💡 **Pense assim:** Anotações de reunião: em vez de reouvir a gravação inteira a cada pergunta, você consulta suas anotações.

*Aparece em:* Q13, Q37, Q41, Q62, Q65

### Prefix caching

Reaproveitar o KV cache de um início de prompt que se repete entre requisições (por exemplo, o mesmo system prompt), pulando o prefill dessa parte.

*Aparece em:* Q31, Q36, Q41, Q61, Q65

### PagedAttention

Técnica que guarda o KV cache em "páginas" pequenas e não contíguas, como a memória virtual de um sistema operacional, reduzindo desperdício e permitindo atender mais requisições na mesma GPU.

*Aparece em:* Q13, Q31

### MQA e GQA

Formas de encolher o KV cache. Na *MQA* (multi-query), todas as heads compartilham um único par K/V. Na *GQA* (grouped-query), as heads se dividem em grupos e cada grupo compartilha um par K/V. Perde pouca qualidade e economiza muita memória.

*Aparece em:* Q24, Q30, Q31, Q37, Q54

### MLA (Multi-head Latent Attention)

Em vez de guardar K e V completos, guarda uma versão comprimida (um vetor "latente") e reconstrói K e V quando precisa. Reduz o KV cache ainda mais que a GQA.

*Aparece em:* Q24, Q25, Q30, Q36, Q37

### FlashAttention

Um jeito de calcular a attention *exatamente igual*, mas muito mais rápido, porque processa em blocos pequenos (tiles) que cabem na memória rápida da GPU e nunca monta a matriz gigante n×n. O ganho vem de mexer menos na memória, não de aproximar.

*Aparece em:* Q5, Q24, Q25

### Online softmax

Truque que calcula o softmax em pedaços, atualizando o resultado a cada bloco, sem precisar ter todos os scores na memória de uma vez. É o que torna a FlashAttention possível.

*Aparece em:* Q25

### O(n²)

Notação de custo: se você dobra o tamanho do contexto, o custo da attention vira 4×. Por isso contextos longos ficam caros.

*Aparece em:* Q13, Q24, Q25, Q54, Q61

### Compute-bound vs memory-bound

*Compute-bound*: o gargalo é a quantidade de contas (a GPU está ocupada calculando). *Memory-bound*: o gargalo é buscar dados na memória (a GPU espera). O prefill costuma ser compute-bound; o decode, memory-bound.

*Aparece em:* Q13, Q16, Q19, Q24, Q30

### Sliding-window attention

Cada token só olha para uma janela dos últimos N tokens, em vez de todos. Barato, mas perde detalhes distantes; por isso é combinado com camadas de attention global (stacks híbridos local/global).

*Aparece em:* Q24, Q28, Q31, Q35, Q36

### Attention sink

Os primeiros tokens da sequência acabam recebendo muita attention mesmo sem serem importantes: funcionam como "ralo" onde a attention extra é descartada. Se você os remove do KV cache, o modelo se desestabiliza.

*Aparece em:* Q35, Q38, Q44

### Linear attention, SSM e Mamba

Alternativas à attention clássica cujo custo cresce linearmente com o contexto: em vez de olhar todos os tokens anteriores, mantêm um "resumo" de tamanho fixo que é atualizado a cada token. São rápidas, mas têm dificuldade em recuperar detalhes exatos do passado; por isso aparecem em stacks híbridos com camadas de attention.

*Aparece em:* Q1, Q41, Q65

## 3. Posição e contexto longo

### Positional encoding

A attention, sozinha, não sabe a *ordem* dos tokens ("o cão mordeu o homem" e "o homem mordeu o cão" seriam iguais). O positional encoding injeta a informação de posição.

*Aparece em:* Q26, Q28

### Sinusoidal vs learned absolute

Duas formas de posição "absoluta": ondas seno/cosseno fixas (sinusoidal) ou vetores de posição aprendidos no treino (learned). As duas têm dificuldade com sequências mais longas do que as vistas no treino.

*Aparece em:* Q26, Q28

### RoPE

Em vez de somar a posição, gira os vetores Q e K por um ângulo proporcional à posição. O resultado é que o score entre dois tokens passa a depender da *distância relativa* entre eles. É o padrão atual.

> 💡 **Pense assim:** Dois ponteiros de relógio: o que importa é o ângulo *entre* eles, não a hora absoluta de cada um.

*Aparece em:* Q27, Q28, Q29, Q37, Q62

### ALiBi

Em vez de vetores de posição, subtrai dos scores de attention uma penalidade proporcional à distância (quanto mais longe, menos atenção). Extrapola razoavelmente bem para contextos mais longos.

*Aparece em:* Q26, Q28

### NoPE

Modelos sem nenhum encoding de posição explícito: a ordem é inferida a partir da máscara causal.

*Aparece em:* Q27, Q28

### Length extrapolation

Fazer o modelo funcionar com sequências mais longas do que as vistas no treino. Normalmente falha sem ajustes.

*Aparece em:* Q26, Q28, Q47, Q57

### Position interpolation, NTK-aware, YaRN

Truques para "esticar" o RoPE e permitir contextos maiores que o do treino (por exemplo, de 8k para 128k), normalmente seguidos de um fine-tune curto. Em vez de usar posições nunca vistas, "comprimem" as novas posições para dentro do intervalo conhecido.

*Aparece em:* Q27, Q28, Q29, Q62

### Context window

Quantidade máxima de tokens (prompt + resposta) que o modelo consegue considerar de uma vez.

*Aparece em:* Q23, Q29, Q41, Q44, Q54

### Needle-in-a-haystack

Teste de contexto longo: esconder um fato (a "agulha") no meio de um texto enorme e perguntar por ele. Passar nesse teste prova pouco: é uma busca fácil, bem diferente de raciocinar sobre vários trechos ao mesmo tempo.

*Aparece em:* Q29, Q39, Q44

### Lost in the middle

Modelos usam melhor a informação que está no começo e no fim do contexto do que a que está no meio.

*Aparece em:* Q44

## 4. Tokenização

### Subword tokenization

Quebrar o texto em pedaços entre letra e palavra. Palavras comuns viram um token; palavras raras são montadas com vários pedaços. Equilibra vocabulário pequeno e sequências curtas.

*Aparece em:* Q9, Q20

### BPE (Byte-Pair Encoding)

Algoritmo que constrói o vocabulário assim: começa com caracteres e repete "junte o par de símbolos mais frequente num símbolo novo" até chegar ao tamanho de vocabulário desejado.

*Aparece em:* Q9, Q20, Q21, Q61

### WordPiece, SentencePiece, Unigram-LM

Variantes de tokenizer. *WordPiece* (BERT) escolhe junções pela probabilidade; *Unigram-LM* começa grande e vai podando; *SentencePiece* é uma biblioteca que trata o texto bruto (inclusive espaços) sem depender de pré-tokenização por idioma.

*Aparece em:* Q20, Q21

### Byte-level BPE

BPE que opera sobre bytes em vez de caracteres. Nunca encontra um caractere desconhecido (qualquer texto vira bytes), por isso é o padrão em LLMs modernos.

*Aparece em:* Q9, Q20, Q21, Q23, Q61

### Pre-tokenization

Etapa que divide o texto grosseiramente (por espaços, pontuação, números) antes do BPE, para evitar junções esquisitas entre palavras.

*Aparece em:* Q20, Q21

### Glitch / under-trained tokens

Tokens que existem no vocabulário mas quase não apareceram no treino (por exemplo, nomes de usuário de fóruns). Seus embeddings ficam mal treinados e o modelo reage de forma estranha quando os vê.

*Aparece em:* Q22, Q26, Q61

### Tokenizer-free

Modelos que leem bytes ou caracteres diretamente, sem tokenizer. Eliminam os problemas da tokenização, mas as sequências ficam muito mais longas e caras.

*Aparece em:* Q18, Q61

## 5. Como o modelo escolhe o próximo token

### Temperature

Um botão que controla o quanto o modelo aposta no token mais provável. Temperature baixa afunila a distribuição (respostas previsíveis); alta a achata (respostas mais variadas e arriscadas). Na conta, os logits são divididos por T antes do softmax.

*Aparece em:* Q11, Q12, Q29, Q32, Q62

### Greedy decoding

Escolher sempre o token mais provável. Determinístico, mas tende a ficar repetitivo.

*Aparece em:* Q11, Q12, Q32, Q40, Q53

### Top-k / top-p (nucleus)

Formas de cortar candidatos antes de sortear. *Top-k*: fica só com os k tokens mais prováveis. *Top-p*: fica com o menor grupo de tokens cuja probabilidade somada atinge p (por exemplo, 90%), então o número de candidatos se adapta à confiança do modelo.

*Aparece em:* Q11, Q12, Q32, Q33, Q48

### Min-p

Descarta tokens cuja probabilidade seja menor do que uma fração (min-p) da probabilidade do token mais provável. Adapta o corte à confiança do modelo e funciona bem em temperaturas altas.

*Aparece em:* Q11, Q12, Q32

### Repetition / frequency / presence penalty

Penalidades que reduzem a chance de tokens que já apareceram. *Repetition*: qualquer repetição; *frequency*: cresce com o número de vezes; *presence*: pune apenas por já ter aparecido.

*Aparece em:* Q32

### Beam search

Em vez de seguir um único caminho, mantém várias sequências candidatas ao mesmo tempo (os "beams") e fica com a de maior probabilidade total. Bom em tradução, ruim em texto aberto (sai genérico e repetitivo).

*Aparece em:* Q31, Q53

### Logprobs

O logaritmo da probabilidade que o modelo deu a cada token gerado. Úteis em produção para medir confiança, fazer classificação com score e detectar respostas duvidosas.

*Aparece em:* Q33, Q50, Q52, Q53

### Speculative decoding

Um modelo pequeno e rápido (draft) propõe vários tokens de uma vez e o modelo grande verifica todos em paralelo, aceitando os corretos. A saída continua *exatamente* com a mesma distribuição do modelo grande, só que mais rápida.

> 💡 **Pense assim:** Um estagiário rascunha a resposta e o sênior só revisa, em vez de escrever tudo do zero.

*Aparece em:* Q16, Q31, Q40

### Perplexity

Mede o quanto o modelo "se surpreende" com um texto; menor é melhor. Não dá para comparar entre modelos com tokenizers diferentes, porque a conta é por token.

*Aparece em:* Q18, Q33, Q35, Q36, Q41

### Calibration

Um modelo é bem calibrado quando "80% de confiança" realmente significa que ele acerta 80% das vezes. LLMs pós-treinados tendem a ficar superconfiantes.

*Aparece em:* Q12, Q33, Q42, Q50

### Hallucination

Quando o modelo gera algo fluente e confiante, mas falso. Vem de ele ser treinado para produzir texto plausível, não verdadeiro.

*Aparece em:* Q33, Q34, Q50

## 6. Treinamento e pós-treinamento

### Pretraining

A fase inicial e mais cara: o modelo lê trilhões de tokens de texto e aprende a prever o próximo token. Dela vem o conhecimento geral.

*Aparece em:* Q17, Q18, Q29, Q34, Q59

### Mid-training

Uma fase intermediária entre o pretraining e o pós-treino: continua treinando com dados de maior qualidade ou voltados a um objetivo (código, contexto longo, raciocínio).

*Aparece em:* Q29, Q34, Q59

### SFT (Supervised Fine-Tuning)

Ajustar o modelo com exemplos de pares "instrução → boa resposta", para ele aprender a seguir instruções e conversar.

*Aparece em:* Q18, Q34, Q52, Q59, Q62

### RLHF e DPO

Formas de alinhar o modelo às preferências humanas. *RLHF*: treina um reward model com comparações humanas e usa RL para maximizá-lo. *DPO*: aprende diretamente das comparações "resposta A é melhor que B", sem o RL.

*Aparece em:* Q7, Q33, Q34, Q50, Q56

### RLVR e GRPO

*RLVR*: RL com recompensa *verificável* por programa (a resposta de matemática está certa? o código passou nos testes?), sem reward model humano. *GRPO*: algoritmo de RL que compara várias respostas do mesmo prompt entre si, dispensando o critic (rede extra) que o PPO exige.

*Aparece em:* Q34, Q51, Q56

### Reasoning models e test-time compute

Modelos treinados para "pensar" em voz alta antes de responder. Gastar mais tokens de raciocínio na hora da resposta (*test-time compute*) melhora o resultado em problemas difíceis, ao custo de latência e dinheiro. O *thinking budget* ou *reasoning effort* controla quanto ele pode pensar.

*Aparece em:* Q10, Q34, Q42, Q51, Q56

### Distillation

Treinar um modelo menor (*student*) para imitar um maior (*teacher*). Pode imitar só o texto gerado (black-box) ou as probabilidades inteiras (*soft targets* / logits), que carregam mais informação.

*Aparece em:* Q16, Q33, Q46, Q52, Q55

### In-context learning (ICL)

Aprender a tarefa só pelos exemplos colocados no prompt, sem mudar nenhum peso do modelo.

*Aparece em:* Q6, Q8, Q17, Q57

### Induction heads

Um circuito de duas heads que implementa "já vi A seguido de B; agora vi A de novo, então provavelmente vem B". É considerado a base mecânica do in-context learning.

*Aparece em:* Q6, Q17, Q57

### Scaling laws (Kaplan, Chinchilla)

Fórmulas empíricas que prevêem a qualidade do modelo conforme crescem parâmetros (N) e dados (D). Kaplan dizia para crescer sobretudo o modelo; o *Chinchilla* mostrou que parâmetros e dados devem crescer juntos, com cerca de 20 tokens por parâmetro.

*Aparece em:* Q1, Q45, Q46, Q48

### Overtraining / uptrain

Treinar um modelo com muito mais dados do que o ótimo do Chinchilla. Fica mais caro de treinar, mas o modelo resultante é menor e mais barato de rodar para a mesma qualidade, o que compensa quando ele será usado em larga escala.

*Aparece em:* Q30, Q45, Q46, Q48, Q52

### Emergent abilities e mirage

Habilidades que parecem surgir de repente quando o modelo passa de certo tamanho. A crítica do *mirage* diz que boa parte disso vem de métricas "tudo ou nada" (acerto exato) que escondem uma melhoria gradual.

*Aparece em:* Q35, Q47, Q51, Q58

### Grokking

Fenômeno em que o modelo primeiro decora o treino e, bem depois, já sem melhorar no treino, "entende" a regra e passa a generalizar.

*Aparece em:* Q58

### Model merging

Combinar os pesos de vários modelos (por exemplo, médias ponderadas) para obter um que reúne habilidades de todos, sem treinar de novo.

*Aparece em:* Q20, Q21, Q59

### Reversal curse

Um modelo treinado com a frase "A é B" não aprende automaticamente "B é A": se ele viu "Fulano dirigiu o filme X", responde bem "quem é Fulano?", mas falha em "quem dirigiu o filme X?". Mostra que o conhecimento fica armazenado de forma direcional.

*Aparece em:* Q43

### Superposition e sparse autoencoders

*Superposition*: o modelo guarda mais conceitos do que tem neurônios, sobrepondo-os nos mesmos números. *Sparse autoencoders* são uma ferramenta de interpretabilidade que tenta separar esses conceitos em "features" legíveis.

*Aparece em:* Q63

### Multi-token prediction

Treinar o modelo para prever vários tokens à frente em vez de só o próximo. Dá um sinal de treino mais rico e pode acelerar a inferência.

*Aparece em:* Q40

### Diffusion language models

Em vez de gerar da esquerda para a direita, começam com um texto "mascarado/ruidoso" e refinam todas as posições em paralelo, em várias passadas.

*Aparece em:* Q8, Q64

## 7. Mixture-of-Experts (MoE) e eficiência

### Mixture-of-Experts (MoE)

Em vez de uma FFN grande, o bloco tem várias FFNs (os *experts*) e cada token usa só alguns deles. O modelo tem muitos parâmetros no total, mas gasta pouco cálculo por token.

> 💡 **Pense assim:** Um hospital com vários especialistas: cada paciente é encaminhado a poucos, e não a todos.

*Aparece em:* Q14, Q46, Q48, Q49, Q54

### Router

A pequena rede que decide para quais experts cada token vai.

*Aparece em:* Q33, Q42, Q48, Q49, Q55

### Active vs total parameters

Parâmetros *totais*: o tamanho do modelo na memória. Parâmetros *ativos*: os que realmente são usados para cada token. Num MoE, o custo de cálculo segue os ativos, mas a memória segue os totais.

*Aparece em:* Q46, Q48

### Load balancing e router collapse

Sem cuidado, o router manda quase tudo para poucos experts (*router collapse*). Usa-se uma *auxiliary load-balancing loss* para distribuir a carga e o *router z-loss* para manter os logits do router estáveis.

*Aparece em:* Q48, Q49

### Quantization

Guardar números com menos bits (por exemplo, 8 ou 4 em vez de 16) para economizar memória e acelerar. Quantizar *ativações* é mais difícil que quantizar *pesos* porque as ativações têm valores extremos (outliers) imprevisíveis.

*Aparece em:* Q35, Q38, Q54, Q55, Q62

### Open-weights vs closed-weights

Modelos *open-weights* têm os pesos publicados (você pode rodar e ajustar por conta própria); *closed* só são acessíveis por API.

*Aparece em:* Q46, Q52, Q55, Q59
