# 📚 Glossário de estudo — Multimodal

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Como uma imagem entra em um LLM](#1-como-uma-imagem-entra-em-um-llm)
- [2. CLIP, embeddings e busca entre modalidades](#2-clip-embeddings-e-busca-entre-modalidades)
- [3. Resolução, tokens de imagem e custo](#3-resolução-tokens-de-imagem-e-custo)
- [4. Limites dos VLMs e ferramentas para contorná-los](#4-limites-dos-vlms-e-ferramentas-para-contorná-los)
- [5. Document AI e RAG multimodal](#5-document-ai-e-rag-multimodal)
- [6. Geração de imagem](#6-geração-de-imagem)
- [7. Áudio e voz](#7-áudio-e-voz)
- [8. Vídeo e robótica](#8-vídeo-e-robótica)
- [9. Segurança e proveniência](#9-segurança-e-proveniência)

## 1. Como uma imagem entra em um LLM

### VLM (vision-language model)

Modelo que recebe imagens (e texto) e responde em texto. Em geral é um LLM ao qual se acrescentou um jeito de "enxergar": um vision encoder e um projector.

*Aparece em:* Q11, Q26, Q27, Q30, Q39

### Patch

Quadradinho fixo da imagem (por exemplo 14×14 ou 16×16 pixels). A imagem é cortada em patches, e cada patch vira um vetor, quase como uma palavra num texto.

> 💡 **Pense assim:** Um quebra-cabeça: o modelo não vê a figura inteira de uma vez, vê uma fileira de peças.

*Aparece em:* Q1, Q3, Q19, Q27, Q29

### ViT (Vision Transformer)

Transformer aplicado a imagens: roda sobre a sequência de patches, assim como um LLM roda sobre a sequência de tokens. Uma imagem 224×224 com patches 16×16 gera 196 patches.

*Aparece em:* Q1, Q2, Q19, Q37

### Vision encoder

A parte do VLM que transforma a imagem numa sequência de embeddings, um por patch. Quase sempre é um ViT já pré-treinado com CLIP ou SigLIP, e não treinado do zero.

*Aparece em:* Q1, Q2, Q20, Q29, Q32

### Projector (adapter)

Pequena rede (muitas vezes um MLP) que traduz os embeddings do vision encoder para o espaço de embeddings do LLM. Depois disso o LLM trata os patches como se fossem tokens.

> 💡 **Pense assim:** Um tradutor entre dois idiomas: o encoder "fala" visão, o LLM "fala" texto, e o projector faz a ponte.

*Aparece em:* Q1, Q20, Q25, Q29, Q32

### Resampler e Q-Former

Alternativa ao MLP: um pequeno transformer com N vetores aprendidos (queries, tipicamente 32 a 64) que olha para todos os patches via cross-attention e devolve sempre exatamente N tokens. O custo fica constante e baixo, mas é um gargalo com perda de informação, o que prejudica documentos densos.

*Aparece em:* Q1, Q3, Q29

### Cross-attention insertion

Em vez de colocar os tokens de imagem na sequência, adicionam-se camadas de cross-attention dentro do LLM congelado, que leem as features visuais diretamente. Não consome contexto, mas altera a arquitetura do LLM e complica o serving. Foi o desenho do Flamingo.

*Aparece em:* Q1, Q29

### Visual instruction tuning (treino em estágios)

Receita típica no estilo LLaVA: no estágio 1 treina-se só o projector com pares imagem-legenda (encoder e LLM congelados). No estágio 2 o LLM é descongelado e ajustado com exemplos de imagem, pergunta e resposta. Depois vêm etapas de alta resolução e de preferências.

*Aparece em:* Q1, Q29

### Early fusion e late fusion

Um único transformer treinado do zero com texto, imagem e áudio intercalados, em vez de colar um encoder pronto num LLM pronto. Integra melhor as modalidades e permite gerar imagem e texto no mesmo modelo, mas o pré-treino é caríssimo. Chameleon e Fuyu são exemplos citados. O oposto (late fusion, ou VLM baseado em adapter) reaproveita um encoder e um LLM prontos e treina só a cola entre eles: barato e modular, mas o encoder congelado é um gargalo, pois o que o pré-treino dele não preservou nunca chega ao LLM.

*Aparece em:* Q29, Q32

### M-RoPE (multimodal rotary embedding)

Versão do RoPE (codificação de posição) que usa três índices: tempo, altura e largura. Assim um patch sabe onde está na imagem em 2D, em vez de ser só a posição N de uma fila 1D. Tokens de texto recebem o mesmo índice nos três eixos, então voltam a ser o RoPE normal.

*Aparece em:* Q19

## 2. CLIP, embeddings e busca entre modalidades

### CLIP

Modelo da OpenAI (2021) que treina juntos um encoder de imagem e um de texto com cerca de 400M pares imagem-legenda da web. Imagens e legendas correspondentes acabam perto uma da outra no mesmo espaço de embeddings. É a base de quase todo vision encoder moderno.

*Aparece em:* Q1, Q2, Q8, Q10, Q18

### Contrastive loss

Função de perda (por exemplo a InfoNCE, que usa softmax e uma temperatura aprendida) que, num batch com N pares, puxa para perto os N pares (imagem, legenda) corretos e empurra para longe os N²−N pares errados. É assim que CLIP aprende sem rótulos de classe.

> 💡 **Pense assim:** Uma festa onde cada pessoa precisa achar o seu par entre todos os convidados.

*Aparece em:* Q2, Q4, Q10, Q32, Q36

### Zero-shot classification

Classificar em categorias que o modelo nunca viu como rótulos. Com CLIP, escreve-se "a photo of a {classe}" para cada classe e escolhe-se a legenda mais próxima do embedding da imagem.

*Aparece em:* Q2, Q31

### SigLIP

Substituto moderno do CLIP que troca o softmax sobre o batch inteiro por uma loss sigmoide pareada: cada par (imagem, texto) vira um problema independente de "combina ou não". Dispensa a normalização global entre dispositivos e funciona bem com batches menores. O SigLIP 2 acrescentou objetivos de localização e features densas.

*Aparece em:* Q1, Q2, Q10, Q18, Q36

### Dual encoder

Arquitetura com dois encoders separados (um para imagem, outro para texto) que só se encontram no espaço de embeddings. Permite pré-calcular os embeddings das imagens e buscar por similaridade, mas não faz comparações finas entre os dois lados.

*Aparece em:* Q1, Q10, Q18, Q29, Q36

### In-batch negatives e hard negatives

Negativos são os exemplos "errados" contra os quais o modelo aprende a distinguir. In-batch negatives são os outros itens do mesmo batch, que logo ficam fáceis demais. Hard negatives são itens parecidos mas errados (mesma categoria, produto exibido e não clicado) e é neles que o modelo realmente aprende. Cuidado com falsos negativos: não clicado não significa irrelevante.

*Aparece em:* Q18, Q36

### Modality gap

Mesmo no espaço compartilhado do CLIP, embeddings de imagem e de texto ficam em duas regiões (cones) separadas. Consequência prática: limiares de similaridade não valem entre pares de modalidades diferentes, e um índice misto favorece resultados da mesma modalidade da consulta.

*Aparece em:* Q2, Q8, Q10, Q18, Q36

### Bag-of-words (comportamento) e attribute binding

Encoders contrastivos tendem a captar quais conceitos aparecem, mas não como se relacionam. Por isso "camisa vermelha com logo azul" pode trazer camisas azuis com logo vermelho. Esse erro de ligar atributos aos objetos certos se chama attribute binding e costuma ser corrigido no reranking.

*Aparece em:* Q2, Q4, Q10, Q18

### ANN (HNSW, IVF-PQ)

Approximate Nearest Neighbor: busca aproximada dos vetores mais próximos, bem mais rápida que comparar com todos. HNSW é um índice em grafo; IVF-PQ divide o espaço em grupos e comprime os vetores (product quantization) para caber em menos memória.

*Aparece em:* Q8, Q18, Q36

### Reranking e cross-encoder

Segunda etapa que reordena os melhores candidatos da busca inicial com um modelo mais caro e preciso. Cross-encoder é um modelo que lê consulta e candidato juntos. Aqui, o dual encoder cuida do recall e o reranker (por exemplo um VLM) cuida da precisão.

*Aparece em:* Q18, Q36

### Matryoshka embeddings

Embeddings treinados para que os primeiros k valores já funcionem sozinhos. Assim dá para truncar o vetor, buscar barato numa primeira etapa e reordenar com o vetor completo.

> 💡 **Pense assim:** Bonecas russas: a menor cabe dentro da maior e já é uma boneca completa.

*Aparece em:* Q36

### Recall@k e nDCG

Métricas de busca. Recall@k é a fração dos itens relevantes que aparecem entre os k primeiros resultados. nDCG também considera a ordem: acertos no topo valem mais.

*Aparece em:* Q18, Q27, Q36

## 3. Resolução, tokens de imagem e custo

### Image tokens

Cada patch processado vira uma posição na sequência do LLM, igual a um token de texto, e é cobrado como tal. Por isso o número de tokens cresce com a área da imagem (resolução dividida pelo tamanho do patch, ao quadrado).

*Aparece em:* Q3, Q30, Q32, Q35, Q37

### Tiling e dynamic resolution (AnyRes)

Em vez de encolher tudo para um tamanho fixo, o modelo recebe uma visão global reduzida mais recortes (tiles) em resolução nativa. Gasta tokens proporcionalmente ao tamanho da imagem. Exemplos: AnyRes do LLaVA-NeXT e a resolução dinâmica do Qwen-VL.

*Aparece em:* Q1, Q3, Q29, Q35, Q37

### Prefill

A fase em que o LLM processa todo o prompt de entrada de uma vez, antes de começar a gerar. Num VLM de documentos ela domina o custo: são ~1 a 2 mil tokens de imagem para só algumas centenas de tokens de saída.

*Aparece em:* Q3, Q29, Q35, Q37

### Prompt caching

Reaproveitar o processamento de um prefixo repetido (por exemplo a mesma imagem em várias perguntas). Para aproveitar, coloque a imagem primeiro e a pergunta por último.

*Aparece em:* Q3, Q17, Q35, Q37

### min_pixels e max_pixels

Parâmetros (nos processadores estilo Qwen) que limitam quantos pixels, e portanto quantos tokens, cada imagem pode gerar. Costumam estar configurados acima do necessário. O ideal é varrer o orçamento e achar o ponto em que a acurácia deixa de melhorar.

*Aparece em:* Q20, Q37

### media_resolution (Gemini 3)

Configuração do Gemini 3 que fixa o orçamento de tokens por imagem ou por frame de vídeo, em vez do cálculo por tiles das versões anteriores.

*Aparece em:* Q3, Q17

## 4. Limites dos VLMs e ferramentas para contorná-los

### Numeric hallucination

Quando o modelo não consegue ler um número e escreve um plausível no lugar. É perigoso porque "$1,284.56" no lugar de "$1,234.56" parece perfeitamente fluente. Mitigações: resolução maior, OCR junto, validação aritmética e self-consistency.

*Aparece em:* Q4, Q11, Q30, Q34

### Object hallucination e POPE

O modelo cita objetos que não estão na imagem, em geral os que costumam aparecer em cenas daquele tipo. O benchmark POPE mede esse problema.

*Aparece em:* Q4

### Self-consistency

Rodar a mesma pergunta várias vezes (com variação) e comparar as respostas. Divergência sinaliza que o valor é incerto. Útil para números críticos.

*Aparece em:* Q4

### Grounding

Ligar uma parte do texto a uma região da imagem. O VLM não tem um detection head: ele escreve as coordenadas da caixa (bounding box) como texto, token a token. Daí vêm os problemas típicos: amostragem (use temperature 0), precisão limitada pelo patch e esquecer de desfazer o resize do pré-processador.

*Aparece em:* Q4, Q7, Q11, Q12, Q30

### IoU

Intersection over Union: área de sobreposição entre duas caixas dividida pela área da união. Mede o quanto uma caixa prevista coincide com a real.

*Aparece em:* Q7

### Set-of-marks

Técnica de desenhar números sobre os elementos candidatos do screenshot para que o modelo responda "clique no 7" em vez de inventar coordenadas. Transforma regressão de coordenadas em classificação.

*Aparece em:* Q21

### Accessibility tree e DOM

Representações estruturadas de uma interface: lista elementos com posição exata, papel (botão, campo) e texto, sem erro de percepção. Num agente de computer use, são preferíveis a pixels sempre que existirem.

*Aparece em:* Q21

### SAM (Segment Anything)

Modelos de segmentação que respondem a prompts e devolvem máscaras precisas dos objetos. O SAM 2 estendeu isso a vídeo com tracking por memória, e o SAM 3 aceita frases curtas e devolve todas as instâncias correspondentes.

*Aparece em:* Q26

### Detector e tracker

Detector: modelo especializado que encontra e caixa objetos de classes conhecidas, rápido e barato. Tracker: acompanha o mesmo objeto ao longo dos frames. Em produção, o VLM planeja e verifica e os especialistas medem.

*Aparece em:* Q7, Q26

### Distillation (destilação)

Treinar um modelo pequeno para imitar um grande. Ex.: usar um VLM de ponta para rotular dados e treinar um classificador pequeno com eles. Também é como se obtêm modelos de difusão de 1 a 4 passos.

*Aparece em:* Q5, Q14, Q20, Q38, Q40

### Linear probe

Treinar só uma camada linear em cima de features congeladas. Se ela já separa suas classes, o encoder consegue "ver" o seu domínio e o problema está em outro lugar.

*Aparece em:* Q20

### LoRA

Ajuste fino leve: em vez de alterar todos os pesos, treinam-se pequenas matrizes adicionais (poucos a dezenas de MB). Em VLMs, o caminho usual é LoRA só no LLM; mexer no vision encoder é o último recurso.

*Aparece em:* Q20, Q25

### Catastrophic forgetting

Ao ajustar um modelo para uma tarefa, ele perde habilidades gerais. Para reduzir: misturar 10 a 30% de dados gerais e manter um benchmark genérico para detectar regressões.

*Aparece em:* Q20, Q39

### Test-time compute (think with tools)

Gastar mais processamento na hora de responder (raciocínio mais longo). Em visão ajuda no raciocínio sobre o que foi percebido, mas não recupera um valor ilegível na imagem. O mais eficaz é dar ferramentas: recortar, dar zoom, rodar OCR ou código.

*Aparece em:* Q35

## 5. Document AI e RAG multimodal

### OCR

Optical Character Recognition: transforma imagem de texto em texto digital, normalmente com coordenadas de cada palavra. Falha de forma visível (lixo), diferente do VLM, que falha de forma invisível com valores plausíveis.

*Aparece em:* Q4, Q11, Q30, Q34, Q35

### Layout analysis e reading order

Detectar a estrutura da página (colunas, tabelas, títulos) e a ordem correta de leitura. É onde OCR clássico mais sofre, por exemplo com várias colunas.

*Aparece em:* Q11, Q34

### OCR pipeline vs OCR-free

Duas estradas para extrair dados de documentos. Pipeline: OCR e depois um LLM sobre o texto, barato e auditável. OCR-free: mandar a imagem da página direto para um VLM, que lida melhor com layouts bagunçados, mas custa mais e pode inventar valores.

*Aparece em:* Q11, Q30

### Cascade (cascata)

Padrão de produção: classificar o documento, rodar OCR barato em tudo e escalar só as páginas difíceis ou valiosas para um VLM, alimentando-o com a imagem e o texto do OCR. Termina com validação e revisão humana.

> 💡 **Pense assim:** Triagem de hospital: o caso simples resolve na recepção, só o grave vai para o especialista.

*Aparece em:* Q11, Q30

### Document-specialised small VLMs

VLMs pequenos feitos para converter páginas em Markdown com ordem de leitura e tabelas, a um custo bem mais próximo do OCR que de um VLM de ponta. Ainda podem inventar texto em regiões ilegíveis, então a validação continua necessária.

*Aparece em:* Q11

### Schema validation e constrained decoding

Validar a extração contra regras: tipos, faixas e verificações aritméticas (itens somam o subtotal). Constrained decoding força a saída a respeitar um formato (JSON). Aproveita a redundância dos documentos para pegar erros sem custo extra.

*Aparece em:* Q4, Q11, Q20, Q23, Q34

### Confidence e abstention

Dar uma nota de confiança a cada campo e aceitar automaticamente só os confiantes, mandando o resto para revisão humana. É assim que se chega a 99% de qualidade: a métrica negociada passa a ser a cobertura (fração automática).

*Aparece em:* Q11, Q30, Q34, Q35, Q39

### Table Transformer

Modelo que reconstrói a estrutura de tabelas a partir da imagem. Camelot e Tabula são extratores baseados em regras (linhas e posição do texto), exatos para PDFs digitais com texto embutido.

*Aparece em:* Q12

### ANLS

Average Normalized Levenshtein Similarity: métrica do DocVQA baseada em distância de edição, que perdoa pequenos erros de grafia na resposta.

*Aparece em:* Q28

### Relaxed accuracy

Métrica do ChartQA em que a resposta numérica conta como correta se estiver a até 5% do valor real, porque ler gráfico é estimativa, não extração.

*Aparece em:* Q12, Q28

### Multimodal RAG

RAG sobre documentos com figuras e tabelas. Três níveis: extrair só texto (perde gráficos), legendar imagens com um VLM e indexar as legendas (padrão pragmático) ou recuperar páginas inteiras como imagem (ColPali).

*Aparece em:* Q27

### ColPali e late interaction

ColPali usa um VLM para gerar muitos vetores por página (um por patch) e vetores por token da consulta. A pontuação é late interaction, no estilo ColBERT: para cada token da consulta pega-se a maior similaridade com algum patch, e somam-se. Dispensa parsing, mas exige armazenar ~1k vetores por página.

*Aparece em:* Q27

### Multi-vector storage

Guardar vários vetores por documento (como no ColPali) em vez de um só. Dá mais precisão, mas o armazenamento explode; atenua-se com pooling e quantização binária.

*Aparece em:* Q27

### RRF (Reciprocal Rank Fusion)

Jeito simples de fundir várias listas ranqueadas: cada item ganha pontos conforme a posição em cada lista, e somam-se. Usado para combinar busca por texto e busca por imagem de página.

*Aparece em:* Q27

### VQA e benchmarks de visão

Visual Question Answering: perguntar sobre uma imagem. DocVQA (documentos), ChartQA (gráficos), TextVQA e OCRBench (texto em imagens) e MMMU (raciocínio de nível universitário) são os mais citados. Cuidado com contaminação: o benchmark pode ter vazado para o treino.

*Aparece em:* Q12, Q28, Q29

### VLM-as-judge

Usar um VLM com rubrica explícita para avaliar respostas abertas. Regras: o juiz precisa ver a imagem, e não deve julgar tarefas (contagem, espacial) que ele mesmo erra. Confira a concordância com humanos antes de confiar.

*Aparece em:* Q28

## 6. Geração de imagem

### Diffusion

Família de modelos que geram imagens por remoção gradual de ruído. Treino: adiciona-se ruído a imagens reais e a rede aprende a prever o ruído (forward process). Geração: parte-se de ruído puro e remove-se um pouco por passo (reverse process).

> 💡 **Pense assim:** Esculpir: começa com um bloco informe e cada passo tira um pouco do que não é imagem.

*Aparece em:* Q5, Q13, Q15, Q31, Q38

### Noise schedule

A regra que define quanto ruído é adicionado em cada passo do processo forward. Controla como a informação é destruída ao longo dos passos.

*Aparece em:* Q5

### DDPM e DDIM (samplers)

DDPM (2020) é a formulação clássica de difusão, com ~1000 passos. Sampler é o algoritmo que percorre o caminho do ruído até a imagem; samplers modernos como o DDIM e seus sucessores chegam a boas imagens em 20 a 50 passos.

*Aparece em:* Q5, Q13

### Latent diffusion

Em vez de difundir pixels, difunde-se uma representação comprimida (latent) criada por um VAE. Uma imagem 512×512 vira um latent 64×64×4, ~48× menos valores, o que tornou a geração viável em GPU comum (Stable Diffusion).

*Aparece em:* Q5, Q13

### VAE

Variational Autoencoder: rede que comprime a imagem num latent e a reconstrói. Em latent diffusion o decoder roda uma vez no fim. É também um teto de qualidade: o que o VAE não reconstrói bem (rostos pequenos, texto) a difusão não recupera.

*Aparece em:* Q5, Q13, Q15, Q38

### U-Net e DiT

Redes que fazem o denoising. U-Net é a arquitetura convolucional com atenção usada nos primeiros modelos; DiT (Diffusion Transformer) é a versão com transformer, que escala melhor e é usada no SD3 e no Flux.

*Aparece em:* Q13, Q15, Q38

### Classifier-free guidance (CFG)

Técnica para fazer o modelo obedecer ao prompt: calculam-se duas previsões (com e sem o texto) e extrapola-se na direção do texto, multiplicando a diferença por uma escala s. Escala alta dá mais aderência ao prompt, mas menos diversidade, cores saturadas e artefatos. Dobra o custo por passo. Um negative prompt entra no lugar do prompt vazio do ramo "sem condição", de modo que a extrapolação se afasta dele.

*Aparece em:* Q5, Q13, Q14, Q38

### Flow matching e rectified flow

Alternativa ao DDPM em que se define um caminho reto entre dado e ruído e a rede prevê a velocidade ao longo dele. Trajetórias mais retas pedem menos passos de amostragem. Usado no SD3 e no Flux.

*Aparece em:* Q13, Q14, Q38, Q39

### Cross-attention (texto → imagem)

Mecanismo pelo qual o denoiser consulta os embeddings do texto a cada passo, e assim o prompt condiciona a imagem. Também aparece no Whisper, onde o decoder consulta o áudio codificado.

*Aparece em:* Q5, Q13, Q15, Q25, Q29

### VQ-VAE e geração autoregressiva de imagem

Abordagem alternativa à difusão: um tokenizador (VQ-VAE) transforma a imagem em códigos discretos e um transformer os prevê em sequência, como texto. Integra-se melhor ao LLM (texto na imagem, edição por conversa), mas decodifica de forma sequencial.

*Aparece em:* Q15

### ControlNet

Adiciona controle de estrutura (bordas, profundidade, pose) à difusão por meio de uma cópia treinável do encoder do denoiser, ligada por camadas inicializadas em zero. Controla geometria, não a identidade do objeto.

*Aparece em:* Q13, Q25

### IP-Adapter

Permite usar uma imagem de referência como prompt, por um caminho extra de cross-attention. Não precisa de treino por objeto, mas a identidade é aproximada (forma e cor, não texto legível).

*Aparece em:* Q25

### DreamBooth

Ajuste fino por objeto: com ~10 a 30 fotos, ensina-se o modelo a gerar aquele objeto específico, geralmente como um LoRA. Melhor fidelidade, mas custa GPU, armazenamento e moderação por adaptador.

*Aparece em:* Q25

### Instruction-based editing models

Modelos que recebem uma imagem de referência e uma instrução ("mesmo produto, sobre uma bancada de mármore") e editam sem treino por objeto. Desde 2025 são o ponto de partida padrão para fotos de produto.

*Aparece em:* Q25

### FID, CLIPScore e GenEval

FID compara a distribuição de imagens geradas com a de reais, e só faz sentido em conjunto, não por imagem. CLIPScore mede o alinhamento prompt-imagem via embeddings do CLIP. GenEval checa composição (contagens, cores, posições) com detectores. Nenhuma substitui a avaliação humana.

*Aparece em:* Q28

### Text encoder T5

Modelo de linguagem usado como text encoder em geradores de imagem recentes. Ajuda a seguir prompts longos e a renderizar texto dentro da imagem.

*Aparece em:* Q13, Q15

## 7. Áudio e voz

### ASR / STT

Automatic Speech Recognition (ou speech-to-text): converte fala em texto.

*Aparece em:* Q16, Q17, Q22, Q24, Q28

### Log-Mel spectrogram

Uma "imagem" do som: mostra a energia em cada faixa de frequência ao longo do tempo, numa escala (Mel) próxima da percepção humana. É a entrada do Whisper e a saída intermediária do TTS clássico.

*Aparece em:* Q6, Q31

### Whisper

ASR da OpenAI: um transformer encoder-decoder que lê janelas de 30 s de espectrograma e gera o texto token a token. Tokens especiais escolhem a tarefa (idioma, transcrever ou traduzir, timestamps). Foi treinado com ~680 mil horas de áudio da web.

*Aparece em:* Q6, Q22

### Weak supervision

Treinar com dados rotulados de forma imperfeita, mas em enorme volume (ex.: transcrições da web). A robustez do Whisper vem disso, não da arquitetura. É a mesma lição do CLIP: escala de supervisão fraca vence dados pequenos e curados.

*Aparece em:* Q6

### WER

Word Error Rate: (substituições + remoções + inserções) dividido pelo número de palavras da referência. Pode passar de 100%. É enganoso para produto: todas as palavras pesam igual, a normalização muda o resultado e médias escondem grupos piores (sotaques, ruído).

*Aparece em:* Q6, Q9, Q22, Q28, Q31

### Entity error rate e cpWER

Entity error rate mede erros só nas palavras que importam (nomes, SKUs, remédios). cpWER é um WER que respeita quem falou, útil quando há diarização.

*Aparece em:* Q9, Q22

### Diarisation e DER

Diarização responde "quem falou quando" agrupando segmentos por voz. DER (Diarization Error Rate) mede fala perdida, alarme falso e troca de falante. Em gravações estéreo com um lado por canal, use os canais e dispense a diarização.

*Aparece em:* Q6, Q9, Q22

### VAD e endpointing

Voice Activity Detection detecta se há fala no áudio. Endpointing decide que o usuário terminou de falar, normalmente após X ms de silêncio. É o imposto de latência mais difícil de evitar num voice agent.

*Aparece em:* Q16, Q22, Q24, Q40

### Semantic turn detection

Modelo pequeno que lê a transcrição em streaming e decide se a frase está completa, para espera curta quando parece terminada e longa quando está no meio. Resolve o dilema "corta o usuário ou deixa silêncio sobrando" que um único timeout não resolve.

*Aparece em:* Q16, Q24, Q40

### Barge-in e backchannel

Barge-in é o usuário interromper o agente enquanto ele fala: é preciso parar o TTS, cancelar a geração e truncar o histórico ao que o usuário realmente ouviu. Backchannels são "aham" e "mm-hmm", que não devem contar como interrupção. É preciso também echo cancellation, para o agente não ouvir a própria voz saindo do alto-falante e achar que foi interrompido.

*Aparece em:* Q16, Q22, Q24, Q40

### Pipeline vs speech-to-speech

Pipeline: ASR, depois LLM, depois TTS, com texto no meio (fácil de auditar e controlar). Speech-to-speech: um único modelo recebe e emite áudio, mais rápido (~300 ms) e preserva tom e emoção, mas sem o texto intermediário fica mais difícil colocar guardrails, logs e tools.

*Aparece em:* Q16, Q24, Q31, Q40

### Time-to-first-token e time-to-first-audio

Tempo até a primeira saída. Em voice agents, somam-se endpointing, ASR, TTFT do LLM e TTS. Para ficar sub-segundo, tudo precisa operar em streaming e o TTS começa já na primeira frase.

*Aparece em:* Q16, Q31

### TTS e vocoder

Text-to-Speech. O desenho clássico tem um modelo acústico (texto → espectrograma, como Tacotron 2 ou FastSpeech 2) e um vocoder neural (espectrograma → onda de áudio, como o HiFi-GAN).

*Aparece em:* Q16, Q28, Q31

### Neural audio codec e codec language model

Um codec neural (EnCodec, SoundStream) comprime áudio em tokens discretos. Um transformer gera esses tokens a partir do texto, como um LLM de áudio (estilo VALL-E). Com alguns segundos de voz de referência ele continua naquela voz.

*Aparece em:* Q31, Q40

### Voice cloning

Reproduzir a voz de alguém a partir de poucos segundos de áudio. É ao mesmo tempo a principal vantagem e o principal risco de segurança do TTS moderno, exigindo verificação de consentimento.

*Aparece em:* Q31

### Text normalization

Converter formas escritas em formas faladas: "Dr.", "3/4", "$1,024". Parece trivial, mas segue sendo fonte frequente de bugs em TTS.

*Aparece em:* Q31

### MOS

Mean Opinion Score: nota média dada por pessoas à qualidade da fala sintetizada. Costuma ser combinada com o WER de um ASR sobre a saída, como proxy automático de inteligibilidade.

*Aparece em:* Q28

### Full-duplex (Moshi, Mimi, RVQ)

Modelo que escuta e fala ao mesmo tempo, sem noção de turno. O Moshi (Kyutai) usa o codec Mimi (12,5 frames por segundo, cada um com vários codebooks RVQ) e dois fluxos de áudio. Prevê também o texto da própria fala (inner monologue). Natural, mas difícil de controlar e auditar.

*Aparece em:* Q16, Q40

### PII redaction

Remover dados pessoais (como números de cartão) antes de chegarem ao LLM ou ao índice. Em gravações de chamadas, números de cartão são falados em voz alta.

*Aparece em:* Q22

## 8. Vídeo e robótica

### Frame sampling e custo de vídeo

Em vez de processar o vídeo inteiro, amostram-se frames (cerca de 1 por segundo é o padrão). Alternativas mais espertas: detectar cortes de cena, amostrar mais perto de momentos de interesse ou começar pelo áudio transcrito. O custo é o gargalo: uma hora de vídeo pode chegar a centenas de milhares até cerca de 1M de tokens, por isso se faz uma passada barata para localizar e uma segunda, densa, só na janela relevante.

*Aparece em:* Q17, Q19, Q26, Q38

### Spacetime patches e 3D VAE

Geração de vídeo estende a difusão de imagem: um VAE 3D comprime o vídeo no tempo e no espaço, e o latent é cortado em patches espaço-temporais. Cinco segundos de 720p chegam a ~100 mil tokens, o que explica o custo alto.

*Aparece em:* Q17, Q38

### VLA (vision-language-action)

VLM ajustado para emitir ações de robô (comandos de juntas ou de garra) a partir de imagens e uma instrução. Aproveita o conhecimento da web para seguir instruções novas e objetos nunca vistos em dados de robô.

*Aparece em:* Q39

### Action chunking e dual-system

O VLM é lento (dezenas a centenas de ms) e o controle exige 50 a 200 Hz. Soluções: action chunking (prever ~0,5 a 1 s de ações por inferência) e sistemas duplos, com um VLM lento para entender a cena e uma política pequena e rápida para o motor.

*Aparece em:* Q39

### Action tokens e FAST

Duas formas de emitir ações. Como tokens: cada dimensão é discretizada em 256 faixas (RT-2, OpenVLA), simples, porém lento. Ou com uma cabeça contínua: o pi0 usa um action expert com flow matching. FAST comprime chunks de ação em espaço de frequência.

*Aparece em:* Q39

## 9. Segurança e proveniência

### Prompt injection em imagens

Texto na imagem (às vezes quase invisível) que manda o agente fazer algo. Como instrução e dado chegam pelo mesmo canal, não há como impedir o modelo de "obedecer": a defesa precisa ser arquitetural.

*Aparece em:* Q23

### Privilege separation

Separar quem lê conteúdo não confiável de quem tem ferramentas. O leitor de imagens não tem tools e só devolve dados estruturados; um planner separado, que nunca vê o conteúdo cru, é quem age. Somado a saída com schema e confirmação humana de ações com efeito colateral.

> 💡 **Pense assim:** Um tradutor que lê cartas suspeitas mas não tem a chave do cofre: só passa adiante um resumo.

*Aparece em:* Q23

### C2PA (Content Credentials)

Padrão que anexa ao arquivo um manifesto assinado criptograficamente dizendo quem e com que ferramenta o criou. Forte quando presente, mas é metadado: screenshot ou re-encode removem. A ausência não prova nada.

*Aparece em:* Q33

### Invisible watermarking (SynthID)

Sinal escondido nos próprios pixels (ou no áudio) que sobrevive a compressão, resize e edições leves. Mais robusto que metadado, mas só o dono do detector consegue lê-lo, e adversários determinados podem removê-lo.

*Aparece em:* Q31, Q33

### Deepfake

Mídia sintética que imita uma pessoa real. Detectores genéricos de deepfake são uma corrida armamentista pouco confiável, por isso proveniência, regras de uso e regulação pesam mais.

*Aparece em:* Q33

### Hash-matching (CSAM)

Comparar o hash de uma imagem com bases de material ilegal conhecido (como PhotoDNA ou o NCMEC) para bloqueá-lo, tanto no corpus de treino quanto nos uploads.

*Aparece em:* Q33

### EU AI Act, Artigo 50

Regulação europeia cujas obrigações de transparência (marcar conteúdo gerado por IA de forma legível por máquina e divulgar deepfakes) passam a valer a partir de agosto de 2026, segundo o material.

*Aparece em:* Q33

### NSFW classifier e base rate

Classificador que detecta conteúdo impróprio na entrada e na saída. Com milhões de gerações por dia, até 1% de falsos positivos vira uma avalanche de reclamações: é preciso medir os dois tipos de erro.

*Aparece em:* Q33
