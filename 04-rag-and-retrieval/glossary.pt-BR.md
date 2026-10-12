# 📚 Glossário de estudo — RAG & Retrieval

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Fundamentos: o que é RAG e quando usar](#1-fundamentos-o-que-é-rag-e-quando-usar)
- [2. Ingestão: parsing, chunking e metadados](#2-ingestão-parsing-chunking-e-metadados)
- [3. Embeddings e similaridade](#3-embeddings-e-similaridade)
- [4. Busca vetorial: exata, ANN e bancos vetoriais](#4-busca-vetorial-exata-ann-e-bancos-vetoriais)
- [5. Busca lexical, híbrida e fusão](#5-busca-lexical-híbrida-e-fusão)
- [6. Reranking e arquiteturas de recuperação](#6-reranking-e-arquiteturas-de-recuperação)
- [7. Entendimento da query e RAG avançado](#7-entendimento-da-query-e-rag-avançado)
- [8. Segurança, permissões e índice fresco](#8-segurança-permissões-e-índice-fresco)
- [9. Avaliação e depuração](#9-avaliação-e-depuração)
- [10. Produção: latência, custo, cache e escala](#10-produção-latência-custo-cache-e-escala)

## 1. Fundamentos: o que é RAG e quando usar

### RAG (Retrieval-Augmented Generation)

Em vez de responder só com o que aprendeu no treino, o LLM recebe, na hora da pergunta, trechos relevantes buscados em uma base de documentos e responde a partir deles. O ciclo mínimo é: transformar a pergunta em vetor, buscar os trechos mais próximos, colocá-los no prompt e gerar a resposta.

> 💡 **Pense assim:** Prova com consulta: em vez de responder de memória, o aluno pode abrir o livro nas páginas certas antes de escrever.

*Aparece em:* Q1, Q2, Q37, Q43, Q50

### Parametric memory

O que o modelo "sabe" por ter sido gravado nos pesos durante o treino. É uma compressão com perdas e fica desatualizada, ao contrário do que o RAG traz na hora.

*Aparece em:* Q1, Q28, Q46, Q47

### Knowledge cutoff

A data a partir da qual o modelo não viu mais nada no treino. O RAG contorna isso, porque o índice pode ser atualizado em minutos, enquanto os pesos não.

*Aparece em:* Q1

### Hallucination e grounding

*Hallucination* é o modelo afirmar algo que não tem base. *Grounding* é ancorar cada afirmação nos trechos recuperados. O RAG reduz alucinação, mas não a elimina: o modelo ainda pode ler errado ou ignorar o contexto.

*Aparece em:* Q1, Q44, Q46, Q56, Q60

### Fine-tuning vs RAG

Fine-tuning muda o *comportamento* do modelo (estilo, formato, jeito de usar ferramentas) e é ruim para injetar fatos. RAG fornece o *conhecimento*. Regra prática: fine-tuning ensina habilidade, RAG entrega informação.

*Aparece em:* Q1, Q2, Q19, Q54, Q58

### Long-context stuffing

Colocar o corpus inteiro no prompt, sem busca. Funciona quando o corpus é pequeno, estável e igual para todos os usuários. Deixa de servir quando o corpus é grande, muda sempre, tem permissões por usuário ou exige citações.

*Aparece em:* Q1, Q2, Q17, Q27, Q50

### Context window

O limite de tokens que o modelo consegue considerar de uma vez (prompt mais resposta). Em RAG, cada trecho recuperado gasta parte desse orçamento.

*Aparece em:* Q3, Q5, Q27, Q50, Q57

### Lost in the middle

Modelos usam pior a informação que fica no meio de um contexto longo do que a do começo e do fim. Por isso mais trechos no prompt nem sempre melhoram a resposta.

*Aparece em:* Q2, Q5, Q46, Q50

### Top-k

Os k resultados mais bem ranqueados de uma busca (por exemplo, os 5 ou 20 melhores). O "k" é quantos trechos você pede ou entrega ao modelo.

*Aparece em:* Q10, Q23, Q46, Q51, Q55

### Prompt caching

Recurso dos provedores de LLM que guarda o início fixo do prompt (system prompt, exemplos, documento) para não pagar e reprocessar tudo de novo a cada chamada. Os tokens em cache saem bem mais baratos.

*Aparece em:* Q2, Q17, Q48, Q49, Q50

### Ingestion pipeline (offline) e query pipeline (online)

O RAG tem dois caminhos. *Offline*: ler, quebrar, vetorizar e indexar os documentos (acontece antes, uma vez). *Online*: o que roda a cada pergunta, da reescrita da query até a resposta. A qualidade se perde mais nas etapas offline (parsing e chunking) do que na escolha do banco vetorial.

*Aparece em:* Q3

### Error compounding

Cada etapa do pipeline tem sua taxa de acerto, e elas se multiplicam: 90% de parsing, 90% de retrieval e 90% de geração fiel resultam em cerca de 73% no fim. Por isso cada etapa precisa ser medida separadamente.

*Aparece em:* Q3, Q31, Q36

## 2. Ingestão: parsing, chunking e metadados

### Parsing

A etapa que transforma PDF, HTML, DOCX etc. em texto limpo, preservando tabelas e estrutura. É a que mais gera defeitos: se o parser estraga uma tabela, nada depois consegue consertar.

*Aparece em:* Q13, Q16, Q46, Q47, Q52

### OCR

Reconhecimento óptico de caracteres: lê o texto a partir da imagem de uma página escaneada. Tem a sua própria taxa de erro.

*Aparece em:* Q13, Q16, Q47, Q52

### VLM-based parsing

Renderizar cada página como imagem e pedir a um modelo de visão e linguagem (VLM) que a transcreva para markdown. Lida bem com layouts difíceis, mas custa uma chamada de LLM por página.

*Aparece em:* Q13, Q16, Q30, Q52

### Parse-quality signals

Sinais automáticos para detectar parsing quebrado sem ler tudo: página quase vazia, texto sem sentido, tabela com número inconsistente de colunas, cabeçalho repetido em toda página. As páginas suspeitas vão para uma fila de revisão ou reprocessamento.

*Aparece em:* Q13, Q16

### Chunk e chunking

*Chunk* é cada pedaço em que o documento é dividido para ser vetorizado e buscado. *Chunking* é a estratégia de cortar. É a primeira coisa a revisar quando o recall está ruim, antes de trocar o modelo de embedding.

> 💡 **Pense assim:** Fichas de estudo: um livro inteiro é difícil de consultar, mas cada ficha com uma ideia é fácil de achar.

*Aparece em:* Q4, Q5, Q17, Q45, Q46

### Fixed-size e recursive chunking

*Fixed-size* corta a cada N tokens, sem olhar o sentido. *Recursive* tenta cortar primeiro em parágrafos, depois em frases, depois em tokens, até caber no limite. É o padrão sensato.

*Aparece em:* Q4

### Structure-aware chunking e heading path

Cortar seguindo a estrutura do documento (títulos markdown, seções HTML, funções de código) e anexar ao chunk o caminho de títulos ("Manual > Benefícios > Licença parental"). Só o heading path já resolve muitos erros de busca.

*Aparece em:* Q3, Q4, Q14, Q17, Q47

### Semantic chunking

Cortar onde a similaridade entre frases vizinhas cai, indicando mudança de assunto. Bonito na teoria, mas o ganho é pequeno e custa embeddings extras na ingestão.

*Aparece em:* Q4

### Overlap

Repetir o final de um chunk no começo do próximo, para um fato que fica na fronteira não ser cortado ao meio. O usual é 10 a 20%; mais que isso gera quase duplicatas que ocupam o top-k.

*Aparece em:* Q5, Q33

### Small-to-big (parent-document retrieval)

Buscar com chunks pequenos (match preciso) mas entregar ao LLM o trecho "pai" maior (leitura coerente). Separa a unidade de busca da unidade de geração e faz a pergunta "qual o tamanho ideal de chunk" perder boa parte do sentido.

*Aparece em:* Q4, Q5, Q46, Q47, Q50

### Contextual retrieval

Antes de vetorizar, um LLM escreve uma frase curta situando o chunk no documento ("este trecho é do relatório do 2º tri da ACME...") e ela é colocada na frente dele. Resolve o chunk que diz "a empresa" sem dizer qual. Custa uma chamada por chunk na ingestão, e o prompt caching a torna viável.

*Aparece em:* Q17, Q46, Q47, Q48, Q49

### Late chunking

Vetorizar o documento inteiro com um embedder de contexto longo e só depois agrupar (mean-pooling) os vetores de tokens por chunk. Cada chunk "enxerga" o documento todo, sem chamar um LLM por chunk, mas só ajuda o lado denso, não o BM25.

*Aparece em:* Q4, Q7, Q17

### Chunk metadata

Informações guardadas junto a cada chunk: origem, página, título, permissões (ACL), datas, versão e dados do pipeline (modelo de embedding, versão do parser). Permitem filtrar, citar, depurar e migrar com segurança.

*Aparece em:* Q3, Q14, Q42, Q51, Q56

### Content hash

Uma "impressão digital" do texto, calculada por função hash. Se o hash não mudou, o documento não mudou, então dá para pular a reindexação e detectar duplicatas exatas de graça.

*Aparece em:* Q14, Q29, Q33, Q49

### Deduplicação: MinHash, SimHash e boilerplate

Técnicas para achar documentos quase idênticos (MinHash com LSH, SimHash) e remover "boilerplate", o texto que se repete em milhares de documentos (rodapés legais, assinaturas de e-mail). Sem isso, o top-k se enche do mesmo fato repetido.

*Aparece em:* Q5, Q15, Q31, Q33, Q45

## 3. Embeddings e similaridade

### Embedding

Vetor (lista de números) que representa o significado de um texto. Textos parecidos ficam com vetores próximos. É a base da busca semântica.

> 💡 **Pense assim:** Coordenadas num mapa: textos sobre o mesmo assunto ficam no mesmo bairro.

*Aparece em:* Q7, Q17, Q19, Q49, Q51

### Bi-encoder

Modelo que vetoriza a query e o documento *separadamente*, cada um em um vetor, e mede a relevância pela proximidade entre os vetores. Como os vetores dos documentos são calculados uma vez, a busca escala para milhões. O limite é que o documento é comprimido antes de se saber a pergunta.

*Aparece em:* Q3, Q6, Q10, Q25

### Cosine similarity, dot product e Euclidean

Três formas de medir proximidade entre vetores. Cosseno compara só a direção; produto escalar também recompensa o tamanho do vetor; Euclidiana mede a distância. Com vetores normalizados (comprimento 1) as três dão o *mesmo ranking*. O importante é usar a métrica com que o modelo foi treinado.

*Aparece em:* Q6, Q11, Q34, Q52, Q60

### Anisotropy e score não calibrado

Um score de similaridade de 0,83 não significa "83% relevante": os valores dependem do modelo e se concentram numa faixa estreita. Por isso o cosseno serve para ordenar, não como probabilidade nem como limiar fixo copiado de tutorial.

*Aparece em:* Q6, Q10, Q11, Q24, Q60

### MTEB

Massive Text Embedding Benchmark, o ranking público mais usado de modelos de embedding. Bom para montar uma lista de candidatos, mas é público (modelos podem ter sido ajustados para ele) e não representa o seu domínio. Decida com o seu próprio conjunto de avaliação.

*Aparece em:* Q7, Q51

### Matryoshka embeddings (MRL)

Embeddings treinados para que os primeiros N números do vetor já formem um embedding válido. Dá para truncar (por exemplo, de 3072 para 256 dimensões) e renormalizar, perdendo pouca qualidade. Permite uma primeira busca barata e um refinamento com o vetor completo.

> 💡 **Pense assim:** Bonecas russas: cada boneca menor está dentro da maior e continua sendo uma boneca completa.

*Aparece em:* Q7, Q18

### Quantization (int8, binary)

Guardar cada número do vetor com menos bits (int8 é 4× menor que float32; binário é 32× menor) para economizar memória, aceitando alguma perda. Costuma-se reavaliar os melhores candidatos com o vetor completo.

*Aparece em:* Q8, Q18, Q21, Q22, Q55

### Contrastive training, hard negatives e InfoNCE

Jeito de treinar embedders: aproximar pares (pergunta, trecho relevante) e afastar os irrelevantes. *Hard negatives* são trechos que parecem relevantes mas não são, e são o que realmente ensina distinções finas. *InfoNCE* é a função de perda usual.

*Aparece em:* Q6, Q15, Q19, Q54

### Query/document prefixes e instructions

Muitos embedders esperam marcadores diferentes para query e documento ("query: " e "passage: " no E5) ou uma instrução de tarefa. Ignorar não gera erro, só piora os resultados em silêncio. Guarde a convenção junto à versão do modelo.

*Aparece em:* Q15

### Fine-tuning de embedder

Treinar o embedder com pares (query, trecho) do seu domínio, muitas vezes minerados de logs de uso. É um dos maiores ganhos para domínios cheios de jargão, mas amarra o índice ao modelo: cada novo modelo exige reembedar tudo.

*Aparece em:* Q19

### Embedding model versioning

Vetores de modelos diferentes não são comparáveis. Por isso se guarda a versão do modelo em cada chunk e se verifica, na hora da query, se bate com o modelo da busca: trocar o modelo sem reembedar vira erro alto, não degradação silenciosa.

*Aparece em:* Q7, Q14, Q29, Q47, Q51

## 4. Busca vetorial: exata, ANN e bancos vetoriais

### Nearest-neighbour search (k-NN)

Achar os vetores mais próximos do vetor da query. A versão exata (brute force) compara com todos e é uma multiplicação de matrizes, boa até cerca de 1 milhão de vetores.

*Aparece em:* Q1, Q6, Q8

### ANN (Approximate Nearest Neighbour)

Busca aproximada: visita só um subconjunto bem escolhido dos vetores, trocando um pouco de recall por muito menos trabalho. A falha é silenciosa: o melhor chunk simplesmente não volta e nada dá erro.

*Aparece em:* Q8, Q23, Q42, Q48, Q55

### HNSW

Índice ANN em forma de grafo em várias camadas: as de cima são esparsas, com saltos longos; a de baixo tem todos os vetores. A busca desce de camada em camada, sempre indo ao vizinho mais próximo da query. Ótimo recall e latência, mas fica todo em RAM e lida mal com deletes.

> 💡 **Pense assim:** Achar um endereço de avião, depois de carro, depois a pé: cada camada refina a localização.

*Aparece em:* Q8, Q20, Q21, Q23, Q48

### M, ef_search e ef_construction

Parâmetros do HNSW. *M*: arestas por nó (mais recall e mais memória). *ef_construction*: qualidade da construção do grafo (ingestão mais lenta). *ef_search*: quantos candidatos a busca mantém, o principal botão em tempo de consulta (mais recall, mais latência).

*Aparece em:* Q20, Q21

### IVF

Índice que agrupa (k-means) os vetores em células e, na busca, examina só as `nprobe` células mais próximas. Mais barato de construir, precisa de treino e perde recall quando o vizinho está na célula ao lado.

*Aparece em:* Q20, Q21, Q22, Q23, Q29

### Product quantization (PQ)

Compressão que quebra cada vetor em subvetores e guarda só o código do centroide mais próximo de cada um. Reduz a memória em 10 a 50 vezes, com perda. Costuma ir junto com IVF e com reavaliação dos melhores no vetor completo.

*Aparece em:* Q21, Q22, Q55

### Recall@k vs exact search

Para saber quanto o índice ANN perde, compare seus resultados com a busca exata numa amostra de queries reais. Repita depois de muitos inserts ou deletes.

*Aparece em:* Q8, Q21, Q29, Q47

### Tombstones e compaction

No HNSW, um delete só marca o nó como apagado (tombstone) e o grafo degrada com o tempo. Por isso se agenda compactação ou reconstrução do índice, acompanhando o recall.

*Aparece em:* Q20, Q21, Q29

### Metadata filtering: pre e post-filtering

*Post-filtering* busca o top-k e depois descarta o que não passa no filtro (pode sobrar quase nada, a "starvation"). *Pre-filtering* restringe antes, mas num grafo HNSW os nós que passam ficam isolados e a busca degenera. Motores bons fazem travessia ciente do filtro, ou trocam para força bruta quando o filtro é muito seletivo.

*Aparece em:* Q8, Q22, Q23, Q42, Q55

### pgvector e bancos vetoriais

pgvector é uma extensão do Postgres para vetores: dados, filtros e transações no mesmo lugar. A regra é "use pgvector até doer" (escala muito grande, filtros pesados, hybrid nativo). Aí entram os bancos dedicados (Qdrant, Milvus, Weaviate, Pinecone, Turbopuffer, Vespa) ou motores de busca como Elasticsearch/OpenSearch.

*Aparece em:* Q8, Q9, Q20, Q22, Q25

## 5. Busca lexical, híbrida e fusão

### BM25

Algoritmo clássico de busca por palavras. Pontua cada documento somando, para cada termo da query, o quanto o termo é raro (IDF), com retorno decrescente para repetições (saturação de TF) e desconto para documentos longos. Não entende sinônimos, mas é excelente para códigos, nomes e IDs.

*Aparece em:* Q3, Q9, Q12, Q17, Q32

### IDF, TF saturation e length normalization

As três peças do BM25. *IDF*: termos raros pesam mais. *Saturação de TF*: a 20ª ocorrência quase não conta; o parâmetro k1 controla a velocidade. *Normalização de tamanho*: documentos longos são descontados, controlado pelo parâmetro b.

*Aparece em:* Q9, Q12, Q32, Q45

### Analyzer (tokenização, stemming, stopwords)

O pré-processamento do texto na busca lexical: como quebrar em termos, reduzir palavras à raiz e ignorar palavras comuns. Um analyzer errado para o seu jargão estraga o BM25 sem aviso, e culpam o algoritmo.

*Aparece em:* Q12, Q36

### Lexical vs dense (sparse vs dense)

*Lexical/sparse*: compara palavras literais, via índice invertido (palavra → lista de documentos). *Dense*: compara vetores de significado. O denso pega paráfrases; o lexical pega termos exatos. Cada um falha onde o outro acerta.

*Aparece em:* Q9, Q24, Q32, Q35, Q59

### Hybrid search

Rodar a busca lexical (BM25) e a densa em paralelo e combinar os resultados. É um dos upgrades de maior valor e menor esforço em RAG, porque a busca densa "erra feio" em IDs, códigos de erro e nomes.

*Aparece em:* Q3, Q9, Q22, Q49, Q59

### Reciprocal Rank Fusion (RRF)

Combina listas ranqueadas somando 1/(k + posição) de cada documento em cada lista (com k ≈ 60). Usa *posições*, não scores, porque os scores do BM25 e do cosseno estão em escalas incomparáveis. Não precisa de treino e quase não precisa de ajuste.

*Aparece em:* Q9, Q24, Q26, Q36, Q41

### Learned sparse retrieval (SPLADE)

Um transformer produz um vetor esparso sobre o vocabulário, com pesos aprendidos e termos *expandidos* ("infarto do miocárdio" também acende "ataque cardíaco"). Dá matching semântico servindo de um índice invertido comum, e o resultado é interpretável. Custa um forward pass por documento na indexação.

*Aparece em:* Q9, Q32

### Query expansion e multi-query

Gerar 3 a 5 variações da pergunta, buscar cada uma e fundir com RRF. Protege contra o azar de uma formulação cair no bairro errado do espaço vetorial.

*Aparece em:* Q9, Q24, Q26

### MMR (Maximal Marginal Relevance)

Reordena os resultados equilibrando relevância e novidade: penaliza o chunk muito parecido com os já escolhidos. Evita o top-k cheio de repetições.

*Aparece em:* Q33

## 6. Reranking e arquiteturas de recuperação

### Reranker

Segundo estágio que reavalia, com mais cálculo por par, os candidatos que o primeiro estágio trouxe. Receita padrão: recuperar 100 a 200 de forma híbrida, rerankear e ficar com os 5 a 20 melhores. Costuma ser o melhor ganho de qualidade por esforço depois de arrumar o chunking.

> 💡 **Pense assim:** Triagem em duas fases: um funil rápido seleciona 100 candidatos e uma banca cuidadosa escolhe os 10 finais.

*Aparece em:* Q10, Q43, Q46, Q58, Q60

### Cross-encoder

Modelo que recebe query e documento *juntos* num mesmo transformer, com attention completa entre os dois. É muito mais preciso que o bi-encoder (entende negação e correferência), mas exige uma passada por candidato, então só serve para reranquear dezenas ou centenas.

*Aparece em:* Q6, Q10, Q25, Q58, Q60

### Late interaction (ColBERT) e MaxSim

Meio-termo: guarda um vetor *por token* do documento (pré-computável) e, na consulta, para cada token da query pega a maior similaridade com algum token do documento (MaxSim) e soma. Match fino por token com custo de armazenamento de 10 a 100×.

*Aparece em:* Q6, Q10, Q22, Q25, Q52

### Visual document retrieval (ColPali)

Em vez de extrair texto, renderiza cada página como imagem, gera vetores por pedaço (patch) dela com um VLM e busca por late interaction. Pula o parsing, e brilha em slides, formulários e relatórios com gráficos. O preço é um índice enorme e a exigência de um gerador que leia imagens.

*Aparece em:* Q52

### LLM reranking: pointwise, pairwise e listwise

Usar um LLM como reranker. *Pointwise*: dá nota a cada par isoladamente. *Pairwise*: compara dois trechos de cada vez. *Listwise*: recebe a query e ~20 candidatos e devolve o ranking. Custa bem mais que um cross-encoder, então se justifica em queries difíceis ou como "professor" para treinar um modelo menor (distillation).

*Aparece em:* Q25, Q58

### Position bias

Tendência de dar mais peso ao que aparece em certas posições: o usuário clica no 1º resultado só por ser o 1º, e um LLM reranker favorece a ordem em que os candidatos foram apresentados.

*Aparece em:* Q54, Q58

## 7. Entendimento da query e RAG avançado

### Query rewriting (condensation)

Um LLM reescreve a pergunta do usuário como uma busca autossuficiente, usando o histórico da conversa ("e para o plano enterprise?" vira uma pergunta completa). É indispensável em RAG conversacional.

*Aparece em:* Q26, Q27, Q40, Q46, Q48

### HyDE (Hypothetical Document Embeddings)

O LLM escreve uma resposta hipotética e é *ela* que vira vetor para a busca. A ideia é que respostas ficam mais perto de documentos no espaço vetorial do que perguntas. Se o palpite do LLM sair do domínio, atrapalha: fundir com a busca da query original.

*Aparece em:* Q26

### Query decomposition (sub-question)

Quebrar uma pergunta composta em sub-perguntas buscadas separadamente e depois sintetizadas. *Paralela* para comparações; *sequencial* para multi-hop. Em perguntas simples só aumenta custo e latência.

*Aparece em:* Q26, Q31, Q47

### Multi-hop

Pergunta cuja resposta exige encadear vários fatos: a busca do passo 2 só pode ser escrita depois de ler o resultado do passo 1 ("quem assinou o contrato com o maior fornecedor?"). Nenhum chunk sozinho tem a resposta.

*Aparece em:* Q31, Q39, Q40, Q45, Q47

### Routing

Decidir, antes de buscar, para qual corpus, índice ou ferramenta a pergunta vai (wiki, código, CRM, ou SQL em vez de busca vetorial). Erros de roteamento parecem falhas de retrieval nas métricas, então o router precisa de avaliação própria e de logs.

*Aparece em:* Q16, Q26, Q38, Q40, Q41

### Agentic RAG

A busca vira uma ferramenta que o modelo chama em loop: busca, lê, percebe lacunas, reformula e busca de novo. Ganha em perguntas multi-hop e exploratórias; perde em latência e custo. A falha típica é parar cedo demais, satisfeito com uma busca medíocre.

*Aparece em:* Q27, Q40, Q48, Q56, Q58

### GraphRAG

Na ingestão, um LLM extrai entidades e relações e monta um grafo de conhecimento; comunidades do grafo recebem resumos pré-escritos. Responde bem a perguntas relacionais e "globais" (temas do corpus inteiro), que o top-k não alcança. Indexação cara e atualização difícil.

*Aparece em:* Q1, Q39

### Self-RAG e Corrective RAG (CRAG)

Dois padrões em que o sistema avalia o próprio retrieval. *CRAG*: um avaliador leve classifica os documentos e, se forem ruins, aciona uma ação corretiva (por exemplo, busca na web). *Self-RAG*: o modelo é treinado para emitir "tokens de reflexão" que decidem se busca, se o trecho serve e se a resposta é suportada. As ideias úteis são avaliar o contexto antes de gerar e ter um caminho para "não achei nada bom".

*Aparece em:* Q37

### Text-to-SQL e semantic layer

Para contagens, filtros e agregados, o modelo gera uma consulta SQL em vez de buscar em vetores, porque índice vetorial não conta e aritmética em tokens é pouco confiável. Uma *semantic layer* define cada métrica uma só vez, reduzindo joins inventados. O SQL roda com as permissões do usuário.

*Aparece em:* Q26, Q30, Q38

### Multilingual e cross-lingual retrieval

Buscar quando query e documento estão em idiomas diferentes. O alinhamento dos embeddings é pior em idiomas de poucos recursos e o BM25 não cruza idiomas. Receita: detectar idioma, traduzir a *query* (não o corpus), usar reranker multilíngue e avaliar por par de idiomas.

*Aparece em:* Q7, Q36

### AST e code retrieval

Em código, o chunking segue a árvore sintática (AST): função, método, classe. Perguntas como "quem chama esta função?" são de grafo (call graph), não de similaridade. Para código, um agente com grep costuma competir com um índice vetorial.

*Aparece em:* Q35

### Temporal retrieval (recency e supersession)

Embeddings não têm noção de tempo: "política de 2024" e "de 2025" ficam quase iguais. Modele datas de validade e `superseded_by` nos metadados, filtre o que é incorreto e só então aplique um *decay* suave de recência.

*Aparece em:* Q14, Q27, Q34, Q47, Q55

## 8. Segurança, permissões e índice fresco

### ACL (Access Control List)

A lista de quem pode ver cada documento. Regra de ouro: a permissão é aplicada no *filtro de busca*, com a identidade verificada de quem pergunta, nunca pedindo ao LLM que não revele. O modelo não tem noção de autorização.

*Aparece em:* Q3, Q42, Q49, Q57, Q59

### Multi-tenancy

Um mesmo sistema atendendo vários clientes (tenants) com dados separados. O isolamento usa namespaces ou índices por tenant, o que também resolve o problema de filtro seletivo no ANN.

*Aparece em:* Q8, Q23, Q42, Q53, Q55

### Prompt injection (direta e indireta)

Texto que tenta dar ordens ao modelo. Na versão *indireta*, o ataque vem dentro de um documento que alguém escreveu (ticket, wiki) e que você indexou: ao ser recuperado, o modelo pode obedecê-lo. Não há correção confiável na camada do modelo: a defesa é arquitetural (limitar o que o agente pode fazer, tratar o texto recuperado como dado não confiável).

> 💡 **Pense assim:** Um bilhete plantado numa pilha de documentos: se o assistente obedece a tudo o que lê, quem escreveu o bilhete passa a mandar nele.

*Aparece em:* Q42, Q47, Q56

### Trust tiers e provenance

Marcar cada fonte com um nível de confiança (docs internos curados vs conteúdo que qualquer um escreve) e registrar de onde veio cada informação, para dar prioridade às fontes confiáveis e deixar o usuário ver qual fonte originou a resposta.

*Aparece em:* Q14, Q56

### MCP (Model Context Protocol)

Protocolo padrão para expor ferramentas a agentes de IA. Numa ferramenta de retrieval via MCP, o servidor deve aplicar as permissões do *usuário final* (derivadas do token validado), nunca de um ID que o modelo escreveu nos argumentos nem de uma conta de serviço ampla.

*Aparece em:* Q40, Q41, Q57

### Incremental sync e change detection

Atualizar o índice só com o que mudou (via webhooks, timestamps ou hashes), em vez de reconstruir tudo toda noite. Deletes e mudanças de permissão devem se propagar rápido: uma ACL revogada que ainda aparece no índice é bug de segurança.

*Aparece em:* Q3, Q14, Q29, Q35, Q39

### Blue-green migration e dual-write

Migrar de índice sem parar o serviço: constrói o novo ao lado do antigo, grava nos dois (dual-write) enquanto preenche o histórico (backfill), compara em sombra e só então troca o tráfego, mantendo o antigo para rollback.

*Aparece em:* Q51

## 9. Avaliação e depuração

### Golden set

Conjunto de perguntas reais com os chunks relevantes já rotulados, usado para medir o sistema a cada mudança. Pode começar com 50 a 100 queries e crescer; deve ser versionado junto com um snapshot do corpus.

*Aparece em:* Q34, Q43, Q45, Q51, Q53

### Recall@k

Dos chunks relevantes conhecidos, que fração aparece nos k primeiros resultados. É a métrica principal do retrieval, porque a geração não pode citar o que a busca nunca trouxe.

*Aparece em:* Q43, Q51, Q53, Q57, Q59

### MRR (Mean Reciprocal Rank)

Média de 1/posição do *primeiro* resultado relevante. Bom quando só importa achar o documento certo; cego ao que vem depois do primeiro acerto.

*Aparece em:* Q43

### nDCG

Soma a relevância (em graus) dos resultados com um desconto logarítmico pela posição, normalizada pela ordem ideal. Dá o retrato mais completo quando há relevância graduada, mas exige rótulos mais caros.

*Aparece em:* Q7, Q43, Q53, Q58

### Faithfulness / groundedness

Cada afirmação da resposta é sustentada pelos trechos recuperados? Mede-se quebrando a resposta em afirmações atômicas e verificando cada uma contra o contexto. É julgada contra o contexto, não contra a verdade: uma resposta pode ser fiel a chunks errados.

*Aparece em:* Q3, Q44, Q47, Q53

### RAGAS e LLM-as-judge

*LLM-as-judge* usa um LLM para avaliar respostas (fidelidade, relevância). *RAGAS* é um conjunto de métricas desse tipo. Para confiar no juiz: modelo forte e diferente do gerador, rubrica, calibração com humanos (concordância medida, por exemplo, pelo kappa de Cohen) e versão fixada.

*Aparece em:* Q13, Q44, Q45, Q53, Q58

### NLI / entailment

Verificar se um texto *implica* outro. Usado para checar se o trecho citado realmente sustenta a afirmação a que foi ligado.

*Aparece em:* Q28, Q44, Q60

### Citations: precision e recall

*Citation precision*: o trecho citado sustenta mesmo a afirmação? *Citation recall*: que fração das afirmações tem citação? Checar se o ID citado existe é grátis; checar se sustenta exige entailment.

*Aparece em:* Q28, Q44

### Retrieval miss vs generation miss

O primeiro corte de qualquer triagem: o chunk certo estava no prompt? Se não, a falha é de retrieval (parsing, chunking, query, índice, reranker). Se sim, é de geração (modelo ignorou, leu mal ou foi contradito). Muita "alucinação" é retrieval miss disfarçado.

*Aparece em:* Q40, Q43, Q46, Q47, Q54

### Abstain e selective prediction

Fazer o sistema dizer "não sei" quando o corpus não tem a resposta. Combina sinais (score do reranker, suficiência da evidência, grounding da resposta) e escolhe um limiar numa curva de cobertura × risco, medida com perguntas sem resposta *quase* cobertas pelo corpus.

*Aparece em:* Q10, Q37, Q40, Q53, Q60

### Interleaving e A/B test

*A/B test*: grupos de usuários veem sistemas diferentes. *Interleaving*: os resultados de dois rankers são misturados na mesma lista e vê-se de qual lado vêm os cliques. É justo quanto à posição e precisa de bem menos tráfego.

*Aparece em:* Q4, Q13, Q54

### Implicit feedback e feedback bias

Cliques e polegares são ouro e também enviesados: viés de posição, viés de apresentação (só se recebe feedback do que foi mostrado) e polegar para baixo que não diz onde foi a falha. Útil para minerar hard negatives e agrupar falhas.

*Aparece em:* Q43, Q53, Q54

### Confidence intervals e per-slice

Com poucas queries, uma variação de 1 ponto no recall é ruído, então se reportam intervalos de confiança. Além da média, olha-se por *fatia* (tipo de query, idioma, tenant), porque a média esconde regressões localizadas.

*Aparece em:* Q51, Q53, Q58, Q60

## 10. Produção: latência, custo, cache e escala

### Latência: TTFT e streaming

*TTFT* é o tempo até o primeiro token da resposta. Com *streaming*, o texto aparece conforme é gerado e uma geração de 6 s parece rápida. A geração do LLM domina latência e custo, então é por ela que se começa a otimizar.

*Aparece em:* Q48, Q49

### p95 / p99 (tail latency)

Percentis de latência: p95 é o tempo abaixo do qual ficam 95% das requisições. Em sistemas com vários shards, o p99 da query é o p99 do shard *mais lento*. *Hedged requests* reenviam a requisição lenta a outra réplica para cortar essa cauda.

*Aparece em:* Q29, Q37, Q48, Q55, Q58

### Embedding cache e retrieval cache

Guardar embeddings (chave: hash do conteúdo mais versão do modelo) e resultados de busca (chave: query mais permissões do usuário). Todo cache herda os requisitos do que ele pula: frescor e autorização.

*Aparece em:* Q29, Q49

### Semantic cache

Se uma pergunta nova for parecida o bastante com uma já respondida, devolve a resposta guardada e pula o pipeline todo. Ganho enorme, mas há risco de falso positivo ("cancelar" vs "renovar" assinatura ficam próximos), de resposta velha e de vazar resposta entre usuários com permissões diferentes.

*Aparece em:* Q29, Q42, Q49

### Sharding

Dividir o índice em vários pedaços em máquinas diferentes. Antes, esgote escalar uma máquina e quantizar. Estratégias: hash/aleatória (toda query vai a todos os shards), semântica, ou por atributo (tenant, idioma, tempo), que costuma ser a melhor quando se aplica.

*Aparece em:* Q22, Q55

### Scatter-gather e over-fetch

Em índice fragmentado, a query é enviada a todos os shards (scatter) e os resultados são combinados (gather). Como o top-k de cada shard com índice aproximado não compõe direto no top-k global, pede-se mais do que o necessário a cada shard (over-fetch).

*Aparece em:* Q23, Q55

### Tool design: search e fetch

Uma ferramenta de busca para agentes é desenhada para um modelo com orçamento de tokens: a descrição é o prompt, poucas ferramentas (um `search` com enum de fonte), 3 a 5 resultados curtos com ID estável e um `fetch(id)` para expandir, e erros que dizem o que tentar em seguida.

*Aparece em:* Q57
