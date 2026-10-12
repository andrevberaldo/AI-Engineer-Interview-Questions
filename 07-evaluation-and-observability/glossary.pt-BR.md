# 📚 Glossário de estudo — Evals & Observability

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Conceitos de base: evals, métricas e o ciclo de melhoria](#1-conceitos-de-base-evals-métricas-e-o-ciclo-de-melhoria)
- [2. Observabilidade e tracing](#2-observabilidade-e-tracing)
- [3. Corretores (graders) e métodos de avaliação](#3-corretores-graders-e-métodos-de-avaliação)
- [4. Estatística para evals](#4-estatística-para-evals)
- [5. Dados de avaliação: construção, versionamento e manutenção](#5-dados-de-avaliação-construção-versionamento-e-manutenção)
- [6. Benchmarks públicos e comparação entre modelos](#6-benchmarks-públicos-e-comparação-entre-modelos)
- [7. RAG e recuperação](#7-rag-e-recuperação)
- [8. Agentes, código e sistemas complexos](#8-agentes-código-e-sistemas-complexos)
- [9. Produção: CI, testes online, monitoramento e segurança](#9-produção-ci-testes-online-monitoramento-e-segurança)

## 1. Conceitos de base: evals, métricas e o ciclo de melhoria

### Eval (eval suite)

Um teste automatizado de qualidade para um sistema com LLM: um conjunto de exemplos de entrada, o que se espera como resultado e um corretor (grader) que dá a nota. Funciona como os testes unitários do software comum, com a diferença de que aqui a própria definição de "certo" é vaga, então o eval set acaba sendo a especificação.

> 💡 **Pense assim:** Uma prova com gabarito: cada vez que você muda o prompt ou o modelo, o sistema refaz a prova para ver se a nota subiu ou caiu.

*Aparece em:* Q1, Q9, Q46, Q51, Q54

### Eval-driven development

O ciclo de trabalho baseado em evals: colocar no ar, observar os traces de produção, analisar as falhas, transformar essas falhas em novos casos do eval set, mudar o prompt/modelo/pipeline, rodar os evals e só então publicar. O ponto central é que as falhas reais voltam para o conjunto de testes.

*Aparece em:* Q1, Q42

### Vibes (avaliar "no olhômetro")

Julgar a qualidade pela impressão pessoal ("pareceu bom"), por exemplo testando cinco prompts favoritos e publicando. É o nível mais imaturo de avaliação: não dá para saber se uma mudança melhorou ou piorou de verdade.

*Aparece em:* Q1, Q46

### Regression (regressão) e regression test

Regressão é quando algo que antes funcionava passa a falhar depois de uma mudança. Um regression test é um caso guardado justamente para avisar se isso voltar a acontecer. No fluxo de evals, cada falha de produção corrigida vira um regression test permanente.

*Aparece em:* Q1, Q7, Q25, Q46, Q47

### Eval case (caso de teste)

Um exemplo individual do eval set: a entrada exatamente como o sistema a recebe, o estado do ambiente necessário, o comportamento esperado (resposta de referência ou rubrica), o corretor e metadados (tags, origem, data). Um bom caso é aquele em que dois especialistas, lendo de forma independente, chegariam ao mesmo veredito.

*Aparece em:* Q9, Q14, Q27, Q28, Q29

### Guardrail metrics vs quality metrics

Métricas de *qualidade* são o que você quer melhorar (correção, resolução da tarefa, fidelidade). Métricas *guardrail* são limites que nunca podem piorar, por mais que a qualidade suba: taxa de jailbreak, vazamento de PII, conteúdo nocivo, latência p95, custo. Uma piora de guardrail bloqueia a publicação automaticamente; uma queda pequena de qualidade é negociável.

> 💡 **Pense assim:** Qualidade é a velocidade que você quer aumentar; guardrail é o cinto de segurança, que não se tira para ir mais rápido.

*Aparece em:* Q7, Q25, Q26, Q32, Q42

### Saturation (saturação)

Quando quase todos os modelos ou prompts já acertam quase tudo num teste (por exemplo, 95% ou mais). O teste deixa de separar o bom do ruim e, portanto, deixa de informar. A saída é aposentar os casos fáceis para um conjunto de smoke test e criar casos mais difíceis. Esses casos fáceis aposentados formam o *smoke test*: um conjunto pequeno e barato que só detecta quebras grosseiras.

*Aparece em:* Q4, Q19, Q28, Q44, Q52

### Goodhart's law

"Quando uma medida vira meta, ela deixa de ser uma boa medida." Se você otimiza o sistema contra uma métrica, ele aprende a enganá-la: o juiz LLM vê respostas mais longas e confiantes, o modelo faz hedge para ser "fiel", e a nota sobe sem a qualidade real subir. A defesa é combinar métricas em pares (fidelidade com completude), manter conjuntos de teste reservados e conferir com humanos. No treinamento, o mesmo fenômeno se chama *reward hacking*: o modelo descobre como obter nota alta do avaliador (reward model ou juiz) sem fazer de fato o que se queria.

> 💡 **Pense assim:** Se a escola paga bônus por nota de prova, os alunos passam a estudar para a prova, não para aprender.

*Aparece em:* Q19, Q28, Q40, Q44, Q54

## 2. Observabilidade e tracing

### Observability (observabilidade)

Registrar e poder consultar *o que aconteceu* em cada requisição: quais chamadas foram feitas, quanto tempo levaram, quantos tokens e quanto custaram, que erro ocorreu. Não diz se a resposta era boa; isso é papel dos evals. Difere dos evals: o que te acorda de madrugada (erro, p95, pico de custo) é observabilidade; o que bloqueia um merge (taxa de acerto no golden set) é eval. Os dados são os mesmos, por isso os fornecedores vendem as duas coisas juntas.

> 💡 **Pense assim:** O painel do carro mostra velocidade e temperatura, mas não diz se você está indo para o lugar certo.

*Aparece em:* Q9, Q42, Q46, Q49, Q55

### Trace e span

Um *trace* é o registro de uma requisição inteira do usuário; é uma árvore de *spans*, e cada span é uma unidade de trabalho com início, fim e atributos (uma chamada ao modelo, uma busca, uma execução de ferramenta, uma checagem de guardrail). É a mesma ideia do tracing distribuído de serviços.

> 💡 **Pense assim:** A nota fiscal de um pedido (trace) com cada item listado (spans).

*Aparece em:* Q9, Q13, Q34, Q42, Q53

### OpenTelemetry (OTel) e GenAI semantic conventions

OpenTelemetry é um padrão aberto de instrumentação. As GenAI semantic conventions definem nomes padronizados para descrever chamadas de LLM (por exemplo gen_ai.request.model e gen_ai.usage.input_tokens). Assim você troca de ferramenta de observabilidade sem reescrever o código. Elas cobrem a mecânica (modelo, tokens, latência), não a qualidade: notas de eval e vereditos de política são atributos seus.

*Aparece em:* Q8, Q13, Q42

### Ferramentas de tracing (LangSmith, Langfuse, Braintrust, Phoenix)

Plataformas que oferecem tracing, datasets e execução de evals num só lugar, com diferentes combinações de open source e hospedado. O que importa é a arquitetura, não o fornecedor.

*Aparece em:* Q13, Q42

### TTFT (time-to-first-token)

O tempo até o primeiro token da resposta aparecer. Com streaming, é o que o usuário sente como "demora", enquanto a latência total é o que bate nos timeouts. Convém medir os dois.

*Aparece em:* Q8, Q33, Q42, Q43

### Percentis de latência (p50, p95, p99)

p95 é o valor abaixo do qual ficam 95% das requisições: só 5% são mais lentas que ele. Percentis mostram a "cauda" que a média esconde. Uma média estável pode coexistir com um p99 terrível.

*Aparece em:* Q9, Q26, Q33, Q42, Q49

### Model alias vs versão exata (pinning)

Um alias (como um nome "latest") pode ser reapontado pelo provedor para outro modelo sem aviso. Por isso se faz *pinning*: fixar a versão exata do modelo (e do juiz) e registrá-la em cada trace, para que uma mudança de qualidade possa ser ligada a uma mudança de versão.

> 💡 **Pense assim:** Pedir "o vinho da casa" (alias) em vez de uma safra específica: o restaurante pode trocar o vinho sem avisar.

*Aparece em:* Q8, Q37, Q42, Q43, Q45

### Join keys (trace ID)

Identificadores (como o trace ID) que permitem ligar eventos separados, por exemplo juntar um thumbs-down ao trace exato que o causou, com a versão do prompt e os chunks recuperados. Sem essa ligação, a análise de erros não funciona.

*Aparece em:* Q8, Q12, Q14, Q42, Q46

### Prompt caching (cache hit)

Provedores reaproveitam o início do prompt quando ele se repete, cobrando menos. Uma edição no começo do prompt pode invalidar esse cache e multiplicar o custo sem mudança visível. Por isso a taxa de cache hit entra no monitoramento.

*Aparece em:* Q8, Q33, Q42, Q43

### Reasoning tokens (e thinking budget)

Em modelos de raciocínio, são os tokens gastos "pensando" antes da resposta. Eles são cobrados e aumentam a latência, mas o usuário não os vê. Por isso entram como dimensão própria nos traces, e a qualidade passa a ser uma curva de nota contra orçamento de raciocínio, não um número único.

*Aparece em:* Q33, Q49

## 3. Corretores (graders) e métodos de avaliação

### Code-graded assertion

Verificação feita por código, de forma determinística: igualdade exata, regex, validação de JSON, tolerância numérica, execução de testes. É grátis, rápida e repetível, e deve ser usada sempre que a resposta puder ser checada objetivamente.

*Aparece em:* Q2, Q3, Q39, Q46, Q48

### Exact match, regex e tolerância numérica

Verificações simples de texto e número. *Exact match* só aceita saída idêntica à referência: serve para espaço fechado (rótulo, campo extraído), mas é frágil a formatação ("$4.50" vs "4.5 dollars"), então normalize antes. *Regex/contains* procura um padrão na saída, com risco de falso positivo ("the answer is not 42" contém "42"). *Tolerância numérica* aceita uma pequena diferença em vez de exigir texto idêntico.

*Aparece em:* Q2, Q3, Q10, Q14

### Structural validation (JSON schema, Pydantic)

Fazer o parsing da saída e checar se ela é um JSON válido que segue o schema esperado (campos obrigatórios, valores permitidos). É o corretor mais usado em extração e chamada de ferramentas.

*Aparece em:* Q2, Q3, Q10

### Execution-based eval e hidden tests

Avaliar código executando-o contra testes, e não comparando texto. Os *hidden tests* (testes ocultos) são usados só na correção; se o modelo (ou você, iterando) enxergar os testes, passa a se ajustar a eles, por exemplo escrevendo as respostas esperadas dentro do código.

*Aparece em:* Q3, Q6, Q10, Q21, Q54

### Reference-based vs reference-free

Reference-based compara a saída com uma resposta de ouro (exact match, BLEU, ROUGE). Reference-free julga a saída por si, contra um critério ou contra o contexto: executar o código, validar um schema, um juiz avaliando fidelidade aos chunks recuperados.

*Aparece em:* Q3, Q10

### BLEU, ROUGE e BERTScore

Métricas antigas que medem a sobreposição de palavras (n-gramas) entre a saída e a referência, criadas para tradução e resumo. Falham em respostas de LLM porque há muitas formas corretas de dizer a mesma coisa, e elas não sabem se a afirmação é verdadeira. BERTScore e a similaridade de embeddings capturam paráfrases melhor, mas são insensíveis a negação, números e troca de entidades: "A dose é 5mg" e "A dose não é 5mg" ficam muito próximas.

*Aparece em:* Q3, Q10

### LLM-as-judge

Usar um LLM (geralmente mais forte) para avaliar saídas com base em uma rubrica, emitindo aprovado/reprovado, uma nota ou uma preferência entre duas respostas. Útil para critérios subjetivos, mas ruidoso e enviesado; precisa ser calibrado contra rótulos humanos antes de ser confiável.

> 💡 **Pense assim:** Um professor-assistente que corrige redações: rápido e barato, mas você precisa conferir se ele corrige como o professor titular.

*Aparece em:* Q2, Q5, Q16, Q17, Q37

### Rubric (rubrica)

As instruções escritas que dizem ao juiz (ou ao anotador humano) como decidir: o que significa passar e o que significa falhar em cada critério, com exemplos. Uma boa rubrica transforma um objetivo vago em um procedimento que um estranho cuidadoso conseguiria seguir. Os *few-shot anchors* são exemplos incluídos na rubrica que mostram, em casos limítrofes, o que é aprovado e o que é reprovado; calibram o juiz melhor que qualquer adjetivo.

*Aparece em:* Q15, Q17, Q18, Q30, Q54

### Chain-of-thought judging

Exigir que o juiz escreva sua análise *antes* do veredito. Isso força a engajar com o conteúdo, melhora a concordância com humanos e deixa uma justificativa auditável para cada nota.

*Aparece em:* Q5, Q15, Q16, Q18

### Pointwise vs pairwise

*Pointwise*: o juiz dá uma nota absoluta a uma resposta ("de 1 a 10"). *Pairwise*: o juiz escolhe a melhor entre duas (A, B ou empate). Comparar é mais fácil e confiável do que dar nota absoluta, que costuma ser mal ancorada e variar entre execuções.

> 💡 **Pense assim:** Dar nota de 0 a 10 a um vinho é difícil; dizer qual de dois copos é melhor é fácil.

*Aparece em:* Q2, Q5, Q15, Q35, Q40

### Binary pass/fail por critério

Em vez de uma nota geral, usar um veredito sim/não para cada critério isolado ("cita só fontes recuperadas?"). É bem mais consistente do que escalas de 1 a 10, e as notas compostas podem ser calculadas em código.

*Aparece em:* Q5, Q15, Q18, Q35, Q55

### Position bias

Na comparação pairwise, o juiz favorece sistematicamente uma posição, geralmente a primeira resposta. A mitigação é julgar duas vezes trocando a ordem e só contar a vitória se o veredito for consistente (senão, empate).

> 💡 **Pense assim:** Como o primeiro candidato a falar numa entrevista, que tende a ser lembrado melhor: para neutralizar, inverte-se a ordem e vê-se se o vencedor continua o mesmo.

*Aparece em:* Q15, Q16

### Verbosity bias

O juiz tende a dar notas maiores a respostas mais longas, independentemente da qualidade. Mitigação: rubrica que penaliza enrolação, e reportar o tamanho das respostas junto da taxa de vitória.

*Aparece em:* Q16, Q18, Q40, Q44, Q51

### Self-preference bias

Modelos tendem a preferir as próprias saídas (ou as da mesma família). Mitigação: usar como juiz um modelo de outra família, ou um painel de juízes de famílias diferentes.

*Aparece em:* Q2, Q5, Q16, Q24

### Judge calibration

Comparar os vereditos do juiz com rótulos de humanos (50 a 100 exemplos, incluindo casos limítrofes) e medir a concordância. Só se automatiza quando ela é comparável à concordância entre os próprios humanos, que é o teto realista. Recalibra-se ao trocar juiz, rubrica ou tarefa.

> 💡 **Pense assim:** Aferir a balança com um peso conhecido antes de confiar no que ela mostra.

*Aparece em:* Q2, Q17, Q37, Q46, Q53

### Judge pinning e judge sanity set

O juiz também é prompt mais modelo: se a versão muda, as notas mudam sem que o produto tenha mudado. Por isso se fixa a versão exata, registra-se em cada resultado e se roda um pequeno conjunto de itens com veredito inequívoco antes de cada lote para detectar um juiz desregulado.

*Aparece em:* Q15, Q17, Q25, Q28, Q37

### Human evaluation

Rótulos de especialistas ou anotadores. Lenta e cara, então é usada para criar o ground truth que calibra os juízes, para domínios de alto risco e como auditoria periódica, não para corrigir toda execução.

*Aparece em:* Q2, Q29, Q35, Q53, Q54

## 4. Estatística para evals

### Standard error e intervalo de confiança

Toda nota de eval é uma amostra e vem com margem de erro. Para uma taxa de acerto p em n exemplos, o erro padrão é aproximadamente raiz de p(1-p)/n, e o intervalo de confiança de 95% é cerca de duas vezes isso para cada lado. Com 100 exemplos e 82% de acerto, o intervalo vai de uns 74% a 90%: diferenças de poucos pontos são ruído. Quando os exemplos não são independentes (várias perguntas do mesmo documento), usam-se *clustered standard errors*, que são maiores que a fórmula ingênua.

> 💡 **Pense assim:** Pesquisa eleitoral com margem de erro: 82% ± 8 pontos não se compara com 78%.

*Aparece em:* Q11, Q25, Q41, Q45, Q46

### Noise floor (variabilidade entre execuções)

O quanto a nota varia só por acaso ao rodar a mesma configuração de novo. Mede-se repetindo a execução idêntica várias vezes; essa variação é o limiar mínimo para considerar uma mudança real.

*Aparece em:* Q7, Q11, Q25, Q47, Q52

### Minimum detectable effect e statistical power

Poder estatístico é a chance de detectar uma melhoria real quando ela existe. O *minimum detectable effect* é a menor melhoria que seu teste consegue enxergar com um dado n. Detectar diferenças pequenas exige muitos exemplos (milhares para 3 pontos, em comparação independente); por isso se calcula o n necessário em vez de chutar 100.

*Aparece em:* Q4, Q31, Q41, Q45, Q52

### Paired comparison (McNemar, paired bootstrap)

Quando duas versões rodam nos *mesmos* exemplos, só importam os casos em que o resultado mudou (os pares discordantes). O teste de McNemar (ou um bootstrap das diferenças por caso) compara esses flips e elimina a variância da dificuldade dos exemplos, reduzindo bastante o n necessário. O *bootstrap* reamostra os resultados com reposição milhares de vezes para estimar a incerteza sem fórmula; os percentis 2,5 e 97,5 viram o intervalo de confiança.

> 💡 **Pense assim:** Em vez de comparar a nota média de duas turmas, comparar a nota de cada aluno antes e depois.

*Aparece em:* Q29, Q31, Q41, Q45, Q47

### Inter-annotator agreement (kappa e alpha)

O quanto anotadores humanos concordam entre si. A concordância percentual simples engana quando uma classe domina (todos dizem "pass" e concordam 90%). O *Cohen's kappa* corrige pelo acaso para dois anotadores; *Fleiss* serve para mais de dois; *Krippendorff's alpha* aceita rótulos faltando e escalas ordinais. A concordância entre humanos é o teto do que um juiz pode atingir. Para notas em escala, compara-se a ordem dos itens com correlação de postos (Spearman, Kendall) em vez dos valores brutos.

*Aparece em:* Q5, Q17, Q30

### Multiple comparisons

Ao testar muitas fatias de uma vez, algum alarme falso aparece por puro acaso: com 12 fatias a 5% de significância, a chance de pelo menos um alarme falso é de cerca de 46%. Por isso resultados por fatia servem de triagem (quais transcrições ler), não de gate automático, ou exigem correção.

*Aparece em:* Q47

### Slice (fatia)

Um subconjunto dos dados definido por uma dimensão relevante (idioma, plano do cliente, intenção, tamanho do contexto). Reportar a nota por fatia evita que uma média geral esconda um segmento em queda. Para ter poder em cada fatia, o conjunto é construído com um mínimo de exemplos por fatia (*stratified sampling*), super-representando as raras, e depois reponderado para a mistura real.

*Aparece em:* Q28, Q47, Q48, Q51, Q53

### pass@k

A probabilidade de pelo menos uma de k amostras resolver o problema. É o padrão em geração de código, em que cada amostra é verificável por testes. O jeito ingênuo (gerar só k amostras) tem alta variância; o estimador não enviesado do paper do Codex gera n ≥ k amostras, conta as c corretas e calcula 1 − C(n−c, k)/C(n, k).

*Aparece em:* Q6, Q19, Q21, Q39, Q45

### pass^k

A probabilidade de *todas* as k tentativas darem certo. Mede consistência, que importa em agentes: um agente com 80% de acerto por tentativa tem pass^8 de aproximadamente 17%. O τ-bench, benchmark de agentes cujo resultado é verificado pelo estado final do ambiente, popularizou essa métrica.

*Aparece em:* Q6, Q39, Q45, Q50

### Nondeterminism (temperature, sampling)

O mesmo input pode dar saídas diferentes por causa de temperature e sampling (e até com temperature 0, por detalhes da infraestrutura). Por isso se repete cada caso k vezes, se avalia na mesma temperature da produção e se reportam barras de erro.

> 💡 **Pense assim:** Jogar o mesmo dado várias vezes: um resultado isolado não diz nada, a média de várias jogadas sim.

*Aparece em:* Q6, Q8, Q21, Q41, Q45

## 5. Dados de avaliação: construção, versionamento e manutenção

### Dataset versioning e provenance

Uma nota é função de tudo que a produziu: exemplos, respostas esperadas, rubrica, código do corretor, prompt e versão do juiz, parâmetros de amostragem. Tudo isso é versionado junto e registrado em cada resultado. *Provenance* é a origem de cada exemplo (trace de produção, escrito à mão, sintético e de qual gerador, quem revisou).

*Aparece em:* Q12, Q25, Q37, Q45, Q54

### Frozen regression suite vs development set

Dois conjuntos separados: um congelado, que muda pouco e serve de gate de merge, e um vivo, que absorve as falhas novas de produção. Misturar os dois impede distinguir uma regressão real de um conjunto que ficou mais difícil.

*Aparece em:* Q12, Q28, Q43

### Dev/test split e overfitting ao eval set

Assim como em ML, iterar prompts só contra a parte "dev" e usar a parte "test" apenas para decisões de lançamento. Se os dois divergem, você se ajustou demais ao dev. Meses de iteração contra os mesmos 200 casos geram melhoras específicas para eles.

> 💡 **Pense assim:** O aluno que treina só com as provas antigas e parece ótimo até aparecer uma prova nova.

*Aparece em:* Q19, Q21, Q28, Q44, Q54

### Synthetic data

Casos gerados por um LLM para cobrir lacunas de cobertura. Úteis para bootstrap, mas tendem a ser mais limpos e fáceis que os usuários reais, e precisam de revisão humana antes de entrar no conjunto. Para gerar bem: montar uma *scenario taxonomy* (intenção × persona × dificuldade × idioma) e amostrar combinações. Cuidados: *mode collapse* (o LLM repete o caso mediano, combatido com diversidade e remoção de duplicatas) e circularidade entre gerador e juiz (mesma família em todos os papéis infla as notas).

*Aparece em:* Q4, Q12, Q14, Q28, Q29

### Benchmark contamination

Quando os dados de teste vazam para o treino: o modelo já viu as perguntas (ou paráfrases), e a nota mede memorização, não capacidade. Para seu eval set, mantenha-o privado e renove com dados de produção, que por construção são posteriores ao treino. O GSM1k, com problemas novos no estilo do GSM8K, mostrou quedas de vários pontos em algumas famílias de modelos.

> 💡 **Pense assim:** O aluno que viu a prova antes: a nota alta mede memória, não conhecimento.

*Aparece em:* Q19, Q20, Q28, Q40, Q44

### Detecção de contaminação (perturbação, n-gram, probes, split temporal)

Formas de detectar: reescrever as questões mudando a forma mas não a lógica (queda grande indica memorização), buscar sobreposição de n-gramas entre teste e corpus de treino, sondar se o modelo completa o gabarito textualmente, e avaliar com dados criados depois da data de corte do treino.

*Aparece em:* Q20

### Error analysis

Ler transcrições de falhas e agrupá-las em modos de falha, em vez de olhar só dashboards. Passo a passo: ler 30 a 50 casos anotando uma linha cada (*open coding*, sem categorias pré-definidas), montar a taxonomia, ranquear por frequência × gravidade × facilidade de correção, e consertar o maior grupo primeiro. Dashboards dizem *que* a qualidade caiu; transcrições dizem *por quê*.

> 💡 **Pense assim:** Em vez de olhar só a febre no termômetro (dashboard), examinar o paciente para achar a causa.

*Aparece em:* Q1, Q27, Q42, Q43, Q45

### Escaped defects

Bugs de qualidade que chegaram aos usuários sem a suíte de evals detectar. Quantos dos últimos 10 bugs a suíte teria pego é a melhor métrica de saúde de um eval set, pois é baseada na realidade.

*Aparece em:* Q52

### Discriminative power (crippled config)

A capacidade da suíte de separar sistemas que você sabe que são diferentes. Teste: rode uma configuração propositalmente piorada (modelo mais fraco, retrieval desligado, instrução-chave removida). Se a nota quase não muda, a suíte é decorativa.

*Aparece em:* Q28, Q52

### Flaky (item instável)

Um teste cujo resultado muda de uma execução para outra sem mudança no sistema, por exemplo por dependência de horário, aleatoriedade ou ordem. Evals instáveis acabam ignorados, como testes instáveis; devem ser isolados e diagnosticados.

*Aparece em:* Q11, Q21, Q45, Q52

## 6. Benchmarks públicos e comparação entre modelos

### MMLU, GSM8K, HumanEval e MT-Bench

Quatro benchmarks clássicos. *MMLU*: prova de múltipla escolha de 57 áreas, mede conhecimento amplo. *GSM8K*: cerca de 8,5 mil problemas de matemática de ensino fundamental, em texto. *HumanEval*: 164 problemas de funções em Python corrigidos por execução (pass@k), pequeno e pouco representativo de software real. *MT-Bench*: 80 prompts de conversa em vários turnos julgados por um LLM, herdando os vieses de juiz. Os três primeiros estão saturados no topo e contaminados.

*Aparece em:* Q16, Q17, Q19, Q20, Q21

### SWE-bench (Verified, Pro)

Benchmark em que o agente recebe issues reais do GitHub e precisa gerar um patch que passe nos testes ocultos do repositório. O Verified é um subconjunto de 500 tarefas validadas por humanos. As notas dependem muito do harness (scaffolding, tentativas, orçamento de compute), então comparações entre modelos podem estar medindo o framework do agente. O texto registra que o SWE-bench Verified perdeu valor e que a OpenAI passou a apontar para o SWE-bench Pro, mais difícil. A correção segue o padrão *fail-to-pass* (os testes que falhavam passam depois do patch) e *pass-to-pass* (os que já passavam continuam passando, o que pega regressões).

*Aparece em:* Q19, Q20, Q21, Q39

### Chatbot Arena (LMArena)

Plataforma em que pessoas enviam um prompt, dois modelos anônimos respondem lado a lado e o voto na melhor resposta alimenta um ranking. Mede preferência humana em prompts abertos e resiste a contaminação, mas preferência não é correção (respostas longas e bem formatadas ganham) e o público votante não representa todos os usos. O *style control* é um ajuste estatístico que remove do ranking o efeito de tamanho e formatação da resposta.

*Aparece em:* Q15, Q16, Q40, Q44

### Elo e Bradley-Terry

Modelos estatísticos para transformar votos de "A venceu B" em um ranking de força. O Bradley-Terry modela a chance de A vencer B como função da diferença de força e é ajustado por máxima verossimilhança sobre todos os votos; é preferível ao Elo incremental, cujo resultado depende da ordem dos votos.

*Aparece em:* Q15, Q40

## 7. RAG e recuperação

### RAG e chunks

Retrieval-Augmented Generation: o sistema busca trechos de documentos (os *chunks*) e os coloca no prompt para o modelo responder com base neles. Avaliar um RAG exige olhar a busca e a geração separadamente.

*Aparece em:* Q22, Q23, Q24, Q29, Q43

### Retrieval metrics: recall@k

A fração das consultas em que um chunk relevante aparece entre os k primeiros resultados. É a métrica principal de RAG porque o gerador só enxerga a janela de k chunks: se a resposta não foi recuperada, nada depois consegue consertar. Define o teto da qualidade fim a fim.

*Aparece em:* Q22, Q23

### MRR, nDCG e precision@k

Métricas de ranking. *MRR*: média de 1/posição do primeiro resultado relevante, útil quando a posição 1 é especial (ignora o que vem depois). *nDCG*: premia colocar os itens mais relevantes no topo e aceita relevância em graus; é a métrica natural de um reranker, mas custa mais para rotular. *Precision@k*: quanto do recuperado é de fato relevante; chunks irrelevantes diluem a atenção, gastam tokens e abrem espaço para alucinação.

*Aparece em:* Q22, Q23

### Reranker

Um segundo estágio que reordena os candidatos da busca inicial (por exemplo, recupera 50 e mantém os 10 melhores) usando um modelo mais preciso.

> 💡 **Pense assim:** Uma segunda peneira: a primeira separa 50 candidatos rápido, a segunda escolhe com mais cuidado os 10 melhores.

*Aparece em:* Q22, Q23

### Pooling problem

Em avaliação de recuperação, chunks não rotulados costumam ser tratados como irrelevantes. Um recuperador novo que acha chunks bons que seus rótulos não cobrem é injustamente punido. Por isso se auditam à mão alguns "erros" antes de concluir que um embedder é pior.

*Aparece em:* Q23

### Faithfulness (groundedness)

Se cada afirmação da resposta é sustentada pelo contexto recuperado. É a métrica anti-alucinação de RAG e é diferente de correção: uma resposta pode ser fiel a um documento errado. Mede-se decompondo a resposta em afirmações atômicas e verificando cada uma contra os chunks (suportada, contradita ou não encontrada). Deve ser pareada com relevância, pois uma resposta que diz "não sei" é perfeitamente fiel. *Atomic claims* são os fatos mínimos da resposta, verificados um a um (a tarefa de NLI, decidir se um texto implica ou contradiz outro, antecedeu os juízes LLM nisso). *Answer relevance* checa se a resposta trata da pergunta. O RAGAS é um framework que empacota essas métricas.

> 💡 **Pense assim:** Um resumo fiel ao livro: pode até ser chato, mas não inventa nada que o livro não diga.

*Aparece em:* Q2, Q18, Q22, Q24, Q44

### Interleaving

Técnica online para ranqueadores: mistura os resultados dos dois ranqueadores numa mesma lista e observa de qual lado vêm os cliques. Por ser comparação dentro do mesmo usuário, chega à significância com muito menos tráfego que um A/B.

> 💡 **Pense assim:** Misturar as figurinhas de dois álbuns numa mesma pilha e ver de qual álbum as pessoas escolhem mais.

*Aparece em:* Q2, Q26

## 8. Agentes, código e sistemas complexos

### Sandbox

Ambiente isolado onde se roda código não confiável ou um agente: sem rede, com limites de CPU/memória e timeouts. Nunca execute saída de modelo no ambiente do seu CI. Para agentes, o sandbox funciona como *environment simulator*: ferramentas e estado inicial realistas, com reset, onde o agente age sem tocar produção. Em tarefas de vários turnos, um LLM faz o papel do cliente (usuário simulado), o que acrescenta variância própria.

> 💡 **Pense assim:** Uma caixa de areia: dá para brincar e fazer bagunça sem estragar nada fora dela.

*Aparece em:* Q21, Q39, Q45, Q50, Q55

### Outcome vs trajectory evaluation

Avaliar o *resultado* (a tarefa foi concluída? verificado no estado final do ambiente) ou o *caminho* (escolha de ferramenta, argumentos, eficiência, recuperação de erros, segurança). O resultado é o padrão-ouro quando checável; a trajetória é a camada de diagnóstico que explica por que falhou.

*Aparece em:* Q35, Q39, Q50

### Milestone grading

Dividir tarefas longas em marcos verificáveis (teste reproduzido, correção compila, migração aplicada) e pontuar o progresso. Em vez de uma parede de zeros, dá um gradiente de melhoria.

*Aparece em:* Q39, Q55

### Time horizon

Enquadramento que agrupa tarefas pelo tempo que um humano habilidoso levaria e mostra a taxa de sucesso por faixa. O comprimento em que o agente acerta metade das vezes resume sua capacidade. O texto atribui a popularização do conceito ao METR.

*Aparece em:* Q55

### Deterministic replay harness

Gravar todos os pontos não determinísticos de uma execução real (respostas de ferramentas, completions, relógio, seeds) e reproduzi-los depois, rodando seu código de orquestração de verdade. Dá testes rápidos, sem custo e sem flakes, e isola bugs de orquestração do acaso do modelo. Mas só cobre caminhos gravados: se o prompt mudar, as chamadas mudam e a gravação não serve, então não mede se um prompt novo é melhor.

> 💡 **Pense assim:** Reassistir à gravação de uma partida: dá para analisar cada lance, mas não para saber como o jogo seria com outra tática.

*Aparece em:* Q50, Q55

### Context compaction

Quando o contexto de um agente longo fica grande, parte dele é resumida ou descartada. Muitas falhas em tarefas longas vêm de o agente perder uma restrição após a compactação, o que nenhuma métrica de tokens ou latência mostra.

*Aparece em:* Q55

### Router (e cascade)

Um *router* decide qual modelo atende cada consulta (barato para as fáceis, de fronteira para as difíceis). Uma *cascade* tenta o barato primeiro e escala se a confiança for baixa. O router é avaliado como troca entre custo e qualidade, não como classificador. O *oracle router* é o teto: manda ao modelo forte só o que o barato erra. O erro assimétrico é o *routed-down failure* (mandar ao barato uma consulta que ele erra); mandar para cima sem necessidade só custa dinheiro.

*Aparece em:* Q38, Q53

### Distilled evaluator

Um avaliador pequeno e barato, treinado com os rótulos de um juiz forte e de humanos, para pontuar muito mais tráfego. Herda os vieses do professor, pode degradar em intenções novas e precisa ser calibrado contra humanos.

*Aparece em:* Q53

### Perturbation robustness e OCR

Em avaliação de documentos e imagens, degradar a entrada (reduzir resolução, comprimir, girar poucos graus) e conferir se a resposta se mantém. Se mudar, a feature é frágil, mesmo com boa nota nominal. Em documentos escaneados, separa-se *perceber* (ler a página, avaliado como extração de campos, em que entra o OCR) de *raciocinar* (responder dado o que foi lido): as falhas e as correções são diferentes.

*Aparece em:* Q48

### Reinforcement fine-tuning (reward grader)

Usar os corretores do eval como sinal de recompensa para treinar o modelo. Muda tudo: o corretor passa a enfrentar um otimizador que procura seus pontos cegos, então precisa ser robusto a saídas adversariais, e o eval deixa de ser independente se usar o mesmo corretor nos mesmos prompts.

*Aparece em:* Q54

## 9. Produção: CI, testes online, monitoramento e segurança

### CI gating (portão de CI)

Rodar a suíte de evals a cada mudança de prompt, modelo ou retrieval, antes do merge. Guardrails bloqueiam de forma rígida; qualidade bloqueia só se a queda passar do limite de ruído. Usa-se uma suíte pequena (50 a 200 casos, minutos) no PR e a completa à noite, e o relatório mostra os casos que mudaram de resultado.

*Aparece em:* Q15, Q25, Q41, Q46, Q47

### Online evaluation e A/B test

Avaliação com usuários reais. No A/B, os usuários (não as requisições) são sorteados entre a versão atual e a candidata, e se compara uma métrica de sucesso real mais os guardrails. Os evals offline preveem; o teste online confirma. Convém rodar por uma ou duas semanas por causa do *novelty effect*: o interesse inicial por algo novo distorce os números da primeira semana.

*Aparece em:* Q2, Q7, Q26, Q32, Q46

### Implicit feedback signals

Sinais indiretos de satisfação já presentes no uso: regenerar a resposta, copiar, aceitar o código, quanto o usuário edita a saída, reformular a pergunta, abandonar, escalar para humano. São mais abundantes que thumbs, que vêm de uma fatia pequena e enviesada das sessões.

*Aparece em:* Q26, Q28, Q35, Q42, Q53

### Shadow, canary e A/B

Estágios de lançamento com exposição crescente. *Shadow*: o candidato recebe tráfego espelhado, mas ninguém vê a saída (cuidado com ferramentas que têm efeitos colaterais). *Canary*: uma fatia pequena (1% a 5%) de usuários reais, com rollback automático; é sobre raio de dano, não estatística. *A/B*: o teste randomizado e com poder que diz se é melhor para os usuários.

> 💡 **Pense assim:** Shadow é ensaiar o show sem plateia; canary é uma estreia para meia dúzia de convidados; A/B é a temporada aberta com as duas versões lado a lado.

*Aparece em:* Q28, Q32, Q43

### Drift (modelo, dados, saída, custo)

Mudança ao longo do tempo. *Model drift*: o provedor altera o modelo. *Data/input drift*: o tráfego muda (idiomas, intenções, superfícies). *Output drift*: as respostas mudam (tamanho, taxa de recusa, mix de ferramentas). *Cost/latency drift*: crescimento do prompt, mais passos de agente, queda do cache. Cada um precisa de seu detector. Contra o drift do modelo usa-se um *scheduled canary eval*: sondas fixas rodadas em horário fixo contra a configuração de produção.

> 💡 **Pense assim:** A bússola que vai se desviando devagar: você só nota quando já está longe do rumo.

*Aparece em:* Q17, Q30, Q43, Q53, Q54

### PII e pseudonymisation

PII é informação pessoal identificável (nomes, e-mails, cartões). Em traces, redige-se na ingestão (antes de gravar). Em vez de apagar tudo com [REDACTED], faz-se pseudonimização: troca-se o valor por um token estável (derivado de um HMAC com chave), de modo que "PERSON_a91f" continue identificável ao longo da conversa sem guardar o nome. Peças técnicas: HMAC é um hash com chave, que gera o token estável; NER (named entity recognition) acha nomes e endereços em texto livre; o algoritmo de Luhn valida números de cartão.

*Aparece em:* Q7, Q8, Q25, Q34, Q42

### Retention e deletion requests

Política de por quanto tempo guardar os dados e capacidade de atender pedidos de exclusão de usuários. Indexar traces por ID de usuário torna a exclusão uma operação por chave, e não uma varredura completa.

*Aparece em:* Q8, Q12, Q32, Q34, Q42

### Human review queue

Fila em que humanos revisam traces selecionados. Amostra-se por quatro fluxos: falhas suspeitas, sinais negativos implícitos, rotas de alto risco e uma pequena amostra uniformemente aleatória (≈5%), que é a única que dá estimativa não enviesada da taxa real de falha. Os rótulos devem voltar para calibração e para o eval set.

*Aparece em:* Q35

### Red teaming

Exercício em que pessoas (ou ferramentas) tentam deliberadamente quebrar o assistente. O valor só se mantém se cada achado virar caso de teste permanente, expandido em variações (paráfrases, outros idiomas, framing diferente) para não corrigir apenas a string literal. Acompanha-se a *attack success rate*, a fração das tentativas que violaram a política, como tendência.

> 💡 **Pense assim:** Contratar um ladrão de mentira para testar a fechadura da casa.

*Aparece em:* Q4, Q7, Q36, Q54

### Jailbreak e prompt injection

*Jailbreak*: induzir o modelo a violar suas políticas. *Prompt injection*: instruções maliciosas escondidas em dados que o modelo lê (um documento recuperado, a saída de uma ferramenta) e que ele obedece como se fossem do usuário. A forma "indireta" vem de conteúdo recuperado, não digitada pelo usuário.

*Aparece em:* Q7, Q25, Q29, Q36
