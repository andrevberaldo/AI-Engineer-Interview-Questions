# 📚 Glossário de estudo — Prompt Engineering & Context Engineering

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Conceitos básicos de prompting](#1-conceitos-básicos-de-prompting)
- [2. In-context learning e exemplos](#2-in-context-learning-e-exemplos)
- [3. Técnicas de reasoning](#3-técnicas-de-reasoning)
- [4. Sampling e geração](#4-sampling-e-geração)
- [5. Structured outputs](#5-structured-outputs)
- [6. Contexto longo, caching e gerenciamento de contexto](#6-contexto-longo-caching-e-gerenciamento-de-contexto)
- [7. RAG, grounding e instruções conflitantes](#7-rag-grounding-e-instruções-conflitantes)
- [8. Segurança: injection, vazamento e guardrails](#8-segurança-injection-vazamento-e-guardrails)
- [9. Avaliação, otimização e produção](#9-avaliação-otimização-e-produção)

## 1. Conceitos básicos de prompting

### Prompt, prompt engineering e context engineering

Prompt é o texto (e os dados) que você entrega ao modelo. Prompt engineering é otimizar esse texto para uma chamada. Context engineering é cuidar de tudo que ocupa a janela do modelo ao longo de uma tarefa inteira: system prompt, tools, documentos, memória, histórico e resultados de tools.

> 💡 **Pense assim:** Prompt engineering é escrever uma boa função; context engineering é gerenciar a memória do programa inteiro: o que entra, onde fica e quando sai.

*Aparece em:* Q6, Q18, Q40, Q42

### Context window

O limite de tokens que o modelo consegue ler de uma vez, contando entrada e saída. Tudo que não está na janela, para o modelo, não existe. Janela grande não significa janela útil: a qualidade pode cair bem antes do limite.

*Aparece em:* Q6, Q12, Q37, Q43, Q48

### Token e tokenizer

Token é o pedaço de texto que o modelo realmente lê (uma palavra, parte de uma palavra, pontuação). O tokenizer é o programa que corta o texto em tokens. Custo, latência e limite de contexto são todos medidos em tokens, não em palavras.

*Aparece em:* Q22, Q29, Q36, Q43, Q49

### Roles: system, user e assistant

São os "papéis" das mensagens. System traz as regras permanentes (persona, escopo, formato, política de tools), user traz a tarefa e os dados do turno, assistant é a saída do modelo. Os modelos são treinados para dar mais autoridade ao system, mas por baixo tudo é uma sequência única de tokens: o papel é uma dica de autoridade, não uma parede de segurança.

*Aparece em:* Q2, Q10, Q33, Q46, Q49

### Instruction-tuned model

Modelo que passou por um pós-treinamento para seguir instruções escritas em linguagem natural, em vez de só continuar texto. Por isso o zero-shot funciona bem nele.

*Aparece em:* Q1, Q4, Q27

### Zero-shot e few-shot

Zero-shot: você dá só a instrução. Few-shot: você junta alguns exemplos resolvidos (entrada → saída) ao prompt. Em modelos modernos o zero-shot é o ponto de partida, e o few-shot entra quando o formato ou a fronteira entre rótulos é difícil de descrever com palavras.

> 💡 **Pense assim:** Explicar uma tarefa a um estagiário: às vezes basta dizer o que fazer, às vezes é mais rápido mostrar dois casos prontos.

*Aparece em:* Q1, Q4, Q22, Q27, Q42

### Format anchoring

Usar exemplos para "ancorar" o formato exato da saída (uma marcação própria, um estilo da casa), quando isso é mais fácil de mostrar do que de descrever.

*Aparece em:* Q1

### Output contract

A descrição combinada do que a saída deve conter e em que formato (campos, tipos, tamanho). Fica no system prompt para o modelo e para o código que consome a resposta falarem a mesma língua.

*Aparece em:* Q20, Q21, Q29, Q40, Q42

### Prefill do turno do assistant

Truque de começar a resposta do assistant por conta própria (por exemplo, com "{" para forçar JSON) para o modelo só continuar. O suporte está diminuindo: alguns modelos novos já rejeitam isso. Hoje o recomendado é usar structured outputs nativos.

> 💡 **Pense assim:** Começar a frase do seu interlocutor para ele só terminá-la.

*Aparece em:* Q2, Q26, Q47, Q50

### Instruction hierarchy

Ordem de autoridade em que o modelo é treinado para obedecer: instruções de system/developer pesam mais que as do user, que pesam mais que o conteúdo de tools e documentos recuperados. É um hábito aprendido, então é probabilístico e enfraquece em conversas e contextos longos.

*Aparece em:* Q46

### Prompt template (slots)

Um prompt com "buracos" (slots) que o código preenche a cada requisição com variáveis como pergunta, documentos e data. Os problemas clássicos são valor vazio, valor sem limite de tamanho e valor que quebra o seu delimitador.

*Aparece em:* Q9, Q25

### Delimiter collision

Quando o texto inserido num slot contém o próprio delimitador do seu template (por exemplo, "</document>") e o modelo vê a estrutura quebrar no meio do conteúdo. É a versão de prompt do SQL injection; a defesa é escapar ou remover seus delimitadores do conteúdo antes de renderizar.

> 💡 **Pense assim:** Se você escreve uma carta e o texto que está copiando contém a frase "fim da carta", o leitor acha que ela acabou ali.

*Aparece em:* Q9, Q20, Q28, Q30, Q45

### XML tags (como formato)

Marcadores como <documents> e <answer> usados para separar partes do prompt ou da saída. Delimitam sozinhos, não exigem escape de aspas ou chaves e continuam aproveitáveis se a saída for cortada no meio.

*Aparece em:* Q7, Q20, Q30

## 2. In-context learning e exemplos

### In-context learning (ICL)

A capacidade de o modelo "aprender" uma tarefa só pelo que está no prompt, sem mudar nenhum peso. Na prática, os exemplos ajudam o modelo a descobrir qual das tarefas que ele já conhece está sendo pedida, mais do que ensinam algo novo.

> 💡 **Pense assim:** Como reconhecer o gênero de um jogo pelas primeiras jogadas: você não aprendeu nada novo, só identificou qual jogo é.

*Aparece em:* Q15

### Induction heads

Circuitos de attention identificados pela interpretabilidade da Anthropic (Olsson et al., 2022) que fazem "ache uma ocorrência anterior parecida e copie o que veio depois". Surgem cedo no treino, junto com o ICL, e são um candidato a explicar como o modelo continua padrões do prompt.

*Aparece em:* Q15

### Majority-label bias e recency bias

Vieses do few-shot: o modelo puxa para o rótulo mais frequente nos exemplos (majority-label) e para o rótulo do último exemplo (recency). Por isso se balanceiam os rótulos e se presta atenção na ordem dos exemplos.

*Aparece em:* Q14, Q27

### Dynamic few-shot e example store

Em vez de exemplos fixos, você guarda um repositório de exemplos rotulados e busca os k mais parecidos com cada entrada. Costuma render mais accuracy em entradas diversas, mas exige curadoria do store e quebra o prefix caching, porque os exemplos mudam a cada requisição.

*Aparece em:* Q1, Q14, Q22, Q27, Q38

### MMR (maximal marginal relevance)

Técnica de seleção que escolhe resultados relevantes mas diferentes entre si. Evita que o top-k traga cinco exemplos quase idênticos, que ensinariam um padrão estreito.

*Aparece em:* Q27

### Perturbation set e format sensitivity (FormatSpread)

Os modelos podem mudar bastante de accuracy por mudanças de formato que um humano acha irrelevantes (separadores, caixa alta, Q:/A: versus Question:/Answer:, ordem dos campos). O FormatSpread (Sclar et al.) mede essa variação. Um perturbation set é o conjunto dessas variações usadas para medir a média e o spread de um prompt, não só um número.

*Aparece em:* Q47

## 3. Técnicas de reasoning

### Chain-of-thought (CoT)

Fazer o modelo escrever os passos intermediários do raciocínio antes da resposta final. Ajuda em matemática, lógica e planejamento, porque cada token intermediário dá ao modelo mais cálculo serial. Em tarefas simples (extração, lookup) só adiciona custo e latência.

> 💡 **Pense assim:** Resolver uma conta no rascunho em vez de de cabeça.

*Aparece em:* Q1, Q3, Q4, Q18, Q42

### Zero-shot CoT

Acrescentar uma frase como "Let's think step by step" para provocar o raciocínio passo a passo sem nenhum exemplo (Kojima et al., 2022). Ficou famoso porque uma frase genérica aumentou muito a accuracy em aritmética de modelos da época. Hoje está em boa parte obsoleto: modelos instruction-tuned e reasoning models já fazem isso sozinhos.

*Aparece em:* Q3, Q4, Q42

### Faithfulness do CoT

Se o raciocínio escrito pelo modelo reflete de fato o que o levou à resposta. Um chain-of-thought convincente pode ser só uma racionalização de uma resposta errada, por isso se avaliam as saídas e não as justificativas.

*Aparece em:* Q3, Q42

### Reasoning model

Modelo treinado com RL para gerar uma cadeia longa de raciocínio interno antes de responder (série o da OpenAI, Claude com extended thinking, modelos thinking do Gemini, DeepSeek-R1). Você controla a quantidade de raciocínio por parâmetro da API, não pedindo "pense passo a passo" no prompt.

*Aparece em:* Q13, Q16, Q17, Q26, Q42

### Thinking budget e reasoning effort

Parâmetros que dizem quanto o reasoning model pode "pensar" antes de responder. É o botão de custo, latência e qualidade desses modelos. Os tokens de raciocínio são cobrados como saída.

*Aparece em:* Q3, Q8, Q23, Q42

### Test-time compute

Gastar mais computação na hora de responder (raciocinar mais, amostrar várias vezes, buscar) em vez de só treinar um modelo maior. Self-consistency, best-of-n e reasoning models são formas disso.

*Aparece em:* Q3, Q4, Q16

### Self-consistency

Gerar k cadeias de CoT independentes com temperature acima de 0 e escolher a resposta final por votação majoritária. Funciona quando a resposta é curta e comparável (número, rótulo), custa k vezes mais, e o grau de concordância serve como sinal de confiança.

> 💡 **Pense assim:** Pedir a cinco pessoas para resolverem o problema sozinhas: se quatro chegam ao mesmo resultado, você confia mais nele.

*Aparece em:* Q8, Q16, Q42

### Best-of-n e verifier

Gerar n candidatos e deixar um verificador (testes unitários, reward model ou um judge) escolher o melhor. É da mesma família da self-consistency, para saídas longas ou livres em que votar não funciona.

*Aparece em:* Q16

### Decomposição e least-to-most

Quebrar uma tarefa difícil em subtarefas, cada uma com um contexto limpo. No least-to-most (Zhou et al., 2022), o modelo primeiro lista os subproblemas e depois os resolve em ordem, passando as respostas anteriores adiante. Dá observabilidade e controle de custo, mas um erro cedo contamina tudo que vem depois (error cascade).

*Aparece em:* Q17, Q23, Q40

### ReAct

Padrão em loop de Thought (raciocínio), Action (chamada de tool) e Observation (resultado), repetido até a resposta final (Yao et al., 2022). A tool calling nativa de hoje é um ReAct com a estrutura garantida pela API, em vez de texto que o código precisa interpretar.

*Aparece em:* Q18, Q38

### Tool calling (function calling)

Recurso em que o modelo emite uma chamada estruturada a uma função (nome e argumentos validados por schema), o seu código a executa e devolve o resultado. A descrição da tool é lida pelo modelo a cada requisição, por isso é parte do prompt.

*Aparece em:* Q18, Q19, Q26, Q49, Q50

### Tool definitions / schemas

O nome, a descrição e os parâmetros de cada tool que o modelo enxerga. Devem dizer o que a tool faz, quando usá-la em vez das vizinhas e o formato dos argumentos. Muitas tools parecidas pioram a escolha do modelo e gastam tokens.

*Aparece em:* Q6, Q19, Q22, Q33, Q50

### Effort routing

Escolher, por requisição, o modelo e o nível de esforço de raciocínio adequados: baixo para perguntas simples, alto para as difíceis, ou nem usar reasoning model. Evita pagar raciocínio caro em tráfego trivial.

*Aparece em:* Q42

## 4. Sampling e geração

### Logits e softmax

Logits são as notas brutas que o modelo dá a cada token do vocabulário para ser o próximo. O softmax transforma essas notas em probabilidades que somam 1, e é daí que o token é sorteado.

*Aparece em:* Q8, Q36

### Temperature

Divide os logits antes do softmax: abaixo de 1 a distribuição fica mais afiada, acima de 1 fica mais plana, e 0 vira sempre escolher o token mais provável (greedy). Não "adiciona criatividade": só deixa tokens menos prováveis serem sorteados com mais frequência, o que em tarefas factuais costuma significar mais erros.

> 💡 **Pense assim:** O volume do sorteio: em 0 você sempre tira a bola mais pesada; quanto mais alto, mais as bolas leves entram na disputa.

*Aparece em:* Q8, Q16, Q32

### top_p e top_k

Filtros do conjunto de candidatos. top_p (nucleus) mantém o menor grupo de tokens cuja probabilidade acumulada chega a p. top_k mantém os k mais prováveis. A temperature muda a forma da distribuição; top_p e top_k mudam o tamanho do pool.

> 💡 **Pense assim:** Cortar a lista de candidatos antes do sorteio: top_k pega os k primeiros da fila, top_p pega a fila até somar p% de chance.

*Aparece em:* Q8

### Determinismo e seed

Temperature 0 não garante a mesma saída sempre: batching, não associatividade do ponto flutuante e roteamento de MoE podem causar diferenças. Alguns provedores oferecem uma seed fixa, mas mesmo assim a reprodutibilidade é de melhor esforço.

*Aparece em:* Q8, Q22, Q32, Q38, Q46

### max_tokens

Limite duro de tokens na saída. Corta a resposta onde estiver, inclusive no meio de uma frase, então não funciona como instrução de tamanho. Em reasoning models o limite costuma contar também os tokens de raciocínio.

*Aparece em:* Q7, Q9, Q13

### Streaming e time-to-first-token (TTFT)

Streaming é enviar a resposta aos poucos, token a token. TTFT é o tempo até o primeiro token chegar, dominado pelo processamento do prompt de entrada (prefill). Guardrails de saída que esperam a resposta inteira acabam com o ganho do streaming.

*Aparece em:* Q7, Q11, Q22, Q42, Q43

## 5. Structured outputs

### JSON mode

Modo em que o modelo é instruído ou treinado a emitir JSON válido, mas nada garante o seu schema: podem faltar campos, vir tipo errado ou valor de enum inventado. Você precisa validar e fazer retry.

*Aparece em:* Q36

### Schema-constrained decoding (structured outputs)

Em vez de torcer para o modelo obedecer, o schema é compilado numa gramática e, a cada passo, os tokens que quebrariam a estrutura têm o logit mascarado para menos infinito antes do sorteio. A saída é válida por construção (sem retry), mas válida não significa correta: o formato está garantido, a verdade não.

> 💡 **Pense assim:** Um formulário em que as caixas só aceitam o tipo certo de caractere: é impossível digitar um formato inválido, mas dá para digitar um valor errado.

*Aparece em:* Q7, Q36, Q40, Q42, Q47

### JSON Schema

Padrão para descrever a forma de um JSON (campos, tipos, enums, obrigatórios). Os modos strict costumam suportar só um subconjunto dele, e schemas complexos aumentam o tempo de compilação.

*Aparece em:* Q36

### Logit masking

Zerar a probabilidade de certos tokens colocando o logit deles em menos infinito antes do softmax. É o mecanismo do constrained decoding.

*Aparece em:* Q36

### FSM e gramática (CFG, GBNF)

Formas de descrever quais sequências são válidas. Uma FSM (máquina de estados finitos) cobre padrões simples como regex; uma gramática livre de contexto (CFG, no llama.cpp chamada GBNF) cobre aninhamento arbitrário. O constrained decoding acompanha o estado dessa estrutura para saber quais tokens ainda são permitidos.

*Aparece em:* Q36

### Campo reasoning primeiro

Num schema com raciocínio, o campo de texto livre reasoning deve vir antes do campo answer, porque o modelo gera em ordem: se a resposta sai primeiro, o raciocínio só justifica o que já foi sorteado. Também mitiga a queda de qualidade de formatos muito restritos (Tam et al., 2024).

*Aparece em:* Q3, Q36

## 6. Contexto longo, caching e gerenciamento de contexto

### Lost in the middle

Resultado (Liu et al., 2023) de que, em contextos longos, a qualidade ao achar uma informação tem forma de U: é melhor no começo e no fim e pior no meio. Por isso as instruções vão no topo e são repetidas no fim, e os documentos mais importantes ficam nas bordas.

> 💡 **Pense assim:** Numa lista longa lida de uma vez, a gente lembra bem do começo e do fim e esquece o meio.

*Aparece em:* Q6, Q20, Q21, Q30, Q37

### Context rot (pollution)

A degradação do comportamento do modelo conforme a janela se enche de tokens de baixo valor: erros antigos, saídas de tools enormes, planos abandonados, retrievals irrelevantes. Acontece bem antes do limite da janela.

> 💡 **Pense assim:** Uma mesa de trabalho que vai acumulando papéis velhos até você não achar mais o que importa.

*Aparece em:* Q6, Q19, Q32, Q37, Q43

### Distraction e self-conditioning

Os dois mecanismos do context rot. Distraction: a attention se espalha sobre conteúdo plausível mas irrelevante. Self-conditioning: o modelo imita padrões do próprio histórico, então erros passados e loops de tool calls viram "modelos" que se reforçam.

*Aparece em:* Q21, Q37

### Prompt caching e KV cache

O provedor guarda o resultado já calculado (chaves e valores de attention, o KV cache) de um prefixo de prompt e reaproveita em requisições futuras que comecem igual, pulando o processamento daquele trecho. Leituras de cache custam uma fração do preço normal de entrada e reduzem o TTFT.

> 💡 **Pense assim:** Marcar a página até onde você já leu um livro: da próxima vez, você só lê o que foi acrescentado.

*Aparece em:* Q12, Q21, Q22, Q27, Q34

### Exact prefix, breakpoint e cache invalidation

O cache só vale para um prefixo exatamente igual, desde o primeiro token. Se qualquer token mudar (um timestamp, um ID de requisição), tudo dali em diante é recalculado. Por isso se ordena o prompt do estável para o volátil e se marcam breakpoints nas fronteiras. Compaction e reescrever o histórico também invalidam o cache.

*Aparece em:* Q1, Q12, Q22, Q37, Q43

### Cache TTL e preços de leitura e escrita

O cache expira após um tempo (na Anthropic, 5 minutos por padrão, renovado a cada uso). Ler do cache custa em torno de 0,1× o preço normal de entrada e escrever custa um pouco mais que o normal (~1,25× no TTL padrão). Os números exatos mudam entre provedores e gerações.

*Aparece em:* Q22

### Compaction

Resumir os turnos antigos de um agent em um digest (decisões, estado atual, pendências, erros já encontrados), mantendo os turnos recentes e artefatos importantes literais. Perde detalhes de forma irreversível e invalida o cache, então deve acontecer em pontos deliberados e não a cada turno.

> 💡 **Pense assim:** Resumir o caderno de uma reunião de três horas numa ata de uma página, mantendo as últimas anotações como estão.

*Aparece em:* Q12, Q28, Q37, Q43, Q50

### Offloading e scratchpad

Gravar estado intermediário ou resultados grandes em arquivos e manter no contexto só uma referência e um resumo, relendo sob demanda. Troca contexto (caro e que degrada) por armazenamento (barato e durável).

*Aparece em:* Q6, Q28, Q29, Q37

### Truncation, pagination e pruning

Truncar corta o resultado (com um aviso visível); paginar entrega em páginas com instrução de como pedir a próxima; podar remove do contexto o que ficou obsoleto (leituras de arquivo antigas, tentativas falhas). Paginar é melhor que truncar em silêncio porque o modelo sabe que há mais dados.

*Aparece em:* Q19, Q28, Q32, Q37, Q43

### Sub-agent (context firewall)

Um agent auxiliar que trabalha numa janela de contexto própria e devolve só um resumo destilado. Mantém o contexto do agent principal limpo e funciona bem para leitura e pesquisa em paralelo; vai mal em trabalho de escrita com estado compartilhado, porque as decisões implícitas de cada agent conflitam.

> 💡 **Pense assim:** Pedir a um colega que leia 30 arquivos e volte com um parágrafo, em vez de você ler todos.

*Aparece em:* Q6, Q37, Q48

### Token budget e ContextBuilder

Token budget é a quantidade de tokens que você decide gastar, que deve vir da curva de qualidade da sua própria eval e não do limite da janela. Cada componente do prompt recebe um teto e uma estratégia de degradação, aplicados por um único componente de montagem (o ContextBuilder), com contagem de tokens por componente nos traces.

*Aparece em:* Q42, Q43

### Session state e memória entre sessões

Session state é um pequeno objeto estruturado (nome, restrições, decisões) re-renderizado a cada turno. Memória entre sessões guarda fatos duráveis em registros tipados, com timestamp e origem, e precisa de regras de atualização (marcar o fato antigo como superseded), de expiração (TTL) e defesa contra poisoning, injection e vazamento entre usuários.

*Aparece em:* Q12, Q37, Q43, Q44

### Knowledge cutoff e data atual

O modelo só sabe o que havia nos dados de treino até uma data de corte e, sem a data atual no prompt, age como se ainda estivesse nela. A prática é injetar a data (em granularidade de dia, por causa do cache), mandar buscar o que pode ter mudado e fazer aritmética de datas em código.

*Aparece em:* Q13, Q34

### Prompt compression e LLMLingua

Reduzir o tamanho do prompt. LLMLingua usa a perplexity de um modelo pequeno para descartar tokens de pouca informação; o texto fica estranho para humanos mas o modelo ainda o lê. Raramente é a primeira escolha: curar o contexto, fazer retrieval melhor e usar prompt caching costumam render mais e sem perda de qualidade.

*Aparece em:* Q29, Q44

### AGENTS.md / CLAUDE.md

Arquivo de instruções do repositório que um coding agent lê em toda sessão. Deve ter só o que o agent não descobre sozinho: comandos exatos, convenções que nenhum linter pega e armadilhas. Fica no prefixo de toda sessão, então precisa ser curto, e precisa ser mantido atualizado.

*Aparece em:* Q35

### Chat template, special tokens e BOS

O chat template é o modelo de texto que transforma a lista de mensagens na sequência exata de tokens (com tokens especiais de papel e de fim de turno) que o modelo viu no pós-treino. BOS é o token de início de sequência. Template errado, BOS duplicado ou stop token faltando fazem o mesmo modelo render pior em um servidor próprio.

*Aparece em:* Q2, Q49

### Open-weight, self-hosted e quantization

Open-weight é um modelo com pesos disponíveis para rodar você mesmo (self-hosted). Quantization reduz a precisão numérica dos pesos para ocupar menos memória, com possível perda de qualidade. Só se suspeita dela depois de confirmar que os tokens enviados são idênticos aos da referência.

*Aparece em:* Q22, Q29, Q49

### Thinking blocks e encrypted reasoning

Em um loop de tools com reasoning model, o raciocínio do passo anterior precisa voltar à API exatamente como foi devolvido (na Anthropic, thinking blocks com assinatura; na Responses API da OpenAI, reasoning items, até criptografados). Sem isso o modelo perde o fio do plano e repete passos.

*Aparece em:* Q18, Q38, Q50

### Multilingual e tokenização desigual

O mesmo texto custa mais tokens em alguns idiomas (como hindi ou tailandês) do que em inglês, e a qualidade varia por idioma. Por isso se declara o idioma de saída explicitamente, se avalia por idioma e, em idiomas de poucos recursos, às vezes compensa traduzir para inglês, processar e traduzir de volta.

*Aparece em:* Q24

## 7. RAG, grounding e instruções conflitantes

### RAG e retrieval

Retrieval-augmented generation: buscar trechos relevantes numa base de documentos e colocá-los no prompt para o modelo responder com base neles. Fatos mudam e pertencem ao retrieval; o prompt deve guardar comportamento, não dados.

*Aparece em:* Q21, Q30, Q32, Q40, Q44

### Chunk e reranker

Chunk é cada trecho de documento recuperado. Reranker é um segundo modelo que reordena os candidatos por relevância. Mandar poucos chunks bons (precisão) costuma ser melhor que mandar muitos (recall), porque trechos irrelevantes distraem.

> 💡 **Pense assim:** Chunk é uma página xerocada de um livro; o reranker é o bibliotecário que ordena as páginas pelo quanto ajudam na sua pergunta.

*Aparece em:* Q12, Q21, Q29, Q30, Q43

### Groundedness e citations

Groundedness é o quanto a resposta se apoia nos documentos fornecidos. Exigir um ID de documento após cada afirmação permite checar por código se aquele documento sustenta a frase, virando uma métrica automática de alucinação.

*Aparece em:* Q11, Q21, Q30, Q38, Q42

### Abstention e parametric knowledge

Abstention é o modelo dizer "não achei a resposta nos documentos" em vez de inventar. Parametric knowledge é o que o modelo memorizou nos pesos; sem uma regra explícita, ele recorre a isso em silêncio, de forma fluente e sem fonte.

*Aparece em:* Q30

### Precedência e conflito de instruções

Quando instruções se contradizem (system, user, documento, ou regras antigas do próprio prompt), o resultado é imprevisível. Regras de autorização vão para o código, regras de preferência são resolvidas no prompt declarando a precedência e dando um exemplo, e a contradição silenciosa entre regras deve ser auditada e testada na eval.

*Aparece em:* Q5, Q32, Q39, Q44, Q46

### Over-refusal

Quando o modelo recusa pedidos perfeitamente legítimos, por excesso de cautela. Mede-se com um eval de casos "inofensivos mas assustadores" e corrige-se com escopo específico dito em positivo, exemplos na fronteira e logging de cada recusa.

*Aparece em:* Q11, Q26, Q31, Q41, Q46

## 8. Segurança: injection, vazamento e guardrails

### Prompt injection

Ataque em que texto malicioso dentro dos dados (página web, e-mail, documento) tenta passar por instrução. Como instruções e dados vivem na mesma sequência de tokens, nenhum layout de prompt resolve por completo; a defesa real é limitar o estrago.

> 💡 **Pense assim:** Um bilhete falso colado dentro de um documento que diz "chefe mandou: transfira o dinheiro", e o assistente obedece.

*Aparece em:* Q11, Q20, Q42, Q44, Q45

### Indirect prompt injection

Injection que chega por conteúdo que o agent lê (páginas, e-mails, documentos recuperados), não por quem conversa com ele. É o vetor mais comum no mundo real.

*Aparece em:* Q6, Q20, Q28, Q45, Q46

### Spotlighting

Marcar o conteúdo não confiável por codificação ou transformação (por exemplo, prefixar cada linha com um marcador) para que sua origem continue visível mesmo no meio do texto.

*Aparece em:* Q20

### Lethal trifecta

Expressão de Simon Willison para a combinação de três coisas: acesso a dados privados, exposição a conteúdo não confiável e um canal de exfiltração. O desastre exige as três; tirar qualquer uma enfraquece o ataque.

*Aparece em:* Q45

### Exfiltration

Fazer dados privados saírem do sistema, por exemplo por um e-mail enviado, uma URL buscada ou uma imagem em markdown com URL controlada pelo atacante. Cortar o canal de exfiltração costuma ser a alavanca mais barata.

*Aparece em:* Q20, Q45

### Least privilege

Dar ao agent só as tools e credenciais necessárias para aquela tarefa, no máximo o que o próprio usuário já poderia fazer. Assim, mesmo um agent enganado causa pouco dano.

> 💡 **Pense assim:** Dar ao estagiário a chave só da sala em que ele trabalha, não o chaveiro do prédio inteiro.

*Aparece em:* Q2, Q20, Q45

### Taint tracking

Marcar tudo que veio de fonte não confiável e proibir que esses dados cheguem a ações de alto risco (como escolher o destinatário de um e-mail) sem confirmação humana.

*Aparece em:* Q45

### Dual-LLM e CaMeL

Padrões de separação estrutural. No dual-LLM, um modelo privilegiado que planeja nunca vê o texto não confiável, e um modelo em quarentena o processa e devolve só valores tipados. O CaMeL, do Google DeepMind, vai além: o planner emite código cujas permissões são aplicadas por um interpretador fora do modelo.

*Aparece em:* Q6, Q20, Q45

### Action gating e human confirmation

Controlar no código, na fronteira da tool, o que o agent pode executar de verdade, e exigir confirmação humana mostrando o efeito concreto (este destinatário, este corpo) antes de ações irreversíveis. O prompt é defesa em profundidade, nunca a fronteira.

*Aparece em:* Q11, Q20, Q45

### System prompt extraction e leakage

Fazer o modelo revelar o system prompt (por role-play, tradução, codificação etc.). Parta do princípio de que vaza: não coloque credenciais, PII ou decisões de autorização nele. O system prompt é configuração, não um cofre.

*Aparece em:* Q10, Q11, Q29, Q44, Q46

### Input e output guardrails

Verificações em volta do modelo. Input guardrails filtram o pedido antes (PII, fora de escopo, padrões de injection); output guardrails filtram a resposta antes de chegar ao usuário (schema, groundedness, vazamento, toxicidade). Pegam falhas diferentes, e agents precisam de uma terceira camada de gating nas tools.

*Aparece em:* Q10, Q11, Q23, Q41

### Fail open vs fail closed

O que fazer quando uma checagem de segurança falha ou dá timeout: deixar passar (fail open) ou bloquear (fail closed). É uma decisão de produto por checagem: aceitável em um bot de suporte, não em uma superfície médica.

*Aparece em:* Q10, Q11

## 9. Avaliação, otimização e produção

### Eval set, golden set e regression suite

Eval set é o conjunto de casos (30 a 200 no começo) com a forma de pontuar cada um. Golden set é o núcleo de casos com resposta conhecida; a regression suite junta isso com falhas passadas. Sem isso, melhorar um prompt é anedota e você conserta um caso quebrando três.

*Aparece em:* Q23, Q24, Q39, Q40, Q41

### LLM-as-judge

Usar um modelo para dar nota a saídas abertas segundo um rubric. Deve ser conferido contra rótulos humanos antes, e um otimizador pode explorar um judge fraco.

*Aparece em:* Q11, Q16, Q23, Q38, Q41

### Overfitting da eval e held-out

Ajustar o prompt tanto a um conjunto de casos que ele só funciona neles. A defesa é a mesma do machine learning clássico: separar casos de teste que o otimizador nunca vê e renovar o conjunto com falhas novas de produção.

> 💡 **Pense assim:** Decorar as questões da prova antiga: você vai bem nela e mal na prova de verdade.

*Aparece em:* Q23, Q26, Q38, Q39, Q47

### Paired test e intervalo de confiança

Em vez de comparar duas médias, compara-se o resultado de cada caso entre duas variantes (teste pareado), com várias rodadas por caso e um intervalo de confiança. Uma melhora de 3 pontos em 100 casos pode ser ruído de poucos casos que viraram.

*Aparece em:* Q7, Q20, Q32, Q41, Q47

### DSPy: signatures, modules e optimizers

Framework (Khattab et al., 2023) que trata o prompt como parâmetro. Você declara o que o pipeline faz (signature, como "question, context -> answer") e monta módulos (Predict, ChainOfThought, ReAct); um optimizer então busca instruções e exemplos que maximizem sua métrica. Exige uma métrica programática e um conjunto de treino/dev.

*Aparece em:* Q38, Q39, Q50

### MIPROv2 e GEPA

Otimizadores de prompt. MIPROv2 propõe instruções candidatas e busca a melhor combinação de instruções e demos com um método estilo otimização bayesiana. GEPA usa um modelo que lê os traces de falha e o feedback e propõe instruções evoluídas, e tende a precisar de menos execuções.

*Aparece em:* Q38

### Meta-prompting

Usar um modelo forte para criticar ou reescrever o seu prompt, de preferência a partir de transcripts de falhas reais. É um gerador barato de candidatos, mas o resultado precisa vencer o baseline na eval: prompts reescritos parecem melhores, ficam mais longos (verbosity bias) e podem perder uma sutileza que existia por causa de um incidente.

*Aparece em:* Q23, Q39

### Anti-patterns de prompt

Os mais comuns: instruções cheias de negação ("não mencione X" planta o X na cabeça do modelo), restrições contraditórias, mega-prompts com dezenas de regras competindo e prompt-as-database (colar fatos que pertencem ao retrieval).

*Aparece em:* Q5, Q25

### Fine-tuning, SFT, LoRA e distillation

Fine-tuning ajusta os pesos com exemplos seus. SFT é o ajuste supervisionado com pares entrada/saída; LoRA é um jeito barato de fazê-lo atualizando poucos parâmetros. Distillation é treinar um modelo pequeno com as saídas de um grande. Funciona para consistência, formato e estilo, não para injetar conhecimento novo, e você passa a ser dono de um artefato de modelo.

*Aparece em:* Q31, Q37, Q38, Q40, Q48

### A/B test, shadow mode e canary

Rollout em estágios para mudar um prompt com segurança: primeiro a eval offline, depois shadow (a variante roda em tráfego real sem ser servida), depois canary (uma pequena porcentagem de usuários, com métricas de guarda e rollback automático). Em produtos multi-turno, randomiza-se por usuário, não por requisição.

> 💡 **Pense assim:** Testar uma receita nova: primeiro na cozinha, depois servida só para um garçom provar, depois a uma mesa, e só então ao salão.

*Aparece em:* Q23, Q25, Q41

### Guardrail metrics e proxies

Em testes online faltam rótulos de verdade, então se usa uma hierarquia: métricas de guarda (validade, latência, custo, segurança), proxies de qualidade (taxa de retry ou de reformulação, conclusão da tarefa, escalonamento para humano) e amostras avaliadas por judge.

*Aparece em:* Q23, Q41

### Prompt versioning e registry

Tratar o prompt como artefato versionado com ID imutável, registrado em cada requisição. A unidade implantável é o conjunto de template, modelo, parâmetros de sampling, schemas de tools e schema de saída. Um registry com labels (production, staging) permite rollback trocando um ponteiro, sem deploy.

*Aparece em:* Q1, Q14, Q15, Q25, Q27

### Tracing e observabilidade

Registrar, por requisição, o prompt exatamente como foi renderizado, o modelo, os parâmetros, os IDs recuperados e cada chamada de tool com resultado. Sem isso não dá para reproduzir nem depurar uma falha.

*Aparece em:* Q32, Q33, Q38, Q43, Q48

### Prompt-model co-adaptation e portabilidade

Cada ajuste de redação para consertar um caso embute os hábitos de um modelo específico no prompt, então ele quebra ao trocar de modelo ou provedor. O ativo portável é a eval, não o prompt: mantém-se o mesmo eval e um adaptador fino por modelo.

*Aparece em:* Q26, Q35, Q47, Q50

### Bisect do contexto

Depurar cortando o contexto pela metade (documentos, exemplos, regras) até achar a menor entrada que ainda reproduz a falha. Costuma revelar regras em conflito ou um exemplo que ensina o comportamento errado.

*Aparece em:* Q32
