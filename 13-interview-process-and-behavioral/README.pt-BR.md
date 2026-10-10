# 💼 Interview Process & Behavioral

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Todo loop de AI engineering termina da mesma forma: uma rodada comportamental em que decidem se de fato gostariam de trabalhar com você, e um debrief em que "forte tecnicamente, mas não soube explicar o que entregou" mata ofertas. Esta seção é um guia de campo de todo o processo - como o loop de 2026 realmente se parece estágio por estágio, como ele difere por tipo de empresa, o que take-homes e portfólios precisam provar e como contar suas histórias de trabalho com AI para que elas causem impacto. Todos são questionados sobre este material, de recém-formados a candidatos staff; os seniores são questionados com mais rigor.

## Anatomia do loop de AI Engineer de 2026

Um loop completo típico tem 4-6 rodadas ao longo de 2-5 semanas. Os estágios variam, mas este é o esqueleto comum:

**1. Recruiter screen (30 min).** Logística, motivação, calibração de nível, expectativas de remuneração. Duas coisas importam: tenha um resumo nítido de 90 segundos do seu projeto de AI mais impressionante (com um número nele) e não ancore baixo na remuneração - desvie com "estou otimizando pela vaga certa; confio que podemos fazer a remuneração funcionar se houver fit mútuo" até ter dados (veja a seção de oferta abaixo).

**2. Coding (1-2 rodadas).** A barra é real e *subiu* em empresas de AI, não caiu. Espere um destes:
- **LeetCode-lite / DSA prático**: strings, hashmaps, heaps, BFS/DFS em dificuldade média, avaliados por código funcionando e comunicação mais do que por complexidade assintótica ótima.
- **"Construa algo que chama um LLM"**: cada vez mais comum. Ex.: "implemente um chat loop com tool calling", "construa um client async com rate limit que faz batching de requisições", "faça o parsing desta saída bagunçada para um schema com retries". Testa fluência de API, Python async, tratamento de erros e se você de fato já fez isso antes. Pratique escrever um tool-use loop do zero sem framework.
- **Rodadas de coding assistido por AI**: algumas empresas agora *permitem ou exigem* que você use Copilot/Claude durante a entrevista e avaliam como você dirige a ferramenta, verifica a saída dela e pega os bugs dela. Pergunte ao recruiter quais regras se aplicam.

**3. Deep-dive de domínio ML/AI (1 rodada).** Profundidade conceitual sobre as afirmações do seu currículo: transformers, sampling, fine-tuning vs RAG vs prompting, evals, mitigação de hallucination. As seções 01-10 deste repositório são esta rodada. O entrevistador sonda até você quebrar - saber onde termina o seu conhecimento e dizer isso vence blefar sempre.

**4. AI system design (1 rodada).** "Projete um support copilot / RAG corporativo / plataforma de agents." A seção 11 cobre isso. Diferencial: começar por evals, custo e modos de falha em vez de caixas e setas.

**5. Behavioral (1 rodada, às vezes distribuída ao longo das demais).** As perguntas em [questions.pt-BR.md](questions.pt-BR.md). Em níveis sênior+ esta rodada pesa tanto quanto qualquer rodada técnica.

**6. Às vezes: take-home ou work trial pago.** Coberto abaixo. Especialmente em startups; algumas fazem trials presenciais pagos de 1-5 dias no lugar de partes do loop.

## Como os loops diferem por tipo de empresa

| | Frontier labs (OpenAI, Anthropic, Google DeepMind, Meta) | Big tech (AWS, Azure, orgs do GCP, times de produto) | Startups AI-native | Empresas tradicionais adotando AI |
|---|---|---|---|---|
| **Barra de coding** | Muito alta; frequentemente várias rodadas, com sabor de mundo real | Alta, DSA padrão + prático | Prático > algorítmico; entregar algo funcionando rápido | Moderada; fluência na linguagem/stack |
| **Rodada de domínio** | Profunda - espere fluência no nível de papers sobre attention, RLHF/GRPO, scaling, inference | Amplitude + como você colocaria em produção | Aplicada: RAG, agents, evals, "o que você construiria com X" | Integração: APIs, data pipelines, tradeoffs de fornecedores |
| **System design** | Infra de pesquisa ou serving em escala extrema | Sistemas distribuídos clássicos + camada de AI | Orientado a produto: design de feature 0→1 | Confiabilidade, compliance, governança de custo |
| **O que ganha ofertas** | Profundidade + fluência em pesquisa + coding excepcional | Experiência em escala, maturidade de processo, histórias cross-team | Velocidade de entrega, amplitude, product sense, portfólio | Gestão de risco, habilidade com stakeholders, pragmatismo |
| **Cuidado com** | Loops longos, curveballs de pesquisa | Jogos de mapeamento de nível, barra dependente da org | Rodadas de feeling do fundador, work trials | Screens de buzzwords feitos por não especialistas |

Calibre sua preparação: para um frontier lab, invista demais nas seções 01-02 e 05; para uma startup, tenha um projeto de portfólio que você consiga demonstrar e defender de ponta a ponta; para big tech, ensaie histórias comportamentais cross-funcionais com escopo e números.

## Decodificador de títulos de cargo

Os títulos são inconsistentes entre empresas - leia a job description, não o título. Decodificação aproximada:

- **AI Engineer / GenAI Engineer**: constrói produtos *em cima de* modelos - RAG, agents, prompting, evals, integração de serving. Principalmente engenharia de software com uma camada específica de LLM. Perguntam: tudo neste repositório, exceto matemática profunda de treinamento.
- **ML Engineer (MLE)**: treina, ajusta e faz deploy de modelos; é dono de pipelines, feature stores, infra de treinamento. Mais ML clássico + infra. Perguntam: fundamentos de ML, MLOps, treinamento distribuído, e cada vez mais a camada de LLM.
- **Research Engineer (RE)**: implementa e escala ideias de pesquisa em labs - runs de treinamento, ablations, performance engineering. Barra pesada de coding + teoria sólida de ML; publicações são opcionais, mas fluência em papers é obrigatória. Perguntam: implementar attention do zero, debugar um run de treinamento, perguntas de GPU/paralelismo.
- **Applied Scientist / Research Scientist**: projeta experimentos, é dono de métricas e escolhas de modelagem; RS normalmente exige publicações. Perguntam: estatística, design experimental, profundidade em modelagem, discussão de papers.

Se a JD de um "AI Engineer" lista CUDA e pretraining, é uma vaga de RE disfarçada. Se a JD de um "MLE" é só LangChain e vector DBs, é AI Engineering. A preparação para a entrevista deve seguir o conteúdo real da JD.

## Take-home assignments

Pedido típico: "Construa um pequeno serviço de RAG/agent/extraction sobre estes dados; ~4-8 horas." O que os avaliadores de fato pontuam, em ordem aproximada de prioridade:

1. **Ele roda.** Setup com um comando, dependências fixadas, sem surpresas de API key faltando. Funcionando e chato vence ambicioso e quebrado - os avaliadores gastam 15 minutos, e um crash no primeiro minuto é fatal.
2. **Evals incluídos.** O maior diferencial em 2026. Mesmo 15 casos de teste escritos à mão com um script de pass/fail sinalizam mais senioridade do que qualquer escolha de arquitetura. Inclua uma análise de falhas: "falha em X por causa de Y."
3. **README com tradeoffs.** O que você construiu, o que deliberadamente não construiu, o que faria com mais uma semana. Nomear seus atalhos ("sem streaming, apenas chamadas síncronas - aceitável nesta escala porque...") soa sênior; escondê-los soa júnior.
4. **Engenharia sensata**: código tipado, tratamento de erros em toda chamada de LLM, noção de custo/tokens, sem sopa de frameworks para um problema de 300 linhas.

**Dimensionando seu tempo**: limite ao número de horas informado mais ~25%. Gaste cerca de 50% em funcionalidade principal, 25% em evals + análise de erros, 25% em README e limpeza. Não perfumaria a UI.

**Quando recusar**: trabalho não remunerado que exceda ~um dia, assignments suspeitamente idênticos ao roadmap real da empresa, ou take-homes enviados antes de qualquer humano ter conversado com você. Tudo bem fazer uma contraproposta: "Não consigo fazer 20 horas não remuneradas, mas aqui está um repo meu que demonstra as mesmas habilidades - tenho prazer em apresentá-lo ao vivo."

**Sobre assistência de AI**: assuma que vão pedir que você defenda cada linha em uma call de follow-up. Use as ferramentas que quiser, mas se você não consegue explicar uma escolha de design, você não a fez.

## Portfólio e projetos que realmente impressionam

Um projeto profundo com evals e um writeup vence dez repos de "cloned a YouTube tutorial". Ideias concretas, cada uma com o que sinaliza:

1. **App de RAG sobre um corpus real e bagunçado** (suas anotações, um site de docs complicado, processos judiciais) com um eval harness de verdade - métricas de retrieval, checagens de faithfulness e um writeup de análise de erros agrupando as falhas. *Demonstra: fundamentos de retrieval + a maturidade em evals que falta à maioria dos candidatos.*
2. **Um agent com execução de tools em sandbox** - ex.: um agent de código ou análise de dados que roda tools em Docker com permission gating, timeouts e um audit log. *Demonstra: design de agents mais o raciocínio de segurança que os entrevistadores raramente veem.*
3. **Tokenizer + transformer do zero** - tokenizer BPE e um pequeno modelo estilo GPT em PyTorch puro (o caminho zero-to-hero do Karpathy), treinado em um corpus de brinquedo, com notas do que você aprendeu. *Demonstra: fundamentos que sobrevivem à rodada de domínio mais profunda.*
4. **Um writeup de benchmark de inference** - o mesmo modelo em vLLM vs llama.cpp vs uma API de provedor: curvas de TTFT/throughput vs batch size e context length, custo por milhão de tokens, metodologia declarada. *Demonstra: conhecimento de serving e rigor empírico.*
5. **Um fine-tune com evals antes/depois** - LoRA em um pequeno modelo open source numa tarefa estreita; compare com um modelo frontier via prompt em qualidade, latency e custo; publique o veredito mesmo que o fine-tuning tenha *perdido*. *Demonstra: você sabe quando fine-tuning vale ou não a pena - um sinal de entrevista de primeira linha.*
6. **Estudo de calibração de LLM-as-judge** - construa um judge para alguma tarefa, meça a concordância contra seus próprios rótulos humanos, documente os vieses de posição/tamanho/self-preference que encontrou. *Demonstra: sofisticação em evals além de "usamos o GPT para avaliar."*
7. **Pipeline de structured-extraction** - milhares de documentos bagunçados → schema validado, com retries, rastreamento de custo e um relatório de acurácia. *Demonstra: engenharia de produção pouco glamorosa, que é a maior parte do trabalho real.*
8. **Um MCP server para uma ferramenta ou fonte de dados real** - com auth, validação de input e testes, utilizável a partir de clients Claude/IDE. *Demonstra: padrões de integração atuais e design de API.*
9. **Um model router / semantic cache** - pequeno classificador ou heurística roteando queries fáceis para um modelo barato e as difíceis para um modelo frontier, com uma fronteira qualidade-vs-custo medida. *Demonstra: engenharia de custo com dados.*
10. **Um app de voz ou multimodal com um latency budget** - documente o pipeline de ponta a ponta e para onde vai cada milissegundo, mais o que você fez sobre o pior hop. *Demonstra: streaming, UX sob latency, pensamento de sistemas.*

Todo projeto precisa de três artefatos: um README com números, um diretório evals/ e um writeup curto do que te surpreendeu. O writeup é o que os entrevistadores de fato leem.

## Dicas de currículo para vagas de AI

- **Métricas em cada bullet, evals sempre que possível.** "Entreguei assistente de RAG" é ruído. "Entreguei assistente de RAG de suporte para 40k usuários; elevei a taxa de aceitação de respostas de 61%→78% em um eval de 500 casos; cortei o custo/query em ~70% via caching + model routing" consegue uma entrevista. Se você não pode compartilhar números internos, use deltas relativos e indicadores de escala.
- **Datas de entrega e durações.** "Protótipo→GA em 9 semanas" sinaliza velocidade. Recência importa neste campo - comece pelo trabalho de 2024-2026.
- **Linke artefatos.** Repos do GitHub, writeups, demos, papers. Uma afirmação no currículo com prova clicável vale três sem ela.
- **Nomeie a camada de que você foi dono.** "Fui dono do pipeline de retrieval + eval" vence um vago "trabalhei no chatbot de AI." Os entrevistadores aprofundam exatamente onde seus bullets reivindicam ownership - garanta que você consegue ir três perguntas a fundo em cada um.
- **Elimine a sopa de palavras-chave.** Uma linha de skills com 30 frameworks soa júnior. Liste o que você se sentiria confortável sendo sabatinado, nada além disso.
- **Não infle.** "Fine-tuned GPT-4" quando você escreveu prompts, ou "trained an LLM" quando você rodou LoRA em um 7B por uma tarde, será descoberto na primeira rodada e envenena tudo depois.

## Mantendo-se atualizado sem se afogar

Você não consegue ler tudo; os entrevistadores não esperam isso. Eles esperam um *sistema*. Uma dieta semanal sensata (~3 horas):

- **Um digest curado** para amplitude - escolha um ou dois entre: uma boa newsletter, Latent Space, o blog do Simon Willison, threads de AI no Hacker News.
- **Uma coisa lida a fundo** - um paper, um model card ou um post sério de blog de engenharia. Profundidade em uma coisa por semana se acumula; passar os olhos em quarenta abstracts não.
- **Release notes dos provedores que você usa** - deprecações de modelos, mudanças de API, preços. Isso é relevante para o trabalho de uma forma que a maioria dos papers não é.
- **Uma hora prática** - experimente o novo modelo/ferramenta no *seu próprio* eval set ou side project. "Rodei no meu benchmark e notei X" é uma resposta de entrevista de elite; "vi uma thread sobre isso" não é.

E explicitamente *pule*: o feed o dia todo, hot takes do dia de lançamento e qualquer paper que você não consiga conectar a algo que você constrói.

## Perguntas que VOCÊ deve fazer aos entrevistadores

Boas perguntas são diagnósticas para você *e* sinalizam senioridade. Pergunte sobre:

- **Cultura de evals**: "Como vocês sabem que uma mudança de prompt ou de modelo é segura para entregar? Me conduzam pela última." (Sem resposta real = você vai ficar apagando incêndios de regressões guiadas por vibes.)
- **Acesso a dados**: "Que dados o time pode de fato usar para treinamento/evals, e quanto tempo leva o acesso?" (Aprovação de dados que leva meses mata times de AI silenciosamente.)
- **Realidade de GPU/orçamento**: "Como está a situação de orçamento de compute e API - experimentos entram em fila atrás de produção?"
- **Como entregam**: "Qual é o caminho do notebook à produção? Quem pode fazer push de uma mudança de prompt, e o que a bloqueia?"
- **On-call**: "Features de AI têm on-call? Como é um incidente - quem debuga uma regressão de qualidade às 2h da manhã?"
- **Estratégia de modelos**: "O que aconteceu na última migração/deprecação de modelo de vocês?" (Diz o quanto estão acoplados a um único fornecedor e quanta dívida técnica você está herdando.)
- **Para startups**: runway, quem são os design partners/clientes e qual foi o último pivot.

## Noções básicas de avaliação de oferta

Apenas orientação geral - faça sua própria pesquisa com dados atuais:

- **Mapeamento de nível é o jogo.** A mesma experiência mapeia para níveis diferentes em empresas diferentes, e a remuneração é definida pelo nível. Antes de negociar valores, negocie o nível - um nível normalmente vale muito mais do que qualquer aumento dentro da faixa. Use [levels.fyi](https://www.levels.fyi/) e conversas com pares para calibrar títulos entre empresas.
- **Equity de startup de AI**: entenda a diferença entre seu strike price (409A) e o preço preferencial da última rodada; o "seu equity vale $X" de destaque usa o segundo. Pergunte sobre expectativas de diluição total, janelas de exercício e se a empresa já permitiu vendas secundárias. Valuations altas em AI significam preference stacks altas - modele o cenário negativo, não só o caso unicórnio.
- **Mix de caixa vs. equity**: frontier labs e big tech pagam pesadamente em equity (líquido em empresas públicas, menos nas demais). Uma oferta de startup 40% abaixo do caixa de big tech precisa de uma história de equity crível para competir - faça-os construí-la.
- **Fontes de pesquisa de remuneração**: levels.fyi, divulgações salariais de H-1B (registro público nos EUA), conversas com recruiters de empresas pares e simplesmente perguntar à sua rede. Nunca negocie a partir de um único ponto de dados.
- **Termos não salariais que importam em vagas de AI**: acesso a compute/GPU, política de publicação, carve-outs de IP para side projects, expectativas de on-call e política de remoto. Para vagas próximas de pesquisa, direitos de publicação podem valer mais do que um aumento salarial para o seu valor de mercado de longo prazo.

## Mock interviews e mentoria

Ler respostas não é o mesmo que dá-las sob pressão. Antes de um loop real, faça pelo menos uma mock completa: use o [mock interview kit](mock-interview-kit.md) com um amigo, ou agende uma sessão com um entrevistador experiente para pontuação honesta e um plano de preparação construído em torno das suas lacunas, em [enginebogie.com/u/om](https://enginebogie.com/u/om) ou [topmate.io/ombharatiya](https://topmate.io/ombharatiya).

## Perguntas de entrevista

42 perguntas comportamentais e de experiência específicas do trabalho com AI, com orientação de resposta, em [questions.pt-BR.md](questions.pt-BR.md).

## Red flags que os entrevistadores observam

- **Nenhum número em lugar algum.** Toda história de projeto sem usuários, latency, custo ou métricas de eval - sinaliza que você viu o trabalho acontecer em vez de ser dono dele.
- **"Não tivemos tempo para evals."** O rebaixamento de senioridade mais rápido de 2026. Até times improvisados rotulam à mão 50 casos.
- **Culpar o modelo.** "O GPT-whatever simplesmente alucinou" sem história de mitigação mostra que você parou de debugar na fronteira da API.
- **Reivindicar a vitória do time na primeira pessoa do singular** - e depois falhar em duas perguntas de follow-up sobre como aquilo de fato funcionava.
- **Inflação de currículo em afirmações de treinamento.** Dizer "fine-tuned"/"trained" para o que foi prompt engineering se desfaz em minutos e contamina todas as outras afirmações.
- **Name-dropping de frameworks em vez de entendimento** - sabe recitar LangChain/LlamaIndex, mas não consegue descrever o que um agent loop faz sem eles.
- **Nenhuma pergunta para o entrevistador**, ou só perguntas sobre benefícios - soa como baixo ownership e baixa curiosidade.
- **Falar mal de um empregador ou time anterior** - mesmo quando justificado, transfere a negatividade para você.

## Leituras complementares

- [levels.fyi](https://www.levels.fyi/) - mapeamento de níveis e dados de remuneração entre empresas.
- [Machine Learning Interviews Book - Chip Huyen](https://huyenchip.com/ml-interviews-book/) - o livro gratuito clássico sobre loops de entrevista de ML, papéis e preparação.
- [Applied LLMs - What We Learned from a Year of Building with LLMs](https://applied-llms.org/) - lições de praticantes (evals, RAG, ops) que mapeiam diretamente para histórias de entrevista.
- [Your AI Product Needs Evals - Hamel Husain](https://hamel.dev/blog/posts/evals/) - o playbook de maturidade em evals que os entrevistadores cada vez mais esperam que você tenha internalizado.
- [Simon Willison's blog](https://simonwillison.net/) - cobertura prática e de alto sinal de novos modelos e ferramentas; um modelo de "manter-se atualizado" bem feito.
- [Latent Space](https://www.latent.space/) - podcast/newsletter sobre AI engineering como disciplina, incluindo contratação e definições de papéis.
- [Neural Networks: Zero to Hero - Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) - o caminho canônico do zero para o projeto de portfólio de fundamentos.
- [The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/) - contexto da indústria sobre processos de contratação, níveis e como as empresas de tecnologia realmente conduzem loops.
