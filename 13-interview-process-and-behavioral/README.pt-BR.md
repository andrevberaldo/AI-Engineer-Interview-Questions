# 💼 Interview Process & Behavioral

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

Todo loop de AI engineering termina do mesmo jeito: uma rodada comportamental, em que decidem se querem mesmo trabalhar com você, e um debrief em que "forte tecnicamente, mas não soube explicar o que entregou" derruba ofertas. Esta seção é um guia de campo do processo inteiro: como o loop de 2026 realmente é, estágio por estágio, como ele muda conforme o tipo de empresa, o que take-homes e portfólios precisam provar e como contar suas histórias de trabalho com AI para que elas marquem. Todo mundo é perguntado sobre esse material, de recém-formado a candidato staff, mas os seniores são cobrados com mais rigor.

## Anatomia do loop de AI Engineer de 2026

Um loop completo costuma ter de 4 a 6 rodadas ao longo de 2 a 5 semanas. Os estágios variam, mas este é o esqueleto comum:

**1. Recruiter screen (30 min).** Logística, motivação, calibração de nível e expectativas de remuneração. Duas coisas importam. Primeiro, tenha na ponta da língua um resumo de 90 segundos do seu projeto de AI mais impressionante, com um número no meio. Segundo, não ancore a remuneração lá embaixo: desvie com "estou otimizando pela vaga certa; confio que, havendo fit dos dois lados, a gente fecha a remuneração" até ter dados (veja a seção de oferta abaixo).

**2. Coding (1-2 rodadas).** A barra é real e *subiu* nas empresas de AI, não caiu. Espere um destes formatos:
- **LeetCode-lite / DSA prático**: strings, hashmaps, heaps, BFS/DFS em dificuldade média. O que pesa é código funcionando e boa comunicação, mais do que complexidade assintótica ótima.
- **"Construa algo que chama um LLM"**: cada vez mais comum. Por exemplo: "implemente um chat loop com tool calling", "construa um client async com rate limit que agrupa requisições em batches", "transforme esta saída bagunçada em um schema, com retries". Isso testa fluência com APIs, Python async, tratamento de erros e se você já fez isso de verdade. Pratique escrever um tool-use loop do zero, sem framework.
- **Rodadas de coding assistido por AI**: algumas empresas agora *permitem ou exigem* que você use Copilot/Claude durante a entrevista, e avaliam como você conduz a ferramenta, verifica a saída dela e pega os bugs que ela cria. Pergunte ao recruiter quais regras valem.

**3. Deep-dive de domínio ML/AI (1 rodada).** Profundidade conceitual sobre o que consta no seu currículo: transformers, sampling, fine-tuning vs RAG vs prompting, evals, mitigação de hallucination. As seções 01-10 deste repositório cobrem essa rodada. O entrevistador vai apertando até você travar, e saber onde termina o seu conhecimento (e dizer isso) sempre vence blefar.

**4. AI system design (1 rodada).** "Projete um support copilot / RAG corporativo / plataforma de agents." A seção 11 cobre esse tema. O diferencial é começar por evals, custo e modos de falha, em vez de caixas e setas.

**5. Behavioral (1 rodada, às vezes diluída nas demais).** As perguntas estão em [questions.pt-BR.md](questions.pt-BR.md). Em níveis sênior ou acima, essa rodada pesa tanto quanto qualquer rodada técnica.

**6. Às vezes: take-home ou work trial pago.** Veja mais abaixo. É mais comum em startups, e algumas fazem trials presenciais pagos de 1 a 5 dias no lugar de parte do loop.

## Como os loops mudam conforme o tipo de empresa

| | Frontier labs (OpenAI, Anthropic, Google DeepMind, Meta) | Big tech (AWS, Azure, orgs do GCP, times de produto) | Startups AI-native | Empresas tradicionais adotando AI |
|---|---|---|---|---|
| **Barra de coding** | Muito alta; geralmente várias rodadas, com cara de problema do mundo real | Alta, DSA padrão + prático | Prático > algorítmico; entregar algo funcionando rápido | Moderada; fluência na linguagem/stack |
| **Rodada de domínio** | Profunda: espere fluência no nível de papers sobre attention, RLHF/GRPO, scaling, inference | Amplitude + como você levaria isso para produção | Aplicada: RAG, agents, evals, "o que você construiria com X" | Integração: APIs, data pipelines, tradeoffs entre fornecedores |
| **System design** | Infra de pesquisa ou serving em escala extrema | Sistemas distribuídos clássicos + camada de AI | Orientado a produto: design de feature 0→1 | Confiabilidade, compliance, governança de custo |
| **O que garante ofertas** | Profundidade + fluência em pesquisa + coding excepcional | Experiência em escala, maturidade de processo, histórias cross-team | Velocidade de entrega, amplitude, product sense, portfólio | Gestão de risco, habilidade com stakeholders, pragmatismo |
| **Atenção a** | Loops longos, curveballs de pesquisa | Jogos de mapeamento de nível, barra que depende da org | Rodadas de feeling do fundador, work trials | Screens de buzzwords feitos por quem não é especialista |

Calibre a preparação: para um frontier lab, invista pesado nas seções 01-02 e 05; para uma startup, tenha um projeto de portfólio que você consiga demonstrar e defender de ponta a ponta; para big tech, ensaie histórias comportamentais cross-funcionais, com escopo e números.

## Decodificador de títulos de cargo

Os títulos variam muito entre empresas, então leia a job description, não o título. Uma decodificação aproximada:

- **AI Engineer / GenAI Engineer**: constrói produtos *em cima de* modelos (RAG, agents, prompting, evals, integração de serving). É basicamente engenharia de software com uma camada específica de LLM. O que cai: tudo neste repositório, exceto a matemática mais pesada de treinamento.
- **ML Engineer (MLE)**: treina, ajusta e faz deploy de modelos; cuida de pipelines, feature stores e infra de treinamento. Tem mais ML clássico + infra. O que cai: fundamentos de ML, MLOps, treinamento distribuído e, cada vez mais, a camada de LLM.
- **Research Engineer (RE)**: implementa e escala ideias de pesquisa nos labs (runs de treinamento, ablations, performance engineering). Exige coding forte + teoria sólida de ML; publicações são opcionais, mas fluência em papers é obrigatória. O que cai: implementar attention do zero, debugar um run de treinamento, perguntas de GPU/paralelismo.
- **Applied Scientist / Research Scientist**: desenha experimentos e responde por métricas e escolhas de modelagem; RS normalmente exige publicações. O que cai: estatística, design experimental, profundidade em modelagem, discussão de papers.

Se a JD de um "AI Engineer" pede CUDA e pretraining, é uma vaga de RE disfarçada. Se a JD de um "MLE" é só LangChain e vector DBs, é AI Engineering. Prepare-se de acordo com o conteúdo real da JD.

## Take-home assignments

O pedido típico é: "Construa um pequeno serviço de RAG/agent/extraction sobre estes dados; ~4-8 horas." O que os avaliadores realmente pontuam, em ordem aproximada de prioridade:

1. **Roda.** Setup com um comando, dependências fixadas, nada de surpresa com API key faltando. Funcionando e sem graça vence ambicioso e quebrado: o avaliador gasta 15 minutos, e um crash no primeiro minuto é fatal.
2. **Evals incluídos.** É o maior diferencial em 2026. Até 15 casos de teste escritos à mão, com um script de pass/fail, sinalizam mais senioridade do que qualquer escolha de arquitetura. Inclua uma análise de falhas: "falha em X por causa de Y."
3. **README com tradeoffs.** O que você construiu, o que deixou de fora de propósito, o que faria com mais uma semana. Assumir seus atalhos ("sem streaming, só chamadas síncronas: aceitável nesta escala porque...") soa sênior; escondê-los soa júnior.
4. **Engenharia sensata**: código tipado, tratamento de erros em toda chamada de LLM, noção de custo/tokens e nada de sopa de frameworks para um problema de 300 linhas.

**Como dimensionar seu tempo**: limite-se às horas informadas mais ~25%. Distribua cerca de 50% na funcionalidade principal, 25% em evals + análise de erros e 25% em README e limpeza. Não perca tempo caprichando na UI.

**Quando recusar**: trabalho não remunerado que passe de ~um dia, assignments suspeitamente iguais ao roadmap real da empresa, ou take-homes enviados antes de qualquer pessoa ter conversado com você. Vale fazer uma contraproposta: "Não consigo fazer 20 horas sem remuneração, mas aqui está um repo meu que demonstra as mesmas habilidades. Posso apresentá-lo ao vivo, com prazer."

**Sobre assistência de AI**: parta do princípio de que vão pedir que você defenda cada linha numa call de follow-up. Use as ferramentas que quiser, mas, se você não consegue explicar uma escolha de design, é porque não foi você que a fez.

## Portfólio e projetos que realmente impressionam

Um projeto profundo, com evals e um writeup, vale mais que dez repos de "clonei um tutorial do YouTube". Aqui vão ideias concretas, cada uma com o sinal que ela transmite:

1. **App de RAG sobre um corpus real e bagunçado** (suas anotações, um site de docs complicado, processos judiciais) com um eval harness de verdade: métricas de retrieval, checagens de faithfulness e um writeup de análise de erros que agrupa as falhas. *Demonstra: fundamentos de retrieval + a maturidade em evals que falta à maioria dos candidatos.*
2. **Um agent com execução de tools em sandbox**, por exemplo um agent de código ou de análise de dados que roda tools em Docker, com permission gating, timeouts e um audit log. *Demonstra: design de agents e o raciocínio de segurança que os entrevistadores raramente veem.*
3. **Tokenizer + transformer do zero**: um tokenizer BPE e um modelo pequeno estilo GPT em PyTorch puro (o caminho zero-to-hero do Karpathy), treinado num corpus de brinquedo, com notas sobre o que você aprendeu. *Demonstra: fundamentos que sobrevivem à rodada de domínio mais profunda.*
4. **Um writeup de benchmark de inference**: o mesmo modelo em vLLM vs llama.cpp vs a API de um provedor, com curvas de TTFT/throughput vs batch size e context length, custo por milhão de tokens e metodologia declarada. *Demonstra: conhecimento de serving e rigor empírico.*
5. **Um fine-tune com evals antes/depois**: LoRA num modelo open source pequeno, em uma tarefa estreita; compare com um modelo frontier usado via prompt em qualidade, latency e custo, e publique o veredito mesmo que o fine-tuning tenha *perdido*. *Demonstra: você sabe quando fine-tuning vale a pena e quando não vale, um sinal de entrevista de primeira linha.*
6. **Estudo de calibração de LLM-as-judge**: construa um judge para alguma tarefa, meça a concordância com seus próprios rótulos humanos e documente os vieses de posição, tamanho e self-preference que encontrar. *Demonstra: sofisticação em evals, indo além de "usamos o GPT para avaliar."*
7. **Pipeline de structured-extraction**: milhares de documentos bagunçados → schema validado, com retries, controle de custo e um relatório de accuracy. *Demonstra: engenharia de produção sem glamour, que é a maior parte do trabalho de verdade.*
8. **Um MCP server para uma ferramenta ou fonte de dados real**, com auth, validação de input e testes, utilizável a partir de clients Claude/IDE. *Demonstra: padrões atuais de integração e design de API.*
9. **Um model router / semantic cache**: um classificador pequeno ou uma heurística que manda as queries fáceis para um modelo barato e as difíceis para um modelo frontier, com uma fronteira qualidade-vs-custo medida. *Demonstra: engenharia de custo baseada em dados.*
10. **Um app de voz ou multimodal com latency budget**: documente o pipeline de ponta a ponta, para onde vai cada milissegundo e o que você fez a respeito do hop mais lento. *Demonstra: streaming, UX sob latency e pensamento de sistemas.*

Todo projeto precisa de três artefatos: um README com números, um diretório evals/ e um writeup curto sobre o que te surpreendeu. O writeup é o que os entrevistadores de fato leem.

## Dicas de currículo para vagas de AI

- **Métricas em cada bullet, e evals sempre que der.** "Entreguei assistente de RAG" é ruído. "Entreguei assistente de RAG de suporte para 40k usuários; subi a taxa de aceitação de respostas de 61%→78% num eval de 500 casos; cortei o custo/query em ~70% com caching + model routing" consegue entrevista. Se não pode compartilhar números internos, use deltas relativos e indicadores de escala.
- **Datas de entrega e durações.** "Protótipo→GA em 9 semanas" sinaliza velocidade. Atualidade pesa nesta área, então comece pelo trabalho de 2024-2026.
- **Coloque links para os artefatos.** Repos no GitHub, writeups, demos, papers. Uma afirmação no currículo com prova clicável vale três sem ela.
- **Diga qual camada era sua.** "Fui responsável pelo pipeline de retrieval + eval" vence um vago "trabalhei no chatbot de AI." Os entrevistadores aprofundam exatamente onde seus bullets reivindicam ownership, então garanta que você aguenta três perguntas de profundidade em cada um.
- **Acabe com a sopa de palavras-chave.** Uma linha de skills com 30 frameworks soa júnior. Liste só aquilo em que você toparia ser sabatinado.
- **Não infle.** Escrever "Fine-tuned GPT-4" quando você só escreveu prompts, ou "trained an LLM" quando rodou LoRA num 7B durante uma tarde, vai ser descoberto na primeira rodada e envenena tudo o que vem depois.

## Como se manter atualizado sem se afogar

Você não consegue ler tudo, e os entrevistadores não esperam isso. Eles esperam que você tenha um *sistema*. Uma dieta semanal sensata (~3 horas):

- **Um digest curado** para ter amplitude: escolha um ou dois entre uma boa newsletter, Latent Space, o blog do Simon Willison ou threads de AI no Hacker News.
- **Uma coisa lida a fundo**: um paper, um model card ou um post sério de blog de engenharia. A profundidade em uma coisa por semana se acumula; passar os olhos em quarenta abstracts, não.
- **Release notes dos provedores que você usa**: deprecações de modelos, mudanças de API, preços. Isso tem relação direta com o trabalho, coisa que a maioria dos papers não tem.
- **Uma hora de prática**: teste o novo modelo ou ferramenta no *seu próprio* eval set ou side project. "Rodei no meu benchmark e notei X" é resposta de entrevista de elite; "vi uma thread sobre isso" não é.

E *pule* de propósito: o feed o dia inteiro, os hot takes do dia de lançamento e qualquer paper que você não consiga conectar a algo que você constrói.

## Perguntas que VOCÊ deve fazer aos entrevistadores

Boas perguntas servem de diagnóstico para você *e* sinalizam senioridade. Pergunte sobre:

- **Cultura de evals**: "Como vocês sabem que uma mudança de prompt ou de modelo é segura para entregar? Me conduzam pela última." (Sem resposta de verdade, você vai passar o tempo apagando incêndio de regressões guiadas por vibes.)
- **Acesso a dados**: "Que dados o time pode de fato usar para treinamento/evals, e quanto tempo leva para liberar o acesso?" (Aprovação de dados que leva meses mata times de AI em silêncio.)
- **Realidade de GPU/orçamento**: "Como está o orçamento de compute e de API? Os experimentos entram na fila atrás da produção?"
- **Como entregam**: "Qual é o caminho do notebook até a produção? Quem pode fazer push de uma mudança de prompt, e o que a barra?"
- **On-call**: "Features de AI têm on-call? Como é um incidente? Quem debuga uma regressão de qualidade às 2h da manhã?"
- **Estratégia de modelos**: "O que aconteceu na última migração ou deprecação de modelo de vocês?" (Mostra o quanto estão acoplados a um único fornecedor e quanta dívida técnica você estaria herdando.)
- **Para startups**: runway, quem são os design partners/clientes e qual foi o último pivot.

## Noções básicas de avaliação de oferta

Isto é só orientação geral; faça sua própria pesquisa com dados atuais:

- **O jogo é o mapeamento de nível.** A mesma experiência cai em níveis diferentes em empresas diferentes, e a remuneração é definida pelo nível. Antes de negociar valores, negocie o nível: subir um nível normalmente vale muito mais do que qualquer aumento dentro da faixa. Use o [levels.fyi](https://www.levels.fyi/) e conversas com pares para calibrar títulos entre empresas.
- **Equity de startup de AI**: entenda a diferença entre o seu strike price (409A) e o preço preferencial da última rodada; o "seu equity vale $X" que aparece em destaque usa o segundo. Pergunte sobre a expectativa de diluição total, as janelas de exercício e se a empresa já permitiu vendas secundárias. Valuations altas em AI significam preference stacks altas, então modele o cenário negativo, não só o caso unicórnio.
- **Mix de caixa vs. equity**: frontier labs e big tech pagam muito em equity (líquido em empresas públicas, menos nas demais). Uma oferta de startup 40% abaixo do caixa de uma big tech precisa de uma história de equity crível para competir. Faça-os construir essa história.
- **Fontes para pesquisar remuneração**: levels.fyi, divulgações salariais de H-1B (registro público nos EUA), conversas com recruiters de empresas pares e, simplesmente, perguntar à sua rede. Nunca negocie com base em um único ponto de dados.
- **Termos não salariais que importam em vagas de AI**: acesso a compute/GPU, política de publicação, carve-outs de IP para side projects, expectativas de on-call e política de trabalho remoto. Em vagas próximas de pesquisa, o direito de publicar pode valer mais do que um aumento salarial para o seu valor de mercado no longo prazo.

## Mock interviews e mentoria

Ler respostas não é o mesmo que dá-las sob pressão. Antes de um loop real, faça pelo menos uma mock completa: use o [mock interview kit](mock-interview-kit.md) com um amigo, ou agende uma sessão com um entrevistador experiente, que dá uma avaliação honesta e monta um plano de preparação em torno das suas lacunas, em [enginebogie.com/u/om](https://enginebogie.com/u/om) ou [topmate.io/ombharatiya](https://topmate.io/ombharatiya).

## Perguntas de entrevista

42 perguntas comportamentais e de experiência, específicas do trabalho com AI e com orientação de resposta, em [questions.pt-BR.md](questions.pt-BR.md).

## Red flags que os entrevistadores observam

- **Nenhum número em lugar nenhum.** Toda história de projeto sem usuários, latency, custo ou métricas de eval sinaliza que você viu o trabalho acontecer, em vez de ser dono dele.
- **"Não tivemos tempo para evals."** É o jeito mais rápido de perder senioridade em 2026. Até times improvisados rotulam 50 casos à mão.
- **Culpar o modelo.** "O GPT-whatever simplesmente alucinou", sem uma história de mitigação, mostra que você parou de debugar na fronteira da API.
- **Reivindicar a vitória do time na primeira pessoa do singular**, e depois travar em duas perguntas de follow-up sobre como aquilo de fato funcionava.
- **Inflar o currículo em afirmações sobre treinamento.** Dizer "fine-tuned"/"trained" para o que foi prompt engineering se desfaz em minutos e contamina todas as outras afirmações.
- **Name-dropping de frameworks no lugar de entendimento**: recita LangChain/LlamaIndex, mas não consegue descrever o que um agent loop faz sem eles.
- **Nenhuma pergunta para o entrevistador**, ou só perguntas sobre benefícios, o que passa a impressão de pouco ownership e pouca curiosidade.
- **Falar mal de um empregador ou time anterior.** Mesmo quando é justificado, a negatividade acaba sobrando para você.

## Leituras complementares

- [levels.fyi](https://www.levels.fyi/) - mapeamento de níveis e dados de remuneração entre empresas.
- [Machine Learning Interviews Book - Chip Huyen](https://huyenchip.com/ml-interviews-book/) - o livro gratuito clássico sobre loops de entrevista de ML, papéis e preparação.
- [Applied LLMs - What We Learned from a Year of Building with LLMs](https://applied-llms.org/) - lições de quem trabalha com isso na prática (evals, RAG, ops) que se encaixam direto em histórias de entrevista.
- [Your AI Product Needs Evals - Hamel Husain](https://hamel.dev/blog/posts/evals/) - o playbook de maturidade em evals que os entrevistadores esperam, cada vez mais, que você tenha internalizado.
- [Simon Willison's blog](https://simonwillison.net/) - cobertura prática e de alto sinal sobre novos modelos e ferramentas; um bom exemplo de como se manter atualizado.
- [Latent Space](https://www.latent.space/) - podcast/newsletter sobre AI engineering como disciplina, incluindo contratação e definição de papéis.
- [Neural Networks: Zero to Hero - Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) - o caminho canônico, do zero, para o projeto de portfólio de fundamentos.
- [The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/) - contexto do mercado sobre processos de contratação, níveis e como as empresas de tecnologia realmente conduzem loops.
