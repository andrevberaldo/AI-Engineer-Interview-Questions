# 🏗️ AI System Design

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

AI system design já é uma etapa padrão do loop de entrevistas para vagas de AI/GenAI/LLM engineer em frontier labs, big techs e startups de IA. Costuma ser uma sessão de whiteboard de 45 a 60 minutos, com enunciados como "Projete um agent de suporte ao cliente" ou "Projete uma busca semântica para o nosso catálogo". O objetivo é ver se você consegue transformar uma ideia de produto vaga em um sistema que dá para entregar, medir e pagar, com um modelo dentro. Candidatos seniores caem nesta rodada mais por falta de evals e de conta de custo do que por falta de caixas no diagrama.

## Como essas entrevistas diferem do system design clássico

As intuições do system design clássico continuam valendo (load balancing, sharding, caching, filas), mas quatro coisas mudam de forma fundamental:

| Dimensão | System design clássico | AI system design |
|---|---|---|
| Componente central | Serviços determinísticos que você escreve | Um **modelo que você não controla por completo**, tratado como uma dependência falível e versionada |
| Correção | Demonstrável: testes, invariantes, ACID | **Qualidade probabilística**: não dá para provar que o sistema está certo, então **evals substituem as provas de correção** e você argumenta com accuracy, faithfulness e taxas de sucesso de tarefa medidas |
| Modelo de custo | Infra majoritariamente fixa; custo marginal por requisição ≈ 0 | **O custo por requisição pesa** (de frações de centavo a dólares). A conta de tokens é o novo capacity planning, e a escolha do modelo é uma das principais alavancas de custo |
| Latency | Um único número (p99 de uma RPC) | **Latency com várias componentes**: time-to-first-token vs tokens/s vs total; UX de streaming; loops de agent levam de segundos a minutos |
| Modos de falha | Crashes, timeouts, perda de dados | Tudo isso, mais **falhas semânticas**: hallucination, prompt injection, recusa, drift depois de uma atualização silenciosa do modelo |
| Iteração | Entregar features | **Quality flywheel**: registrar traces → montar eval sets → melhorar prompts/retrieval/modelos → medir de novo |

Se a sua resposta seria idêntica com um serviço determinístico no lugar do modelo, você não está respondendo a uma pergunta de AI system design.

## O framework de resposta em 8 passos

Use este esqueleto para qualquer enunciado. Numa entrevista de 45 minutos, a divisão é: ~5 min no passo 1, ~10 min nos passos 2-3, ~15 min no passo 4 e ~15 min nos passos 5-8. Os entrevistadores dão muito peso a evals e ops, então não deixe o tempo acabar antes de chegar neles.

### 1. Requisitos e métricas de sucesso
Esclareça usuários, escala (QPS, tamanho do corpus, crescimento), expectativas de latency e **o que "bom" significa em números**. Por exemplo: "accuracy das respostas ≥ 90% em um golden set, deflection rate ≥ 50%, p95 de TTFT < 1,5s, custo < $0,05/query". Antes de desenhar qualquer coisa, diga qual é a north-star metric e quais são as guardrail metrics.

### 2. Estratégia de modelo
API vs self-host (sensibilidade dos dados, economia de escala, controle de latency, carga operacional) e **tiering**: mande o tráfego fácil e de alto volume para um modelo pequeno e barato, e o tráfego difícil para um modelo de fronteira. Diga quais são os fallbacks entre provedores e como você vai fixar (pin) e atualizar as versões do modelo.

### 3. Estratégia de dados e contexto
Como o modelo obtém o conhecimento de que precisa? A ordem de decisão padrão é **prompting → RAG → tools → fine-tuning**. RAG serve para fatos que mudam; fine-tuning, para estilo, formato e habilidades específicas, não para manter o conhecimento atualizado; tools, para estado em tempo real e ações. Defina aqui chunking, indexação, pipeline de freshness e ACLs.

### 4. Arquitetura de alto nível
Desenhe o caminho da requisição: client → gateway → (router) → orquestração → retrieval/tools → modelo → guardrails → resposta, além dos caminhos offline (ingestão, eval, treinamento). Diga o que é síncrono e o que vai para fila, o que fica em cache e onde o estado mora.

### 5. Evals e loop de qualidade
Este é o passo que separa os seniores: um **golden set** (100-1.000 casos rotulados), scoring automatizado (exact-match quando possível; LLM-as-judge com rubrica quando não, calibrado contra rótulos humanos), métricas por componente (recall@k do retrieval separado da faithfulness da resposta), CI gating em mudanças de prompt/modelo e A/B online ou interleaving.

### 6. Deployment e ops
Orçamento de latency por estágio, streaming, caching (prompt caching, semantic caching, implicações do KV cache), rate limits e gestão de cotas dos provedores, degradação graciosa (modelo de fallback, respostas "não sei", respostas em cache) e tracing de toda chamada de LLM com prompt, versão, tokens e latency.

### 7. Segurança e guardrails
Input guards (detecção de injection, redação de PII, filtros de tópico), output guards (validação de schema, checagem de citações, moderação), **action guards** para agents (tools em allowlist, limites de gasto e de blast radius, aprovação humana para ações irreversíveis) e isolamento de tenants.

### 8. Plano de iteração
O que entra na semana 1 e o que entra no trimestre 1: comece pelo sistema mais simples que dê para medir (em geral, um único prompt + retrieval), instrumente tudo e deixe os dados de eval ditarem a complexidade. Diga explicitamente o que você *ainda não* construiria.

## Building blocks reutilizáveis

A maioria dos sistemas de IA é montada com os mesmos ~10 blocos. Falar deles com naturalidade sinaliza experiência:

| Bloco | O que faz | Notas de design |
|---|---|---|
| **Model gateway** | Ponto único de passagem para todas as chamadas de LLM: auth, cotas, retries, failover entre provedores, medição de uso | Compre ou construa algo fino; nunca deixe os serviços chamarem os provedores diretamente |
| **Router** | Envia cada requisição ao tier de modelo certo (pequeno e rápido vs de fronteira) por classificador, heurística ou tipo de tarefa | A maior alavanca de custo isolada; meça a diferença de qualidade por tier |
| **Retrieval service** | Busca híbrida (BM25 + dense) sobre um corpus dividido em chunks e marcado com permissões, com reranking | Serviço próprio, com evals próprios (recall@k, nDCG) |
| **Context builder** | Monta o prompt: instruções de system, chunks recuperados, memória e schemas de tools, dentro de um token budget | Determinístico e testável; registre o prompt final exato |
| **Orchestrator** | Executa fluxos de várias etapas: loops de agent, dispatch de tools, retries, timeouts, limite máximo de iterações | Durable execution (com fila) para agents de longa duração |
| **Guardrails layer** | Checagens de input, output e ação: injection, PII, validação de schema, moderação, policy engine para tool calls | Fail closed em ações, fail open (com logging) em estilo |
| **Eval harness** | Roda golden sets contra qualquer configuração de prompt/modelo/retrieval; pontua, compara diffs e bloqueia o CI | Versione tudo: prompts, datasets, prompts de judge |
| **Tracing e observability** | Trace por requisição de cada chamada de modelo, tool call e retrieval, com tokens, custo, latency e feedback anexados | A matéria-prima para eval sets e debugging |
| **Caches** | Cache de resposta por exact-match, semantic cache, prompt caching do provedor para prefixos compartilhados | Prompt caching costuma cortar o custo de input em 50-90% em contextos longos compartilhados |
| **Feedback store** | Thumbs, edições, aceitações e escalonamentos, ligados aos traces | Alimenta o quality flywheel e os dados de fine-tuning |
| **Human-in-the-loop queue** | UI de revisão para saídas de baixa confiança e ações de alto risco | Projete o caminho de escalonamento antes de passar a confiar no modelo |

## O que os entrevistadores avaliam

- **Perguntas de esclarecimento primeiro.** Pular direto para a arquitetura é a reprovação mais comum. Escopo, escala, nível de qualidade esperado e restrições nunca vêm totalmente especificados, e isso é de propósito.
- **Articulação de tradeoffs.** Toda escolha ("RAG em vez de fine-tuning", "self-host do modelo de completion") vem com a alternativa, o motivo e o que faria você mudar de ideia.
- **Eval literacy.** Você consegue dizer, de forma concreta, como saberia que o sistema funciona: datasets, métricas, judges, experimentos online? Em 2026, este é o critério que mais separa sênior de júnior nos loops.
- **Consciência de custo.** Conta de padaria com tokens: requisições/dia × tokens/requisição × $/token. O padrão esperado é chegar a uma estimativa dentro de 10x, com as premissas declaradas com clareza.
- **Pragmatismo.** Comece pelo mais simples e só adicione complexidade quando a necessidade for medida. "Eu começaria com um prompt e um golden set" vale mais do que um enxame de agents com 12 caixas.
- **Pensamento em modos de falha.** Injection, citações alucinadas, queda de provedor, atualizações silenciosas de modelo, índices desatualizados: traga esses pontos sem esperar ser perguntado.
- **Realismo de latency.** Saiba que TTFT, streaming e o número de iterações do loop de agent dominam a velocidade percebida, e faça o orçamento por estágio.

## Red flags que os entrevistadores observam

- Projetar tratando o modelo como um oráculo infalível: sem plano de eval, sem fallback, sem caminho para "quando ele erra".
- Fine-tuning como primeira resposta para um problema de conhecimento desatualizado.
- Nenhum número em lugar nenhum: sem QPS, sem contagem de tokens, sem estimativa de custo, sem orçamento de latency.
- Maximalismo de agents: enxames multi-agent para um problema que um único prompt + retrieval resolve.
- Ignorar ACLs e tenancy em cenários enterprise ("é só embedar tudo em um único índice").
- Tratar o "LLM-as-judge" como gratuito e infalível, em vez de um componente que também precisa ser calibrado.
- Nenhuma menção a prompt injection quando o design ingere conteúdo não confiável ou executa ações.

## Prompts de prática rápida

Pratique o esqueleto de 8 passos nestes enunciados até virar reflexo. Eles têm uma linha só de propósito: gerar as perguntas de esclarecimento faz parte do exercício.

1. Projete um assistente de conhecimento corporativo sobre wikis, documentos e tickets.
2. Projete um assistente de código com IA, com completion inline, chat e edições agentic.
3. Projete um agent de suporte ao cliente que consulta pedidos e emite reembolsos.
4. Projete uma busca semântica para um catálogo de e-commerce com 100M de itens.
5. Projete um sistema de moderação de conteúdo para posts de usuários a 10k posts/s.
6. Projete um pipeline de document intelligence para faturas e contratos.
7. Projete um agent de analytics que converte linguagem natural em SQL para usuários de negócio.
8. Projete um assistente de reuniões: transcrição, resumos, action items e busca de follow-ups.
9. Projete um assistente de triagem e redação de e-mails para um time de vendas.
10. Projete um sumarizador personalizado de notícias/feed para 10M de usuários diários.
11. Projete uma feature flag de IA: lançar um novo modelo para 5% do tráfego com segurança.
12. Projete um pipeline em batch para classificar e etiquetar 500M de documentos legados.
13. Projete um voice agent para reservas de restaurante, de ponta a ponta.
14. Projete um bot de code review para um monorepo com 2.000 engenheiros.
15. Projete um LLM gateway para uma empresa com 40 times chamando 5 provedores de modelo.

## Case studies

Soluções completas e resolvidas, seguindo um template único (problema → esclarecimentos → requisitos → arquitetura → deep-dives → estratégia de dados → evals → conta de custo → modos de falha → ops → follow-ups):

1. [Enterprise RAG Assistant](case-studies/01-enterprise-rag-assistant.md) - assistente de conhecimento corporativo: ACLs, freshness, citações, busca híbrida, eval harness.
2. [AI Code Assistant](case-studies/02-ai-code-assistant.md) - completion na IDE + chat + agents: tiers de latency, FIM, contexto do repositório, privacidade, evals de taxa de aceitação.
3. [Customer Support Agent](case-studies/03-customer-support-agent.md) - agent com tools sobre pedidos/reembolsos: guardrails de ação, handoff, métricas de deflection, defesa contra injection.
4. [Semantic Search at Scale](case-studies/04-semantic-search.md) - busca de e-commerce: retrieval híbrido, reranking, operação de índice ANN, refresh de embeddings, orçamentos abaixo de 100ms.
5. [Content Moderation](case-studies/05-content-moderation-pipeline.md) - classificação de alto throughput com modelos em tiers e revisão humana.
6. [Document Intelligence](case-studies/06-document-intelligence-pipeline.md) - extração de faturas/contratos com validação de schema e HITL.
7. [Text-to-SQL Agent](case-studies/07-text-to-sql-agent.md) - analytics em linguagem natural sobre um warehouse, com verificação de correção.
8. [Meeting Assistant](case-studies/08-meeting-assistant.md) - transcrição, sumarização, action items e busca em reuniões.
9. [Real-time Voice Agent](case-studies/09-realtime-voice-agent.md) - um agent telefônico com orçamento de voz para voz abaixo de um segundo, com barge-in e turn-taking.
10. [LLM Gateway and Serving Platform](case-studies/10-llm-gateway-and-serving-platform.md) - roteamento multi-tenant, cotas, failover e o data plane de serving por trás de tudo isso.

## Leituras complementares

- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents) - o argumento de referência para "comece simples e só adicione agência quando necessário".
- [Patterns for Building LLM-based Systems & Products - Eugene Yan](https://eugeneyan.com/writing/llm-patterns/) - evals, RAG, guardrails e caching como padrões reutilizáveis.
- [What We Learned from a Year of Building with LLMs - O'Reilly](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) - notas de campo de quem trabalha com isso, ao longo de toda a stack.
- [LLM Powered Autonomous Agents - Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/) - a taxonomia de referência para os componentes de um agent.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - por que a montagem e a ordem do contexto importam.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) - o clássico sobre por que o modelo é a caixa pequena do diagrama.
- [Designing Data-Intensive Applications - Kleppmann](https://dataintensive.net/) - a metade de system design clássico da entrevista ainda se apoia nele.
