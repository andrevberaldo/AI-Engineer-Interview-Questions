# 🤖 Agents, Tool Use & MCP

> 🇧🇷 Tradução em português do Brasil. Versão original em inglês: [README.md](README.md).

> 📚 Material de apoio: [Glossário de estudo](glossary.pt-BR.md), com explicações simples dos termos técnicos deste módulo.

Um dos tópicos de AI engineering mais cobrados em 2025-2026. Todo frontier lab, time de AI de big tech e startup de agents pergunta sobre o assunto, e ele domina as rodadas de system design para vagas de AI engineer, GenAI engineer e LLM platform. Os entrevistadores usam o tema para separar quem colocou um agent em produção de quem só assistiu a uma demo: as perguntas exploram matemática de confiabilidade, segurança e engenharia de custo, não apenas vocabulário.

## Crash course

### Workflows vs agents

O enquadramento da Anthropic (do artigo *Building Effective Agents*) é o vocabulário padrão de entrevista. **Workflows** orquestram LLMs e tools por *caminhos de código predefinidos*: prompt chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer. **Agents** deixam o LLM *dirigir o próprio processo de forma dinâmica*: ele decide quais tools chamar, em que ordem e quando terminou. Agência é um dial, não um binário. Cada passo em direção a mais agência traz flexibilidade, mas custa previsibilidade, latency, dinheiro e facilidade de avaliação.

**A regra de design mais importante:** se você consegue desenhar o fluxograma, escreva o workflow. Só recorra a um agent quando o número de passos e as ramificações realmente não puderem ser enumerados de antemão ("corrija este teste que está falhando", "pesquise esta empresa") e o valor de resolver o problema for maior que o custo do loop.

### O loop central do agent

Um agent é: **um modelo + tools + contexto/memória, rodando em um loop, com condições de parada.** É só isso.

```mermaid
flowchart TD
    U[Requisição do usuário] --> CTX["Montar o contexto:<br/>system prompt + definições de tools + memória"]
    CTX --> LLM[Chamada ao LLM]
    LLM -->|"emite tool calls"| EXEC["Seu runtime executa as tools<br/>(valida, sandbox, checa permissões)"]
    EXEC --> RES[Anexar resultados das tools ao contexto]
    RES --> STOP{"Parada forçada?<br/>máx. iterações / orçamento / timeout"}
    STOP -->|"não"| LLM
    STOP -->|"sim"| HITL[Escalar para um humano]
    LLM -->|"texto simples, sem tool calls"| DONE["Resposta final ou ação concluída"]
```

O esqueleto cabe em uma tela de Python. As versões de produção envolvem o mesmo formato com retries, streaming e tracing. Os entrevistadores costumam pedir que você o esboce:

```python
def run_agent(user_msg: str, tools: list, max_iters: int = 15):
    messages = [{"role": "user", "content": user_msg}]
    spend = Budget(max_tokens=500_000, max_cost_usd=5.0)
    for _ in range(max_iters):
        resp = llm(messages, tools=tools)        # o modelo pode emitir tool calls
        messages.append(resp.message)
        spend.add(resp.usage)
        if not resp.tool_calls:                  # parada natural: resposta só com texto
            return resp.text
        for call in resp.tool_calls:             # VOCÊ executa; o modelo nunca executa
            result = execute(call.name, call.arguments)   # valide os args, sandbox,
            messages.append(tool_result(call.id, result)) # controle ações irreversíveis
        if spend.exceeded():
            break
    return escalate_to_human(messages)           # nunca entre em loop infinito
```

### Mecânica de tool calling

- Você passa as **definições de tools** (nome, descrição, **JSON Schema** dos parâmetros) a cada requisição. Elas consomem tokens de input.
- O modelo não executa nada. Ele emite uma **mensagem estruturada**: nome da tool, argumentos em JSON e um call ID. Os modelos são pós-treinados (SFT + RL) para produzir esse formato. Os providers também podem usar **constrained decoding** (por exemplo, definições de tools com `strict: true` nas APIs da OpenAI e da Anthropic) para garantir argumentos válidos conforme o schema. Válido não é o mesmo que correto: uma chamada bem formada ainda pode carregar um ID inventado.
- Seu código executa a chamada e anexa uma **mensagem de tool result** associada ao call ID. Aí o loop continua.
- **Parallel tool calls:** o modelo pode emitir várias chamadas independentes em um único turno (buscar três URLs). Execute-as em paralelo e devolva todos os resultados.
- **Forçar a escolha de tool (tool choice):** `auto` (o modelo decide), `required`/`any` (precisa chamar *alguma* tool), uma tool específica pelo nome (truque clássico para garantir extração estruturada) ou `none`.

### Design de tools

A parte da engenharia de agents que mais rende e menos aparece:

- **Poucas tools, distintas e bem nomeadas.** Tools que se sobrepõem ("search", "find", "lookup") causam erros de tool errada. Use namespace nas tools relacionadas (`jira_create_issue`).
- **Descrições são prompts.** Escreva como se fosse a documentação para quem acabou de entrar no time: o que a tool faz, quando usar, quando *não* usar, o significado dos argumentos e um exemplo.
- **Consolide cadeias previsíveis.** `schedule_meeting(attendees, when)` é melhor que `list_users` → `get_availability` → `create_event`: menos round trips e menos chances de erro. Deixe as tools granulares só onde o modelo realmente precisa ramificar. Mapeie tools para *intenções*, não para endpoints de API.
- **Outputs econômicos em tokens.** O agent paga por cada token de resultado até o fim da trajetória. Pagine, filtre, retorne IDs semânticos, trunque deixando ponteiros ou grave resultados grandes em um arquivo e retorne o caminho.
- **Erros que o modelo consegue usar para agir.** Retorne as falhas como tool results, não como crashes: "`status` inválido; valores válidos: open, closed, merged" transforma um beco sem saída em uma correção de rota. Stack traces brutos gastam tokens e confundem o modelo.

### Grandes superfícies de tools: tool search e code execution

Conecte seis servers MCP e você pode ter mais de 100 definições de tools e dezenas de milhares de tokens de schema no contexto antes de o usuário digitar qualquer coisa. O prompt caching resolve o custo, mas não ajuda na accuracy, porque os erros de tool errada crescem com o número de opções quase sinônimas. As alavancas são:

- **Curar e dividir:** faça allowlist de tools por agent, não por server, use namespace por serviço e dê a cada subagent só as tools do próprio domínio.
- **Progressive disclosure:** mantenha um conjunto pequeno sempre carregado, mais um `search_tools(query)` ou um loader por domínio que traz as definições sob demanda. Assim, a seleção de tools vira um problema de retrieval. Algumas APIs de providers já oferecem isso como tool search nativa, com carregamento adiado.
- **Code execution:** apresente as tools como uma API de código tipada dentro de um sandbox. O agent lê só os módulos de que precisa e escreve um script que combina as chamadas. Assim, resultados intermediários grandes são filtrados no sandbox e apenas o output final entra no contexto. O preço: operar um sandbox de verdade, mover os approval gates para um broker que enxergue cada chamada com efeito colateral e fazer tracing dentro do sandbox.

### MCP (Model Context Protocol)

**O problema:** N apps × M integrações = N×M adapters sob medida. O MCP é um padrão aberto lançado pela Anthropic em nov. de 2024, doado à Agentic AI Foundation, da Linux Foundation, em dez. de 2025, e hoje suportado nos ecossistemas da OpenAI, do Google e da Microsoft. Ele padroniza a interface e transforma a conta em N+M: você constrói um server uma vez e qualquer host MCP pode usá-lo.

**Arquitetura:** um **host** (Claude Desktop, uma IDE, seu app) executa um **client** por conexão com um **server** (que encapsula uma fonte de dados ou uma API). As mensagens são JSON-RPC 2.0. Desde a revisão **2026-07-28**, o protocolo é stateless: o handshake `initialize` e o header `Mcp-Session-Id` deixaram de existir, e cada requisição carrega a própria versão do protocolo e suas capabilities. Com isso, qualquer instância de server atrás de um load balancer comum pode respondê-la. O handshake antigo continua como caminho de compatibilidade para as revisões até 2025-11-25.

- **Primitivas do server:** **tools** (ações controladas pelo modelo), **resources** (dados/contexto controlados pelo app), **prompts** (templates acionados pelo usuário).
- **Funcionalidades do client:** **elicitation** (o server pede input ao usuário no meio de uma chamada), que na 2026-07-28 virou uma requisição com múltiplos round trips: o server retorna um resultado input-required e o client reenvia a chamada com as respostas. **Sampling** (o server usa emprestado o LLM do host), **roots** (escopo do filesystem) e logging estão deprecated na 2026-07-28, com pelo menos doze meses antes de qualquer remoção.
- **Transports:** **stdio** (server como subprocesso local: simples, mas herda os privilégios locais) e **Streamable HTTP** (servers remotos, autorização baseada em OAuth; substituiu o antigo transport HTTP+SSE na revisão 2025-03-26).

**Segurança:** um server MCP de terceiros injeta texto não confiável (descrições de tools, resultados) no contexto do seu modelo e, via stdio, executa código na sua máquina. Tool poisoning, atualizações de descrição do tipo rug-pull e tool shadowing são classes de ataque reais. Faça auditoria, fixe versões (pin) e rode com least-privilege.

### A2A (agent-to-agent)

O MCP é o eixo **vertical**: o agent descendo até as tools e os dados que você controla. O **A2A** é o eixo **horizontal**: o agent indo até um *agent par (peer)*, em geral do outro lado da fronteira de um time ou de uma empresa. O protocolo chegou à **v1.0**, é hospedado pela Linux Foundation e passou de 150 organizações apoiadoras já no primeiro ano (AWS, Cisco, Google, IBM, Microsoft, Salesforce, SAP, ServiceNow). "MCP vs A2A, e como eles se combinam" virou pergunta padrão de triagem em infra de agents.

- **Agent Card:** metadados em JSON publicados em uma URI well-known, declarando identidade, skills, endpoint, requisitos de autenticação e capabilities opcionais (streaming, push notifications). Pode ser assinado com **JWS**, para que quem chama consiga verificá-lo.
- **Tasks, não chamadas:** uma task tem um ciclo de vida: submitted, working, completed, failed, canceled, rejected, além dos estados interrompidos **input-required** e **auth-required**. As atualizações chegam por polling, streaming ou push para um webhook, porque a task de um peer pode levar minutos.
- **Opacidade é a propriedade central do design:** os peers colaboram em *capabilities declaradas*, sem expor prompts internos, tools, memória ou raciocínio. O que você tem é uma caixa-preta com contrato, não uma função, e é exatamente por isso que não é "MCP com passos extras".
- **Do outro lado de uma fronteira de organização:** credenciais por peer (OAuth 2.0 client credentials, mTLS) em vez de um token compartilhado; auditoria de dois principals (qual peer, agindo em nome de qual usuário final); o output do peer tratado como **conteúdo não confiável**, como qualquer tool result; e quotas e limites de profundidade por peer, porque o incidente mais provável é o denial of wallet.

Os dois se combinam: seu agent usa MCP para os seus sistemas e A2A para chegar ao agent de um vendor, que por sua vez usa servers MCP próprios que você nunca vê.

### Agent Skills (SKILL.md)

Uma **skill** é uma pasta com um `SKILL.md` (frontmatter YAML, com no mínimo `name` e `description`, mais instruções em Markdown) e, opcionalmente, `scripts/`, `references/` e `assets/` empacotados. Ela é carregada por **progressive disclosure em três estágios**: nome e descrição na inicialização, instruções completas quando uma task corresponde e arquivos empacotados sob demanda. A Anthropic a lançou como padrão aberto, e hoje ela é carregada por Claude Code, ChatGPT/Codex, Cursor, VS Code, GitHub Copilot, Gemini CLI, Goose, Kiro, Databricks, Snowflake, JetBrains Junie e outros.

**A regra de decisão:** tools e servers MCP dão ao agent uma *capability ou conexão* que ele não tem. Skills dão *conhecimento procedural* sobre um trabalho que ele já consegue fazer tecnicamente. RAG dá *fatos*. Skills também são um **artefato de supply chain**: são instruções executáveis mais scripts, carregadas em um contexto que contém suas credenciais. Por isso, revisão, pinning e proveniência valem exatamente como para uma dependência.

### Padrões de planning

- **ReAct** (Yao et al., 2022): alterne Thought → Action → Observation. É adaptativo, mas míope. Os modelos modernos de tool-calling e reasoning internalizam boa parte disso.
- **Plan-then-execute:** produza um plano logo no início, execute os passos (paralelizáveis, com modelos mais baratos em cada passo) e replaneje se algo falhar. É melhor para tasks longas e para revisão humana dos planos, mas fica frágil quando o ambiente surpreende você.
- **Reflection / self-critique:** gerar → criticar → revisar. Funciona melhor quando verificar é mais fácil que gerar e quando há sinais externos como âncora (resultados de testes, compiladores), porque os modelos são condescendentes ao avaliar o próprio trabalho. Os ganhos diminuem rápido depois de 1-2 rodadas.

### Multi-agent e subagents

O ganho real dos padrões **orchestrator-worker / subagent** é o **isolamento de contexto**: um worker gasta 100k tokens pesquisando e devolve um resumo de 1k tokens, mantendo limpo o contexto do orchestrator. Paralelismo e especialização vêm em segundo plano.

Multi-agent **ajuda** em trabalho de muita leitura e paralelizável (pesquisa: muitas pistas independentes). **Atrapalha** em trabalho de muita escrita com estado compartilhado (vários agents editando a mesma codebase entram em conflito), e é caro: a Anthropic relatou que o sistema multi-agent de pesquisa deles usou ~15× os tokens de uma interação de chat. **Handoffs** (passar o controle da conversa para um agent especialista) servem a domínios distintos, como a triagem de suporte. Já a orquestração serve a tasks que dá para decompor. Comece com um único agent com boas tools e só mude quando ele falhar de forma comprovada.

### Memória e gerenciamento de contexto

- **Working memory** = a context window. **Memória persistente** = arquivos, bancos de dados e vector stores que sobrevivem entre sessões e são trazidos de volta sob demanda.
- Trajetórias longas degradam: o contexto enche e a qualidade cai antes de chegar ao limite rígido ("context rot"). As contramedidas são **compaction** (resumir os turnos antigos e manter os recentes e as decisões-chave), **limpar tool results obsoletos**, **structured note-taking** (listas de todo e arquivos de scratchpad ficam fora do contexto e sobrevivem à compaction) e **isolamento por subagent**.
- Agents de longo horizonte (horas ou dias) precisam de **checkpointing e durable execution**: persistir o estado nas fronteiras entre etapas, tornar as tool calls idempotentes e retomar a partir de event logs, em vez de rezar para o processo nunca morrer.

### Confiabilidade, guardrails, segurança

- **A matemática do erro composto:** 95% de sucesso por passo ao longo de 20 passos = 0,95²⁰ ≈ **36%** de ponta a ponta. É por isso que demos não são produtos. Ataque o problema reduzindo passos, aumentando a confiabilidade por passo e adicionando *recuperação* (passos de verificação, erros acionáveis, checkpoints), para que as falhas deixem de ser terminais.
- **Guardrails:** máximo de iterações, orçamentos de tokens/custo, timeouts, detecção de chamadas duplicadas e **tiers de permissão**: leituras aprovadas automaticamente, escritas controladas e ações irreversíveis (enviar, deletar, pagar) atrás de aprovação humana. A reversibilidade define o grau de autonomia.
- **Prompt injection:** tool results são um canal que o atacante controla, e os modelos não conseguem separar dados de instruções de forma confiável. A **lethal trifecta** de Simon Willison (acesso a dados privados + conteúdo não confiável + um canal de exfiltração) é o teste de design: nunca deixe um único agent reunir os três. Defesa em profundidade: tools com least-privilege, controles de egress, execução de código em sandbox e approval gates. Nenhuma correção no nível do prompt basta.

### Avaliação, custo, observabilidade

- Avalie **resultados finais** (o estado final ficou correto?) para medir robustez, e **trajetórias** (tools certas, caminho eficiente, conformidade com políticas, em geral julgadas por um LLM contra uma rubrica) para fazer diagnóstico.
- **pass@k vs pass^k:** pass@k (pelo menos uma das k tentativas dá certo) mede capability. **pass^k** (todas as k tentativas dão certo, popularizado pelo τ-bench) mede a consistência que o usuário realmente sente. Agents em produção são julgados por pass^k.
- **Engenharia de custo/latency:** use tiers de modelos por passo (o modelo grande planeja, o pequeno extrai e roteia), aproveite o **prompt caching** (system prompt + definições de tools + prefixo do histórico formam um cache hit estável a cada iteração do loop), paralelize tool calls e compacte o contexto. Meça o **custo por task bem-sucedida**, não por chamada.
- **Observabilidade:** faça trace de cada chamada ao LLM e de cada tool call (args, resultados, tokens, latency) como spans, tenha suporte a replay e acompanhe taxa de sucesso, passos por task, gasto de tokens e taxas de erro das tools. Não dá para depurar um agent que você não rastreou.

### Frameworks

LangGraph, OpenAI Agents SDK, Claude Agent SDK, Google ADK, Microsoft Agent Framework, CrewAI e companhia entregam o scaffolding do loop, estado/checkpointing, tracing e a infraestrutura de handoff. Em troca, você perde transparência nas abstrações e controle sobre os prompts. A postura de um candidato forte: **construa o loop cru você mesmo primeiro** (ele é pequeno), adote um framework quando precisar da infraestrutura específica dele e mantenha prompts e tools independentes de framework, para poder trocar.

## Perguntas de entrevista

Veja [questions.pt-BR.md](questions.pt-BR.md): **65 perguntas** com respostas, da mecânica básica ao design avançado de produção.

## Red flags que os entrevistadores observam

- Dizer que o modelo "executa" tools: não saber que o modelo só emite uma intenção estruturada, e que o runtime do client executa, valida e devolve os resultados.
- Recorrer a um framework multi-agent para uma task que um único prompt ou um workflow fixo resolve, e não saber explicar quando *não* construir um agent.
- Projetar um loop sem condições de parada, orçamentos ou controle de permissões: "ele roda até terminar."
- Errar a matemática de confiabilidade: tratar 90-95% de sucesso por passo como pronto para produção em tasks de 20 passos, ou não ter resposta para erros compostos além de "prompts melhores."
- Afirmar que prompt injection se resolve mandando o modelo ignorar instruções injetadas, ou nunca ter ouvido falar da lethal trifecta e do design de tools com least-privilege.
- Avaliar agents só com pass@1 nas respostas finais: sem trajectory evals, sem pensar em consistência (pass^k), sem uma história de tracing.
- Despejar 50 tools granulares no contexto sem opinião sobre consolidação de tools, qualidade das descrições ou custo em tokens dos resultados.
- Citar nomes de frameworks sem conseguir esboçar o loop cru do agent em Python.

## Leituras complementares

- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents) - o ensaio canônico sobre workflows vs agents.
- [How we built our multi-agent research system - Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system) - números honestos sobre custos de tokens em multi-agent e lições de orquestração.
- [Writing effective tools for agents - Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents) - orientações sobre design e avaliação de tools.
- [Model Context Protocol - official docs & spec](https://modelcontextprotocol.io) - arquitetura, primitivas, transports e o changelog de cada revisão da spec.
- [A2A Protocol specification](https://a2a-protocol.org/latest/specification/) - Agent Cards, ciclo de vida das tasks, transports e o princípio de opacidade dos agents.
- [Agent Skills](https://agentskills.io/) - o formato SKILL.md, progressive disclosure e a vitrine de clients.
- [Code execution with MCP - Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp) - apresentar tools como uma API de código, em vez de definições de função carregadas de uma vez.
- [ReAct: Synergizing Reasoning and Acting in Language Models - Yao et al.](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning - Shinn et al.](https://arxiv.org/abs/2303.11366)
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains - Yao et al.](https://arxiv.org/abs/2406.12045) - fonte da métrica de confiabilidade pass^k.
- [The lethal trifecta for AI agents - Simon Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) - o modelo mental essencial de segurança para agents.
