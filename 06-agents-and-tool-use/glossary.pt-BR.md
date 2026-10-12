# 📚 Glossário de estudo — Agents & Tool Use

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. Agentes: ideias centrais](#1-agentes-ideias-centrais)
- [2. Desenho de ferramentas](#2-desenho-de-ferramentas)
- [3. MCP e A2A](#3-mcp-e-a2a)
- [4. Skills e padrões de planejamento](#4-skills-e-padrões-de-planejamento)
- [5. Memória e contexto](#5-memória-e-contexto)
- [6. Confiabilidade e guardrails](#6-confiabilidade-e-guardrails)
- [7. Segurança](#7-segurança)
- [8. Avaliação, observabilidade e custo](#8-avaliação-observabilidade-e-custo)

## 1. Agentes: ideias centrais

### Agent

Um sistema em que o LLM decide sozinho o que fazer a seguir: qual ferramenta chamar, em que ordem e quando terminar. Na prática é um modelo + ferramentas + contexto/memória rodando em um loop, com condições de parada.

> 💡 **Pense assim:** Um funcionário a quem você dá um objetivo e acesso a ferramentas, em vez de uma lista de passos.

*Aparece em:* Q1, Q2, Q3, Q7, Q14

### Workflow

Um fluxo em que o *seu código* define o caminho (qual chamada de LLM vem depois de qual), e o LLM só preenche cada etapa. Mais previsível, barato e fácil de avaliar do que um agent. Regra de ouro: se dá para desenhar o fluxograma, escreva um workflow.

> 💡 **Pense assim:** Uma linha de montagem com etapas fixas, em oposição a um funcionário que decide o que fazer.

*Aparece em:* Q1, Q2, Q46, Q47, Q63

### Agent loop

O ciclo central: montar o contexto, chamar o LLM, executar as ferramentas que ele pediu, anexar os resultados e repetir até o modelo responder só com texto (ou até uma condição de parada forçada).

*Aparece em:* Q13, Q22, Q30, Q47, Q53

### Stop conditions (max iterations, budget, timeout)

Limites que encerram o loop à força: número máximo de iterações, orçamento de tokens ou de dinheiro, tempo limite. Sem eles, um agent com erro pode girar para sempre e queimar dinheiro.

*Aparece em:* Q3, Q30, Q35, Q56, Q57

### Tool / function calling

Mecanismo em que o modelo, em vez de texto, devolve uma mensagem estruturada dizendo "quero chamar a ferramenta X com estes argumentos". Quem executa de verdade é o *seu código*, nunca o modelo.

*Aparece em:* Q4, Q7, Q14, Q51, Q62

### Tool definition

A descrição que você envia ao modelo a cada requisição: nome da ferramenta, descrição em texto e os parâmetros em JSON Schema. Ela consome tokens de entrada e funciona como um prompt.

*Aparece em:* Q7, Q15, Q31, Q33, Q56

### JSON Schema

Formato padrão para descrever a estrutura de um JSON: quais campos existem, seus tipos, quais são obrigatórios, valores permitidos. É assim que se declaram os parâmetros de uma ferramenta.

*Aparece em:* Q4, Q15

### Tool result

A resposta da execução da ferramenta, devolvida ao modelo como mensagem ligada ao ID da chamada. É o que o modelo "vê" no passo seguinte.

*Aparece em:* Q33, Q43, Q48, Q52, Q54

### Tool call ID

Identificador único de cada chamada que o modelo emite. O resultado volta com o mesmo ID, para o modelo saber qual resultado pertence a qual chamada.

*Aparece em:* Q4, Q6, Q14, Q54

### Parallel tool calls

O modelo pede várias chamadas independentes no mesmo turno (por exemplo, buscar três URLs). Você as executa em paralelo e devolve todos os resultados, poupando idas e vindas.

*Aparece em:* Q4, Q6, Q48

### Tool choice (auto, required, none)

Controle de quanto o modelo é obrigado a usar ferramentas: *auto* (ele decide), *required/any* (precisa chamar alguma), uma ferramenta específica (truque clássico para extração estruturada garantida) ou *none* (proibido chamar).

*Aparece em:* Q6, Q13, Q51

### Strict mode / constrained decoding

O provedor restringe a geração token a token para que os argumentos *sempre* sigam o JSON Schema. Garante a forma, não a verdade: um ID bem formatado ainda pode ser inventado, e o modelo pode ter escolhido a ferramenta errada. A validação semântica continua sendo sua.

*Aparece em:* Q4, Q15, Q29, Q51

### Reasoning model

Modelo que "pensa" por mais tokens antes de responder. Ajuda em planejamento e problemas difíceis dentro de um agent, mas é mais lento e caro, então não vale para passos simples como roteamento ou extração.

*Aparece em:* Q9, Q13, Q21, Q38

### Orchestration vs. agency

Autonomia é um dial, não uma chave liga/desliga. Cada passo rumo a mais autonomia compra flexibilidade, mas custa previsibilidade, latência, dinheiro e facilidade de avaliar.

*Aparece em:* Q1, Q22

## 2. Desenho de ferramentas

### Tool description as prompt

A descrição da ferramenta é lida pelo modelo para decidir quando e como usá-la. Escreva como documentação para um funcionário novo: o que faz, quando usar, quando *não* usar, o que significa cada argumento, um exemplo.

*Aparece em:* Q27, Q29, Q36, Q42, Q50

### Namespacing

Prefixar o nome das ferramentas pelo serviço (como `jira_create_issue`) para evitar confusão entre ferramentas parecidas.

*Aparece em:* Q10, Q31, Q55

### Consolidated vs granular tools

Consolidar significa criar uma ferramenta que resolve uma intenção completa (`schedule_meeting`) em vez de expor três chamadas de API encadeadas. Menos idas e vindas, menos chance de erro. Só mantenha granular onde o modelo realmente precisa decidir entre caminhos.

*Aparece em:* Q10, Q11, Q17, Q42, Q48

### Actionable errors

Devolver a falha como *resultado* da ferramenta, com instrução do que corrigir ("status inválido; valores aceitos: open, closed"), em vez de derrubar o loop ou jogar um stack trace. Transforma um beco sem saída em correção de curso.

*Aparece em:* Q16, Q30, Q42, Q50, Q51

### Token-efficient outputs

O agent paga por cada token de resultado pelo resto da trajetória. Por isso se pagina, filtra, devolve IDs semânticos, trunca com um ponteiro ou grava o resultado grande num arquivo e devolve só o caminho.

*Aparece em:* Q10, Q18, Q34, Q48, Q54

### Wrong-tool errors

Quando há ferramentas com funções sobrepostas, o modelo escolhe a errada. Esse erro cresce com o número de opções parecidas, e prompt caching não resolve (só reduz custo).

*Aparece em:* Q2, Q10, Q15, Q31, Q49

### Tool-call hallucination

O modelo inventa algo na chamada: uma ferramenta que não existe, um argumento que não está no schema, ou um ID plausível mas falso. Defende-se validando tudo no executor antes de agir.

*Aparece em:* Q4, Q6, Q10, Q15, Q51

### Progressive disclosure

Manter só um conjunto pequeno de definições sempre carregado e buscar o resto sob demanda. Aplica-se a ferramentas (via `search_tools`) e a skills (nome e descrição primeiro, instruções completas só quando a tarefa casa).

*Aparece em:* Q19, Q31, Q39, Q62

### Tool search

Uma ferramenta que busca, por consulta, as definições de outras ferramentas e as carrega só quando necessárias. Transforma a escolha de ferramenta em um problema de retrieval.

*Aparece em:* Q31

### Code execution / code API

Em vez de dezenas de ferramentas no prompt, expõe-se uma API tipada dentro de um sandbox. O agent lê só os módulos de que precisa e escreve um script que combina as chamadas, de modo que resultados intermediários grandes ficam no sandbox e só a saída final entra no contexto. Em troca, exige operar um sandbox de verdade.

*Aparece em:* Q17, Q26, Q45, Q62

## 3. MCP e A2A

### MCP (Model Context Protocol)

Padrão aberto para conectar aplicações de IA a ferramentas e dados. Resolve o problema N×M (cada app com um adaptador para cada integração): o servidor é feito uma vez e qualquer host MCP o usa.

> 💡 **Pense assim:** O USB-C das integrações de IA: um encaixe só para vários aparelhos.

*Aparece em:* Q7, Q11, Q40, Q60, Q64

### Host, client e server (MCP)

O *host* é a aplicação (um IDE, um app de chat). Ele roda um *client* por conexão, e cada client fala com um *server*, que expõe uma fonte de dados ou API.

*Aparece em:* Q8, Q11, Q39, Q55, Q64

### JSON-RPC

Formato de mensagem (chamadas de procedimento remoto em JSON) que o MCP usa para toda a comunicação entre client e server.

*Aparece em:* Q7, Q8, Q26, Q40

### Tools, resources e prompts (MCP)

As três primitivas que um servidor MCP oferece. *Tools*: ações que o modelo decide chamar. *Resources*: dados/contexto que a aplicação anexa. *Prompts*: modelos de instrução que o usuário invoca.

*Aparece em:* Q8, Q12, Q45, Q60, Q64

### stdio vs Streamable HTTP

Os dois transportes do MCP. *stdio*: o servidor roda como processo local na sua máquina (simples, mas herda seus privilégios). *Streamable HTTP*: servidor remoto, com autorização baseada em OAuth.

*Aparece em:* Q8, Q26, Q27, Q55, Q64

### Stateless MCP

Na revisão 2026-07-28, o MCP deixa de ter sessão: cada requisição leva sua própria versão de protocolo e capacidades. Assim qualquer instância de servidor atrás de um load balancer simples pode atendê-la, sem estado guardado na memória de uma instância.

*Aparece em:* Q8, Q26, Q37, Q57, Q65

### Elicitation

Recurso em que o servidor MCP pede uma informação ao usuário no meio de uma chamada (por exemplo, "qual classe de assento?"). Na versão sem sessão, vira uma chamada em várias idas e vindas: o servidor devolve "preciso de input" e o client repete a chamada com as respostas.

*Aparece em:* Q8, Q11, Q37, Q65

### Sampling (MCP)

Recurso em que o servidor pede ao host que rode uma completion de LLM por ele, sem precisar de chave de API própria. Pouco adotado (suporte irregular, o host paga por prompts que não escreveu) e marcado como deprecated na revisão 2026-07-28.

*Aparece em:* Q8, Q26, Q37, Q54, Q56

### Roots

Recurso do client MCP que delimita a quais partes do sistema de arquivos o servidor pode acessar. Também marcado como deprecated na revisão 2026-07-28.

*Aparece em:* Q8, Q26, Q27

### MCP Apps

Extensão do MCP que permite a um servidor entregar uma interface interativa (um template HTML) junto com a ferramenta. O host a exibe em um iframe isolado. Muda a segurança: agora o servidor envia código que roda dentro do seu host.

*Aparece em:* Q40

### Sandboxed iframe / CSP

Iframe é uma "janela" de página dentro de outra página. Com sandbox e CSP (Content Security Policy, regras sobre o que a página pode carregar e para onde pode enviar dados), ela fica isolada e não consegue acessar o resto do app nem enviar dados para qualquer lugar.

*Aparece em:* Q40

### Tool poisoning, rug pull e tool shadowing

Ataques via servidores MCP de terceiros. *Poisoning*: instruções maliciosas escondidas na descrição da ferramenta. *Rug pull*: a descrição é alterada depois de você aprová-la. *Shadowing*: uma ferramenta maliciosa imita ou interfere com outra confiável. Mitigação: auditar, fixar versões, menor privilégio.

*Aparece em:* Q27, Q39, Q40

### OAuth e PKCE

OAuth é o padrão de autorização em que o usuário concede acesso limitado sem entregar a senha. PKCE é uma proteção extra no fluxo de código de autorização, que impede que um código interceptado seja usado por outro. Servidores MCP remotos funcionam como resource servers OAuth.

*Aparece em:* Q7, Q26, Q27, Q61, Q64

### Token audience e token passthrough

*Audience* é para qual servidor o token foi emitido: o servidor deve checá-la sempre. *Token passthrough* (proibido) é repassar o token do usuário a outra API; o servidor deve obter um token próprio para cada upstream.

*Aparece em:* Q40, Q64

### Confused deputy

Um intermediário com mais poder do que o usuário (por exemplo, uma credencial única de serviço) é induzido a usá-lo em nome de alguém sem permissão. Em agents, uma injeção de prompt vira acesso a tudo que a credencial alcança. Cura: credenciais por usuário/tenant.

*Aparece em:* Q55, Q64

### A2A (Agent2Agent)

Protocolo para um agent falar com outro agent *par*, geralmente de outra equipe ou empresa. É o eixo horizontal; o MCP é o vertical (agent para ferramentas e dados). Os pares colaboram por capacidades declaradas, sem expor prompts ou ferramentas internas.

*Aparece em:* Q60

### Agent Card

Metadados JSON publicados em um endereço conhecido, declarando a identidade do agent, suas skills, endpoint e requisitos de autenticação. Pode ser assinado com JWS para o chamador verificar.

> 💡 **Pense assim:** O cartão de visitas de um agent.

*Aparece em:* Q60, Q61

### JWS, mTLS

*JWS* é uma assinatura digital em formato JSON, usada para provar que um documento não foi adulterado. *mTLS* é TLS em que os dois lados apresentam certificado, autenticando cliente e servidor.

*Aparece em:* Q60

### A2A task lifecycle

No A2A, o trabalho é uma *task* com estados (submitted, working, completed, failed, canceled, rejected) e estados de interrupção como input-required e auth-required. As atualizações chegam por polling, streaming ou webhook, porque a task de um par pode durar minutos.

*Aparece em:* Q8, Q37, Q60, Q65

### Webhook

O sistema que termina o trabalho chama uma URL sua para avisar, em vez de você ficar perguntando.

*Aparece em:* Q60

### MCP Task

Padrão para chamadas longas: o servidor aceita o trabalho e devolve imediatamente um identificador durável. O client consulta o status, busca o resultado depois ou cancela. O estado precisa ficar em armazenamento durável, não na memória de uma instância.

*Aparece em:* Q17, Q41, Q56, Q60, Q65

## 4. Skills e padrões de planejamento

### Agent Skills (SKILL.md)

Uma pasta com um arquivo `SKILL.md` (nome, descrição e instruções em Markdown) e, opcionalmente, scripts e referências. Dá ao agent *conhecimento procedural* sobre como fazer um trabalho. Ferramentas dão capacidade, RAG dá fatos, skills dão o método.

*Aparece em:* Q39, Q60, Q61

### ReAct

Padrão em que o modelo alterna Thought (raciocínio), Action (chamada de ferramenta) e Observation (resultado). Adaptativo, mas míope (só vê o próximo passo). Modelos modernos já internalizam boa parte disso.

*Aparece em:* Q9, Q21

### Plan-then-execute

O agent produz um plano completo antes e depois executa os passos, replanejando se algo falhar. Bom para tarefas longas e revisão humana do plano, mas frágil quando o ambiente surpreende.

*Aparece em:* Q9, Q21

### Reflection / self-critique

Gerar, criticar o próprio resultado e revisar. Funciona melhor quando a verificação é mais fácil que a geração e se apoia em sinais externos (testes, compilador). Modelos tendem a se avaliar com leniência, e o retorno cai rápido após 1 ou 2 rodadas.

*Aparece em:* Q20, Q22, Q50

### Orchestrator-worker / subagents

Um agent principal divide a tarefa e delega a subagents. O maior ganho é *isolamento de contexto*: o worker gasta 100k tokens pesquisando e devolve um resumo de 1k, mantendo o contexto do orquestrador limpo.

> 💡 **Pense assim:** Um chefe que delega pesquisas e recebe só o resumo.

*Aparece em:* Q23, Q25, Q31, Q35, Q48

### Handoff

Transferir o controle da conversa para um agent especialista (por exemplo, da triagem para o suporte de cobrança). Diferente de orquestração, em que o agent principal continua no comando e recebe os resultados.

*Aparece em:* Q17, Q24, Q25, Q53

### Fan-out / parallelisation

Disparar vários trabalhos independentes ao mesmo tempo. Multi-agent ajuda em tarefas de leitura paralelizáveis (pesquisa) e atrapalha em escrita com estado compartilhado, porque vários agents editando o mesmo código conflitam.

*Aparece em:* Q23, Q24, Q35

### Agent Workflow Memory / procedural memory

Memória de *como fazer* uma classe de tarefa: o agent destila trajetórias passadas em procedimentos reutilizáveis (gatilho, passos, verificações, armadilhas) e os recupera na hora de planejar. Candidatos devem ser validados antes de promovidos e versionados como código.

*Aparece em:* Q63

## 5. Memória e contexto

### Working memory vs persistent memory

*Working memory* é a janela de contexto da execução atual. *Persistent memory* são arquivos, bancos e vector stores que sobrevivem entre sessões e são recuperados quando necessário.

*Aparece em:* Q3, Q20, Q47, Q59

### Context rot

A qualidade do agent cai à medida que o contexto enche, mesmo antes do limite rígido. Contramedidas: compaction, limpar resultados de ferramentas antigos, anotações estruturadas e subagents.

*Aparece em:* Q17, Q42, Q52

### Compaction

Resumir os turnos antigos, mantendo os recentes e as decisões-chave, para liberar espaço na janela de contexto.

*Aparece em:* Q19, Q33, Q48, Q52, Q54

### Clearing stale tool results

Apagar do contexto resultados de ferramentas antigos que já foram usados, porque costumam ser grandes e deixaram de ser úteis.

*Aparece em:* Q19, Q33, Q48, Q52

### Structured note-taking / scratchpad

O agent mantém arquivos (lista de tarefas, decisões, progresso) fora do contexto. Eles sobrevivem à compaction e a reinícios: em tarefas longas, as notas, e não o histórico de tokens, são a memória real.

*Aparece em:* Q9, Q19, Q21, Q47, Q52

### Vector store vs rolling summarisation

Duas formas de lembrar entre sessões. *Vector store*: guarda trechos e busca por similaridade. *Rolling summarisation*: mantém um resumo que é reescrito a cada sessão. O vector store recupera detalhes pontuais; o resumo é compacto, mas perde detalhes.

*Aparece em:* Q20, Q32, Q55, Q63

### Episodic vs semantic memory

Dois tipos de memória. *Episódica*: o que aconteceu em eventos passados específicos. *Semântica*: fatos gerais e estáveis sobre o usuário ou o mundo.

*Aparece em:* Q20, Q59

### Memory consolidation

Rotina periódica que une duplicatas, resolve contradições e expira memórias velhas ou pouco usadas, para a memória não virar uma pilha que só cresce e atrapalha a recuperação.

*Aparece em:* Q13, Q17, Q20, Q58, Q63

### Memory poisoning

Conteúdo não confiável (como um resultado de ferramenta) é gravado na memória do agent e depois volta como se fosse conhecimento próprio, ganhando confiança indevida. Por isso a memória autogravada é um canal de persistência para injeção de prompt.

*Aparece em:* Q27, Q39, Q52, Q63

### Prompt caching

O provedor reaproveita o processamento de um prefixo de prompt idêntico entre chamadas, cobrando bem menos. Em agents o system prompt, as definições de ferramentas e o histórico formam um prefixo estável. Qualquer mudança no meio do prefixo invalida o cache dali em diante.

*Aparece em:* Q31, Q33, Q48, Q49, Q55

### Cache TTL

Tempo de vida de uma entrada (de cache ou de resultado armazenado): depois dele, o dado expira e é descartado.

*Aparece em:* Q33, Q65

## 6. Confiabilidade e guardrails

### Compounding errors

Probabilidade de sucesso se multiplica passo a passo: 95% por passo em 20 passos dá cerca de 36% de sucesso final. Por isso demos não são produtos. Ataca-se reduzindo passos, aumentando a confiabilidade por passo e adicionando recuperação (verificações, erros acionáveis, checkpoints).

*Aparece em:* Q2, Q16, Q42, Q46, Q62

### Guardrails

Limites de segurança no loop: máximo de iterações, orçamento de tokens/custo, timeouts, detecção de chamadas repetidas e níveis de permissão.

*Aparece em:* Q2, Q29, Q30, Q53

### Permission tiers / reversibility

Classificar ações por risco: leituras são aprovadas automaticamente, escritas passam por portão, ações irreversíveis (enviar, apagar, pagar) exigem aprovação humana. A reversibilidade determina quanta autonomia dar.

*Aparece em:* Q13, Q30, Q36, Q41, Q62

### Human-in-the-loop (HITL)

Pontos em que o agent para e pede aprovação ou ajuda de uma pessoa. O desafio é desenhar os portões para que o usuário não clique em "aprovar" no automático.

*Aparece em:* Q13, Q21, Q30, Q36, Q62

### Idempotency / idempotency key

Uma operação é idempotente quando repeti-la produz o mesmo efeito de fazê-la uma vez. Uma *idempotency key* estável por operação lógica faz a API devolver o resultado original em vez de, por exemplo, cobrar o cartão duas vezes numa retentativa.

> 💡 **Pense assim:** O botão do elevador: apertar cinco vezes não chama cinco elevadores.

*Aparece em:* Q5, Q14, Q47, Q60, Q65

### At-least-once delivery

Garantia de que uma operação será tentada uma ou mais vezes, nunca zero, mas possivelmente duplicada. Um agent se comporta assim, então toda ferramenta com efeito colateral precisa ser segura contra repetição.

*Aparece em:* Q5, Q14, Q47, Q62, Q65

### Side effect

Efeito de uma ferramenta que muda o mundo (enviar e-mail, gravar no banco, pagar), em contraste com leituras puras. Define o tratamento: leituras podem ser repetidas livremente, escritas não.

*Aparece em:* Q5, Q41, Q43, Q47, Q62

### Circuit breaker e backoff

Mecanismos contra falhas em cascata e loops. *Backoff*: esperar cada vez mais entre retentativas. *Circuit breaker*: parar de chamar um serviço que está falhando por um tempo. Detecção de chamadas duplicadas pega o agent repetindo a mesma ferramenta com os mesmos argumentos.

*Aparece em:* Q34, Q47, Q50, Q57, Q61

### Checkpointing / durable execution

Persistir o estado da execução (histórico, plano, artefatos) a cada fronteira do loop, de forma que após uma queda ou deploy o agent retome do ponto em que parou. Motores de workflow como o Temporal oferecem essa semântica pronta.

*Aparece em:* Q14, Q36, Q47, Q53, Q65

### Heartbeat / stall alert

Sinal periódico de que a execução longa continua viva e progredindo. A ausência dele (ou de progresso) dispara um alerta de travamento.

*Aparece em:* Q27, Q39, Q45, Q47, Q60

### Cancellation

Interromper o agent no meio de uma ação (o botão Stop). A ferramenta em andamento é tratada conforme seu efeito colateral, e o histórico precisa de um resultado sintético para cada chamada aberta, senão a próxima requisição quebra.

*Aparece em:* Q41, Q60, Q65

### Streaming / partial JSON

Argumentos de ferramentas chegam aos poucos, como JSON incompleto. Pode-se exibir "buscando..." na tela, mas nunca executar antes de o bloco terminar e ser validado.

*Aparece em:* Q8, Q41, Q60

## 7. Segurança

### Prompt injection

Texto malicioso dentro de dados que o agent lê (página web, e-mail, resultado de ferramenta) que o modelo trata como instrução. Modelos não separam com confiabilidade dados de instruções, então não existe correção apenas no prompt.

*Aparece em:* Q43, Q44, Q45, Q61, Q64

### Untrusted content

Qualquer texto que o atacante pode influenciar: páginas, documentos, resultados de ferramentas, saídas de outro agent, descrições de servidores MCP de terceiros. Deve ser tratado como dados, nunca como ordem.

*Aparece em:* Q43, Q44, Q45, Q61, Q63

### Lethal trifecta

Expressão de Simon Willison para a combinação perigosa de três coisas num mesmo agent: acesso a dados privados, exposição a conteúdo não confiável e um canal para enviar dados para fora. O teste de design é nunca juntar as três.

*Aparece em:* Q27, Q43, Q44, Q45

### Exfiltration / egress

*Exfiltração* é tirar dados de dentro para fora. *Egress* é o tráfego de saída da rede. Bloquear por padrão o egress e liberar só destinos necessários fecha o canal que o atacante precisaria.

*Aparece em:* Q27, Q40, Q43, Q44, Q45

### Least privilege / allowlist

Dar a cada agent e ferramenta apenas o acesso estritamente necessário. Allowlist é a lista do que *é* permitido (tudo o mais é negado).

*Aparece em:* Q27, Q30, Q43, Q45, Q55

### Defense in depth

Combinar várias camadas de proteção (menor privilégio, controle de egress, sandbox, portões de aprovação) em vez de confiar em uma só, porque cada uma pode falhar.

*Aparece em:* Q55

### Sandbox

Ambiente isolado onde o código gerado pelo modelo roda, tratado como não confiável. Camadas: isolamento (container, gVisor, microVM), diretórios limitados, rede com egress negado por padrão, limites de recursos e nenhuma credencial ambiente.

> 💡 **Pense assim:** Uma caixa de areia: pode bagunçar à vontade, mas não sai de lá.

*Aparece em:* Q40, Q43, Q45, Q55, Q62

### gVisor e Firecracker

Tecnologias de isolamento mais fortes que containers comuns. *gVisor* intercepta as chamadas de sistema do código. *Firecracker* cria microVMs, pequenas máquinas virtuais leves. São comuns em serviços multi-tenant de execução de código.

*Aparece em:* Q45

### Computer-use / browser agents

Agents que operam interfaces gráficas vendo screenshots e simulando mouse e teclado. São mais frágeis que agents de API (percepção imprecisa, ações sem validação, cadeias longas). Quando possível, prefira API ou DOM/accessibility tree e reserve pixels para o resto.

*Aparece em:* Q46

### Multi-tenant isolation

Um tenant é um cliente da plataforma. Em agents há canais extras de vazamento: contexto compartilhado, caches compartilhados (prompt cache e cache de respostas) e execução compartilhada. Isola-se com credenciais por tenant, cache e vector store particionados, sandbox por tenant e cotas.

*Aparece em:* Q38, Q45, Q55, Q56, Q61

### Noisy neighbour

Um tenant que consome recursos demais e prejudica os outros. Resolve-se com cotas de tokens e limites de concorrência por tenant.

*Aparece em:* Q55

### Denial of wallet

Ataque ou acidente que não derruba o serviço, mas faz o custo explodir (loops, delegações em cadeia sem limite). Em agents, a conta é o risco real. Combate-se com orçamentos, cotas e limites de profundidade.

*Aparece em:* Q61

### PII e redação

PII é informação pessoal identificável (nome, CPF, e-mail etc.). Redigir é remover ou mascarar esses dados, por exemplo nos traces, que carregam argumentos e raciocínios e podem vazar muito.

*Aparece em:* Q49, Q55, Q62

## 8. Avaliação, observabilidade e custo

### Trajectory vs final-outcome evals

*Final-outcome*: o estado final ficou correto? É robusto. *Trajectory*: o caminho foi bom (ferramentas certas, eficiente, seguiu as políticas)? Útil para diagnóstico, geralmente julgada por um LLM contra uma rubrica.

*Aparece em:* Q28, Q42, Q48, Q58, Q63

### pass@k vs pass^k

*pass@k*: probabilidade de pelo menos uma entre k tentativas dar certo (mede capacidade). *pass^k*: probabilidade de *todas* as k darem certo (mede consistência, o que o usuário sente). Agents em produção são julgados por pass^k.

*Aparece em:* Q29, Q42, Q48, Q56, Q58

### τ-bench, SWE-bench, WebArena, OSWorld, GAIA

Benchmarks de agents. SWE-bench: corrigir issues reais de software. WebArena: tarefas em sites. OSWorld: uso de computador. τ-bench: conversas com ferramentas e políticas de negócio, popularizou o pass^k.

*Aparece em:* Q28, Q29, Q46, Q58, Q63

### LLM-as-judge

Usar um LLM para avaliar as saídas de outro, normalmente contra uma rubrica. Barato e escalável, mas ruidoso: precisa ser calibrado com rótulos humanos.

*Aparece em:* Q28, Q57

### Rubric

Lista de critérios explícitos (com o que vale nota boa ou ruim) que o avaliador, humano ou LLM, usa para pontuar de forma consistente.

*Aparece em:* Q22, Q28, Q44

### User simulator

Um LLM que faz o papel do usuário (mudando de ideia, errando, ambíguo) para testar o agent em conversas multi-turno que casos de teste estáticos não cobrem.

*Aparece em:* Q58

### Regression test / golden set

Conjunto fixo de casos que roda a cada mudança para garantir que o que já funcionava continue funcionando.

*Aparece em:* Q28, Q38, Q54, Q56, Q59

### Canary / A/B / shadow

Formas de lançar uma mudança (por exemplo, um novo system prompt) com segurança. *Canary*: para uma fatia pequena do tráfego. *A/B*: duas versões comparadas lado a lado. *Shadow*: a versão nova roda em paralelo sem afetar o usuário. *Rollback*: voltar à versão anterior.

*Aparece em:* Q46, Q50, Q56, Q59, Q63

### Deterministic vs counterfactual replay

Dois modos de reproduzir uma execução a partir de um log de eventos completo. *Determinístico*: reencena as respostas gravadas do modelo e das ferramentas, sem chamar o modelo (testa o seu código). *Contrafactual*: modelo real com resultados de ferramentas fixos (testa se um novo prompt resolve; rode k vezes).

*Aparece em:* Q17, Q34, Q48, Q54, Q62

### Event log

Registro append-only de tudo que aconteceu em uma execução (requisição exata, resposta bruta, resultados das ferramentas, versões). É a fonte de verdade da qual se derivam os traces e o replay.

*Aparece em:* Q33, Q54, Q63

### Trace / span

Um *trace* é o registro completo de uma execução; cada chamada de LLM ou de ferramenta é um *span* (com argumentos, resultado, tokens e latência). Você não consegue depurar um agent que não rastreou.

*Aparece em:* Q38, Q49, Q54, Q56, Q62

### SLO / SLI

*SLI* é a métrica medida (como a taxa de sucesso de tarefas). *SLO* é a meta para ela. Em agents, falhas são silenciosas (tudo "200 OK"), então se mede o sucesso semântico e indicadores antecedentes: runs que estouram o limite de iterações, passos por tarefa, taxa de erro por ferramenta, intervenção humana.

*Aparece em:* Q57

### Cost per successful task

Custo total dividido pelas tarefas que realmente deram certo (não por chamada). Incentiva medir o que importa: um agent barato por chamada que falha muito sai caro.

*Aparece em:* Q41, Q48, Q56, Q57, Q63

### Model tiering

Usar modelos de tamanhos diferentes por passo: o grande planeja, o pequeno extrai ou roteia. Corta custo e latência sem destruir a qualidade.

*Aparece em:* Q21, Q48

### Agent frameworks (LangGraph, Agents SDK, ADK)

Bibliotecas que dão o esqueleto do loop, estado, checkpointing, tracing e handoffs. Custo: abstração opaca e menos controle dos prompts. Postura forte: construa o loop cru primeiro (é pequeno) e adote um framework quando precisar da infraestrutura dele.

*Aparece em:* Q25, Q47, Q53
