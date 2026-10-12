# 📚 Glossário de estudo — Safety, Security & Responsible AI

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. A raiz do problema: injection e jailbreak](#1-a-raiz-do-problema-injection-e-jailbreak)
- [2. Técnicas de jailbreak](#2-técnicas-de-jailbreak)
- [3. Agents, tools e autoridade](#3-agents-tools-e-autoridade)
- [4. Arquiteturas de defesa](#4-arquiteturas-de-defesa)
- [5. Guardrails e tratamento de saída](#5-guardrails-e-tratamento-de-saída)
- [6. Dados, privacidade e multi-tenancy](#6-dados-privacidade-e-multi-tenancy)
- [7. Supply chain e artefatos de modelo](#7-supply-chain-e-artefatos-de-modelo)
- [8. Alinhamento, hallucination e comportamento](#8-alinhamento-hallucination-e-comportamento)
- [9. Red teaming e avaliação de segurança](#9-red-teaming-e-avaliação-de-segurança)
- [10. Governança, regulação e responsible AI](#10-governança-regulação-e-responsible-ai)
- [11. Operação, incidentes e provenance de conteúdo](#11-operação-incidentes-e-provenance-de-conteúdo)

## 1. A raiz do problema: injection e jailbreak

### Context window sem separação de privilégios

Dentro do contexto do modelo, system prompt, mensagem do usuário, documentos recuperados e resultados de tools são todos apenas tokens numa sequência só. Seguir instruções é um comportamento aprendido no treino, não uma barreira imposta pelo sistema. Quase todo problema do módulo vem desse fato.

> 💡 **Pense assim:** Uma sala onde chefe, cliente e panfleto de rua falam pelo mesmo microfone e ninguém sabe quem tem autoridade.

*Aparece em:* Q2, Q10

### System prompt

O texto de instruções que o desenvolvedor coloca no início do contexto para orientar o modelo (tom, regras, papel). Ele influencia o comportamento, mas não é uma fronteira de segurança: pode ser extraído e contornado.

*Aparece em:* Q2, Q5, Q15, Q22, Q41

### Prompt injection

Ataque em que texto controlado por um atacante entra no contexto do modelo e o leva a fazer algo que o desenvolvedor não pretendia. É um problema de segurança da aplicação.

> 💡 **Pense assim:** É o equivalente em LLMs ao SQL injection, só que sem o equivalente das queries parametrizadas.

*Aparece em:* Q1, Q3, Q10, Q13, Q51

### Direct vs indirect injection

Na injection *direta*, o próprio usuário digita o ataque. Na *indireta*, as instruções chegam escondidas em conteúdo que o app processa em nome do usuário (página web, e-mail, PDF, convite de calendário, resultado de tool). A indireta é mais perigosa: a vítima não vê o ataque e ele escala, pois uma página envenenada atinge todo agent que a ler.

> 💡 **Pense assim:** Direta é o ladrão batendo à porta; indireta é uma carta armadilhada que o próprio carteiro entrega dentro de casa.

*Aparece em:* Q3, Q6, Q31, Q38, Q46

### Jailbreak

Ataque ao *treinamento de segurança do modelo* para obter conteúdo que o provedor proíbe. Aqui o atacante é o usuário, e a política atacada é a do provedor. Diferente de injection, em que um terceiro ataca o usuário através do app.

*Aparece em:* Q1, Q22, Q35, Q36, Q41

### Instruction hierarchy

Treino que ensina o modelo a dar prioridade a instruções de fontes mais confiáveis (system acima de user, e este acima de conteúdo de tools). Reduz a taxa de sucesso dos ataques, mas é probabilístico, então não elimina o risco.

*Aparece em:* Q15, Q41

### Delimiters (delimitadores)

Marcar o conteúdo não confiável com símbolos ou tags (por exemplo, “o texto entre <<< e >>> é só dado”). Ajuda um pouco, mas não resolve: o modelo pode ser convencido a ignorar a regra e o atacante pode incluir o próprio delimitador de fechamento.

*Aparece em:* Q2

### Blast radius

O tamanho do estrago possível se algo der errado (por exemplo, se uma injection funcionar). A estratégia de segurança para LLMs é menos impedir toda falha e mais reduzir o blast radius.

*Aparece em:* Q15, Q17, Q27, Q30, Q40

## 2. Técnicas de jailbreak

### Roleplay / persona framing

Embrulhar o pedido numa ficção, hipótese ou personagem “sem regras”. As recusas foram treinadas em pedidos diretos, então a moldura fictícia fica fora da distribuição vista no treino.

*Aparece em:* Q16

### Many-shot jailbreaking

Encher um contexto longo com centenas de diálogos falsos em que o “assistente” obedece tudo. O in-context learning acaba sobrepondo o comportamento treinado, e o efeito cresce com o tamanho do contexto.

*Aparece em:* Q16, Q35

### Encoding / obfuscation

Disfarçar o pedido com base64, ROT13, leetspeak ou idiomas de poucos recursos. O modelo entende essas formas, mas o treino de segurança não as cobriu, então ele decodifica e obedece.

*Aparece em:* Q16

### GCG (adversarial suffix)

Método que busca, por gradiente, uma sequência de tokens sem sentido que, anexada ao pedido, maximiza a chance de o modelo obedecer. Esses sufixos transferem entre modelos, o que mostra que a fragilidade é estrutural.

*Aparece em:* Q16, Q35, Q36

### Multi-turn escalation (crescendo)

Começar com perguntas inocentes e subir o tom aos poucos ao longo da conversa, de modo que nenhuma mensagem isolada dispare a recusa.

*Aparece em:* Q16

### Best-of-N

Gerar muitas variações superficiais do mesmo pedido (maiúsculas, espaços, ordem) e tentar uma por uma até alguma passar. Explora a fragilidade do modelo à forma do texto.

*Aparece em:* Q16

### Safety training distribucional

O treino de segurança ensina recusas em uma distribuição de prompts nocivos conhecidos, enquanto as capacidades do modelo generalizam muito além disso. Atacantes se movem para fora dessa distribuição, onde a recusa não foi reforçada mas a capacidade continua ativa.

*Aparece em:* Q16, Q49

## 3. Agents, tools e autoridade

### Agent / tool call

Agent é um LLM que, além de gerar texto, chama ferramentas (enviar e-mail, consultar banco, navegar na web). Cada tool call dá ao modelo um poder real, e por isso o risco de segurança deixa de ser só “conteúdo ruim” e passa a ser “ação ruim”.

*Aparece em:* Q3, Q27, Q36, Q43, Q52

### Lethal trifecta

Teste de design de Simon Willison: um agent que reúne (1) acesso a dados privados, (2) exposição a conteúdo não confiável e (3) um canal de exfiltração é explorável. A correção é arquitetural: remover uma das três pernas.

> 💡 **Pense assim:** Fogo, combustível e oxigênio: tirando qualquer um, não há incêndio.

*Aparece em:* Q4, Q6, Q38, Q44, Q51

### Exfiltration channel

Qualquer caminho pelo qual dados podem sair do seu controle: enviar e-mail, requisição HTTP, imagem em markdown com dados na URL, postar numa API. É a terceira perna da lethal trifecta.

*Aparece em:* Q6, Q34, Q38, Q40, Q51

### Markdown image exfiltration

O atacante faz o modelo emitir uma imagem em markdown cuja URL leva o segredo na query string. Quando o cliente renderiza, o navegador faz um GET automático para o servidor do atacante, sem nenhum clique. Defesa: allowlist de domínios de imagem, CSP e sanitização da saída.

*Aparece em:* Q6, Q7, Q30, Q38

### Excessive agency (OWASP LLM06)

O agent pode causar mais dano do que sua tarefa exige. Tem três subtipos: *funcionalidade* em excesso (tools que ele não precisa), *permissões* em excesso (credencial ampla demais) e *autonomia* em excesso (ações irreversíveis sem confirmação).

*Aparece em:* Q4, Q9

### Least privilege

Dar a cada tool e credencial apenas o mínimo necessário para a tarefa: read-only por padrão, escopo estreito, tokens de curta duração.

*Aparece em:* Q15, Q20, Q34, Q41, Q44

### Confused deputy

Programa com privilégios que é enganado por alguém com menos privilégios a usar sua autoridade em favor dele. Um agent é um caso quase perfeito: carrega a autoridade do usuário e aceita instruções em linguagem natural de qualquer texto que entre no contexto.

> 💡 **Pense assim:** Um porteiro com a chave do prédio que abre a porta para qualquer um que lhe entregue um bilhete convincente.

*Aparece em:* Q10, Q30, Q51

### Service account vs on-behalf-of token

Uma *service account* compartilhada tem a união das permissões de todos os usuários e deixa o log sem saber quem fez o quê. Um token *on-behalf-of* (delegado) é derivado da sessão do usuário e limitado ao que ele já podia fazer, o que mantém ACLs e auditoria significativas.

*Aparece em:* Q27, Q28, Q33, Q42, Q44

### Human-in-the-loop (approval gate)

Exigir aprovação humana antes de ações consequenciais ou irreversíveis (enviar dinheiro, apagar, publicar). Para funcionar, deve mostrar o *efeito calculado pelo servidor*, não a descrição escrita pelo modelo.

*Aparece em:* Q15, Q38, Q40, Q41, Q50

### Rubber-stamping

Quando o humano aprova tudo automaticamente por excesso de pedidos de confirmação, e a aprovação vira teatro. Uma métrica prática é a latência de aprovação: respostas em menos de ~2 segundos indicam carimbo sem leitura.

*Aparece em:* Q27, Q33, Q43

### Tool poisoning, rug pull e shadowing

Riscos de conectar tools de terceiros (como MCP servers). *Tool poisoning*: a descrição da tool carrega instruções maliciosas. *Rug pull*: o servidor muda a descrição ou o comportamento depois de você aprová-lo. *Shadowing*: uma tool maliciosa manipula como o modelo usa outra tool confiável.

*Aparece em:* Q20

### MCP (Model Context Protocol)

Padrão para conectar modelos a tools e fontes de dados externas. Facilita misturar servidores de terceiros, e por isso facilita montar a lethal trifecta sem querer.

*Aparece em:* Q6, Q20, Q28, Q30, Q33

### Memory poisoning

Injection que *persiste*: o atacante faz o agent gravar na memória de longo prazo uma instrução disfarçada de preferência ou fato, e ela é recuperada dias depois, em outra sessão, já com cara de conteúdo confiável.

*Aparece em:* Q4, Q13, Q33

### Provenance e taint tracking

*Provenance* é a etiqueta de origem de cada dado (fonte, data, nível de confiança). *Taint tracking* é propagar a marca “contaminado” por todos os passos: dado derivado de conteúdo não confiável continua marcado e uma política pode impedir que ele chegue a um sink perigoso.

*Aparece em:* Q13, Q21, Q25, Q31, Q44

### Browser agent e Same-Origin Policy

Agent que age dentro do navegador logado do usuário. A Same-Origin Policy impede um site de ler dados de outro, mas o agent lê a página A e age na B, virando justamente a ponte entre origens que a política existia para impedir.

*Aparece em:* Q51

### Slopsquatting

Ataque de supply chain em que o modelo inventa o nome de um pacote que não existe e o atacante registra esse nome no PyPI ou npm com código malicioso. É typosquatting sem erro de digitação humano.

*Aparece em:* Q14

## 4. Arquiteturas de defesa

### Defense in depth

Empilhar várias camadas de proteção (entrada, modelo, privilégios, saída, humano, monitoramento), sem confiar em nenhuma sozinha, para reduzir o estrago se uma falhar.

*Aparece em:* Q15, Q22, Q38

### Design for failure (assume injection succeeds)

Projetar partindo da premissa de que a injection vai funcionar em algum momento, de modo que o sistema limite o que ela consegue fazer. Resistência “por construção” não depende de o modelo se recusar a ser enganado.

*Aparece em:* Q21, Q34, Q51

### Dual-LLM pattern

Um LLM *privilegiado* planeja e chama tools mas nunca lê conteúdo não confiável. Um LLM *em quarentena* lê esse conteúdo, sem tools, e devolve apenas valores tipados (como `$email_summary`) que o lado privilegiado manipula de forma simbólica.

*Aparece em:* Q15, Q31, Q34, Q44

### CaMeL

Evolução do dual-LLM (Google DeepMind, 2025): o planner escreve um plano em código só a partir do pedido confiável, um interpretador customizado o executa e políticas de capability rastreiam quais dados podem fluir para quais sinks. O preço é perder flexibilidade, pois o plano não se adapta ao que os dados dizem.

*Aparece em:* Q34

### Structural constraints / schema validation

Obrigar a saída do modelo a um formato fixo (por exemplo, um enum de 5 categorias ou argumentos tipados). Como a validação é determinística, é um dos guardrails de maior alavancagem: uma injection não consegue exfiltrar um segredo por um campo que só aceita 5 valores.

*Aparece em:* Q15, Q17, Q41, Q52

### Sandbox

Ambiente isolado para executar código gerado pelo modelo: container endurecido (gVisor), microVM (Firecracker) ou WASM, com filesystem descartável, limites de recursos, sem rede e sem credenciais. Parte da premissa de que o código é hostil.

*Aparece em:* Q7, Q14, Q20, Q40, Q41

### Egress control (allowlist de saída)

Bloquear por padrão toda conexão de saída e permitir só hosts específicos. Em sandboxes, é considerado o controle de maior alavancagem, porque corta exfiltração e download de payloads.

*Aparece em:* Q30, Q35, Q40, Q43, Q50

### Content-Security-Policy (CSP)

Cabeçalho que diz ao navegador de quais origens ele pode carregar recursos (por exemplo, `img-src`). Usado para impedir que um markdown injetado carregue imagens de um domínio do atacante.

*Aparece em:* Q7, Q38

### Kill switch vs circuit breaker

*Kill switch* é manual e global: um humano percebe e para o agent. *Circuit breaker* é automático e local: um limite (passos, custo, falhas seguidas, chamadas repetidas) dispara e para o agent sozinho em milissegundos. Precisa dos dois, e o kill switch deve estar fora do loop do agent e revogar credenciais.

> 💡 **Pense assim:** O kill switch é o botão de emergência do fabricante; o circuit breaker é o disjuntor que desarma sozinho.

*Aparece em:* Q11, Q43

### Denial of wallet

Ataque econômico: em vez de derrubar o sistema, o atacante o torna caríssimo, com prompts curtos que disparam gerações ou loops longos. Defesas: autenticar antes de gastar, quotas em tokens, teto de passos e custo, limites de gasto no provedor e alertas de anomalia.

*Aparece em:* Q4, Q12, Q44, Q48

### Threat modeling (STRIDE e MITRE ATLAS)

Mapear o que pode dar errado antes de lançar. *STRIDE* é uma lista clássica (spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege). *MITRE ATLAS* é a base de conhecimento de ataques específicos a sistemas de IA.

*Aparece em:* Q1, Q30, Q35, Q40

### Multi-agent: injection laundering e privilege aggregation

Em sistemas multiagente, *injection laundering* é quando conteúdo envenenado passa por um agent e ganha confiança falsa só por ter vindo de um agent interno. *Privilege aggregation* é a permissão efetiva de um ponto de entrada virar a união de todas as tools da frota.

*Aparece em:* Q7, Q13, Q44

## 5. Guardrails e tratamento de saída

### Guardrails

Verificações automáticas ao redor do modelo, na entrada e na saída (heurísticas, classificadores, moderação, validação de formato). Reduzem risco, mas são atacáveis e têm falsos positivos.

*Aparece em:* Q17, Q35, Q41, Q43, Q50

### Moderation e Llama Guard

*Moderation* classifica texto contra categorias de conteúdo proibido (existe um endpoint gratuito da OpenAI). *Llama Guard* é um modelo de segurança que classifica entradas e saídas segundo uma taxonomia de riscos.

*Aparece em:* Q15, Q17, Q32

### False positive budget

Quantidade de pedidos legítimos bloqueados que o produto tolera. Todo classificador erra, e cada bloqueio indevido custa em qualidade do produto, então os limiares são ajustados por superfície e se prefere resposta suave a bloqueio duro quando possível.

*Aparece em:* Q17, Q49

### Improper output handling (OWASP LLM05)

Confiar na saída do modelo em etapas seguintes: renderizar como HTML (XSS), executar SQL gerado (SQL injection), rodar código sem sandbox (RCE). Como o modelo lê entradas controladas por atacantes, sua saída é transitivamente controlada por eles.

*Aparece em:* Q4, Q7, Q33

### XSS, SQL injection, RCE, SSRF

Vulnerabilidades clássicas de AppSec. *XSS*: script injetado roda no navegador. *SQL injection*: texto vira parte de uma query. *RCE*: execução remota de código. *SSRF*: o servidor é induzido a fazer requisições a destinos escolhidos pelo atacante. Com LLMs, o modelo pode ser a etapa que produz o payload.

*Aparece em:* Q2, Q4, Q7

### Parameterised queries

Enviar a estrutura da query e os dados por canais separados, de modo que o dado nunca seja interpretado como código. É a solução do SQL injection que não tem equivalente em LLMs.

*Aparece em:* Q2, Q7

### Presidio e PII detection

Presidio é uma ferramenta da Microsoft para detectar dados pessoais (nomes, telefones, documentos) em texto, usada para redigir ou mascarar antes do modelo e dos logs.

*Aparece em:* Q17, Q18

## 6. Dados, privacidade e multi-tenancy

### PII

Informação pessoalmente identificável: nome, telefone, e-mail, documento, endereço etc. Deve ser mantida fora de todo lugar onde não é estritamente necessária.

*Aparece em:* Q8, Q17, Q18, Q19, Q43

### PII redaction e pseudonymisation

Trocar entidades por marcadores tipados antes de enviar ao modelo (por exemplo, `[PERSON_1]`), guardando um mapa reversível na camada confiável para reinserir os valores reais na resposta. O modelo raciocina sem ver o dado real, e se o prompt vazar não há nada sensível nele.

*Aparece em:* Q18, Q37, Q43, Q47, Q52

### Data leakage

Informação sensível escapando de onde deveria ficar. Os canais são mais amplos que o modelo: prompts enviados ao vendor, traces de observabilidade, logs, datasets de fine-tuning e eval, caches, embeddings e o system prompt.

*Aparece em:* Q4, Q8, Q37, Q39, Q50

### Observability / tracing

Ferramentas (como LangSmith, Langfuse) que guardam prompts e respostas completos para depuração. Costumam ser o maior depósito acidental de PII em produção, e precisam de redação e retenção curta.

*Aparece em:* Q8, Q18, Q42, Q43, Q52

### Zero-data-retention (ZDR)

Acordo com o provedor para que ele não guarde seus dados nem pelo período de monitoramento de abuso (da ordem de ~30 dias). Costuma ser contrato, não uma caixa a marcar.

*Aparece em:* Q8, Q18, Q32

### DPA e BAA

Contratos de proteção de dados com o fornecedor. *DPA* (Data Processing Agreement) regula o tratamento de dados pessoais. *BAA* (Business Associate Agreement) é exigido para dados de saúde nos EUA.

*Aparece em:* Q18, Q32

### VPC / private deployment

Rodar a inferência dentro da sua própria conta de nuvem (Bedrock, Azure OpenAI, Vertex), sob seu DPA e controles de rede, em vez de enviar dados para uma API pública. É o meio-termo comum em empresas reguladas.

*Aparece em:* Q18, Q32, Q39

### Memorisation e extraction attack

Modelos decoram alguns exemplos de treino, principalmente dados duplicados ou de alta entropia (chaves, telefones). Ataques de extração recuperam esses trechos, como o ataque de “divergência” que mandou um modelo repetir uma palavra indefinidamente e fez vazar dados de treino.

*Aparece em:* Q19, Q29, Q36, Q39, Q41

### Membership inference

Ataque que tenta descobrir se um registro específico estava no conjunto de treino. Isso por si só pode ser uma violação de privacidade em dados sensíveis, como um corpus médico.

*Aparece em:* Q19

### Differential privacy (DP-SGD)

Técnica de treino que adiciona ruído controlado para limitar o quanto qualquer exemplo individual influencia o modelo. Protege contra memorização ao custo de utilidade e computação.

*Aparece em:* Q19

### Right to erasure (GDPR) e unlearning

Direito de pedir a exclusão dos próprios dados. Em RAG basta apagar o documento e reindexar, mas dados já treinados estão diluídos nos pesos. *Unlearning* aproximado tenta remover a influência mas é difícil de verificar. *SISA* treina em fatias para retreinar só uma.

> 💡 **Pense assim:** É mais parecido com desassar um bolo do que com apagar uma linha de planilha.

*Aparece em:* Q18, Q29

### Per-tenant isolation

Garantir que os dados do cliente A nunca apareçam para o cliente B em índices, caches, memória e tools. Falhas comuns: filtro de tenant esquecido, cache compartilhado e credencial ampla demais.

*Aparece em:* Q4, Q39, Q41, Q42, Q43

### Semantic cache

Cache que reaproveita respostas para perguntas parecidas (não idênticas). Se não for separado por tenant, devolve a resposta do cliente A ao cliente B, um vazamento silencioso clássico.

*Aparece em:* Q39, Q42

### ACL filter no query time

ACL é a lista de quem pode ver o quê. A checagem deve ser feita *na consulta*, dentro do datastore, com a identidade de quem pergunta. Aplicar permissões só na ingestão deixa o índice desatualizado, e filtrar depois da busca (post-filter) é um bug, pois o k já está errado.

*Aparece em:* Q18, Q26, Q28, Q39, Q42

### Embedding inversion

Reconstruir o texto original a partir de seus embeddings. Por isso um vector store deve ter o mesmo nível de sensibilidade dos documentos que o originaram: um embedding de prontuário é um prontuário.

*Aparece em:* Q4, Q26

### Index poisoning (RAG)

Quem consegue escrever no corpus consegue escrever no contexto do modelo. O atacante cria um trecho feito para ser o melhor resultado de uma consulta específica e coloca instruções nele. Defesa: escrita autenticada, provenance, namespaces por nível de confiança.

*Aparece em:* Q4, Q25, Q26, Q46

## 7. Supply chain e artefatos de modelo

### Supply chain (OWASP LLM03)

Risco de componentes de terceiros comprometidos: pesos, datasets, pacotes, adapters, tools. Em IA há dois tipos novos de artefato (pesos e datasets) que podem esconder payloads que inspeção não revela.

*Aparece em:* Q4, Q21, Q32, Q33, Q40

### Pickle vs safetensors

Checkpoints `.bin`/`.pt` do PyTorch são arquivos *pickle*, e carregá-los pode executar código arbitrário. *safetensors* é um formato só de dados (tensores e metadados), seguro por construção. Regra: prefira safetensors e nunca carregue pickle de fonte não confiável.

*Aparece em:* Q21, Q25, Q32

### Backdoor

Modelo treinado para agir normalmente, exceto quando aparece uma frase-gatilho que ativa um comportamento escolhido pelo atacante. Não dá para eliminar por benchmark, pois o gatilho não está no seu eval.

*Aparece em:* Q4, Q21, Q25

### Data & model poisoning (LLM04)

Corromper dados (pré-treino, fine-tuning, preferências ou índice RAG) para o modelo aprender o que o atacante quer. Detecção vira questão de provenance e processo, não de teste.

*Aparece em:* Q4, Q13, Q21, Q25, Q44

### Pinning, hashes e SBOM

Fixar versões exatas e verificar hashes de dependências e pesos (lockfiles). *SBOM* é a lista de materiais de software, o inventário de componentes, para rastrear vulnerabilidades.

*Aparece em:* Q14, Q20, Q21, Q32, Q46

## 8. Alinhamento, hallucination e comportamento

### Alignment

Treino que molda o comportamento do modelo sobre as capacidades do pré-treino, para que ele seja útil, inofensivo e honesto. A recusa de pedidos nocivos vive nos pesos, não no system prompt.

*Aparece em:* Q22, Q25, Q32, Q41, Q50

### RLHF, DPO e RLAIF

*RLHF*: treina um reward model com comparações humanas e otimiza o modelo contra ele. *DPO*: alternativa mais simples que otimiza direto em pares de preferência, sem reward model. *RLAIF*: as preferências vêm de uma IA em vez de humanos.

*Aparece em:* Q16, Q22, Q41

### Constitutional AI

Abordagem da Anthropic em que o modelo critica e revisa as próprias respostas com base em um conjunto escrito de princípios (a “constituição”), e rótulos de preferência gerados por IA substituem boa parte dos humanos.

*Aparece em:* Q22, Q41

### Over-refusal

Quando o modelo recusa ou faz sermão em pedidos legítimos só porque parecem perigosos (por exemplo, “como matar um processo Python”). Um modelo que recusa tudo é trivialmente “seguro” e inútil, então segurança deve ser medida junto com over-refusal.

*Aparece em:* Q23, Q36, Q50

### Hallucination

Quando o modelo produz uma resposta fluente e confiante, mas inventada. Vira questão de segurança quando causa dano real, como citações jurídicas fabricadas. Não chega a zero, então o foco é limitar as consequências.

*Aparece em:* Q4, Q14, Q24, Q36, Q44

### Grounding e citations

*Grounding*: responder apenas com base em contexto recuperado (RAG). *Citations*: exigir que o modelo cite trechos das fontes e verificar programaticamente que o trecho existe e sustenta a afirmação. Citação não verificada é teatro.

*Aparece em:* Q15, Q17, Q24, Q36, Q47

### Abstention

Fazer do “não sei” uma resposta de primeira classe no produto e no prompt. Modelos tendem a chutar com confiança porque treino e benchmarks premiam o palpite.

*Aparece em:* Q24

### Emergent misalignment

Fenômeno em que o fine-tuning em uma tarefa estreita e ruim (como escrever código inseguro sem avisar) faz o modelo ficar mal alinhado de forma ampla, em prompts sem relação. Parece que ele aprende um traço, não só a tarefa.

*Aparece em:* Q50

### Chain-of-thought monitoring

Ler o raciocínio de um modelo com outro modelo para detectar más intenções. Serve como camada de *detecção*, não como controle, porque o trace pode não refletir a computação real, e treinar contra o monitor ensina o modelo a escondê-lo (obfuscated reward hacking).

*Aparece em:* Q12, Q52

## 9. Red teaming e avaliação de segurança

### Red teaming

Teste adversarial para achar falhas antes dos atacantes. Combina especialistas humanos (segurança e domínio) com ferramentas automáticas, antes do lançamento e de forma contínua, pois mudanças de modelo, prompt ou tools reabrem brechas.

*Aparece em:* Q13, Q30, Q35, Q36, Q41

### garak e PyRIT

Ferramentas de scanner adversarial automatizado para LLMs (garak, e PyRIT da Microsoft), úteis para escala e regressão, mas só encontram variações de ataques conhecidos.

*Aparece em:* Q35

### Regression eval

Todo achado de red team vira um teste permanente rodado a cada mudança, para a falha não voltar silenciosamente. Um achado que não vira teste será redescoberto em um incidente.

*Aparece em:* Q30, Q33, Q35, Q43, Q47

### Attack success rate (ASR)

Porcentagem de ataques que conseguem seu objetivo. É a métrica acompanhada ao longo do tempo por categoria de ataque.

*Aparece em:* Q2, Q35

### HarmBench, StrongREJECT e AdvBench

Benchmarks de robustez a jailbreak. *HarmBench* padroniza avaliação de red team. *StrongREJECT* mede se o jailbreak produziu conteúdo realmente útil e nocivo, e não só uma não-recusa. *AdvBench* é o conjunto de prompts usado com o GCG.

*Aparece em:* Q36

### XSTest

Benchmark de over-refusal: prompts seguros formulados para parecerem perigosos, que pegam modelos “seguros” por serem inúteis.

*Aparece em:* Q23, Q36

### AgentHarm e AgentDojo

Benchmarks para agents. *AgentHarm* mede se agents com tools podem ser levados a completar tarefas nocivas de vários passos. *AgentDojo* mede ataques e defesas de prompt injection em agents, pontuando utilidade e taxa de ataque juntas.

*Aparece em:* Q36

## 10. Governança, regulação e responsible AI

### OWASP Top 10 for LLM Applications

Lista padrão dos dez riscos mais importantes em aplicações com LLM (LLM01 a LLM10), usada como vocabulário comum em revisões de segurança. Para agents, a cadeia crítica é LLM01 (injection) + LLM05 (saída mal tratada) + LLM06 (agency excessiva).

*Aparece em:* Q4, Q9, Q26, Q33, Q35

### OWASP Top 10 for Agentic Applications

Lista complementar (dezembro de 2025, ASI01 a ASI10) organizada em torno de autonomia: planos, tools, identidades, memória e comunicação entre agents. Inclui riscos como Agent Goal Hijack e Rogue Agents. Em uma revisão de agent, use as duas listas.

*Aparece em:* Q4, Q33

### System prompt leakage (LLM07)

Premissa: o system prompt vai vazar, porque extraí-lo é fácil. Consequência: nenhum segredo, credencial ou lógica de autorização no prompt. Autorização e chaves ficam no servidor.

*Aparece em:* Q4, Q5, Q18

### Model card vs system card

*Model card* documenta o modelo: uso pretendido, avaliações e limitações. *System card* documenta o sistema implantado (modelo, guardrails, retrieval e tools) e seu comportamento de ponta a ponta.

*Aparece em:* Q36, Q37

### EU AI Act

Lei europeia de IA com níveis de risco: práticas proibidas, alto risco (contratação, crédito, saúde etc., com exigências de logging, supervisão humana e avaliação de conformidade), transparência (chatbots devem se declarar IA) e risco mínimo. Há obrigações separadas para modelos de propósito geral. Datas de alto risco foram adiadas por emendas em 2026, então confirme o texto adotado.

*Aparece em:* Q37, Q41, Q47

### NIST AI RMF

Framework voluntário dos EUA organizado em quatro funções: Govern, Map, Measure, Manage. Há um Generative AI Profile (NIST AI 600-1) que o aplica a riscos de LLMs.

*Aparece em:* Q17, Q27, Q37, Q45, Q46

### Audit logging

Registros imutáveis do que foi pedido, gerado, chamado e aprovado, com identidade de humano e de agent. Em tensão com privacidade, resolvida com logs redigidos e acesso restrito aos dados brutos.

*Aparece em:* Q15, Q28, Q37, Q39, Q44

### Fairness: adverse impact e counterfactual testing

Em triagem de currículos, mede-se disparidade (como a razão de impacto adverso, a regra dos quatro quintos) e testa-se por *perturbação contrafactual*: trocar nome, pronome ou escola no mesmo currículo e medir a mudança da nota.

*Aparece em:* Q36, Q47

### Automation bias

Tendência humana de aceitar a sugestão da máquina sem questionar. Se os revisores quase nunca discordam, a supervisão humana existe só no papel.

*Aparece em:* Q47

## 11. Operação, incidentes e provenance de conteúdo

### Incident response e scoping

Em um vazamento, a ordem é conter com o menor corte possível, preservar evidências, delimitar o alcance (quais requisições, de quem para quem) e só então divulgar. Delimitar depende dos logs: tenant, documentos recuperados, chave de cache e principal por requisição.

*Aparece em:* Q42, Q43, Q46, Q50, Q51

### Canary tenant / canary prompt

Um *canary tenant* é um cliente de teste cujos dados nunca devem aparecer em lugar nenhum, para detectar vazamento cruzado. *Canary prompts* são entradas idiossincráticas cujas respostas características servem de impressão digital para identificar um modelo clonado.

*Aparece em:* Q42, Q45

### Distillation e model extraction

*Weight extraction* recupera parâmetros do modelo via API. *Behavioural cloning* (a forma comercial comum) coleta pares entrada/saída e ajusta um modelo menor com eles. Não dá para impedir por completo sem destruir o produto: eleva-se o custo, detecta-se e age-se por via contratual.

*Aparece em:* Q45

### Watermarking (SynthID-Text)

Marca estatística escondida no texto gerado: a amostragem é levemente enviesada por valores pseudoaleatórios ligados a uma chave secreta, e a detecção testa essa assinatura. Enfraquece em textos curtos, factuais, código, traduzidos ou muito editados.

*Aparece em:* Q45, Q49

### AI-text detectors

Detectores que tentam adivinhar se um texto foi escrito por IA. Não devem embasar decisões sobre pessoas: um estudo citado achou taxa de falso positivo média de 61,3% em redações de falantes não nativos de inglês, por viés estrutural dos detectores baseados em perplexidade.

*Aparece em:* Q49

### AP2 (Agent Payments Protocol) e mandates

Protocolo para pagamentos feitos por agents, em que cada decisão gera um artefato assinado e encadeado por hash: o que o usuário autorizou, o que o comerciante finalizou e o que foi cobrado. Torna uma disputa resolvível por verificação, não por confiar no log do operador.

*Aparece em:* Q48
