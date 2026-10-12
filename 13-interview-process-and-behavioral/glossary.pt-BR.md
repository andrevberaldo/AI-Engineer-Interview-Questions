# 📚 Glossário de estudo — Interview Process & Behavioral

> 🇧🇷 Material de apoio em português do Brasil. Cada termo traz uma explicação simples (e, quando ajuda, uma analogia) e indica as perguntas de [questions.pt-BR.md](questions.pt-BR.md) em que ele aparece. Os termos técnicos ficam em inglês, como no resto do material.

**Como usar:** leia o glossário antes das perguntas para destravar o vocabulário, e volte a ele quando um termo te travar no meio de uma resposta. As explicações são deliberadamente simplificadas; as respostas completas estão nas perguntas.

## Índice

- [1. O processo seletivo e os tipos de rodada](#1-o-processo-seletivo-e-os-tipos-de-rodada)
- [2. Como responder: estrutura e sinais de senioridade](#2-como-responder-estrutura-e-sinais-de-senioridade)
- [3. Evals e medição de qualidade](#3-evals-e-medição-de-qualidade)
- [4. Produção, incidentes e operação](#4-produção-incidentes-e-operação)
- [5. Técnicas de AI citadas nas histórias](#5-técnicas-de-ai-citadas-nas-histórias)
- [6. Segurança, dados e governança](#6-segurança-dados-e-governança)

## 1. O processo seletivo e os tipos de rodada

### Interview loop

O conjunto completo de rodadas de uma contratação, da triagem inicial ao debrief final. Um loop típico de AI engineer tem várias rodadas (recruiter, código, conhecimento de domínio, system design, comportamental) ao longo de algumas semanas.

*Aparece em:* Q1, Q15, Q21

### Debrief

Reunião em que os entrevistadores comparam suas notas e decidem se contratam. É nela que frases como "forte tecnicamente, mas não soube explicar o que entregou" derrubam ofertas.

*Aparece em:* Q10

### Recruiter screen

Primeira conversa, curta, com o recrutador: logística, motivação, nível da vaga e expectativa salarial. Vale ter um resumo de 90 segundos do seu melhor projeto de AI, com um número.

*Aparece em:* Q21

### Behavioral round

Rodada em que perguntam sobre experiências passadas ("conte uma vez em que...") para avaliar julgamento, comunicação e como é trabalhar com você. Em níveis seniores pesa tanto quanto as rodadas técnicas.

*Aparece em:* Q33, Q42

### STAR

Roteiro para contar histórias: Situation (contexto), Task (o que cabia a você), Action (o que você fez) e Result (resultado, com números). Ajuda a não se perder em detalhes e a mostrar o que você, pessoalmente, decidiu.

> 💡 **Pense assim:** Uma notícia bem escrita: contexto, seu papel, o que aconteceu e o desfecho, sem enrolação.

*Aparece em:* cabeçalho do módulo (a estrutura vale para todas as respostas comportamentais)

### Domain deep-dive

Rodada técnica de conhecimento de AI (transformers, fine-tuning vs RAG, evals etc.) em que o entrevistador aprofunda até você chegar ao limite do que sabe. Saber dizer onde esse limite está vale mais do que chutar.

*Aparece em:* Q7

### Take-home

Tarefa para fazer em casa, geralmente de algumas horas (por exemplo, construir um pequeno serviço de RAG). Os avaliadores olham, nesta ordem aproximada: se roda, se tem evals, se o README explica os trade-offs e se a engenharia é sensata.

*Aparece em:* Q8, Q20

### Defence round (walkthrough do take-home)

Conversa posterior em que você explica e defende cada decisão do seu take-home. Por isso, mesmo usando ferramentas de AI para fazer, você precisa saber justificar cada escolha: se não sabe explicar, na prática não foi você que decidiu.

*Aparece em:* Q8, Q10, Q20

### AI-assisted coding round

Rodada de código em que a empresa permite ou exige o uso de um assistente de AI (Copilot, Claude Code, Cursor etc.) e avalia como você o dirige, verifica o resultado e pega os erros dele. As regras variam, então vale perguntar ao recrutador antes.

*Aparece em:* Q21, Q25, Q41

### Forward-deployed engineer (FDE)

Engenheiro que trabalha dentro do cliente (geralmente corporativo) para descobrir o problema, definir o escopo, construir e colocar em produção a solução. Os mais valorizados fazem discovery antes de escrever código.

*Aparece em:* Q39

## 2. Como responder: estrutura e sinais de senioridade

### Blast radius

O tamanho do estrago possível se algo der errado. Reduzir o blast radius é limitar as consequências de uma falha: manter um humano na decisão, restringir as ações de um agente, liberar para poucos usuários primeiro.

> 💡 **Pense assim:** O raio de uma explosão: você não evita todo acidente, mas pode fazer o laboratório ficar longe da vizinhança.

*Aparece em:* Q13, Q19, Q39, Q40, Q42

### Steelman

O contrário de espantalho: apresentar a melhor versão possível do argumento do outro lado antes de discordar. Em disputas técnicas, mostra respeito e melhora a decisão.

*Aparece em:* Q29

### Calibrated uncertainty

Expressar a confiança que de fato se tem: dizer "não sei" onde não sabe, mostrar o que sabe ao redor e como fecharia a lacuna. Sinaliza julgamento confiável, ao contrário de enrolar ou chutar.

*Aparece em:* Q7, Q42

### Postmortem (blameless)

Documento escrito após um incidente: linha do tempo, impacto, causa e ações corretivas. É "blameless" quando o foco está nos sistemas e processos que permitiram a falha, não em culpar pessoas.

*Aparece em:* Q35, Q41

### Detection latency

O tempo entre um problema começar e alguém perceber. Na resposta de postmortem do repositório, é a descoberta principal: a causa é pontual, mas a falta de visibilidade é estrutural.

*Aparece em:* Q35

### Sunk cost

Esforço e dinheiro já gastos e que não voltam. Falácia comum: continuar um projeto ruim só porque já se investiu muito nele. Matar o projeto cedo costuma ser a decisão correta.

*Aparece em:* Q22

## 3. Evals e medição de qualidade

### Evals e eval set (golden set)

Evals (evaluations) são testes sistemáticos de uma feature de AI; o eval set é o conjunto de exemplos reais com resultado esperado usado neles. Mesmo 40 a 60 casos feitos à mão e um script de passa/falha já ajudam muito, e a maturidade em evals é um dos sinais de senioridade mais fortes nas entrevistas. "Conferimos a olho" pesa contra.

> 💡 **Pense assim:** Uma prova com gabarito: você refaz a prova a cada mudança para ver se a nota subiu ou caiu.

*Aparece em:* Q16, Q18, Q30, Q32, Q33

### LLM-as-judge

Usar um LLM para dar nota às respostas de outro modelo (por exemplo, "esta resposta é fiel aos documentos?"). Útil porque escala, mas precisa ser calibrado contra rótulos humanos, pois tem vieses e pontos cegos.

*Aparece em:* Q12, Q18, Q22, Q30, Q33

### Judge calibration

Medir o quanto o juiz automático concorda com avaliadores humanos num conjunto rotulado (no exemplo do repositório, cerca de 92% de concordância) e conhecer onde ele erra. Sem isso, a nota do juiz não é confiável.

*Aparece em:* Q12, Q33

### Code-graded assertions

Checagens feitas por código determinístico (o formato é válido? há PII na saída? o JSON bate com o schema?), em vez de um juiz LLM. São grátis, rápidas e exatas, então devem ser usadas sempre que possível.

*Aparece em:* Q12, Q17

### Faithfulness

Medida de quanto a resposta se sustenta nos documentos recuperados, sem inventar nada além deles. Em RAG, é uma das métricas centrais.

*Aparece em:* Q3, Q12, Q13, Q22, Q35

### Recall@k

Fração dos casos em que o documento correto aparece entre os k primeiros resultados da busca. Se o retriever nem traz o trecho certo, melhorar o reranker não adianta: por isso se mede antes.

*Aparece em:* Q8, Q20

### Regression test

Teste que garante que algo que funcionava continua funcionando. Todo incidente de AI deve virar um caso de teste permanente no eval set, para o mesmo erro não voltar.

*Aparece em:* Q10, Q17, Q30, Q35, Q38

### CI gate

Etapa do pipeline de integração contínua (CI) que bloqueia a mudança se as evals caírem abaixo do limite. É o que transforma o eval de painel decorativo em disciplina que decide o deploy.

*Aparece em:* Q12, Q17, Q30, Q38, Q41

### Proxy metric

Métrica indireta, barata de medir em tráfego real, que sinaliza qualidade sem medi-la diretamente: taxa de recusa, falhas de schema, retrieval vazio, thumbs-down. Costumam avisar horas ou dias antes das reclamações.

*Aparece em:* Q23, Q35

### Holdout (grupo de controle)

Um grupo deliberadamente deixado sem a feature para servir de comparação. Dá um contrafactual honesto do impacto, em vez de comparar com o trimestre passado ou com a adoção bruta.

*Aparece em:* Q26

### Guardrail metric

Métrica de segurança que acompanha a principal para pegar um custo escondido. Exemplo do repositório: o tempo de atendimento caiu, mas a taxa de reabertura ficou estável, provando que não ficou mais rápido porque ficou pior.

*Aparece em:* Q26

### Selection bias e novelty effect

Selection bias: quem adota primeiro costuma ser a pessoa mais forte, então comparar usuários com não usuários favorece a feature. Novelty effect: as primeiras semanas exageram o ganho real, que só aparece depois de assentar.

*Aparece em:* Q26

### Inter-annotator agreement

O quanto duas pessoas concordam ao rotular os mesmos exemplos. Se humanos discordam entre si, os rótulos são ruído e um modelo que concorda com eles não está provando nada.

*Aparece em:* Q22

### Eval staleness (eval defasado)

Quando o eval set já não representa o tráfego real (os usuários mudaram, o eval ficou parado). As notas sobem, mas o produto não melhora. Corrigir: renovar o eval com amostras do tráfego atual.

*Aparece em:* Q18, Q35

## 4. Produção, incidentes e operação

### Non-determinism

O fato de o mesmo input poder gerar saídas diferentes. Vem da amostragem (temperature), mas também de batching, atualizações do provider e variação no contexto montado. Por isso o debug vira estatística: medir uma taxa de falha em vez de reproduzir uma vez.

*Aparece em:* Q9, Q13, Q15

### Replay e logging completo

Registrar todo o request (prompt, contexto, versão do modelo) para poder reexecutar exatamente a mesma situação dezenas de vezes. Transforma um bug fantasma em uma taxa de falha mensurável.

*Aparece em:* Q15

### Model alias vs snapshot

Alias é um nome estável (como "latest") que o provider pode reapontar para um modelo novo sem avisar. Snapshot é uma versão datada e fixa. Fixar o snapshot evita mudanças silenciosas de comportamento.

*Aparece em:* Q15, Q23, Q35, Q41

### Prompt como código

A ideia de que prompts devem ser versionados, revisados, testados e revertidos como código. Editar prompt ao vivo num painel, sem versionamento, é a origem de muitos incidentes.

*Aparece em:* Q17

### Canary e staged rollout

Liberar uma mudança primeiro para uma pequena fração do tráfego (por exemplo, 5% ou 10%), comparar a qualidade e só então expandir. Limita o estrago se algo quebrar. No shadow canary, a versão nova roda em paralelo ao tráfego real sem mostrar o resultado ao usuário, só para comparar com a antiga.

*Aparece em:* Q1, Q17, Q33, Q35, Q39

### Feature flag e rollback

Feature flag é uma chave que liga ou desliga uma funcionalidade sem novo deploy. Rollback é voltar à versão anterior. Juntos, tornam reverter uma mudança quase instantâneo.

*Aparece em:* Q17, Q19, Q23, Q33, Q36

### p95 (latência)

O tempo de resposta abaixo do qual ficam 95% das requisições: só 5% são mais lentas. Mostra a experiência dos casos ruins, que a média esconde.

*Aparece em:* Q9, Q23, Q29

### On-call e paging

On-call: escala de quem fica de plantão. Paging: o alerta que acorda essa pessoa. A pergunta de AI é o que merece acordar alguém (queda de serviço) e o que espera até o horário comercial (queda sutil de qualidade).

*Aparece em:* Q10, Q13, Q14, Q23, Q42

### Prompt caching

Reaproveitar o processamento de um início de prompt que se repete (system prompt fixo), reduzindo custo e latência. Uma das primeiras otimizações, por ser quase de graça.

*Aparece em:* Q16

### Model routing

Mandar consultas simples para um modelo barato e as difíceis para um modelo de ponta, por um classificador ou heurística. Corta custo, mas exige medir quanto de qualidade se perde quando o roteador erra.

*Aparece em:* Q16, Q22, Q23

### Abstraction layer (camada de modelo)

Camada de código que isola o resto do sistema do provider específico, facilitando trocar de modelo. Reduz o acoplamento a um fornecedor, que é o que encarece uma deprecation.

*Aparece em:* Q14, Q33

### Model deprecation e migração

Quando o provider aposenta um modelo (muitas vezes com meses de aviso), toda a base precisa migrar. É um problema de evals: o eval set é a especificação que o substituto precisa cumprir, e o comportamento pode mudar silenciosamente mesmo com modelo "melhor".

*Aparece em:* Q10, Q14, Q32, Q33, Q36

## 5. Técnicas de AI citadas nas histórias

### RAG

Retrieval-Augmented Generation: antes de responder, o sistema busca trechos relevantes numa base de documentos e os coloca no prompt, para o modelo responder com base neles. Permite conteúdo atualizado e citações sem retreinar o modelo.

> 💡 **Pense assim:** Prova com consulta: o aluno pode olhar o material antes de responder.

*Aparece em:* Q3, Q8, Q13, Q14, Q20

### Chunking

Dividir documentos em pedaços para indexar e buscar. Um corte ruim pode separar uma linha de tabela do seu cabeçalho, levando o modelo a responder com confiança a partir de meia tabela.

*Aparece em:* Q1, Q8, Q15, Q20, Q23

### Reranker

Segundo estágio da busca que reordena os candidatos recuperados, usando um modelo mais preciso, para colocar os mais relevantes no topo. Só ajuda se o retriever já trouxe o trecho certo.

*Aparece em:* Q1, Q7, Q8, Q20

### Fine-tuning vs retrieval vs prompting

Três formas de adaptar um modelo: prompting (instruir no texto), retrieval (trazer conhecimento externo) e fine-tuning (ajustar os pesos). Retrieval serve melhor para conhecimento que muda; fine-tuning, para ensinar formato ou estilo de forma consistente. Em geral decide-se medindo, não por preferência.

*Aparece em:* Q9, Q14, Q16, Q24, Q33

### LoRA

Técnica barata de fine-tuning que treina só um pequeno conjunto de pesos adicionais em vez do modelo todo. No exemplo do repositório, serviu para aprender onde o fine-tuning ganha ou perde para um prompt bem feito.

*Aparece em:* Q9

### Frontier model e open-weights

Frontier model: os modelos mais capazes de ponta, geralmente via API. Open-weights: modelos cujos pesos são públicos, que dá para rodar na própria infraestrutura (mais controle de dados e custo, mas muitas vezes menos qualidade em tarefas difíceis).

*Aparece em:* Q9, Q14, Q29, Q31, Q33

### Tool calling e agentes

Tool calling: o modelo decide chamar funções externas (busca, banco, APIs) e usa o resultado. Um agente é um loop em que o modelo escolhe ações com essas ferramentas até concluir a tarefa.

*Aparece em:* Q15, Q21, Q25, Q40, Q41

### Structured output e schema validation

Pedir que a saída siga um formato fixo (como JSON com campos definidos) e validá-la por código. Evita que um campo faltando quebre o sistema que consome a resposta.

*Aparece em:* Q17, Q22, Q23, Q35, Q36

### Hallucination

Quando o modelo gera algo fluente e confiante, mas falso ou sem base. Culpar a "alucinação" como causa final é visto como parar de depurar cedo demais: mostra-se a mitigação (citações, medir fidelidade).

*Aparece em:* Q3, Q13, Q32

### Prompt injection

Texto malicioso (ou apenas inesperado) dentro dos dados que o modelo lê, que o faz seguir instruções que não deveria. Tratar entradas e saídas do modelo como não confiáveis é regra básica de segurança.

*Aparece em:* Q32, Q37, Q40

### Human-in-the-loop e confidence routing

Colocar uma pessoa para revisar quando o risco é alto ou o sistema está inseguro. No confidence routing, resultados de alta confiança passam direto e os de baixa vão para revisão humana, limitando o estrago.

*Aparece em:* Q4, Q12, Q19, Q28, Q41

## 6. Segurança, dados e governança

### PII

Personally Identifiable Information: dados que identificam uma pessoa (nome, número de conta, endereço). Mandar PII a uma API de terceiros envolve contrato, regulação e consentimento.

*Aparece em:* Q12, Q31

### DPA e zero-retention

DPA (data processing agreement) é o contrato que define como o provider trata seus dados. Zero-retention e no-training significam que ele não guarda nem usa os dados para treinar modelos. Também entram a residência dos dados (processar numa região específica) e leis como GDPR (UE) e HIPAA (saúde, EUA). Leia o que o contrato de fato cobre, em vez de supor.

*Aparece em:* Q31, Q37

### Redaction e pseudonymisation

Substituir os dados sensíveis por marcadores (por exemplo, [NOME_1]) antes da chamada à API e restaurá-los depois (rehydration). Costuma ser barato e perder pouca qualidade. Para achar os dados, usam-se regex e NER (Named Entity Recognition, que reconhece nomes, lugares, organizações).

*Aparece em:* Q31

### Least privilege e policy no tool

Dar ao agente apenas as permissões estritamente necessárias e validar as regras no servidor, dentro da ferramenta. Uma instrução no prompt é um pedido; só o controle no código é garantia.

> 💡 **Pense assim:** Um caixa de banco: o limite de saque está no sistema, não num aviso pedindo ao caixa que não passe do valor.

*Aparece em:* Q40

### Reversibility e human approval

Classificar as ações do agente por reversibilidade: leituras livres, escritas reversíveis com log, e o que move dinheiro ou não pode ser desfeito exige aprovação humana.

*Aparece em:* Q30, Q40, Q41

### Automation bias

Tendência de confiar em um sistema que acerta quase sempre e parar de conferir. Se o revisor aprova 100% das sugestões, ele não está revisando. Acompanhar a override rate (quando o humano discorda) ajuda a detectar isso.

*Aparece em:* Q42

### Build vs buy vs API

Decisão entre construir internamente, comprar um produto pronto ou chamar uma API de modelo. A regra prática do repositório: API para a inteligência, construir as camadas que tocam seus dados e usuários, comprar só onde o fornecedor vive daquele problema.

*Aparece em:* Q14, Q36

### Sunset (descontinuar uma feature)

Desligar uma funcionalidade de forma planejada: anunciar, mantê-la atrás de uma flag por um ciclo e contatar os poucos usuários que dependiam dela. Combinar de antemão o critério de decisão evita discutir o critério depois.

*Aparece em:* Q36
