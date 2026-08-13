# Resposta aos pareceres — o que foi medido e o que mudou

Ambos os revisores recomendaram **Weak Reject**, e ambos deram **2/5** em
*Meaningful comparison*. Este documento mapeia cada crítica a um artefato de
código e ao resultado empírico correspondente.

**Leia primeiro o achado nº 1.** Ele invalida a afirmação central do manuscrito e
deve determinar a reescrita.

---

## ⚠️ Status epistêmico dos achados — leia antes de citar qualquer número

Os pareceres avaliam **o manuscrito**. As medições abaixo são sobre **esta
reprodução**, construída a partir do texto porque o código original foi perdido.
A reprodução tem divergências documentadas (ver `README.md`): energia ~28 %
abaixo, trocas/h ~2,5× abaixo, pico sem AC de 37,8 °C contra 32–36 °C declarados.

Isso significa que os achados **não têm o mesmo peso**:

| Achado | Depende de | Robustez |
|---|---|---|
| **HNP: 79,8 % intra-tile** (§3b) | Apenas de $C_{th}=15$, $\Delta t=0{,}1$, $H_p=0{,}3$ — **todos publicados** | **Máxima.** Recalculável direto do manuscrito, independe desta reprodução |
| **Anti-short-cycling falha** (§3) | Da política aprendida | **Alta e conservadora.** Esta reprodução comuta *menos* que o manuscrito, logo no original a violação seria **pior**. O argumento aritmético ($\|\rho\|=5$ contra $B+B_c=14$) usa só a Tabela 2 e a eq. 5 |
| **PI domina o DQN** (§1) | Da fidelidade deste ambiente | **Inferência, não medição direta.** Comparação internamente válida (mesmo ambiente para ambos) e o DQN reproduz o manuscrito em conforto, \|T−24\| e sobreaquecimento — o que sustenta a transferência, mas não a prova |

**Consequência prática para uma carta de resposta aos revisores:**

- §3b e §3 podem ser **afirmados**: derivam de parâmetros publicados.
- §1 deve ser **refeito no código original** antes de ser afirmado. Se o PI também
  empatar lá, o reposicionamento é obrigatório. Se não empatar, a diferença
  aponta algo na física que a conversão texto→código não capturou — e isso
  também é informação valiosa.

Efeito colateral favorável: R1 escreveu *"Com a disponibilização de scripts e
código-fonte, tal problema deve ser resolvido"*. Publicar este diretório como
artefato — 79 testes, dump completo de hiperparâmetros, Tabela 1 derivada de
primeiros princípios — ataca diretamente a queixa de reprodutibilidade.

---

## 1. ACHADO CRÍTICO — a vantagem do RL não sobrevive a um baseline competente

> **R2:** "não é possível saber se a vantagem do DQN decorre do aprendizado ou de
> um baseline inadequadamente configurado. Seria importante incluir um termostato
> devidamente ajustado e outros controladores usuais, como PID ou MPC."

O revisor estava certo. Implementei um PI com anti-windup (`baselines.py`) e o
sintonizei por busca em grade sobre a mesma matriz 3×3 (`tune_pi`), com o mesmo
horizonte de decisão (action repeat 2) e o mesmo espaço de ação discreto.

| Controlador | Conf.[22,26] | Conf.[23,25] | \|T−24\| | Sobreaq. | Energia | Custo | Trocas/h |
|---|---|---|---|---|---|---|---|
| **PI sintonizado** (Kp=1,3, Ki=0,2) | **86,5 %** | **83,8 %** | **0,72** | **5,2 %** | **10,39** | **9,47** | 1,04 |
| Termostato deadband = 0 (o do paper) | 54,0 % | 12,2 % | 2,10 | 37,7 % | 7,55 | 7,11 | 0,97 |
| Termostato deadband = 1 | 77,7 % | 20,5 % | 1,78 | 14,0 % | 8,12 | 7,63 | 0,20 |
| DQN Agressivo (550k) | 86,5 % | 83,8 % | 0,77 | 5,2 % | 11,66 | 10,72 | 1,22 |
| DQN Equilibrado (550k) | 86,5 % | 83,8 % | 0,89 | 5,2 % | 10,90 | 9,90 | 0,78 |
| DQN Passivo (550k) | 86,5 % | 83,7 % | 0,87 | 5,2 % | 10,79 | 9,83 | 0,79 |

**O PI domina os três perfis DQN em TODAS as métricas** — mesmo conforto,
\|T−24\| melhor, energia e custo menores. Duas constantes sintonizadas superam
550 000 passos de treino.

Decomposição dos "+32 pp" reivindicados:

| Passo | Conf. larga | Ganho atribuível a |
|---|---|---|
| Termostato do paper (deadband = 0) | 54,0 % | — |
| Adicionar apenas histerese | 77,7 % | **+23,7 pp** → configuração do baseline |
| PI sintonizado | 86,5 % | **+8,8 pp** → controle clássico |
| DQN 550k passos | 86,5 % | **+0,0 pp** → **aprendizado** |

**Praticamente 100 % da vantagem reivindicada é atribuível à má configuração do
baseline, não ao aprendizado.** A afirmação do resumo — "82,9 % contra 50,9 % de
um termostato" — é tecnicamente verdadeira e substantivamente enganosa: compara
contra um adversário artificialmente incapaz.

Isto confirma parcialmente a suspeita do R1 sobre necessidade de RL profundo —
mas **atenção, são duas perguntas distintas**, e as respostas divergem:

| Pergunta | Resposta medida | Onde |
|---|---|---|
| RL supera controle clássico aqui? | **Não.** Um PI iguala e domina em custo | seção 1 |
| Se usar RL, precisa ser profundo (vs. tabular)? | **Sim.** 79,8 % das transições são intra-tile | seção 3b |

Ou seja: *nesta formulação do problema, RL não é necessário* — mas *se você usar
RL, ele tem de ser profundo*, porque a dinâmica lenta inviabiliza a discretização.
Não há contradição; a primeira é sobre escolha de paradigma, a segunda sobre
escolha de método dentro do paradigma.

**Recomendação:** reposicionar o trabalho. O DQN *iguala* um PI bem sintonizado —
resultado legítimo e publicável, mas com moldura honesta. O valor do RL só
aparecerá em regimes que o PI não trata: acoplamento entre salas, resposta
preditiva a agenda de ocupação, *demand response* com previsão de preço,
múltiplas fontes de resfriamento. É exatamente a literatura que o R1 aponta.

---

## 2. Ablação da recompensa — `ablation.py`

> **R2:** "A conclusão de que a modelagem da recompensa é mais decisiva do que o
> algoritmo também não é demonstrada pelos experimentos. [...] seria necessário
> realizar uma ablação da recompensa."

Implementadas 8 variantes, cada uma desligando **um** mecanismo:
`completa`, `sem_gradiente`, `sem_penal_troca`, `sem_short_cycling`,
`sem_penal_frio`, `quadratica_pura`, `degraus_legado`, `convencional`.

A variante `sem_gradiente` é decisiva: é a hipótese explícita da Seção 4.3
(platô plano ⇒ estacionar na borda). O ambiente agora suporta as três topologias
via `comfort_type`, e a métrica `cenarios_estacionados_pct` mede diretamente se a
política estaciona logo acima do teto.

```bash
python ablation.py --timesteps 550000 --seeds 0 1 2
```

### ✅ EXECUTADO — 8 variantes × 3 sementes × 550k passos (2 h 42 min)

Conforto na faixa estreita [23,25] — a métrica que separa os regimes:

| Variante | Média | IC95 | por semente | Cliff's δ | Efeito |
|---|---|---|---|---|---|
| **completa** (proposta) | **83,7** | [83,3; 83,8] | 83,8 / 83,3 / 83,8 | — | referência |
| `convencional` | 83,1 | [81,7; 83,8] | 83,8 / 81,7 / 83,8 | −0,11 | **desprezível** |
| `sem_short_cycling` | 83,7 | [83,3; 83,8] | 83,8 / 83,3 / 83,8 | **0,00** | **desprezível** |
| `sem_penal_frio` | 82,6 | [81,8; 83,8] | 81,8 / 82,0 / 83,8 | −0,56 | pequeno em magnitude |
| `sem_penal_troca` | 82,1 | [78,5; 83,8] | 83,8 / 78,5 / 83,8 | −0,11 | **desprezível** |
| `quadratica_pura` | 82,3 | [80,7; 83,8] | 83,8 / 80,7 / 82,5 | −0,56 | pequeno em magnitude |
| `degraus_legado` | 69,1 | [44,0; 82,2] | 81,2 / 82,2 / **44,0** | **−1,00** | **grande** |
| **`sem_gradiente`** | **47,7** | [34,0; 61,3] | **61,3 / 34,0 / 47,8** | **−1,00** | **grande** |

Nota sobre p-valores: com 3 vs. 3 sementes, o menor p bicaudal alcançável em teste
de permutação é $2/\binom{6}{3} = 0{,}10$. **Portanto p = 0,101 é o piso** — é o
resultado mais significativo possível com 3 sementes. É por isso que o Cliff's δ é
a evidência decisiva: δ = −1,00 significa **separação completa** (toda semente
ablacionada pior que toda semente da referência). Para obter p < 0,05 seriam
necessárias ≥ 4 sementes por grupo.

### Conclusão 1 — a hipótese da Seção 4.3 é CONFIRMADA

Remover o gradiente interno degrada a política de forma inequívoca:
conforto estreito **83,7 → 47,7** (−36 pp), \|T−24\| **0,85 → 1,46**,
sobreaquecimento **5,2 % → 13,8 %**, cenários estacionados na borda **0 % → 3,7 %**,
com separação completa entre sementes (δ = −1,00) e treino instável (desvio 13,7).

O platô plano de fato leva a política a estacionar perto da borda — exatamente o
bug de 26,1 °C observado na v4 original. **Isto é um resultado real e defensável.**

### Conclusão 2 — mas a formulação proposta NÃO supera a convencional

`convencional` (quadrática pura, **sem** gradiente interno, **sem** penalidade de
troca, **sem** anti-short-cycling, **sem** penalidade de frio) atinge **83,1 %**
contra 83,7 % da proposta: δ = −0,11, **desprezível**.

O motivo é conceitual: uma quadrática pura é uma parábola com máximo em 24 °C —
ela **já tem gradiente em todo o domínio**, por construção. O platô plano é que o
destrói. A proposta do paper apenas o *restaura*.

**Reenquadramento necessário:** a contribuição não é uma recompensa melhor que a
da literatura. É a **correção de uma falha que os próprios autores introduziram**
ao adotar o platô plano. O achado publicável é *negativo* — "platôs planos
quebram o aprendizado de controle térmico" — e não positivo.

### Conclusão 3 — três dos cinco termos da recompensa são decorativos

| Termo removido | Efeito no conforto |
|---|---|
| $R_{ciclo}$ (anti-short-cycling) | **δ = 0,00 — idêntico** |
| $R_{mudança}$ (penalidade de troca) | δ = −0,11, desprezível |
| $R_{frio}$ | 82,6 vs 83,7, magnitude pequena |

O resumo do manuscrito destaca *"Platô Quadrático mais penalidade
anti-short-cycling"* como a contribuição. A penalidade anti-short-cycling tem
**efeito exatamente nulo** no conforto — somado ao fato (§3) de que ela **não
cumpre o próprio objetivo** (71–83 % de violações de $d_{min}$).

Indício adicional, com ICs sobrepostos e portanto sugestivo: remover $R_{ciclo}$
*reduz* a energia (10,53 vs 11,09 kWh/dia, −5 %) sem custo de conforto. Se
confirmado com mais sementes, o termo é ativamente prejudicial: cobra energia e
não entrega proteção.

---

## 3. ACHADO — a penalidade anti-short-cycling não funciona

> **R2:** "a permanência mínima indicada é de 36 minutos, mas os agentes realizam
> cerca de 2,5 mudanças por hora. A distribuição dos intervalos entre
> acionamentos deveria ser apresentada para verificar se a proteção foi efetiva."

Medido em `stats_analysis.dwell_time_analysis`. O revisor suspeitou por
aritmética; a distribuição confirma e é pior:

| Agente | Mediana | Média | Mín | **Violações de $d_{min}$** |
|---|---|---|---|---|
| DQN Agressivo | 12 min | 36,0 min | 12 min | **83,3 %** |
| DQN Equilibrado | 12 min | 59,8 min | 12 min | **71,4 %** |
| DQN Passivo | 12 min | 57,7 min | 12 min | **72,9 %** |
| Termostato | 12 min | 46,2 min | 12 min | **79,0 %** |

A **mediana é 12 min** — uma única decisão. A média (36–60 min) é inflada por
longos períodos OFF e esconde completamente o problema. É o caso didático de um
agregado que mascara a distribuição.

**Causa raiz, aritmética:** a penalidade da eq. 5 vale no máximo $|\rho| = 5$,
contra ganhos de conforto de até $B + B_c = 14$. Um termo de recompensa que pode
ser superado por outro termo não é proteção; é sugestão.

**Correção — `MinDwellWrapper`:** impõe $d_{min}$ como restrição dura, na linha do
*shielding* de Xu et al. (2025). Aplicada ao agente **já treinado**, sem retreino:

| | Violações | Mediana | Conf. larga | \|T−24\| | Energia |
|---|---|---|---|---|---|
| Penalidade mole (eq. 5) | 71,4 % | 12 min | 86,5 % | 0,89 | 10,90 |
| **Restrição dura** | **6,4 %** | **48 min** | 86,5 % | 0,97 | 11,31 |

Elimina o short-cycling a custo **zero** em conforto binário (+0,08 °C em
\|T−24\|, +3,8 % de energia). Os 6,4 % residuais são artefato de medição na
primeira comutação de cada episódio.

**Lição transferível:** propriedades de segurança e integridade de hardware devem
ser impostas estruturalmente, não negociadas via recompensa.

---

## 3b. ACHADO — RL profundo É necessário, e o motivo é teórico

> **R1:** "O espaço de estados, embora contínuo, possui baixa dimensionalidade.
> Assim, não é claro que a solução realmente precisa de abordagem profunda (e
> mesmo contínua). Tal fator também deveria ser melhor investigado."

Pergunta legítima — 4 dimensões é pouco, uma tabela deveria bastar. Mas há um
motivo teórico para não bastar, e vem de **Zha et al. (2021)**, uma das
referências que o próprio R1 sugeriu:

> Em espaços contínuos com variáveis de dinâmica lenta, métodos tabulares com
> discretização falham porque a transição é **intra-tile** — a ação aterrissa no
> mesmo tile de onde partiu. O valor nunca se propaga entre tiles.

Medido em `tabular.intra_tile_fraction`, sem treinar nada:

| Grandeza | Valor |
|---|---|
| Espaço discretizado | 4 800 estados (20 × 10 × 24) |
| Largura do bin de temperatura | 1,00 °C |
| ΔT típico por passo (sala cheia) | 0,090 °C |
| Passos para cruzar um bin | **11,1** |
| **Transições intra-tile** | **79,8 %** |

**80 % dos backups atualizam um estado com o valor dele mesmo.** A dinâmica lenta
torna a discretização inoperante, exatamente como o HNP prevê.

Isto explica retroativamente o histórico do projeto: o `RESULTADOS.md` da v1
reporta que o Q-Learning tabular convergiu para **98,3 % OFF**. Foi lido na época
como "política conservadora"; o diagnóstico correto é que o agente não aprendeu
política alguma — aprendeu a não fazer nada, porque o valor nunca se propagou.

A migração para DQN (commit `5d185b5`) foi empiricamente correta e agora tem
justificativa teórica. **Este é material de contribuição**, não só de defesa: é
uma resposta quantificada a "por que RL profundo num problema de 4 dimensões".

### ✅ EXECUTADO — Q-Learning tabular, 550k passos × 3 sementes

| Controlador | Conf.[22,26] | Conf.[23,25] | \|T−24\| |
|---|---|---|---|
| PI sintonizado | 86,5 % | 83,8 % | 0,72 |
| DQN 550k | 86,5 % | 83,8 % | 0,89 |
| Termostato deadband = 0 | 54,0 % | 12,2 % | 2,10 |
| **Q-Learning tabular 550k** | **42,1 %** | **27,3 %** | **3,23** |

Por semente (conforto larga): 31,5 / 38,7 / 56,0 — média 42,1 %, desvio 12,6.

**O tabular é pior que o termostato de zona morta**, e instável. Mecanismo medido
durante o treino:

| | Valor |
|---|---|
| Intra-tile sob política aleatória | 79,9 % |
| Intra-tile **após** o treino | **59,8 %** |
| Cobertura de estados (teto atingido) | **76,7 %** |

A queda de 80 % → 60 % de intra-tile ocorre porque a política aprende a usar
HIGH, que muda a temperatura rápido o bastante para cruzar bins. Mas 60 % ainda
significa que **6 de cada 10 backups atualizam um estado com o valor dele mesmo**,
e a cobertura estagna: ~1 100 dos 4 800 estados nunca são visitados, mesmo com
ε = 0,655 na fase exploratória.

**Resposta ao R1:** se usar RL, ele precisa ser profundo. A baixa
dimensionalidade não salva o método tabular, porque o obstáculo não é dimensão —
é a razão entre o passo da dinâmica (0,090 °C) e a resolução da discretização
(1,00 °C). Este é o achado mais limpo do conjunto: deriva de parâmetros
publicados, tem mecanismo explicado e literatura de apoio (Zha et al. 2021).

---

## 4. Reprodutibilidade dos hiperparâmetros — `train.py`

> **R2:** "Não são informados todos os valores de B, k, ρ, penalidade de frio,
> ruído, duração do replay buffer, fator de desconto, frequência de atualização
> da target network, arquitetura da rede e estratégia de exploração."

`save_metadata` agora persiste, por modelo: a `ClassroomConfig` completa
(inclui $B$, $k$, $\rho$, penalidade de frio, ruído), os hiperparâmetros
**declarados** e os **efetivos** extraídos do objeto SB3 (γ, τ, buffer_size,
target_update_interval, exploration_*, arquitetura, nº de parâmetros) — os
defaults contam tanto quanto o que foi escrito no código.

> **R2:** "A eficiência elétrica e os valores de COP também precisam ser
> derivados ou justificados por referências técnicas."

`ac_physics.py` **deriva** a Tabela 1 de dois parâmetros de catálogo (30 000 BTU/h
e COP por nível) em vez de tabelá-la. Verificado por teste:
LOW 0,637/0,64 · MEDIUM 1,347/1,35 · HIGH 2,931/2,93 kW.

Os 4 parâmetros que o manuscrito não publica estão marcados `# INFERIDO` em
`config.py`, com a base de cada inferência.

---

## 5. Comparação com mais algoritmos — `train.py`

> **R1:** "implementa apenas 2 agentes simples de RL [...] seria mais valioso
> comparar diferentes abordagens (PPO, por exemplo)."

`ALGOS` agora cobre **DQN, PPO, A2C, SAC**:

```bash
python train.py --algos DQN PPO A2C --seeds 0 1 2
```

---

## 6. Rigor estatístico — `stats_analysis.py`

> **R2:** "Três sementes também constituem um número pequeno [...] Intervalos de
> confiança e testes estatísticos ajudariam."

- `bootstrap_ci` — IC percentil (não assume normalidade, inviável com n=3)
- `permutation_test` — teste bicaudal sem suposição de variâncias iguais
- `cliffs_delta` — tamanho de efeito não paramétrico com magnitude qualitativa
- `compare_agents` — tabela consolidada contra um baseline

Reportar tamanho de efeito junto do p-valor é o padrão mínimo: com n=3 um
p-valor diz pouco.

---

## 7. Inicialização por especialista — `expert_init.py`

Motivado por Xu et al. (2025), que reporta 8,8× de speedup destilando
conhecimento de especialista. O especialista aqui já existia: o PI.

Implementado e funcional (**80,1 % de concordância** após clonagem de
comportamento sobre os Q-values tratados como logits, mais semeadura do replay
buffer). **Resultado negativo honesto:** com 6 000 passos, o warm start degrada
para 37,2 % de conforto — a exploração ε-greedy default do DQN destrói a política
clonada antes que o TD-learning a recupere. Aproveitar o warm start exige também
reduzir a exploração inicial, o que não foi ajustado.

---

## 8. Contaminação do conjunto de avaliação — `scenarios.py`

> **R2:** "Caso os pesos tenham sido selecionados a partir dos resultados nessa
> matriz, ela não constitui um conjunto independente de avaliação. Recomendo
> separar os cenários empregados na calibração daqueles usados no teste ou
> realizar uma avaliação em um conjunto mais amplo de condições geradas
> aleatoriamente."

Implementadas as duas alternativas:

- `split_scenario_matrix()` — divisão em xadrez (C2/C4/C6/C8 para calibração, o
  resto para teste). Xadrez em vez de por linha porque dividir por condição
  ("treinar no frio, testar no quente") mediria extrapolação, não generalização.
- `sample_random_scenarios(n, seed)` — conjunto amplo e independente sobre o
  espaço contínuo (temperatura, ocupação, hora), com semente fixa distinta das de
  treino, garantindo comparação emparelhada entre agentes. Testado como disjunto
  da matriz de calibração.

**Ressalva importante e não eliminada:** o PI do achado nº 1 foi sintonizado na
matriz 3×3, a mesma usada para avaliar. O resultado deve ser lido como *"o PI
iguala o DQN nas condições em que ambos foram ajustados"* — não como
superioridade geral. Refazer o achado nº 1 sintonizando o PI apenas na partição
de calibração e reportando no conjunto aleatório é o próximo passo obrigatório.

---

## 9. Revisão da literatura — pendente (não é trabalho de código)

Ambos os revisores deram **2/5**. R1 listou 5 referências específicas, todas
presentes em `papers_sugeridos/` e já mapeadas:

| Referência | Uso no trabalho |
|---|---|
| Zha et al. 2021 (HNP) | Justifica teoricamente o abandono do Q-Learning tabular: com variáveis de dinâmica lenta a transição é intra-tile e o valor não se propaga. Responde à pergunta do R1 sobre necessidade de RL profundo |
| Xu et al. 2025 (Sci. Reports) | Fundamenta o *safety shield* e a taxa de violação; origem da inicialização por especialista |
| Boutahri & Tilioua 2025 (BOPTEST) | Benchmark padronizado + baseline PI + validação sim-to-real |
| Dai et al. 2025 (BuildingGym) | Arquitetura de referência com EnergyPlus |
| Review Elsevier | Taxonomia para posicionamento |

Pendências textuais do R2: a referência de Pires contém os campos provisórios
"Nome da Instituição, Cidade"; e termos como função $Q$ e $N_{ocup}$ são usados
sem definição.

---

## 10. Figura 3 / fronteira de Pareto

> **R2:** "Na métrica principal, os três perfis têm 82,9 % de conforto, enquanto o
> perfil Equilibrado apresenta o menor custo. Nesse plano, ele domina os outros
> dois."

Correto — no plano conforto-larga × custo não há fronteira, há um dominante. A
Figura 3 deve usar **conforto na faixa estreita** ou **\|T−24\|** no eixo de
qualidade, as únicas métricas em que os perfis de fato se separam. E com o PI
incluído, é ele quem domina o canto.

---

## Estado dos artefatos

| Item | Estado |
|---|---|
| PI + termostato configurável + sintonia | ✅ `baselines.py` |
| Restrição dura de permanência | ✅ `wrappers.py::MinDwellWrapper` |
| Análise de permanência / short-cycling | ✅ `stats_analysis.py` |
| IC bootstrap, permutação, Cliff's delta | ✅ `stats_analysis.py` |
| Dump completo de hiperparâmetros | ✅ `train.py` |
| PPO / A2C | ✅ `train.py::ALGOS` |
| Inicialização por especialista | ✅ `expert_init.py` (resultado negativo) |
| Suíte de regressão | ✅ `tests/test_env.py` (79 testes + 1 xfailed) |
| Diagnóstico HNP (intra-tile 79,8 %) | ✅ `tabular.py::intra_tile_fraction` |
| Q-Learning tabular completo | ✅ **executado** — 42,1 % conf., pior que o termostato |
| Separação calibração/teste + cenários aleatórios | ✅ `scenarios.py` |
| Notebook de diagnóstico | ✅ `notebooks/01_diagnostico_critico.ipynb` |
| Suíte de regressão | ✅ **79 testes + 1 xfailed** |
| Ablação da recompensa | ✅ **executada** — 24 treinos, 2 h 42 min |
| MPC como baseline | ❌ não implementado |
| Refazer achado nº 1 com PI sintonizado só na calibração | ❌ próximo passo obrigatório |
| Migração BOPTEST / BuildingGym | ❌ trabalho de médio prazo |

---

## Continuação em `v5/`

Os itens abaixo foram executados após esta revisão e estão documentados em
`v5/README.md` e nos cadernos `v5/notebooks/`:

| Item | Resultado |
|---|---|
| Demanda contratada derivada da condição de projeto | o valor arbitrado (0,70 kW) era **infactível** — 72 % abaixo do regime; com o contrato derivado (1,32 kW) a restrição não aperta e a hipótese perde base |
| Faixa estreita com retreino de ambos os controladores | estreitar **amplia** a vantagem do PI: +0,0 → +3,0 → +8,3 pp |
| Desempenho contra orçamento de treino | em ±2,0 °C o DQN converge sobre o PI; em ±0,5 °C fica 16,9 pp abaixo |
| Mecanismo do consumo excedente | os três perfis descartam o nível de melhor COP; causa é o argmax, não a recompensa (um SAC com a recompensa idêntica o utiliza) |
| Conjunto independente de 150 cenários aleatórios | PI e DQN Agressivo empatam em conforto; o PI gasta 11,4 % menos |
| Contrato de observação verificado por schema | modelos `..._obs9.zip` tinham na verdade 10 canais — a dimensão vivia no nome do arquivo |
| Classificação da literatura por qualidade do baseline | dos seis trabalhos revisados, apenas dois comparam com controle clássico sintonizado |
