# -*- coding: utf-8 -*-
"""
Gera e executa os notebooks de resultados.

Os notebooks não recomputam nada por conta própria: chamam `hvac.results` e
`hvac.figures`, as mesmas funções que alimentam o artigo. É o que impede que
notebook, tabela e figura divirjam — o defeito que a v4 tinha.

    python notebooks/build_notebooks.py
"""
from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)

# O %% vira % após a interpolação de RAIZ abaixo: a magic precisa dele escapado.
PREAMBULO = """\
%%matplotlib inline
import sys, warnings
sys.path.insert(0, %r)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hvac import results as R
from hvac import figures as F

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
F.aplicar_estilo()
""" % RAIZ


def md(txt):
    return nbf.v4.new_markdown_cell(txt.strip())


def code(txt):
    return nbf.v4.new_code_cell(txt.strip())


# ============================================================ notebook 01
nb1 = nbf.v4.new_notebook(cells=[
    md("""
# 01 — Auditoria da vantagem reivindicada

**Pergunta:** a superioridade do DQN sobre o baseline decorre do aprendizado, ou
da configuração do adversário?

O manuscrito auditado reporta 82,9 % de conforto contra 50,9 % de um termostato —
uma vantagem de aproximadamente +32 pontos percentuais. Este notebook decompõe
esse número, atribuindo cada incremento à sua causa.

Todos os controladores operam sob **condições idênticas**: mesma física, mesma
matriz de cenários, mesmo horizonte de decisão (*action repeat* 2) e mesmo espaço
de ação discreto. Sem essa equalização, a comparação mediria o protocolo, e não a
política.
"""),
    code(PREAMBULO),

    md("""
## 1.1 Comparação sob condições idênticas

O controlador PI foi sintonizado por busca em grade sobre a matriz de cenários
(K_p = 1,3; K_i = 0,2). Os agentes DQN vêm de 550 000 passos de treinamento.
"""),
    code("""
comp = R.comparacao_controladores()
comp.round(2)
"""),

    md("""
### Leitura

O PI **domina os três perfis DQN simultaneamente em qualidade e em custo**: mesmo
conforto na faixa larga (86,5 %), mesmo conforto na faixa estreita (83,8 %), menor
desvio absoluto do ideal e menor consumo. Duas constantes sintonizadas igualam ou
superam 550 000 passos de treinamento.

Repare também que os **três perfis empatam** no conforto binário. A métrica que o
manuscrito usa como principal está saturada: ela não separa perfis que diferem em
gradiente de conforto, penalidade de energia e penalidade de troca. É por isso
que a faixa estreita e o desvio absoluto são reportados ao lado — são as métricas
em que os perfis de fato se distinguem.
"""),

    md("""
## 1.2 Decomposição da vantagem

Cada linha acrescenta **um único fator** ao anterior, de modo que o incremento é
atribuível.
"""),
    code("""
dec = R.decomposicao_da_vantagem()
dec.round(2)
"""),
    code("""
fig = F.fig_decomposicao(dec)
plt.close(fig)   # evita a exibição dupla do backend inline
fig
"""),

    md("""
### Leitura — o achado central

| Etapa | Ganho | Atribuível a |
|---|---|---|
| Termostato do manuscrito | — | ponto de partida |
| Adicionar apenas histerese | **+23,7 pp** | configuração do baseline |
| PI sintonizado | **+8,8 pp** | controle clássico |
| DQN, 550k passos | **+0,0 pp** | **aprendizado** |

Praticamente **toda** a vantagem reivindicada é atribuível à configuração
inadequada do adversário. Um termostato com zona morta nula comuta ao menor
desvio e estaciona no teto da faixa; acrescentar histerese — uma linha de código,
prática padrão em qualquer termostato comercial — já recupera 23,7 dos 32 pontos.

A afirmação "82,9 % contra 50,9 %" é tecnicamente verdadeira e substantivamente
enganosa: compara-se contra um adversário artificialmente incapaz.

> **Ressalva metodológica, não eliminada.** O PI foi sintonizado sobre a mesma
> matriz usada para avaliar. O resultado deve ser lido como *"o PI iguala o DQN
> nas condições em que ambos foram ajustados"*, e não como superioridade geral.
> Refazer a sintonia apenas numa partição de calibração é trabalho pendente.
"""),

    md("""
## 1.3 Por que isso não é acidente

O ambiente é uma planta **monovariável de primeira ordem**, essencialmente linear
na faixa de operação, com **modelo conhecido** de dois parâmetros, perturbação
mensurável e objetivo de rastrear um setpoint. Esse é o caso canônico em que
controle proporcional-integral é ótimo ou quase-ótimo.

O aprendizado por reforço tem vantagem estabelecida em regimes que esta
formulação não exibe: não-linearidade forte, acoplamento de alta dimensão,
objetivos não expressáveis como custo quadrático, decisões combinatórias de
alocação, ou modelo desconhecido. O notebook 03 documenta três tentativas de
construir tais regimes.
"""),
])

# ============================================================ notebook 02
nb2 = nbf.v4.new_notebook(cells=[
    md("""
# 02 — Ablação da recompensa e a falha do anti-short-cycling

**Perguntas:**
1. A função de recompensa proposta é de fato responsável pelo desempenho?
2. A penalidade anti-short-cycling — destacada como contribuição — cumpre o seu
   objetivo?
3. Um método tabular bastaria, dado que o espaço de estados tem 4 dimensões?
"""),
    code(PREAMBULO),

    md("""
## 2.1 Ablação: 8 variantes × 3 sementes

Cada variante desativa **um único** mecanismo da recompensa, mantendo todo o
resto idêntico. A métrica é o conforto na faixa estreita, que discrimina regimes
onde a faixa larga satura.

**Sobre a estatística.** Com 3 sementes por grupo, o menor valor-p bicaudal
alcançável em teste de permutação é 2/C(6,3) = 0,10 — portanto p ≈ 0,101 é o
*piso*, não um resultado marginal. Por isso reportamos o δ de Cliff, tamanho de
efeito não paramétrico sem esse teto: δ = −1,00 indica **separação completa**
(toda semente ablacionada pior que toda semente da referência).
"""),
    code("""
abl = R.ablacao()
abl[["rotulo", "media", "desvio", "minimo", "maximo", "cliffs_delta", "magnitude"]].round(2)
"""),
    code("""
fig = F.fig_ablacao(abl, R.ablacao_por_semente())
plt.close(fig)
fig
"""),

    md("""
### Leitura 1 — a hipótese do platô plano é confirmada

Remover o gradiente interno degrada o conforto estreito de **83,7 % para 47,7 %**
(−36 pp), com separação completa entre sementes (δ = −1,00) e treinamento
instável (desvio 13,7). Um platô verdadeiramente plano não recompensa *entrar* na
faixa — o ganho marginal junto à borda é nulo —, e a política estaciona logo
acima do teto. Este é um resultado real e defensável.

### Leitura 2 — mas a formulação proposta **não supera** a convencional

A variante `convencional` (quadrática pura, sem gradiente interno, sem penalidade
de troca, sem anti-short-cycling, sem penalidade de frio) atinge **83,1 %** contra
83,7 % da proposta: δ = −0,11, **desprezível**.

O motivo é conceitual: uma quadrática pura é uma parábola com máximo em 24 °C —
ela **já possui gradiente em todo o domínio**, por construção. O platô plano é que
o destrói; a proposta apenas o *restaura*.

**Reenquadramento necessário:** a contribuição não é uma recompensa melhor que a
da literatura. É a correção de uma falha que os próprios autores introduziram ao
adotar o platô plano. O achado publicável é *negativo*.

### Leitura 3 — cuidado ao interpretar o δ de Cliff

`sem_penal_frio` e `quadratica_pura` têm δ = −0,56, que a convenção classifica
como *grande*. Mas a diferença de média é de apenas ~1,2 pp (83,7 → 82,5). Não há
contradição: **o δ de Cliff mede separação ordinal, não magnitude da diferença.**
Um deslocamento minúsculo e consistente entre grupos produz δ alto. Reportar
apenas o δ, sem a diferença de médias ao lado, exageraria o efeito — e é por isso
que a tabela traz as duas colunas.
"""),

    md("""
## 2.2 A penalidade anti-short-cycling não cumpre seu objetivo

O requisito declarado é d_min = 36 min de permanência mínima entre comutações. A
métrica agregada "comutações por hora" não permite verificá-lo: ela não distingue
"sempre respeita 36 min" de "metade das trocas viola gravemente".
"""),
    code("""
por_agente = R.permanencia_por_agente()
por_agente[["agente", "n_comutacoes", "mediana_min", "media_min",
            "min_min", "violacoes_pct"]].round(1)
"""),
    code("""
perm = R.permanencia()
perm["resumo"].round(1)
"""),
    code("""
fig = F.fig_permanencia(perm)
plt.close(fig)
fig
"""),

    md("""
### Leitura

A **mediana é 12 minutos** — uma única decisão — contra um requisito de 36. A
média de 36 a 60 minutos é inflada por longos períodos com o equipamento
desligado e **mascara completamente o problema**: é o caso didático de um agregado
que oculta a distribuição.

**Causa raiz, aritmética:** a penalidade da eq. 5 vale no máximo |ρ| = 5, contra
ganhos de conforto de até B + B_c = 14. Um termo de recompensa que pode ser
superado por outro termo não é uma proteção; é uma sugestão.

**Correção:** impondo d_min como restrição dura sobre o agente **já treinado**, sem
retreinamento, as violações caem de 71,4 % para 6,4 % e a mediana sobe de 12 para
48 minutos — mantendo o conforto binário inalterado. Os 6,4 % residuais são
artefato de medição na primeira comutação de cada episódio.

**Lição transferível:** propriedades de segurança e de integridade de hardware
devem ser impostas **estruturalmente**, não negociadas via função de recompensa.
"""),

    md("""
## 2.3 Se RL for utilizado, ele precisa ser profundo

Com apenas 4 dimensões de estado, um método tabular deveria bastar. Zha et al.
(2021) fornecem a razão teórica pela qual não basta: em espaços contínuos com
variáveis de **dinâmica lenta**, a transição permanece no mesmo hiper-tile de
origem, e o valor nunca se propaga entre células.
"""),
    code("""
tab = R.tabular()
tab.round(2)
"""),
    code("""
print(f"conforto larga : {tab['conf_larga_pct'].mean():.1f}% "
      f"(desvio {tab['conf_larga_pct'].std():.1f}, por semente: "
      f"{', '.join(f'{v:.1f}' for v in tab['conf_larga_pct'])})")
print(f"intra-tile pós-treino : {tab['intra_tile_treino_pct'].mean():.1f}%")
print(f"cobertura de estados  : {tab['cobertura_estados'].mean():.1f}%")
"""),

    md("""
### Leitura

| Grandeza | Valor |
|---|---|
| Espaço discretizado | 4 800 estados (20 × 10 × 24) |
| Largura do bin de temperatura | 1,00 °C |
| ΔT típico por passo (sala cheia) | 0,090 °C |
| Passos para cruzar um bin | **11,1** |
| Transições intra-tile (política aleatória) | **79,8 %** |
| Transições intra-tile (após treino) | 59,8 % |
| Cobertura de estados | 76,7 % |

Cerca de 80 % dos *backups* atualizam um estado com o próprio valor. O
Q-Learning tabular atinge **42,1 %** de conforto — **pior que o termostato de zona
morta** — e é instável entre sementes.

A queda de 80 % para 60 % após o treino ocorre porque a política aprende a usar
HIGH, que altera a temperatura rápido o bastante para cruzar bins. Ainda assim,
seis de cada dez atualizações permanecem inócuas, e ~1 100 dos 4 800 estados nunca
são visitados.

O obstáculo **não é a dimensionalidade**, e sim a razão entre o passo da dinâmica
(0,090 °C) e a resolução da discretização (1,00 °C). Este é o achado mais robusto
do conjunto: deriva inteiramente de parâmetros publicados.
"""),
])

# ============================================================ notebook 03
nb3 = nbf.v4.new_notebook(cells=[
    md("""
# 03 — Regimes estendidos: onde o RL poderia ganhar

Estabelecido que o controle clássico domina na formulação original, resta a
pergunta: **existe alguma extensão do problema em que o RL leve vantagem?**

Três regimes foram construídos e avaliados. Os três falharam, por razões
estruturalmente distintas — e a análise dessas razões é o resultado mais útil
deste notebook.
"""),
    code(PREAMBULO),

    md("""
## 3.1 Rastreamento de precisão

Um laboratório com tolerância de ±0,5 °C em torno do setpoint, equipamento
reversível (aquece e resfria), ação contínua e observação enriquecida com erro
escalado pela tolerância, integral com fuga e derivada. Agentes TD3 e SAC
treinados sob três perfis de custo.
"""),
    code("""
lab = R.laboratorio_precisao()
lab.round(3)
"""),
    code("""
fig = F.fig_precisao(lab)
plt.close(fig)
fig
"""),

    md("""
### Leitura

O **PI bidirecional permanece superior**: 100 % do tempo dentro da tolerância com
σ = 0,088 °C e R$ 14,76/dia, contra σ = 0,198 °C e R$ 16,44/dia do melhor TD3.
Ele ocupa sozinho o canto ótimo — mais barato *e* mais estável.

Tornar o problema **mais difícil** não mudou a **classe** do problema. Um problema
de rastreamento difícil continua sendo rastreamento, e responde a uma sintonia
mais agressiva do controlador clássico.
"""),

    md("""
## 3.2 Antecipação tarifária

Sob tarifa branca real (razão de ponta 2,21×), a oportunidade de pré-resfriamento
é limitada pela própria tolerância: com ±0,5 °C, o armazenamento térmico
disponível cobre apenas ~29 % da duração do posto de ponta. A margem para
antecipação é estruturalmente pequena — o "reservatório" é raso demais.
"""),

    md("""
## 3.3 Demanda contratada — e uma auditoria do próprio experimento

A hipótese mais promissora: uma restrição sobre o **máximo** da média integrada em
15 min, e não sobre a média. Um PI minimiza erro instantâneo e não tem como
representar "não ultrapasse X kW em nenhum momento"; respeitá-la exigiria
**antecipar** carga.

Uma versão preliminar fixava a demanda contratada em **0,70 kW**, valor escolhido
por varredura como "ponto de máxima tensão" entre conforto e restrição. Aplicando
a esse valor a mesma exigência que este trabalho aplica ao manuscrito auditado,
constatou-se que ele era **infactível**.
"""),
    code("""
dem = R.dimensionamento_demanda()
print(dem["sizing"].as_table())
"""),

    md("""
O regime permanente de pior caso — 45 ocupantes, externa a 36 °C, mantendo o
setpoint — exige **1,204 kW**. Um contrato de 0,70 kW está **72 % abaixo** disso e
sustenta apenas ~17 dos 45 ocupantes. **Nenhuma política respeita tal restrição**,
porque antecipar não cria regime permanente: pré-resfriar compra um transitório
limitado pela tolerância, não potência contínua.

O valor foi substituído por dimensionamento **derivado da condição de projeto**
(regime de pior caso + margem de contratação de 10 %), resultando em 1,324 kW.
"""),
    code("""
dem["comparacao"].round(3)
"""),

    md("""
### Leitura — a hipótese perde sua base

Sob o contrato **arbitrado** (0,70 kW), o PI ingênuo viola em 7 de 9 cenários e o
PI ciente da restrição não viola: parecia haver um dilema real.

Sob o contrato **derivado** (1,324 kW), **nenhum** dos nove cenários excede o
limite, e os dois PIs tornam-se indistinguíveis. Não há o que antecipar.

O "dilema" era artefato de uma restrição impossível, e não evidência de que
antecipação fosse necessária.
"""),
    code("""
fig = F.fig_demanda(dem)
plt.close(fig)
fig
"""),
    code("""
dem["origem_da_pressao"]
"""),

    md("""
### Leitura — a contradição estrutural

Os únicos cenários que excedem o contrato derivado são os que **partem fora do
setpoint** (17 °C exige aquecimento pleno, 2,198 kW; 30 °C exige pulldown,
2,931 kW). Neles a violação ocorre no **passo inicial** — instante em que
antecipação é impossível por definição, pois não há como pré-resfriar antes de
t = 0.

Ao construir a matriz de cenários partindo do setpoint — decisão correta, tomada
justamente para eliminar essa violação impossível —, elimina-se **simultaneamente
a única fonte de demanda acima do regime permanente**.

**O conjunto no qual a hipótese pode ser testada é vazio.**
"""),

    md("""
## 3.4 Síntese: por que as três tentativas falharam

| Tentativa | Resultado | Razão |
|---|---|---|
| Precisão (±0,5 °C) | PI vence: σ 0,088 vs 0,198 | rastreamento mais apertado ainda é rastreamento |
| Antecipação tarifária | oportunidade ~29 % da ponta | tolerância limita o armazenamento térmico |
| Demanda contratada | conjunto de teste vazio | pressão só existe onde antecipar é impossível |

As três introduziram **dificuldade**, não **estrutura**. Tornar o problema mais
difícil não muda a classe a que ele pertence.

### A limitação de método que fecha o argumento

O nicho do RL é o **descasamento de modelo** — situações em que o modelo é
desconhecido, impreciso ou variável. Num simulador de autoria própria, o modelo é
**exato por construção**.

Qualquer comparação conduzida integralmente dentro do próprio ambiente favorece
estruturalmente os métodos que exploram o modelo, incluindo controle preditivo.
**Este ambiente não pode, em princípio, exibir o regime em que o RL tem
vantagem.** Não é limitação da implementação; é limitação do método de
investigação.

Testar a hipótese de verdade exige transferência simulação-realidade ou benchmark
de terceiros — BOPTEST (Boutahri e Tilioua, 2025) ou BuildingGym (Dai et al.,
2025).
"""),
])


# ============================================================ notebook 04
nb4 = nbf.v4.new_notebook(cells=[
    md("""
# 04 — Trajetórias, e a faixa estreita muda o vencedor?

Duas perguntas:

1. **Como cada controlador se comporta ao longo do dia**, cenário a cenário? Quem
   de fato mantém a temperatura na faixa, e por qual mecanismo?
2. **Estreitar a faixa alvo favorece o RL?** A intuição é que, quanto mais
   exigente a especificação, mais o aprendizado compensaria.

A segunda é testada com retreinamento, não com re-medição: cobrar ±0,5 °C de um
agente treinado para ±2,0 °C mediria descasamento de objetivo, não capacidade.
"""),
    code(PREAMBULO),

    md("""
## 4.1 O dia inteiro, cenário a cenário

Pequenos múltiplos: a comparação que interessa é entre controladores **dentro**
de cada cenário. A faixa alvo aparece sombreada como referência visual comum.
"""),
    code("""
tr = R.trajetorias()
fig = F.fig_trajetorias_grade(tr)
plt.close(fig)
fig
"""),

    md("""
### Leitura

O padrão é imediato e consistente nos oito cenários controláveis: **o termostato
estaciona acima da faixa**, encostado no teto de 26 °C, enquanto PI e DQN ficam
quase sobrepostos em torno de 24 °C.

C1 (Frio + Poucas) é o cenário em que as três curvas coincidem — é exatamente o
excluído pelo filtro de controlabilidade: sem ninguém na sala e partindo de
17 °C, a temperatura sobe sozinha até a faixa e resfriar seria contraprodutivo.
Nenhum controlador se distingue ali, e por isso ele não entra nas médias.
"""),

    md("""
## 4.2 Um dia em detalhe — o mecanismo

A trajetória mostra **quanto** cada um acerta; a ação mostra **como**.
"""),
    code("""
fig = F.fig_trajetoria_detalhe(tr, "C6")
plt.close(fig)
fig
"""),

    md("""
### Leitura

O painel inferior explica o superior. O **termostato** liga em MEDIUM ao cruzar
26 °C e desliga ao voltar — daí o dente-de-serra permanentemente acima da faixa.

O **DQN** comuta entre OFF e HIGH de forma agressiva, com pulsos curtos e
repetidos: é o *short-cycling* visível, o mesmo fenômeno que a distribuição de
permanência quantifica (mediana de 12 min contra requisito de 36).

O **PI** modula: usa LOW e MEDIUM de forma sustentada e recorre a HIGH raramente.
Mesmo resultado de temperatura, mecanismo mais suave — e, como a Seção 4.6
mostra, mais barato.
"""),

    md("""
## 4.3 Apertando a régua, sem retreinar

Primeiro a versão barata da pergunta: mantendo os mesmos controladores, o que
acontece quando se exige mais precisão? Aqui o DQN está em desvantagem
declarada — foi treinado para ±2,0 °C.
"""),
    code("""
sens = R.sensibilidade_a_largura()
sens.pivot(index="tolerancia", columns="controlador",
           values="na_tolerancia_pct").round(1)
"""),
    code("""
fig = F.fig_sensibilidade_largura(sens)
plt.close(fig)
fig
"""),

    md("""
### Leitura

Em ±2,0 °C, PI e DQN empatam em 86,5 %. À medida que a régua aperta, eles **se
separam**, e a distância cresce monotonicamente: em ±0,25 °C o PI mantém 73,2 %
contra 34,0 % do DQN — mais que o dobro.

Os termostatos colapsam bem antes: de 54,0 % para 1,7 % o de zona morta nula.
Isso confirma que a métrica de faixa larga do manuscrito estava saturada e
escondia a diferença real entre as estratégias.

Mas este teste é injusto com o DQN, e por isso não conclui nada sozinho.
"""),

    md("""
## 4.4 O teste justo: ambos preparados para a faixa

Para cada largura, **dois** controladores são preparados para *aquela* largura:
o DQN é retreinado com a faixa de conforto correspondente, e o PI é
**re-sintonizado** por busca em grade.

Re-sintonizar o PI é a parte não negociável. Congelar os ganhos de ±2,0 °C e
cobrar precisão de ±0,5 °C reproduziria, dentro deste experimento, o mesmo
defeito que a auditoria acusa no manuscrito: vencer um adversário mal
configurado.
"""),
    code("""
fx = R.faixa_estreita()
fx.groupby(["tolerancia", "controlador"])["na_tolerancia_pct"].agg(
    ["mean", "std", "min", "max"]).round(2)
"""),
    code("""
fig = F.fig_faixa_estreita(fx)
plt.close(fig)
fig
"""),

    md("""
### Leitura — a hipótese é refutada, e o efeito é o inverso

| Faixa alvo | PI | DQN (3 sementes) | Vantagem do PI |
|---|---|---|---|
| ±2,0 °C | 86,5 % | 86,5 % | **+0,0 pp** |
| ±1,0 °C | 83,8 % | 80,8 % | **+3,0 pp** |
| ±0,5 °C | 79,8 % | 71,5 % | **+8,3 pp** |

Estreitar a faixa **amplia** a vantagem do controle clássico, monotonicamente —
o oposto da intuição. E há um segundo efeito: o desvio entre sementes do DQN
cresce de 0,00 para 5,20 e 4,04. Quando a especificação aperta, o treinamento não
só entrega menos, como fica **instável**.
"""),

    md("""
## 4.5 "Mas treinou o suficiente?"

É a objeção óbvia, e nenhum experimento com orçamento fixo a responde. Seguindo
as Figs. 9-10 de Yuan et al. — que traçam custo e desconforto ano a ano —,
avaliamos o agente periodicamente **durante** o treino, contra a linha do PI.

O PI aparece como reta horizontal por construção: ele não aprende.
"""),
    code("""
curva = R.curva_aprendizado()
curva.groupby(["tolerancia", "controlador", "passos"])["na_tolerancia_pct"].mean(
    ).unstack(level=0).tail(8).round(1)
"""),
    code("""
fig = F.fig_curva_aprendizado(curva)
plt.close(fig)
fig
"""),

    md("""
### Leitura

Em **±2,0 °C** o DQN sai de 14 %, cruza rapidamente entre 100k e 250k passos e
**satura exatamente sobre a linha do PI** (86,5 % nas três sementes). O empate já
reportado não é coincidência de um ponto de parada: é o patamar de convergência.

Em **±0,5 °C** o quadro é outro. A curva sobe de 4 % para ~63 % e a 400k passos
está **16,9 pp abaixo** do PI. O crescimento desacelera muito — de 4 → 57 entre
25k e 250k, contra 57 → 63 nos 150k seguintes.

**Ressalva honesta:** a curva desacelera fortemente, mas **não é plana**. A
tendência nos últimos 150k passos ainda é de +2,9 pp. Mantida essa taxa — hipótese
otimista, que o próprio achatamento contradiz —, seriam necessários cerca de
**870k passos adicionais** para fechar a diferença. Portanto o que estes dados
sustentam não é "o DQN converge para pior", e sim que **o custo de amostra para
alcançar o PI é alto e cresce quando a faixa aperta**. A afirmação mais forte
exigiria um orçamento maior.

Note ainda a dispersão ao final: 73,3 / **43,7** / 71,8. Uma das três sementes
colapsa — a instabilidade da Seção 4.4 aparece aqui como trajetória, não só como
desvio-padrão.

O paralelo com Yuan et al. é direto: eles reportam o RL superando o PID apenas
após **dois anos de exploração mais dois anos de buffer**, com melhor desempenho
no sétimo ano. Alto custo de amostra neste domínio é um achado da literatura, não
uma particularidade desta implementação.
"""),

    md("""
## 4.6 De onde vem a diferença de energia

As tabelas mostram que o DQN gasta ~4,7 % mais que o PI, mas não mostram **em
quê**. Seguindo a Fig. 11 de Yuan et al., que decompõe o consumo por item do
sistema, decompomos por **nível de potência acionado**.
"""),
    code("""
from hvac.config import config_for_profile
cop = {k.name: v for k, v in config_for_profile("Equilibrado").physics.cop.items()}
dec = R.consumo_decomposto()
dec["por_nivel"].pivot(index="controlador", columns="nivel",
                       values="kwh_dia").round(2)
"""),
    code("""
fig = F.fig_consumo_decomposto(dec, cop=cop)
plt.close(fig)
fig
"""),

    md("""
### Leitura — o mecanismo do desperdício

| Controlador | OFF | LOW (COP 3,45) | MEDIUM (COP 3,59) | HIGH (COP 3,00) |
|---|---|---|---|---|
| PI | 55,4 % | 26,4 % | **13,5 %** | 4,7 % |
| DQN | 54,0 % | 36,6 % | **0,0 %** | 9,5 % |

**O DQN nunca usa MEDIUM** — precisamente o nível de maior COP do equipamento — e
compensa com o dobro de HIGH, o de pior COP. A energia extra não vem de operar
mais tempo; vem de operar nos níveis errados.

A causa é estrutural: uma política *greedy* determinística escolhe o argmax dos
Q-values. Uma ação que nunca seja o argmax **desaparece por completo** da
política, mesmo que seja quase ótima em muitos estados. Um controlador
proporcional, ao contrário, atravessa naturalmente todos os níveis ao percorrer a
faixa de erro.

Isso liga os dois achados: descartar o nível intermediário obriga a alternar
entre extremos, que é exatamente o *short-cycling* observado no painel de ações
da Seção 4.2.
"""),

    md("""
## 4.7 Uma hipótese sobre a causa — testada e refutada

A degradação em faixa estreita sugeria uma explicação: PI e DQN compartilham o
mesmo espaço de ação, mas o PI tem **integrador**, e a observação do manuscrito é
puramente reativa (temperatura, ocupação, hora). Sem integrar o erro, não se
elimina offset — e com ±0,5 °C o offset passa a ser a diferença entre estar
dentro ou fora.

Hipótese testável: dar ao agente o erro escalado e o erro integral deveria
recuperar parte da distância.
"""),
    code("""
ti = R.teste_integral()
ti.groupby(["variante", "n_canais"])["na_tolerancia_pct"].agg(
    ["mean", "std", "min", "max"]).round(2)
"""),

    md("""
### Leitura — resultado negativo

| Observação | Na tolerância |
|---|---|
| Do manuscrito (4 canais) | **71,5 %** |
| + erro escalado e integral (6 canais) | **65,9 %** |

Acrescentar informação **piorou**. A hipótese não se sustenta: com o mesmo
orçamento de 300k passos, os canais extras ampliam o que há para aprender sem
compensar em desempenho.

Fica registrado como resultado negativo. A degradação do DQN em faixa estreita
tem mecanismo parcialmente explicado — o descarte do nível MEDIUM, Seção 4.6 —
mas a contribuição do estado insuficiente **não** foi demonstrada, e seria
desonesto apresentá-la como se tivesse sido.
"""),

    md("""
## 4.8 Síntese

1. **Trajetórias.** O termostato estaciona acima da faixa; PI e DQN a mantêm. A
   diferença entre os dois não está na temperatura, está na **ação**: o PI modula,
   o DQN alterna entre extremos.
2. **Estreitar a faixa não favorece o RL** — favorece o PI, e de forma crescente:
   +0,0 → +3,0 → +8,3 pp. Com ambos preparados para cada largura.
3. **Não é simplesmente falta de treino.** Em ±2,0 °C o DQN converge sobre a
   linha do PI; em ±0,5 °C fica 16,9 pp abaixo com a curva já bastante achatada.
   O que se afirma é o custo de amostra elevado, não a impossibilidade — a
   distinção importa e está registrada na Seção 4.5.
4. **A energia extra tem mecanismo identificado**: a política aprendida descarta o
   nível de melhor eficiência do equipamento.
5. **Uma hipótese explicativa foi testada e refutada** — dar estado suficiente ao
   agente não recuperou o desempenho.

O conjunto reforça a conclusão dos notebooks anteriores, agora com mecanismo: em
rastreamento de setpoint com modelo conhecido, o controle clássico não é apenas
competitivo — ele é estruturalmente mais adequado, e a vantagem **cresce** com a
exigência de precisão.
"""),
])


# ============================================================ notebook 05
nb5 = nbf.v4.new_notebook(cells=[
    md("""
# 05 — O nível MEDIUM descartado, e quem mantém melhor a faixa

Quatro perguntas:

1. O DQN **não usar MEDIUM** é um problema, do ponto de vista de custo-benefício?
2. A causa está na **recompensa**? Valeria ajustar penalidades?
3. Em percentual, **qual controlador mantém melhor a temperatura na faixa**?
4. Num conjunto **amplo e aleatório** de cenários, **onde** os modelos divergem?

Os três perfis do manuscrito — Agressivo, Equilibrado e Passivo — são avaliados
lado a lado, porque é a variação entre eles que separa "causa na recompensa" de
"causa no algoritmo".
"""),
    code(PREAMBULO),

    md("""
## 5.1 Os três perfis descartam MEDIUM

Os perfis diferem bastante nos pesos: o gradiente de conforto vai de 7,0 a 3,0, a
penalidade de energia de 0,03 a 0,12 (**4×**), a de troca de −0,5 a −1,5 (**3×**).
Se a recompensa fosse a causa, esperaríamos comportamentos distintos.
"""),
    code("""
from hvac.config import config_for_profile, REWARD_PROFILES
cop = {k.name: v for k, v in config_for_profile("Equilibrado").physics.cop.items()}
niveis = R.uso_dos_niveis()
niveis.round(1)
"""),
    code("""
fig = F.fig_uso_dos_niveis(niveis, cop=cop)
plt.close(fig)
fig
"""),

    md("""
### Leitura

**Os três perfis usam MEDIUM 0,0 % do tempo**, apesar de pesos que variam por
fatores de 3× a 4×. O Agressivo é ainda mais extremo: descarta LOW **e** MEDIUM,
operando só em OFF (83,4 %) e HIGH (16,6 %) — política puramente liga-desliga.

Já o **SAC**, treinado com a **mesma** função de recompensa do DQN Equilibrado,
usa MEDIUM 13,3 % do tempo — praticamente igual ao PI (12,0 %).

Isso é um controle experimental limpo: mesma recompensa, comportamentos opostos.
A causa **não é a recompensa**; é a política discreta de argmax.
"""),

    md("""
## 5.2 É um problema de custo-benefício? Sim.

MEDIUM é o nível de **melhor COP** do equipamento (3,59, contra 3,45 do LOW e
3,00 do HIGH) — o modelo físico reproduz o pico de eficiência em carga parcial,
comportamento de equipamento *inverter*. Descartá-lo significa operar nos níveis
menos eficientes.
"""),
    code("""
dec = R.consumo_decomposto()
dec["por_nivel"].pivot(index="controlador", columns="nivel",
                       values="kwh_dia").round(2)
"""),
    code("""
fig = F.fig_consumo_decomposto(dec, cop=cop)
plt.close(fig)
fig
"""),

    md("""
### Leitura

O DQN Equilibrado gasta 10,90 kWh/dia contra 10,40 do PI — **4,7 % a mais** para
entregar o mesmo conforto. E a energia extra não vem de operar mais tempo (ambos
ficam ~55 % em OFF): vem de **operar nos níveis errados**, com o dobro de HIGH.

Portanto sim, é um problema de custo-benefício — e é o único mecanismo
identificado que explica quantitativamente a diferença de energia entre os dois.
"""),

    md("""
## 5.3 Vale ajustar a recompensa? O que os Q-values dizem

Se MEDIUM fosse fortemente dominado, ajustar pesos não adiantaria. Se estivesse
em segundo lugar por pouco, um ajuste pequeno o traria de volta. A resposta exige
olhar a **escala** dos Q-values, e não só o ranking.
"""),
    code("""
R.analise_q_values().round(2)
"""),

    md("""
### Leitura — cuidado com a escala

MEDIUM fica em **terceiro lugar** no ranking (posição média 3,2 de 4) nos três
perfis, e nunca é o argmax.

Mas o déficit precisa ser lido na escala certa. O Q-value absoluto é da ordem de
**980**; a faixa inteira entre as quatro ações vale **~1,9 %** desse valor, e o
déficit de MEDIUM para o topo é de **~1,2 %**.

> Dizer que "MEDIUM perde 63 % da faixa de Q" seria tecnicamente verdadeiro e
> substantivamente enganoso — sugeriria uma ação ruim, quando as quatro são
> **quase equivalentes em valor**.

O mecanismo real é *winner-take-all*: **uma política greedy determinística
converte uma margem de ~1 % em uso de 0 %.** Não existe "usar MEDIUM às vezes"
sob argmax — ou a ação é a melhor num estado, ou não aparece nunca.

**Consequência prática:** mexer nos pesos da recompensa é o caminho errado. Os
três perfis já cobrem uma variação de 3–4× e todos colapsam igual. O que restaura
o uso dos níveis intermediários é mudar a **classe de política** — estocástica ou
contínua, como o SAC demonstra —, não reponderar termos.
"""),

    md("""
## 5.4 Quem mantém melhor a faixa? Conjunto amplo e independente

A matriz 3×3 tem nove pontos e foi usada para sintonizar o PI. Empregam-se
aqui **150 cenários aleatórios** sobre o espaço contínuo (temperatura inicial 16–33 °C,
ocupação 0–45, hora 0–23), com semente fixa distinta das de treino. Todos os
controladores veem **exatamente os mesmos** cenários — comparação emparelhada.

Para o PI este é um teste **fora da amostra**: ele foi sintonizado noutro
conjunto.
"""),
    code("""
al = R.cenarios_aleatorios_cache()
al.groupby("controlador").agg(
    conf_larga=("conf_larga_pct", "mean"),
    conf_estreita=("conf_estreita_pct", "mean"),
    na_tolerancia=("na_tolerancia_pct", "mean"),
    desvio=("desvio_ideal", "mean"),
    kwh_dia=("energia_kwh", "mean"),
).sort_values("na_tolerancia", ascending=False).round(2)
"""),

    md("""
### Leitura — a resposta depende de qual faixa se pergunta

| Critério | Melhor | Valor |
|---|---|---|
| Faixa larga [22, 26] °C | **empate** entre PI e os 3 DQN | 87,88 % |
| Faixa estreita [23, 25] °C | empate PI / DQN Agressivo | 83,28 % |
| Tolerância ±0,5 °C | DQN Agressivo, por 0,44 pp | 79,62 % vs 79,18 % |
| Desvio médio \\|T−24\\| | **PI** | 0,62 °C |
| Energia | **PI** (entre os que controlam) | 10,77 kWh/dia |

Na faixa larga os quatro dão **exatamente 87,88 %** — a métrica está saturada, e
o valor é determinado pelo transitório de *pulldown*, que é limitado pela física
e não pelo controlador. Ela não discrimina nada.

Na tolerância de ±0,5 °C aparece separação real, e ali o **DQN Agressivo alcança
o PI** (79,62 % contra 79,18 %). Este é um resultado favorável ao RL, e mais
honesto do que a matriz 3×3 sugeria — mas veja o custo na Seção 5.6.

O PI **se sustenta fora da amostra**, o que atenua (sem eliminar) a ressalva de
que ele havia sido sintonizado no mesmo conjunto de avaliação.
"""),

    md("""
## 5.5 Onde os modelos divergem

A média esconde a distribuição. Agrupando por condição do cenário:
"""),
    code("""
fig = F.fig_divergencia_por_condicao(al)
plt.close(fig)
fig
"""),
    code("""
cond = al.drop_duplicates("cenario").set_index("cenario")[["start_temp","occupancy","hour"]]
piv = al.pivot(index="cenario", columns="controlador",
               values="na_tolerancia_pct").join(cond)
piv["dif"] = piv["DQN Equilibrado"] - piv["PI sintonizado"]
piv.nsmallest(6, "dif")[["start_temp", "occupancy", "hour",
                         "PI sintonizado", "DQN Equilibrado", "dif"]].round(1)
"""),

    md("""
### Leitura — a falha é concentrada, não difusa

O déficit dos perfis Equilibrado e Passivo vive quase todo em **salas vazias**:
−21,7 pp e −18,2 pp com 0–10 ocupantes, contra −4,4 pp e −3,1 pp com a sala
cheia.

Os piores cenários têm assinatura comum: **ocupação 0–4 pessoas, madrugada
(2h–6h)**. Ali o PI atinge 100 % dentro da tolerância e o DQN cai para 37–58 %.

**Mecanismo:** com a sala vazia de madrugada, a carga térmica é mínima, e manter
±0,5 °C exige potência muito baixa e finamente dosada. O menor nível não-nulo
disponível (LOW, 25 % da capacidade) já é excessivo — então o agente oscila entre
resfriar demais e deixar subir. O PI resolve alternando com o ciclo certo, obtendo
uma média efetiva menor que qualquer nível isolado.

O DQN Agressivo **não** sofre disso (+1,7 pp em sala vazia), porque opera em
liga-desliga puro e, sem penalidade de energia relevante, aciona HIGH sem
hesitação — mas paga na conta de luz.
"""),

    md("""
## 5.6 A fronteira de Pareto: cada perfil falha de um jeito
"""),
    code("""
fig = F.fig_pareto_aleatorios(al)
plt.close(fig)
fig
"""),
    code("""
g = al.groupby("controlador").agg(tol=("na_tolerancia_pct","mean"),
                                  kwh=("energia_kwh","mean"))
pi = g.loc["PI sintonizado"]
comp = g.loc[["DQN Agressivo","DQN Equilibrado","DQN Passivo","SAC Equilibrado"]].copy()
comp["dif_tolerancia_pp"] = comp["tol"] - pi["tol"]
comp["dif_energia_pct"] = (comp["kwh"] / pi["kwh"] - 1) * 100
comp.round(2)
"""),

    md("""
### Leitura — nenhum perfil domina o PI

| Perfil | Conforto vs PI | Energia vs PI | Diagnóstico |
|---|---|---|---|
| DQN Agressivo | **+0,4 pp** | **+12,9 %** | compra conforto com energia |
| DQN Equilibrado | −6,3 pp | +2,7 % | perde conforto sem economizar |
| DQN Passivo | −7,5 pp | +0,7 % | perde conforto sem economizar |
| SAC Equilibrado | −43,6 pp | +10,5 % | pior nos dois eixos |

Esta é a assinatura de estar **abaixo da fronteira de Pareto**: para igualar o PI
em conforto é preciso gastar 12,9 % mais energia; para igualar em energia é
preciso perder 6–7 pontos de conforto. O PI ocupa sozinho o canto ótimo.

E note que os três perfis foram obtidos **variando apenas pesos da recompensa** —
que era a tese central do manuscrito. A variação move o agente ao longo de uma
curva que passa **inteiramente por dentro** da fronteira, sem tocá-la.
"""),

    md("""
## 5.8 Velocidade de resposta: quanto tempo até a sala ficar utilizável

Todas as métricas até aqui medem **qualidade em regime** — fração do tempo dentro
da faixa, desvio, energia. Nenhuma media **velocidade**, que é requisito
operacional real: uma sala que leva quatro horas para ficar utilizável é um
problema mesmo que depois se mantenha perfeita.

Esta seção também testa uma afirmação feita anteriormente **sem medição**: a de
que o conforto idêntico entre controladores (Seção 5.4) decorre de um transitório
limitado pela física. Se os tempos diferirem, aquela explicação está errada.

Reportamos três grandezas distintas — entrar na faixa, parar de sair dela, e
quanto se passou do outro lado — sobre os cenários que **começam fora** da faixa.
"""),
    code("""
tr = R.resposta_transitoria()
tr.groupby("controlador").agg(
    entrada_h=("entrada_h", "mean"), desvio=("entrada_h", "std"),
    acomodacao_h=("acomodacao_h", "mean"), taxa_c_h=("taxa_c_por_h", "mean"),
    saturacao_pct=("saturacao_pct", "mean"),
    kwh_transitorio=("energia_transitorio_kwh", "mean"),
).sort_values("entrada_h").round(2)
"""),
    code("""
fig = F.fig_transitorio(tr)
plt.close(fig)
fig
"""),

    md("""
### Leitura — a explicação anterior se confirma

| Controlador | Entrar | Acomodar | Potência máxima |
|---|---|---|---|
| PI e os três DQN | **1,27 h** | **1,27 h** | **100 %** |
| Termostato (ambos) | 1,99 h | 19–21 h | 28 % |
| SAC Equilibrado | 3,91 h | 4,73 h | 31 % |

**PI e os três perfis DQN têm tempo idêntico — divergem em 0 de 48 cenários.** O
motivo aparece na última coluna: todos saturam o atuador, operando em potência
máxima durante 100 % do transitório. No *pulldown* não existe decisão a tomar; a
única ação sensata é potência total, e a velocidade fica limitada pela física do
equipamento. Isso confirma a explicação dada na Seção 5.4 para o conforto
idêntico, que até aqui era inferência.

O **termostato** revela seu defeito na distância entre as duas barras: entra na
faixa em 1,99 h, mas leva cerca de **20 horas** para parar de sair dela. Ele
estaciona no teto e oscila em torno de 26 °C — entra e sai repetidamente. A
métrica de primeira entrada o favoreceria indevidamente; a de acomodação expõe o
comportamento.

O **SAC** é **3,1× mais lento** que o PI, e a causa é a mesma coluna: satura o
atuador em apenas 31 % do transitório. Ele hesita onde não há razão para hesitar.

### Uma observação sobre o conjunto de métricas

Esta é a **terceira métrica saturada** encontrada: conforto na faixa larga,
conforto na faixa estreita sobre cenários aleatórios, e agora velocidade de
resposta. Em todas, PI e DQN produzem valores indistinguíveis.

Isso não é coincidência estatística — é a assinatura de um problema com pouco
espaço para diferenciação. Fora do regime permanente, a política ótima é trivial
(potência máxima) e qualquer controlador competente a encontra. A diferenciação
existe apenas no regime permanente, e ali quem decide é a resolução da atuação —
onde o agente discreto perde, pelas razões da Seção 5.3.
"""),
    md("""
## 5.9 Síntese

1. **O descarte de MEDIUM é real e custa dinheiro.** Os três perfis o descartam
   completamente; o nível tem o melhor COP; a energia extra do DQN vem daí.
2. **Não adianta ajustar a recompensa.** Os perfis já variam os pesos por 3–4× e
   colapsam igual. As quatro ações têm Q-values dentro de ~2 %, e o argmax
   converte ~1 % de margem em 0 % de uso.
3. **O que restaura os níveis intermediários é a classe de política.** O SAC, com
   a mesma recompensa, usa MEDIUM como o PI. Política contínua ou estocástica —
   não reponderação.
4. **Na faixa larga ninguém se distingue** (87,88 % para todos): a métrica é
   limitada pela física do *pulldown*. Só a tolerância estreita discrimina.
5. **A falha é concentrada em salas vazias de madrugada**, onde o menor nível
   disponível já é potência demais.
6. **Nenhum perfil domina o PI** — cada um erra por um eixo diferente.
7. **Velocidade de resposta não discrimina**: PI e DQN entram na faixa em tempo
   idêntico, porque ambos saturam o atuador. O termostato entra rápido mas leva
   ~20 h para parar de sair; o SAC é 3,1× mais lento por hesitar.

**Recomendação prática:** se o objetivo for aplicar RL neste equipamento, o ganho
mais direto não está na recompensa, e sim em dar ao agente **autoridade de
atuação mais fina** — ação contínua, ou modulação por ciclo de trabalho entre
níveis. Sem isso, a política aprendida fica presa aos extremos e paga a diferença
em energia.
"""),
])


def gerar(nb, nome):
    caminho = os.path.join(AQUI, nome)
    nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                 "language": "python", "name": "python3"}
    print(f"executando {nome} ...", flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": AQUI}}).execute()
    with open(caminho, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    n_err = sum(1 for c in nb.cells for o in c.get("outputs", [])
                if o.get("output_type") == "error")
    print(f"  -> {caminho}  (células: {len(nb.cells)}, erros: {n_err})")
    return n_err


if __name__ == "__main__":
    erros = 0
    erros += gerar(nb1, "01_auditoria_baselines.ipynb")
    erros += gerar(nb2, "02_ablacao_e_short_cycling.ipynb")
    erros += gerar(nb3, "03_regimes_estendidos.ipynb")
    erros += gerar(nb4, "04_trajetorias_e_faixa_estreita.ipynb")
    erros += gerar(nb5, "05_niveis_de_potencia_e_cenarios_aleatorios.ipynb")
    print("\nTOTAL DE ERROS:", erros)
    sys.exit(1 if erros else 0)
