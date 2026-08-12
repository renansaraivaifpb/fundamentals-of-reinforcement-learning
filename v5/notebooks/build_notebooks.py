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
    print("\nTOTAL DE ERROS:", erros)
    sys.exit(1 if erros else 0)
