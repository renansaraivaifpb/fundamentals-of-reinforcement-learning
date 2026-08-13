# Controle de HVAC por Aprendizado por Reforço — auditoria de reprodutibilidade

Este repositório começou como uma implementação de RL para climatização de salas
de aula e se tornou outra coisa: **uma auditoria do próprio trabalho**, que
acabou refutando a afirmação central do manuscrito que lhe deu origem.

O resultado principal é negativo e está medido:

> A vantagem de **+32 pontos percentuais** que o manuscrito reivindica sobre o
> baseline é praticamente toda atribuível à **configuração inadequada do
> adversário**, e não ao aprendizado. Um controlador PI com duas constantes
> iguala o DQN em conforto e o supera em desvio do setpoint e em custo.

| Etapa | Conforto [22,26] °C | Ganho | Atribuível a |
|---|---|---|---|
| Termostato do manuscrito (zona morta = 0) | 54,0 % | — | — |
| Adicionar apenas histerese | 77,7 % | **+23,7 pp** | configuração do baseline |
| PI sintonizado (Kp=1,3; Ki=0,2) | 86,5 % | **+8,8 pp** | controle clássico |
| DQN, 550k passos | 86,5 % | **+0,0 pp** | **aprendizado** |

---

## Por onde começar

| Se você quer… | Vá para |
|---|---|
| entender os achados, com tabelas e gráficos | `v5/notebooks/` (5 cadernos executados) |
| o artigo pronto | `v5/paper_auditoria_hvac_rl.docx` (e a versão curta) |
| o código atual | `v5/hvac/` |
| a reprodução fiel do manuscrito | `v4/paper/` |
| a resposta ponto a ponto aos pareceres | `v4/paper/REVISAO.md` |

```bash
cd v5
pip install -e .
python -m pytest tests/ -q          # 18 testes
python notebooks/build_notebooks.py # regera os cadernos
python gerar_paper.py               # regera o artigo
```

## Estrutura do repositório

| Diretório | Papel | Estado |
|---|---|---|
| raiz, `v2/`, `v3/` | primeiras iterações (Q-Learning tabular, DQN) | **histórico** |
| `v4/` | ambiente pré-artigo | **congelado — contém bugs conhecidos** |
| `v4/paper/` | reprodução fiel do manuscrito + experimentos | referência |
| **`v5/`** | **código atual**, notebooks, artigo | **ativo** |
| `papers_sugeridos/` | referências da literatura | — |

> **Atenção:** `v4/classroom_ac_env_v4.py` contém três bugs documentados
> (hora dos cenários descartada, observação fora do espaço declarado,
> descasamento treino/avaliação) que `v4/paper/` corrigiu. Ele ainda é importado
> por `v4/main_train_v4.1.py`. Não use esses scripts para produzir resultados.

## Os achados, em ordem de robustez

**1. RL profundo é necessário, se RL for usado.** O espaço tem 4 dimensões, mas
métodos tabulares falham: **79,8 %** das transições são *intra-tile* — o passo da
dinâmica (0,090 °C) é onze vezes menor que o bin da discretização (1,00 °C), e o
valor não se propaga. O Q-Learning tabular atinge 42,1 % de conforto, **pior que
um termostato com zona morta**. Deriva apenas de parâmetros publicados, e é o
achado mais robusto do conjunto.

**2. A penalidade anti-short-cycling não cumpre seu objetivo.** Requisito de
36 min de permanência mínima; **mediana medida de 12 min** e 71–83 % de
violações. A causa é aritmética: a penalidade vale no máximo 5, contra ganhos de
conforto de até 14. Uma restrição dura reduz as violações a 6,4 % **sem custo de
conforto**.

**3. A formulação de recompensa proposta não supera a convencional.** Ablação de
8 variantes × 3 sementes: a quadrática simples empata (δ de Cliff = −0,11,
desprezível) e três dos cinco termos são inertes. O que de fato importa é o
*gradiente interno* — sem ele o conforto cai de 83,7 % para 47,7 %. A contribuição
real é a correção de uma falha introduzida pelo platô plano, não uma recompensa
melhor.

**4. O baseline mal configurado explica a vantagem reivindicada** (tabela acima).

**5. Precisão maior não favorece o RL — favorece o clássico.** Com ambos os
controladores repreparados para cada largura de faixa, a vantagem do PI **cresce**:
+0,0 → +3,0 → +8,3 pp para ±2,0, ±1,0 e ±0,5 °C.

**6. O agente descarta o nível de melhor eficiência.** Os três perfis acionam
MEDIUM — o de maior COP — em **0,0 %** do tempo, apesar de pesos que variam por
3–4×. A causa não é a recompensa: um SAC treinado com a recompensa **idêntica** o
usa como o PI. É *winner-take-all* — a política determinística converte uma
margem de valor de ~1 % em uso de 0 %.

## O que este trabalho **não** conclui

Não conclui que RL seja inútil em HVAC. Conclui que, **nesta formulação** —
planta monovariável de primeira ordem, modelo conhecido, objetivo de rastreamento
— o controle clássico é a ferramenta adequada, e que a literatura da área compara
majoritariamente contra adversários fracos (ver a classificação por qualidade de
baseline na Seção 2 do artigo).

Há uma limitação de método que consideramos a mais importante: **o nicho do RL é
o descasamento de modelo, e um simulador de autoria própria não o exibe por
construção**. Testar a hipótese exige transferência simulação-realidade ou
benchmark de terceiros — BOPTEST, EnergyPlus, BuildingGym.

## Trabalho em aberto

- Refazer o achado nº 4 com o PI sintonizado **apenas** na partição de
  calibração (a ressalva metodológica que permanece)
- MPC como controlador de referência
- Migração para BOPTEST ou BuildingGym
- Multi-zona com capacidade compartilhada (`v5/hvac/multizone.py`), onde a
  decisão de alocação não admite lei de controle local

## Referência

O manuscrito auditado é `29914_Paper_manuscript.pdf`. O código original foi
perdido; `v4/paper/` é uma reimplementação a partir do texto, com quatro
parâmetros não publicados marcados `# INFERIDO` e a base de cada inferência
registrada.
