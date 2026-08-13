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
| o código atual | `v5/hvac/` |
| a transferência para benchmark de terceiros | `v5/hvac/boptest/` |
| a implementação fiel do manuscrito auditado | `v4/paper/` |
| a resposta ponto a ponto aos pareceres | `v4/paper/REVISAO.md` |

Os artigos **não são versionados**: são saída dos geradores, e versionar o
`.docx` ao lado do código que o produz cria duas fontes de verdade e um
conflito binário a cada regeração. O que se revisa é o gerador.

```bash
cd v5
pip install -e .
python -m pytest tests/ -q           # 35 testes
python notebooks/build_notebooks.py  # regera os 5 cadernos
python gerar_paper.py                # artigo em português (22 tab., 17 fig.)
python gerar_paper.py --curto        # versão reduzida (20 tab., 13 fig.)
python gerar_paper_eb.py             # manuscrito em inglês (Energy & Buildings)
python gerar_cover_letter.py         # cover letter da submissão
python gerar_paper_eb.py --tex       # o mesmo texto na classe LaTeX da Elsevier
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

**6. O resultado sobrevive a um emulador de terceiros — desde que o baseline
também seja repreparado.** Os mesmos agentes, **sem retreino**, foram executados
contra o caso `bestest_air` do [BOPTEST](https://ibpsa.github.io/project1-boptest/),
emulador Modelica mantido pelo IBPSA. Com os ganhos do PI **congelados** da
planta local, os agentes vencem por 10,7 e 17,0 pp; re-sintonizado dentro do
emulador, o PI volta a liderar (84,0 % contra 81,3 % no dia de pico; 100,0 %
contra 97,7 % no dia típico, este **fora da amostra** da sintonia). O
experimento reproduziu, contra o baseline deste próprio trabalho, o defeito que
o artigo audita: basta congelar o adversário em condições novas para o RL
"ganhar" dez pontos. Ver `v5/hvac/boptest/`.

**7. O agente descarta o nível de melhor eficiência.** Os três perfis acionam
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

Havia uma limitação de método que considerávamos a mais importante: **o nicho do
RL é o descasamento de modelo, e um simulador de autoria própria não o exibe por
construção**. O achado nº 6 endereça essa limitação, executando os mesmos
controladores num emulador que não escrevemos. A ressalva que **permanece** é a
outra: a constante de tempo adotada (30 h) excede a duração do episódio e torna a
planta local mais benigna que uma sala real. O BOPTEST mede isso de fora — lá a
plena carga move a zona 8,8 °C em 12 min, contra 0,090 °C por passo aqui.

## Submissão

O manuscrito é gerado, nunca editado à mão. Alvo atual: **Energy & Buildings**
(Elsevier), que exige dois arquivos.

```bash
cd v5
python gerar_paper_eb.py       # submissao_eb/manuscript.docx + figures/
python gerar_cover_letter.py   # submissao_eb/cover_letter.docx
python gerar_paper_eb.py --tex # submissao_eb/cas/manuscript.tex (classe cas-sc)
```

O texto em inglês não é uma tradução paralela: reusa `hvac.results` e
`hvac.figures`, os mesmos objetos que alimentam os cadernos e o artigo em
português. A camada `hvac/i18n_en.py` traduz rótulos de figura e rótulos vindos
de DataFrame, e **levanta exceção** em vez de deixar passar um rótulo sem
tradução.

A saída LaTeX usa a classe CAS distribuída pela Elsevier. O corpo do artigo
continua escrito uma única vez: os helpers de `gerar_paper_eb.py` despacham para
o backend `.docx` ou para `hvac/tex_backend.py`, e as equações são descritas por
uma árvore que cada backend percorre à sua maneira. A classe CAS não é
versionada aqui; baixe o pacote da editora e deixe `els-cas-templates/` na raiz.

## Trabalho em aberto

- Reexecutar o protocolo sob capacidade térmica fisicamente plausível — é a
  ameaça de maior peso, e o experimento de maior retorno entre os pendentes
- Ampliar a transferência: mais casos do BOPTEST, mais períodos, e comparação
  contra o controlador baseline nativo de cada caso
- Refazer o achado nº 4 com o PI sintonizado **apenas** na partição de
  calibração (a ressalva metodológica que permanece)
- Elevar de 3 para 8–10 sementes nas comparações principais
- MPC como controlador de referência
- Multi-zona com capacidade compartilhada (`v5/hvac/multizone.py`), onde a
  decisão de alocação não admite lei de controle local

## Sobre a implementação auditada

`v4/paper/` implementa o controlador auditado a partir de sua especificação.
Quatro constantes da função de recompensa não são publicadas por ela e estão
marcadas `# INFERIDO`, com a base de cada inferência registrada; a Seção 4.11 do
artigo examina quais delas podem ser obtidas por argumento em vez de leitura de
gráfico, e mede o que acontece quando são substituídas.
