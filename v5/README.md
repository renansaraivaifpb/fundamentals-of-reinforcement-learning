# v5 — o que mudou em relação à `v4/paper`

A v5 não muda a física, os cenários nem as métricas: **a paridade com a v4 foi
verificada e é bit-a-bit** (PI 86,50 % / 0,7120 °C / 10,4040 kWh; termostato
zona morta 0: 54,00 / 2,1047 / 7,5480; zona morta 1 °C: 77,67 / 1,7827 / 8,1167).
Tudo que já foi medido continua valendo.

O que muda é **a chance que o agente tem de aprender** e **a capacidade de saber
se ele aprendeu**.

```bash
cd v5
pip install -e .                        # imports deixam de depender do cwd
python -m pytest tests/ -q              # 35 testes
python notebooks/build_notebooks.py     # regera os 5 cadernos executados
python gerar_paper.py                   # artigo completo (22 tab., 17 fig.)
python gerar_paper.py --curto           # versão reduzida (20 tab., 13 fig.)
python gerar_paper_eb.py                # manuscrito em inglês (Energy & Buildings)
python gerar_paper_eb.py --tex          # o mesmo texto na classe LaTeX cas-sc
python gerar_cover_letter.py            # cover letter da submissão
python -m hvac.train --lab2 --algos TD3 --seeds 0 1 2
```

## Mapa do diretório

| Caminho | Conteúdo |
|---|---|
| `hvac/` | pacote: ambiente, agentes, baselines, métricas, resultados, figuras |
| `notebooks/` | 5 cadernos **executados**, com saídas embutidas |
| `experimentos/` | scripts de treino longo + CSVs de resultado |
| `figuras_v5/` | PNGs em 300 dpi |
| `hvac/boptest/` | ponte para o BOPTEST: cliente REST, ambiente, avaliação |
| `gerar_paper.py` | artigo em português, a partir de `hvac.results`/`hvac.figures` |
| `gerar_paper_eb.py` | manuscrito em inglês; `--tex` emite a classe CAS da Elsevier |
| `gerar_cover_letter.py` | cover letter, com os números vindos dos mesmos objetos |
| `hvac/i18n_en.py` | glossário pt→en; erra alto em rótulo sem tradução |
| `hvac/tex_backend.py` | backend LaTeX dos mesmos helpers do gerador |

Os `.docx`/`.tex` **não são versionados**: são saída dos geradores. Reproduza
com os comandos acima.

### Os cadernos

| Caderno | Pergunta |
|---|---|
| `01_auditoria_baselines` | a vantagem vem do aprendizado ou do baseline? |
| `02_ablacao_e_short_cycling` | a recompensa é responsável? a proteção funciona? RL precisa ser profundo? |
| `03_regimes_estendidos` | existe algum regime que favoreça o RL? |
| `04_trajetorias_e_faixa_estreita` | como se comportam ao longo do dia? precisão maior favorece o RL? |
| `05_niveis_de_potencia_e_cenarios_aleatorios` | por que o agente gasta mais? quem mantém melhor a faixa? |

### Arquitetura de dados: fonte única

```
hvac/results.py  →  números   ─┬─→  notebooks/*.ipynb
hvac/figures.py  →  gráficos  ─┴─→  gerar_paper.py → .docx
```

`results.py` **recomputa** o que é barato (rodar controladores nos cenários) e
**lê dos CSVs** o que exigiu treino. `figures.py` apenas desenha, sobre dados já
computados. Divergência entre uma tabela e a figura ao lado é impossível por
construção — era o defeito da v4, cujos PNGs vinham de execuções diferentes das
que produziram os números reportados.

A numeração de tabelas e figuras do artigo é **derivada da composição**, por
chave simbólica: cortar uma seção renumera tudo sozinho. Foi assim que a versão
curta foi produzida sem quebrar referências — e o mecanismo já revelou uma
legenda ausente e uma citação com número escrito à mão.

---

## 1. Observação declarativa — `hvac/features.py`

Na v4 a observação era declarada **duas vezes**: o `Box` no `__init__` do
ambiente e o vetor em `_get_obs()`. Duas listas paralelas, mantidas à mão, em
pontos diferentes do arquivo. Já produziram o bug documentado de `o_norm = 1,33`
fora do espaço declarado, e deixavam aberto um modo de falha pior: trocar a ordem
de dois canais **não gera erro** — o `Box` aceita, o agente treina, e a política
aprende com os canais invertidos.

O sintoma mais eloquente estava nos nomes dos arquivos: os modelos se chamavam
`TD3_..._lab2_obs9.zip`. A dimensão da observação tinha migrado para o nome do
arquivo porque não havia onde mais guardá-la. **Quando um dado estrutural começa
a viver no nome do arquivo, falta uma estrutura no código.** (E o sufixo já era
mentira: ao habilitar a restrição de demanda, a observação passou a ter 10
canais, silenciosamente.)

Agora há uma lista de `Feature` — nome, limites, extrator, condição de ativação —
e dela derivam, sem possibilidade de divergência, o `observation_space`, o vetor
de cada passo e o **schema** (nomes ordenados) gravado no metadado.

## 2. Contrato de observação verificado na carga — `hvac/model_io.py`

A v4 já comparava `observation_space.shape` ao carregar um modelo. Isso pega o
caso grosseiro (9 canais contra 10) e é **cego para o perigoso**: duas
configurações podem dar a mesma dimensão com canais diferentes ou trocados.

`assert_schema_compatible` confere **forma e ordem**. Incompatibilidade vira erro
no carregamento em vez de virar um número errado numa tabela. Modelos da v4 (sem
schema) carregam com aviso explícito de que só a forma pôde ser conferida.

## 3. Seleção do melhor modelo, não do último — `hvac/train.py`

A v4 salvava o estado ao fim de `learn()`. O comentário do próprio `ALGOS` da v4
registra a variância entre sementes do DQN na mesma configuração:
**97,8 % / 96,4 % / 5,3 %** — uma semente colapsou. Salvando o estado final, um
colapso perto do fim é o que vai para o disco, mesmo que o agente tenha passado
quase todo o treino numa política boa.

Um `EvalCallback` avalia periodicamente, num ambiente separado com sementes
distintas, e guarda o melhor. É a correção de maior retorno por linha do arquivo.

## 4. Normalização da recompensa — `hvac/train.py`

A observação já sai normalizada por construção, então `norm_obs=False` de
propósito — renormalizar por estatística móvel destruiria o significado fixo de
cada canal (o zero do erro escalado deixaria de ser o setpoint).

A **recompensa** é outra história: o conforto vale até +10, mas a topologia
`band` aplica parede de inclinação 40 por grau fora da faixa. O próprio código da
v4 documenta o sintoma no conforto Huber — *"faixa dinâmica de 204x, que faz os
alvos de TD explodirem e o DQN não converge"*. `VecNormalize(norm_reward=True)`
ataca a causa sem reescrever a recompensa.

## 5. Warm start com exploração reduzida — `hvac/train.py`

A v4 obteve 80,1 % de concordância com o PI após a clonagem, e então o resultado
degradava para 37,2 % de conforto. O diagnóstico já estava escrito no
`REVISAO.md`: *"a exploração ε-greedy default do DQN destrói a política clonada
antes que o TD-learning a recupere. Aproveitar o warm start exige também reduzir
a exploração inicial, o que não foi ajustado."*

Fica ajustado: sob `--expert-init`, `exploration_initial_eps = 0,15`,
`exploration_fraction = 0,05` e `learning_starts = 0` (o buffer já chega
semeado). O fio solto era explícito.

## 6. Avaliação multi-semente com incerteza — `hvac/evaluation.py`

O `evaluate.py` da v4 — que produz a comparação central entre PI e DQN — roda com
`--seed` de valor único. A diferença entre 0,72 °C e 0,77 °C foi lida de uma
amostra de tamanho **um**.

Aqui a unidade de reporte é sempre `(média, IC95, n)`, sobre sementes de
avaliação **disjuntas das de treino**, e todos os controladores são avaliados nas
**mesmas** sementes — comparação pareada, em que o ruído comum cancela na
diferença. `diferenca_afirmavel()` só afirma diferença com separação completa
(|δ de Cliff| = 1), critério deliberadamente conservador para n pequeno.

## 7. Eixos de ablação declarados — `hvac/config.py`

`LAB2_BASE` altera seis subsistemas de uma vez, e o comentário do próprio
dicionário afirma que as mudanças são "independentes e testáveis isoladamente" —
mas nunca foram, porque a config não oferecia eixo para isso. Quando o conjunto
não superou o PI, ficou impossível dizer qual mudança ajudou.

`ABLATION_GROUPS` declara nove eixos (`obs_pid`, `obs_derivada`, `obs_tarifa`,
`aquecimento`, `acao_continua`, …). A ablação vira uma flag:

```bash
python -m hvac.train --lab2 --ablate-group obs_pid --algos TD3 --seeds 0 1 2
```

## 8. Empacotamento — `pyproject.toml`

A v4 não tinha arquivo de dependências, e a consequência já havia aparecido: um
shim de compatibilidade numpy 1.x/2.x escrito à mão dentro de um script de
avaliação. Versões fixadas e pacote instalável em modo editável.

## 9. Transferência para terceiros — `hvac/boptest/`

Responde à ameaça declarada como a mais séria: todos os resultados vinham de um
simulador de autoria própria, que por construção não exibe descasamento de
modelo. O módulo executa os **mesmos** controladores contra o
[BOPTEST](https://ibpsa.github.io/project1-boptest/), emulador Modelica mantido
pelo IBPSA, sem retreinar nada.

```bash
# serviço (uma vez; as imagens levam ~10 min para construir)
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest && docker compose up -d web worker provision

cd v5 && python experimentos/boptest_transferencia.py --url http://127.0.0.1:8000
```

Três decisões sustentam o experimento:

**A observação não é redeclarada.** Vem de `hvac/features.py`, a mesma
declaração única que alimenta o ambiente local. É o que permite carregar um
agente treinado aqui e executá-lo lá sabendo que cada canal chega à política com
o significado com que foi treinado. Foi também por isso que não se adotou o
`boptest-gym` oficial: ele traz sua própria declaração de observação, e manter
duas em paralelo é exatamente o defeito que `features.py` existe para eliminar.

**As métricas não são recalculadas.** O episódio é reduzido ao mesmo DataFrame do
ambiente local e entregue a `metrics.episode_metrics`, para que "conforto na
faixa" e "comutações por hora" tenham a mesma definição nos dois ambientes. Ao
lado delas reporta-se o conjunto de KPIs nativo do BOPTEST, que é a moeda com que
a literatura de *building performance simulation* compara controladores.

**O intervalo de controle é derivado, não arbitrado.** O protocolo decide a cada
12 min, o que é benigno numa planta cuja plena carga move a sala 0,090 °C por
passo; no `bestest_air` ela move **8,8 °C** no mesmo intervalo, mais que o dobro
da faixa de conforto inteira. Mantê-lo transformaria o problema em liga-desliga
para todos os controladores. `boptest/calibracao.py` mede a autoridade do atuador
e deriva o intervalo de uma condição declarada antes de medir — *a plena carga
não deve atravessar mais que a meia-faixa em uma decisão* —, o que dá 180 s.
Esse número é, por si, uma medição externa da ameaça: a planta local é lenta, e
seu equipamento pequeno, frente a uma zona real.

**O achado.** Com os ganhos do PI congelados da planta local, os agentes vencem
por 10,7 e 17,0 pp. Re-sintonizado dentro do emulador (Kp = 0,2; Ki = 0,05), o PI
volta a liderar: 84,0 % contra 81,3 % no dia de pico e 100,0 % contra 97,7 % no
dia típico, este fora da amostra da sintonia. A transferência reproduziu, contra
o baseline deste próprio trabalho, o defeito que o artigo audita.

Duas coisas **não** transferem, e estão declaradas: o fancoil do emulador não tem
o pico de COP em carga parcial do equipamento local, de modo que o achado sobre o
nível de melhor eficiência não é testável ali; e o emulador não publica contagem
de ocupantes, então o canal de ocupação é nominal.

`tests/test_boptest.py` roda **sem Docker**: o cliente é substituído por um
emulador de brinquedo com a mesma interface, porque a camada que erra em silêncio
é a de tradução — unidades, nomes de ponto, ordem de canais.
---

## O que a v5 **não** faz

Não muda a conclusão do trabalho. As melhorias dão ao RL a melhor chance
disponível e tornam a medição confiável — não alteram o fato de que, para uma
planta SISO de primeira ordem com modelo conhecido e objetivo de rastreamento, o
controle clássico é a ferramenta indicada.

O valor de rodar a v5 é outro: se o RL continuar perdendo, agora se perde num
teste em que ele teve tudo a favor, com incerteza reportada e sem risco de canal
trocado — o que torna o resultado negativo **muito mais difícil de refutar**.

## Adições vindas da literatura

Após a leitura de dois trabalhos que você indicou, três itens foram
incorporados — cada um marcado no código com a referência.

**Wei, Wang e Zhu (DAC 2017)** — primeiro trabalho a aplicar DRL a HVAC:

- `comfort_type='hinge'` — a recompensa da eq. 1 deles: `-λ·([T−T_max]⁺ +
  [T_min−T]⁺)`, linear fora da faixa e nula dentro. Serve de contraponto ao
  "Platô Quadrático com gradiente" na ablação: é ainda mais simples que a
  variante `convencional`, que já empatava com a proposta.
- `observe_outdoor_forecast` — previsão multi-passo da externa no estado, que
  eles usam para permitir controle proativo. **Ressalva registrada no código:**
  aqui a externa é uma senoide determinística, então a previsão é *perfeita*;
  num prédio real teria erro, e parte da vantagem não sobreviveria.
- `multizone.py` — ação combinatória discreta (`mᶻ`: 64 / 256 / 1024 para 3/4/5
  zonas) e `MultiNivelWrapper`, adaptação do princípio de decomposição deles,
  que reduz para 12 / 16 / 20 — linear em vez de exponencial.

**Yuan et al. (Building Simulation)** — compara RBC, PID e RL no mesmo
experimento:

- `results.curva_aprendizado()` e `figures.fig_curva_aprendizado()` — desempenho
  contra orçamento de treino, análogo às Figs. 9-10 deles. Responde à objeção
  "o DQN perde porque treinou pouco?", que nenhum experimento de orçamento fixo
  responde.
- `results.consumo_decomposto()` e `figures.fig_consumo_decomposto()` — consumo
  por nível de potência, análogo à Fig. 11 deles. Revelou o mecanismo do
  desperdício: **o DQN nunca aciona MEDIUM**, o nível de melhor COP (3,59), e
  usa o dobro de HIGH, o de pior (3,00).

## Experimentos desta rodada

| Script | Pergunta | Resultado |
|---|---|---|
| `experimentos/faixa_estreita.py` | estreitar a faixa favorece o RL? | **Não** — a vantagem do PI cresce: +0,0 → +3,0 → +8,3 pp |
| `experimentos/curva_aprendizado.py` | é falta de treino? | em ±2,0 °C converge sobre o PI; em ±0,5 °C fica 16,9 pp abaixo, com a curva ainda subindo devagar — custo de amostra alto, não impossibilidade |
| `experimentos/teste_integral.py` | falta o integrador ao agente? | **Refutado** — dar o estado piorou (71,5 → 65,9 %) |
| `hvac.results.cenarios_aleatorios` | e num conjunto amplo e independente? | 150 cenários: PI e DQN Agressivo empatam em conforto; o PI gasta 11,4 % menos |
| `experimentos/boptest_transferencia.py` | o resultado sobrevive a um emulador de terceiros? | **Sim, com uma condição** — ver abaixo |
| `experimentos/boptest_sintonia_pi.py` | os ganhos do PI transferem entre plantas? | **Não**: re-sintonizar recupera 13,3 pp |

## Achados desta versão

**O nível de melhor eficiência é descartado.** Os três perfis do manuscrito
acionam MEDIUM — de maior COP (3,59) — em **0,0 %** do tempo, apesar de pesos que
variam por 3–4×. O perfil Agressivo descarta LOW também, virando liga-desliga
puro. Um **SAC treinado com a recompensa idêntica** usa MEDIUM em 13,3 %,
praticamente igual ao PI (12,0 %): mesma recompensa, comportamentos opostos.

A análise dos valores de ação mostra que as quatro opções são **quase
equivalentes** — a faixa entre elas vale ~1,9 % do valor absoluto, e MEDIUM perde
~1,2 %. O mecanismo é *winner-take-all*: uma política determinística converte
margem de ~1 % em uso de 0 %. **Reponderar a recompensa é o caminho errado**; o
que restaura os níveis intermediários é mudar a classe de política.

**A falha é concentrada, não difusa.** Nos cenários aleatórios, o déficit dos
perfis Equilibrado e Passivo vive quase todo em **salas vazias de madrugada**
(−21,7 e −18,2 pp com até 10 ocupantes, contra −4,4 e −3,1 com a sala cheia).
Com carga mínima, o menor nível não-nulo já é 25 % da capacidade — potência demais.

**Nenhum perfil domina o PI.** Para igualar em precisão, o Agressivo consome
12,9 % mais; para igualar em consumo, os outros perdem 6–7 pontos de precisão.
Como os três saem apenas de variar pesos — a tese central do manuscrito —, essa
variação percorre uma curva inteiramente **dentro** da fronteira de Pareto.

**Métrica saturada.** No conjunto independente, quatro controladores distintos
dão conforto idêntico na faixa larga (87,88 %): o número é fixado pelo transitório
de partida, limitado pela física. Uma métrica que não discrimina não deveria ser
a principal de um artigo.

## Migração pendente

- `evaluate.py` e `evaluate_lab2.py` não foram portados: continuam na `v4/paper`.
  Portá-los sobre `hvac.evaluation` é o próximo passo natural.
- `tabular.py`, `ablation.py` e `compressor_health.py` idem.
- `multizone.py` segue órfão, e a decisão sobre ele é anterior ao código: o
  adversário certo ali é MPC, não PI.
