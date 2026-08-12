# v5 — o que mudou em relação à `v4/paper`

A v5 não muda a física, os cenários nem as métricas: **a paridade com a v4 foi
verificada e é bit-a-bit** (PI 86,50 % / 0,7120 °C / 10,4040 kWh; termostato
zona morta 0: 54,00 / 2,1047 / 7,5480; zona morta 1 °C: 77,67 / 1,7827 / 8,1167).
Tudo que já foi medido continua valendo.

O que muda é **a chance que o agente tem de aprender** e **a capacidade de saber
se ele aprendeu**.

```bash
cd v5
pip install -e .                       # imports deixam de depender do cwd
python -m pytest tests/ -q             # 13 testes
python -m hvac.train --lab2 --algos TD3 --seeds 0 1 2
```

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

---

## O que a v5 **não** faz

Não muda a conclusão do trabalho. As melhorias dão ao RL a melhor chance
disponível e tornam a medição confiável — não alteram o fato de que, para uma
planta SISO de primeira ordem com modelo conhecido e objetivo de rastreamento, o
controle clássico é a ferramenta indicada.

O valor de rodar a v5 é outro: se o RL continuar perdendo, agora se perde num
teste em que ele teve tudo a favor, com incerteza reportada e sem risco de canal
trocado — o que torna o resultado negativo **muito mais difícil de refutar**.

## Migração pendente

- `evaluate.py` e `evaluate_lab2.py` não foram portados: continuam na `v4/paper`.
  Portá-los sobre `hvac.evaluation` é o próximo passo natural.
- `tabular.py`, `ablation.py` e `compressor_health.py` idem.
- `multizone.py` segue órfão, e a decisão sobre ele é anterior ao código: o
  adversário certo ali é MPC, não PI.
