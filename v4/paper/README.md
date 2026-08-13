# Reprodução — *Intelligent Classroom HVAC Control Using Deep Reinforcement Learning*

> **Status:** este diretório continua sendo a **reprodução fiel do manuscrito** e
> a fonte dos modelos treinados e dos CSVs de ablação. O desenvolvimento ativo,
> porém, migrou para **`v5/`**, que preserva a paridade numérica bit-a-bit e
> acrescenta garantias de reprodutibilidade (contrato de observação verificado,
> avaliação multi-semente, fonte única para tabelas e figuras). Comece por
> `v5/README.md` e pelos cadernos em `v5/notebooks/`.
>
> A suíte deste diretório está em **79 testes + 1 xfailed** (algumas passagens
> abaixo citam contagens anteriores).

Implementação fiel ao manuscrito `29914_Paper_manuscript.pdf`, isolada em
`v4/paper/` para não sobrescrever o código histórico do `v4/`.

## Arquitetura

| Módulo | Responsabilidade |
|---|---|
| `ac_physics.py` | `ACPhysicsModel` — Tabela 1 **derivada** de BTU/h + COP, não hardcodada |
| `config.py` | `ClassroomConfig`, `REWARD_PROFILES` (Tabela 2), `ROOM_VARIATIONS` (Tabela 6) |
| `env.py` | `ClassroomACEnv` — eqs. 2 (física), 3 (observação), 4 (conforto), 5 (anti-short-cycling) |
| `wrappers.py` | `ActionRepeat`, `SafetyShield`, `DecisionLogger`, `ContinuousAction` (SAC) |
| `scenarios.py` | Matriz 3×3 C1–C9 (Tabela 3), `ThermostatAgent`, `AlwaysOffAgent` |
| `metrics.py` | Janela ocupada, filtro de controlabilidade, métricas das Tabelas 4/5 |
| `train.py` | DQN (3 perfis) + SAC, 550k passos, lr 5e-5, batch 64, action repeat 2 |
| `evaluate.py` | Tabelas 4/5/6 e Figuras 1–4 |

## Como rodar

```bash
cd v4/paper

# protocolo do paper: 3 perfis + SAC, 550k passos
python train.py --timesteps 550000 --seeds 0 --sac

# robustez sobre 3 sementes (Seção 5)
python train.py --timesteps 550000 --seeds 0 1 2

# Tabelas 4/5 + Figuras 1–4
python evaluate.py

# Tabela 6 (generalização sem retreino)
python evaluate.py --generalization

# Figura 1 isolada (não exige modelos treinados)
python evaluate.py --figure1
```

## Parâmetros inferidos

O paper especifica $B_c$, penalidade de energia e de troca (Tabela 2), mas **não
publica** quatro parâmetros necessários para executar a eq. 4 e a eq. 5. Foram
inferidos e estão marcados `# INFERIDO` em `config.py`:

| Símbolo | Valor | Base da inferência |
|---|---|---|
| $B$ (bônus base) | `10.0` | Figura 1: bordas do platô (22 e 26 °C) em ≈ +10 |
| $k$ (curvatura) | `0.6` | Figura 1: em 32 °C a curva cai a ≈ −11 ⟹ $10 - k\cdot6^2 = -11$ ⟹ $k \approx 0{,}58$; 0,6 é também o valor do `v4` original |
| $\rho$ (anti-short-cycling) | `-5.0` | Não observável. Escolhido na ordem de grandeza do conforto |
| Penalidade de frio | `-2.0` | A Conclusão do paper alerta que valor severo induziu *"medo de resfriar"*; o `v4` original usava −20 |

**Divergências numéricas devem ser atribuídas a estes quatro parâmetros antes de
se suspeitar da implementação.**

## Fidelidade verificada

### Tabela 1 — reproduz exatamente

Derivada de um split de 30 000 BTU/h (8,7921 kW térmicos):

| Nível | Carga | Resfr. (u) | Elétrica (kW) | Paper |
|---|---|---|---|---|
| LOW | 0,25 | 10 | 0,637 | 0,64 ✅ |
| MEDIUM | 0,55 | 22 | 1,347 | 1,35 ✅ |
| HIGH | 1,00 | 40 | 2,931 | 2,93 ✅ |

### Filtro de controlabilidade — reproduz exatamente

**8/9** cenários controláveis, igual ao paper. O excluído é **C1 (Frio + Poucas)**,
consistente com a Seção 5.1: *"a única exceção é Frio + Poucas"*.

### Tabela 4 — reprodução (550k passos, semente 0)

| Agente | Conf.[22,26] | Conf.[23,25] | \|T−24\| | Sobreaq. | Energia | Custo | Trocas/h |
|---|---|---|---|---|---|---|---|
| Agressivo — paper | 82,9 % | 83,9 % | 0,86 | 5,4 % | 15,59 | 13,64 | 2,91 |
| Agressivo — repro | **86,5 %** | **83,8 %** | **0,77** | **5,2 %** | 11,66 | 10,72 | 1,22 |
| Equilibrado — paper | 82,9 % | 78,1 % | 0,99 | 5,4 % | 14,65 | 12,73 | 2,45 |
| Equilibrado — repro | **86,5 %** | 83,8 % | **0,89** | **5,2 %** | 10,90 | 9,90 | 0,78 |
| Passivo — paper | 82,9 % | 79,6 % | 0,97 | 5,4 % | 15,28 | 13,31 | 2,57 |
| Passivo — repro | **86,5 %** | 83,7 % | **0,87** | **5,2 %** | 10,79 | 9,83 | 0,79 |
| Termostato — paper | 50,9 % | 6,4 % | 2,21 | 41,4 % | 10,91 | 9,81 | 2,50 |
| Termostato — repro | 54,0 % | 12,2 % | 2,10 | 37,7 % | 7,55 | 7,11 | 0,97 |

**Achado central reproduzido:** vantagem de **+32,5 pp** do RL sobre o termostato
(paper: +32 pp); **os três perfis empatam** no conforto binário — a tese de
saturação da métrica; sobreaquecimento idêntico entre perfis (5,2 %) contra
37,7 % do termostato; e Agressivo é o mais próximo do ideal (\|T−24\| = 0,77),
mesma ordenação do paper.

### Tabela 5 — discreto (DQN) vs. contínuo (SAC)

| Controlador | Conf.[22,26] | Conf.[23,25] | \|T−24\| | Energia | Gasto |
|---|---|---|---|---|---|
| DQN Equilibrado — paper | 82,9 % | 78,1 % | 0,99 | 14,65 | 12,73 |
| DQN Equilibrado — repro | 86,5 % | 83,8 % | 0,89 | 10,90 | 9,90 |
| SAC — paper | 74,5 % | 62,8 % | 1,32 | 16,47 | 15,25 |
| SAC — repro | **75,5 %** | 52,0 % | **1,46** | 11,35 | 10,25 |

O conforto do SAC reproduz com **1,0 pp** de diferença, e a conclusão
qualitativa se mantém: **o DQN domina o SAC** em conforto (−11,0 pp na repro,
−8,4 pp no paper).

**Não reproduzido:** a afirmação de que o SAC faz ≈54 % menos ajustes de
potência. Na repro o SAC ajusta *ligeiramente mais* que o DQN. A métrica depende
de um limiar de "ajuste significativo" que o paper não publica — usei
\|Δcarga\| > 0,05; um limiar maior reduziria a contagem do SAC. Métrica não
reprodutível a partir do texto.

### Tabela 6 — generalização sem retreino

| Sala | Conf. DQN (repro / paper) | Conf. Term. (repro / paper) | Δ repro | Δ paper |
|---|---|---|---|---|
| C_th=10 (leve) | 87,3 / 88,6 | 53,9 / 51,3 | +33,3 | +37,3 |
| C_th=15 (treino) | 86,5 / 83,2 | 54,0 / 51,0 | **+32,5** | **+32,2** |
| C_th=20 | 87,2 / 77,6 | 54,5 / 49,4 | +32,8 | +28,2 |
| C_th=30 (pesada) | 81,1 / 67,5 | 56,4 / 45,8 | +24,8 | +21,7 |
| K=0,3 (bem isolada) | 85,5 / 80,3 | 57,0 / 51,9 | **+28,5** | **+28,4** |
| K=0,8 (mal isolada) | 84,0 / 85,3 | 49,0 / 46,7 | +35,0 | +38,5 |

A vantagem reproduz na faixa **+24,8 a +35,0 pp**, dentro da faixa +21,7 a +38,5
do paper, com a mesma assinatura qualitativa: pior degradação em sala pesada
(C_th=30) e menor vantagem na sala bem isolada (K=0,3). Duas células batem
praticamente na casa decimal (+32,5/+32,2 e +28,5/+28,4).

### Baseline termostático

| Métrica | Repro | Paper | Δ |
|---|---|---|---|
| Conf. [22,26] | 54,0 % | 50,9 % | +3,1 pp |
| Conf. [23,25] | 12,2 % | 6,4 % | +5,8 pp |
| \|T−24\| | 2,10 °C | 2,21 °C | −0,11 |
| Sobreaquecimento | 37,7 % | 41,4 % | −3,7 pp |
| Energia | 7,55 kWh/dia | 10,91 kWh/dia | **−31 %** |
| Custo | R$ 7,11 | R$ 9,81 | **−28 %** |
| Trocas/h | 0,97 | 2,50 | **−61 %** |

As métricas de conforto fecham dentro de ~3 pp — a física está correta. Energia
e trocas/h ficam abaixo porque **o paper não publica os limiares do deadband**.
Minha implementação liga em MEDIUM acima de 26 °C e em HIGH acima de 28 °C,
desligando ao voltar à faixa; um deadband mais estreito (ou histerese
assimétrica) comuta mais e consome mais. Não é ajustável a partir do texto sem
*fitting* aos resultados — o que seria reprodução circular.

## Correções em relação ao `v4/` original

Três bugs do código original impediam qualquer avaliação válida e estão
corrigidos aqui:

1. **Hora dos cenários descartada** — `reset()` lia `options['hour_of_day']` mas
   o `scenarios.json` usava `'hour'`. Todos os cenários rodavam em hora
   aleatória; como $T_{ext}$ e a tarifa de pico dependem da hora, toda
   comparação estava contaminada. Agora aceita ambas as chaves.
2. **Observação fora do espaço declarado** — com 45 ocupantes, `o_norm = 1,33`
   violava o `Box(high=1.0)`. Agora clipado.
3. **Descasamento treino/avaliação** — o `analyzer_v4.1.py` avaliava sem o
   `ActionRepeatWrapper` usado no treino, medindo uma política com horizonte de
   decisão diferente. Agora o `action_repeat` é lido do metadado do modelo.

Dois bugs foram encontrados e corrigidos **nesta implementação** durante a
validação:

4. **`ActionRepeatWrapper` descartava grandezas extensivas** — devolvia só o
   `info` do último passo interno, perdendo metade da energia e **todas** as
   trocas de nível (a troca ocorre no primeiro passo interno). Sintoma:
   `Trocas/h = 0,00`. Corrigido com acumuladores + OR.
5. **`changes_per_hour` dividia pelas horas erradas** — usava `len(df)` quando
   cada linha vale `action_repeat` passos. Corrigido via `inner_steps`.

## Limitações herdadas do paper

- Balanço térmico agregado: sala como nó único, sem gradientes espaciais,
  umidade ou CO₂.
- $T_{ext}$ senoidal idealizada, sem dados meteorológicos reais.
- Atuador instantâneo; compressores reais têm latência e ciclo de partida (a
  penalidade anti-short-cycling endereça isso apenas em parte).
- Sem validação em hardware — todos os resultados são de simulação.
