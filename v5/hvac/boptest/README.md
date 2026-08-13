# Transferência para o BOPTEST

Ponte entre os controladores deste trabalho e o [BOPTEST](https://ibpsa.github.io/project1-boptest/)
(IBPSA Project 1), o benchmark de referência para avaliação de controle predial.

## Por que existe

A ameaça à validade declarada como a mais séria do artigo é que todos os
resultados vivem dentro de um simulador de autoria própria — que, por
construção, não exibe descasamento de modelo, precisamente o regime em que se
espera que o aprendizado por reforço tenha vantagem. Este módulo executa os
**mesmos** controladores, **sem retreino**, contra um emulador Modelica revisado
por pares que ninguém aqui escreveu.

## Como rodar

```bash
# 1. serviço (uma vez; as imagens levam ~10 min para construir)
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest && docker compose up -d web worker provision

# se a porta 8000 estiver ocupada, crie docker-compose.override.yml:
#   services: {web: {ports: !override ["8098:8000"]}}

# 2. experimento
cd v5
python experimentos/boptest_transferencia.py --url http://127.0.0.1:8098
```

Resultado em `experimentos/resultados_boptest.csv`.

## Decisões de projeto que o leitor precisa conhecer

**Contrato de observação.** A observação não é redeclarada aqui: vem de
`hvac.features`, a mesma declaração única que alimenta o ambiente local. É o que
permite carregar um agente treinado localmente e executá-lo aqui sabendo que cada
canal chega à política com o significado com que foi treinado.

**Mapeamento do atuador.** O `bestest_air` expõe o ventilador do fancoil como
sinal normalizado em [0, 1]. Com a temperatura de insuflamento fixa, ele vira uma
fração de carga de resfriamento — a mesma grandeza dos quatro níveis discretos do
artigo, cujas frações são 0,00, 0,25, 0,55 e 1,00. O mapeamento é direto.

**Intervalo de controle derivado, não arbitrado.** O protocolo do artigo decide a
cada 12 min. No ambiente local isso é benigno: a plena carga move a sala 0,09 °C
por passo. No `bestest_air` a plena carga move **8,8 °C** em 12 min — mais que o
dobro da largura inteira da faixa de conforto. Manter os 12 min não preservaria o
protocolo; transformaria o problema em liga-desliga puro para todos os
controladores, e a comparação deixaria de discriminar.

O intervalo é, então, derivado de uma condição de projeto declarada antes de
medir — *a plena carga não deve atravessar mais que a meia-faixa de conforto em
um intervalo de decisão* — implementada em `calibracao.medir_autoridade`. Para o
`peak_cool_day` isso dá **180 s**. É a mesma disciplina que a Seção 5.2 do artigo
recomenda ao tratar de demanda contratada: derivar de condição de projeto, em vez
de calibrar até o resultado ficar bom.

Esse número é, por si, um resultado: ele mede de fora a ameaça que a Seção 5.3
declara. A planta do artigo é lenta demais e o equipamento pequeno demais frente
a uma zona real do BESTEST.

## O que **não** transfere

1. **Curva de COP.** O fancoil do emulador não reproduz o pico de eficiência em
   carga parcial que o modelo local tem. As conclusões da Seção 4.8, sobre o
   descarte do nível de melhor COP, não são testáveis aqui.
2. **Ocupação.** O emulador não publica contagem de ocupantes. O canal é
   *nominal* (janela ocupada da config), que é a semântica do treino;
   `--ocupacao co2` troca-o pela concentração medida, observável no mundo real
   mas de escala distinta da do treino.
3. **Clima.** Denver, com ganhos solares e internos próprios, contra a senoide
   tropical do ambiente local. Nos períodos de resfriamento a zona sem
   climatização chega a 35,5 °C (`peak_cool_day`) e 30,0 °C
   (`typical_cool_day`).

## Testes

`tests/test_boptest.py` roda **sem Docker**: o cliente é substituído por um
emulador de brinquedo com a mesma interface, porque a camada que erra em silêncio
é a de tradução (unidades, nomes de ponto, ordem de canais) e um ambiente que só
pudesse ser testado com o serviço de pé não seria testado. O teste de integração
real é pulado automaticamente quando `BOPTEST_URL` não responde.

```bash
BOPTEST_URL=http://127.0.0.1:8098 python -m pytest tests/test_boptest.py -q
```
