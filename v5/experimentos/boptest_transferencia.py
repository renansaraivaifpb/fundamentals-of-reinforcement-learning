# -*- coding: utf-8 -*-
"""
Transferência para um emulador de terceiros (BOPTEST).

PERGUNTA
--------
A Seção 4.2 mede que um PI sintonizado iguala o DQN dentro do simulador escrito
por nós. A Seção 5.1 declara que esse resultado não é conclusivo, porque o nicho
do aprendizado por reforço é o descasamento de modelo e um simulador próprio não
o exibe por construção. Este experimento executa exatamente os mesmos
controladores contra o `bestest_air` do IBPSA Project 1 — um emulador Modelica
que ninguém aqui escreveu, com física, clima e ganhos internos independentes.

O QUE MUDA E O QUE NÃO MUDA
---------------------------
Não muda: a política. Os agentes são carregados dos mesmos arquivos avaliados no
artigo e executados SEM RETREINO, em modo determinístico. O PI mantém os ganhos
sintonizados no ambiente local (Kp = 1,3; Ki = 0,2), sem re-sintonia. É esse
congelamento que torna o teste informativo: mede-se transferência, não ajuste.

Muda: a planta. O `bestest_air` tem constante de tempo compatível com uma zona
real, clima medido de Denver, ganhos internos e solares próprios, e um fancoil
cuja eficiência não reproduz o pico de COP em carga parcial do nosso modelo. Por
isso as conclusões da Seção 4.8, sobre o descarte do nível de melhor COP, NÃO
são testáveis aqui e não devem ser lidas nestes números.

    python experimentos/boptest_transferencia.py --url http://127.0.0.1:8098
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvac.baselines import PIController, ThermostatAgent          # noqa: E402
from hvac.boptest import ConfigBoptest, avaliar, medir_autoridade   # noqa: E402
from hvac.config import ClassroomConfig                           # noqa: E402
from hvac.model_io import load_agent                              # noqa: E402
from hvac.results import _pm                                      # noqa: E402

SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_boptest.csv")

# Os mesmos perfis da Tabela de comparação do artigo.
AGENTES = [("DQN_Agressivo_seed0.zip", "DQN Agressivo"),
           ("DQN_Equilibrado_seed0.zip", "DQN Equilibrado"),
           ("DQN_Passivo_seed0.zip", "DQN Passivo")]


def _agente(caminho: str):
    """Carrega um agente e verifica o contrato de observação contra a ponte."""
    modelo, _cfg_treino, _meta = load_agent(_pm(caminho))
    return modelo


def controladores(incluir_agentes: bool = True):
    """
    Fábricas dos controladores, na ordem em que aparecem no artigo.

    São fábricas, e não instâncias, porque cada par (controlador, período) roda
    em um teste novo do BOPTEST e o controlador precisa partir com estado limpo —
    um PI que herdasse o integrador do episódio anterior mediria outra coisa.
    """
    ctrl = {
        # Ganhos do artigo, sintonizados na planta LOCAL e aqui congelados: é o
        # que mede transferência.
        "PI sintonizado (ganhos locais)": lambda cfg: PIController(cfg, kp=1.3, ki=0.2),
        # Ganhos re-sintonizados NO emulador por busca em grade
        # (`boptest_sintonia_pi.py`), com desempate pela faixa estreita, que é a
        # métrica que discrimina. Sem esta linha o experimento cometeria, contra
        # o PI, o mesmo defeito de baseline mal configurado que o artigo audita.
        "PI re-sintonizado (emulador)": lambda cfg: PIController(cfg, kp=0.2, ki=0.05),
        "Termostato zm=1 °C": lambda cfg: ThermostatAgent(cfg, deadband=1.0),
        "Termostato zm=0 (baseline)": lambda cfg: ThermostatAgent(cfg, deadband=0.0),
    }
    if incluir_agentes:
        for arquivo, nome in AGENTES:
            ctrl[nome] = (lambda a: (lambda cfg: _agente(a)))(arquivo)
    return ctrl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("BOPTEST_URL",
                                                    "http://127.0.0.1:8000"))
    ap.add_argument("--periodos", nargs="+",
                    default=["peak_cool_day", "typical_cool_day"])
    ap.add_argument("--intervalo", default="auto",
                    help="intervalo de controle em segundos, ou 'auto' para "
                         "derivá-lo da autoridade medida do atuador")
    ap.add_argument("--horas", type=float, default=24.0,
                    help="duração do episódio")
    ap.add_argument("--ocupacao", default="nominal", choices=["nominal", "co2"])
    ap.add_argument("--sem-agentes", action="store_true",
                    help="só controladores clássicos (não exige modelos)")
    ap.add_argument("--saida", default=SAIDA)
    args = ap.parse_args()

    cfg = ClassroomConfig()

    # O intervalo de decisão do artigo (12 min) é benigno numa planta cuja plena
    # carga move 0,09 °C por passo; no emulador ela move vários graus, e mantê-lo
    # transformaria o problema em liga-desliga puro para TODOS os controladores.
    # Deriva-se, então, de condição de projeto declarada — ver `boptest.calibracao`.
    if args.intervalo == "auto":
        aut = medir_autoridade(config=cfg, url=args.url,
                               periodo=args.periodos[0])
        print(f"calibração: {aut}")
        intervalo = aut.intervalo_derivado_s
    else:
        intervalo = float(args.intervalo)

    passos = int(round(args.horas * 3600.0 / intervalo))
    bop = ConfigBoptest(intervalo_controle_s=intervalo, passos=passos,
                        fonte_ocupacao=args.ocupacao)

    print(f"BOPTEST em {args.url} — caso {bop.caso}, "
          f"passo {intervalo:.0f} s × {passos} = {args.horas:.0f} h, "
          f"ocupação {args.ocupacao}")
    df = avaliar(controladores(not args.sem_agentes),
                 config=cfg, periodos=args.periodos, bop=bop, url=args.url)

    df.to_csv(args.saida, index=False)
    print(f"\ngravado: {args.saida}")

    cols = [c for c in ["controlador", "periodo", "comfort_wide_pct",
                        "comfort_narrow_pct", "abs_dev_from_ideal",
                        "energy_kwh_day", "changes_per_hour",
                        "kpi_tdis_tot", "kpi_ener_tot", "kpi_cost_tot"]
            if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
