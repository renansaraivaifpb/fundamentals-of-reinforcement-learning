# -*- coding: utf-8 -*-
"""
Re-sintonia do PI dentro do emulador de terceiros.

POR QUE ESTE EXPERIMENTO É OBRIGATÓRIO
--------------------------------------
O experimento de transferência executa o PI com os ganhos sintonizados na planta
local (Kp = 1,3; Ki = 0,2). Essa planta tem constante de tempo de 30 h e uma
plena carga que move 0,09 °C por passo; o `bestest_air` responde vários graus no
mesmo tempo. Cobrar desempenho de um controlador clássico cujos ganhos foram
ajustados para outra dinâmica é EXATAMENTE o defeito que a Seção 4.2 identifica
no manuscrito auditado — um adversário mal configurado — apenas deslocado de
ambiente.

A Seção 4.7 já estabeleceu a regra correta: quando a especificação muda, ambos os
controladores são repreparados. Aqui a planta muda, e a mesma regra se aplica. Se
o PI re-sintonizado recuperar a liderança, a leitura do experimento de
transferência é "os ganhos não transferem"; se não recuperar, a leitura é que o
agente de fato lida melhor com esta planta. As duas conclusões são publicáveis, e
só uma delas é honesta sem este experimento.

    python experimentos/boptest_sintonia_pi.py --url http://127.0.0.1:8098
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvac.baselines import PIController                           # noqa: E402
from hvac.boptest import BoptestClient, ConfigBoptest             # noqa: E402
from hvac.boptest.avaliacao import rodar_episodio                 # noqa: E402
from hvac.boptest.env import BoptestClassroomEnv                  # noqa: E402
from hvac.config import ClassroomConfig                           # noqa: E402

SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_boptest_sintonia.csv")

# A grade cobre duas ordens de grandeza em torno do ganho local, porque a razão
# entre as autoridades das duas plantas é dessa ordem. Ki inclui zero: numa
# planta rápida a ação integral pode ser francamente prejudicial.
KP = (0.2, 0.5, 1.3, 3.0, 6.0)
KI = (0.0, 0.05, 0.2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("BOPTEST_URL",
                                                    "http://127.0.0.1:8000"))
    ap.add_argument("--periodo", default="peak_cool_day",
                    help="período de sintonia; a avaliação final usa os demais")
    ap.add_argument("--intervalo", type=float, default=180.0)
    ap.add_argument("--horas", type=float, default=24.0)
    ap.add_argument("--saida", default=SAIDA)
    args = ap.parse_args()

    cfg = ClassroomConfig()
    bop = ConfigBoptest(intervalo_controle_s=args.intervalo,
                        passos=int(round(args.horas * 3600.0 / args.intervalo)))

    print(f"sintonia do PI em {args.periodo}: {len(KP) * len(KI)} combinações")
    linhas = []
    for kp, ki in itertools.product(KP, KI):
        env = BoptestClassroomEnv(config=cfg, bop=bop,
                                  cliente=BoptestClient(url=args.url))
        try:
            r = rodar_episodio(env, PIController(cfg, kp=kp, ki=ki),
                               opcoes={"periodo": args.periodo})
        finally:
            env.close()
        m = r["metricas"]
        linha = {"kp": kp, "ki": ki, "periodo": args.periodo,
                 "conf_larga": m.get("comfort_wide_pct"),
                 "conf_estreita": m.get("comfort_narrow_pct"),
                 "desvio": m.get("abs_dev_from_ideal"),
                 "kwh": m.get("energy_kwh_day"),
                 "trocas_h": m.get("changes_per_hour"),
                 "tdis_tot": (r["kpis"] or {}).get("tdis_tot")}
        linhas.append(linha)
        print(f"  Kp={kp:4.1f} Ki={ki:4.2f}  conf {linha['conf_larga']:5.1f}%  "
              f"estreita {linha['conf_estreita']:5.1f}%  "
              f"|T-24| {linha['desvio']:4.2f}  tdis {linha['tdis_tot']:6.2f}")

    df = pd.DataFrame(linhas).sort_values("conf_larga", ascending=False)
    df.to_csv(args.saida, index=False)
    print(f"\ngravado: {args.saida}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    melhor = df.iloc[0]
    print(f"\nmelhor sintonia local do emulador: "
          f"Kp = {melhor['kp']}, Ki = {melhor['ki']} "
          f"({melhor['conf_larga']:.1f}% de conforto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
