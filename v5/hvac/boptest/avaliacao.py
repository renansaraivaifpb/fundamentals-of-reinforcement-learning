# -*- coding: utf-8 -*-
"""
Laço de avaliação sobre o BOPTEST.

As métricas do artigo NÃO são recalculadas aqui. O episódio é reduzido ao mesmo
DataFrame que o ambiente local produz e entregue a `hvac.metrics.episode_metrics`,
de modo que "conforto na faixa", "desvio absoluto do ideal" e "comutações por
hora" tenham exatamente a mesma definição nos dois ambientes. Reimplementá-las
aqui permitiria que uma diferença de definição fosse lida como diferença de
desempenho — que é o gênero de erro que este artigo audita.

Reporta-se, ao lado delas, o conjunto de KPIs nativo do BOPTEST (`tdis_tot`,
`ener_tot`, `cost_tot`, `pele_tot`, ...), que é a moeda com que a literatura de
building performance simulation compara controladores. Ter as duas famílias na
mesma tabela é o que permite ligar este resultado ao do artigo e, ao mesmo tempo,
a trabalhos de terceiros.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ClassroomConfig
from ..metrics import episode_metrics
from .client import BoptestClient
from .env import BoptestClassroomEnv, ConfigBoptest


def _tarifa_brl(cfg: ClassroomConfig, hora: float) -> float:
    """Tarifa da config, para manter a coluna de custo comparável à do artigo."""
    try:
        return float(cfg.tariff_rate(hora))
    except Exception:                                  # config sem tarifa horária
        return float(getattr(cfg.tariff, "off_peak_brl_kwh", 0.0))


def rodar_episodio(env: BoptestClassroomEnv, controlador, *,
                   passa_info: bool = True,
                   opcoes: Optional[Dict] = None) -> Dict[str, object]:
    """
    Executa um episódio e devolve série, métricas do artigo e KPIs do framework.

    `controlador` segue a interface `predict(obs, deterministic=True)` do
    Stable-Baselines3; os controladores clássicos aceitam ainda `info=`, por onde
    recebem a temperatura sem depender da normalização da observação.
    """
    cfg = env.config
    if hasattr(controlador, "reset"):
        controlador.reset()

    obs, info = env.reset(options=opcoes or {})
    horas_por_passo = env.bop.intervalo_controle_s / 3600.0
    # `inner_steps` converte o passo desta ponte para os passos do protocolo do
    # artigo, de modo que "comutações por hora" signifique a mesma coisa aqui.
    passos_internos = max(int(round(horas_por_passo / cfg.dt)), 1)

    linhas: List[Dict] = []
    for _ in range(env.bop.passos):
        if passa_info:
            try:
                acao, _ = controlador.predict(obs, deterministic=True, info=info)
            except TypeError:
                acao, _ = controlador.predict(obs, deterministic=True)
        else:
            acao, _ = controlador.predict(obs, deterministic=True)

        obs, _, terminado, truncado, info = env.step(acao)

        hora = float(info["hour"])
        energia_kwh = float(info["power_w"]) / 1000.0 * horas_por_passo
        ocupado = (float(getattr(cfg, "occupied_start_hour", 7.0)) <= hora
                   < float(getattr(cfg, "occupied_end_hour", 22.0)))
        linhas.append({
            "temperature": float(info["temperature"]),
            "occupied": ocupado,
            "hour": hora,
            "energy_kwh": energia_kwh,
            "cost_brl": energia_kwh * _tarifa_brl(cfg, hora),
            "action_changed": bool(info["switched"]),
            "inner_steps": passos_internos,
            "level": int(info["level"]),
            "load": float(info["load"]),
            "outdoor": float(info["outdoor_temperature"]),
            "power_w": float(info["power_w"]),
        })
        if terminado or truncado:
            break

    df = pd.DataFrame(linhas)
    return {"df": df,
            "metricas": episode_metrics(df, cfg),
            "kpis": env.kpis()}


def avaliar(controladores: Dict[str, Callable[[ClassroomConfig], object]],
            *,
            config: Optional[ClassroomConfig] = None,
            periodos: Sequence[str] = ("peak_cool_day", "typical_cool_day"),
            bop: Optional[ConfigBoptest] = None,
            url: Optional[str] = None,
            verbose: bool = True) -> pd.DataFrame:
    """
    Avalia cada controlador em cada período, um teste do BOPTEST por execução.

    Um teste novo por par (controlador, período) é deliberado: o emulador guarda
    estado entre chamadas, e reaproveitar o mesmo teste faria o segundo
    controlador partir do estado que o primeiro deixou. Todos partem do mesmo
    ponto porque todos usam o mesmo rótulo de período, que carrega o warmup
    definido pelos autores do caso.
    """
    cfg = config or ClassroomConfig()
    base = bop or ConfigBoptest()
    saida: List[Dict] = []

    for periodo in periodos:
        for nome, fabrica in controladores.items():
            cliente = BoptestClient(url=url) if url else BoptestClient()
            env = BoptestClassroomEnv(config=cfg, bop=base, cliente=cliente)
            try:
                r = rodar_episodio(env, fabrica(cfg), opcoes={"periodo": periodo})
            finally:
                env.close()

            linha = {"controlador": nome, "periodo": periodo}
            linha.update({k: v for k, v in r["metricas"].items()
                          if isinstance(v, (int, float, np.floating))})
            for k, v in (r["kpis"] or {}).items():
                if isinstance(v, (int, float)):
                    linha[f"kpi_{k}"] = float(v)
            linha["externa_min"] = float(r["df"]["outdoor"].min())
            linha["externa_max"] = float(r["df"]["outdoor"].max())
            saida.append(linha)

            if verbose:
                m = r["metricas"]
                print(f"  {periodo:18s} {nome:28s} "
                      f"conf {m.get('comfort_wide_pct', float('nan')):5.1f}%  "
                      f"|T-24| {m.get('abs_dev_from_ideal', float('nan')):4.2f}  "
                      f"kWh {m.get('energy_kwh_day', float('nan')):5.2f}  "
                      f"tdis {(r['kpis'] or {}).get('tdis_tot', float('nan')):6.2f}")

    return pd.DataFrame(saida)
