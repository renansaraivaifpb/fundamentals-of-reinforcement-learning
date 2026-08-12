# -*- coding: utf-8 -*-
"""
Métricas de integridade do compressor.

Motivação: "ficar mudando de temperatura várias vezes pode prejudicar o
funcionamento do ar-condicionado". Correto, e a métrica `changes_per_hour` que eu
vinha usando é inadequada para isso — ela conta MODULAÇÃO, e modular é
precisamente a função de um equipamento inverter. O que danifica hardware é
específico:

1. CICLAGEM ON/OFF — partidas do compressor. É o dano dominante: cada partida
   tem corrente de inrush (5–7x a nominal), estresse no enrolamento e migração de
   óleo. Fabricantes especificam número máximo de partidas/hora.

2. REVERSÃO DE CICLO (resfria <-> aquece) — em equipamento reversível exige
   inverter a válvula de 4 vias. Envolve equalização de pressão e, em unidade
   real, o compressor deve parar durante a reversão. Alternar sentido é MUITO
   mais severo que modular dentro de um sentido. Esta métrica só existe porque
   habilitei aquecimento; antes não havia como ocorrer.

3. TEMPO MÍNIMO DE OPERAÇÃO — partir e desligar em poucos minutos não permite
   retorno de óleo ao cárter.

4. TAXA DE RAMPA (d carga/dt) — inverters têm limite de rampa; degraus abruptos
   estressam a eletrônica de potência e o compressor.

Modular suavemente 20x/h dentro de um sentido é benigno. Ciclar ON/OFF 20x/h ou
reverter o ciclo 5x/h não é. A distinção não aparece em `changes_per_hour`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import ClassroomConfig

# Limites de referência para split inverter residencial/comercial. Não são de uma
# folha de dados específica — são ordens de grandeza usuais em engenharia de
# refrigeração, e servem de critério de aprovação, não de verdade absoluta.
LIMITES = {
    "partidas_por_hora": 6.0,        # ~1 partida a cada 10 min
    "reversoes_por_hora": 2.0,       # reversão de válvula é severa
    "tempo_min_operacao_min": 5.0,   # retorno de óleo
    "rampa_max_por_min": 0.35,       # fração de carga por minuto
}


def _minutes_per_row(df: pd.DataFrame, config: ClassroomConfig) -> np.ndarray:
    """Minutos representados por cada linha (respeita action repeat)."""
    passos = (df["inner_steps"].to_numpy() if "inner_steps" in df
              else np.ones(len(df)))
    return passos * config.dt * 60.0


def compressor_metrics(df: pd.DataFrame, config: ClassroomConfig) -> Dict[str, float]:
    """
    Métricas de desgaste a partir do histórico de um episódio.

    Trabalha sobre `load` (carga com sinal) quando disponível; se o histórico for
    do modo discreto sem carga, cai para `action`.
    """
    if "load" in df:
        carga = df["load"].to_numpy(dtype=float)
    else:
        carga = df["action"].to_numpy(dtype=float)

    minutos = _minutes_per_row(df, config)
    horas = float(minutos.sum()) / 60.0
    if horas <= 0 or len(carga) < 2:
        return {}

    ligado = np.abs(carga) > 1e-6
    sentido = np.sign(carga)                      # -1 aquece, 0 off, +1 resfria

    # --- 1. Partidas do compressor (OFF -> ON) ---
    partidas = int(np.sum(~ligado[:-1] & ligado[1:]))

    # --- 2. Reversões de ciclo: troca de sentido ignorando as paradas ---
    # Comparar sentidos adjacentes contaria OFF como reversão; o que importa é a
    # sequência de sentidos ATIVOS.
    ativos = sentido[ligado]
    reversoes = int(np.sum(ativos[:-1] * ativos[1:] < 0)) if len(ativos) > 1 else 0

    # --- 3. Duração dos períodos ligados ---
    duracoes: List[float] = []
    atual = 0.0
    for on, m in zip(ligado, minutos):
        if on:
            atual += m
        elif atual > 0:
            duracoes.append(atual); atual = 0.0
    if atual > 0:
        duracoes.append(atual)
    duracoes_arr = np.asarray(duracoes) if duracoes else np.array([0.0])

    # --- 4. Taxa de rampa ---
    passo_min = float(np.mean(minutos))
    rampa = np.abs(np.diff(carga)) / max(passo_min, 1e-9)

    return {
        "partidas_dia": float(partidas),
        "partidas_por_hora": partidas / horas,
        "reversoes_dia": float(reversoes),
        "reversoes_por_hora": reversoes / horas,
        "tempo_op_mediano_min": float(np.median(duracoes_arr)),
        "tempo_op_minimo_min": float(duracoes_arr.min()),
        "op_curtas_pct": float(
            (duracoes_arr < LIMITES["tempo_min_operacao_min"]).mean() * 100.0
        ),
        "rampa_p95_por_min": float(np.quantile(rampa, 0.95)) if len(rampa) else 0.0,
        "fracao_ligado_pct": float(np.average(ligado, weights=minutos) * 100.0),
        "horas": horas,
    }


def avaliar_limites(m: Dict[str, float]) -> Dict[str, bool]:
    """Aprovação por critério. True = dentro do limite."""
    return {
        "partidas": m.get("partidas_por_hora", 0) <= LIMITES["partidas_por_hora"],
        "reversoes": m.get("reversoes_por_hora", 0) <= LIMITES["reversoes_por_hora"],
        "tempo_op": m.get("tempo_op_mediano_min", 0) >= LIMITES["tempo_min_operacao_min"],
        "rampa": m.get("rampa_p95_por_min", 0) <= LIMITES["rampa_max_por_min"],
    }


def relatorio(nome: str, m: Dict[str, float]) -> str:
    ok = avaliar_limites(m)
    marca = lambda b: "ok " if b else "!! "
    return (
        f"{nome}\n"
        f"  {marca(ok['partidas'])}partidas do compressor: {m['partidas_por_hora']:.2f}/h "
        f"({m['partidas_dia']:.0f}/dia)  [limite {LIMITES['partidas_por_hora']:.0f}/h]\n"
        f"  {marca(ok['reversoes'])}reversões de ciclo:     {m['reversoes_por_hora']:.2f}/h "
        f"({m['reversoes_dia']:.0f}/dia)  [limite {LIMITES['reversoes_por_hora']:.0f}/h]\n"
        f"  {marca(ok['tempo_op'])}tempo ligado (mediana): {m['tempo_op_mediano_min']:.1f} min "
        f"(mín {m['tempo_op_minimo_min']:.1f})  [limite {LIMITES['tempo_min_operacao_min']:.0f} min]\n"
        f"  {marca(ok['rampa'])}rampa p95:              {m['rampa_p95_por_min']:.3f}/min "
        f" [limite {LIMITES['rampa_max_por_min']:.2f}/min]\n"
        f"    ligado {m['fracao_ligado_pct']:.0f}% do tempo"
    )


def tabela(resultados: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Consolida em DataFrame com coluna de aprovação."""
    linhas = []
    for nome, m in resultados.items():
        ok = avaliar_limites(m)
        linhas.append({
            "controlador": nome,
            "partidas_h": round(m["partidas_por_hora"], 2),
            "reversoes_h": round(m["reversoes_por_hora"], 2),
            "t_op_med_min": round(m["tempo_op_mediano_min"], 1),
            "op_curtas_%": round(m["op_curtas_pct"], 1),
            "rampa_p95": round(m["rampa_p95_por_min"], 3),
            "ligado_%": round(m["fracao_ligado_pct"], 1),
            "aprovado": all(ok.values()),
            "falhas": ",".join(k for k, v in ok.items() if not v) or "-",
        })
    return pd.DataFrame(linhas)
