# -*- coding: utf-8 -*-
"""
Rigor estatístico e verificação do anti-short-cycling.

Responde a duas críticas do Revisor 2:

  "Três sementes também constituem um número pequeno para avaliar a
   variabilidade do treinamento. Intervalos de confiança e testes estatísticos
   ajudariam a verificar se as diferenças observadas são consistentes."

  "a permanência mínima indicada é de 36 minutos, mas os agentes realizam cerca
   de 2,5 mudanças por hora. A distribuição dos intervalos entre acionamentos
   deveria ser apresentada para verificar se a proteção foi efetiva."

A segunda crítica é grave e o revisor está certo: 2,5 trocas/h significa uma
troca a cada 24 min, abaixo do d_min de 36 min declarado. Ou a proteção não
funcionou, ou a métrica agregada esconde a distribuição. `dwell_time_analysis`
resolve empiricamente: mede cada intervalo entre comutações e reporta a fração
que viola d_min.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ClassroomConfig


# ------------------------------------------------------- intervalos e testes

def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    IC por bootstrap percentil. Devolve (média, inferior, superior).

    Bootstrap em vez de IC normal porque com 3–5 sementes a suposição de
    normalidade não se sustenta, e o t de Student com n=3 dá intervalos tão
    largos que nada é distinguível. O bootstrap não conserta n pequeno — apenas
    deixa de assumir o que não se pode verificar.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])

    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(v.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Teste de permutação bicaudal sobre a diferença de médias.

    Escolhido em vez do t-test porque não assume normalidade nem variâncias
    iguais — apropriado para comparar sementes de RL, cuja distribuição é
    tipicamente assimétrica e multimodal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float(a.mean() - b.mean())

    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[: a.size].mean() - pooled[a.size :].mean()
        if abs(diff) >= abs(observed):
            count += 1

    return {
        "diff": observed,
        "p_value": (count + 1) / (n_perm + 1),  # correção de continuidade
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """
    Tamanho de efeito não paramétrico (Cliff's delta) ∈ [−1, 1].

    Um p-valor com n=3 diz pouco; o tamanho de efeito diz quanto a diferença
    importa. Reportar ambos é o padrão mínimo em RL empírico.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {"delta": float("nan"), "magnitude": "indefinido"}

    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    delta = float((greater - less) / (a.size * b.size))

    m = abs(delta)
    magnitude = (
        "desprezível" if m < 0.147
        else "pequeno" if m < 0.33
        else "médio" if m < 0.474
        else "grande"
    )
    return {"delta": delta, "magnitude": magnitude}


def compare_agents(
    per_seed: Dict[str, Sequence[float]],
    metric_name: str = "métrica",
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """Tabela de ICs e, se `baseline` for dado, testes contra ele."""
    rows: List[Dict] = []
    for name, values in per_seed.items():
        mean, lo, hi = bootstrap_ci(values)
        row = {
            "agente": name,
            "n_sementes": len(values),
            f"{metric_name}_media": round(mean, 2),
            "ic95_inf": round(lo, 2),
            "ic95_sup": round(hi, 2),
        }
        if baseline and name != baseline and baseline in per_seed:
            test = permutation_test(values, per_seed[baseline])
            eff = cliffs_delta(values, per_seed[baseline])
            row.update({
                "diff_vs_baseline": round(test["diff"], 2),
                "p_valor": round(test["p_value"], 4),
                "cliffs_delta": round(eff["delta"], 2),
                "efeito": eff["magnitude"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------- anti-short-cycling (Revisor 2)

def dwell_times(df: pd.DataFrame, config: ClassroomConfig) -> np.ndarray:
    """
    Intervalos (em minutos) entre comutações consecutivas de nível.

    Trabalha em passos do AMBIENTE, não em linhas do DataFrame, para não
    confundir horizonte de decisão (action repeat) com tempo físico.
    """
    steps_per_row = (
        df["inner_steps"].to_numpy() if "inner_steps" in df
        else np.ones(len(df))
    )
    changed = df["action_changed"].to_numpy().astype(bool)

    minutes_per_step = config.dt * 60.0
    intervals: List[float] = []
    accumulated = 0.0
    for i in range(len(df)):
        accumulated += steps_per_row[i] * minutes_per_step
        if changed[i]:
            intervals.append(accumulated)
            accumulated = 0.0
    return np.asarray(intervals, dtype=float)


def dwell_time_analysis(
    dfs: Sequence[pd.DataFrame],
    config: ClassroomConfig,
) -> Dict[str, float]:
    """
    Verifica se a penalidade anti-short-cycling foi EFETIVA.

    A crítica do Revisor 2 é aritmética: 2,5 trocas/h ⇒ uma troca a cada 24 min,
    abaixo do d_min de 36 min. Um agregado como "trocas/h" não distingue "sempre
    respeita 36 min" de "metade das trocas viola gravemente". Esta função reporta
    a distribuição e a fração de violações — a evidência que o revisor pediu.
    """
    all_intervals = np.concatenate(
        [dwell_times(df, config) for df in dfs if len(df)]
    ) if dfs else np.array([])

    if all_intervals.size == 0:
        return {"n_comutacoes": 0, "violacoes_pct": 0.0}

    d_min = config.min_dwell_minutes
    return {
        "n_comutacoes": int(all_intervals.size),
        "d_min_min": d_min,
        "mediana_min": float(np.median(all_intervals)),
        "media_min": float(all_intervals.mean()),
        "p10_min": float(np.quantile(all_intervals, 0.10)),
        "p90_min": float(np.quantile(all_intervals, 0.90)),
        "min_min": float(all_intervals.min()),
        # A métrica decisiva: proteção efetiva ⇒ este valor deve ser baixo.
        "violacoes_pct": float((all_intervals < d_min).mean() * 100.0),
        "trocas_por_hora_implicita": float(60.0 / all_intervals.mean()),
    }


def format_dwell_report(name: str, stats: Dict[str, float]) -> str:
    if not stats.get("n_comutacoes"):
        return f"{name}: nenhuma comutação registrada"
    return (
        f"{name}\n"
        f"  comutações: {stats['n_comutacoes']}  |  d_min = {stats['d_min_min']:.0f} min\n"
        f"  permanência: mediana {stats['mediana_min']:.1f} min, "
        f"média {stats['media_min']:.1f} min, "
        f"p10 {stats['p10_min']:.1f}, p90 {stats['p90_min']:.1f}, "
        f"mín {stats['min_min']:.1f}\n"
        f"  VIOLAÇÕES de d_min: {stats['violacoes_pct']:.1f}%  "
        f"(trocas/h implícita: {stats['trocas_por_hora_implicita']:.2f})"
    )
