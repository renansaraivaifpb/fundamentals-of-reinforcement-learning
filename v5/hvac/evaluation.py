# -*- coding: utf-8 -*-
"""
Avaliação com múltiplas sementes e incerteza reportada.

O PROBLEMA QUE ISTO RESOLVE
---------------------------
Na v4 o `evaluate.py` — o script que produz a comparação central entre o PI e os
perfis DQN — roda com `--seed` de valor único (default 0). Com ruído de processo
no ambiente e ocupação estocástica, a diferença reportada entre 0,72 °C do PI e
0,77 °C do DQN Agressivo foi lida de uma amostra de tamanho UM.

O detalhe revelador é que o próprio projeto já sabia fazer melhor: o
`evaluate_lab2.py`, escrito depois, roda `args.eval_seeds` e tira média. O
experimento mais recente era metodologicamente superior àquele que sustenta a
conclusão principal — e mesmo ele reporta só a média, sem dispersão, de modo que
não dá para saber se sigma 0,088 contra 0,198 sobrevive ao ruído.

Aqui a unidade de reporte é sempre (média, IC95, n). Uma diferença entre dois
controladores só é afirmável se os intervalos não se sobrepõem — ou, melhor, se
o teste pareado sobre as MESMAS sementes a sustenta.

O EMPARELHAMENTO IMPORTA
------------------------
Todos os controladores são avaliados exatamente nas mesmas sementes. Isso
transforma a comparação em pareada: o ruído do ambiente é comum aos dois braços e
cancela na diferença, o que dá muito mais poder estatístico do que comparar duas
médias independentes com n pequeno.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ClassroomConfig
from .metrics import evaluate_agent
from .stats_analysis import bootstrap_ci, cliffs_delta, permutation_test

# Sementes de avaliação DISJUNTAS das de treino (0,1,2), para que o desempenho
# não seja lido nas mesmas realizações de ruído em que a política foi ajustada.
EVAL_SEEDS: Sequence[int] = (100, 101, 102, 103, 104)

METRICAS_PADRAO = ("comfort_wide_pct", "comfort_narrow_pct", "abs_dev_from_ideal",
                   "in_tolerance_pct", "temp_std", "energy_kwh_day",
                   "cost_brl_day", "changes_per_hour")


def avaliar_multissemente(
    agente_factory: Callable[[], object],
    env_factory: Callable[[], object],
    scenarios: Sequence[Dict],
    config: ClassroomConfig,
    seeds: Sequence[int] = EVAL_SEEDS,
    pass_info_to_agent: bool = False,
) -> Dict[str, object]:
    """
    Avalia o agente em cada semente e devolve valores por semente + resumo.

    `agente_factory` é uma callable, e não um agente pronto, porque controladores
    com estado (integrador do PI, histerese do termostato) precisam nascer limpos
    a cada semente — reaproveitar a instância faria o estado de uma avaliação
    vazar para a seguinte.
    """
    por_semente: List[Dict[str, float]] = []
    for s in seeds:
        r = evaluate_agent(agente_factory(), env_factory, scenarios, config,
                           seed=int(s), pass_info_to_agent=pass_info_to_agent)
        linha = {"seed": int(s)}
        linha.update({k: float(v) for k, v in r["summary"].items()
                      if isinstance(v, (int, float))})
        por_semente.append(linha)

    df = pd.DataFrame(por_semente)
    resumo: Dict[str, Dict[str, float]] = {}
    for m in df.columns:
        if m == "seed":
            continue
        vals = df[m].dropna().to_numpy()
        if not len(vals):
            continue
        _, baixo, alto = bootstrap_ci(vals)   # (média, limite inf., limite sup.)
        resumo[m] = {
            "media": float(np.mean(vals)),
            "desvio": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ic_baixo": float(baixo),
            "ic_alto": float(alto),
            "n": int(len(vals)),
        }
    return {"por_semente": df, "resumo": resumo, "seeds": list(seeds)}


def comparar(
    resultados: Dict[str, Dict[str, object]],
    referencia: str,
    metricas: Sequence[str] = METRICAS_PADRAO,
) -> pd.DataFrame:
    """
    Compara cada controlador contra a referência, nas MESMAS sementes.

    Reporta tamanho de efeito junto do valor-p porque, com poucas sementes, um
    valor-p diz pouco: o menor p bicaudal alcançável em teste de permutação com
    n = m = 3 é 0,10, de modo que "p = 0,101" é o piso e não um resultado
    marginal. O delta de Cliff não tem esse teto.
    """
    if referencia not in resultados:
        raise KeyError(f"referência '{referencia}' ausente. Há: {list(resultados)}")

    base = resultados[referencia]["por_semente"]
    linhas = []
    for nome, r in resultados.items():
        if nome == referencia:
            continue
        df = r["por_semente"]
        for m in metricas:
            if m not in df or m not in base:
                continue
            a, b = df[m].to_numpy(), base[m].to_numpy()
            d = cliffs_delta(a, b)
            perm = permutation_test(a, b)
            linhas.append({
                "controlador": nome, "metrica": m,
                "media": float(np.mean(a)),
                f"media_{referencia}": float(np.mean(b)),
                "diferenca": float(np.mean(a) - np.mean(b)),
                "cliffs_delta": d["delta"],
                "magnitude": d["magnitude"],
                "p_permutacao": perm["p_value"],
            })
    return pd.DataFrame(linhas)


def tabela_resumo(
    resultados: Dict[str, Dict[str, object]],
    metricas: Sequence[str] = METRICAS_PADRAO,
    decimais: int = 3,
) -> pd.DataFrame:
    """Tabela publicável: 'média ± desvio' por controlador e métrica."""
    linhas = []
    for nome, r in resultados.items():
        linha = {"controlador": nome, "n_seeds": len(r["seeds"])}
        for m in metricas:
            if m not in r["resumo"]:
                continue
            e = r["resumo"][m]
            linha[m] = f"{e['media']:.{decimais}f} ± {e['desvio']:.{decimais}f}"
        linhas.append(linha)
    return pd.DataFrame(linhas)


def diferenca_afirmavel(
    resultados: Dict[str, Dict[str, object]],
    a: str, b: str, metrica: str,
) -> Dict[str, object]:
    """
    Responde à única pergunta que importa: dá para afirmar que A difere de B?

    Critério deliberadamente conservador — exige separação COMPLETA entre as
    sementes (|delta de Cliff| = 1). Com n pequeno, qualquer critério mais frouxo
    produz afirmações que não sobrevivem a uma semente a mais.
    """
    va = resultados[a]["por_semente"][metrica].to_numpy()
    vb = resultados[b]["por_semente"][metrica].to_numpy()
    delta = cliffs_delta(va, vb)["delta"]
    return {
        "metrica": metrica,
        f"{a}": float(np.mean(va)), f"{b}": float(np.mean(vb)),
        "diferenca": float(np.mean(va) - np.mean(vb)),
        "cliffs_delta": delta,
        "separacao_completa": abs(delta) == 1.0,
        "afirmavel": bool(abs(delta) == 1.0),
        "n": int(min(len(va), len(vb))),
    }
