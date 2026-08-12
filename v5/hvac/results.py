# -*- coding: utf-8 -*-
"""
Coleta dos resultados — separada do desenho.

Por que separar `results` de `figures`: notebooks e o gerador do artigo precisam
dos MESMOS números. Se cada um recomputasse à sua maneira, tabela e figura
voltariam a divergir — o defeito que a v4 tinha, com PNGs vindos de execuções
diferentes das que produziram os números reportados.

Aqui há uma única função por resultado. Quem quiser a tabela e quem quiser o
gráfico chamam a mesma função.

Resultados CAROS (treino) são lidos dos CSVs canônicos; resultados BARATOS
(rodar um controlador na matriz de cenários) são recomputados na hora, o que os
mantém sempre coerentes com o código vigente.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Os artefatos de treino (modelos, CSVs de ablação) vivem na v4, que foi onde os
# experimentos rodaram. A v5 os LÊ; não os duplica.
V4 = os.path.join(os.path.dirname(__file__), "..", "..", "v4", "paper")


def _p(*partes) -> str:
    return os.path.normpath(os.path.join(V4, *partes))


# ------------------------------------------------------- infraestrutura comum

def _fabrica(cfg, repeat: int = 2, dwell: bool = False, continuo: bool = False):
    """
    Fábrica de ambientes respeitando o contrato do agente.

    `continuo` aplica o `ContinuousActionWrapper`, exigido pelos agentes de ação
    contínua (SAC) no regime do manuscrito, onde o ambiente base é discreto.
    Avaliar um SAC sem ele produziria erro de espaço de ação — ou, pior, uma
    coerção silenciosa que mediria outra política.
    """
    from .env import ClassroomACEnv
    from .wrappers import (ActionRepeatWrapper, ContinuousActionWrapper,
                           MinDwellWrapper)

    def f():
        env = ClassroomACEnv(config=cfg)
        if continuo and not cfg.continuous_action:
            env = ContinuousActionWrapper(env)
        if dwell:
            env = MinDwellWrapper(env)
        return ActionRepeatWrapper(env, repeat=repeat)
    return f


# =========================================================== 1. decomposição

def decomposicao_da_vantagem() -> pd.DataFrame:
    """
    Recomputa a decomposição dos "+32 pp": quanto vem de configurar o baseline,
    quanto de controle clássico e quanto de aprendizado.

    É o resultado central da auditoria, e é barato — por isso recomputado, e não
    lido de arquivo.
    """
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import evaluate_agent
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cfg = config_for_profile("Equilibrado")
    cen = build_scenario_matrix()
    fab = _fabrica(cfg)

    etapas = [
        ("Termostato (zona morta = 0)", ThermostatAgent(cfg, deadband=0.0),
         "baseline do manuscrito"),
        ("+ histerese (zona morta = 1 °C)", ThermostatAgent(cfg, deadband=1.0),
         "configuração do baseline"),
        ("PI sintonizado (Kp=1,3; Ki=0,2)", PIController(cfg, kp=1.3, ki=0.2),
         "controle clássico"),
    ]
    linhas = []
    for rot, ag, atrib in etapas:
        s = evaluate_agent(ag, fab, cen, cfg, seed=0,
                           pass_info_to_agent=True)["summary"]
        linhas.append({"etapa": rot, "conforto_larga_pct": s["comfort_wide_pct"],
                       "atribuivel_a": atrib})

    modelo, cfg_t, meta = load_agent(_p("models_paper", "DQN_Equilibrado_seed0.zip"))
    s = evaluate_agent(modelo, _fabrica(cfg_t, meta.get("action_repeat", 2)),
                       cen, cfg_t, seed=0)["summary"]
    linhas.append({"etapa": "DQN (550k passos)",
                   "conforto_larga_pct": s["comfort_wide_pct"],
                   "atribuivel_a": "aprendizado"})

    df = pd.DataFrame(linhas)
    df["ganho_pp"] = df["conforto_larga_pct"].diff().fillna(
        df["conforto_larga_pct"].iloc[0])
    return df


def comparacao_controladores() -> pd.DataFrame:
    """Tabela principal: PI, três perfis DQN e os dois termostatos."""
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import evaluate_agent
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cfg = config_for_profile("Equilibrado")
    cen = build_scenario_matrix()
    campos = ("comfort_wide_pct", "comfort_narrow_pct", "abs_dev_from_ideal",
              "overheat_pct", "energy_kwh_day", "cost_brl_day", "changes_per_hour")

    linhas = []
    for rot, ag in [("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2)),
                    ("Termostato zona morta = 1 °C", ThermostatAgent(cfg, deadband=1.0)),
                    ("Termostato zona morta = 0", ThermostatAgent(cfg, deadband=0.0))]:
        s = evaluate_agent(ag, _fabrica(cfg), cen, cfg, seed=0,
                           pass_info_to_agent=True)["summary"]
        linhas.append({"controlador": rot, **{k: s[k] for k in campos}})

    # DQN (discreto) e SAC (contínuo). O SAC é a comparação discreto x contínuo
    # da Tabela 5 do manuscrito, e omiti-lo deixaria a auditoria incompleta.
    for arq, rot, continuo in [
        ("DQN_Agressivo_seed0.zip", "DQN Agressivo (550k)", False),
        ("DQN_Equilibrado_seed0.zip", "DQN Equilibrado (550k)", False),
        ("DQN_Passivo_seed0.zip", "DQN Passivo (550k)", False),
        ("SAC_Equilibrado_seed0.zip", "SAC Equilibrado (550k)", True),
    ]:
        caminho = _p("models_paper", arq)
        if not os.path.exists(caminho):
            continue
        m, c, meta = load_agent(caminho)
        s = evaluate_agent(
            m, _fabrica(c, meta.get("action_repeat", 2), continuo=continuo),
            cen, c, seed=0)["summary"]
        linhas.append({"controlador": rot, **{k: s[k] for k in campos}})

    ordem = ["PI sintonizado", "DQN Agressivo (550k)", "DQN Equilibrado (550k)",
             "DQN Passivo (550k)", "SAC Equilibrado (550k)",
             "Termostato zona morta = 1 °C", "Termostato zona morta = 0"]
    df = pd.DataFrame(linhas)
    return df.set_index("controlador").reindex(
        [o for o in ordem if o in set(df["controlador"])]).reset_index()


# ================================================================ 2. ablação

def ablacao() -> pd.DataFrame:
    """Ablação da recompensa: 8 variantes × 3 sementes (lido do CSV canônico)."""
    raw = pd.read_csv(_p("results_ablation", "ablation_raw.csv"))
    nomes = {
        "completa": "completa (proposta)",
        "convencional": "convencional (quadrática pura)",
        "sem_short_cycling": "sem anti-short-cycling",
        "sem_penal_frio": "sem penalidade de frio",
        "quadratica_pura": "quadrática pura",
        "sem_penal_troca": "sem penalidade de troca",
        "degraus_legado": "degraus (legado)",
        "sem_gradiente": "sem gradiente interno",
    }
    g = (raw.groupby("variante")["conf_estreita_pct"]
            .agg(media="mean", minimo="min", maximo="max", desvio="std")
            .reindex(list(nomes)))
    g["rotulo"] = [nomes[i] for i in g.index]

    from .stats_analysis import cliffs_delta
    ref = raw.loc[raw["variante"] == "completa", "conf_estreita_pct"].to_numpy()
    d = [cliffs_delta(raw.loc[raw["variante"] == v, "conf_estreita_pct"].to_numpy(),
                      ref) for v in g.index]
    g["cliffs_delta"] = [x["delta"] for x in d]
    g["magnitude"] = [x["magnitude"] for x in d]
    g.loc["completa", ["cliffs_delta", "magnitude"]] = [np.nan, "referência"]
    return g.reset_index()


def ablacao_por_semente() -> pd.DataFrame:
    return pd.read_csv(_p("results_ablation", "ablation_raw.csv"))


# ========================================================= 3. short-cycling

def permanencia() -> Dict[str, object]:
    """
    Distribuição dos tempos de permanência, sob penalidade mole e sob restrição
    dura. Recomputado: depende da política, e é barato.
    """
    from .metrics import is_controllable, run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix
    from .stats_analysis import dwell_times

    modelo, cfg, meta = load_agent(_p("models_paper", "DQN_Equilibrado_seed0.zip"))
    rep = meta.get("action_repeat", 2)
    cen = build_scenario_matrix()

    def coletar(dura: bool) -> np.ndarray:
        fab = _fabrica(cfg, rep, dwell=dura)
        dfs = [run_episode(modelo, fab(), c, seed=0)
               for c in cen if is_controllable(fab(), c, seed=0)]
        return np.concatenate([dwell_times(df, cfg) for df in dfs if len(df)])

    mole, dura = coletar(False), coletar(True)
    d_min = cfg.min_dwell_minutes
    resumo = pd.DataFrame([
        {"regime": "Penalidade de recompensa (eq. 5)", "n": len(mole),
         "mediana_min": float(np.median(mole)), "media_min": float(mole.mean()),
         "minimo_min": float(mole.min()),
         "violacoes_pct": float((mole < d_min).mean() * 100)},
        {"regime": "Restrição dura (MinDwell)", "n": len(dura),
         "mediana_min": float(np.median(dura)), "media_min": float(dura.mean()),
         "minimo_min": float(dura.min()),
         "violacoes_pct": float((dura < d_min).mean() * 100)},
    ])
    return {"mole": mole, "dura": dura, "d_min": d_min, "resumo": resumo}


def permanencia_por_agente() -> pd.DataFrame:
    """Permanência dos três perfis DQN e do termostato."""
    from .baselines import ThermostatAgent
    from .config import config_for_profile
    from .metrics import is_controllable, run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix
    from .stats_analysis import dwell_time_analysis

    cen = build_scenario_matrix()
    linhas = []
    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        caminho = _p("models_paper", f"DQN_{perfil}_seed0.zip")
        if not os.path.exists(caminho):
            continue
        m, cfg, meta = load_agent(caminho)
        fab = _fabrica(cfg, meta.get("action_repeat", 2))
        dfs = [run_episode(m, fab(), c, seed=0)
               for c in cen if is_controllable(fab(), c, seed=0)]
        s = dwell_time_analysis(dfs, cfg)
        linhas.append({"agente": f"DQN {perfil}", **s})

    cfg = config_for_profile("Equilibrado")
    fab = _fabrica(cfg)
    dfs = [run_episode(ThermostatAgent(cfg, deadband=0.0), fab(), c, seed=0,
                       pass_info_to_agent=True)
           for c in cen if is_controllable(fab(), c, seed=0)]
    linhas.append({"agente": "Termostato", **dwell_time_analysis(dfs, cfg)})
    return pd.DataFrame(linhas)


# ==================================================== 4. regimes estendidos

def laboratorio_precisao() -> pd.DataFrame:
    """Regime de precisão (lab2): TD3/SAC contra o PI bidirecional."""
    return pd.read_csv(_p("results_lab2.csv"))


def tabular() -> pd.DataFrame:
    """Q-Learning tabular: o diagnóstico de que RL precisa ser profundo."""
    return pd.read_csv(_p("results_tabular.csv"))


def dimensionamento_demanda() -> Dict[str, object]:
    """
    Demanda contratada: o valor derivado da condição de projeto contra o valor
    arbitrado por varredura, e a pressão que cada um exerce nos cenários.
    """
    from dataclasses import replace

    from .baselines import DemandAwarePI, PIController
    from .config import config_for_lab2
    from .metrics import episode_metrics, run_episode
    from .scenarios import build_demand_scenario_matrix, build_scenario_matrix

    base = config_for_lab2("Lab_Equilibrado")
    antigo = replace(base, demand_contracted_kw_manual=0.70)

    def varre(cfg, Agente, cen):
        picos, tol = [], []
        for c in cen:
            env = _fabrica(cfg)()
            df = run_episode(Agente(cfg, kp=1.3, ki=0.2, discrete=False), env, c,
                             seed=0, pass_info_to_agent=True)
            m = episode_metrics(df, cfg)
            picos.append(m["demand_peak_kw"])
            tol.append(m["in_tolerance_ss_pct"])
        return np.array(picos), float(np.nanmean(tol))

    cen_d = build_demand_scenario_matrix()
    linhas = []
    for cfg, tag in [(antigo, "0,70 kW (varredura)"),
                     (base, f"{base.demand_contracted_kw:.3f} kW (derivado)")]:
        for Ag, nome in [(PIController, "PI ingênuo"),
                         (DemandAwarePI, "PI ciente da demanda")]:
            picos, tol = varre(cfg, Ag, cen_d)
            lim = cfg.demand_contracted_kw
            linhas.append({
                "contrato": tag, "controlador": nome,
                "tolerancia_pct": tol, "pico_max_kw": float(picos.max()),
                "excedente_kw": float(max(0.0, picos.max() - lim)),
                "cenarios_violando": int((picos > lim + 1e-9).sum()),
                "n_cenarios": len(cen_d),
            })

    # De onde vem a pressão: só os cenários que partem FORA do setpoint estouram.
    picos_c, _ = varre(base, PIController, build_scenario_matrix())
    origem = pd.DataFrame({
        "cenario": [c["id"] for c in build_scenario_matrix()],
        "temp_inicial": [c["start_temp"] for c in build_scenario_matrix()],
        "pico_kw": picos_c,
        "excede": picos_c > base.demand_contracted_kw,
    })
    return {"comparacao": pd.DataFrame(linhas), "origem_da_pressao": origem,
            "sizing": base.demand_sizing}


# ============================================ 5. trajetórias e faixa estreita

def trajetorias(cenarios: Optional[List[Dict]] = None) -> Dict[str, object]:
    """
    Séries de 24 h por cenário, para DQN, termostato e PI.

    Devolve temperatura E carga do equipamento: ver só a temperatura mostra
    QUANTO cada controlador acerta, mas não COMO — e a diferença de mecanismo
    (comutar em degraus contra modular continuamente) é o que explica o
    resultado.
    """
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cen = cenarios if cenarios is not None else build_scenario_matrix()
    cfg = config_for_profile("Equilibrado")
    modelo, cfg_dqn, meta = load_agent(_p("models_paper", "DQN_Equilibrado_seed0.zip"))
    rep = meta.get("action_repeat", 2)

    controladores = [
        ("DQN Equilibrado", modelo, cfg_dqn, rep, False),
        ("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2), cfg, 2, True),
        ("Termostato (zm = 0)", ThermostatAgent(cfg, deadband=0.0), cfg, 2, True),
    ]
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for c in cen:
        out[c["id"]] = {"cenario": c}
        for nome, ag, cfg_a, r, info in controladores:
            df = run_episode(ag, _fabrica(cfg_a, r)(), c, seed=0,
                             pass_info_to_agent=info)
            out[c["id"]][nome] = df
    return {"series": out, "config": cfg,
            "controladores": [c[0] for c in controladores]}


def sensibilidade_a_largura(
    tolerancias: tuple = (2.0, 1.5, 1.0, 0.75, 0.5, 0.25),
) -> pd.DataFrame:
    """
    Aperta a RÉGUA sem retreinar nada: mede os mesmos controladores sob
    tolerâncias progressivamente menores.

    Responde "quem degrada menos quando a exigência aumenta?" — pergunta
    distinta de "quem vence quando ambos são preparados para a exigência", que
    exige retreinamento e está em `experimentos/faixa_estreita.py`.
    """
    from dataclasses import replace

    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import evaluate_agent
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cen = build_scenario_matrix()
    base = config_for_profile("Equilibrado")
    modelo, cfg_dqn, meta = load_agent(_p("models_paper", "DQN_Equilibrado_seed0.zip"))
    rep = meta.get("action_repeat", 2)

    linhas = []
    for tol in tolerancias:
        cfg = replace(base, lab_tolerance=tol)
        cfg_d = replace(cfg_dqn, lab_tolerance=tol)
        for nome, ag, c, r, info in [
            ("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2), cfg, 2, True),
            ("DQN Equilibrado", modelo, cfg_d, rep, False),
            ("Termostato (zm = 1 °C)", ThermostatAgent(cfg, deadband=1.0), cfg, 2, True),
            ("Termostato (zm = 0)", ThermostatAgent(cfg, deadband=0.0), cfg, 2, True),
        ]:
            s = evaluate_agent(ag, _fabrica(c, r), cen, c, seed=0,
                               pass_info_to_agent=info)["summary"]
            linhas.append({"tolerancia": tol, "controlador": nome,
                           "na_tolerancia_pct": s["in_tolerance_pct"],
                           "desvio_ideal": s["abs_dev_from_ideal"],
                           "sigma": s["temp_std"]})
    return pd.DataFrame(linhas)


def faixa_estreita() -> Optional[pd.DataFrame]:
    """
    Resultado do experimento com RETREINO por largura de faixa.

    Caro (9 treinos), portanto lido do CSV — gerado por
    `experimentos/faixa_estreita.py`.
    """
    caminho = os.path.join(os.path.dirname(__file__), "..", "experimentos",
                           "resultados_faixa_estreita.csv")
    caminho = os.path.normpath(caminho)
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)
