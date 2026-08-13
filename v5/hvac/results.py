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


# Diretório de modelos vigente. `None` usa os treinados sob os parâmetros
# INFERIDOS, que reproduzem o manuscrito; apontar para outro diretório permite
# reportar a mesma bateria de análises sob outra configuração de recompensa sem
# duplicar código nem arriscar que as duas se misturem.
_DIR_MODELOS: Optional[str] = None


def usar_modelos(caminho: Optional[str]) -> None:
    """Escolhe de qual diretório os agentes treinados serão carregados."""
    global _DIR_MODELOS
    _DIR_MODELOS = caminho


def modelos_vigentes() -> str:
    return _DIR_MODELOS or _p("models_paper")


def _pm(arquivo: str) -> str:
    """Caminho de um modelo no diretório vigente."""
    return os.path.normpath(os.path.join(modelos_vigentes(), arquivo))


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

    modelo, cfg_t, meta = load_agent(_pm("DQN_Equilibrado_seed0.zip"))
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
        caminho = _pm(arq)
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

    modelo, cfg, meta = load_agent(_pm("DQN_Equilibrado_seed0.zip"))
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
        caminho = _pm(f"DQN_{perfil}_seed0.zip")
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
    modelo, cfg_dqn, meta = load_agent(_pm("DQN_Equilibrado_seed0.zip"))
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
    modelo, cfg_dqn, meta = load_agent(_pm("DQN_Equilibrado_seed0.zip"))
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


# ================================== 6. decomposição do consumo (Yuan et al.)

def consumo_decomposto() -> Dict[str, pd.DataFrame]:
    """
    De ONDE vem a diferença de energia entre os controladores.

    Inspirado na Fig. 11 de Yuan et al., que separa o consumo por item do
    sistema (resfriamento, transmissão/distribuição). O análogo aqui, num split
    de 4 níveis, é decompor por NÍVEL DE POTÊNCIA acionado e por POSTO TARIFÁRIO.

    Por que isso importa: as tabelas mostram que o DQN gasta ~4,7 % mais que o
    PI, mas não mostram EM QUÊ. Se o excedente estiver concentrado em HIGH, a
    causa é o padrão de comutação; se estiver espalhado, é o ponto de operação.
    A resposta muda o que se recomenda ao engenheiro.
    """
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import is_controllable, run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cen = build_scenario_matrix()
    cfg = config_for_profile("Equilibrado")
    modelo, cfg_d, meta = load_agent(_pm("DQN_Equilibrado_seed0.zip"))

    controladores = [
        ("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2), cfg, 2, True, False),
        ("DQN Equilibrado", modelo, cfg_d, meta.get("action_repeat", 2), False, False),
        ("Termostato (zm = 0)", ThermostatAgent(cfg, deadband=0.0), cfg, 2, True, False),
    ]
    try:
        sac, cfg_s, meta_s = load_agent(_pm("SAC_Equilibrado_seed0.zip"))
        controladores.append(("SAC Equilibrado", sac, cfg_s,
                              meta_s.get("action_repeat", 2), False, True))
    except Exception:
        pass

    por_nivel, por_posto = [], []
    for nome, ag, c, rep, info, continuo in controladores:
        fab = _fabrica(c, rep, continuo=continuo)
        # TODOS os cenários, sem o filtro de controlabilidade: é assim que
        # `evaluate_agent` promedia a energia (o filtro vale para o conforto,
        # não para o consumo). Filtrar aqui faria a decomposição não somar o
        # total reportado na tabela — divergência figura/tabela, justamente o
        # defeito que esta camada existe para impedir.
        dfs = [run_episode(ag, fab(), s, seed=0, pass_info_to_agent=info)
               for s in cen]
        df = pd.concat(dfs, ignore_index=True)
        # Energia por passo já vem em kWh; a soma sobre os cenários é dividida
        # pelo número de dias simulados para ficar em kWh/dia comparável.
        dias = len(dfs)
        g = df.groupby("ac_state")["energy_kwh"].sum() / dias
        for nivel in ("OFF", "LOW", "MEDIUM", "HIGH"):
            por_nivel.append({"controlador": nome, "nivel": nivel,
                              "kwh_dia": float(g.get(nivel, 0.0))})
        if "posto" in df:
            gp = df.groupby("posto")["energy_kwh"].sum() / dias
            for posto, v in gp.items():
                por_posto.append({"controlador": nome, "posto": str(posto),
                                  "kwh_dia": float(v)})
    return {"por_nivel": pd.DataFrame(por_nivel),
            "por_posto": pd.DataFrame(por_posto)}


def curva_aprendizado() -> Optional[pd.DataFrame]:
    """Desempenho contra orçamento de treino (caro; lido do CSV)."""
    caminho = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experimentos",
        "resultados_curva_aprendizado.csv"))
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


def boptest_transferencia() -> Optional[pd.DataFrame]:
    """
    Transferência para o emulador de terceiros (caro; lido do CSV).

    Gerado por `experimentos/boptest_transferencia.py`, que exige o serviço
    BOPTEST em execução. A ordem das linhas é imposta aqui, e não no CSV, para
    que a tabela do artigo leia como argumento: os dois PI primeiro, porque a
    comparação entre eles é o achado da seção.
    """
    caminho = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experimentos",
        "resultados_boptest.csv"))
    if not os.path.exists(caminho):
        return None
    df = pd.read_csv(caminho)
    ordem = ["PI re-sintonizado (emulador)", "PI (ganhos locais, congelados)",
             "DQN Agressivo", "DQN Equilibrado", "DQN Passivo",
             "Termostato zm=1 °C", "Termostato zm=0 (baseline)"]
    periodos = ["peak_cool_day", "typical_cool_day"]
    df["_ord"] = df["controlador"].apply(
        lambda c: ordem.index(c) if c in ordem else len(ordem))
    df["_per"] = df["periodo"].apply(
        lambda p: periodos.index(p) if p in periodos else len(periodos))
    return (df.sort_values(["_per", "_ord"])
              .drop(columns=["_ord", "_per"]).reset_index(drop=True))


def boptest_sintonia() -> Optional[pd.DataFrame]:
    """Busca em grade do PI dentro do emulador (`boptest_sintonia_pi.py`)."""
    caminho = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experimentos",
        "resultados_boptest_sintonia.csv"))
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


def boptest_resumo() -> Optional[Dict]:
    """
    Números que o texto da seção cita, derivados e não digitados.

    Reunir aqui o que o parágrafo afirma impede a divergência clássica entre um
    número no corpo do texto e a tabela ao lado — a mesma garantia que vale para
    o restante do artigo.
    """
    df = boptest_transferencia()
    if df is None or df.empty:
        return None

    def _pega(periodo: str, ctrl: str, col: str) -> Optional[float]:
        s = df[(df["periodo"] == periodo) & (df["controlador"] == ctrl)][col]
        return float(s.iloc[0]) if len(s) else None

    out: Dict = {"periodos": list(dict.fromkeys(df["periodo"]))}
    for per in out["periodos"]:
        sub = df[df["periodo"] == per]
        agentes = sub[sub["controlador"].str.startswith("DQN")]
        melhor = (agentes.sort_values("comfort_wide_pct", ascending=False)
                  .iloc[0] if len(agentes) else None)
        congelado = _pega(per, "PI (ganhos locais, congelados)",
                          "comfort_wide_pct")
        resint = _pega(per, "PI re-sintonizado (emulador)", "comfort_wide_pct")
        out[per] = {
            "pi_congelado": congelado,
            "pi_resintonizado": resint,
            "ganho_resintonia_pp": (None if None in (congelado, resint)
                                    else resint - congelado),
            "melhor_agente": None if melhor is None else melhor["controlador"],
            "melhor_agente_conf": None if melhor is None
            else float(melhor["comfort_wide_pct"]),
            "vantagem_agente_sobre_pi_congelado_pp": (
                None if melhor is None or congelado is None
                else float(melhor["comfort_wide_pct"]) - congelado),
            "vantagem_pi_resintonizado_pp": (
                None if melhor is None or resint is None
                else resint - float(melhor["comfort_wide_pct"])),
            "externa_min": float(sub["externa_min"].iloc[0]),
            "externa_max": float(sub["externa_max"].iloc[0]),
        }
    return out


def teste_integral() -> Optional[pd.DataFrame]:
    """Hipótese do integrador — resultado NEGATIVO (ver notebook 04)."""
    caminho = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experimentos",
        "resultados_teste_integral.csv"))
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


# ========================= 7. cenários aleatórios e uso dos níveis de potência

def uso_dos_niveis() -> pd.DataFrame:
    """
    Fração do tempo em cada nível de potência, por controlador.

    Existe porque a decomposição de energia revelou que os agentes DQN nunca
    acionam MEDIUM — o nível de melhor COP. Aqui a medida é de TEMPO, não de
    energia, para separar "usa pouco" de "não usa".
    """
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cen = build_scenario_matrix()
    cfg = config_for_profile("Equilibrado")
    linhas = []

    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        caminho = _pm(f"DQN_{perfil}_seed0.zip")
        if not os.path.exists(caminho):
            continue
        m, c, meta = load_agent(caminho)
        fab = _fabrica(c, meta.get("action_repeat", 2))
        df = pd.concat([run_episode(m, fab(), s, seed=0) for s in cen])
        pct = df["ac_state"].value_counts(normalize=True) * 100
        linhas.append({"controlador": f"DQN {perfil}",
                       **{n: float(pct.get(n, 0.0))
                          for n in ("OFF", "LOW", "MEDIUM", "HIGH")}})

    try:
        sac, cs, ms = load_agent(_pm("SAC_Equilibrado_seed0.zip"))
        fab = _fabrica(cs, ms.get("action_repeat", 2), continuo=True)
        df = pd.concat([run_episode(sac, fab(), s, seed=0) for s in cen])
        pct = df["ac_state"].value_counts(normalize=True) * 100
        linhas.append({"controlador": "SAC Equilibrado",
                       **{n: float(pct.get(n, 0.0))
                          for n in ("OFF", "LOW", "MEDIUM", "HIGH")}})
    except Exception:
        pass

    for nome, ag in [("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2)),
                     ("Termostato (zm = 0)", ThermostatAgent(cfg, deadband=0.0))]:
        fab = _fabrica(cfg)
        df = pd.concat([run_episode(ag, fab(), s, seed=0, pass_info_to_agent=True)
                        for s in cen])
        pct = df["ac_state"].value_counts(normalize=True) * 100
        linhas.append({"controlador": nome,
                       **{n: float(pct.get(n, 0.0))
                          for n in ("OFF", "LOW", "MEDIUM", "HIGH")}})
    return pd.DataFrame(linhas)


def cenarios_aleatorios(n: int = 200, seed: int = 12345,
                        tolerancia: Optional[float] = None) -> pd.DataFrame:
    """
    Avaliação num conjunto AMPLO e independente, cenário a cenário.

    A matriz 3x3 tem nove pontos e foi usada para sintonizar o PI; um conjunto
    aleatório sobre o espaço contínuo (temperatura, ocupação, hora), com semente
    fixa e distinta das de treino, é o conjunto de teste independente que o
    parecer pede. Todos os controladores veem EXATAMENTE os mesmos cenários, o
    que torna a comparação emparelhada.

    Devolve uma linha por (controlador, cenário) para que a análise possa
    perguntar ONDE eles divergem, e não apenas por quanto.
    """
    from dataclasses import replace

    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import episode_metrics, run_episode
    from .model_io import load_agent
    from .scenarios import sample_random_scenarios

    cen = sample_random_scenarios(n=n, seed=seed)
    base = config_for_profile("Equilibrado")
    if tolerancia is not None:
        base = replace(base, lab_tolerance=tolerancia)

    controladores = []
    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        caminho = _pm(f"DQN_{perfil}_seed0.zip")
        if os.path.exists(caminho):
            m, c, meta = load_agent(caminho)
            if tolerancia is not None:
                c = replace(c, lab_tolerance=tolerancia)
            controladores.append((f"DQN {perfil}", m, c,
                                  meta.get("action_repeat", 2), False, False))
    try:
        sac, cs, ms = load_agent(_pm("SAC_Equilibrado_seed0.zip"))
        if tolerancia is not None:
            cs = replace(cs, lab_tolerance=tolerancia)
        controladores.append(("SAC Equilibrado", sac, cs,
                              ms.get("action_repeat", 2), False, True))
    except Exception:
        pass
    controladores += [
        ("PI sintonizado", PIController(base, kp=1.3, ki=0.2), base, 2, True, False),
        ("Termostato (zm = 1 °C)", ThermostatAgent(base, deadband=1.0), base, 2, True, False),
        ("Termostato (zm = 0)", ThermostatAgent(base, deadband=0.0), base, 2, True, False),
    ]

    linhas = []
    for nome, ag, c, rep, info, continuo in controladores:
        fab = _fabrica(c, rep, continuo=continuo)
        for s in cen:
            df = run_episode(ag, fab(), s, seed=0, pass_info_to_agent=info)
            met = episode_metrics(df, c)
            linhas.append({
                "controlador": nome, "cenario": s.get("id", s.get("name")),
                "start_temp": s["start_temp"], "occupancy": s["occupancy"],
                "hour": s["hour"],
                "conf_larga_pct": met["comfort_wide_pct"],
                "conf_estreita_pct": met["comfort_narrow_pct"],
                "na_tolerancia_pct": met["in_tolerance_pct"],
                "desvio_ideal": met["abs_dev_from_ideal"],
                "energia_kwh": met["energy_kwh_day"],
            })
    return pd.DataFrame(linhas)


def analise_q_values(perfis=("Agressivo", "Equilibrado", "Passivo")) -> pd.DataFrame:
    """
    Por que MEDIUM some da política: dominância ou margem estreita?

    Reporta a posição de MEDIUM no ranking dos Q-values, o déficit para o topo E
    a escala absoluta. A escala importa: dizer que MEDIUM "perde 63 % da faixa de
    Q" soa devastador, mas a faixa inteira entre as quatro ações vale ~1,9 % do
    valor absoluto. O que o dado mostra é que as ações são quase equivalentes, e
    que o argmax converte uma margem de ~1 % em uso de 0 %.
    """
    import torch

    from .metrics import run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cen = build_scenario_matrix()
    linhas = []
    for perfil in perfis:
        caminho = _pm(f"DQN_{perfil}_seed0.zip")
        if not os.path.exists(caminho):
            continue
        m, c, meta = load_agent(caminho)
        fab = _fabrica(c, meta.get("action_repeat", 2))
        obs = []
        for s in cen:
            env = fab()
            o, _ = env.reset(seed=0, options=s)
            for _ in range(120):
                a, _ = m.predict(o, deterministic=True)
                o, _, t, tr, _ = env.step(a)
                obs.append(o)
                if t or tr:
                    break
        with torch.no_grad():
            q = m.q_net(torch.as_tensor(np.array(obs, dtype=np.float32))).numpy()
        rank = (-q).argsort(axis=1).argsort(axis=1)
        faixa = (q.max(1) - q.min(1)).mean()
        linhas.append({
            "controlador": f"DQN {perfil}",
            "posicao_medium": float(rank[:, 2].mean() + 1),
            "deficit_medium": float((q.max(1) - q[:, 2]).mean()),
            "faixa_entre_acoes": float(faixa),
            "q_absoluto": float(q.mean()),
            "faixa_pct_do_valor": float(faixa / abs(q.mean()) * 100),
            "deficit_pct_do_valor": float((q.max(1) - q[:, 2]).mean()
                                          / abs(q.mean()) * 100),
            "argmax_medium_pct": float((q.argmax(1) == 2).mean() * 100),
        })
    return pd.DataFrame(linhas)


def cenarios_aleatorios_cache() -> Optional[pd.DataFrame]:
    """Avaliação nos cenários aleatórios (gerada por `cenarios_aleatorios`)."""
    caminho = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experimentos",
        "resultados_cenarios_aleatorios.csv"))
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


# ================================== 8. resposta transitória (velocidade)

def resposta_transitoria(n: int = 120, seed: int = 12345) -> pd.DataFrame:
    """
    Quanto tempo cada controlador leva para trazer a sala à faixa.

    MÉTRICA QUE FALTAVA. Todo o trabalho anterior mede QUALIDADE em regime
    (fração do tempo dentro da faixa, desvio, energia) e nunca mediu VELOCIDADE
    de resposta. A omissão importa por dois motivos: velocidade é requisito
    operacional real — uma sala que leva quatro horas para ficar utilizável é um
    problema mesmo que depois fique perfeita —, e ela testa uma afirmação que o
    restante do trabalho fazia sem medir, a de que o conforto idêntico entre
    controladores decorre de um transitório limitado pela física.

    São reportadas três grandezas distintas, e a diferença entre elas é o que
    revela o comportamento do termostato:
      * `entrada_h`   — primeira vez que cruza para dentro da faixa;
      * `acomodacao_h`— a partir de quando NÃO sai mais;
      * `overshoot_c` — quanto passou do outro lado ao frear.

    Apenas cenários que começam FORA da faixa entram na conta: nos demais o
    transitório não existe e a média seria diluída por zeros.
    """
    from .baselines import PIController, ThermostatAgent
    from .config import config_for_profile
    from .metrics import run_episode
    from .model_io import load_agent
    from .scenarios import sample_random_scenarios

    cfg = config_for_profile("Equilibrado")
    cen = [s for s in sample_random_scenarios(n=n, seed=seed)
           if s["start_temp"] > cfg.temp_comfort_max]

    controladores = []
    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        caminho = _pm(f"DQN_{perfil}_seed0.zip")
        if os.path.exists(caminho):
            m, c, meta = load_agent(caminho)
            controladores.append((f"DQN {perfil}", m, c,
                                  meta.get("action_repeat", 2), False, False))
    try:
        sac, cs, ms = load_agent(_pm("SAC_Equilibrado_seed0.zip"))
        controladores.append(("SAC Equilibrado", sac, cs,
                              ms.get("action_repeat", 2), False, True))
    except Exception:
        pass
    controladores += [
        ("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2), cfg, 2, True, False),
        ("Termostato (zm = 1 °C)", ThermostatAgent(cfg, deadband=1.0), cfg, 2, True, False),
        ("Termostato (zm = 0)", ThermostatAgent(cfg, deadband=0.0), cfg, 2, True, False),
    ]

    linhas = []
    for nome, ag, c, rep, info, continuo in controladores:
        fab = _fabrica(c, rep, continuo=continuo)
        for s in cen:
            df = run_episode(ag, fab(), s, seed=0, pass_info_to_agent=info)
            t = df["temperature"].to_numpy()
            dt = c.dt * rep                      # horas por linha do DataFrame
            dentro = t <= c.temp_comfort_max
            i = int(np.argmax(dentro)) if dentro.any() else len(t)
            # Acomodação: primeiro índice a partir do qual nunca mais sai.
            j = len(t)
            for k in range(len(t)):
                if dentro[k:].all():
                    j = k
                    break
            trans = df.iloc[:max(i, 1)]
            linhas.append({
                "controlador": nome, "cenario": s.get("id", s.get("name")),
                "start_temp": s["start_temp"], "occupancy": s["occupancy"],
                "entrada_h": i * dt, "acomodacao_h": j * dt,
                "overshoot_c": max(0.0, c.temp_comfort_min - t[:max(j, 1)].min()),
                # Taxa média de aproximação. Indefinida quando o cenário entra
                # na faixa já no primeiro passo (i = 0): dividir por ~zero
                # produzia valores da ordem de 10^7 na primeira versão.
                "taxa_c_por_h": ((t[0] - t[i]) / (i * dt)) if i > 0 else np.nan,
                "saturacao_pct": float((trans["ac_state"] == "HIGH").mean() * 100.0),
                "energia_transitorio_kwh": float(trans["energy_kwh"].sum()),
            })
    return pd.DataFrame(linhas)


# ============================ 9. correspondência física e hiperparâmetros

def correspondencia_fisica(config=None) -> Dict[str, float]:
    """
    Traduz as unidades de simulação para grandezas físicas.

    LACUNA QUE ISTO FECHA. O parecer aponta que o ambiente "representa uma sala
    fictícia" cujos parâmetros foram escolhidos para tornar ocupação e potência
    relevantes, sem calibração com dados reais. O modelo, de fato, opera em
    unidades adimensionais: `thermal_mass = 15` e `heat_transfer_coeff = 0,5` não
    dizem nada sobre a sala que representam.

    A conversão é possível porque a capacidade do equipamento ancora a escala:
    40 unidades correspondem a 8,79 kW térmicos. Dessa âncora derivam a
    capacidade térmica em kJ/°C, o volume equivalente de ar, a condutância em
    kW/°C e — a grandeza mais informativa — a CONSTANTE DE TEMPO do ambiente.
    """
    from .config import config_for_profile

    cfg = config or config_for_profile("Equilibrado")
    fis = cfg.physics
    u_kw = fis.capacity_kw_thermal / fis.cooling_units_at_full_load

    c_kwh = u_kw / (1.0 / cfg.thermal_mass)      # kWh/°C
    c_kj = c_kwh * 3600.0
    rho, cp = 1.184, 1.005                       # ar a ~25 °C
    k_kw = cfg.heat_transfer_coeff * u_kw

    return {
        "kw_por_unidade": u_kw,
        "capacidade_kwh_por_c": c_kwh,
        "capacidade_kj_por_c": c_kj,
        "volume_ar_equivalente_m3": c_kj / (rho * cp),
        "condutancia_kw_por_c": k_kw,
        "constante_tempo_h": c_kwh / k_kw,
        "ganho_por_pessoa_w": cfg.heat_gain_per_person * u_kw * 1000.0,
        "carga_ocupacao_plena_kw": cfg.max_occupancy * cfg.heat_gain_per_person * u_kw,
        "capacidade_equipamento_kw": fis.capacity_kw_thermal,
        "duracao_episodio_h": cfg.episode_steps * cfg.dt,
    }


def hiperparametros(algo: str = "DQN", perfil: str = "Equilibrado") -> pd.DataFrame:
    """
    Hiperparâmetros EFETIVOS do agente, extraídos do objeto treinado.

    O parecer registra que o manuscrito não informa duração do replay buffer,
    fator de desconto, frequência de atualização da rede-alvo, arquitetura nem
    estratégia de exploração. Lê-los do modelo — e não do que foi escrito no
    código — garante que os valores default do arcabouço, que nunca aparecem
    explicitamente, sejam igualmente reportados.
    """
    from .model_io import load_agent

    m, _, _ = load_agent(_pm(f"{algo}_{perfil}_seed0.zip"))
    campos = [
        ("learning_rate", "taxa de aprendizado"),
        ("batch_size", "tamanho do lote"),
        ("gamma", "fator de desconto γ"),
        ("buffer_size", "capacidade do replay buffer"),
        ("learning_starts", "passos antes do primeiro ajuste"),
        ("gradient_steps", "passos de gradiente por atualização"),
        ("target_update_interval", "intervalo de atualização da rede-alvo"),
        ("tau", "coeficiente de atualização suave τ"),
        ("exploration_initial_eps", "ε inicial"),
        ("exploration_final_eps", "ε final"),
        ("exploration_fraction", "fração do treino em decaimento de ε"),
        ("max_grad_norm", "recorte de gradiente"),
    ]
    linhas = []
    for attr, rot in campos:
        if not hasattr(m, attr):
            continue
        v = getattr(m, attr)
        if callable(v):
            try:
                v = float(v(1.0))
            except Exception:
                v = str(v)
        linhas.append({"parametro": rot, "valor": v})

    freq = getattr(m, "train_freq", None)
    if freq is not None:
        linhas.append({"parametro": "frequência de treino (passos)",
                       "valor": getattr(freq, "frequency", str(freq))})
    n = sum(p.numel() for p in m.policy.parameters() if p.requires_grad)
    dim = m.observation_space.shape[0]
    linhas += [
        {"parametro": "arquitetura (MLP)", "valor": f"{dim}–64–64–{m.action_space.n}"},
        {"parametro": "parâmetros treináveis", "valor": f"{n:,}".replace(",", ".")},
    ]
    return pd.DataFrame(linhas)


# ============================ 10. os quatro parâmetros inferidos

def analise_parametros_recompensa(config=None) -> Dict[str, object]:
    """
    O que se pode DECIDIR por argumento sobre B, k, rho e a penalidade de frio.

    Os quatro não são publicados pelo manuscrito e foram inferidos da figura de
    recompensa. Nem todos, porém, precisam ser medidos: alguns se resolvem por
    análise da própria formulação, o que é preferível, pois não depende de
    treinar nada.
    """
    from .config import config_for_profile

    cfg = config or config_for_profile("Equilibrado")
    B, Bc, k = cfg.comfort_bonus, cfg.comfort_gradient, cfg.comfort_sensitivity
    Tmin, Tmax, Tid = cfg.temp_comfort_min, cfg.temp_comfort_max, cfg.ideal_temp
    hw = (Tmax - Tmin) / 2.0

    def conforto(T):
        if Tmin <= T <= Tmax:
            return B + Bc * (1 - abs(T - Tid) / hw)
        d = T - Tmax if T > Tmax else Tmin - T
        return B - k * d ** 2

    custo_meia_faixa = conforto(Tid) - conforto(Tmax)          # centro -> borda
    custos_fora = {d: conforto(Tmax) - conforto(Tmax + d) for d in (0.5, 1, 2, 4)}

    # k que restaura a coerência: sair `delta` da faixa custa o mesmo que
    # atravessar a meia-faixa por dentro.
    k_coerente = {d: Bc / d ** 2 for d in (1.0, 2.0)}

    # rho: a penalidade da eq. 5 é PROPORCIONAL à antecipação, de modo que
    # comutar quase no limite custa quase nada.
    d_min = cfg.min_dwell_steps
    penal_por_antecipacao = {d: cfg.short_cycle_penalty * (d_min - d) / d_min
                             for d in range(0, d_min + 1)}

    return {
        "B": B, "Bc": Bc, "k": k, "rho": cfg.short_cycle_penalty,
        "frio": cfg.cold_action_penalty,
        "B_e_aditivo": True,
        "custo_centro_ate_borda": custo_meia_faixa,
        "custo_fora_por_delta": custos_fora,
        "inclinacao_interna_por_c": Bc / hw,
        "inclinacao_externa_em": {d: 2 * k * d for d in (0.1, 0.5, 1.0, 2.0)},
        "k_que_iguala_meia_faixa": k_coerente,
        "d_min_passos": d_min,
        "penalidade_por_antecipacao": penal_por_antecipacao,
    }


def ativacao_penalidade_frio() -> pd.DataFrame:
    """
    Com que frequência a penalidade de frio chega a ser acionada.

    Ela só incide quando a sala está abaixo do piso da faixa E o equipamento
    resfria. Se isso praticamente não ocorre, calibrar o parâmetro é discutir um
    termo que não participa do treinamento, e a questão deixa de ser empírica.
    """
    from .baselines import PIController
    from .config import config_for_profile
    from .metrics import run_episode
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix

    cfg = config_for_profile("Equilibrado")
    cen = build_scenario_matrix()
    linhas = []
    for nome, ag, c, rep, info in [
        ("PI sintonizado", PIController(cfg, kp=1.3, ki=0.2), cfg, 2, True),
    ] + [(f"DQN {p}", *load_agent(_pm(f"DQN_{p}_seed0.zip"))[:2],
          2, False) for p in ("Agressivo", "Equilibrado", "Passivo")]:
        fab = _fabrica(c if not isinstance(c, int) else cfg, rep)
        cfg_uso = c if not isinstance(c, int) else cfg
        df = pd.concat([run_episode(ag, fab(), s, seed=0, pass_info_to_agent=info)
                        for s in cen])
        # condição da penalidade: T abaixo do piso e carga de resfriamento ativa
        ativa = (df["temperature"] < cfg_uso.temp_comfort_min) & (df["load"] > 0.2)
        linhas.append({"controlador": nome,
                       "passos": len(df),
                       "passos_com_penalidade": int(ativa.sum()),
                       "fracao_pct": float(ativa.mean() * 100.0)})
    return pd.DataFrame(linhas)


def calibracao_parametros() -> Optional[pd.DataFrame]:
    """Varredura dos quatro parâmetros inferidos (caro; lido do CSV)."""
    c = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                      "experimentos", "resultados_calibracao.csv"))
    return pd.read_csv(c) if os.path.exists(c) else None


def combinacao_derivados() -> Optional[pd.DataFrame]:
    """Referência contra a configuração com B e k derivados."""
    c = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                      "experimentos", "resultados_combinacao.csv"))
    return pd.read_csv(c) if os.path.exists(c) else None


def orcamento_e_parametros() -> Optional[Dict[str, pd.DataFrame]]:
    """
    O efeito dos parâmetros derivados DEPENDE do orçamento de treinamento.

    Este é o achado que corrige uma conclusão anterior deste trabalho. A
    calibração, conduzida com 300 mil passos, indicava que zerar o bônus base e
    elevar a curvatura melhorava o conforto em 16,7 pontos percentuais. Sob o
    orçamento do protocolo, 550 mil passos, o efeito se inverte: a configuração
    derivada converge mais rápido e depois degrada, ao passo que a inferida
    aprende devagar e continua melhorando.

    A lição metodológica é geral e vale além deste caso: ajustar parâmetros de
    recompensa sob orçamento reduzido e extrapolar para o orçamento completo é
    inválido, porque a interação entre formulação e duração do treinamento não é
    monotônica.
    """
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                         "experimentos"))
    c300 = os.path.join(base, "resultados_combinacao.csv")
    d550 = os.path.join(base, "resultados_derivados_550k.csv")
    i550 = os.path.join(base, "resultados_inferidos_550k.csv")
    if not (os.path.exists(c300) and os.path.exists(d550)):
        return None

    curto = pd.read_csv(c300)
    linhas = [
        {"orcamento": "300k", "config": "inferidos",
         "conf_estreita": v, "perfil": "Equilibrado"}
        for v in curto[curto["config"].str.startswith("referência")]["conf_estreita"]
    ] + [
        {"orcamento": "300k", "config": "derivados",
         "conf_estreita": v, "perfil": "Equilibrado"}
        for v in curto[curto["config"].str.startswith("derivados")]["conf_estreita"]
    ]

    der = pd.read_csv(d550)
    linhas += [{"orcamento": "550k", "config": "derivados",
                "conf_estreita": r["conf_estreita"], "perfil": r["perfil"]}
               for _, r in der.iterrows()]

    if os.path.exists(i550):
        inf = pd.read_csv(i550)
        linhas += [{"orcamento": "550k", "config": "inferidos",
                    "conf_estreita": r["conf_estreita"], "perfil": r["perfil"]}
                   for _, r in inf.iterrows()]
    # a semente 0 dos inferidos a 550k vem dos modelos do protocolo original
    from .metrics import evaluate_agent
    from .model_io import load_agent
    from .scenarios import build_scenario_matrix
    cen = build_scenario_matrix()
    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        caminho = _p("models_paper", f"DQN_{perfil}_seed0.zip")
        if not os.path.exists(caminho):
            continue
        m, c, meta = load_agent(caminho)
        s = evaluate_agent(m, _fabrica(c, meta.get("action_repeat", 2)),
                           cen, c, seed=0)["summary"]
        linhas.append({"orcamento": "550k", "config": "inferidos",
                       "conf_estreita": s["comfort_narrow_pct"], "perfil": perfil})

    df = pd.DataFrame(linhas)
    resumo = (df.groupby(["orcamento", "config"])["conf_estreita"]
                .agg(media="mean", desvio="std", n="count").reset_index())
    return {"bruto": df, "resumo": resumo}
