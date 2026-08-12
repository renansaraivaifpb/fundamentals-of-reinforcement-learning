# -*- coding: utf-8 -*-
"""
Simulação e métricas de avaliação (Tabelas 4, 5 e 6 do paper).

As métricas seguem três regras que o paper explicita e que mudam os números
de forma substancial — implementá-las errado é a principal fonte de
divergência numa reprodução:

1. JANELA OCUPADA: conforto, |T−24| e sobreaquecimento são medidos apenas em
   7h–22h. Medir 24 h infla o conforto (a sala vazia esfria de graça).
2. FILTRO DE CONTROLABILIDADE: um cenário só entra na média de conforto se o
   contrafactual (AC sempre desligado) ultrapassaria 26 °C. Cenários de
   partida fria são incontroláveis por um equipamento que apenas resfria, e
   incluí-los pune todos os agentes igualmente sem informar nada.
3. ENERGIA/CUSTO em base DIÁRIA, sobre todos os 9 cenários; conforto e
   métricas contínuas sobre os controláveis.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ClassroomConfig


def run_episode(
    agent,
    env,
    scenario: Dict,
    seed: Optional[int] = None,
    pass_info_to_agent: bool = False,
) -> pd.DataFrame:
    """Roda um episódio de 24 h e devolve o histórico como DataFrame."""
    # Controladores com estado (integrador do PI, histerese do termostato) devem
    # ser zerados por episódio; sem isto o estado vaza de um cenário para o
    # próximo e a avaliação deixa de ser independente por célula da matriz.
    if hasattr(agent, "reset"):
        agent.reset()

    obs, info = env.reset(seed=seed, options=scenario)
    rows: List[Dict] = []

    terminated = truncated = False
    while not (terminated or truncated):
        if pass_info_to_agent:
            action, _ = agent.predict(obs, deterministic=True, info=info)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(dict(info, reward=reward))

    return pd.DataFrame(rows)


def is_controllable(env, scenario: Dict, seed: Optional[int] = None) -> bool:
    """
    Contrafactual do paper: com o AC sempre desligado, a sala passaria de
    26 °C na janela ocupada? Se não, resfriar não era necessário.
    """
    from .scenarios import AlwaysOffAgent

    df = run_episode(AlwaysOffAgent(), env, scenario, seed=seed)
    occupied = df[df["occupied"]]
    if occupied.empty:
        return False
    return bool(occupied["temperature"].max() > env.unwrapped.config.temp_comfort_max)


def episode_metrics(df: pd.DataFrame, config: ClassroomConfig) -> Dict[str, float]:
    """Métricas de um único episódio."""
    occupied = df[df["occupied"]]
    if occupied.empty:
        occupied = df

    temp = occupied["temperature"]
    in_wide = (temp >= config.temp_comfort_min) & (temp <= config.temp_comfort_max)
    in_narrow = (temp >= config.temp_narrow_min) & (temp <= config.temp_narrow_max)
    overheat = temp > config.temp_comfort_max

    # Energia e custo são do dia inteiro (o AC opera fora da janela também).
    energy_kwh = float(df["energy_kwh"].sum())
    cost_brl = float(df["cost_brl"].sum())

    # Trocas por hora: comutações de nível / horas simuladas.
    # Cada linha pode representar mais de um passo do ambiente (action repeat),
    # então as horas vêm de `inner_steps` quando disponível — usar len(df)
    # subestimaria a duração e inflaria trocas/h pelo fator de repetição.
    steps = float(df["inner_steps"].sum()) if "inner_steps" in df else float(len(df))
    hours = steps * config.dt
    changes = int(df["action_changed"].sum())

    # Taxa de violação de temperatura (Xu et al. 2025): fração do tempo OCUPADO
    # fora da faixa de conforto. É a métrica de segurança da literatura de
    # shielding, e é mais informativa que "conforto" porque não satura: mede o
    # que o controlador falhou em evitar, não o que ele acertou.
    violation = (temp < config.temp_comfort_min) | (temp > config.temp_comfort_max)
    # Graus·hora de violação: pondera a violação pela severidade. Uma sala a
    # 26,1 °C e outra a 32 °C têm a mesma taxa de violação e gravidades
    # completamente distintas.
    excess = np.maximum(temp - config.temp_comfort_max, 0.0) + np.maximum(
        config.temp_comfort_min - temp, 0.0
    )

    # --- Métricas de LABORATÓRIO de precisão ---
    # Para controle minucioso a pergunta não é "ficou na faixa de 4 °C" e sim
    # "quanto tempo ficou dentro de ±tolerance do setpoint" e "qual a dispersão".
    # O desvio-padrão é o que um laboratório de fato especifica (estabilidade),
    # e não aparece em nenhuma métrica do manuscrito.
    tol = config.lab_tolerance
    in_tolerance = (temp - config.ideal_temp).abs() <= tol
    peak = df[df["posto"] == "ponta"] if "posto" in df else df.iloc[0:0]

    # --- Métricas de ERRO da teoria de controle ---
    # O projeto reporta apenas médias e frações de tempo. Um engenheiro de
    # controle especifica outra coisa: IAE/ISE (erro acumulado), sobressinal,
    # tempo de acomodação e erro de regime. Nenhuma delas existia aqui, e é
    # justamente onde o RL perde do PI — a média esconde transiente, como ficou
    # claro quando o subresfriamento de 20 °C do PI com windup passou batido por
    # métricas agregadas boas (86,5 % de conforto, |T-24| de 0,72).
    erro = temp - config.ideal_temp
    dt_h = config.dt
    passos = float(df["inner_steps"].sum()) if "inner_steps" in df else float(len(df))
    horas_tot = passos * dt_h

    iae = float(erro.abs().sum() * dt_h)          # integral do erro absoluto (°C·h)
    ise = float((erro ** 2).sum() * dt_h)         # integral do erro quadrático
    itae = float((erro.abs() * np.arange(len(erro)) * dt_h).sum() * dt_h)  # pondera cauda

    # Sobressinal: maior excursão ALÉM do setpoint no sentido da correção. Num
    # pulldown (partida quente) é o quanto passou para baixo do alvo.
    inicio = float(df["temperature"].iloc[0])
    if inicio > config.ideal_temp:
        overshoot = float(max(config.ideal_temp - temp.min(), 0.0))
    else:
        overshoot = float(max(temp.max() - config.ideal_temp, 0.0))

    # Tempo de acomodação: primeira vez que o erro entra na tolerância e
    # PERMANECE (não apenas cruza). Sem o "permanece", um cruzamento
    # transitório contaria como acomodado.
    tol = config.lab_tolerance
    dentro = (erro.abs() <= tol).to_numpy()
    settling = float("nan")
    for k in range(len(dentro)):
        if dentro[k:].all():
            settling = k * dt_h
            break

    # Erro de regime: média do erro no último quarto do episódio, COM sinal.
    # Viés persistente é o que uma política sem integral não consegue eliminar.
    cauda = erro.iloc[int(len(erro) * 0.75):]
    erro_regime = float(cauda.mean()) if len(cauda) else float("nan")

    # --- SEGMENTAÇÃO transitório / regime ---
    # Sem isto as métricas agregadas medem a CONDIÇÃO INICIAL, não o controle:
    # 15 sintonias distintas do PI davam exatamente 90,1 % de tolerância porque o
    # número era dominado pelo transiente de partida (17 °C ou 30 °C até a faixa),
    # não pela qualidade de regime. Depois de acomodar, PI e todos os agentes
    # ficam em 100 % — o que só aparece quando se separa.
    k = int(round(settling / dt_h)) if settling == settling else len(temp)
    k = min(max(k, 0), len(temp))
    tem_regime = (len(temp) - k) >= 10        # exige cauda mínima para ser métrica

    if tem_regime:
        t_ss = temp.iloc[k:]
        e_ss = erro.iloc[k:]
        ss = {
            "in_tolerance_ss_pct": float((e_ss.abs() <= tol).mean() * 100.0),
            "temp_std_ss": float(t_ss.std()),
            "abs_dev_ss": float(e_ss.abs().mean()),
            "max_abs_dev_ss": float(e_ss.abs().max()),
        }
    else:
        ss = {"in_tolerance_ss_pct": float("nan"), "temp_std_ss": float("nan"),
              "abs_dev_ss": float("nan"), "max_abs_dev_ss": float("nan")}

    # Transitório: só até acomodar. IAE aqui é o custo da partida.
    iae_transient = float(erro.iloc[:k].abs().sum() * dt_h) if k > 0 else 0.0

    # --- Demanda contratada ---
    # O que fatura é o PICO da média de 15 min no ciclo, não a média do dia:
    # reduzir consumo médio não adianta se houve um único período acima do
    # contratado.
    dem = {}
    if "demand_peak_kw" in df:
        dem["demand_peak_kw"] = float(df["demand_peak_kw"].max())
        limite = getattr(config, "demand_contracted_kw", float("inf"))
        dem["demand_overrun_kw"] = float(max(0.0, dem["demand_peak_kw"] - limite))
        dem["demand_overrun_pct"] = float((df["demand_kw"] > limite).mean() * 100.0)

    return {
        **dem,
        # --- transitório ---
        "iae_transient": iae_transient,
        # --- regime permanente (depois de acomodar) ---
        **ss,
        # --- episódio inteiro ---
        "iae_c_h": iae,
        "ise": ise,
        "itae": itae,
        "overshoot_c": overshoot,
        "settling_h": settling,
        "erro_regime_c": erro_regime,
        "in_tolerance_pct": float(in_tolerance.mean() * 100.0),
        "temp_std": float(temp.std()),
        "max_abs_dev": float((temp - config.ideal_temp).abs().max()),
        "p95_abs_dev": float((temp - config.ideal_temp).abs().quantile(0.95)),
        "peak_cost_brl": float(peak["cost_brl"].sum()) if len(peak) else 0.0,
        "peak_energy_kwh": float(peak["energy_kwh"].sum()) if len(peak) else 0.0,
        "comfort_wide_pct": float(in_wide.mean() * 100.0),
        "comfort_narrow_pct": float(in_narrow.mean() * 100.0),
        "abs_dev_from_ideal": float((temp - config.ideal_temp).abs().mean()),
        "overheat_pct": float(overheat.mean() * 100.0),
        "violation_rate_pct": float(violation.mean() * 100.0),
        "violation_degree_hours": float(excess.sum() * config.dt),
        "shield_interventions": float(
            df["shield_interventions"].max() if "shield_interventions" in df else 0.0
        ),
        "energy_kwh_day": energy_kwh,
        "cost_brl_day": cost_brl,
        "changes_per_hour": changes / hours if hours else 0.0,
        "changes_per_day": float(changes),
        "mean_temp": float(temp.mean()),
    }


def evaluate_agent(
    agent,
    env_factory,
    scenarios: Sequence[Dict],
    config: ClassroomConfig,
    seed: Optional[int] = 0,
    pass_info_to_agent: bool = False,
) -> Dict[str, object]:
    """
    Avalia um agente na matriz 3×3, aplicando o filtro de controlabilidade.

    `env_factory` é uma callable sem argumentos que devolve um ambiente novo —
    necessário porque o contrafactual precisa de um ambiente limpo, e porque o
    DQN e o SAC exigem wrappers distintos.
    """
    per_scenario: List[Dict] = []
    temps_controllable: List[np.ndarray] = []

    for scenario in scenarios:
        controllable = is_controllable(env_factory(), scenario, seed=seed)
        df = run_episode(
            agent, env_factory(), scenario, seed=seed,
            pass_info_to_agent=pass_info_to_agent,
        )
        m = episode_metrics(df, config)
        m.update(
            {
                "scenario": scenario["name"],
                "scenario_id": scenario["id"],
                "condition": scenario["condition"],
                "occupancy_label": scenario["occupancy_label"],
                "controllable": controllable,
            }
        )
        per_scenario.append(m)

        if controllable:
            occ = df[df["occupied"]]
            temps_controllable.append(occ["temperature"].to_numpy())

    df_scen = pd.DataFrame(per_scenario)
    ctrl = df_scen[df_scen["controllable"]]
    basis = ctrl if not ctrl.empty else df_scen

    summary = {
        # Conforto e métricas contínuas: apenas cenários controláveis.
        "comfort_wide_pct": basis["comfort_wide_pct"].mean(),
        "comfort_narrow_pct": basis["comfort_narrow_pct"].mean(),
        "abs_dev_from_ideal": basis["abs_dev_from_ideal"].mean(),
        "overheat_pct": basis["overheat_pct"].mean(),
        **({"demand_peak_kw": df_scen["demand_peak_kw"].max(),
            "demand_overrun_kw": df_scen["demand_overrun_kw"].max(),
            "demand_overrun_pct": df_scen["demand_overrun_pct"].mean()}
           if "demand_peak_kw" in df_scen else {}),
        "iae_transient": basis["iae_transient"].mean(),
        "in_tolerance_ss_pct": basis["in_tolerance_ss_pct"].mean(),
        "temp_std_ss": basis["temp_std_ss"].mean(),
        "abs_dev_ss": basis["abs_dev_ss"].mean(),
        "max_abs_dev_ss": basis["max_abs_dev_ss"].max(),
        "iae_c_h": basis["iae_c_h"].mean(),
        "ise": basis["ise"].mean(),
        "overshoot_c": basis["overshoot_c"].max(),
        "settling_h": basis["settling_h"].mean(),
        "erro_regime_c": basis["erro_regime_c"].mean(),
        "in_tolerance_pct": basis["in_tolerance_pct"].mean(),
        "temp_std": basis["temp_std"].mean(),
        "max_abs_dev": basis["max_abs_dev"].max(),
        "p95_abs_dev": basis["p95_abs_dev"].mean(),
        "peak_cost_brl": df_scen["peak_cost_brl"].mean(),
        "peak_energy_kwh": df_scen["peak_energy_kwh"].mean(),
        "violation_rate_pct": basis["violation_rate_pct"].mean(),
        "violation_degree_hours": basis["violation_degree_hours"].mean(),
        "shield_interventions": df_scen["shield_interventions"].mean(),
        # Energia e custo: todos os cenários.
        "energy_kwh_day": df_scen["energy_kwh_day"].mean(),
        "cost_brl_day": df_scen["cost_brl_day"].mean(),
        "changes_per_hour": df_scen["changes_per_hour"].mean(),
        "changes_per_day": df_scen["changes_per_day"].mean(),
        "n_controllable": int(df_scen["controllable"].sum()),
        "n_scenarios": len(df_scen),
    }

    return {
        "summary": summary,
        "per_scenario": df_scen,
        "temperatures": (
            np.concatenate(temps_controllable) if temps_controllable else np.array([])
        ),
    }


def format_table4(results: Dict[str, Dict]) -> str:
    """Renderiza a Tabela 4 no formato do paper."""
    header = (
        f"{'Agente':<22}{'Conf.[22,26]':>14}{'Conf.[23,25]':>14}"
        f"{'|T-24|':>9}{'Sobreaq.':>10}{'Energia':>10}{'Custo':>9}{'Trocas/h':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, res in results.items():
        s = res["summary"]
        lines.append(
            f"{name:<22}{s['comfort_wide_pct']:>13.1f}%{s['comfort_narrow_pct']:>13.1f}%"
            f"{s['abs_dev_from_ideal']:>9.2f}{s['overheat_pct']:>9.1f}%"
            f"{s['energy_kwh_day']:>10.2f}{s['cost_brl_day']:>9.2f}"
            f"{s['changes_per_hour']:>10.2f}"
        )
    return "\n".join(lines)
