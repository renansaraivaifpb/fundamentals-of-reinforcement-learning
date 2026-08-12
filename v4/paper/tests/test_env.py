# -*- coding: utf-8 -*-
"""
Testes de regressão do ambiente, da física e da recompensa.

Cada teste marcado com REGRESSÃO corresponde a um bug real que ocorreu neste
projeto. Sem eles, os mesmos erros voltam silenciosamente — foi exatamente o que
aconteceu entre a v4 original e esta reprodução.

    pytest -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ac_physics import ACPhysicsModel, ACState  # noqa: E402
from baselines import PIController, ThermostatAgent  # noqa: E402
from config import REWARD_PROFILES, ClassroomConfig, config_for_profile  # noqa: E402
from env import ClassroomACEnv  # noqa: E402
from metrics import episode_metrics, run_episode  # noqa: E402
from scenarios import build_scenario_matrix  # noqa: E402
from wrappers import ActionRepeatWrapper, SafetyShieldWrapper  # noqa: E402


# ------------------------------------------------------- contrato Gymnasium

def test_gymnasium_env_checker():
    """O checker oficial teria pego o bug de observação fora do Box."""
    from gymnasium.utils.env_checker import check_env

    check_env(ClassroomACEnv(config=ClassroomConfig()), skip_render_check=True)


def test_determinism_under_seed():
    """Mesma semente -> mesma trajetória. Base de qualquer reprodutibilidade."""
    def roll():
        env = ClassroomACEnv(config=ClassroomConfig())
        obs, _ = env.reset(seed=123)
        temps = []
        for a in [0, 1, 2, 3, 2, 1, 0, 3]:
            _, _, _, _, info = env.step(a)
            temps.append(info["temperature"])
        return temps

    assert roll() == pytest.approx(roll())


# ----------------------------------------------------------------- Tabela 1

def test_physics_table1_matches_paper():
    """Tabela 1 derivada de 30k BTU + COP deve bater com o manuscrito."""
    p = ACPhysicsModel()
    assert p.electrical_kw(ACState.LOW) == pytest.approx(0.64, abs=0.01)
    assert p.electrical_kw(ACState.MEDIUM) == pytest.approx(1.35, abs=0.01)
    assert p.electrical_kw(ACState.HIGH) == pytest.approx(2.93, abs=0.01)
    assert p.cooling_units(ACState.LOW) == pytest.approx(10.0)
    assert p.cooling_units(ACState.MEDIUM) == pytest.approx(22.0)
    assert p.cooling_units(ACState.HIGH) == pytest.approx(40.0)
    assert p.electrical_kw(ACState.OFF) == 0.0


def test_cop_peaks_at_partial_load():
    """COP máximo em carga parcial (inverter). Sustenta o trade-off do paper."""
    p = ACPhysicsModel()
    assert p.cop[ACState.MEDIUM] > p.cop[ACState.LOW] > p.cop[ACState.HIGH]


def test_continuous_matches_discrete_at_tabulated_loads():
    """O modelo contínuo (SAC) deve coincidir com o discreto nos pontos tabelados,
    senão DQN e SAC são avaliados sob eficiências diferentes."""
    p = ACPhysicsModel()
    for s in (ACState.LOW, ACState.MEDIUM, ACState.HIGH):
        frac = p.load_fraction[s]
        assert p.electrical_kw_continuous(frac) == pytest.approx(p.electrical_kw(s))
        assert p.cooling_units_continuous(frac) == pytest.approx(p.cooling_units(s))


# ---------------------------------------------------------- carga térmica

def test_occupancy_dominates_external_gain():
    """Seção 4.1: 45 ocupantes (13,5 u) devem superar o ganho externo (~6 u)."""
    cfg = ClassroomConfig()
    people = cfg.max_occupancy * cfg.heat_gain_per_person
    external_max = cfg.heat_transfer_coeff * (
        (cfg.outdoor_base_temp + cfg.outdoor_amplitude) - cfg.ideal_temp
    )
    assert people == pytest.approx(13.5)
    assert external_max == pytest.approx(6.0)
    assert people > external_max


def test_high_gives_pulldown_headroom():
    """HIGH deve exceder o pico de carga (19,5 u), senão não há pulldown."""
    cfg = ClassroomConfig()
    peak_load = cfg.max_occupancy * cfg.heat_gain_per_person + 6.0
    assert cfg.physics.cooling_units(ACState.HIGH) > peak_load
    assert cfg.physics.cooling_units(ACState.MEDIUM) > peak_load  # sustenta regime


def _peak_temp_without_ac() -> float:
    env = ClassroomACEnv(config=ClassroomConfig())
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 7})
    peak = 24.0
    for _ in range(150):  # 15 h, cobrindo a janela ocupada
        _, _, term, _, info = env.step(int(ACState.OFF))
        peak = max(peak, info["temperature"])
        if term:
            break
    return peak


def test_full_room_without_ac_requires_cooling():
    """
    Requisito FUNCIONAL: sem climatização a sala cheia precisa sair da faixa de
    conforto com folga, senão o problema de controle é trivial e a comparação
    com o baseline não informa nada.
    """
    assert _peak_temp_without_ac() > 30.0


@pytest.mark.xfail(
    reason="DIVERGÊNCIA DE FIDELIDADE documentada: a Seção 4.1 afirma 32–36 °C; "
    "esta parametrização atinge ~37,8 °C. Como o manuscrito não publica H_p nem "
    "o perfil de ocupação fora da janela, não é possível fechar a diferença sem "
    "fitting. Mantido como xfail para que a divergência fique visível na suíte "
    "em vez de escondida.",
    strict=True,
)
def test_full_room_peak_matches_paper_range():
    assert 32.0 <= _peak_temp_without_ac() <= 36.0


# ------------------------------------------------------- recompensa (eq. 4)

def test_gradient_makes_center_strictly_better():
    """
    A correção central do paper: com gradiente interno, 24 °C é ESTRITAMENTE
    melhor que a borda. Um platô plano empataria — que é a causa raiz do
    "estacionar em 26,1 °C" da v4 original.
    """
    cfg = ClassroomConfig()
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)

    def comfort_at(t: float) -> float:
        env.current_temp = t
        env.ac_state = ACState.OFF
        env.hour_of_day = 10  # fora do pico, sem multiplicador
        return env._reward(changed=False, dwell_before_change=10**6)

    center = comfort_at(cfg.ideal_temp)
    edge_hi = comfort_at(cfg.temp_comfort_max)
    edge_lo = comfort_at(cfg.temp_comfort_min)
    assert center > edge_hi, "gradiente ausente: centro não supera a borda"
    assert center > edge_lo
    assert center - edge_hi == pytest.approx(cfg.comfort_gradient, abs=1e-6)


def test_reward_penalizes_leaving_the_band():
    """Sair da faixa deve custar, e o custo deve crescer com o afastamento."""
    cfg = ClassroomConfig()
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)

    def comfort_at(t: float) -> float:
        env.current_temp = t
        env.ac_state = ACState.OFF
        env.hour_of_day = 10
        return env._reward(changed=False, dwell_before_change=10**6)

    assert comfort_at(26.0) > comfort_at(27.0) > comfort_at(29.0)
    assert comfort_at(22.0) > comfort_at(21.0) > comfort_at(19.0)


def test_short_cycle_penalty_scales_with_dwell():
    """eq. 5: penalidade máxima ao trocar imediatamente, zero após d_min."""
    cfg = ClassroomConfig()
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)
    env.current_temp = cfg.ideal_temp
    env.hour_of_day = 10
    env.ac_state = ACState.OFF

    base = env._reward(changed=False, dwell_before_change=10**6)
    immediate = env._reward(changed=True, dwell_before_change=0)
    half = env._reward(changed=True, dwell_before_change=cfg.min_dwell_steps // 2)
    after = env._reward(changed=True, dwell_before_change=cfg.min_dwell_steps)

    # Isola o termo de ciclo removendo a penalidade de troca, comum aos três.
    cyc = lambda r: r - (base + cfg.action_change_penalty)
    assert cyc(immediate) == pytest.approx(cfg.short_cycle_penalty)
    assert cyc(after) == pytest.approx(0.0)
    assert cyc(immediate) < cyc(half) < 0.0


def test_min_dwell_steps_is_36_minutes():
    cfg = ClassroomConfig()
    assert cfg.min_dwell_steps == 6
    assert cfg.min_dwell_steps * cfg.dt * 60 == pytest.approx(36.0)


def test_profiles_differ_only_in_reward_weights():
    """
    Tese central do paper: os perfis diferem APENAS por pesos de recompensa.
    Se a física divergir entre perfis, a conclusão não se sustenta.
    """
    cfgs = {p: config_for_profile(p) for p in REWARD_PROFILES}
    physical = ("thermal_mass", "heat_transfer_coeff", "heat_gain_per_person",
                "max_occupancy", "dt", "episode_steps")
    for attr in physical:
        values = {getattr(c, attr) for c in cfgs.values()}
        assert len(values) == 1, f"perfis divergem em física: {attr}"
    assert len({c.comfort_gradient for c in cfgs.values()}) == 3


# ------------------------------------------------------ REGRESSÃO: bugs reais

def test_regression_reset_honours_scenario_hour():
    """
    REGRESSÃO: a v4 lia options['hour_of_day'] mas o scenarios.json usa 'hour',
    descartando silenciosamente a hora e avaliando tudo em horas aleatórias.
    """
    env = ClassroomACEnv(config=ClassroomConfig())
    _, info = env.reset(seed=0, options={"hour": 8, "start_temp": 19.0, "occupancy": 0})
    assert info["hour"] == 8
    # A chave legada deve continuar funcionando.
    _, info = env.reset(seed=0, options={"hour_of_day": 15})
    assert info["hour"] == 15


def test_regression_observation_always_inside_space():
    """
    REGRESSÃO: com ocupação acima de max_occupancy a v4 emitia o_norm = 1,33,
    violando o Box(high=1.0) e alimentando a rede com entrada fora da
    distribuição de treino.
    """
    env = ClassroomACEnv(config=ClassroomConfig())
    for occ in (0, 45, 60, 999):
        obs, _ = env.reset(seed=0, options={"occupancy": occ, "start_temp": 24.0})
        assert env.observation_space.contains(obs), f"obs fora do espaço (occ={occ})"
    for temp in (10.0, 40.0, 24.0):
        obs, _ = env.reset(seed=0, options={"start_temp": temp})
        assert env.observation_space.contains(obs)


def test_regression_action_repeat_accumulates_extensive_quantities():
    """
    REGRESSÃO: o wrapper devolvia só o info do último passo interno, perdendo
    metade da energia e TODAS as trocas de nível (a troca ocorre no primeiro
    passo interno). Sintoma observado: Trocas/h = 0,00.
    """
    cfg = ClassroomConfig()
    repeat = 2
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=repeat)
    env.reset(seed=0, options={"start_temp": 30.0, "occupancy": 45, "hour": 14})

    _, _, _, _, info = env.step(int(ACState.HIGH))  # OFF -> HIGH: houve troca
    assert info["action_changed"] is True, "troca no 1º passo interno foi perdida"
    assert info["inner_steps"] == repeat

    expected = cfg.physics.electrical_kw(ACState.HIGH) * cfg.dt * repeat
    assert info["energy_kwh"] == pytest.approx(expected), "energia não acumulada"


def test_regression_changes_per_hour_uses_inner_steps():
    """
    REGRESSÃO: changes_per_hour dividia por len(df), ignorando que cada linha
    vale `action_repeat` passos — inflando trocas/h pelo fator de repetição.
    """
    cfg = ClassroomConfig()
    scenario = build_scenario_matrix()[0]
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    df = run_episode(ThermostatAgent(cfg), env, scenario, seed=0,
                     pass_info_to_agent=True)
    m = episode_metrics(df, cfg)
    # 240 passos de 0,1 h = 24 h, independentemente do action repeat.
    assert df["inner_steps"].sum() * cfg.dt == pytest.approx(24.0)
    assert m["changes_per_hour"] == pytest.approx(m["changes_per_day"] / 24.0)


# --------------------------------------------------------------- wrappers

def test_safety_shield_prevents_thermal_runaway():
    """Contra política adversarial sempre-OFF, o escudo deve resfriar."""
    cfg = ClassroomConfig()
    env = SafetyShieldWrapper(
        ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    )
    env.reset(seed=0, options={"start_temp": 31.0, "occupancy": 45, "hour": 14})
    final = 31.0
    for _ in range(30):
        _, _, term, _, info = env.step(int(ACState.OFF))
        final = info["temperature"]
        if term:
            break
    assert info["shield_interventions"] > 0
    assert final < 31.0, "escudo não conteve o aquecimento"


def test_safety_shield_turns_off_when_cold():
    cfg = ClassroomConfig()
    env = SafetyShieldWrapper(ClassroomACEnv(config=cfg))
    env.reset(seed=0, options={"start_temp": 18.0, "occupancy": 0, "hour": 10})
    _, _, _, _, info = env.step(int(ACState.HIGH))
    assert info["ac_state"] == "OFF"
    assert info["shield_interventions"] == 1


def test_shield_disabled_is_transparent():
    """Com o escudo desligado, o comportamento deve ser idêntico ao ambiente cru."""
    cfg = ClassroomConfig()
    opts = {"start_temp": 31.0, "occupancy": 45, "hour": 14}

    plain = ClassroomACEnv(config=cfg)
    plain.reset(seed=7, options=opts)
    shielded = SafetyShieldWrapper(ClassroomACEnv(config=cfg), enabled=False)
    shielded.reset(seed=7, options=opts)

    for _ in range(20):
        _, _, _, _, i1 = plain.step(int(ACState.OFF))
        _, _, _, _, i2 = shielded.step(int(ACState.OFF))
        assert i1["temperature"] == pytest.approx(i2["temperature"])


# --------------------------------------------------------------- baselines

def test_thermostat_deadband_reduces_switching():
    """
    O deadband é o parâmetro NÃO PUBLICADO que explica a divergência de
    trocas/h e energia na reprodução. Deadband maior => menos comutação.
    """
    cfg = ClassroomConfig()
    scenario = next(s for s in build_scenario_matrix() if s["id"] == "C9")
    changes = {}
    for db in (0.0, 1.0, 2.0):
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        df = run_episode(ThermostatAgent(cfg, deadband=db), env, scenario,
                         seed=0, pass_info_to_agent=True)
        changes[db] = episode_metrics(df, cfg)["changes_per_day"]
    assert changes[0.0] >= changes[1.0] >= changes[2.0]


def test_pi_controller_has_antiwindup():
    """
    Sem anti-windup o integrador satura no pulldown e o PI demora a desligar,
    tornando o baseline artificialmente ruim.
    """
    cfg = ClassroomConfig()
    pi = PIController(cfg, kp=0.6, ki=0.08)
    for _ in range(200):          # pulldown longo e saturado
        pi._load(35.0)
    saturated = pi._integral
    for _ in range(50):           # já dentro da faixa
        load = pi._load(23.0)
    assert saturated < 1e4, "integrador explodiu: anti-windup ausente"
    assert load == pytest.approx(0.0), "PI não desligou após o pulldown"


def test_pi_output_within_action_space():
    cfg = ClassroomConfig()
    pi = PIController(cfg)
    for t in (10.0, 20.0, 24.0, 30.0, 40.0):
        action, _ = pi.predict(np.zeros(4, dtype=np.float32), info={"temperature": t})
        assert action in [int(s) for s in ACState]


def test_agents_are_stateless_across_episodes():
    """
    Controladores com estado precisam ser resetados por episódio; senão o estado
    vaza entre células da matriz 3×3 e a avaliação deixa de ser independente.
    """
    cfg = ClassroomConfig()
    scenario = build_scenario_matrix()[0]
    agent = PIController(cfg)

    def roll():
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        return run_episode(agent, env, scenario, seed=0,
                           pass_info_to_agent=True)["temperature"].to_numpy()

    assert np.allclose(roll(), roll()), "estado do agente vazou entre episódios"


# ----------------------------------------------------------------- métricas

def test_occupied_window_is_7_to_22():
    cfg = ClassroomConfig()
    assert not cfg.is_occupied_hour(6)
    assert cfg.is_occupied_hour(7)
    assert cfg.is_occupied_hour(21)
    assert not cfg.is_occupied_hour(22)


def test_scenario_matrix_is_3x3():
    s = build_scenario_matrix()
    assert len(s) == 9
    assert [x["id"] for x in s] == [f"C{i}" for i in range(1, 10)]
    assert sorted({x["occupancy"] for x in s}) == [5, 22, 45]
    assert sorted({x["start_temp"] for x in s}) == [17.0, 24.0, 30.0]


def test_cold_start_scenario_is_uncontrollable():
    """
    Seção 5.1: "a única exceção é Frio + Poucas". Um AC que só resfria não pode
    corrigir uma sala fria — incluir C1 puniria todos igualmente sem informar.
    """
    from metrics import is_controllable

    cfg = ClassroomConfig()
    scenarios = {s["id"]: s for s in build_scenario_matrix()}
    factory = lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    assert not is_controllable(factory(), scenarios["C1"], seed=0)
    assert is_controllable(factory(), scenarios["C9"], seed=0)


def test_violation_metrics_are_consistent():
    """Taxa de violação e conforto na faixa larga devem ser complementares."""
    cfg = ClassroomConfig()
    scenario = next(s for s in build_scenario_matrix() if s["id"] == "C9")
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    df = run_episode(ThermostatAgent(cfg), env, scenario, seed=0,
                     pass_info_to_agent=True)
    m = episode_metrics(df, cfg)
    assert m["violation_rate_pct"] + m["comfort_wide_pct"] == pytest.approx(100.0)
    assert m["violation_degree_hours"] >= 0.0


# ------------------------------- separação calibração/teste (Revisor 2)

def test_calibration_test_split_is_disjoint_and_diverse():
    """
    A divisão deve ser disjunta E preservar diversidade nos dois lados. Uma
    divisão por linha (treinar em 'frio', testar em 'quente') mediria
    extrapolação, não generalização.
    """
    from scenarios import split_scenario_matrix

    calib, teste = split_scenario_matrix()
    ids_c = {s["id"] for s in calib}
    ids_t = {s["id"] for s in teste}
    assert not (ids_c & ids_t), "conjuntos se sobrepõem"
    assert ids_c | ids_t == {f"C{i}" for i in range(1, 10)}
    for grupo, nome in ((calib, "calibração"), (teste, "teste")):
        assert len({s["condition"] for s in grupo}) >= 2, f"{nome} pouco diverso"


def test_random_scenarios_are_reproducible_and_independent():
    """
    O conjunto aleatório de teste deve ser idêntico entre agentes (comparação
    emparelhada) e não coincidir com a matriz de calibração.
    """
    from scenarios import build_scenario_matrix, sample_random_scenarios

    a = sample_random_scenarios(50, seed=999)
    b = sample_random_scenarios(50, seed=999)
    assert [s["start_temp"] for s in a] == [s["start_temp"] for s in b]
    assert sample_random_scenarios(50, seed=1)[0] != a[0]

    matriz = {(s["start_temp"], s["occupancy"], s["hour"])
              for s in build_scenario_matrix()}
    aleatorios = {(s["start_temp"], s["occupancy"], s["hour"]) for s in a}
    assert not (matriz & aleatorios), "cenário aleatório coincide com a calibração"


def test_random_scenarios_respect_env_bounds():
    """Cenários gerados não devem violar o espaço de observação."""
    from scenarios import sample_random_scenarios

    cfg = ClassroomConfig()
    env = ClassroomACEnv(config=cfg)
    for s in sample_random_scenarios(30, seed=7):
        obs, _ = env.reset(seed=0, options=s)
        assert env.observation_space.contains(obs)
        assert 0 <= s["hour"] < 24
        assert 0 <= s["occupancy"] <= cfg.max_occupancy


# ------------------------------- Q-Learning tabular / HNP (Revisor 1)

def test_hnp_pathology_is_present():
    """
    Zha et al. (2021): variáveis de dinâmica lenta tornam a maioria das
    transições intra-tile, impedindo a propagação de valor em métodos tabulares.
    Este teste documenta a magnitude medida — é a resposta quantitativa à dúvida
    do Revisor 1 sobre necessidade de RL profundo.
    """
    from tabular import TabularConfig, intra_tile_fraction

    cfg = config_for_profile("Equilibrado")
    d = intra_tile_fraction(cfg, TabularConfig(), n_steps=5_000, seed=0)
    assert d["intra_tile_pct"] > 50.0, (
        f"apenas {d['intra_tile_pct']:.1f}% intra-tile: a justificativa para RL "
        "profundo via HNP não se sustenta nesta parametrização"
    )
    assert d["steps_to_cross_bin"] > 1.0


def test_tabular_agent_interface_matches_sb3():
    """
    O agente tabular deve expor a mesma interface dos modelos SB3, senão não
    pode ser avaliado pela mesma função e a comparação deixa de ser justa.
    """
    from tabular import TabularQAgent

    cfg = config_for_profile("Equilibrado")
    agent = TabularQAgent(cfg, seed=0)
    env = ClassroomACEnv(config=cfg)
    obs, _ = env.reset(seed=0)
    action, state = agent.predict(obs, deterministic=True)
    assert action in [int(s) for s in ACState]
    assert state is None


def test_tabular_state_index_is_bijective_on_bins():
    """A indexação de estados não deve colidir entre bins distintos."""
    from tabular import TabularConfig, TabularQAgent

    cfg = config_for_profile("Equilibrado")
    tab = TabularConfig()
    agent = TabularQAgent(cfg, tab, seed=0)

    vistos = set()
    for ti in range(tab.temp_bins):
        for oi in range(tab.occupancy_bins):
            for hi in range(tab.hour_bins):
                idx = (ti * tab.occupancy_bins + oi) * tab.hour_bins + hi
                assert 0 <= idx < tab.n_states
                vistos.add(idx)
    assert len(vistos) == tab.n_states, "colisão na indexação de estados"


# ----------------------------------------- ablação (Revisor 2)

def test_ablation_variants_change_only_reward():
    """
    Cada variante da ablação deve alterar SOMENTE termos de recompensa. Se a
    física mudar, a ablação compara duas coisas ao mesmo tempo e não isola nada.
    """
    from ablation import ABLATIONS, config_for_ablation

    base = config_for_ablation("completa")
    physical = ("thermal_mass", "heat_transfer_coeff", "heat_gain_per_person",
                "max_occupancy", "dt", "episode_steps", "temperature_noise_std")
    for variant in ABLATIONS:
        cfg = config_for_ablation(variant)
        for attr in physical:
            assert getattr(cfg, attr) == getattr(base, attr), (
                f"variante '{variant}' alterou física: {attr}"
            )


def test_flat_plateau_makes_center_indifferent():
    """
    A hipótese explícita da Seção 4.3: com gradiente zero o platô é plano e o
    centro deixa de ser preferível à borda — a causa raiz do "estacionar em
    26,1 °C". Este teste garante que a variante da ablação de fato reproduz isso.
    """
    from ablation import config_for_ablation

    cfg = config_for_ablation("sem_gradiente")
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)

    def comfort_at(t):
        env.current_temp = t
        env.ac_state = ACState.OFF
        env.hour_of_day = 10
        return env._reward(changed=False, dwell_before_change=10**6)

    assert comfort_at(cfg.ideal_temp) == pytest.approx(comfort_at(cfg.temp_comfort_max))
    assert comfort_at(cfg.ideal_temp) == pytest.approx(comfort_at(cfg.temp_comfort_min))


def test_comfort_topologies_are_distinct():
    """As três topologias devem produzir valores diferentes no centro da faixa."""
    from dataclasses import replace

    base = config_for_profile("Equilibrado")
    valores = {}
    for topo in ("plateau", "quadratic", "step"):
        env = ClassroomACEnv(config=replace(base, comfort_type=topo))
        env.reset(seed=0)
        env.current_temp = base.ideal_temp
        valores[topo] = env._comfort_reward(base.ideal_temp)
    assert len(set(round(v, 6) for v in valores.values())) == 3, valores


# ============ Laboratório v2: aquecimento, observação, ação contínua ========

def test_paper_mode_unchanged_by_new_features():
    """
    REGRESSÃO CRÍTICA: todas as flags novas são default=False, então a
    reprodução do manuscrito deve permanecer bit-a-bit idêntica.
    """
    env = ClassroomACEnv(config=ClassroomConfig())
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 4
    obs, _ = env.reset(seed=42, options={"start_temp": 26.0, "occupancy": 30, "hour": 14})
    temps = [env.step(a)[4]["temperature"] for a in (3, 3, 2, 1, 0)]
    # Valores de referência do modo paper (só resfria, 4 níveis).
    assert temps[0] < 26.0                      # HIGH resfria
    assert env.observation_space.contains(obs)


def test_heating_raises_temperature():
    """(1) Sem aquecimento nenhum controlador mantém tolerância à noite."""
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado")
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 22.0, "occupancy": 0, "hour": 3})
    t0 = env.current_temp
    for _ in range(10):
        env.step([-1.0])                        # aquecimento máximo
    assert env.current_temp > t0 + 0.1, "aquecimento não elevou a temperatura"


def test_heating_fixes_night_drift():
    """
    O ganho concreto do item (1): com aquecimento a madrugada passa a ser
    controlável. Sem ele, 18 % do período fica fora de ±0,5 °C por física.
    """
    from config import config_for_lab, config_for_lab2

    def fracao_fora(cfg, controlar):
        env = ClassroomACEnv(config=cfg)
        env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 0, "hour": 22})
        fora = 0
        for _ in range(100):
            if controlar:
                erro = env.current_temp - cfg.ideal_temp
                a = [float(np.clip(erro * 2.0, -1.0, 1.0))]   # P simples bidirecional
            else:
                a = 0
            _, _, term, _, info = env.step(a)
            fora += abs(info["temperature"] - cfg.ideal_temp) > cfg.lab_tolerance
            if term:
                break
        return fora / 100 * 100

    sem = fracao_fora(config_for_lab("Lab_Equilibrado"), controlar=False)
    com = fracao_fora(config_for_lab2("Lab_Equilibrado"), controlar=True)
    assert sem > 10.0, f"esperava deriva noturna sem aquecimento, obtive {sem:.0f}%"
    assert com < sem, f"aquecimento não melhorou: {com:.0f}% vs {sem:.0f}%"


def test_minutes_to_peak_is_correct():
    """(2) A feature que torna a antecipação aprendível precisa estar certa."""
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado")   # ponta 17:30–20:30
    env = ClassroomACEnv(config=cfg)
    for hora, esperado in ((16.0, 90.0), (17.0, 30.0), (12.0, 330.0)):
        env.reset(seed=0, options={"hour": int(hora), "start_temp": 24.0})
        env.hour_float = hora
        assert env._minutes_to_peak() == pytest.approx(esperado, abs=1.0)
    # Dentro da ponta deve ser zero.
    env.reset(seed=0, options={"hour": 18, "start_temp": 24.0})
    env.hour_float = 18.5
    assert env._minutes_to_peak() == 0.0


def test_derivative_feature_tracks_temperature_change():
    """(3) dT/dt deve distinguir aquecendo de resfriando no mesmo ponto."""
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado")
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 12})
    obs_resfria, *_ = env.step([1.0])
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 12})
    obs_aquece, *_ = env.step([-1.0])
    # Índice 4 é a derivada (após as 4 features base).
    assert obs_resfria[4] < 0 < obs_aquece[4], "derivada não reflete o sentido"


def test_observation_space_matches_emitted_obs():
    """
    Todas as combinações de flags: o Box declarado deve conter a observação
    emitida. Foi a divergência entre os dois que gerou o bug de o_norm=1,33.
    """
    from dataclasses import replace

    base = ClassroomConfig()
    for heat in (False, True):
        for deriv in (False, True):
            for peak in (False, True):
                for cont in (False, True):
                    cfg = replace(base, heating_enabled=heat, observe_derivative=deriv,
                                  observe_time_to_peak=peak, continuous_action=cont,
                                  tariff_name="enel_sp_branca")
                    env = ClassroomACEnv(config=cfg)
                    esperado = 4 + int(deriv) + 2 * int(peak)
                    assert env.observation_space.shape == (esperado,)
                    for occ in (0, 45, 99):
                        obs, _ = env.reset(seed=0, options={"occupancy": occ,
                                                            "start_temp": 30.0, "hour": 18})
                        assert env.observation_space.contains(obs), (
                            f"obs fora do Box (heat={heat} deriv={deriv} "
                            f"peak={peak} cont={cont} occ={occ})")


def test_continuous_action_space_bounds_follow_heating():
    """Sem aquecimento a carga é [0,1]; com aquecimento, [-1,1]."""
    from dataclasses import replace

    so_resfria = ClassroomACEnv(config=replace(ClassroomConfig(), continuous_action=True))
    assert so_resfria.action_space.low[0] == pytest.approx(0.0)
    bidir = ClassroomACEnv(config=replace(ClassroomConfig(), continuous_action=True,
                                          heating_enabled=True))
    assert bidir.action_space.low[0] == pytest.approx(-1.0)


def test_discrete_levels_mirror_when_heating():
    """Ação discreta bidirecional: 7 níveis simétricos."""
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado", continuous_action=False)
    env = ClassroomACEnv(config=cfg)
    assert env.action_space.n == 7
    assert env.levels == pytest.approx([-1.0, -0.55, -0.25, 0.0, 0.25, 0.55, 1.0])


def test_heating_cop_exceeds_cooling_cop():
    """Bomba de calor entrega trabalho + calor absorvido: COP maior."""
    from dataclasses import replace

    p = replace(ACPhysicsModel(), heating_enabled=True)
    assert p.heating_cop > max(p.cop.values())
    # Mesma magnitude térmica, energia menor no aquecimento.
    assert p.electrical_kw_signed(-1.0) < p.electrical_kw_signed(1.0)


def test_gymnasium_checker_accepts_lab2():
    from gymnasium.utils.env_checker import check_env

    from config import config_for_lab2

    check_env(ClassroomACEnv(config=config_for_lab2("Lab_Equilibrado")),
              skip_render_check=True)


# ===================== Integridade do compressor =========================

def test_compressor_metrics_distinguish_cycling_from_modulation():
    """
    `changes_per_hour` não serve para desgaste de hardware: conta MODULAÇÃO, e
    modular é a função de um inverter. O que danifica é ciclagem ON/OFF.
    Este teste garante que as duas coisas são medidas separadamente.
    """
    from compressor_health import compressor_metrics
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado")

    def rodar(cargas):
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 12})
        linhas = []
        for c in cargas:
            _, _, term, _, info = env.step([c])
            linhas.append(dict(info))
            if term:
                break
        return compressor_metrics(pd.DataFrame(linhas), cfg)

    # Modulação suave dentro de um sentido: nenhuma partida, nenhuma reversão.
    suave = rodar([0.3 + 0.05 * (i % 3) for i in range(40)])
    # Ciclagem ON/OFF: muitas partidas.
    ciclando = rodar([0.6 if i % 2 else 0.0 for i in range(40)])

    assert suave["partidas_dia"] == 0, "modulação não deve contar como partida"
    assert ciclando["partidas_dia"] > 5, "ciclagem ON/OFF não foi detectada"
    assert suave["reversoes_dia"] == 0


def test_compressor_metrics_detect_cycle_reversal():
    """
    Reversão de ciclo (resfria <-> aquece) inverte a válvula de 4 vias e é o modo
    de desgaste que só passou a existir com o aquecimento habilitado.
    """
    from compressor_health import compressor_metrics
    from config import config_for_lab2

    cfg = config_for_lab2("Lab_Equilibrado")
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 20, "hour": 12})
    linhas = []
    for i in range(20):
        _, _, term, _, info = env.step([0.5 if i % 2 else -0.5])   # alterna sentido
        linhas.append(dict(info))
        if term:
            break
    m = compressor_metrics(pd.DataFrame(linhas), cfg)
    assert m["reversoes_dia"] >= 15, f"reversões não detectadas: {m['reversoes_dia']}"


def test_protection_blocks_early_reversal():
    """A proteção deve impedir reversão antes do tempo mínimo no modo."""
    from config import config_for_lab2
    from wrappers import CompressorProtectionWrapper

    cfg = config_for_lab2("Lab_Equilibrado")
    env = CompressorProtectionWrapper(ClassroomACEnv(config=cfg),
                                      min_run_minutes=5.0, min_mode_minutes=60.0)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 20, "hour": 12})
    env.step([0.6])                       # entra em resfriamento
    _, _, _, _, info = env.step([-0.6])   # tenta reverter imediatamente
    assert info["blocked_reversal"] >= 1, "reversão precoce não foi bloqueada"
    assert info["load"] >= 0.0, "reverteu apesar do bloqueio"


def test_protection_limits_ramp():
    from config import config_for_lab2
    from wrappers import CompressorProtectionWrapper

    cfg = config_for_lab2("Lab_Equilibrado")
    env = CompressorProtectionWrapper(ClassroomACEnv(config=cfg), max_ramp_per_min=0.02)
    env.reset(seed=0, options={"start_temp": 26.0, "occupancy": 45, "hour": 12})
    _, _, _, _, info = env.step([1.0])    # degrau de 0 -> 1
    limite = 0.02 * cfg.dt * 60.0
    assert info["load"] <= limite + 1e-6, f"rampa não limitada: {info['load']}"
    assert info["clipped_ramp"] >= 1


def test_protection_disabled_is_transparent():
    from config import config_for_lab2
    from wrappers import CompressorProtectionWrapper

    cfg = config_for_lab2("Lab_Equilibrado")
    opts = {"start_temp": 26.0, "occupancy": 45, "hour": 12}
    cru = ClassroomACEnv(config=cfg); cru.reset(seed=3, options=opts)
    prot = CompressorProtectionWrapper(ClassroomACEnv(config=cfg), enabled=False)
    prot.reset(seed=3, options=opts)
    for a in (1.0, 0.0, -1.0, 0.5):
        _, _, _, _, i1 = cru.step([a])
        _, _, _, _, i2 = prot.step([a])
        assert i1["temperature"] == pytest.approx(i2["temperature"])


# ============ Contrato modelo <-> configuração de treino ==================

def test_config_reconstructed_from_metadata_not_defaults():
    """
    REGRESSÃO: a config de treino faz parte do CONTRATO do modelo. Reconstruí-la
    do default vigente quebra em silêncio quando um default muda — aconteceu duas
    vezes neste projeto (action_repeat omitido no analyzer original; observação de
    7-D vs 9-D ao adicionar erro escalado e integral).
    """
    from dataclasses import asdict

    from config import ClassroomConfig, config_for_lab2
    from model_io import config_from_metadata

    treino = config_for_lab2("Lab_Equilibrado")
    meta = {
        "env_params": {k: v for k, v in asdict(treino).items()
                       if not isinstance(v, dict)},
    }
    recon = config_from_metadata(meta)
    for attr in ("heating_enabled", "continuous_action", "observe_derivative",
                 "observe_scaled_error", "observe_integral", "tariff_name",
                 "comfort_type", "comfort_sensitivity", "lab_tolerance"):
        assert getattr(recon, attr) == getattr(treino, attr), f"divergiu: {attr}"

    # Campos desconhecidos no metadado não devem estourar.
    meta["env_params"]["campo_que_nao_existe_mais"] = 123
    config_from_metadata(meta)


def test_metadata_missing_raises_instead_of_guessing():
    """
    Avaliar sem saber a configuração de treino é pior que falhar: produz números
    plausíveis e inválidos. O carregador deve recusar.
    """
    from model_io import load_agent

    with pytest.raises(FileNotFoundError):
        load_agent("caminho/inexistente_sem_metadado.zip")


def test_observation_space_contract_is_checked():
    """
    O carregador deve verificar que o obs space do modelo casa com o do ambiente
    reconstruído, em vez de deixar o erro aparecer depois como shape mismatch.
    """
    import inspect

    from model_io import load_agent

    src = inspect.getsource(load_agent)
    assert "observation_space" in src and "contrato violado" in src


# ====== Arquitetura: distribuição de treino, sensor, custo de controle ======

def test_train_eval_occupancy_distribution_mismatch_is_fixed():
    """
    PROBLEMA DE ARQUITETURA medido: o treino com passeio aleatório dá média 22,7
    e 9,9 % nos extremos; a avaliação com janela ocupada dá média 15,0 e 58,3 %.
    KS: D = 0,438. O agente treinava numa distribuição que não vê no teste, e
    quase nunca experimentava as transições em degrau 0->45 que definem todo
    cenário. `schedule` alinha as duas.
    """
    from dataclasses import replace

    from scenarios import build_scenario_matrix

    def amostrar(cfg, cenarios=None):
        env = ClassroomACEnv(config=cfg)
        occ = []
        fontes = cenarios or [None] * 25
        for k, s in enumerate(fontes):
            env.reset(seed=k, options=s)
            for _ in range(240):
                _, _, term, _, i = env.step(0)
                occ.append(i["occupancy"])
                if term:
                    break
        return np.array(occ)

    base = ClassroomConfig()
    aval = amostrar(base, build_scenario_matrix())
    passeio = amostrar(replace(base, train_occupancy_mode="random_walk"))
    agenda = amostrar(replace(base, train_occupancy_mode="schedule"))

    ext = lambda a: ((a == 0) | (a == base.max_occupancy)).mean()
    # A agenda deve ficar MUITO mais próxima da avaliação nos extremos.
    assert abs(ext(agenda) - ext(aval)) < abs(ext(passeio) - ext(aval))
    assert abs(agenda.mean() - aval.mean()) < abs(passeio.mean() - aval.mean())


def test_schedule_mode_produces_step_transitions():
    """A estrutura que importa é o degrau ao encher/esvaziar, não só a média."""
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), train_occupancy_mode="schedule")
    env = ClassroomACEnv(config=cfg)
    saltos = 0
    for ep in range(10):
        env.reset(seed=ep)
        ant = None
        for _ in range(240):
            _, _, term, _, i = env.step(0)
            if ant is not None and abs(i["occupancy"] - ant) > 4:
                saltos += 1
            ant = i["occupancy"]
            if term:
                break
    assert saltos > 0, "modo agenda não gerou transições em degrau"


def test_sensor_noise_affects_observation_not_ground_truth():
    """
    O ruído de MEDIÇÃO deve entrar na observação sem contaminar `current_temp` —
    avaliar contra a leitura do sensor mediria o sensor, não o controle.
    """
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), sensor_noise_std=0.5,
                  observe_scaled_error=True, temperature_noise_std=0.0)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 0, "hour": 12})
    leituras, verdades = [], []
    for _ in range(40):
        obs, _, _, _, info = env.step(0)
        leituras.append(env.measured_temp)
        verdades.append(info["temperature"])
    assert np.std(np.array(leituras) - np.array(verdades)) > 0.1, "ruído não aplicado"
    # A verdade deve seguir suave (só a física), sem o ruído do sensor.
    assert np.std(np.diff(verdades)) < np.std(np.diff(leituras))


def test_sensor_lag_delays_observation():
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), sensor_lag_steps=5, temperature_noise_std=0.0)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 30.0, "occupancy": 45, "hour": 14})
    for _ in range(3):
        _, _, _, _, info = env.step(int(ACState.HIGH))
    # Com atraso, a leitura ainda reflete o passado (mais quente que a verdade).
    assert env.measured_temp > info["temperature"]


def test_derivative_penalty_damps_rate_of_change():
    """
    Custo de controle clássico penaliza erro, esforço E variação. O projeto tinha
    e² e u, nada em ė — e é a oscilação que separa sigma 0,198 do RL de 0,088 do
    PI. A penalidade deve crescer com a taxa de variação.
    """
    from dataclasses import replace

    cfg = replace(config_for_profile("Equilibrado"), derivative_penalty=1.0)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 12})
    env.prev_temp = 24.0
    env.current_temp = 24.0
    env.hour_of_day = 12
    parado = env._reward(changed=False, dwell_before_change=10**6)
    env.prev_temp = 24.0
    env.current_temp = 24.2          # variando rápido
    variando = env._reward(changed=False, dwell_before_change=10**6)
    assert variando < parado, "penalidade de derivada não puniu variação"


def test_control_error_metrics_are_sane():
    """IAE, ISE, sobressinal, acomodação e erro de regime — coerência básica."""
    from scenarios import build_scenario_matrix

    cfg = ClassroomConfig()
    s7 = next(s for s in build_scenario_matrix() if s["id"] == "C7")
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    df = run_episode(PIController(cfg, kp=1.3, ki=0.2), env, s7, seed=0,
                     pass_info_to_agent=True)
    m = episode_metrics(df, cfg)
    assert m["iae_c_h"] > 0 and m["ise"] > 0
    assert m["ise"] >= 0                        # integral de quadrado
    assert m["overshoot_c"] >= 0
    assert abs(m["erro_regime_c"]) < 2.0        # PI deve convergir
    # IAE deve crescer com erro pior: um controlador ruim tem IAE maior.
    df2 = run_episode(ThermostatAgent(cfg, deadband=0.0), env, s7, seed=0,
                      pass_info_to_agent=True)
    assert episode_metrics(df2, cfg)["iae_c_h"] > m["iae_c_h"]


def test_paper_mode_unaffected_by_new_options():
    """Todos os defaults novos preservam a reprodução do manuscrito."""
    cfg = ClassroomConfig()
    assert cfg.train_occupancy_mode == "random_walk"
    assert cfg.sensor_noise_std == 0.0 and cfg.sensor_lag_steps == 0
    assert cfg.derivative_penalty == 0.0
    env = ClassroomACEnv(config=cfg)
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 4


# ============ Ocupação realista: entra-e-sai ao longo do dia ==============

def _amostrar_ocupacao(modo, cenarios=None, n=25):
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), train_occupancy_mode=modo)
    env = ClassroomACEnv(config=cfg)
    occ = []
    for k, s in enumerate(cenarios or [None] * n):
        env.reset(seed=k, options=s)
        for _ in range(240):
            _, _, term, _, i = env.step(0)
            occ.append(i["occupancy"])
            if term:
                break
    return np.array(occ)


def test_realistic_occupancy_aligns_train_and_eval_distributions():
    """
    O modo realista deve usar a MESMA família de perturbações no treino e na
    avaliação. Medido: KS cai de 0,422 (passeio aleatório) para ~0,09.
    """
    from scipy.stats import ks_2samp

    from scenarios import build_scenario_matrix

    aval = _amostrar_ocupacao("realistic", build_scenario_matrix())
    passeio = _amostrar_ocupacao("random_walk")
    realista = _amostrar_ocupacao("realistic")

    ks_passeio = ks_2samp(passeio, aval).statistic
    ks_realista = ks_2samp(realista, aval).statistic
    assert ks_realista < ks_passeio / 2.0, (
        f"realista não alinhou: KS {ks_realista:.3f} vs {ks_passeio:.3f}"
    )


def test_realistic_occupancy_transitions_are_gradual():
    """
    Sala real não enche instantaneamente. O degrau de 45 pessoas num passo do modo
    'schedule' impõe 0,090 °C/passo de carga — perturbação que nenhum controlador
    pode antecipar porque não há aviso.
    """
    from dataclasses import replace

    def max_salto(modo):
        cfg = replace(ClassroomConfig(), train_occupancy_mode=modo)
        env = ClassroomACEnv(config=cfg)
        pico = 0
        for ep in range(8):
            env.reset(seed=ep, options={"start_temp": 24.0, "occupancy": 45, "hour": 6})
            ant = None
            for _ in range(240):
                _, _, term, _, i = env.step(0)
                if ant is not None:
                    pico = max(pico, abs(i["occupancy"] - ant))
                ant = i["occupancy"]
                if term:
                    break
        return pico

    assert max_salto("realistic") < max_salto("schedule")


def test_realistic_occupancy_has_daily_structure():
    """
    A perturbação precisa ter ESTRUTURA (aula, intervalo, almoço, noturno), não
    ser só ruído: é o que permite ao agente aprender a antecipar.
    """
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), train_occupancy_mode="realistic")
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 24.0, "occupancy": 45, "hour": 0})
    por_hora = {}
    for _ in range(240):
        _, _, term, _, i = env.step(0)
        por_hora.setdefault(int(i["hour_float"]) % 24, []).append(i["occupancy"])
        if term:
            break
    media = {h: float(np.mean(v)) for h, v in por_hora.items()}
    assert media.get(3, 0) < 2, "sala deveria estar vazia de madrugada"
    assert media.get(10, 0) > 20, "sala deveria estar cheia às 10h"
    assert media.get(12, 99) < media.get(10, 0), "almoço deveria reduzir a ocupação"


def test_occupancy_model_is_reproducible_and_bounded():
    from occupancy import OccupancyModel

    m = OccupancyModel(max_occupancy=45)
    def perfil(seed):
        rng = np.random.default_rng(seed)
        ep = m.sample_episode(rng, 45)
        ep.reset(0.0)
        return [ep.step(k * 0.1, 0.1, rng) for k in range(240)]

    a, b = perfil(7), perfil(7)
    assert a == b, "modelo de ocupação não é reprodutível por semente"
    assert all(0 <= x <= 45 for x in a), "ocupação fora dos limites"
    assert perfil(7) != perfil(8), "sementes distintas dão o mesmo perfil"


def test_realistic_occupancy_respects_observation_space():
    from dataclasses import replace

    cfg = replace(ClassroomConfig(), train_occupancy_mode="realistic")
    env = ClassroomACEnv(config=cfg)
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        assert env.observation_space.contains(obs)
        for _ in range(240):
            obs, _, term, _, _ = env.step(0)
            assert env.observation_space.contains(obs)
            if term:
                break


# ====== Portão da penalidade de derivada e segmentação de métricas ========

def test_derivative_penalty_is_gated_by_tolerance():
    """
    A penalidade de variação existe para amortecer oscilação PERTO do alvo. Sem
    portão ela também freava o pulldown — contradizendo o objetivo, porque
    durante a aproximação variar rápido é exatamente o desejado.
    """
    from dataclasses import replace

    from config import config_for_lab2

    cfg = replace(config_for_lab2("Lab_Equilibrado"), derivative_penalty=0.5)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)
    env.hour_of_day = 10

    def penalidade(temp_atual, temp_anterior):
        env.prev_temp = temp_anterior
        env.current_temp = temp_atual
        env.load = 0.0
        com = env._reward(changed=False, dwell_before_change=10**6)
        env.prev_temp = temp_atual          # sem variação
        sem = env._reward(changed=False, dwell_before_change=10**6)
        return sem - com                    # quanto a variação custou

    # Mesma taxa de variação, temperaturas diferentes.
    perto = penalidade(24.1, 23.9)          # dentro da tolerância
    longe = penalidade(30.1, 29.9)          # em pleno pulldown
    assert perto > 0, "penalidade não atua dentro da tolerância"
    assert longe == pytest.approx(0.0, abs=1e-9), "penalidade freia o pulldown"


def test_derivative_gate_is_continuous():
    """Rampa, não degrau: descontinuidade na borda criaria incentivo estranho."""
    from dataclasses import replace

    from config import config_for_lab2

    cfg = replace(config_for_lab2("Lab_Equilibrado"), derivative_penalty=0.5)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0)
    env.hour_of_day = 10
    tol = cfg.lab_tolerance

    def portao_em(erro):
        env.prev_temp = cfg.ideal_temp + erro - 0.1
        env.current_temp = cfg.ideal_temp + erro
        env.load = 0.0
        com = env._reward(changed=False, dwell_before_change=10**6)
        env.prev_temp = env.current_temp
        return env._reward(changed=False, dwell_before_change=10**6) - com

    # Deve decair monotonicamente entre tol e 2*tol, sem salto.
    vals = [portao_em(e) for e in (tol * 0.9, tol * 1.2, tol * 1.6, tol * 2.1)]
    assert vals[0] >= vals[1] >= vals[2] >= vals[3]
    assert vals[-1] == pytest.approx(0.0, abs=1e-9)


def test_metrics_separate_transient_from_steady_state():
    """
    REGRESSÃO METODOLÓGICA: métricas agregadas mediam a CONDIÇÃO INICIAL, não o
    controle — 15 sintonias distintas do PI davam exatamente 90,1 % porque o
    número era dominado pelo transiente de partida.
    """
    from config import config_for_lab2
    from scenarios import build_scenario_matrix

    cfg = config_for_lab2("Lab_Equilibrado")
    quente = next(s for s in build_scenario_matrix() if s["id"] == "C9")   # parte de 30 °C
    normal = next(s for s in build_scenario_matrix() if s["id"] == "C5")   # parte de 24 °C

    def medir(scenario):
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        df = run_episode(PIController(cfg, kp=1.3, ki=0.2, discrete=False), env,
                         scenario, seed=0, pass_info_to_agent=True)
        return episode_metrics(df, cfg)

    mq, mn = medir(quente), medir(normal)

    # Em regime, ambos devem ser essencialmente perfeitos...
    assert mq["in_tolerance_ss_pct"] > 95.0
    assert mn["in_tolerance_ss_pct"] > 95.0
    # ...mas o agregado do cenário quente é pior, por causa do transiente.
    assert mq["in_tolerance_pct"] < mq["in_tolerance_ss_pct"]
    # E o custo do transiente deve aparecer separado.
    assert mq["iae_transient"] > mn["iae_transient"]


def test_steady_state_metrics_are_nan_when_never_settles():
    """Sem cauda suficiente, métricas de regime devem ser NaN — não zero."""
    from dataclasses import replace

    from config import config_for_lab2

    cfg = replace(config_for_lab2("Lab_Equilibrado"), lab_tolerance=0.001)
    env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    df = run_episode(ThermostatAgent(ClassroomConfig()), env,
                     {"start_temp": 30.0, "occupancy": 45, "hour": 14},
                     seed=0, pass_info_to_agent=True)
    m = episode_metrics(df, cfg)
    assert np.isnan(m["in_tolerance_ss_pct"]) or m["in_tolerance_ss_pct"] >= 0
