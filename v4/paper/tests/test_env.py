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
