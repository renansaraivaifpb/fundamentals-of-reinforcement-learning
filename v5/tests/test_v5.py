# -*- coding: utf-8 -*-
"""
Testes das garantias que a v5 acrescenta.

Cada teste corresponde a um modo de falha REAL observado na v4, e não a
cobertura genérica.
"""
from dataclasses import replace

import numpy as np
import pytest

from hvac import features
from hvac.config import (ABLATION_GROUPS, ClassroomConfig, config_ablacao,
                         config_for_lab2, config_for_profile)
from hvac.env import ClassroomACEnv


# ============ Declaração única da observação =============================

def test_box_e_vetor_vem_da_mesma_declaracao():
    """
    A garantia central da v5: é impossível o `Box` e o vetor divergirem.

    Na v4 eram duas listas paralelas mantidas à mão; a divergência entre elas
    produziu o bug documentado de `o_norm = 1,33` fora do espaço declarado.
    """
    for cfg in (ClassroomConfig(), config_for_lab2("Lab_Equilibrado"),
                config_for_profile("Agressivo")):
        env = ClassroomACEnv(config=cfg)
        n = len(features.features_ativas(cfg))
        assert env.observation_space.shape == (n,)
        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            assert obs.shape == (n,)
            assert env.observation_space.contains(obs)
            for _ in range(25):
                obs, *_ = env.step(env.action_space.sample())
                assert env.observation_space.contains(obs), (
                    f"observação fora do Box declarado: {obs}")


def test_observacao_nunca_escapa_dos_limites_em_condicao_extrema():
    """
    Regressão direta do bug de o_norm = 1,33: 45 ocupantes com `max_occupancy`
    menor produzia razão > 1. O clip vive agora nos MESMOS números que declaram
    o Box, então a classe inteira de bug deixa de ser possível.
    """
    cfg = replace(config_for_lab2("Lab_Equilibrado"), max_occupancy=30)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"start_temp": 39.0, "occupancy": 30, "hour": 14})
    env.occupancy = 45                      # acima do máximo declarado
    obs = env._get_obs()
    assert env.observation_space.contains(obs)
    assert obs[features.nomes(cfg).index("ocupacao")] <= 1.0


def test_schema_nomeia_os_canais_na_ordem():
    cfg = config_for_lab2("Lab_Equilibrado")
    esquema = features.schema(cfg)
    assert esquema["n"] == len(esquema["nomes"]) == ClassroomACEnv(
        config=cfg).observation_space.shape[0]
    assert esquema["nomes"][0] == "t_norm"
    # Canais opcionais desligados somem do schema, e não viram zeros mudos.
    sem = replace(cfg, observe_integral=False)
    assert "integral" not in features.schema(sem)["nomes"]
    assert features.schema(sem)["n"] == esquema["n"] - 1


def test_ablacao_de_observacao_muda_schema_e_espaco_juntos():
    """Desligar uma feature reduz Box e vetor simultaneamente — nunca só um."""
    base = config_for_lab2("Lab_Equilibrado")
    for grupo in ("obs_pid", "obs_derivada", "obs_tarifa"):
        abl = config_ablacao("Lab_Equilibrado", grupo)
        e_base, e_abl = ClassroomACEnv(config=base), ClassroomACEnv(config=abl)
        assert (e_abl.observation_space.shape[0]
                < e_base.observation_space.shape[0])
        obs, _ = e_abl.reset(seed=0)
        assert obs.shape == e_abl.observation_space.shape
        assert e_abl.obs_schema["n"] == obs.shape[0]


# ============ Contrato modelo <-> ambiente ===============================

class _ModeloFalso:
    def __init__(self, n):
        from gymnasium import spaces
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)


def test_contrato_rejeita_ordem_trocada_com_mesma_dimensao():
    """
    O modo de falha que a checagem de FORMA da v4 não pegava: mesma dimensão,
    canais em ordem diferente. O modelo carregaria e produziria métricas
    plausíveis lendo cada canal com o significado errado.
    """
    from hvac.model_io import assert_schema_compatible

    cfg = config_for_lab2("Lab_Equilibrado")
    env = ClassroomACEnv(config=cfg)
    n = env.observation_space.shape[0]

    trocado = list(env.obs_schema["nomes"])
    trocado[4], trocado[5] = trocado[5], trocado[4]
    with pytest.raises(ValueError, match="ordem/identidade"):
        assert_schema_compatible(_ModeloFalso(n), env,
                                 {"obs_schema": {"n": n, "nomes": trocado}}, "teste")

    # O schema correto passa.
    assert_schema_compatible(_ModeloFalso(n), env,
                             {"obs_schema": env.obs_schema}, "teste")


def test_contrato_avisa_quando_metadado_e_anterior_ao_schema():
    """Modelo da v4 não tem schema: verificar só a forma precisa ser explícito."""
    from hvac.model_io import assert_schema_compatible

    env = ClassroomACEnv(config=config_for_lab2("Lab_Equilibrado"))
    n = env.observation_space.shape[0]
    with pytest.warns(RuntimeWarning, match="obs_schema"):
        assert_schema_compatible(_ModeloFalso(n), env, {}, "modelo_v4")


def test_contrato_rejeita_dimensao_diferente():
    from hvac.model_io import assert_schema_compatible

    env = ClassroomACEnv(config=config_for_lab2("Lab_Equilibrado"))
    with pytest.raises(ValueError, match="contrato violado"):
        assert_schema_compatible(_ModeloFalso(3), env, {}, "teste")


# ============ Avaliação multi-semente ====================================

def test_avaliacao_multissemente_reporta_incerteza():
    """
    A v4 avaliava o achado central numa única semente. Aqui a unidade de reporte
    é (média, IC95, n) — sem isso não se pode afirmar que dois controladores
    diferem.
    """
    from hvac.baselines import PIController
    from hvac.evaluation import avaliar_multissemente, tabela_resumo
    from hvac.scenarios import build_scenario_matrix
    from hvac.wrappers import ActionRepeatWrapper

    cfg = config_for_profile("Equilibrado")
    fab = lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    cen = build_scenario_matrix()[:3]

    r = avaliar_multissemente(lambda: PIController(cfg, kp=1.3, ki=0.2),
                              fab, cen, cfg, seeds=(100, 101, 102),
                              pass_info_to_agent=True)
    assert len(r["por_semente"]) == 3
    m = r["resumo"]["comfort_wide_pct"]
    assert m["n"] == 3 and m["ic_baixo"] <= m["media"] <= m["ic_alto"]
    assert "±" in tabela_resumo({"PI": r})["comfort_wide_pct"].iloc[0]


def test_agente_com_estado_nasce_limpo_a_cada_semente():
    """
    A fábrica recebe uma callable, e não um agente pronto: o integrador do PI
    precisa zerar entre sementes, senão o estado de uma avaliação vaza para a
    seguinte e a comparação deixa de ser independente.
    """
    from hvac.baselines import PIController
    from hvac.evaluation import avaliar_multissemente
    from hvac.scenarios import build_scenario_matrix
    from hvac.wrappers import ActionRepeatWrapper

    cfg = config_for_profile("Equilibrado")
    fab = lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    cen = build_scenario_matrix()[:2]
    kw = dict(scenarios=cen, config=cfg, seeds=(100, 101), pass_info_to_agent=True)

    a = avaliar_multissemente(lambda: PIController(cfg, kp=1.3, ki=0.2), fab, **kw)
    b = avaliar_multissemente(lambda: PIController(cfg, kp=1.3, ki=0.2), fab, **kw)
    np.testing.assert_allclose(a["por_semente"]["comfort_wide_pct"],
                               b["por_semente"]["comfort_wide_pct"])


# ============ Grupos de ablação ==========================================

def test_todo_grupo_de_ablacao_muda_a_config():
    """Um grupo que não altera nada é um eixo morto, e mascara o experimento."""
    base = config_for_lab2("Lab_Equilibrado")
    for grupo in ABLATION_GROUPS:
        abl = config_ablacao("Lab_Equilibrado", grupo)
        difs = [k for k, v in ABLATION_GROUPS[grupo].items()
                if getattr(abl, k) != getattr(base, k)]
        assert difs, f"grupo '{grupo}' não altera a config"


def test_grupo_inexistente_falha_cedo():
    with pytest.raises(KeyError):
        config_ablacao("Lab_Equilibrado", "nao_existe")


# ============ Correções herdadas =========================================

def test_demanda_contratada_continua_derivada():
    cfg = config_for_lab2("Lab_Equilibrado")
    s = cfg.demand_sizing
    assert cfg.demand_contracted_kw == pytest.approx(
        s.steady_state_kw * (1.0 + cfg.demand_contract_margin))
    assert s.steady_state_kw < cfg.demand_contracted_kw < s.pulldown_kw


def test_config_propaga_aquecimento_sem_env():
    cfg = config_for_lab2("Lab_Equilibrado")
    assert cfg.physics.heating_enabled
    assert cfg.physics.electrical_kw_signed(-1.0) > 0.0


# ============ Aproveitamentos de Wei et al. (DAC 2017) ====================

def test_recompensa_hinge_reproduz_eq1_de_wei():
    """
    r = -lambda * ([T - T_max]+ + [T_min - T]+): zero dentro da faixa, linear
    fora. É a formulação mais simples da literatura fundacional, e o contraponto
    ao "Platô Quadrático com gradiente" do manuscrito.
    """
    cfg = replace(config_for_profile("Equilibrado"), comfort_type="hinge",
                  comfort_hinge_lambda=10.0)
    env = ClassroomACEnv(config=cfg)
    for t in (22.0, 24.0, 26.0):
        assert env._comfort_reward(t) == pytest.approx(0.0), "sem bônus dentro"
    assert env._comfort_reward(28.0) == pytest.approx(-20.0)   # 2 °C acima
    assert env._comfort_reward(20.0) == pytest.approx(-20.0)   # 2 °C abaixo
    # Linear, e não quadrática: dobrar o desvio dobra a penalidade.
    assert env._comfort_reward(30.0) == pytest.approx(2 * env._comfort_reward(28.0))


def test_previsao_externa_antecipa_a_senoide():
    """
    A previsão tem de ADIANTAR a externa, não repeti-la. Sem isso a feature seria
    uma cópia ruidosa do presente e não permitiria antecipação.
    """
    cfg = replace(config_for_profile("Equilibrado"),
                  observe_outdoor_forecast=True, forecast_steps=3,
                  forecast_horizon_hours=2.0)
    env = ClassroomACEnv(config=cfg)
    env.reset(seed=0, options={"hour": 8, "start_temp": 24.0, "occupancy": 20})
    nomes = features.nomes(cfg)
    obs = env._get_obs()
    prev = [obs[nomes.index(f"previsao_ext_{k}")] for k in (1, 2, 3)]
    # Às 8h a externa sobe rumo ao pico das 14h: a previsão tem de crescer.
    assert prev[0] < prev[1] < prev[2], f"previsão não acompanha a tendência: {prev}"
    # E tem de bater com a física, não ser um número qualquer.
    esperado = (env._outdoor_temp() - 15.0) / 20.0
    assert prev[0] > esperado, "previsão de 40 min à frente deveria exceder a atual"


def test_espaco_de_acao_multizona_explode_e_a_heuristica_contem():
    """
    Wei et al.: o espaço combinatório é m^z e degrada o treino. A heurística
    multinível o torna linear em z. Este teste fixa os dois em números.
    """
    from hvac.multizone import (MultiNivelWrapper, MultiZoneConfig,
                                MultiZoneEnv, ZoneSpec)

    for z, esperado in [(3, 64), (5, 1024)]:
        cfg = MultiZoneConfig(zones=tuple(ZoneSpec(chr(65 + i)) for i in range(z)))
        comb = MultiZoneEnv(cfg, acao_discreta=True)
        assert comb.contagem_acoes() == esperado
        assert comb.action_space.n == esperado
        mn = MultiNivelWrapper(MultiZoneEnv(cfg))
        assert mn.contagem_acoes() == 4 * z          # linear, não exponencial
        assert mn.contagem_acoes() < comb.contagem_acoes()


def test_heuristica_multinivel_respeita_o_nivel_global_e_a_prioridade():
    """A decomposição precisa fazer o que promete: teto de consumo + prioridade."""
    import numpy as np

    from hvac.multizone import MultiNivelWrapper, MultiZoneEnv

    env = MultiNivelWrapper(MultiZoneEnv())
    env.reset(seed=0)
    n = env.n
    # Nível global 0 => não aciona nada.
    assert np.allclose(env._traduz(0), 0.0)
    # Nível máximo => a soma das frações satura em 1.
    topo = (env.niveis_totais - 1) * n
    assert env._traduz(topo).sum() == pytest.approx(1.0)
    # A zona escolhida recebe a maior parcela, para o mesmo nível global.
    for zona in range(n):
        aloc = env._traduz(topo + zona)
        assert aloc.argmax() == zona, f"prioridade {zona} não recebeu o máximo"


def test_acao_discreta_multizona_decodifica_sem_ambiguidade():
    """Índice -> vetor de níveis tem de ser bijetivo, senão duas ações colidem."""
    from hvac.multizone import MultiZoneEnv

    env = MultiZoneEnv(acao_discreta=True)
    vistos = {tuple(env._decodifica(i)) for i in range(env.contagem_acoes())}
    assert len(vistos) == env.contagem_acoes(), "decodificação não é bijetiva"
