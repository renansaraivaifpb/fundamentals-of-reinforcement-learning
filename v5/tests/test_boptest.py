# -*- coding: utf-8 -*-
"""
Testes da ponte para o BOPTEST.

Rodam SEM Docker: o cliente é substituído por um emulador de brinquedo que
responde no mesmo formato da API. Um ambiente que só pudesse ser testado com o
serviço de pé não seria testado na prática, e é justamente a camada de tradução
— unidades, nomes de ponto, ordem dos canais — que erra em silêncio.

O teste de integração real vive em `test_boptest_integracao`, marcado para ser
pulado quando `BOPTEST_URL` não estiver acessível.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hvac.baselines import PIController, ThermostatAgent
from hvac.boptest.avaliacao import rodar_episodio
from hvac.boptest.client import BoptestClient
from hvac.boptest.env import BoptestClassroomEnv, ConfigBoptest, ZERO_C
from hvac.config import ClassroomConfig
from hvac import features


class ClienteFalso:
    """
    Emulador mínimo com a interface do `BoptestClient`.

    A física é deliberadamente grosseira — primeira ordem com ganho constante —
    porque o que está sob teste é a TRADUÇÃO entre a API e o ambiente, não o
    comportamento térmico. Registra as ações recebidas para que os testes possam
    afirmar o que foi efetivamente enviado ao emulador.
    """

    def __init__(self, t0_c: float = 30.0):
        self.testid = None
        self.temp_c = t0_c
        self.tempo_s = 0.0
        self.recebidas = []
        self.encerrado = False

    def selecionar(self, caso):
        self.testid = "teste-falso"
        return self.testid

    def cenario(self, **kw):
        return {"time_period": kw.get("periodo")}

    def inicializar(self, **kw):
        self.tempo_s = float(kw.get("inicio_s", 0.0))
        return {}

    def passo(self, segundos):
        self.dt_s = float(segundos)
        return {"step": segundos}

    def avancar(self, acao=None):
        acao = acao or {}
        self.recebidas.append(dict(acao))
        carga = float(acao.get("fcu_oveFan_u", 0.0))
        # Deriva para cima sem atuação, resfria proporcional à carga.
        self.temp_c += 0.4 - 2.0 * carga
        self.tempo_s += getattr(self, "dt_s", 720.0)
        return {"time": self.tempo_s,
                "zon_reaTRooAir_y": self.temp_c + ZERO_C,
                "zon_weaSta_reaWeaTDryBul_y": 30.0 + ZERO_C,
                "zon_reaCO2RooAir_y": 700.0,
                "fcu_reaPCoo_y": 3000.0 * carga,
                "fcu_reaPFan_y": 200.0 * carga,
                "fcu_reaPHea_y": 0.0}

    def kpis(self):
        return {"tdis_tot": 1.23, "ener_tot": 4.56, "cost_tot": 7.89}

    def encerrar(self):
        self.encerrado = True


@pytest.fixture
def env():
    return BoptestClassroomEnv(config=ClassroomConfig(),
                               bop=ConfigBoptest(passos=20),
                               cliente=ClienteFalso())


def test_contrato_de_observacao_identico_ao_ambiente_local(env):
    """
    O ponto que sustenta o experimento inteiro.

    Se o schema divergir, um agente treinado no ambiente local lê cada canal com
    o significado errado e produz números plausíveis e inválidos — a falha
    silenciosa que `hvac.features` existe para impedir.
    """
    cfg = ClassroomConfig()
    assert env.obs_schema == features.schema(cfg)
    low, high = features.limites(cfg)
    assert np.allclose(env.observation_space.low, low)
    assert np.allclose(env.observation_space.high, high)


def test_observacao_dentro_do_espaco_declarado(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), obs


def test_kelvin_para_celsius(env):
    """Erro de unidade aqui passaria despercebido e invalidaria todo conforto."""
    env.reset()
    env.cliente.temp_c = 25.0
    obs, _, _, _, info = env.step(0)
    # A física falsa soma 0,4 °C sem atuação.
    assert info["temperature"] == pytest.approx(25.4, abs=1e-6)


def test_acao_discreta_vira_fracao_de_carga(env):
    """As quatro ações devem chegar ao emulador como as frações do artigo."""
    env.reset()
    esperado = list(env.config.physics.discrete_levels())
    for acao, fracao in enumerate(esperado):
        env.step(acao)
        enviado = env.cliente.recebidas[-1]
        assert enviado["fcu_oveFan_u"] == pytest.approx(fracao)
        assert enviado["fcu_oveFan_activate"] == 1
        # Insuflamento fixo é o que torna o ventilador a única variável.
        assert enviado["fcu_oveTSup_activate"] == 1
        assert enviado["fcu_oveTSup_u"] == pytest.approx(env.bop.t_insuflamento_k)


def test_comutacao_detectada_apenas_na_troca(env):
    env.reset()
    _, _, _, _, i1 = env.step(3)
    _, _, _, _, i2 = env.step(3)
    _, _, _, _, i3 = env.step(0)
    assert i1["switched"] is True      # partiu de OFF
    assert i2["switched"] is False
    assert i3["switched"] is True


def test_controladores_classicos_operam_pela_temperatura(env):
    """
    Termostato e PI leem a temperatura do `info`, não da observação normalizada.

    É o que permite usá-los sem adaptação entre os dois ambientes.
    """
    r = rodar_episodio(env, ThermostatAgent(env.config, deadband=1.0))
    assert len(r["df"]) == env.bop.passos
    assert r["df"]["level"].nunique() > 1
    assert set(["comfort_wide_pct", "abs_dev_from_ideal"]) <= set(r["metricas"])
    assert r["kpis"]["tdis_tot"] == 1.23


def test_pi_reduz_temperatura_partindo_de_sala_quente():
    env = BoptestClassroomEnv(config=ClassroomConfig(),
                              bop=ConfigBoptest(passos=40),
                              cliente=ClienteFalso(t0_c=32.0))
    r = rodar_episodio(env, PIController(env.config, kp=1.3, ki=0.2))
    temps = r["df"]["temperature"]
    assert temps.iloc[-1] < temps.iloc[0], "o PI deveria resfriar a sala"


def test_horas_do_episodio_batem_com_o_protocolo(env):
    """
    `inner_steps` converte o passo da ponte para o passo do artigo.

    Sem isso, "comutações por hora" teria escala diferente entre os ambientes e a
    comparação entre as duas tabelas seria inválida.
    """
    r = rodar_episodio(env, ThermostatAgent(env.config))
    horas = r["df"]["inner_steps"].sum() * env.config.dt
    assert horas == pytest.approx(env.bop.passos
                                  * env.bop.intervalo_controle_s / 3600.0)


def test_energia_acumulada_usa_o_intervalo_de_controle(env):
    env.reset()
    _, _, _, _, info = env.step(3)                      # carga plena
    esperado_w = 3000.0 + 200.0
    assert info["power_w"] == pytest.approx(esperado_w)


def test_encerrar_libera_o_teste(env):
    env.reset()
    env.close()
    assert env.cliente.encerrado


def test_payload_vazio_vira_erro_explicito(env):
    """Teste expirado devolve vazio; falhar alto é melhor que métrica inválida."""
    from hvac.boptest.client import BoptestError
    env.reset()
    env.cliente.avancar = lambda acao=None: {}
    with pytest.raises(BoptestError):
        env.step(0)


# --------------------------------------------------------------- integração

def _servico_no_ar(url: str) -> bool:
    try:
        BoptestClient(url=url, timeout=5, tentativas=1).verificar_versao()
        return True
    except Exception:
        return False


URL = os.environ.get("BOPTEST_URL", "http://127.0.0.1:8000")


@pytest.mark.skipif(not _servico_no_ar(URL),
                    reason=f"serviço BOPTEST indisponível em {URL}")
def test_boptest_integracao():
    """Laço curto contra o serviço real, quando ele estiver de pé."""
    cliente = BoptestClient(url=URL)
    env = BoptestClassroomEnv(bop=ConfigBoptest(passos=6), cliente=cliente)
    try:
        r = rodar_episodio(env, PIController(env.config, kp=1.3, ki=0.2),
                           opcoes={"periodo": "peak_cool_day"})
        assert len(r["df"]) == 6
        assert 0.0 < r["df"]["temperature"].mean() < 60.0
        assert "tdis_tot" in r["kpis"]
    finally:
        env.close()
