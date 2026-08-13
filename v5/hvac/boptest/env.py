# -*- coding: utf-8 -*-
"""
Ambiente Gymnasium sobre um emulador do BOPTEST.

O QUE ESTE MÓDULO EXISTE PARA PERMITIR
--------------------------------------
A limitação declarada como a mais séria do artigo é que todos os resultados
vivem dentro de um simulador de autoria própria, que por construção não exibe
descasamento de modelo — precisamente o regime em que se espera que o
aprendizado por reforço tenha vantagem. Este ambiente executa os MESMOS
controladores contra um emulador de terceiros, revisado por pares e mantido pelo
IBPSA Project 1, cujo modelo ninguém aqui escreveu.

A DECISÃO DE PROJETO CENTRAL
----------------------------
A observação NÃO é redeclarada aqui. Ela é montada por `hvac.features.vetor`,
a mesma função que alimenta o ambiente local, o que exige apenas que esta classe
exponha os atributos que os extratores leem (`measured_temp`, `occupancy`,
`hour_float`, ...). A consequência é o que torna o experimento possível: um
agente treinado no ambiente local pode ser avaliado aqui SEM RETREINO, e a
verificação de schema em `model_io` garante que cada canal chega à política com
o significado com que foi treinado. Redeclarar a observação neste arquivo
reintroduziria exatamente a falha silenciosa que `features.py` foi escrito para
tornar impossível.

MAPEAMENTO DO ATUADOR
---------------------
O caso `bestest_air` expõe o ventilador do fancoil como sinal normalizado pela
vazão de projeto, `fcu_oveFan_u` em [0, 1], e a temperatura de insuflamento,
`fcu_oveTSup_u`. Com a insuflação fixa em valor frio, o ventilador passa a ser
uma fração de carga de resfriamento — a mesma grandeza que os quatro níveis
discretos do ambiente local, cujas frações são 0,00, 0,25, 0,55 e 1,00. O
mapeamento é, portanto, direto e não introduz reinterpretação da ação.

O QUE NÃO TRANSFERE, E ESTÁ DECLARADO
-------------------------------------
1. O emulador não expõe contagem de ocupantes; o canal de ocupação é NOMINAL,
   derivado da janela ocupada da config, e não medido. `fonte_ocupacao="co2"`
   troca-o pelo CO2 medido, normalizado, o que é uma leitura fisicamente
   disponível mas de semântica distinta da do treino.
2. O equipamento do emulador não tem a curva de COP com pico em carga parcial
   que o ambiente local modela. Conclusões sobre uso do nível de melhor COP não
   transferem, e não devem ser lidas aqui.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                    # pragma: no cover
    import gym
    from gym import spaces

from .. import features
from ..config import ClassroomConfig
from .client import BoptestClient, BoptestError

ZERO_C = 273.15

# Pontos do caso `bestest_air`. Reunidos em um único lugar porque trocar de caso
# é trocar este bloco, e não caçar strings pelo arquivo.
PONTO_TEMP = "zon_reaTRooAir_y"
PONTO_CO2 = "zon_reaCO2RooAir_y"
PONTO_EXTERNA = "zon_weaSta_reaWeaTDryBul_y"
PONTOS_POTENCIA = ("fcu_reaPCoo_y", "fcu_reaPFan_y", "fcu_reaPHea_y")
# Nomes-BASE dos pontos de sobrescrito: a API espera o sufixo "_u" para o
# valor e "_activate" para a habilitação.
ENTRADA_VENT = "fcu_oveFan"
ENTRADA_TSUP = "fcu_oveTSup"


@dataclass
class ConfigBoptest:
    """Parâmetros da ponte, separados da config do problema."""

    caso: str = "bestest_air"
    periodo: str = "peak_cool_day"
    preco: str = "constant"
    # 12 min = passo de 6 min com `action repeat` 2, o horizonte de decisão do
    # protocolo do artigo. Igualá-lo é condição para a comparação isolar a
    # política, e não o intervalo de amostragem.
    intervalo_controle_s: float = 720.0
    # 285,15 K = 12 °C é o mínimo admitido pelo ponto de insuflamento. Fixá-lo
    # no mínimo torna o ventilador a única variável de controle.
    t_insuflamento_k: float = 285.15
    warmup_s: float = 24 * 3600.0
    passos: int = 120                       # 120 × 12 min = 24 h, um episódio
    fonte_ocupacao: str = "nominal"         # "nominal" | "co2"
    co2_base_ppm: float = 400.0
    co2_saturacao_ppm: float = 1000.0


class BoptestClassroomEnv(gym.Env):
    """
    Emulador do BOPTEST exposto com o contrato de observação do ambiente local.

    O cliente é injetável para que os testes exercitem o laço completo sem
    exigir o serviço em execução — um ambiente que só pode ser testado com
    Docker de pé não seria testado.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[ClassroomConfig] = None,
                 bop: Optional[ConfigBoptest] = None,
                 cliente: Optional[BoptestClient] = None):
        super().__init__()
        self.config = config or ClassroomConfig()
        self.bop = bop or ConfigBoptest()
        self.cliente = cliente or BoptestClient()

        self.levels = self.config.physics.discrete_levels()
        if self.config.continuous_action:
            lo = -1.0 if self.config.heating_enabled else 0.0
            self.action_space = spaces.Box(low=lo, high=1.0, shape=(1,),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.levels))

        low, high = features.limites(self.config)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._init_state()

    # O mesmo contrato do ambiente local: é ele que `model_io` confere.
    @property
    def obs_schema(self) -> Dict:
        return features.schema(self.config)

    # ------------------------------------------------------------------ estado

    def _init_state(self) -> None:
        cfg = self.config
        self.measured_temp: float = cfg.ideal_temp
        self.current_temp: float = cfg.ideal_temp
        self.prev_temp: float = cfg.ideal_temp
        self.outdoor_temp: float = cfg.ideal_temp
        self.occupancy: int = 0
        self.hour_float: float = 0.0
        self.integral_error: float = 0.0
        self.demand = None
        self.time_step: int = 0
        self.tempo_s: float = 0.0
        self.carga_atual: float = 0.0
        self.nivel_atual: int = 0
        self.history: List[Dict] = []

    def minutes_to_peak(self) -> float:
        """Minutos até o próximo posto de ponta, pela tarifa da config."""
        cfg = self.config
        inicio = float(getattr(cfg.tariff, "peak_start_hour", 18.0))
        delta = (inicio - self.hour_float) % 24.0
        return delta * 60.0

    # ------------------------------------------------------------- observação

    def _fracao_ocupacao(self) -> float:
        """
        Canal de ocupação.

        O emulador não publica contagem de ocupantes. Em `nominal` reproduz-se a
        janela ocupada da config, que é a semântica com que a política foi
        treinada; em `co2` usa-se a concentração medida, que é observável no
        mundo real mas cuja escala não é a do treino. A escolha é do experimento,
        e o resultado deve declarar qual foi usada.
        """
        cfg = self.config
        if self.bop.fonte_ocupacao == "co2":
            faixa = max(self.bop.co2_saturacao_ppm - self.bop.co2_base_ppm, 1e-9)
            return float(np.clip((self._co2_ppm - self.bop.co2_base_ppm) / faixa,
                                 0.0, 1.0))
        ini = float(getattr(cfg, "occupied_start_hour", 7.0))
        fim = float(getattr(cfg, "occupied_end_hour", 22.0))
        return 1.0 if ini <= self.hour_float < fim else 0.0

    def _get_obs(self) -> np.ndarray:
        self.occupancy = int(round(self._fracao_ocupacao()
                                   * self.config.max_occupancy))
        return features.vetor(self)

    def _info(self) -> Dict:
        """`temperature` é o que os controladores clássicos leem."""
        return {"temperature": self.measured_temp,
                "true_temperature": self.current_temp,
                "outdoor_temperature": self.outdoor_temp,
                "co2_ppm": self._co2_ppm,
                "hour": self.hour_float,
                "occupancy": self.occupancy,
                "power_w": self._potencia_w,
                "level": self.nivel_atual,
                "load": self.carga_atual,
                "time_s": self.tempo_s}

    # ---------------------------------------------------------- leitura do emulador

    def _absorver(self, y: Dict) -> None:
        """Traduz uma resposta do emulador para o estado deste ambiente."""
        if not y:
            raise BoptestError("emulador devolveu payload vazio; teste expirou?")
        self.prev_temp = self.measured_temp
        temp_k = float(y.get(PONTO_TEMP, ZERO_C + self.config.ideal_temp))
        self.measured_temp = temp_k - ZERO_C
        # O emulador não distingue verdade de leitura: não há ruído de sensor
        # modelado aqui, e igualar as duas é mais honesto que inventar um.
        self.current_temp = self.measured_temp
        self.outdoor_temp = float(y.get(PONTO_EXTERNA, ZERO_C)) - ZERO_C
        self._co2_ppm = float(y.get(PONTO_CO2, self.bop.co2_base_ppm))
        self._potencia_w = sum(float(y.get(p, 0.0)) for p in PONTOS_POTENCIA)
        self.tempo_s = float(y.get("time", self.tempo_s))
        self.hour_float = (self.tempo_s / 3600.0) % 24.0

        erro = self.measured_temp - self.config.ideal_temp
        self.integral_error = float(np.clip(
            self.integral_error + erro * (self.bop.intervalo_controle_s / 3600.0),
            -self.config.integral_clip_degree_hours,
            self.config.integral_clip_degree_hours))

    # ------------------------------------------------------------------ ciclo

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        options = options or {}
        self._init_state()
        self._co2_ppm = self.bop.co2_base_ppm
        self._potencia_w = 0.0

        if self.cliente.testid is None:
            self.cliente.selecionar(options.get("caso", self.bop.caso))

        self.cliente.cenario(periodo=options.get("periodo", self.bop.periodo),
                             preco=options.get("preco", self.bop.preco))
        # `scenario` já posiciona o tempo inicial do período; `initialize` é
        # chamado apenas quando o experimento pede um instante específico.
        if "inicio_s" in options:
            self.cliente.inicializar(
                inicio_s=float(options["inicio_s"]),
                warmup_s=float(options.get("warmup_s", self.bop.warmup_s)))
        self.cliente.passo(self.bop.intervalo_controle_s)

        # Um avanço sem sobrescrito estabelece o estado inicial observável sem
        # que o controlador tenha agido — o análogo do `reset` local.
        self._absorver(self.cliente.avancar({}))
        return self._get_obs(), self._info()

    def _carga_de(self, action) -> float:
        if isinstance(self.action_space, spaces.Box):
            return float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        idx = int(np.asarray(action).reshape(-1)[0])
        return float(self.levels[idx])

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        carga = self._carga_de(action)
        nivel = int(np.argmin([abs(carga - f) for f in self.levels]))

        y = self.cliente.avancar({
            ENTRADA_VENT + "_u": carga,
            ENTRADA_VENT + "_activate": 1,
            ENTRADA_TSUP + "_u": self.bop.t_insuflamento_k,
            ENTRADA_TSUP + "_activate": 1,
        })
        trocou = nivel != self.nivel_atual
        self.carga_atual, self.nivel_atual = carga, nivel
        self._absorver(y)
        self.time_step += 1

        obs = self._get_obs()
        info = self._info()
        info["switched"] = trocou
        self.history.append(dict(info))

        # A recompensa não governa nada nesta avaliação — os agentes são
        # executados em modo determinístico e o julgamento é feito pelos KPIs do
        # framework. Devolve-se conforto simples apenas para satisfazer a
        # interface do Gymnasium.
        cfg = self.config
        dentro = cfg.temp_comfort_min <= self.measured_temp <= cfg.temp_comfort_max
        recompensa = float(dentro) - 0.01 * self._potencia_w / 1000.0

        terminado = False
        truncado = self.time_step >= self.bop.passos
        return obs, recompensa, terminado, truncado, info

    def kpis(self) -> Dict:
        return self.cliente.kpis()

    def close(self) -> None:
        self.cliente.encerrar()
