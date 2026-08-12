# -*- coding: utf-8 -*-
"""
Ambiente multi-zona com capacidade COMPARTILHADA.

POR QUE ESTA E A MUDANCA ESTRUTURAL MAIS FORTE
----------------------------------------------
Todas as tentativas anteriores falharam em favorecer RL porque um PI continuava
sendo a ferramenta certa: rastrear setpoint rejeitando perturbacao e exatamente
o que realimentacao linear faz bem. Multi-zona com capacidade compartilhada
quebra isso de forma que nao ha conserto local:

1. NAO HA LEI LOCAL OTIMA. Quando a demanda somada das zonas excede a capacidade
   do compressor, alguem precisa ser sacrificado. Um PI por zona nao tem
   mecanismo para decidir QUEM: cada um pede o que precisa, e o rateio que sobra
   e proporcional -- que nao e otimo. O certo e priorizar a zona mais proxima de
   violar, ou a de maior ocupacao, ou a que tem menos inercia. Isso e uma decisao
   de ALOCACAO, e alocacao nao e expressavel como realimentacao local.

2. ACOPLAMENTO ENTRE ZONAS. Paredes internas trocam calor: resfriar a zona A
   ajuda a zona B. Um PI por zona ignora isso e briga contra o vizinho. A politica
   otima explora o acoplamento.

3. ESPACO DE ACAO COMBINATORIO. Com N zonas a acao e um vetor de alocacao no
   simplex, nao um escalar. Sintonizar N PIs independentes nao produz coordenacao.

E a lacuna que o Revisor 1 nomeou explicitamente: "a literatura aborda a
interacao entre salas, multiplas fontes de resfriamento, variacao de ocupacao".

PRECEDENTE NA LITERATURA — Wei, Wang e Zhu (DAC 2017)
----------------------------------------------------
O primeiro trabalho a aplicar DRL a HVAC formula exatamente este problema e
enuncia a dificuldade central: com z zonas e m niveis de atuacao por zona, o
espaco de acao tem m^z elementos, e "the dimension of action space will increase
rapidly with larger number of zones and air flow rate levels, which will then
greatly increase the training time and degrade the control performance".

Duas consequencias para este modulo:

1. A acao combinatoria DISCRETA e a formulacao canonica, e esta implementada em
   `MultiZoneEnv` com `acao_discreta=True`. `contagem_acoes()` expoe m^z para
   que a explosao seja um numero no relatorio, e nao uma afirmacao.
2. Wei et al. propoem uma heuristica de controle MULTINIVEL para contornar a
   explosao. `MultiNivelWrapper` implementa o PRINCIPIO descrito por eles —
   decompor a decisao em "quanta capacidade usar" e "quem tem prioridade" —
   reduzindo o espaco de m^z para m + z. Nao e reproducao literal: o artigo
   descreve a heuristica em figura, e o que se afirma aqui e a adaptacao do
   principio, nao a identidade com o metodo original.

RESSALVA DE BASELINE, herdada do achado nº 1 desta auditoria: Wei et al. comparam
contra rule-based e reportam 20-70% de reducao de custo. Repetir esse baseline
aqui reproduziria, num problema mais complexo, exatamente o defeito que este
trabalho acusa. Os adversarios validos sao `PriorityAllocator` e, idealmente, MPC.

RESSALVA DE HONESTIDADE
-----------------------
O adversario certo aqui DEIXA de ser o PI e passa a ser MPC: com o modelo
conhecido, MPC resolve alocacao por otimizacao explicita. O nicho do RL e onde o
modelo e ruim ou a dimensionalidade cresce demais para otimizar online. Comparar
multi-zona so contra PI reproduziria, num problema mais complexo, o mesmo erro
de baseline que os revisores apontaram no manuscrito.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ClassroomConfig


@dataclass
class ZoneSpec:
    """Uma zona: fisica propria, ocupacao propria, setpoint proprio."""
    name: str
    thermal_mass: float = 15.0
    heat_transfer_coeff: float = 0.5
    max_occupancy: int = 45
    ideal_temp: float = 24.0
    tolerance: float = 0.5


@dataclass
class MultiZoneConfig:
    """
    Configuracao do predio.

    `capacity_fraction` abaixo de 1,0 e o que cria o problema: se houvesse
    capacidade para todas as zonas simultaneamente, cada uma seria independente e
    N PIs resolveriam. A escassez e que exige alocacao.
    """
    zones: Tuple[ZoneSpec, ...] = field(default_factory=lambda: (
        ZoneSpec("A", thermal_mass=12.0, max_occupancy=45),   # leve, cheia
        ZoneSpec("B", thermal_mass=18.0, max_occupancy=30),   # pesada, media
        ZoneSpec("C", thermal_mass=15.0, max_occupancy=20),   # media, pequena
    ))
    # Fracao da capacidade necessaria para atender TODAS as zonas no pico.
    capacity_fraction: float = 0.65
    # Conducao pelas paredes internas (u por grau de diferenca entre zonas).
    inter_zone_coupling: float = 0.8
    base: ClassroomConfig = field(default_factory=ClassroomConfig)

    def peak_load_units(self) -> float:
        """Carga termica somada no pico, para dimensionar a capacidade."""
        b = self.base
        externo = b.heat_transfer_coeff * (
            (b.outdoor_base_temp + b.outdoor_amplitude) - b.ideal_temp)
        return sum(z.max_occupancy * b.heat_gain_per_person + externo
                   for z in self.zones)

    def total_capacity_units(self) -> float:
        return self.capacity_fraction * self.peak_load_units()


class MultiZoneEnv(gym.Env):
    """
    N zonas, um compressor. A acao aloca a capacidade entre as zonas.

    Acao: vetor em [0,1]^N. E NORMALIZADO para o simplex escalado pela capacidade
    quando a soma excede o disponivel -- ou seja, o agente pede e o hardware
    limita, que e o comportamento fisico. Pedir menos que a capacidade e
    permitido (economiza energia).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[MultiZoneConfig] = None,
                 acao_discreta: bool = False,
                 niveis: Sequence[float] = (0.0, 0.34, 0.67, 1.0)):
        super().__init__()
        self.cfg = config or MultiZoneConfig()
        self.n = len(self.cfg.zones)
        b = self.cfg.base

        # Acao contínua (simplex) ou DISCRETA COMBINATORIA (formulacao de Wei et
        # al. 2017). A discreta e a que expoe a explosao m^z.
        self.niveis = tuple(niveis)
        self.acao_discreta = bool(acao_discreta)
        if self.acao_discreta:
            self.action_space = spaces.Discrete(self.contagem_acoes())
        else:
            self.action_space = spaces.Box(0.0, 1.0, shape=(self.n,), dtype=np.float32)
        # Por zona: erro escalado, ocupacao, derivada. Global: sin/cos hora, folga.
        dim = 3 * self.n + 3
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)

        self.capacity = self.cfg.total_capacity_units()
        self._init_state()

    def contagem_acoes(self) -> int:
        """
        m^z — o tamanho do espaco de acao combinatorio.

        Existe como metodo para que a explosao apontada por Wei et al. seja
        verificavel: com 3 zonas e 4 niveis sao 64 acoes; com 5 zonas, 1024.
        """
        return int(len(self.niveis) ** self.n)

    def _decodifica(self, indice: int) -> np.ndarray:
        """Indice do espaco combinatorio -> vetor de fracoes por zona."""
        vals, resto = [], int(indice)
        for _ in range(self.n):
            vals.append(self.niveis[resto % len(self.niveis)])
            resto //= len(self.niveis)
        return np.array(vals, dtype=float)

    def _init_state(self) -> None:
        b = self.cfg.base
        self.temps = np.full(self.n, b.ideal_temp, dtype=float)
        self.prev_temps = self.temps.copy()
        self.occ = np.zeros(self.n, dtype=int)
        self.episodes: List = []
        self.time_step = 0
        self.start_hour = 0
        self.hour_float = 0.0

    # ------------------------------------------------------------------ obs

    def _get_obs(self) -> np.ndarray:
        b = self.cfg.base
        out: List[float] = []
        for i, z in enumerate(self.cfg.zones):
            escala = max(4.0 * z.tolerance, 1e-9)
            out.append(float(np.clip((self.temps[i] - z.ideal_temp) / escala, -1, 1)))
            out.append(float(np.clip(self.occ[i] / max(z.max_occupancy, 1), 0, 1)))
            max_d = (self.capacity / z.thermal_mass) * b.dt
            out.append(float(np.clip((self.temps[i] - self.prev_temps[i]) / max(max_d, 1e-9), -1, 1)))
        ang = 2 * np.pi * self.hour_float / 24.0
        out += [float(np.sin(ang)), float(np.cos(ang))]
        # Folga: quanto da capacidade sobraria se todas as zonas fossem atendidas.
        out.append(float(np.clip(1.0 - self._demanda_termica() / max(self.capacity, 1e-9), -1, 1)))
        return np.array(out, dtype=np.float32)

    def _demanda_termica(self) -> float:
        """Capacidade que seria necessaria para segurar TODAS as zonas agora."""
        b = self.cfg.base
        ext = self._outdoor()
        total = 0.0
        for i, z in enumerate(self.cfg.zones):
            total += max(0.0, self.occ[i] * b.heat_gain_per_person
                         + z.heat_transfer_coeff * (ext - self.temps[i]))
        return total

    def _outdoor(self) -> float:
        b = self.cfg.base
        fase = (self.hour_float % 24.0 - b.outdoor_peak_hour) * np.pi / 12.0
        return float(b.outdoor_base_temp + b.outdoor_amplitude * np.cos(fase))

    # ---------------------------------------------------------------- reset

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._init_state()
        b = self.cfg.base
        from .occupancy import OccupancyModel

        self.start_hour = int(self.np_random.integers(0, 24))
        if options:
            self.start_hour = int(options.get("hour", self.start_hour))
        self.hour_float = float(self.start_hour)

        temp0 = float(options.get("start_temp", b.ideal_temp)) if options else b.ideal_temp
        self.temps[:] = temp0
        self.prev_temps[:] = temp0

        picos = options.get("occupancies") if options else None
        self.episodes = []
        for i, z in enumerate(self.cfg.zones):
            modelo = OccupancyModel(max_occupancy=z.max_occupancy,
                                    tau_minutes=b.occupancy_tau_minutes,
                                    churn_per_hour=b.occupancy_churn_per_hour)
            pico = (int(picos[i]) if picos is not None
                    else int(self.np_random.integers(5, z.max_occupancy + 1)))
            ep = modelo.sample_episode(self.np_random, pico)
            self.occ[i] = ep.reset(self.hour_float)
            self.episodes.append(ep)

        return self._get_obs(), self._info(np.zeros(self.n))

    # ----------------------------------------------------------------- step

    def _alocar(self, pedido: np.ndarray) -> np.ndarray:
        """
        Converte pedido em alocacao efetiva respeitando a capacidade total.

        Rateio PROPORCIONAL quando o pedido excede a capacidade — e o que o
        hardware faz. Note que proporcional NAO e otimo: a decisao de quem
        sacrificar deveria depender de quem esta mais proximo de violar, e essa
        escolha e do agente, feita ANTES, ao dimensionar o proprio pedido.
        """
        pedido = np.clip(np.asarray(pedido, dtype=float).reshape(-1), 0.0, 1.0)
        unidades = pedido * self.capacity
        total = unidades.sum()
        if total > self.capacity and total > 0:
            unidades *= self.capacity / total
        return unidades

    def step(self, action):
        b = self.cfg.base
        self.prev_temps = self.temps.copy()
        if self.acao_discreta:
            action = self._decodifica(int(action))
        resfr = self._alocar(action)
        ext = self._outdoor()

        novas = self.temps.copy()
        for i, z in enumerate(self.cfg.zones):
            ganho = (self.occ[i] * b.heat_gain_per_person
                     + z.heat_transfer_coeff * (ext - self.temps[i]))
            # Acoplamento entre zonas: cada vizinho troca calor pela parede.
            acopl = sum(self.cfg.inter_zone_coupling * (self.temps[j] - self.temps[i])
                        for j in range(self.n) if j != i)
            liquido = ganho + acopl - resfr[i]
            ruido = float(self.np_random.normal(0.0, b.temperature_noise_std))
            novas[i] = float(np.clip(
                self.temps[i] + (b.dt / z.thermal_mass) * liquido + ruido,
                b.temp_min_clip, b.temp_max_clip))
        self.temps = novas

        self.time_step += 1
        self.hour_float = self.start_hour + self.time_step * b.dt
        for i, ep in enumerate(self.episodes):
            self.occ[i] = ep.step(self.hour_float, b.dt, self.np_random)

        reward = self._reward(resfr)
        terminated = self.time_step >= b.episode_steps
        return self._get_obs(), reward, terminated, False, self._info(resfr)

    def _reward(self, resfr: np.ndarray) -> float:
        """
        Soma das zonas: conforto em faixa + custo de energia.

        Ponderado pela OCUPACAO: violar o conforto de uma sala com 45 pessoas e
        pior que numa sala com 5. E esta ponderacao que torna a alocacao uma
        decisao com resposta certa -- e que um rateio proporcional ignora.
        """
        b = self.cfg.base
        total = 0.0
        for i, z in enumerate(self.cfg.zones):
            d = abs(self.temps[i] - z.ideal_temp)
            peso = 0.3 + 0.7 * (self.occ[i] / max(z.max_occupancy, 1))
            conforto = 10.0 if d <= z.tolerance else 10.0 - 40.0 * (d - z.tolerance)
            total += peso * conforto
        kw = resfr.sum() / max(self.capacity, 1e-9) * 3.0     # proxy de potencia
        return total / self.n - 0.5 * kw

    def _info(self, resfr: np.ndarray) -> Dict:
        return {
            "temperatures": self.temps.copy(),
            "occupancies": self.occ.copy(),
            "allocation": resfr.copy(),
            "hour_float": float(self.hour_float),
            "hour": int(self.hour_float) % 24,
            "capacity": self.capacity,
            "thermal_demand": self._demanda_termica(),
            "saturated": bool(self._demanda_termica() > self.capacity),
            "step": self.time_step,
        }


class PerZonePI:
    """
    N PIs independentes — o baseline ingenuo, e o ponto do experimento.

    Cada zona pede o que precisa; o hardware rateia proporcionalmente quando nao
    cabe. Nenhum dos PIs sabe da existencia dos outros nem da restricao de
    capacidade, entao a decisao de quem sacrificar e tomada pelo rateio, nao por
    criterio.
    """

    def __init__(self, cfg: MultiZoneConfig, kp: float = 1.3, ki: float = 0.2):
        self.cfg, self.kp, self.ki = cfg, kp, ki
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(len(self.cfg.zones))

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        temps = info["temperatures"]
        saida = np.zeros(len(self.cfg.zones), dtype=np.float32)
        for i, z in enumerate(self.cfg.zones):
            erro = temps[i] - z.ideal_temp
            bruto = self.kp * erro + self.ki * self.integral[i]
            if 0.0 < bruto < 1.0:
                self.integral[i] += erro
            saida[i] = np.clip(self.kp * erro + self.ki * self.integral[i], 0.0, 1.0)
        return saida, None


class PriorityAllocator(PerZonePI):
    """
    Baseline FORTE: PI por zona mais arbitragem explicita por prioridade.

    Quando a capacidade nao cobre o pedido, aloca primeiro para a zona com maior
    (ocupacao x desvio) — que e o gradiente da recompensa. E o que um engenheiro
    de automacao competente implementaria, e o adversario honesto contra o qual o
    RL precisa ganhar. Comparar so contra PerZonePI reproduziria o erro de
    baseline fraco que os revisores apontaram.

    O que ele NAO faz: antecipar. Ele arbitra o deficit do instante; nao
    pre-resfria a zona pesada antes do pico sabendo que depois faltara capacidade.
    """

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        pedido, _ = super().predict(obs, info=info)
        cap = info["capacity"]
        unidades = np.asarray(pedido, dtype=float) * cap
        if unidades.sum() <= cap:
            return pedido.astype(np.float32), None

        temps, occ = info["temperatures"], info["occupancies"]
        prio = []
        for i, z in enumerate(self.cfg.zones):
            desvio = max(0.0, temps[i] - z.ideal_temp)
            prio.append((0.3 + 0.7 * occ[i] / max(z.max_occupancy, 1)) * (1 + desvio))
        ordem = np.argsort(prio)[::-1]

        restante, alocado = cap, np.zeros(len(self.cfg.zones))
        for i in ordem:
            dar = min(unidades[i], restante)
            alocado[i] = dar
            restante -= dar
        return (alocado / cap).astype(np.float32), None


class MultiNivelWrapper(gym.Wrapper):
    """
    Heuristica de controle MULTINIVEL — adaptacao do principio de Wei et al. (2017).

    O PROBLEMA. Com z zonas e m niveis por zona, a acao combinatoria tem m^z
    elementos: 64 para 3 zonas, 1024 para 5. Wei et al. registram que isso
    "will greatly increase the training time and degrade the control
    performance", e propoem decompor a decisao em niveis.

    A DECOMPOSICAO AQUI. Em vez de escolher a fracao de cada zona
    independentemente, o agente escolhe duas coisas:

      1. QUANTA capacidade total acionar (m niveis);
      2. QUAL zona tem prioridade no rateio (z opcoes).

    O espaco cai de m^z para m x z — de 64 para 12 com 3 zonas, de 1024 para 20
    com 5 zonas. E crescimento LINEAR em z, e nao exponencial.

    O QUE SE PERDE, e e preciso dizer: a decomposicao nao é sem custo. Ela nao
    consegue expressar toda alocacao possivel — por exemplo "zona A no maximo e
    zona B no minimo enquanto C fica no meio" nao tem representacao exata. A
    aposta, que e a mesma de Wei et al., e que as alocacoes uteis estao quase
    todas na familia "priorize quem precisa mais, com esta intensidade total".
    Se a politica otima estiver fora dessa familia, a heuristica impoe um teto.

    HONESTIDADE SOBRE A FONTE: o artigo descreve a heuristica em figura, que a
    extracao de texto nao recupera. O que se afirma aqui e a adaptacao do
    principio enunciado, nao identidade com o metodo original.
    """

    def __init__(self, env: "MultiZoneEnv", niveis_totais: int = 4):
        super().__init__(env)
        self.n = env.n
        self.niveis_totais = int(niveis_totais)
        self.action_space = spaces.Discrete(self.niveis_totais * self.n)

    def contagem_acoes(self) -> int:
        return int(self.niveis_totais * self.n)

    def _traduz(self, indice: int) -> np.ndarray:
        """(nivel total, zona prioritaria) -> vetor de fracoes por zona."""
        indice = int(indice)
        nivel = indice // self.n
        prioritaria = indice % self.n
        fracao_total = nivel / max(self.niveis_totais - 1, 1)

        env = self.env
        # Necessidade de cada zona: desvio acima do setpoint, ponderado pela
        # ocupacao — o mesmo criterio que a recompensa usa, para que a heuristica
        # esteja alinhada ao objetivo e nao lute contra ele.
        need = np.zeros(self.n)
        for i, z in enumerate(env.cfg.zones):
            desvio = max(0.0, env.temps[i] - z.ideal_temp)
            peso = 0.3 + 0.7 * (env.occ[i] / max(z.max_occupancy, 1))
            need[i] = peso * (0.1 + desvio)
        need[prioritaria] *= 3.0        # a escolha do agente entra aqui

        if need.sum() <= 0:
            return np.full(self.n, fracao_total / self.n)
        # A capacidade escolhida e repartida na proporcao da necessidade, de modo
        # que a soma das fracoes vale `fracao_total` — o nivel global e um teto
        # de consumo, e a prioridade decide quem fica com a maior parte dele.
        return fracao_total * need / need.sum()

    def step(self, action):
        return self.env.step(self._traduz(action))
