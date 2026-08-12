# -*- coding: utf-8 -*-
"""
Modelo de ocupação realista: pessoas entram e saem ao longo do dia.

PROBLEMA QUE ISTO RESOLVE
-------------------------
O ambiente tinha dois processos de ocupação incompatíveis:

  treino     passeio aleatório  -> média 22,7, mediana 23,  9,9 % nos extremos
  avaliação  janela ocupada     -> média 15,0, mediana  5, 58,3 % nos extremos
                                   (teste KS: D = 0,438, p ~ 0)

O agente treinava numa distribuição que praticamente não via no teste, e — pior —
quase nunca experimentava as transições em degrau (0->45 às 7h) que definiam
todos os cenários de avaliação. Isso invalida a avaliação: uma política testada
fora da distribuição em que treinou não tem desempenho interpretável.

Nenhum dos dois processos era realista, aliás. Sala de aula real não enche
instantaneamente nem tem ocupação em passeio aleatório: enche em ~15 min no início
da aula, esvazia no intervalo, cai no almoço, e tem entra-e-sai contínuo.

O MODELO
--------
Ocupação-alvo por bloco horário (aulas, intervalos, almoço, noturno), com a
ocupação real **relaxando** para o alvo com constante de tempo, mais *churn*
estocástico de entra-e-sai. Isso dá três propriedades que os processos anteriores
não tinham:

  1. transições GRADUAIS (pessoas levam tempo para entrar e sair);
  2. mesma estrutura no treino e na avaliação — só os parâmetros randomizam;
  3. perturbação contínua, que é o ponto: a temperatura não é totalmente
     controlada porque a carga térmica muda o tempo todo.

O estado interno é contínuo e só é arredondado na saída — arredondar a cada passo
mataria a dinâmica de relaxação para constantes de tempo curtas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# (hora_início, hora_fim, fração_da_lotação_de_pico)
# Grade típica de sala de aula/laboratório brasileiro, com matutino, vespertino e
# noturno. As frações são do PICO do cenário, não absolutas — assim o mesmo
# perfil serve para uma sala de 5 e uma de 45 pessoas.
GRADE_PADRAO: Tuple[Tuple[float, float, float], ...] = (
    (7.5, 9.5, 0.90),     # primeira aula
    (9.5, 9.83, 0.40),    # intervalo curto
    (9.83, 11.83, 1.00),  # segunda aula — lotação máxima
    (11.83, 13.33, 0.10), # almoço: quase todos saem
    (13.33, 15.33, 0.85), # aula da tarde
    (15.33, 15.67, 0.35), # intervalo
    (15.67, 17.67, 0.70), # fim de tarde, turma menor
    (18.5, 22.0, 0.55),   # noturno
)


@dataclass
class OccupancyModel:
    """
    Ocupação por relaxação para alvo horário, com churn.

    `tau_minutes` é o tempo característico de enchimento/esvaziamento: ~15 min é
    o que uma turma leva para entrar. `churn_per_hour` é o fluxo aleatório de
    entra-e-sai dentro do bloco (banheiro, chegadas atrasadas), em pessoas/hora.
    """

    max_occupancy: int = 45
    grade: Tuple[Tuple[float, float, float], ...] = GRADE_PADRAO
    tau_minutes: float = 15.0
    churn_per_hour: float = 6.0

    # Randomização por episódio (domain randomization sobre a AGENDA).
    jitter_hours: float = 0.75      # desloca as bordas dos blocos
    jitter_fraction: float = 0.15   # varia a fração de cada bloco
    jitter_tau: float = 5.0         # varia a velocidade de enchimento

    def sample_episode(self, rng: np.random.Generator, peak: int) -> "OccupancyEpisode":
        """
        Sorteia uma realização da agenda para um episódio.

        A ESTRUTURA (encher, intervalo, almoço, esvaziar) é sempre a mesma; o que
        varia são as bordas, as frações e a velocidade. É isto que faz o treino
        cobrir a mesma família de perturbações da avaliação sem decorá-la.
        """
        desloc = float(rng.uniform(-self.jitter_hours, self.jitter_hours))
        blocos: List[Tuple[float, float, float]] = []
        for ini, fim, frac in self.grade:
            f = float(np.clip(frac * (1.0 + rng.uniform(-self.jitter_fraction,
                                                        self.jitter_fraction)),
                              0.0, 1.0))
            blocos.append((ini + desloc, fim + desloc, f))
        tau = max(2.0, self.tau_minutes + float(rng.uniform(-self.jitter_tau,
                                                            self.jitter_tau)))
        return OccupancyEpisode(model=self, blocos=tuple(blocos),
                                peak=int(np.clip(peak, 0, self.max_occupancy)),
                                tau_minutes=tau)


@dataclass
class OccupancyEpisode:
    """Realização da agenda válida por um episódio."""

    model: OccupancyModel
    blocos: Tuple[Tuple[float, float, float], ...]
    peak: int
    tau_minutes: float
    _cont: float = field(default=0.0, init=False)

    def target(self, hour: float) -> float:
        """Ocupação-alvo na hora decimal (0 fora de todos os blocos)."""
        h = hour % 24.0
        for ini, fim, frac in self.blocos:
            a, b = ini % 24.0, fim % 24.0
            dentro = (a <= h < b) if a <= b else (h >= a or h < b)
            if dentro:
                return self.peak * frac
        return 0.0

    def reset(self, hour: float) -> int:
        """Inicializa no alvo da hora de início (sala já no regime do bloco)."""
        self._cont = self.target(hour)
        return int(round(self._cont))

    def step(self, hour: float, dt_hours: float, rng: np.random.Generator) -> int:
        """
        Avança um passo: relaxa para o alvo e aplica churn.

        Relaxação exponencial discreta: alpha = 1 - exp(-dt/tau). Usar
        `alpha = dt/tau` seria instável para dt comparável a tau; a forma
        exponencial é estável para qualquer dt.
        """
        dt_min = dt_hours * 60.0
        alpha = 1.0 - float(np.exp(-dt_min / max(self.tau_minutes, 1e-9)))
        alvo = self.target(hour)
        self._cont += alpha * (alvo - self._cont)

        # Churn só faz sentido com gente na sala, e não deve empurrar abaixo de
        # zero nem acima do alvo de forma sistemática.
        if self._cont > 0.5:
            sigma = self.model.churn_per_hour * np.sqrt(max(dt_hours, 0.0))
            self._cont += float(rng.normal(0.0, sigma))

        self._cont = float(np.clip(self._cont, 0.0, self.model.max_occupancy))
        return int(round(self._cont))


def describe_profile(ep: OccupancyEpisode, dt_hours: float = 0.1,
                     seed: int = 0) -> np.ndarray:
    """Perfil de 24 h para inspeção/plotagem."""
    rng = np.random.default_rng(seed)
    ep.reset(0.0)
    n = int(round(24.0 / dt_hours))
    out = np.zeros(n, dtype=int)
    for k in range(n):
        out[k] = ep.step(k * dt_hours, dt_hours, rng)
    return out
