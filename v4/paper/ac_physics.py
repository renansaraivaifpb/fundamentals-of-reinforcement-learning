# -*- coding: utf-8 -*-
"""
Modelo físico do ar-condicionado (Tabela 1 do paper).

Decisão de projeto: em vez de hardcodar a Tabela 1, ela é DERIVADA de dois
parâmetros de catálogo (capacidade em BTU/h e COP por nível de carga). Isso
torna o modelo auditável e permite trocar de equipamento sem reescrever a
tabela — a única forma honesta de sustentar a afirmação do paper de que a
penalidade de energia reflete "eficiência fisicamente coerente".

Verificado contra a Tabela 1 (split ~30k BTU):
    LOW    0,25 · 8,7921 / 3,45 = 0,637 kW  (paper 0,64)
    MEDIUM 0,55 · 8,7921 / 3,59 = 1,347 kW  (paper 1,35)
    HIGH   1,00 · 8,7921 / 3,00 = 2,931 kW  (paper 2,93)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

# 1 BTU/h = 0,000293071 kW
BTU_PER_HOUR_TO_KW = 0.000293071


class ACState(IntEnum):
    """Níveis discretos do compressor. IntEnum para indexar direto pela ação."""
    OFF = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class ACPhysicsModel:
    """
    Converte nível de operação -> (potência de resfriamento em unidades de
    simulação, potência elétrica em kW).

    O COP tem pico em carga parcial (MEDIUM), comportamento realista de
    equipamento *inverter*. Isso é o que cria o trade-off interessante: o
    termostato, ao operar em MEDIUM, é energeticamente barato — e é por isso
    que no paper ele ganha em custo apesar de perder em conforto.
    """

    capacity_btu_per_hour: float = 30_000.0

    # Fração da capacidade nominal acionada em cada nível.
    load_fraction: Dict[ACState, float] = field(default_factory=lambda: {
        ACState.OFF: 0.00,
        ACState.LOW: 0.25,
        ACState.MEDIUM: 0.55,
        ACState.HIGH: 1.00,
    })

    # Coeficiente de performance por nível (pico em carga parcial).
    cop: Dict[ACState, float] = field(default_factory=lambda: {
        ACState.LOW: 3.45,
        ACState.MEDIUM: 3.59,
        ACState.HIGH: 3.00,
    })

    # Escala que mapeia carga nominal (fração 1,0) -> unidades de simulação.
    # Fixada em 40 para reproduzir a linha "Potência de resfriamento (u)".
    cooling_units_at_full_load: float = 40.0

    @property
    def capacity_kw_thermal(self) -> float:
        """Capacidade de refrigeração nominal em kW térmicos."""
        return self.capacity_btu_per_hour * BTU_PER_HOUR_TO_KW

    def cooling_units(self, state: ACState) -> float:
        """Potência de resfriamento nas unidades usadas no balanço térmico."""
        return self.cooling_units_at_full_load * self.load_fraction[state]

    def electrical_kw(self, state: ACState) -> float:
        """
        Potência elétrica consumida (kW). É esta grandeza — e não a de
        resfriamento — que alimenta a penalidade de energia e o custo, de modo
        que o agente otimiza a conta de luz real, não uma proxy.
        """
        if state == ACState.OFF:
            return 0.0
        thermal_kw = self.load_fraction[state] * self.capacity_kw_thermal
        return thermal_kw / self.cop[state]

    # --- Interface contínua, usada pelo controlador SAC (Seção 4.5) ---

    def cooling_units_continuous(self, load: float) -> float:
        """Resfriamento para uma fração de potência contínua a ∈ [0, 1]."""
        return self.cooling_units_at_full_load * max(0.0, min(1.0, load))

    def electrical_kw_continuous(self, load: float) -> float:
        """
        Potência elétrica para carga contínua, com COP interpolado
        linearmente entre os pontos tabelados. Sem isso, o SAC seria avaliado
        sob um modelo de eficiência diferente do DQN e a comparação da
        Tabela 5 não seria justa.
        """
        load = max(0.0, min(1.0, load))
        if load <= 0.0:
            return 0.0

        points = sorted(
            (self.load_fraction[s], self.cop[s]) for s in self.cop
        )
        if load <= points[0][0]:
            cop = points[0][1]
        elif load >= points[-1][0]:
            cop = points[-1][1]
        else:
            cop = points[-1][1]
            for (x0, c0), (x1, c1) in zip(points, points[1:]):
                if x0 <= load <= x1:
                    w = (load - x0) / (x1 - x0)
                    cop = c0 + w * (c1 - c0)
                    break
        return load * self.capacity_kw_thermal / cop

    def as_table(self) -> str:
        """Reproduz a Tabela 1 para conferência/auditoria."""
        rows = [f"{'Nível':<8}{'Carga':>8}{'Resfr.(u)':>11}{'Elétr.(kW)':>12}{'COP':>7}"]
        for s in ACState:
            cop = self.cop.get(s)
            rows.append(
                f"{s.name:<8}{self.load_fraction[s]:>8.2f}"
                f"{self.cooling_units(s):>11.0f}"
                f"{self.electrical_kw(s):>12.2f}"
                f"{('—' if cop is None else f'{cop:.2f}'):>7}"
            )
        return "\n".join(rows)
