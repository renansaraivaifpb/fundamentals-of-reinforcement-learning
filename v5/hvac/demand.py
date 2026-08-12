# -*- coding: utf-8 -*-
"""
Demanda contratada (Grupo A) — a restrição que realimentação reativa não expressa.

POR QUE ISTO MUDA A ESTRUTURA DO PROBLEMA
-----------------------------------------
Todas as tentativas anteriores de tornar o RL vantajoso falharam porque não
mudavam a ESTRUTURA: um PI continuava sendo a ferramenta certa para rastrear
setpoint rejeitando perturbação. Demanda contratada é diferente por três motivos:

1. É uma restrição sobre o MÁXIMO, não sobre a média. Um PI minimiza erro
   instantâneo; não tem como representar "não ultrapasse 2 kW em nenhum momento
   deste mês".
2. O horizonte é o CICLO DE FATURAMENTO, não o passo. A penalidade de
   ultrapassagem depende do pico único do mês — é não-Markoviana no estado
   instantâneo.
3. Para respeitá-la é preciso ANTECIPAR: pré-resfriar antes do período de carga
   alta, para nunca precisar de potência máxima. Antecipação é exatamente o que
   realimentação reativa não faz, e o que uma política aprendida pode fazer.

Diferente do caso da tarifa horária — onde a tolerância de ±0,5 °C limitava o
armazenamento a 29 % do posto de ponta —, a restrição de demanda aperta em TODO
momento de carga alta, não só numa janela de 3 h.

COMO A ANEEL MEDE
-----------------
Demanda medida é a média INTEGRADA em 15 minutos, não a potência instantânea.
Isso é importante: um pico de 1 minuto não conta; 15 minutos de potência alta
contam. Com dt = 6 min, a janela são 2,5 passos.

Ultrapassar a demanda contratada gera cobrança de ultrapassagem, historicamente
com fator punitivo sobre o excedente. O valor exato vem da Resolução
Homologatória da distribuidora — os defaults aqui são ordem de grandeza, não
folha de dados.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class DemandContract:
    """
    Contrato de demanda no Grupo A.

    `contracted_kw` dimensionado abaixo da potência máxima do equipamento é o que
    cria a decisão: com folga de sobra a restrição nunca aperta e o problema
    volta a ser rastreamento puro.
    """

    contracted_kw: float = 1.6
    # R$/kW sobre a demanda faturada no mês.
    demand_charge_brl_per_kw: float = 30.0
    # Fator punitivo sobre o excedente (ordem de grandeza; ver REH da concessionária).
    overrun_multiplier: float = 2.0
    # Janela de integração da medição, em minutos (padrão ANEEL: 15).
    window_minutes: float = 15.0

    def window_steps(self, dt_hours: float) -> int:
        """
        Janela de integração em passos.

        ATENÇÃO: 15 min não é divisível pelo passo de 6 min (dá 2,5). A escolha
        importa e tem direção certa: janela MAIS CURTA promedia menos e portanto
        reporta pico MAIOR — mantém o controlador num padrão mais rígido que o
        real. Janela mais longa suavizaria o pico e faria o controlador parecer
        melhor do que é, que é o erro perigoso numa métrica de restrição.

        Por isso `floor`, e não `round`: além de escolher a direção conservadora,
        evita a armadilha do arredondamento bancário do Python, em que
        round(2.5) devolve 2 e round(3.5) devolve 4 — comportamento que mudaria
        silenciosamente com outro dt.
        """
        import math
        return max(1, math.floor(self.window_minutes / (dt_hours * 60.0)))


class DemandTracker:
    """
    Acompanha a demanda medida (média móvel de 15 min) e o pico do ciclo.

    Mantém o PICO porque é ele que determina a fatura: reduzir a média não
    adianta se houve um único período de 15 min acima do contratado.
    """

    def __init__(self, contract: DemandContract, dt_hours: float):
        self.contract = contract
        self.dt = dt_hours
        self.n = contract.window_steps(dt_hours)
        self.buffer: Deque[float] = deque(maxlen=self.n)
        self.peak_kw: float = 0.0
        self.overrun_steps: int = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.peak_kw = 0.0
        self.overrun_steps = 0

    def update(self, power_kw: float) -> float:
        """Registra a potência do passo e devolve a demanda medida vigente."""
        self.buffer.append(float(power_kw))
        medida = float(np.mean(self.buffer))
        self.peak_kw = max(self.peak_kw, medida)
        if medida > self.contract.contracted_kw:
            self.overrun_steps += 1
        return medida

    @property
    def overrun_kw(self) -> float:
        """Quanto o pico do ciclo excedeu o contratado."""
        return max(0.0, self.peak_kw - self.contract.contracted_kw)

    def billed_demand_brl(self) -> float:
        """
        Custo de demanda do ciclo: contratado + excedente com fator punitivo.

        Reportado em base DIÁRIA para ficar comparável com energia/custo do
        episódio — o contrato é mensal, mas o pico que o define pode ocorrer em
        qualquer dia.
        """
        c = self.contract
        base = c.contracted_kw * c.demand_charge_brl_per_kw
        excedente = self.overrun_kw * c.demand_charge_brl_per_kw * c.overrun_multiplier
        return (base + excedente) / 30.0

    def headroom(self) -> float:
        """
        Folga normalizada até o limite, em [0,1]: 1 = livre, 0 = no limite.

        É esta a feature que torna a restrição aprendível — sem ela o agente não
        tem como saber que está perto de estourar a demanda.
        """
        medida = float(np.mean(self.buffer)) if self.buffer else 0.0
        c = max(self.contract.contracted_kw, 1e-9)
        return float(np.clip(1.0 - medida / c, 0.0, 1.0))

    def stats(self) -> Dict[str, float]:
        return {
            "demand_peak_kw": self.peak_kw,
            "demand_overrun_kw": self.overrun_kw,
            "demand_overrun_pct": self.overrun_steps / max(1, len(self.buffer)) * 0.0,
            "demand_cost_brl_day": self.billed_demand_brl(),
        }
