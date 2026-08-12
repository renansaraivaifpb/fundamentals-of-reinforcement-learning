# -*- coding: utf-8 -*-
"""
Tarifas reais de energia elétrica brasileiras (Enel), com resolução de meia hora.

Motivação: o manuscrito modela a tarifa como R$ 0,80/kWh com multiplicador 1,6 no
pico 18h–21h. A estrutura real da Tarifa Branca é diferente e mais rica:
**três** postos tarifários, razão de ponta de ~2,2× e fronteiras em meia hora.
Para avaliar economia em um laboratório real, a tarifa precisa ser a real — é ela
que define onde há dinheiro a economizar.

Fontes (consultadas em agosto/2026):
  Enel SP, Tarifa Branca B1 — https://calculadoraenergia.com.br/tarifa/enel-sp
  Postos horários ANEEL     — https://www.gov.br/aneel/pt-br/assuntos/tarifas/tarifa-branca

RESSALVAS IMPORTANTES
---------------------
1. Valores SEM impostos (ICMS + PIS/COFINS). O custo faturado é maior; use
   `with_taxes=True` para uma estimativa. Como os impostos são aproximadamente
   proporcionais, a estrutura de INCENTIVO (razão entre postos) é praticamente
   invariante a eles — o que importa para a política aprendida.
2. Tarifas mudam a cada revisão anual da ANEEL. Reconfirme antes de usar em
   dimensionamento real.
3. Fins de semana e feriados são integralmente fora de ponta. O ambiente não
   modela dia da semana; os episódios assumem DIA ÚTIL, que é o caso relevante
   para um laboratório e também o pior caso de custo.
4. Um laboratório universitário costuma estar no **Grupo A** (média tensão), com
   Tarifa Horosazonal Verde/Azul, que cobra **demanda contratada (kW)** além da
   energia (kWh). Nesse caso, limitar a POTÊNCIA DE PICO vale tanto quanto reduzir
   energia total, e a ultrapassagem de demanda tem multa pesada. Ver
   `GRUPO_A_VERDE` e a nota em `demand_charge_brl_per_kw`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TariffSchedule:
    """
    Tarifa com postos horários. Janelas em horas decimais (17.5 = 17h30).

    `rate(hour)` recebe hora decimal — usar hora inteira perderia as fronteiras
    de meia hora da Tarifa Branca, que é justamente onde a política precisa agir.
    """

    name: str
    off_peak_brl_kwh: float
    peak_brl_kwh: float
    intermediate_brl_kwh: Optional[float] = None
    peak_windows: Tuple[Tuple[float, float], ...] = ()
    intermediate_windows: Tuple[Tuple[float, float], ...] = ()

    # Impostos estimados (SP). Aplicados multiplicativamente quando solicitado.
    icms_rate: float = 0.18
    pis_cofins_rate: float = 0.09

    # Grupo A apenas: custo da demanda contratada. 0 para Grupo B.
    demand_charge_brl_per_kw: float = 0.0

    source: str = ""

    def _in_windows(self, hour: float, windows) -> bool:
        h = hour % 24.0
        for start, end in windows:
            if start <= end:
                if start <= h < end:
                    return True
            else:  # janela que cruza a meia-noite
                if h >= start or h < end:
                    return True
        return False

    def posto(self, hour: float) -> str:
        """Nome do posto tarifário vigente."""
        if self._in_windows(hour, self.peak_windows):
            return "ponta"
        if self.intermediate_brl_kwh is not None and self._in_windows(
            hour, self.intermediate_windows
        ):
            return "intermediario"
        return "fora_ponta"

    def rate(self, hour: float, with_taxes: bool = False) -> float:
        """R$/kWh vigente na hora decimal informada."""
        p = self.posto(hour)
        base = (
            self.peak_brl_kwh if p == "ponta"
            else self.intermediate_brl_kwh if p == "intermediario"
            else self.off_peak_brl_kwh
        )
        if with_taxes:
            # Aproximação: impostos "por dentro" sobre a tarifa da ANEEL.
            base = base / (1.0 - self.icms_rate - self.pis_cofins_rate)
        return float(base)

    @property
    def peak_ratio(self) -> float:
        """Razão ponta/fora-ponta — mede o incentivo à antecipação."""
        return self.peak_brl_kwh / self.off_peak_brl_kwh

    def describe(self) -> str:
        lines = [f"{self.name}  (razão ponta/fora-ponta = {self.peak_ratio:.2f}x)"]
        lines.append(f"  fora ponta      R$ {self.off_peak_brl_kwh:.4f}/kWh")
        if self.intermediate_brl_kwh is not None:
            w = ", ".join(f"{a:g}h–{b:g}h" for a, b in self.intermediate_windows)
            lines.append(f"  intermediário   R$ {self.intermediate_brl_kwh:.4f}/kWh  ({w})")
        w = ", ".join(f"{a:g}h–{b:g}h" for a, b in self.peak_windows)
        lines.append(f"  ponta           R$ {self.peak_brl_kwh:.4f}/kWh  ({w})")
        if self.demand_charge_brl_per_kw:
            lines.append(f"  demanda         R$ {self.demand_charge_brl_per_kw:.2f}/kW")
        if self.source:
            lines.append(f"  fonte: {self.source}")
        return "\n".join(lines)


# --- Enel São Paulo, Tarifa Branca B1 (residencial) -----------------------
ENEL_SP_BRANCA = TariffSchedule(
    name="Enel SP — Tarifa Branca B1",
    off_peak_brl_kwh=0.669,
    intermediate_brl_kwh=0.985,
    peak_brl_kwh=1.480,
    peak_windows=((17.5, 20.5),),
    intermediate_windows=((16.5, 17.5), (20.5, 21.5)),
    source="calculadoraenergia.com.br/tarifa/enel-sp (ago/2026, sem impostos)",
)

# --- Enel São Paulo, convencional (tarifa única) --------------------------
ENEL_SP_CONVENCIONAL = TariffSchedule(
    name="Enel SP — Convencional B1",
    off_peak_brl_kwh=0.789,
    peak_brl_kwh=0.789,
    source="calculadoraenergia.com.br/tarifa/enel-sp (ago/2026, sem impostos)",
)

# --- Enel Rio de Janeiro, Tarifa Branca ----------------------------------
ENEL_RJ_BRANCA = TariffSchedule(
    name="Enel RJ — Tarifa Branca B1",
    off_peak_brl_kwh=0.745,
    intermediate_brl_kwh=1.229,
    peak_brl_kwh=1.613,
    peak_windows=((17.5, 20.5),),
    intermediate_windows=((16.5, 17.5), (20.5, 21.5)),
    source="calculadoraenergia.com.br/tarifa/enel-rj (ago/2026, sem impostos)",
)

# --- Referência do manuscrito (para comparação) --------------------------
PAPER_TARIFF = TariffSchedule(
    name="Manuscrito (R$0,80 com ×1,6 no pico)",
    off_peak_brl_kwh=0.80,
    peak_brl_kwh=0.80 * 1.6,
    peak_windows=((18.0, 21.0),),
    source="29914_Paper_manuscript.pdf, Seção 5",
)

# --- Grupo A Verde (esqueleto; exige demanda contratada) -----------------
# NÃO calibrado — os valores de Grupo A dependem de contrato e subgrupo (A4, A3).
# Deixado explícito porque um laboratório real provavelmente está aqui, e nesse
# caso a métrica de potência de pico passa a ter custo próprio.
GRUPO_A_VERDE = TariffSchedule(
    name="Grupo A Verde (ESQUELETO — não calibrado)",
    off_peak_brl_kwh=0.45,
    peak_brl_kwh=2.20,
    peak_windows=((17.5, 20.5),),
    demand_charge_brl_per_kw=30.0,
    source="PLACEHOLDER — obter da Resolução Homologatória da distribuidora",
)

TARIFFS = {
    "enel_sp_branca": ENEL_SP_BRANCA,
    "enel_sp_convencional": ENEL_SP_CONVENCIONAL,
    "enel_rj_branca": ENEL_RJ_BRANCA,
    "paper": PAPER_TARIFF,
    "grupo_a_verde": GRUPO_A_VERDE,
}


def get_tariff(name: str) -> TariffSchedule:
    if name not in TARIFFS:
        raise KeyError(f"tarifa '{name}' desconhecida. Use: {list(TARIFFS)}")
    return TARIFFS[name]


if __name__ == "__main__":
    for key in ("enel_sp_branca", "enel_sp_convencional", "enel_rj_branca", "paper"):
        print(get_tariff(key).describe(), "\n")

    print("Perfil horário (Enel SP Branca), passo de 30 min:")
    t = ENEL_SP_BRANCA
    for i in range(32, 45):
        h = i / 2.0
        print(f"  {int(h):02d}:{int((h%1)*60):02d}  {t.posto(h):<14} R$ {t.rate(h):.4f}")
