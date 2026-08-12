# -*- coding: utf-8 -*-
"""
Dimensionamento da demanda contratada — DERIVADO, não sintonizado.

POR QUE ESTE MÓDULO EXISTE
--------------------------
A primeira versão fixava `demand_contracted_kw = 0,70`, escolhido por varredura
como "ponto de tensão máxima": o valor em que o dilema entre conforto e demanda
ficava mais agudo. Isso é o MESMO erro metodológico que esta reprodução acusou no
manuscrito (achado nº 1 do REVISAO.md): ajustar a configuração do problema até
que o resultado desejado apareça. Lá era um baseline mal configurado; aqui seria
uma restrição calibrada para favorecer a hipótese.

Pior: o valor era INFACTÍVEL. O regime permanente de pior caso exige 1,204 kW —
0,70 kW fica 72 % abaixo, e sustenta apenas ~17 das 45 pessoas. Nenhuma política
respeita a restrição, porque antecipar não cria regime permanente: pré-resfriar
compra um transitório limitado pela tolerância de ±0,5 °C, não potência contínua.
O que parecia "o RL precisa antecipar" era, na verdade, "todo controlador é
forçado a violar, e só resta escolher COMO violar".

O critério aqui é o de engenharia elétrica, e ele preserva a estrutura do
experimento sem escolhê-la:

  1. A demanda faturada é a média INTEGRADA em 15 min (ver demand.py), então
     dimensiona-se pelo REGIME SUSTENTADO, nunca pelo transitório de partida.
  2. A condição de projeto é o pior caso sustentado: ocupação máxima, externa no
     pico diário, mantendo o setpoint.
  3. Aplica-se margem de contratação sobre esse valor.

A estrutura que torna o problema interessante EMERGE do cálculo, em vez de ser
imposta: o regime cabe no contrato (por construção), mas o pulldown a plena carga
(2,931 kW) e o aquecimento a plena carga (2,198 kW) não cabem — ambos ~2x acima.
Continua sendo verdade que só quem modula a potência fica dentro; a diferença é
que agora existe pelo menos uma política que consegue.

SOBRE A MARGEM
--------------
A ANEEL prevê tolerância antes da cobrança punitiva de ultrapassagem (da ordem de
5 % para unidades de subgrupos de baixa tensão do Grupo A). Essa tolerância é o
PISO: contrata-se acima dela para absorver variação de carga não modelada
(equipamento auxiliar, iluminação, dias atípicos). `DEFAULT_MARGIN = 0,10` é
escolha de projeto documentada, não folha de dados — e é um parâmetro exposto,
para que a sensibilidade ao valor seja mensurável em vez de escondida.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                        # evita ciclo de import com config.py
    from .physics import ACPhysicsModel

# Margem de contratação sobre a demanda de regime da condição de projeto.
DEFAULT_MARGIN: float = 0.10


@dataclass(frozen=True)
class DemandSizing:
    """
    Memória de cálculo do dimensionamento.

    Existe para que o número seja AUDITÁVEL, no mesmo espírito de
    `ACPhysicsModel.as_table()`: um revisor precisa poder refazer a conta a
    partir de parâmetros publicados, não aceitar uma constante.
    """

    outdoor_design_temp: float      # externa da condição de projeto (°C)
    occupancy_design: int           # ocupação da condição de projeto
    setpoint: float                 # temperatura mantida (°C)
    people_units: float             # carga térmica das pessoas (u)
    envelope_units: float           # carga térmica pelo envelope (u)
    thermal_load_units: float       # carga total sustentada (u)
    load_fraction: float            # fração da capacidade nominal exigida
    steady_state_kw: float          # potência elétrica de regime (kW)
    margin: float                   # margem de contratação aplicada
    contracted_kw: float            # RESULTADO: demanda a contratar (kW)
    pulldown_kw: float              # contrafactual: plena carga resfriando (kW)
    heating_full_kw: float          # contrafactual: plena carga aquecendo (kW)

    @property
    def pulldown_ratio(self) -> float:
        """Quantas vezes o pulldown excede o contrato. > 1 => modular é obrigatório."""
        return self.pulldown_kw / max(self.contracted_kw, 1e-9)

    def as_table(self) -> str:
        """Memória de cálculo para conferência/auditoria."""
        L = [
            "Dimensionamento da demanda contratada (condição de projeto)",
            "-" * 62,
            f"  externa de projeto        {self.outdoor_design_temp:>10.2f} °C",
            f"  ocupação de projeto       {self.occupancy_design:>10d} pessoas",
            f"  setpoint mantido          {self.setpoint:>10.2f} °C",
            "",
            f"  carga: pessoas            {self.people_units:>10.2f} u",
            f"  carga: envelope           {self.envelope_units:>10.2f} u",
            f"  carga: TOTAL sustentada   {self.thermal_load_units:>10.2f} u",
            f"  fração da capacidade      {self.load_fraction:>10.4f}",
            "",
            f"  potência de REGIME        {self.steady_state_kw:>10.4f} kW",
            f"  margem de contratação     {self.margin:>10.1%}",
            f"  DEMANDA CONTRATADA        {self.contracted_kw:>10.4f} kW",
            "",
            "  contrafactuais (a estrutura do problema, não uma escolha):",
            f"    pulldown a plena carga  {self.pulldown_kw:>10.4f} kW"
            f"   ({self.pulldown_ratio:.2f}x o contrato)",
            f"    aquecimento pleno       {self.heating_full_kw:>10.4f} kW",
        ]
        return "\n".join(L)


def size_contracted_demand(
    *,
    physics: "ACPhysicsModel",
    max_occupancy: int,
    heat_gain_per_person: float,
    heat_transfer_coeff: float,
    outdoor_base_temp: float,
    outdoor_amplitude: float,
    ideal_temp: float,
    margin: float = DEFAULT_MARGIN,
) -> DemandSizing:
    """
    Dimensiona a demanda contratada a partir da condição de projeto.

    Recebe escalares em vez de uma `ClassroomConfig` de propósito: mantém o
    módulo livre de ciclo de import com `config.py` e testável isoladamente.

    A externa de projeto é o PICO da senoide diária (base + amplitude) — é o
    instante em que a carga sustentada é máxima e, portanto, o que dimensiona.
    """
    outdoor_design = outdoor_base_temp + outdoor_amplitude

    people = max_occupancy * heat_gain_per_person
    envelope = heat_transfer_coeff * (outdoor_design - ideal_temp)
    # O envelope pode ser negativo (externa abaixo do setpoint); nesse caso ele
    # ALIVIA a carga. Não se dimensiona por alívio, então o piso é a carga das
    # pessoas — mas na condição de projeto (pico diário) o termo é positivo.
    load_units = max(people + envelope, people)

    fraction = min(load_units / physics.cooling_units_at_full_load, 1.0)
    steady_kw = physics.electrical_kw_continuous(fraction)

    return DemandSizing(
        outdoor_design_temp=outdoor_design,
        occupancy_design=max_occupancy,
        setpoint=ideal_temp,
        people_units=people,
        envelope_units=envelope,
        thermal_load_units=load_units,
        load_fraction=fraction,
        steady_state_kw=steady_kw,
        margin=margin,
        contracted_kw=steady_kw * (1.0 + margin),
        pulldown_kw=physics.electrical_kw_continuous(1.0),
        # `electrical_kw_signed` devolve 0 quando o aquecimento está desligado no
        # modelo físico; a conta abaixo é o contrafactual "se fosse reversível",
        # que é o que dimensiona um equipamento reversível de verdade.
        heating_full_kw=(physics.capacity_kw_thermal
                         * physics.heating_capacity_ratio / physics.heating_cop),
    )
