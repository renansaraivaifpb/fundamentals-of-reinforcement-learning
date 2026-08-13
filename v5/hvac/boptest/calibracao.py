# -*- coding: utf-8 -*-
"""
Dimensionamento do intervalo de controle no emulador de terceiros.

O PROBLEMA
----------
O protocolo do artigo decide a cada 12 min, e no ambiente local isso é uma
escolha benigna: a plena carga move a sala 0,09 °C por passo de 6 min, de modo
que atravessar a meia-faixa de conforto exige dezenas de passos. No `bestest_air`
a razão é outra. A zona é leve, o fancoil é dimensionado para ela, e a plena
carga move a temperatura vários graus em um único passo de 12 min — mais que a
largura inteira da faixa de conforto.

Manter o intervalo de 12 min ali não preserva o protocolo: transforma o problema
em liga-desliga puro, no qual nenhum dos controladores pode acertar a faixa e a
comparação deixa de discriminar. Esta é, medida de fora, a mesma propriedade que
a Seção 5.3 declara como a ameaça mais séria à validade externa do artigo — a
planta local é lenta demais, e o emulador torna isso visível.

O CRITÉRIO
----------
O intervalo NÃO é arbitrado até o resultado ficar bom. Ele é derivado de uma
condição de projeto declarada antes de medir:

    a plena carga não deve atravessar mais que a meia-faixa de conforto
    em um único intervalo de decisão

que é a condição para que os quatro níveis discretos consigam, em princípio,
estacionar dentro da faixa. É a mesma disciplina que a Seção 5.2 recomenda ao
tratar da demanda contratada: derivar de condição de projeto em vez de calibrar
para produzir o resultado desejado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import ClassroomConfig
from .client import BoptestClient
from .env import ENTRADA_TSUP, ENTRADA_VENT, PONTO_TEMP, ZERO_C


@dataclass
class Autoridade:
    """Resultado da medição de autoridade do atuador."""

    intervalo_medido_s: float
    delta_plena_c: float          # variação com carga plena, no intervalo medido
    delta_minima_c: float         # variação com o menor nível não nulo
    meia_faixa_c: float
    intervalo_derivado_s: float
    temp_inicial_c: float

    def __str__(self) -> str:
        return (f"autoridade em {self.intervalo_medido_s:.0f} s: "
                f"plena {self.delta_plena_c:+.2f} °C, "
                f"mínima {self.delta_minima_c:+.2f} °C; "
                f"meia-faixa {self.meia_faixa_c:.1f} °C -> "
                f"intervalo derivado {self.intervalo_derivado_s:.0f} s")


def medir_autoridade(*, caso: str = "bestest_air",
                     periodo: str = "peak_cool_day",
                     t_insuflamento_k: float = 285.15,
                     intervalo_sonda_s: float = 720.0,
                     passos_ate_carga: int = 70,
                     config: Optional[ClassroomConfig] = None,
                     url: Optional[str] = None,
                     piso_s: float = 60.0,
                     teto_s: float = 900.0) -> Autoridade:
    """
    Mede a variação de temperatura por nível e devolve o intervalo derivado.

    A sonda avança sem atuação até o período de maior carga do dia, quando a
    diferença entre a zona e o insuflamento é maior e, portanto, a autoridade do
    atuador é máxima. Dimensionar pelo pior caso é deliberado: um intervalo que
    funcione ali funciona no resto do dia.

    O intervalo é saturado em [`piso_s`, `teto_s`]. O piso existe porque um
    intervalo muito curto multiplica o custo de simulação sem alterar a
    conclusão, e o teto porque acima dele o critério já está satisfeito.
    """
    cfg = config or ClassroomConfig()
    meia_faixa = (cfg.temp_comfort_max - cfg.temp_comfort_min) / 2.0

    deltas = {}
    t_ini = float("nan")
    for carga in (cfg.physics.discrete_levels()[1], 1.0):   # menor não nulo e plena
        cliente = BoptestClient(url=url) if url else BoptestClient()
        try:
            cliente.selecionar(caso)
            cliente.cenario(periodo=periodo, preco="constant")
            cliente.passo(intervalo_sonda_s)
            desligado = {ENTRADA_VENT + "_u": 0.0, ENTRADA_VENT + "_activate": 1,
                         ENTRADA_TSUP + "_u": t_insuflamento_k,
                         ENTRADA_TSUP + "_activate": 1}
            y = {}
            for _ in range(passos_ate_carga):
                y = cliente.avancar(desligado)
            t0 = float(y[PONTO_TEMP]) - ZERO_C
            y = cliente.avancar({ENTRADA_VENT + "_u": float(carga),
                                 ENTRADA_VENT + "_activate": 1,
                                 ENTRADA_TSUP + "_u": t_insuflamento_k,
                                 ENTRADA_TSUP + "_activate": 1})
            deltas[carga] = float(y[PONTO_TEMP]) - ZERO_C - t0
            t_ini = t0
        finally:
            cliente.encerrar()

    delta_plena = deltas[1.0]
    # Regra de três sobre o intervalo da sonda: a resposta é aproximadamente
    # linear no intervalo para passos curtos frente à constante de tempo da zona.
    bruto = intervalo_sonda_s * meia_faixa / max(abs(delta_plena), 1e-9)
    derivado = float(min(max(bruto, piso_s), teto_s))
    # Múltiplo de 60 s: intervalos redondos evitam deriva de fase entre o passo
    # de controle e as agendas horárias do emulador.
    derivado = max(round(derivado / 60.0) * 60.0, piso_s)

    return Autoridade(intervalo_medido_s=intervalo_sonda_s,
                      delta_plena_c=delta_plena,
                      delta_minima_c=deltas[cfg.physics.discrete_levels()[1]],
                      meia_faixa_c=meia_faixa,
                      intervalo_derivado_s=derivado,
                      temp_inicial_c=t_ini)
