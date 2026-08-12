# -*- coding: utf-8 -*-
"""
Espaço de observação DECLARATIVO — uma única fonte de verdade.

O PROBLEMA QUE ISTO RESOLVE
---------------------------
Na v4 a observação era declarada em dois lugares: o `Box` era montado no
`__init__` do ambiente e o vetor em `_get_obs()`. Duas listas paralelas,
mantidas à mão, em pontos diferentes do arquivo, que precisavam concordar em
ORDEM e em TAMANHO. Isso já produziu um bug documentado (`o_norm = 1,33` fora do
`Box` declarado) e é uma armadilha permanente: trocar a ordem de duas features
não gera erro algum — o `Box` aceita, o agente treina, e a política aprende com
os canais invertidos. Falha silenciosa, resultado inválido.

O sintoma mais eloquente do problema estava nos nomes dos arquivos: os modelos
se chamavam `TD3_..._lab2_obs9.zip`. A DIMENSÃO DA OBSERVAÇÃO tinha migrado para
o nome do arquivo, porque não havia onde mais guardá-la — o metadado JSON não
registrava nem o formato nem a ordem das features. Quando um dado estrutural
começa a viver no nome do arquivo, é sinal de que falta uma estrutura no código.

A SOLUÇÃO
---------
Uma lista de `Feature`, cada uma com nome, limites e extrator. Dela derivam,
por construção e sem possibilidade de divergência:

  * o `observation_space` (limites, na ordem declarada);
  * o vetor devolvido a cada passo (mesmos extratores, mesma ordem);
  * o SCHEMA (nomes ordenados + shape), gravado no metadado do modelo e
    verificado no carregamento — ver `model_io.assert_schema_compatible`.

Com isso, incompatibilidade entre um modelo treinado e o ambiente de avaliação
vira erro no carregamento, em vez de virar um número errado numa tabela.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Feature:
    """
    Um canal da observação.

    `fn` recebe o ambiente e devolve o valor JÁ normalizado. `ativa` decide, a
    partir da config, se o canal existe — é o que permite ablação de observação
    sem tocar em duas listas.
    """

    nome: str
    low: float
    high: float
    fn: Callable[[object], float]
    ativa: Callable[[object], bool] = lambda cfg: True
    doc: str = ""

    def valor(self, env) -> float:
        """Valor do canal, sempre saturado nos limites declarados.

        O clip aqui não é defensivo por hábito: é o que torna IMPOSSÍVEL repetir
        o bug de emitir um valor fora do `Box`, porque limite declarado e limite
        aplicado são literalmente o mesmo par de números.
        """
        return float(np.clip(self.fn(env), self.low, self.high))


# --------------------------------------------------------------- extratores
#
# Funções livres, e não métodos do ambiente, para que cada canal seja testável
# isoladamente e para que a lista abaixo leia como uma especificação.

def _t_norm(env) -> float:
    """eq. 3 — temperatura medida escalada pela faixa do equipamento."""
    return (env.measured_temp - 15.0) / 20.0


def _ocupacao(env) -> float:
    return env.occupancy / env.config.max_occupancy


def _sen_hora(env) -> float:
    return np.sin(2.0 * np.pi * env.hour_float / 24.0)


def _cos_hora(env) -> float:
    return np.cos(2.0 * np.pi * env.hour_float / 24.0)


def _erro_escalado(env) -> float:
    """
    Erro escalado pela TOLERÂNCIA, não pela faixa do equipamento.

    Com t_norm = (T-15)/20 a faixa de ±0,5 °C ocupa 5 % do intervalo da
    observação — a rede precisa resolver 0,05 em [0,1] exatamente onde vive todo
    o requisito. Escalando por 4·tolerância a mesma faixa passa a ocupar 50 %.
    """
    cfg = env.config
    escala = max(cfg.error_scale_tolerances * cfg.lab_tolerance, 1e-9)
    return (env.measured_temp - cfg.ideal_temp) / escala


def _integral(env) -> float:
    """Erro acumulado com fuga: sem integral não se elimina offset."""
    return env.integral_error / max(env.config.integral_clip_degree_hours, 1e-9)


def _derivada(env) -> float:
    """dT/dt normalizada pela maior variação possível por passo."""
    cfg = env.config
    max_d = (cfg.physics.cooling_units_at_full_load / cfg.thermal_mass) * cfg.dt
    return (env.measured_temp - env.prev_temp) / max(max_d, 1e-9)


def _tempo_ate_ponta(env) -> float:
    look = max(env.config.peak_lookahead_hours * 60.0, 1e-9)
    return env.minutes_to_peak() / look


def _tarifa(env) -> float:
    cfg = env.config
    taxas = [cfg.tariff.off_peak_brl_kwh, cfg.tariff.peak_brl_kwh]
    if cfg.tariff.intermediate_brl_kwh is not None:
        taxas.append(cfg.tariff.intermediate_brl_kwh)
    return cfg.tariff_rate(env.hour_float) / max(taxas)


def _previsao_externa(env, k: int) -> float:
    """
    Temperatura externa prevista k passos à frente, normalizada.

    Wei et al. (2017) mostram que uma sequência curta de previsão permite ao
    agente capturar a tendência e agir proativamente. A normalização usa a mesma
    escala de t_norm, para que previsão e medida vivam no mesmo espaço.
    """
    cfg = env.config
    dt = cfg.forecast_horizon_hours / max(cfg.forecast_steps, 1)
    hora = (env.hour_float + (k + 1) * dt) % 24.0
    fase = (hora - cfg.outdoor_peak_hour) * np.pi / 12.0
    base, amp = ((cfg.outdoor_base_temp, cfg.outdoor_amplitude)
                 if cfg.season == "summer" else (20.0, 6.0))
    return (float(base + amp * np.cos(fase)) - 15.0) / 20.0


def _folga_demanda(env) -> float:
    return env.demand.headroom() if env.demand else 1.0


# ------------------------------------------------------------ especificação
#
# A ORDEM desta lista é a ordem do vetor de observação. Alterá-la muda o
# significado de cada canal para todo modelo já treinado — é por isso que o
# schema é gravado no metadado e verificado no carregamento.

FEATURES: Sequence[Feature] = (
    Feature("t_norm", 0.0, 1.0, _t_norm,
            doc="temperatura medida, escalada pela faixa do equipamento"),
    Feature("ocupacao", 0.0, 1.0, _ocupacao,
            ativa=lambda c: c.observe_occupancy,
            doc="fração da lotação máxima"),
    Feature("sen_hora", -1.0, 1.0, _sen_hora, doc="codificação circular da hora"),
    Feature("cos_hora", -1.0, 1.0, _cos_hora, doc="codificação circular da hora"),
    Feature("erro_escalado", -1.0, 1.0, _erro_escalado,
            ativa=lambda c: c.observe_scaled_error,
            doc="erro escalado pela tolerância (resolução onde importa)"),
    Feature("integral", -1.0, 1.0, _integral,
            ativa=lambda c: c.observe_integral,
            doc="erro acumulado com fuga (anti-windup)"),
    Feature("derivada", -1.0, 1.0, _derivada,
            ativa=lambda c: c.observe_derivative,
            doc="dT/dt normalizada"),
    Feature("tempo_ate_ponta", 0.0, 1.0, _tempo_ate_ponta,
            ativa=lambda c: c.observe_time_to_peak,
            doc="minutos até o posto de ponta, normalizados"),
    Feature("tarifa", 0.0, 1.0, _tarifa,
            ativa=lambda c: c.observe_time_to_peak,
            doc="tarifa vigente, normalizada pela maior"),
    # Previsão da externa: uma feature por passo do horizonte. Declaradas em
    # laço para que mudar `forecast_steps` não exija editar esta lista — ela é a
    # única fonte de verdade e precisa acompanhar a config sozinha.
    *[Feature(f"previsao_ext_{k+1}", 0.0, 1.0,
              (lambda k: lambda env: _previsao_externa(env, k))(k),
              ativa=(lambda k: lambda c: c.observe_outdoor_forecast
                     and k < c.forecast_steps)(k),
              doc=f"externa prevista, passo {k+1} do horizonte")
      for k in range(8)],
    Feature("folga_demanda", 0.0, 1.0, _folga_demanda,
            ativa=lambda c: c.demand_limit_enabled and c.observe_demand_headroom,
            doc="folga até a demanda contratada"),
)


def features_ativas(config) -> List[Feature]:
    """Canais habilitados pela config, na ordem canônica."""
    return [f for f in FEATURES if f.ativa(config)]


def nomes(config) -> List[str]:
    return [f.nome for f in features_ativas(config)]


def limites(config):
    """(low, high) como arrays float32, para construir o `Box`."""
    ativas = features_ativas(config)
    return (np.array([f.low for f in ativas], dtype=np.float32),
            np.array([f.high for f in ativas], dtype=np.float32))


def vetor(env) -> np.ndarray:
    """Observação do passo atual, na mesma ordem que gerou o `Box`."""
    return np.array([f.valor(env) for f in features_ativas(env.config)],
                    dtype=np.float32)


def schema(config) -> dict:
    """
    Contrato da observação, gravado no metadado do modelo.

    `shape` sozinho não basta: duas configurações diferentes podem produzir a
    mesma dimensão com semânticas distintas — que é precisamente o modo de falha
    silenciosa que a v4 não conseguia detectar.
    """
    ativas = features_ativas(config)
    return {"n": len(ativas), "nomes": [f.nome for f in ativas]}


def descrever(config) -> str:
    """Tabela legível dos canais ativos, para conferência."""
    linhas = [f"{'#':>2}  {'canal':<16}{'faixa':<14}descrição",
              "-" * 74]
    for i, f in enumerate(features_ativas(config)):
        linhas.append(f"{i:>2}  {f.nome:<16}[{f.low:+.1f}, {f.high:+.1f}]  {f.doc}")
    return "\n".join(linhas)
