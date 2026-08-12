# -*- coding: utf-8 -*-
"""
Configuração do ambiente e os três perfis operacionais (Tabela 2 do paper).

NOTA DE REPRODUÇÃO — parâmetros inferidos
------------------------------------------
O paper especifica numericamente `B_c`, a penalidade de energia e a de troca
(Tabela 2), mas NÃO publica `B` (bônus base), `k` (curvatura fora da faixa),
`rho` (peso do anti-short-cycling) nem a penalidade de frio. Os valores abaixo
foram inferidos da Figura 1 e estão marcados com `# INFERIDO`:

  B = 10.0  -> a Figura 1 mostra as bordas do platô (22 e 26 °C) em ≈ +10.
  k = 0.6   -> em 32 °C a curva laranja cai a ≈ −11: 10 − k·6² = −11 => k ≈ 0,58.
               0,6 também é o valor usado no v4 original (comfort_sensitivity).
  rho = -5.0 -> não observável na Figura 1. Escolhido na ordem de grandeza do
               conforto, para conter a comutação sem dominar a recompensa.
  cold_action_penalty = -2.0 -> a Conclusão do paper alerta que uma penalidade
               severa no frio induziu "medo de resfriar" e degradou a política.
               Mantida deliberadamente branda (o v4 original usava -20).

Qualquer divergência dos números da Tabela 4 deve ser lida à luz destes quatro
parâmetros antes de se suspeitar da implementação.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

from .physics import ACPhysicsModel
from .demand_sizing import DEFAULT_MARGIN, DemandSizing, size_contracted_demand


@dataclass
class ClassroomConfig:
    """Parâmetros da sala, da recompensa e da tarifa."""

    # --- Geometria e ocupação (Seção 4.1: sala fictícia, 45 ocupantes) ---
    max_occupancy: int = 45
    heat_gain_per_person: float = 0.3   # 45 · 0,3 = 13,5 u -> carga dominante

    # --- Dinâmica térmica (eq. 2) ---
    thermal_mass: float = 15.0          # C_th da sala de treino (Seção 5.5)
    heat_transfer_coeff: float = 0.5    # K_transf: 0,5·(36−24) = 6 u no pico
    temperature_noise_std: float = 0.01
    dt: float = 0.1                     # 0,1 h = 6 min por passo
    episode_steps: int = 240            # 240 · 0,1 h = 24 h
    temp_min_clip: float = 10.0
    temp_max_clip: float = 40.0

    # --- Temperatura externa senoidal (pico às 14h) ---
    season: str = "summer"
    outdoor_base_temp: float = 28.0
    outdoor_amplitude: float = 8.0
    outdoor_peak_hour: float = 14.0

    # --- Faixas de conforto ---
    ideal_temp: float = 24.0
    temp_comfort_min: float = 22.0
    temp_comfort_max: float = 26.0
    temp_narrow_min: float = 23.0       # faixa estreita [23,25] da Tabela 4
    temp_narrow_max: float = 25.0

    # --- Janela ocupada (Seção 4.4) ---
    occupied_hour_start: int = 7
    occupied_hour_end: int = 22

    # --- Recompensa de conforto (eq. 4) ---
    # Topologia: 'plateau' é a eq. 4 do paper (platô + gradiente interno);
    # 'quadratic' e 'step' existem para a ABLAÇÃO exigida pelo Revisor 2 —
    # sem poder desligar a contribuição, a tese "recompensa > algoritmo" não é
    # verificável. Com comfort_gradient = 0, 'plateau' vira o platô plano, que é
    # a hipótese explícita do paper para o "estacionar na borda".
    # 'band' trata a tolerância como FAIXA ACEITÁVEL, não como setpoint: plano
    # dentro, íngreme fora. É o que uma especificação de laboratório de fato diz
    # ("mantenha entre 23,5 e 24,5"), e muda a natureza do problema — deixa de
    # ser rastreamento e vira otimização de custo sujeita a restrição. Um PI vai
    # ao centro e desperdiça a faixa; a política ótima FLUTUA dentro dela usando-a
    # como armazenamento térmico.
    #
    # RISCO CONHECIDO: platô plano foi o que colapsou o treino na ablação
    # (conforto estreito 83,7 -> 47,7, delta = -1,00). A diferença aqui é que lá
    # NÃO havia outro gradiente dentro da faixa; aqui o custo de energia e a folga
    # de demanda fornecem o gradiente. Um platô plano só é zona morta quando nada
    # mais varia dentro dele.
    # 'hinge' é a recompensa de Wei et al. (DAC 2017), referência fundacional de
    # DRL para HVAC: r_conforto = -lambda * ([T - T_max]+ + [T_min - T]+). Sem
    # bônus base, sem platô, sem gradiente interno. Existe aqui para que a
    # formulação do manuscrito seja comparável à da literatura que a antecede —
    # a ablação já mostrou que a quadrática convencional empata com a proposta, e
    # esta é ainda mais simples.
    comfort_type: str = "plateau"  # 'plateau'|'quadratic'|'step'|'huber'|'band'|'hinge'
    # Raio da zona quadratica do conforto Huber (usado so por comfort_type='huber').
    comfort_huber_delta: float = 0.5
    # Inclinação da parede fora da faixa (só para comfort_type='band').
    band_wall_slope: float = 40.0
    # Peso da violação de faixa em comfort_type='hinge' (lambda da eq. 1 de
    # Wei et al. 2017).
    comfort_hinge_lambda: float = 10.0
    comfort_bonus: float = 10.0            # B          # INFERIDO
    comfort_gradient: float = 4.0          # B_c        (Tabela 2: perfil)
    comfort_sensitivity: float = 0.6       # k          # INFERIDO

    # --- Demais termos da recompensa ---
    energy_penalty_factor: float = 0.10    # (Tabela 2: perfil)
    action_change_penalty: float = -1.5    # (Tabela 2: perfil)
    cold_action_penalty: float = -2.0                   # INFERIDO
    short_cycle_penalty: float = -5.0      # rho        # INFERIDO
    min_dwell_minutes: float = 36.0        # d_min (eq. 5)

    # --- Tarifa ---
    # `tariff` substitui o modelo do manuscrito (R$0,80 com ×1,6 em 18–21h) por
    # uma tarifa real com resolução de meia hora. Os campos legados abaixo são
    # mantidos apenas para reproduzir o manuscrito.
    tariff_name: str = "paper"
    energy_tariff_brl_per_kwh: float = 0.80          # legado
    peak_hours: Tuple[int, int] = (18, 21)           # legado
    peak_tariff_multiplier: float = 1.6              # legado
    tariff_with_taxes: bool = False

    # O manuscrito escala a PENALIDADE de energia no pico (Seção 4.3) e a TARIFA
    # no pico (Seção 5). Mantidos separados de propósito: um é sinal de treino, o
    # outro é métrica de avaliação. Com `price_aware_penalty=True` a penalidade
    # passa a seguir a tarifa REAL — condição para o agente poder antecipar.
    peak_penalty_multiplier: float = 1.6
    price_aware_penalty: bool = False

    # --- Espaço de ação e observação estendidos ---
    # Todos default=False para preservar a reprodução exata do manuscrito.
    heating_enabled: bool = False
    continuous_action: bool = False

    # dT/dt na observação. Sem derivada o agente não distingue "24,0 subindo
    # rápido" de "24,0 estável" — é o papel do termo derivativo de um PID. Sem
    # ela o estado não é Markoviano para fins de controle: sistemas térmicos têm
    # inércia, e pedir precisão a um agente cego para velocidade é contraditório.
    observe_derivative: bool = False

    # Tempo até o posto de ponta + tarifa vigente. Sem isso o agente precisa
    # INFERIR de sin/cos da hora que faltam 90 min para o preço subir 2,2x, e a
    # exploração ε-greedy não encontra a sequência coerente de ~2 h que o
    # pré-resfriamento exige. Com a feature explícita, deixa de ser um problema
    # de exploração de horizonte longo.
    observe_time_to_peak: bool = False
    peak_lookahead_hours: float = 4.0

    # PREVISÃO DA PERTURBAÇÃO EXTERNA, seguindo Wei et al. (DAC 2017): eles
    # incluem no estado não só a externa atual, mas uma sequência curta de
    # previsão, "enabling the DRL algorithm to capture the trend of the
    # environment, perform proactive control". É a informação que separa reagir
    # de antecipar — e um PI, por construção, não a utiliza.
    #
    # RESSALVA: neste ambiente a externa é uma senoide determinística, logo a
    # previsão é PERFEITA. Num prédio real ela tem erro, e parte da vantagem
    # medida aqui não sobreviveria. É otimismo declarado, não escondido.
    observe_outdoor_forecast: bool = False
    forecast_steps: int = 3               # passos à frente
    forecast_horizon_hours: float = 2.0   # janela total prevista

    # ERRO escalado pela tolerância, em vez de temperatura absoluta escalada pela
    # faixa do equipamento. Medido: com t_norm = (T-15)/20, a faixa de ±0,5 °C
    # ocupa 5 % do intervalo da observação — a rede precisa resolver 0,05 em
    # [0,1] exatamente onde vive todo o requisito. Escalando por 4·tolerância a
    # mesma faixa passa a ocupar 50 %: ganho de 10x em resolução onde importa.
    # É a mesma lógica do conforto Huber, aplicada à observação em vez da
    # recompensa.
    observe_scaled_error: bool = False
    error_scale_tolerances: float = 4.0

    # ERRO ACUMULADO. O estado suficiente para rastreamento de setpoint é
    # (e, integral de e, derivada de e). Uma política sem memória não pode
    # integrar — só reagir — então qualquer viés persistente produz offset
    # permanente. É o que separa sigma = 0,088 do PI de 0,198 do melhor RL.
    # Integral com fuga (leaky) para não divergir: é o análogo do anti-windup.
    observe_integral: bool = False
    integral_leak: float = 0.98
    integral_clip_degree_hours: float = 2.0

    # --- Distribuição de ocupação no TREINO ---
    # PROBLEMA DE ARQUITETURA medido: o treino usa passeio aleatório (média 22,7,
    # 9,9 % nos extremos) e a avaliação usa a janela ocupada (média 15,0, 58,3 %
    # nos extremos, mediana 5). Teste KS: D = 0,438, p ~ 0. O agente é treinado
    # numa distribuição que praticamente não vê no teste, e quase nunca
    # experimenta as transições em degrau 0->45 que definem todo cenário de
    # avaliação. 'schedule' treina sob a MESMA estrutura da avaliação, com
    # parâmetros randomizados (domain randomization sobre a agenda).
    # 'realistic' é o único que usa a MESMA estrutura no treino e na avaliação:
    # grade de aulas/intervalos/almoço com entra-e-sai contínuo. Ver occupancy.py.
    train_occupancy_mode: str = "random_walk"   # 'random_walk'|'schedule'|'realistic'
    schedule_occupancy_jitter: int = 8          # variação da lotação de pico
    schedule_hour_jitter: float = 2.0           # variação das bordas da janela
    occupancy_tau_minutes: float = 15.0         # tempo de enchimento/esvaziamento
    occupancy_churn_per_hour: float = 6.0       # entra-e-sai dentro do bloco

    # --- Modelo de SENSOR ---
    # O agente observa a temperatura verdadeira, sem ruído nem atraso. Sensor
    # real tem ambos. `temperature_noise_std` é ruído de PROCESSO (no estado),
    # não de MEDIÇÃO. Sem isso, qualquer afirmação de precisão de ±0,5 °C é
    # otimista: parte do orçamento de erro real é do próprio sensor.
    sensor_noise_std: float = 0.0
    sensor_lag_steps: int = 0

    # --- Penalidade de DERIVADA (custo de controle) ---
    # A recompensa é puramente instantânea. Um custo de controle clássico (LQR)
    # penaliza erro, esforço E variação: -(w_e*e^2 + w_d*edot^2 + w_u*u^2). O
    # projeto tem e^2 (Huber) e u (energia), mas NADA em edot — e é exatamente a
    # oscilação que separa sigma 0,198 do RL de 0,088 do PI.
    derivative_penalty: float = 0.0

    # --- Demanda contratada (Grupo A) ---
    # Restrição sobre o MÁXIMO (média de 15 min), não sobre a média — e com
    # horizonte de ciclo de faturamento. Um PI não tem como representá-la: ele
    # minimiza erro instantâneo. Respeitá-la exige ANTECIPAR carga, que é o que
    # realimentação reativa não faz. Ver demand.py.
    #
    # O VALOR CONTRATADO NÃO É UM PARÂMETRO LIVRE. Ele é DERIVADO da condição de
    # projeto (ver demand_sizing.py), pela mesma razão que a Tabela 1 é derivada
    # de BTU/h + COP em vez de tabelada: uma constante escolhida à mão numa
    # restrição é indistinguível de uma restrição calibrada para produzir o
    # resultado desejado — exatamente a crítica que esta reprodução faz ao
    # manuscrito no achado nº 1.
    demand_limit_enabled: bool = False
    demand_penalty: float = 3.0            # peso na recompensa ao exceder
    observe_demand_headroom: bool = True   # folga até o limite, na observação
    # Margem de contratação sobre a demanda de regime da condição de projeto.
    demand_contract_margin: float = DEFAULT_MARGIN
    # Override explícito, para ANÁLISE DE SENSIBILIDADE ao valor contratado.
    # Deixar em None (o padrão) usa o valor derivado. Existe para que variar a
    # restrição seja um experimento declarado, e não uma constante silenciosa.
    demand_contracted_kw_manual: Optional[float] = None

    # Ocupação observada. Num laboratório real raramente há contagem de pessoas;
    # desligar torna o cenário mais honesto (crítica do R1), mas NÃO favorece o
    # RL — o PI nunca usou ocupação, então esconder informação só penaliza o
    # agente. Mantido como opção de realismo, não como hipótese de vantagem.
    observe_occupancy: bool = True

    # --- Tolerância de laboratório ---
    # Um laboratório de precisão não tem "faixa de conforto" de 4 °C: tem
    # tolerância em torno de um setpoint. `tolerance` é a métrica que importa
    # (fração do tempo dentro de ±tolerance do setpoint) e NÃO altera a
    # recompensa — é medição, não sinal de treino.
    lab_tolerance: float = 0.5

    # --- Safety shield (Seção 5.4) ---
    shield_force_cool_above: float = 28.0
    shield_force_off_below: float = 20.0

    # --- Modelo do equipamento ---
    physics: ACPhysicsModel = field(default_factory=ACPhysicsModel)

    def __post_init__(self) -> None:
        """
        Propaga `heating_enabled` para o modelo físico.

        A config é a FONTE DE VERDADE. Antes isto era feito por `ClassroomACEnv.
        __init__`, o que deixava a config incompleta até que um env fosse
        instanciado: `config_for_lab2(...).physics.heating_enabled` era False
        mesmo com `heating_enabled=True`. Qualquer consumidor que lesse a física
        sem passar pelo env via um equipamento só-frio.

        REGRESSÃO CONCRETA que isso causou: `DemandAwarePI._max_heat_load()` lê
        `physics.heating_enabled` e devolvia 0,0, de modo que o limitador de
        demanda nunca exercitava o lado do aquecimento — e o teste que existia
        para proteger exatamente essa simetria passava de forma vacuosa.
        """
        if self.heating_enabled and not self.physics.heating_enabled:
            self.physics = replace(self.physics, heating_enabled=True)

    @property
    def min_dwell_steps(self) -> int:
        """d_min convertido para passos: 36 min / 6 min = 6 passos."""
        return int(round(self.min_dwell_minutes / (self.dt * 60.0)))

    # --- Demanda contratada: derivada da condição de projeto ---------------

    @property
    def demand_sizing(self) -> DemandSizing:
        """
        Memória de cálculo do dimensionamento (auditável, ver demand_sizing.py).

        Memoizada pela mesma razão que `tariff`: é lida a cada passo pela
        recompensa, e recomputar por passo é desperdício num treino de 250k.
        A chave cobre tudo que entra na conta, então mudar qualquer parâmetro de
        projeto — inclusive via `dataclasses.replace` — re-deriva o valor.
        """
        chave = (
            self.max_occupancy, self.heat_gain_per_person,
            self.heat_transfer_coeff, self.outdoor_base_temp,
            self.outdoor_amplitude, self.ideal_temp, self.demand_contract_margin,
            self.physics.capacity_btu_per_hour,
            self.physics.cooling_units_at_full_load,
            self.physics.heating_capacity_ratio, self.physics.heating_cop,
            tuple(sorted(self.physics.cop.items())),
            tuple(sorted(self.physics.load_fraction.items())),
        )
        if getattr(self, "_sizing_cache", (None,))[0] != chave:
            self._sizing_cache = (chave, size_contracted_demand(
                physics=self.physics,
                max_occupancy=self.max_occupancy,
                heat_gain_per_person=self.heat_gain_per_person,
                heat_transfer_coeff=self.heat_transfer_coeff,
                outdoor_base_temp=self.outdoor_base_temp,
                outdoor_amplitude=self.outdoor_amplitude,
                ideal_temp=self.ideal_temp,
                margin=self.demand_contract_margin,
            ))
        return self._sizing_cache[1]

    @property
    def demand_contracted_kw(self) -> float:
        """
        Demanda contratada em kW: derivada, salvo override explícito.

        É property, e não campo, para que continue correta sob `replace()` — as
        variações de sala da Tabela 6 mudam `heat_transfer_coeff`, que ENTRA na
        condição de projeto. Um campo congelaria o valor da sala de treino e
        aplicaria silenciosamente um contrato errado às demais.
        """
        if self.demand_contracted_kw_manual is not None:
            return float(self.demand_contracted_kw_manual)
        return self.demand_sizing.contracted_kw

    def is_occupied_hour(self, hour: int) -> bool:
        return self.occupied_hour_start <= hour < self.occupied_hour_end

    def is_peak_hour(self, hour: int) -> bool:
        """Legado: usado apenas pelo modo de reprodução do manuscrito."""
        return self.peak_hours[0] <= hour <= self.peak_hours[1]

    @property
    def tariff(self):
        """Schedule tarifário resolvido (memoizado por instância)."""
        from .tariff import get_tariff
        if not hasattr(self, "_tariff_cache") or self._tariff_cache[0] != self.tariff_name:
            object.__setattr__(self, "_tariff_cache",
                               (self.tariff_name, get_tariff(self.tariff_name)))
        return self._tariff_cache[1]

    def tariff_rate(self, hour_float: float) -> float:
        """R$/kWh na hora decimal — meia hora importa na Tarifa Branca."""
        return self.tariff.rate(hour_float, with_taxes=self.tariff_with_taxes)


# --- Tabela 2: os três perfis operacionais -------------------------------
# Obtidos APENAS variando pesos da recompensa — arquitetura e hiperparâmetros
# do DQN são idênticos entre perfis. É esta a tese central do paper:
# "a modelagem da recompensa, mais que o algoritmo, é o fator determinante".
REWARD_PROFILES: Dict[str, Dict[str, float]] = {
    "Agressivo": {
        "comfort_gradient": 7.0,
        "energy_penalty_factor": 0.03,
        "action_change_penalty": -0.5,
    },
    "Equilibrado": {
        "comfort_gradient": 4.0,
        "energy_penalty_factor": 0.10,
        "action_change_penalty": -1.5,
    },
    "Passivo": {
        "comfort_gradient": 3.0,
        "energy_penalty_factor": 0.12,
        "action_change_penalty": -1.5,
    },
}

# --- Tabela 6: variações físicas para o teste de generalização ------------
ROOM_VARIATIONS: Dict[str, Dict[str, float]] = {
    "C_th=10 (leve)":            {"thermal_mass": 10.0, "heat_transfer_coeff": 0.5},
    "C_th=15 (treino)":          {"thermal_mass": 15.0, "heat_transfer_coeff": 0.5},
    "C_th=20":                   {"thermal_mass": 20.0, "heat_transfer_coeff": 0.5},
    "C_th=30 (pesada)":          {"thermal_mass": 30.0, "heat_transfer_coeff": 0.5},
    "K=0,3 (bem isolada)":       {"thermal_mass": 15.0, "heat_transfer_coeff": 0.3},
    "K=0,8 (mal isolada)":       {"thermal_mass": 15.0, "heat_transfer_coeff": 0.8},
}


def config_for_profile(profile: str, **overrides) -> ClassroomConfig:
    """Constrói a config de um perfil da Tabela 2, com overrides opcionais."""
    if profile not in REWARD_PROFILES:
        raise KeyError(f"Perfil '{profile}' inexistente. Use: {list(REWARD_PROFILES)}")
    return replace(ClassroomConfig(), **{**REWARD_PROFILES[profile], **overrides})


# --- Configuração de LABORATÓRIO de precisão ------------------------------
#
# Diferenças em relação ao perfil de sala de aula, e o porquê de cada uma:
#
#   comfort_type='quadratic'  Um laboratório rastreia SETPOINT, não faixa. Toda
#                             desvio custa, proporcionalmente ao quadrado do erro
#                             — que é o custo clássico de controle. A ablação
#                             mostrou que a quadrática pura iguala o platô com
#                             gradiente, então não há perda em adotá-la, e ela é
#                             mais defensável para precisão.
#   comfort_sensitivity alta  0,6 dá penalidade de 0,15 a ±0,5 °C: invisível
#                             frente a um conforto de 10. Para tolerância
#                             estreita a curvatura precisa ser agressiva.
#   energy_penalty_factor     Elevado ~30x: no valor do manuscrito a energia é
#                             1–4 % do sinal, então o custo é invisível para o
#                             agente e nenhum perfil troca conforto por dinheiro.
#   price_aware_penalty       A penalidade segue a tarifa real. Sem isso o agente
#                             não tem como aprender a antecipar o posto de ponta.
#   tariff_name               Tarifa Branca real da Enel (razão de ponta 2,21x).
LAB_PROFILES: Dict[str, Dict[str, float]] = {
    "Lab_Precisao": {
        # Prioriza tolerância: aceita gastar para não sair do setpoint.
        "comfort_type": "huber",
        "comfort_sensitivity": 8.0,
        "comfort_huber_delta": 0.5,
        "comfort_gradient": 0.0,
        "energy_penalty_factor": 1.0,
        "action_change_penalty": -0.5,
        "lab_tolerance": 0.5,
    },
    "Lab_Equilibrado": {
        "comfort_type": "huber",
        "comfort_sensitivity": 8.0,
        "comfort_huber_delta": 0.5,
        "comfort_gradient": 0.0,
        "energy_penalty_factor": 3.0,
        "action_change_penalty": -0.5,
        "lab_tolerance": 0.5,
    },
    "Lab_Economico": {
        # Prioriza custo: deve aprender a antecipar o posto de ponta.
        "comfort_type": "huber",
        "comfort_sensitivity": 8.0,
        "comfort_huber_delta": 0.5,
        "comfort_gradient": 0.0,
        "energy_penalty_factor": 8.0,
        "action_change_penalty": -0.5,
        "lab_tolerance": 0.5,
    },
}

LAB_BASE = {
    "tariff_name": "enel_sp_branca",
    "price_aware_penalty": True,
    "comfort_bonus": 10.0,
}

# --- Laboratório v2: bidirecional, observação enriquecida, ação contínua ----
# É esta a configuração que corresponde ao objetivo declarado (laboratório de
# precisão). As quatro mudanças em relação ao LAB_BASE são independentes e
# testáveis isoladamente.
LAB2_BASE = {
    **LAB_BASE,
    "heating_enabled": True,        # (1) sem isto 18% da madrugada é inalcançável
    "observe_time_to_peak": True,   # (2) torna a antecipação aprendível
    "observe_derivative": True,     # (3) precisão exige ver velocidade
    "observe_scaled_error": True,   # (5) 10x resolução na faixa de tolerância
    "observe_integral": True,       # (6) sem integral não se elimina offset
    # Ocupação realista: MESMA família de perturbações no treino e na avaliação.
    # Sem isto o KS entre as duas distribuições era 0,422 — o agente era
    # avaliado fora da distribuição em que treinou, e nenhum ajuste de
    # recompensa corrige isso.
    "train_occupancy_mode": "realistic",
    # Custo de controle com termo derivativo, para amortecer a oscilação que
    # separa sigma 0,198 do RL de 0,088 do PI.
    "derivative_penalty": 0.02,
    # Demanda contratada abaixo do pico do equipamento (2,93 kW): manter em
    # regime cabe (1,20 kW), o pulldown a plena carga nao. So quem antecipa
    # fica dentro -- e e a unica das mudancas testadas que favorece RL por
    # ESTRUTURA, nao por dificuldade.
    "demand_limit_enabled": True,
    # O valor contratado NAO aparece aqui: e derivado da condicao de projeto em
    # demand_sizing.py (regime de pior caso + margem de contratacao), o que da
    # ~1,32 kW. A versao anterior fixava 0,70 kW, escolhido por varredura como
    # "ponto de tensao maxima" -- e esse valor era INFACTIVEL: 72% abaixo do
    # regime de 1,204 kW, sustentando so ~17 das 45 pessoas. O que parecia
    # exigir antecipacao apenas forcava todo controlador a violar.
    #
    # A estrutura que interessa sobreviveu a derivacao, e agora nao foi
    # escolhida: o regime cabe no contrato, mas o pulldown a plena carga
    # (2,931 kW) e o aquecimento pleno (2,198 kW) ficam ~2x acima. Modular a
    # potencia continua obrigatorio; a diferenca e que agora existe ao menos uma
    # politica capaz de respeitar a restricao.
    #
    # Para varrer o valor, use `demand_contracted_kw_manual` -- explicito, e
    # portanto declarado como experimento.
    # Faixa como armazenamento, nao setpoint: sem preferencia interna, o
    # gradiente vem do custo. Um PI vai ao centro e desperdica a faixa.
    "comfort_type": "band",
    "continuous_action": True,      # (4) 4 níveis discretos limitam o ripple a ~0,09 °C
}


def config_for_lab2(profile: str, **overrides) -> ClassroomConfig:
    """Laboratório de precisão bidirecional com observação enriquecida."""
    if profile not in LAB_PROFILES:
        raise KeyError(f"Perfil '{profile}' inexistente. Use: {list(LAB_PROFILES)}")
    return replace(ClassroomConfig(),
                   **{**LAB2_BASE, **LAB_PROFILES[profile], **overrides})


# --- Eixos de ablação declarados ------------------------------------------
#
# PROBLEMA QUE ISTO RESOLVE: `LAB2_BASE` altera seis subsistemas de uma vez, e o
# comentário do próprio dicionário afirma que as mudanças são "independentes e
# testáveis isoladamente" — mas elas nunca foram testadas isoladamente, porque a
# config não oferecia um eixo para isso. O resultado prático foi que, quando o
# conjunto todo não superou o PI, ficou impossível dizer QUAL das seis mudanças
# ajudou, qual atrapalhou e qual foi neutra.
#
# Declarar os grupos torna a ablação uma varredura de uma linha em vez de um
# dicionário editado à mão a cada execução. Ver `train.py --ablate-group`.
ABLATION_GROUPS: Dict[str, Dict[str, object]] = {
    "aquecimento":   {"heating_enabled": False},
    "acao_continua": {"continuous_action": False},
    "obs_pid":       {"observe_scaled_error": False, "observe_integral": False},
    "obs_derivada":  {"observe_derivative": False},
    "obs_tarifa":    {"observe_time_to_peak": False},
    "obs_demanda":   {"observe_demand_headroom": False},
    "penal_derivada": {"derivative_penalty": 0.0},
    "restricao_demanda": {"demand_limit_enabled": False},
    "ocupacao_realista": {"train_occupancy_mode": "random_walk"},
    # Eixos vindos de Wei et al. (2017), ver features.py e env.py.
    "obs_previsao": {"observe_outdoor_forecast": False},
}


def config_ablacao(profile: str, grupo: str, **overrides) -> ClassroomConfig:
    """Config do laboratório com UM grupo desligado — o resto idêntico."""
    if grupo not in ABLATION_GROUPS:
        raise KeyError(f"grupo '{grupo}' inexistente. Use: {list(ABLATION_GROUPS)}")
    return config_for_lab2(profile, **{**ABLATION_GROUPS[grupo], **overrides})


def config_for_lab(profile: str, **overrides) -> ClassroomConfig:
    """Config de laboratório de precisão com tarifa real da Enel."""
    if profile not in LAB_PROFILES:
        raise KeyError(f"Perfil '{profile}' inexistente. Use: {list(LAB_PROFILES)}")
    return replace(ClassroomConfig(),
                   **{**LAB_BASE, **LAB_PROFILES[profile], **overrides})
