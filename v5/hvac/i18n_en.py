# -*- coding: utf-8 -*-
"""
Camada inglês para figuras e rótulos de dados.

POR QUE TRADUZIR DEPOIS, E NÃO NA ORIGEM. `figures.py` desenha; se cada função
recebesse um idioma, haveria duas implementações de desenho convivendo e elas
divergiriam na primeira revisão — que é exatamente o defeito que a v5 eliminou
das tabelas. Aqui a figura é construída UMA vez, pelas mesmas funções que
alimentam os cadernos, e só então tem seus textos substituídos. Dados, escalas,
posições e cores são necessariamente idênticos entre a versão pt-BR e a en-US:
a única diferença possível é o texto.

POR QUE FALHAR EM VEZ DE DEIXAR PASSAR. Um rótulo em português esquecido dentro
de uma figura de submissão internacional é um erro que ninguém percebe até o
parecer. Por isso não há tradução "por aproximação": todo texto precisa constar
do glossário, e o que não constar levanta `TraducaoAusente` com a string exata a
acrescentar. Trocar isso por um `dict.get(s, s)` reintroduz a falha silenciosa.

POR QUE A CHAVE IGNORA NÚMEROS. Boa parte dos rótulos embute valor calculado
("contrato derivado = 1,32 kW"). Chavear pela string literal quebraria o
glossário a cada reexecução que mudasse um dígito. A chave normaliza todo número
para "#", e o modelo em inglês recebe os números de volta na ordem original,
já convertidos para o separador decimal inglês.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

# Número com separador decimal opcional, vírgula ou ponto. Não captura o sinal:
# o "−" unicode dos eixos e o "+" dos incrementos fazem parte do modelo.
_RX_NUM = re.compile(r"\d+(?:[.,]\d+)?")


class TraducaoAusente(KeyError):
    """Texto encontrado numa figura sem entrada correspondente no glossário."""


def _chave(texto: str) -> str:
    return _RX_NUM.sub("#", texto)


def _numero_en(texto: str) -> str:
    """Vírgula decimal brasileira para ponto decimal inglês."""
    return texto.replace(",", ".")


# ---------------------------------------------------------------- glossário
#
# Chave: texto em pt-BR com os números normalizados para "#".
# Valor: mesmo texto em en-US, com "#" nas mesmas posições e na mesma ordem.
#
# A verificação de que a contagem de "#" bate entre chave e valor é feita na
# carga do módulo (ver `_conferir_glossario`), porque um "#" a mais ou a menos
# produziria um rótulo truncado, não um erro.

GLOSSARIO: Dict[str, str] = {
    # -------------------------------------------------- puramente numéricos
    "#": "#",
    "−#": "−#",
    "-#": "-#",
    "+#": "+#",
    "#%": "#%",
    "-#%": "-#%",
    "−#%": "−#%",
    "+# pp": "+# pp",
    "-# pp": "-# pp",
    "−# pp": "−# pp",
    "# h": "# h",
    "# min": "# min",
    "#k": "#k",
    "±#": "±#",
    "±# °C": "±# °C",

    # ------------------------------------------------- BOPTEST (terceiros)
    "dia de pico de resfriamento": "peak cooling day",
    "dia típico de resfriamento": "typical cooling day",
    "PI re-sintonizado\n(emulador)": "PI retuned\n(emulator)",
    "PI\n(ganhos locais, congelados)": "PI\n(local gains, frozen)",
    "PI re-sintonizado (emulador)": "PI retuned (emulator)",
    "PI (ganhos locais, congelados)": "PI (local gains, frozen)",
    "Termostato zm=# °C": "Thermostat db=# °C",
    "Termostato zm=#\n(baseline)": "Thermostat db=#\n(baseline)",
    "Termostato zm=# (baseline)": "Thermostat db=# (baseline)",
    "re-sintonia\n+# pp": "retuning\n+# pp",
    "Ganho proporcional $K_p$": "Proportional gain $K_p$",
    "Ganho integral $K_i$": "Integral gain $K_i$",
    "ganhos do artigo\n(planta local)": "gains from the paper\n(local plant)",
    "ótimo nesta planta": "optimum for this plant",

    # ------------------------------------------------------- eixos e títulos
    "Conforto na faixa [#, #] °C (%)": "Time in the [#, #] °C band (%)",
    "Conforto na faixa estreita [#, #] °C (%)":
        "Time in the narrow [#, #] °C band (%)",
    "Conforto na faixa estreita (%)": "Time in the narrow band (%)",
    "Tempo de permanência entre comutações (min)":
        "Dwell time between switches (min)",
    "Fração acumulada das comutações (%)": "Cumulative fraction of switches (%)",
    "Custo diário (R$)": "Daily cost (BRL)",
    "Dispersão da temperatura, σ (°C)": "Temperature dispersion, σ (°C)",
    "Pico da demanda medida (kW)": "Measured peak demand (kW)",
    "Faixa alvo para a qual AMBOS foram preparados":
        "Target band BOTH controllers were prepared for",
    "Tempo dentro da faixa (%)": "Time within the band (%)",
    "passos de treino (mil)": "training steps (thousands)",
    "Fração do tempo em cada nível de potência (%)":
        "Fraction of time at each power level (%)",
    "Consumo diário decomposto por nível acionado (kWh/dia)":
        "Daily energy decomposed by commanded level (kWh/day)",
    "Ocupação do cenário": "Scenario occupancy",
    "Diferença vs PI sintonizado (pp)": "Difference vs tuned PI (pp)",
    "Energia (kWh/dia)": "Energy (kWh/day)",
    "Tempo dentro de ±# °C (%)": "Time within ±# °C (%)",
    "Tempo desde o início do episódio (h)": "Time from episode start (h)",

    # ------------------------------------------------- rótulos de categoria
    "Termostato\n(zona morta = #)": "Thermostat\n(deadband = #)",
    "+ histerese\n(zona morta = # °C)": "+ hysteresis\n(deadband = # °C)",
    "PI sintonizado\n(Kp=#; Ki=#)": "Tuned PI\n(Kp=#; Ki=#)",
    "DQN\n(#k passos)": "DQN\n(#k steps)",
    "Termostato (zm = #)": "Thermostat (db = #)",
    "Termostato (zm = # °C)": "Thermostat (db = # °C)",
    "Termostato (zm=# °C)": "Thermostat (db=# °C)",
    "Termostato\n(zm=#)": "Thermostat\n(db=#)",
    "Termostato\n(zm=# °C)": "Thermostat\n(db=# °C)",
    "Termostato": "Thermostat",
    "PI sintonizado": "Tuned PI",
    "PI (clássico)": "PI (classical)",
    "PI bidirecional": "Bidirectional PI",
    "PI": "PI",
    "DQN": "DQN",
    # "TD3" tem um dígito, então a chave normalizada é "TD#" — ver `_chave`.
    "TD#": "TD#",
    "SAC": "SAC",
    "DQN Agressivo": "DQN Aggressive",
    "DQN Equilibrado": "DQN Balanced",
    "DQN Passivo": "DQN Passive",
    "SAC Equilibrado": "SAC Balanced",
    "TD# Equil.": "TD# Balanced",
    "SAC Equil.": "SAC Balanced",
    "TD# Precisão": "TD# Precision",
    "SAC Precisão": "SAC Precision",
    "TD# Econôm.": "TD# Economy",
    "OFF": "OFF",
    "LOW (COP #)": "LOW (COP #)",
    "MEDIUM (COP #)": "MEDIUM (COP #)",
    "HIGH (COP #)": "HIGH (COP #)",
    "C#\n# °C": "C#\n# °C",
    "#–#\n(vazia)": "#–#\n(empty)",
    "#–#\n(parcial)": "#–#\n(partial)",
    "#–#\n(cheia)": "#–#\n(full)",

    # ---------------------------------------------- legendas e anotações
    "aprendizado não\nacrescenta nada": "learning adds\nnothing",
    "sementes individuais (n = #)": "individual seeds (n = #)",
    "amplitude observada": "observed range",
    "sementes do DQN": "DQN seeds",
    "requisito $d_{min}$ = # min": "requirement $d_{min}$ = # min",
    "Penalidade de recompensa": "Reward penalty",
    "Restrição dura (MinDwell)": "Hard constraint (MinDwell)",
    "↙  melhor: mais barato e mais estável":
        "↙  better: cheaper and more stable",
    "↖  melhor: mais preciso\n     e mais barato":
        "↖  better: more precise\n     and cheaper",
    "parte fora do setpoint — excede no passo #":
        "starts off setpoint — exceeds at step #",
    "parte do setpoint — nunca excede": "starts at setpoint — never exceeds",
    "contrato derivado = # kW": "derived contract = # kW",
    "regime de projeto = # kW": "design steady state = # kW",
    "faixa alvo ±# °C": "target band ±# °C",
    "DQN (média de # sementes)": "DQN (mean of # seeds)",
    "PI sintonizado (não aprende)": "Tuned PI (does not learn)",
    "# kWh/dia": "# kWh/day",
    "entrar na faixa": "first entry into band",
    "não sair mais (acomodação)": "no further exit (settling)",
    "DQN Agressivo\n(# % em potência máxima)":
        "DQN Aggressive\n(# % at maximum power)",
    "DQN Equilibrado\n(# % em potência máxima)":
        "DQN Balanced\n(# % at maximum power)",
    "DQN Passivo\n(# % em potência máxima)":
        "DQN Passive\n(# % at maximum power)",
    "SAC Equilibrado\n(# % em potência máxima)":
        "SAC Balanced\n(# % at maximum power)",
    "PI sintonizado\n(# % em potência máxima)":
        "Tuned PI\n(# % at maximum power)",
    "Termostato (zm = #)\n(# % em potência máxima)":
        "Thermostat (db = #)\n(# % at maximum power)",
    "Termostato (zm = # °C)\n(# % em potência máxima)":
        "Thermostat (db = # °C)\n(# % at maximum power)",

    # ------------------------------------------------------- diagrama do ciclo
    "Agente": "Agent",
    "política π(a | s)": "policy π(a | s)",
    "Ambiente": "Environment",
    "sala + equipamento": "room + equipment",
    "ação  $a_t$": "action  $a_t$",
    "nível de potência acionado": "commanded power level",
    "estado  $s_{t+#}$": "state  $s_{t+#}$",
    "recompensa  $r_{t+#}$": "reward  $r_{t+#}$",
    "temperatura, ocupação e hora": "temperature, occupancy and hour",
    "aprende a maximizar\no retorno acumulado":
        "learns to maximize\nthe cumulative return",
    "evolui segundo o\nbalanço térmico da sala":
        "evolves per the room's\nthermal balance",

    # ------------------------------------------- variantes da ablação
    "completa (proposta)": "complete (proposed)",
    "convencional (quadrática pura)": "conventional (pure quadratic)",
    "sem anti-short-cycling": "no anti-short-cycling",
    "sem penalidade de frio": "no cold penalty",
    "quadrática pura": "pure quadratic",
    "sem penalidade de troca": "no switching penalty",
    "degraus (legado)": "step reward (legacy)",
    "sem gradiente interno": "no inner gradient",
    "referência": "reference",
    "desprezível": "negligible",
    "pequena": "small",
    "média": "medium",
    "grande": "large",

    # -------------------------------- parâmetros derivados e orçamento
    "por parâmetro, isoladamente": "per parameter, in isolation",
    "combinados": "combined",
    "referência\n(B=#, k=#)": "reference\n(B=#, k=#)",
    "derivados\n(B=#, k=#)": "derived\n(B=#, k=#)",
    "referência (B=#, k=#)": "reference (B=#, k=#)",
    "derivados (B=#, k=#)": "derived (B=#, k=#)",
    "inferidos (B=#, k=#)": "inferred (B=#, k=#)",
    "inferidos": "inferred",
    "derivados": "derived",
    "# mil passos\n(orçamento da calibração)": "#k steps\n(calibration budget)",
    "# mil passos\n(orçamento do protocolo)": "#k steps\n(protocol budget)",
    "B = #": "B = #",
    "k = #": "k = #",
    "rho = -#": "ρ = −#",
    "frio = -#": "cold = −#",

    # ------------------------------------------- decomposição da vantagem
    "Termostato (zona morta = #)": "Thermostat (deadband = #)",
    "+ histerese (zona morta = # °C)": "+ hysteresis (deadband = # °C)",
    "PI sintonizado (Kp=#; Ki=#)": "Tuned PI (Kp=#; Ki=#)",
    "DQN (# passos)": "DQN (# steps)",
    "DQN (#k passos)": "DQN (#k steps)",
    "baseline do manuscrito": "baseline of the audited manuscript",
    "configuração do baseline": "baseline configuration",
    "controle clássico": "classical control",
    "aprendizado": "learning",

    # -------------------------------------------- comparação de controladores
    "DQN Agressivo (#k)": "DQN Aggressive (#k)",
    "DQN Equilibrado (#k)": "DQN Balanced (#k)",
    "DQN Passivo (#k)": "DQN Passive (#k)",
    "SAC Equilibrado (#k)": "SAC Balanced (#k)",
    "Termostato zona morta = # °C": "Thermostat, deadband = # °C",
    "Termostato zona morta = #": "Thermostat, deadband = #",

    # ------------------------------------------------ laboratório de precisão
    "TD# Lab_Equilibrado": "TD# Balanced",
    "TD# Lab_Precisao": "TD# Precision",
    "SAC Lab_Precisao": "SAC Precision",
    "SAC Lab_Equilibrado": "SAC Balanced",
    "TD# Lab_Economico": "TD# Economy",
    "Antecipatório (manual)": "Anticipatory (hand-coded)",

    # ------------------------------------------------------- hiperparâmetros
    "taxa de aprendizado": "learning rate",
    "tamanho do lote": "batch size",
    "fator de desconto γ": "discount factor γ",
    "capacidade do replay buffer": "replay buffer capacity",
    "passos antes do primeiro ajuste": "steps before first update",
    "passos de gradiente por atualização": "gradient steps per update",
    "intervalo de atualização da rede-alvo": "target network update interval",
    "coeficiente de atualização suave τ": "soft update coefficient τ",
    "ε inicial": "initial ε",
    "ε final": "final ε",
    "fração do treino em decaimento de ε": "fraction of training in ε decay",
    "recorte de gradiente": "gradient clipping",
    "frequência de treino (passos)": "train frequency (steps)",
    "arquitetura (MLP)": "architecture (MLP)",
    "parâmetros treináveis": "trainable parameters",
}


def _conferir_glossario() -> None:
    """Chave e modelo precisam ter o mesmo número de lacunas."""
    ruins = [(k, val) for k, val in GLOSSARIO.items()
             if k.count("#") != val.count("#")]
    if ruins:
        raise AssertionError(
            "entradas do glossário com contagem de '#' divergente entre chave e "
            f"modelo: {ruins}")


_conferir_glossario()


def t(texto: str) -> str:
    """
    Traduz um texto único. Levanta `TraducaoAusente` se não houver entrada.

    Recebe também rótulos vindos de DataFrame, que é o caminho pelo qual as
    tabelas do artigo e as figuras compartilham o mesmo glossário.
    """
    if texto is None or not str(texto).strip():
        return texto
    texto = str(texto)
    chave = _chave(texto)
    modelo = GLOSSARIO.get(chave)
    if modelo is None:
        raise TraducaoAusente(
            f"sem tradução para {texto!r} (chave {chave!r}); acrescente a "
            "entrada em hvac/i18n_en.py")
    numeros = [_numero_en(n) for n in _RX_NUM.findall(texto)]
    partes = modelo.split("#")
    saida = partes[0]
    for numero, parte in zip(numeros, partes[1:]):
        saida += numero + parte
    return saida


def ts(textos: Iterable[str]) -> List[str]:
    return [t(x) for x in textos]


# --------------------------------------------------------- tradução de figura

def _traduzir_ticks(ax, eixo: str) -> None:
    """
    Reescreve os rótulos de um eixo apenas se algum deles mudar.

    A guarda importa: fixar os rótulos converte o localizador em `FixedLocator`,
    o que congela o eixo. Fazer isso num eixo puramente numérico não traz
    benefício e traz risco de o eixo deixar de acompanhar mudanças de limite.
    """
    obter = ax.get_xticklabels if eixo == "x" else ax.get_yticklabels
    originais = [obj.get_text() for obj in obter()]
    if not originais:
        return
    traduzidos = [t(s) if s.strip() else s for s in originais]
    if traduzidos == originais:
        return
    posicoes = ax.get_xticks() if eixo == "x" else ax.get_yticks()
    if eixo == "x":
        ax.set_xticks(posicoes)
        ax.set_xticklabels(traduzidos, fontsize=obter()[0].get_fontsize())
    else:
        ax.set_yticks(posicoes)
        ax.set_yticklabels(traduzidos, fontsize=obter()[0].get_fontsize())


def traduzir_figura(fig):
    """
    Substitui, no lugar, todo texto de `fig` pelo equivalente em inglês.

    Devolve a mesma `Figure`, para encadear com `savefig`. Qualquer texto fora
    do glossário interrompe a geração — ver a nota no topo do módulo.
    """
    # Os rótulos de escala só existem depois do primeiro desenho; ler antes
    # devolveria a lista vazia e o eixo sairia em português.
    fig.canvas.draw()

    for ax in fig.axes:
        for obter, definir in ((ax.get_title, ax.set_title),
                               (ax.get_xlabel, ax.set_xlabel),
                               (ax.get_ylabel, ax.set_ylabel)):
            atual = obter()
            if atual.strip():
                definir(t(atual))
        _traduzir_ticks(ax, "x")
        _traduzir_ticks(ax, "y")
        for obj in ax.texts:
            obj.set_text(t(obj.get_text()))
        legenda = ax.get_legend()
        if legenda is not None:
            titulo = legenda.get_title()
            if titulo.get_text().strip():
                titulo.set_text(t(titulo.get_text()))
            for obj in legenda.get_texts():
                obj.set_text(t(obj.get_text()))

    for legenda in getattr(fig, "legends", []):
        titulo = legenda.get_title()
        if titulo.get_text().strip():
            titulo.set_text(t(titulo.get_text()))
        for obj in legenda.get_texts():
            obj.set_text(t(obj.get_text()))

    for obj in fig.texts:
        obj.set_text(t(obj.get_text()))

    return fig
