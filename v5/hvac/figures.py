# -*- coding: utf-8 -*-
"""
Figuras do artigo — desenho puro, sobre os dados de `results.py`.

Cada função recebe DADOS e devolve uma `Figure`. Nada aqui recomputa nada: é o
que garante que a figura do notebook e a figura do artigo sejam o mesmo objeto,
e que ambas concordem com a tabela ao lado.

PALETA. Slots 1–3 do tema categórico de referência, validados para deficiência de
visão de cores em todos os pares (pior par CVD ΔE 9,2; visão normal 24,0, ambos
acima dos pisos). O alívio exigido pelo aviso de contraste do aqua é atendido por
rótulos diretos e pelas tabelas equivalentes. Marcadores distintos acompanham a
cor, para que a identidade sobreviva à impressão em escala de cinza.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# O BACKEND É ESCOLHA DO CHAMADOR, não deste módulo. Forçar "Agg" aqui impedia o
# backend inline dos notebooks, e as figuras saíam vazias — script headless usa
# `matplotlib.use("Agg")` antes de importar; notebook usa `%matplotlib inline`.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, MUDO, GRADE = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3e0"

ESTILO = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": MUDO, "axes.labelcolor": TINTA, "axes.titlesize": 9.5,
    "text.color": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
}


def aplicar_estilo() -> None:
    plt.rcParams.update(ESTILO)


def _limpar(ax, eixo="y"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=eixo, color=GRADE, linewidth=0.7)


def _virgula(x, casas=1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


# =============================================================== Figura 1

def fig_decomposicao(df: pd.DataFrame):
    """
    Cascata: a decomposição de um total em contribuições atribuíveis.

    Forma escolhida pelo trabalho do dado — barras lado a lado perderiam a
    relação de acumulação, que é justamente o argumento.
    """
    aplicar_estilo()
    rotulos = [r.replace(" (", "\n(") for r in df["etapa"]]
    ganhos = df["ganho_pp"].to_numpy()
    vals = df["conforto_larga_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    cores = [MUDO, LARANJA, AZUL, AQUA]
    base = 0.0
    for i, g in enumerate(ganhos):
        ax.bar(i, g, bottom=base, color=cores[i % 4], width=0.62,
               edgecolor="white", linewidth=1.2)
        if i > 0:
            ax.plot([i - 1 + 0.31, i - 0.31], [base, base], color=MUDO,
                    linewidth=0.8, linestyle=(0, (3, 3)))
        txt = (f"{_virgula(vals[i])}%" if i == 0
               else "+0,0 pp" if abs(g) < 0.05 else f"+{_virgula(g)} pp")
        ax.text(i, base + g + 1.6, txt, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TINTA)
        # Incremento nulo não desenha barra: sem marca explícita, o leitor lê
        # "dado ausente" em vez de "ganho zero" — que é o achado.
        if abs(g) < 0.05:
            ax.plot([i - 0.31, i + 0.31], [base, base], color=cores[i % 4],
                    linewidth=3.0, solid_capstyle="butt")
        base += g

    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_ylabel("Conforto na faixa [22, 26] °C (%)")
    ax.set_ylim(0, 104)
    _limpar(ax)
    ax.annotate("aprendizado não\nacrescenta nada",
                xy=(len(ganhos) - 1, vals[-1]), xytext=(len(ganhos) - 1.45, 44),
                fontsize=8, color=TINTA2, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUDO, linewidth=0.9))
    return fig


# =============================================================== Figura 2

def fig_ablacao(resumo: pd.DataFrame, por_semente: pd.DataFrame):
    """Barras com amplitude e as três sementes sobrepostas."""
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(resumo))[::-1]

    for i, linha in resumo.iterrows():
        yy = y[i]
        destaque = linha["variante"] == "completa"
        ax.barh(yy, linha["media"], color=AZUL if destaque else MUDO,
                height=0.52, alpha=1.0 if destaque else 0.55)
        ax.plot([linha["minimo"], linha["maximo"]], [yy, yy], color=TINTA,
                linewidth=1.4, solid_capstyle="butt")
        pts = por_semente.loc[por_semente["variante"] == linha["variante"],
                              "conf_estreita_pct"]
        ax.scatter(pts, [yy] * len(pts), s=17, color=TINTA, zorder=3,
                   edgecolor="white", linewidth=0.7)
        # Coluna de valores fixa, fora da área de dados: nunca colide com as
        # sementes nem com a barra de amplitude.
        ax.text(104, yy, _virgula(linha["media"]), va="center", ha="right",
                fontsize=8.5, color=TINTA,
                fontweight="bold" if destaque else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(resumo["rotulo"], fontsize=8)
    ax.set_xlabel("Conforto na faixa estreita [23, 25] °C (%)")
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    _limpar(ax, eixo="x")
    ax.scatter([], [], s=17, color=TINTA, edgecolor="white",
               label="sementes individuais (n = 3)")
    ax.plot([], [], color=TINTA, linewidth=1.4, label="amplitude observada")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, fontsize=7.5)
    return fig


# =============================================================== Figura 3

def fig_permanencia(dados: Dict[str, object]):
    """
    ECDF: a afirmação é sobre a FRAÇÃO abaixo de um limiar, que a ECDF lê direto
    no eixo — ao contrário de um histograma.
    """
    aplicar_estilo()
    d_min = dados["d_min"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for chave, cor, marca, rot in [
        ("mole", LARANJA, "o", "Penalidade de recompensa (eq. 5)"),
        ("dura", AZUL, "s", "Restrição dura (MinDwell)"),
    ]:
        x = np.sort(dados[chave])
        yv = np.arange(1, len(x) + 1) / len(x) * 100.0
        ax.step(np.concatenate([[0], x]), np.concatenate([[0], yv]),
                where="post", color=cor, linewidth=2.0, label=rot)
        idx = np.linspace(0, len(x) - 1, 7).astype(int)
        ax.scatter(x[idx], yv[idx], color=cor, marker=marca, s=22, zorder=3,
                   edgecolor="white", linewidth=0.8)
        viol = (dados[chave] < d_min).mean() * 100.0
        ax.scatter([d_min], [viol], color=cor, s=48, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{_virgula(viol)}%", xy=(d_min, viol),
                    xytext=(d_min + 14, viol + (5 if viol > 40 else 9)),
                    fontsize=9, fontweight="bold", color=TINTA,
                    arrowprops=dict(arrowstyle="-", color=MUDO, linewidth=0.8))

    ax.axvline(d_min, color=TINTA, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text(d_min + 5, 112, f"requisito $d_{{min}}$ = {d_min:.0f} min",
            ha="left", va="center", fontsize=8, color=TINTA2)
    ax.set_xlabel("Tempo de permanência entre comutações (min)")
    ax.set_ylabel("Fração acumulada das comutações (%)")
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    _limpar(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return fig


# =============================================================== Figura 4

def fig_precisao(df: pd.DataFrame):
    """Custo × dispersão; área do marcador ∝ tempo na tolerância."""
    aplicar_estilo()
    d = df[df["controlador"] != "Antecipatório (manual)"].copy()

    def familia(nome):
        if nome.startswith("PI"):
            return "PI (clássico)", AZUL, "D"
        if nome.startswith("TD3"):
            return "TD3", LARANJA, "o"
        return "SAC", AQUA, "^"

    # Deslocamentos por ponto: com seis rótulos numa área pequena, um offset
    # único garante colisão.
    desloc = {"PI bidirecional": (30, -4), "TD3 Lab_Equilibrado": (0, -14),
              "TD3 Lab_Precisao": (0, -14), "SAC Lab_Precisao": (0, 11),
              "SAC Lab_Equilibrado": (0, -14), "TD3 Lab_Economico": (0, 11)}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for _, r in d.iterrows():
        _, cor, marca = familia(r["controlador"])
        ax.scatter(r["custo_dia"], r["sigma"],
                   s=28 + (r["na_tol_%"] - 85) * 5.2, color=cor, marker=marca,
                   alpha=0.9, zorder=3, edgecolor="white", linewidth=1.0)
        curto = (r["controlador"].replace("Lab_", "").replace(" bidirecional", "")
                 .replace("Equilibrado", "Equil.").replace("Precisao", "Precisão")
                 .replace("Economico", "Econôm."))
        ax.annotate(curto, (r["custo_dia"], r["sigma"]),
                    xytext=desloc.get(r["controlador"], (0, -14)),
                    textcoords="offset points", fontsize=7.5, color=TINTA2,
                    ha="left" if r["controlador"] == "PI bidirecional" else "center",
                    va="center")

    ax.set_xlabel("Custo diário (R$)")
    ax.set_ylabel("Dispersão da temperatura, σ (°C)")
    ax.set_xlim(14.0, 19.0)
    ax.set_ylim(0.045, 0.335)
    _limpar(ax)
    ax.grid(axis="x", color=GRADE, linewidth=0.7)
    ax.text(14.08, 0.055, "↙  melhor: mais barato e mais estável",
            fontsize=8, color=TINTA2, ha="left", va="center")
    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c,
                      markeredgecolor="white", markersize=7, label=n)
               for n, c, m in [("PI (clássico)", AZUL, "D"),
                               ("TD3", LARANJA, "o"), ("SAC", AQUA, "^")]]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    return fig


# =============================================================== Figura 5

def fig_demanda(dados: Dict[str, object]):
    """
    Origem da pressão de demanda: só cenários que partem FORA do setpoint
    excedem o contrato — e neles a violação ocorre no passo inicial, onde
    antecipar é impossível.
    """
    aplicar_estilo()
    origem = dados["origem_da_pressao"]
    limite = dados["sizing"].contracted_kw
    regime = dados["sizing"].steady_state_kw

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cores = [LARANJA if e else AZUL for e in origem["excede"]]
    ax.bar(range(len(origem)), origem["pico_kw"], color=cores, width=0.62,
           edgecolor="white", linewidth=1.0)

    ax.axhline(limite, color=TINTA, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axhline(regime, color=MUDO, linewidth=1.0, linestyle=(0, (1, 2)))

    ax.set_xticks(range(len(origem)))
    ax.set_xticklabels([f"{c}\n{t:.0f} °C" for c, t in
                        zip(origem["cenario"], origem["temp_inicial"])], fontsize=7.5)
    ax.set_ylabel("Pico da demanda medida (kW)")
    # Folga no topo para a legenda de quatro entradas não tocar as barras altas.
    ax.set_ylim(0, 3.9)
    _limpar(ax)
    # Os valores das linhas de referência vão para a LEGENDA, e não para textos
    # soltos sobre o gráfico: em qualquer altura escolhida eles cairiam sobre as
    # barras que excedem o contrato, que são justamente as mais altas.
    handles = [
        Line2D([], [], marker="s", color="none", markerfacecolor=LARANJA,
               markersize=8, label="parte fora do setpoint — excede no passo 0"),
        Line2D([], [], marker="s", color="none", markerfacecolor=AZUL,
               markersize=8, label="parte do setpoint — nunca excede"),
        Line2D([], [], color=TINTA, linewidth=1.2, linestyle=(0, (4, 3)),
               label=f"contrato derivado = {_virgula(limite, 2)} kW"),
        Line2D([], [], color=MUDO, linewidth=1.0, linestyle=(0, (1, 2)),
               label=f"regime de projeto = {_virgula(regime, 2)} kW"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5)
    return fig


# ======================================= Figuras 6-9: trajetórias e faixa

CORES_CTRL = {"DQN Equilibrado": LARANJA, "PI sintonizado": AZUL,
              "Termostato (zm = 0)": AQUA, "Termostato (zm = 1 °C)": AQUA,
              "DQN": LARANJA, "Termostato (zm=1 °C)": AQUA}
MARCAS_CTRL = {"DQN Equilibrado": "o", "PI sintonizado": "D",
               "Termostato (zm = 0)": "^", "Termostato (zm = 1 °C)": "^",
               "DQN": "o", "Termostato (zm=1 °C)": "^"}


def fig_trajetorias_grade(dados: Dict[str, object], tol: Optional[float] = None):
    """
    Grade 3×3 dos cenários: temperatura ao longo do dia.

    Pequenos múltiplos, e não um gráfico único com 27 curvas: a comparação que
    interessa é entre controladores DENTRO de cada cenário, e é ela que o painel
    isola. A faixa alvo aparece sombreada em todos, dando uma referência visual
    comum — quem passa mais tempo dentro do sombreado vence.
    """
    aplicar_estilo()
    series, cfg = dados["series"], dados["config"]
    ids = list(series)
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 6.4), sharex=True, sharey=True)

    lo = cfg.ideal_temp - tol if tol else cfg.temp_comfort_min
    hi = cfg.ideal_temp + tol if tol else cfg.temp_comfort_max

    for k, cid in enumerate(ids):
        ax = axes[k // 3][k % 3]
        bloco = series[cid]
        cen = bloco["cenario"]
        ax.axhspan(lo, hi, color=AQUA, alpha=0.13, lw=0)
        ax.axhline(cfg.ideal_temp, color=MUDO, lw=0.8, ls=(0, (1, 2)))
        for nome in dados["controladores"]:
            df = bloco[nome]
            ax.plot(df["hour_float"] % 24 if df["hour_float"].max() < 24
                    else np.arange(len(df)) * cfg.dt * 2,
                    df["temperature"], color=CORES_CTRL[nome], lw=1.5,
                    label=nome if k == 0 else None)
        ax.set_title(f"{cid} — {cen['condition']} + {cen['occupancy_label']}",
                     fontsize=8, color=TINTA)
        ax.set_ylim(15, 39)
        _limpar(ax)
        if k % 3 == 0:
            ax.set_ylabel("T (°C)", fontsize=8)
        if k // 3 == 2:
            ax.set_xlabel("horas desde o início", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.13))
    labels.append(f"faixa alvo [{lo:.1f}, {hi:.1f}] °C")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.045),
               ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_trajetoria_detalhe(dados: Dict[str, object], cid: str,
                           tol: Optional[float] = None):
    """
    Um cenário em detalhe: temperatura em cima, ação do equipamento embaixo.

    Eixos empilhados e compartilhando o tempo — nunca duas escalas no mesmo eixo.
    O painel inferior é o que explica o superior: mostra COMO cada controlador
    chega ao resultado.
    """
    aplicar_estilo()
    series, cfg = dados["series"], dados["config"]
    bloco = series[cid]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.35, 1]})

    lo = cfg.ideal_temp - tol if tol else cfg.temp_comfort_min
    hi = cfg.ideal_temp + tol if tol else cfg.temp_comfort_max
    a1.axhspan(lo, hi, color=AQUA, alpha=0.13, lw=0,
               label=f"faixa alvo [{lo:.1f}, {hi:.1f}] °C")
    a1.axhline(cfg.ideal_temp, color=MUDO, lw=0.8, ls=(0, (1, 2)))

    for nome in dados["controladores"]:
        df = bloco[nome]
        t = np.arange(len(df)) * cfg.dt * 2
        a1.plot(t, df["temperature"], color=CORES_CTRL[nome], lw=1.8, label=nome)
        # Degrau para a carga: o comando é constante entre decisões, e
        # interpolar sugeriria uma modulação contínua que não existe.
        a2.step(t, df["load"], where="post", color=CORES_CTRL[nome], lw=1.5,
                alpha=0.95)

    cen = bloco["cenario"]
    a1.set_title(f"{cid} — {cen['condition']} + {cen['occupancy_label']} "
                 f"(início {cen['hour']}h, {cen['occupancy']} ocupantes)",
                 fontsize=9)
    a1.set_ylabel("Temperatura (°C)")
    a1.set_ylim(15, 39)
    _limpar(a1)
    a1.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=2)

    a2.set_ylabel("Carga do equipamento")
    a2.set_xlabel("horas desde o início do episódio")
    a2.set_ylim(-0.05, 1.08)
    a2.set_yticks([0.0, 0.25, 0.55, 1.0])
    a2.set_yticklabels(["OFF", "LOW", "MED", "HIGH"], fontsize=8)
    _limpar(a2)
    fig.tight_layout()
    return fig


def fig_sensibilidade_largura(df: pd.DataFrame):
    """Tempo na tolerância conforme a régua aperta (sem retreino)."""
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for nome, g in df.groupby("controlador", sort=False):
        g = g.sort_values("tolerancia", ascending=False)
        ax.plot(g["tolerancia"], g["na_tolerancia_pct"],
                color=CORES_CTRL.get(nome, MUDO),
                marker=MARCAS_CTRL.get(nome, "o"), markersize=5, lw=1.8,
                markeredgecolor="white", markeredgewidth=0.8, label=nome)
    ax.invert_xaxis()      # da régua frouxa para a apertada
    ax.set_xlabel("Meia-largura da faixa alvo, ± °C  (mais exigente →)")
    ax.set_ylabel("Tempo dentro da faixa (%)")
    ax.set_ylim(0, 104)
    _limpar(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    return fig


def fig_faixa_estreita(df: pd.DataFrame):
    """
    Retreino por largura: PI re-sintonizado contra DQN retreinado.

    Barras agrupadas com as sementes do DQN sobrepostas — com n = 3 a média
    isolada esconderia a dispersão, que aqui é a informação decisiva.
    """
    aplicar_estilo()
    tols = sorted(df["tolerancia"].unique(), reverse=True)
    ctrls = ["PI sintonizado", "DQN", "Termostato (zm=1 °C)"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    largura = 0.26

    for j, c in enumerate(ctrls):
        xs, ys = [], []
        for i, t in enumerate(tols):
            sub = df[(df["tolerancia"] == t) & (df["controlador"] == c)]
            if sub.empty:
                continue
            x = i + (j - 1) * largura
            xs.append(x)
            ys.append(sub["na_tolerancia_pct"].mean())
            if c == "DQN" and len(sub) > 1:
                ax.scatter([x] * len(sub), sub["na_tolerancia_pct"], s=15,
                           color=TINTA, zorder=4, edgecolor="white", linewidth=0.6)
        ax.bar(xs, ys, width=largura, color=CORES_CTRL.get(c, MUDO),
               label=c, edgecolor="white", linewidth=1.0,
               alpha=1.0 if c != "Termostato (zm=1 °C)" else 0.75)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.6, _virgula(y), ha="center", fontsize=7.5,
                    color=TINTA)

    ax.set_xticks(range(len(tols)))
    ax.set_xticklabels([f"±{_virgula(t, 1)} °C" for t in tols])
    ax.set_xlabel("Faixa alvo para a qual AMBOS foram preparados")
    ax.set_ylabel("Tempo dentro da faixa (%)")
    ax.set_ylim(0, 112)
    _limpar(ax)
    ax.scatter([], [], s=15, color=TINTA, edgecolor="white",
               label="sementes do DQN")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=2)
    return fig
