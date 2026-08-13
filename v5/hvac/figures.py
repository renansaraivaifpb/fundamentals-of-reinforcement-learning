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
        ("mole", LARANJA, "o", "Penalidade de recompensa"),
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


# ============================ Figuras 10-11: inspiradas em Yuan et al. (2021)

def fig_consumo_decomposto(dados: Dict[str, pd.DataFrame], cop: Optional[Dict] = None):
    """
    Consumo por nível de potência — análogo da Fig. 11 de Yuan et al., que
    decompõe o consumo por item do sistema.

    Barras EMPILHADAS porque o dado é uma composição de um total: a pergunta é
    "de que se compõe o consumo diário de cada controlador", e o total importa
    tanto quanto as partes. Empilhar responde as duas de uma vez.
    """
    aplicar_estilo()
    piv = dados["por_nivel"].pivot(index="controlador", columns="nivel",
                                   values="kwh_dia")
    ordem_niveis = [n for n in ("LOW", "MEDIUM", "HIGH") if n in piv.columns]
    piv = piv[ordem_niveis]
    piv = piv.loc[piv.sum(axis=1).sort_values().index]

    # Cor por EFICIÊNCIA do nível, não por posição: o argumento da figura é que
    # o nível de melhor COP é justamente o que o agente aprendido descarta.
    cores = {"LOW": AZUL, "MEDIUM": AQUA, "HIGH": LARANJA}
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    esq = np.zeros(len(piv))
    for n in ordem_niveis:
        v = piv[n].to_numpy()
        rot = f"{n} (COP {cop[n]:.2f})" if cop and n in cop else n
        ax.barh(range(len(piv)), v, left=esq, color=cores.get(n, MUDO),
                height=0.55, label=rot, edgecolor="white", linewidth=1.2)
        for i, (x, w) in enumerate(zip(esq, v)):
            if w > 0.7:                      # rotula só o que cabe
                ax.text(x + w / 2, i, _virgula(w), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        esq += v
    for i, tot in enumerate(esq):
        ax.text(tot + 0.18, i, f"{_virgula(tot, 2)} kWh/dia", va="center",
                fontsize=8.5, color=TINTA)

    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8.5)
    ax.set_xlabel("Consumo diário decomposto por nível acionado (kWh/dia)")
    ax.set_xlim(0, esq.max() * 1.28)
    _limpar(ax, eixo="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=8)
    return fig


def fig_curva_aprendizado(df: pd.DataFrame):
    """
    Desempenho contra orçamento de treino — análogo das Figs. 9-10 de Yuan et
    al., que traçam custo e desconforto ano a ano.

    Responde à objeção "o DQN perde porque treinou pouco?". A linha do PI é
    horizontal por construção: ele não aprende, e é isso que a figura precisa
    deixar visualmente óbvio.
    """
    aplicar_estilo()
    tols = sorted(df["tolerancia"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(tols), figsize=(3.6 * len(tols), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, tol in zip(axes, tols):
        sub = df[df["tolerancia"] == tol]
        pi = sub[sub["controlador"] == "PI sintonizado"]["na_tolerancia_pct"]
        dqn = sub[sub["controlador"] == "DQN"]

        for seed, g in dqn.groupby("seed"):
            g = g.sort_values("passos")
            ax.plot(g["passos"] / 1000, g["na_tolerancia_pct"], color=LARANJA,
                    lw=1.0, alpha=0.45)
        med = dqn.groupby("passos")["na_tolerancia_pct"].mean().sort_index()
        ax.plot(med.index / 1000, med.to_numpy(), color=LARANJA, lw=2.2,
                marker="o", markersize=3.5, markeredgecolor="white",
                markeredgewidth=0.6, label="DQN (média de 3 sementes)")
        if len(pi):
            ax.axhline(pi.iloc[0], color=AZUL, lw=2.0, ls=(0, (5, 2)),
                       label="PI sintonizado (não aprende)")
        ax.set_title(f"faixa alvo ±{_virgula(tol, 1)} °C", fontsize=9)
        ax.set_xlabel("passos de treino (mil)")
        _limpar(ax)
    axes[0].set_ylabel("Tempo dentro da faixa (%)")
    axes[0].set_ylim(0, 104)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


# =================== Figuras 12-14: níveis de potência e cenários aleatórios

def fig_uso_dos_niveis(df: pd.DataFrame, cop: Optional[Dict] = None):
    """
    Fração do tempo em cada nível — o descarte de MEDIUM pelos agentes DQN.

    Barras empilhadas somando 100 %: a pergunta é como cada controlador REPARTE
    o dia entre os níveis, e a ausência de um segmento é o achado.
    """
    aplicar_estilo()
    ordem = ["OFF", "LOW", "MEDIUM", "HIGH"]
    cores = {"OFF": "#c9c9c4", "LOW": AZUL, "MEDIUM": AQUA, "HIGH": LARANJA}
    d = df.set_index("controlador")[ordem]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    esq = np.zeros(len(d))
    y = np.arange(len(d))[::-1]
    for n in ordem:
        v = d[n].to_numpy()
        rot = f"{n} (COP {cop[n]:.2f})" if cop and n in cop else n
        ax.barh(y, v, left=esq, color=cores[n], height=0.58, label=rot,
                edgecolor="white", linewidth=1.2)
        for i, (x, w) in enumerate(zip(esq, v)):
            if w > 6:
                ax.text(x + w / 2, y[i], _virgula(w), ha="center", va="center",
                        fontsize=8, color="white" if n != "OFF" else TINTA2,
                        fontweight="bold")
        esq += v
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=8.5)
    ax.set_xlabel("Fração do tempo em cada nível de potência (%)")
    ax.set_xlim(0, 100)
    _limpar(ax, eixo="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              frameon=False, fontsize=8)
    return fig


def fig_divergencia_por_condicao(df: pd.DataFrame, metrica: str = "na_tolerancia_pct",
                                 referencia: str = "PI sintonizado"):
    """
    ONDE cada agente se afasta da referência, por faixa de ocupação.

    A média sobre todos os cenários esconde que a falha é concentrada: agrupar
    por condição mostra que o déficit vive quase todo em salas vazias.
    """
    aplicar_estilo()
    cond = df.drop_duplicates("cenario").set_index("cenario")[["occupancy"]]
    piv = df.pivot(index="cenario", columns="controlador", values=metrica).join(cond)
    piv["faixa"] = pd.cut(piv["occupancy"], [-1, 10, 25, 45],
                          labels=["0–10\n(vazia)", "11–25\n(parcial)", "26–45\n(cheia)"])

    alvos = [c for c in ("DQN Agressivo", "DQN Equilibrado", "DQN Passivo")
             if c in piv.columns]
    cores = {"DQN Agressivo": LARANJA, "DQN Equilibrado": AZUL, "DQN Passivo": AQUA}
    marcas = {"DQN Agressivo": "o", "DQN Equilibrado": "s", "DQN Passivo": "^"}

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    faixas = list(piv["faixa"].cat.categories)
    x = np.arange(len(faixas))
    largura = 0.26
    for j, a in enumerate(alvos):
        g = (piv[a] - piv[referencia]).groupby(piv["faixa"], observed=True).mean()
        vals = [g.get(f, np.nan) for f in faixas]
        ax.bar(x + (j - 1) * largura, vals, width=largura, color=cores[a],
               label=a, edgecolor="white", linewidth=1.0)
        for xi, v in zip(x + (j - 1) * largura, vals):
            if not np.isnan(v):
                ax.text(xi, v + (0.8 if v >= 0 else -2.2), _virgula(v),
                        ha="center", fontsize=7.5, color=TINTA)
    ax.axhline(0, color=TINTA, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(faixas, fontsize=8)
    # Folga para os rótulos das barras negativas, que sem isso saem da área de
    # plotagem e são cortados no PNG.
    todos = [v for a in alvos
             for v in (piv[a] - piv[referencia]).groupby(
                 piv["faixa"], observed=True).mean().values if not np.isnan(v)]
    lo, hi = min(todos), max(todos)
    ax.set_ylim(lo - 0.22 * (hi - lo) - 3.0, hi + 0.20 * (hi - lo) + 2.0)
    ax.set_ylabel(f"Diferença vs {referencia} (pp)")
    ax.set_xlabel("Ocupação do cenário")
    _limpar(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    return fig


def fig_pareto_aleatorios(df: pd.DataFrame):
    """
    Conforto contra energia nos cenários aleatórios — a fronteira de Pareto.

    Cada perfil DQN falha de um jeito: um compra conforto com energia, os outros
    igualam a energia perdendo conforto. Nenhum ocupa o canto.
    """
    aplicar_estilo()
    g = df.groupby("controlador").agg(tol=("na_tolerancia_pct", "mean"),
                                      kwh=("energia_kwh", "mean"))
    estilo = {
        "PI sintonizado": (AZUL, "D", 90), "DQN Agressivo": (LARANJA, "o", 70),
        "DQN Equilibrado": (LARANJA, "s", 70), "DQN Passivo": (LARANJA, "^", 70),
        "SAC Equilibrado": (AQUA, "v", 70), "Termostato (zm = 1 °C)": (MUDO, "P", 55),
        "Termostato (zm = 0)": (MUDO, "X", 55),
    }
    # Deslocamentos ajustados após inspecionar o PNG: PI, Equilibrado e Passivo
    # ficam a poucos décimos de kWh uns dos outros e colidem com offset único.
    desloc = {"PI sintonizado": (-6, 16), "DQN Agressivo": (0, 15),
              "DQN Equilibrado": (40, 6), "DQN Passivo": (-42, -6),
              "SAC Equilibrado": (0, -15), "Termostato (zm = 1 °C)": (0, 15),
              "Termostato (zm = 0)": (0, 15)}

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for nome, r in g.iterrows():
        cor, marca, tam = estilo.get(nome, (MUDO, "o", 60))
        ax.scatter(r["kwh"], r["tol"], s=tam, color=cor, marker=marca, zorder=3,
                   edgecolor="white", linewidth=1.1)
        ax.annotate(nome.replace(" (zm = ", "\n(zm="), (r["kwh"], r["tol"]),
                    xytext=desloc.get(nome, (0, 12)), textcoords="offset points",
                    fontsize=7.5, color=TINTA2, ha="center", va="center")
    ax.set_xlabel("Energia (kWh/dia)")
    ax.set_ylabel("Tempo dentro de ±0,5 °C (%)")
    ax.set_ylim(-4, 96)
    _limpar(ax)
    ax.grid(axis="x", color=GRADE, linewidth=0.7)
    ax.set_xlim(7.2, 12.9)
    ax.text(12.7, 12, "↖  melhor: mais preciso\n     e mais barato", fontsize=8,
            color=TINTA2, ha="right")
    return fig


def fig_transitorio(df: pd.DataFrame):
    """
    Velocidade de resposta: entrar na faixa contra permanecer nela.

    Duas barras por controlador porque as duas grandezas contam histórias
    diferentes, e a DISTÂNCIA entre elas é o achado: o termostato entra cedo e
    demora horas para parar de sair, porque estaciona no teto e oscila em torno
    dele. Uma barra só esconderia isso.
    """
    aplicar_estilo()
    g = df.groupby("controlador").agg(
        entrada=("entrada_h", "mean"), acomoda=("acomodacao_h", "mean"),
        satur=("saturacao_pct", "mean")).sort_values("entrada")

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(g))[::-1]
    alt = 0.34
    ax.barh(y + alt / 2, g["entrada"], height=alt, color=AZUL,
            label="entrar na faixa", edgecolor="white", linewidth=1.0)
    ax.barh(y - alt / 2, g["acomoda"], height=alt, color=LARANJA,
            label="não sair mais (acomodação)", edgecolor="white", linewidth=1.0)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(r["entrada"] + 0.3, y[i] + alt / 2, f"{_virgula(r['entrada'], 2)} h",
                va="center", fontsize=7.5, color=TINTA)
        ax.text(r["acomoda"] + 0.3, y[i] - alt / 2, f"{_virgula(r['acomoda'], 2)} h",
                va="center", fontsize=7.5, color=TINTA)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n({_virgula(s, 0)} % em potência máxima)"
                        for n, s in zip(g.index, g["satur"])], fontsize=7.5)
    ax.set_xlabel("Tempo desde o início do episódio (h)")
    ax.set_xlim(0, max(g["acomoda"]) * 1.16)
    _limpar(ax, eixo="x")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return fig


def fig_ciclo_rl():
    """
    Diagrama do ciclo de interação agente-ambiente.

    Existe para dar ao leitor não especialista a estrutura do problema antes das
    equações. É desenhado, e não importado, para seguir a mesma paleta e a mesma
    tipografia das demais figuras, e para ser reprodutível junto com elas.

    O espaçamento é generoso de propósito: numa primeira versão os rótulos das
    setas invadiram as caixas, porque o vão entre elas era estreito demais para
    o texto que precisava caber ali.
    """
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def caixa(x, y, w, h, titulo, sub, cor):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=cor, alpha=0.14,
                                   edgecolor=cor, linewidth=1.6, zorder=2))
        ax.text(x + w / 2, y + h * 0.60, titulo, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=TINTA, zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=7.6, color=TINTA2, zorder=3)

    LX, LW = 0.2, 2.9          # caixa esquerda
    RX = 10 - LX - LW          # caixa direita
    caixa(LX, 2.0, LW, 1.9, "Agente", "política π(a | s)", AZUL)
    caixa(RX, 2.0, LW, 1.9, "Ambiente", "sala + equipamento", AQUA)

    vao_e, vao_d = LX + LW, RX
    meio = (vao_e + vao_d) / 2

    # ação: de cima, da esquerda para a direita
    ax.annotate("", xy=(vao_d - 0.05, 3.45), xytext=(vao_e + 0.05, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=LARANJA, linewidth=2.0))
    ax.text(meio, 4.30, "ação  $a_t$", ha="center", fontsize=9.5, color=TINTA)
    ax.text(meio, 3.85, "nível de potência acionado", ha="center", fontsize=7.4,
            color=TINTA2)

    # estado e recompensa: de baixo, da direita para a esquerda
    ax.annotate("", xy=(vao_e + 0.05, 2.45), xytext=(vao_d - 0.05, 2.45),
                arrowprops=dict(arrowstyle="-|>", color=MUDO, linewidth=2.0))
    # Empilhado em duas linhas: numa única linha o rótulo excedia o vão entre as
    # caixas e se sobrepunha a elas.
    ax.text(meio, 2.00, "estado  $s_{t+1}$", ha="center", fontsize=9.5,
            color=TINTA)
    ax.text(meio, 1.58, "recompensa  $r_{t+1}$", ha="center", fontsize=9.5,
            color=TINTA)
    ax.text(meio, 1.16, "temperatura, ocupação e hora", ha="center",
            fontsize=7.4, color=TINTA2)

    # notas de rodapé, alinhadas sob cada caixa
    ax.text(LX + LW / 2, 0.45, "aprende a maximizar\no retorno acumulado",
            ha="center", va="center", fontsize=7.6, color=TINTA2, style="italic")
    ax.text(RX + LW / 2, 0.45, "evolui segundo o\nbalanço térmico da sala",
            ha="center", va="center", fontsize=7.6, color=TINTA2, style="italic")
    return fig


def fig_parametros_derivados(comb: pd.DataFrame, calib: pd.DataFrame):
    """
    Efeito de derivar B e k, isolados e combinados.

    Painel esquerdo: cada eixo da varredura, com as sementes sobrepostas.
    Painel direito: a configuração combinada contra a referência. O que a figura
    precisa mostrar não é só a média, é que a dispersão entre sementes encolhe,
    porque a instabilidade do treino é parte do problema que se está corrigindo.
    """
    aplicar_estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    ordem = ["B = 10.0", "B = 0.0", "k = 0.6", "k = 1.0", "k = 4.0"]
    sub = calib[calib["variante"].isin(ordem)]
    y = np.arange(len(ordem))[::-1]
    for i, var in enumerate(ordem):
        v = sub[sub["variante"] == var]["conf_estreita"]
        derivado = var in ("B = 0.0", "k = 1.0")
        a1.barh(y[i], v.mean(), height=0.55, color=AZUL if derivado else MUDO,
                alpha=1.0 if derivado else 0.55)
        a1.scatter(v, [y[i]] * len(v), s=16, color=TINTA, zorder=3,
                   edgecolor="white", linewidth=0.6)
        # Coluna de valores fixa: junto à barra, o rótulo caía sobre os pontos
        # das sementes.
        a1.text(99, y[i], _virgula(v.mean()), va="center", ha="right",
                fontsize=8, color=TINTA,
                fontweight="bold" if derivado else "normal")
    a1.set_yticks(y)
    a1.set_yticklabels([o.replace(".0", "").replace("=", "=") for o in ordem],
                       fontsize=8.5)
    a1.set_xlabel("Conforto na faixa estreita (%)")
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 25, 50, 75])
    a1.set_title("por parâmetro, isoladamente", fontsize=9)
    _limpar(a1, eixo="x")

    rots = ["referência\n(B=10, k=0,6)", "derivados\n(B=0, k=1,0)"]
    chaves = ["referência", "derivados"]
    for j, (rot, ch) in enumerate(zip(rots, chaves)):
        v = comb[comb["config"].str.startswith(ch)]["conf_estreita"]
        a2.bar(j, v.mean(), width=0.5, color=AZUL if j else MUDO,
               alpha=1.0 if j else 0.55, edgecolor="white", linewidth=1.2)
        a2.scatter([j] * len(v), v, s=20, color=TINTA, zorder=3,
                   edgecolor="white", linewidth=0.7)
        a2.text(j, v.mean() + 3.4, _virgula(v.mean()), ha="center",
                fontsize=9, fontweight="bold" if j else "normal", color=TINTA)
    a2.set_xticks(range(2))
    a2.set_xticklabels(rots, fontsize=8)
    a2.set_ylabel("Conforto na faixa estreita (%)")
    a2.set_ylim(0, 100)
    a2.set_title("combinados", fontsize=9)
    _limpar(a2)
    # Sem caixa de legenda: um marcador vazio no canto era lido como se fosse um
    # dado. Os pontos são identificados na legenda da figura.
    fig.tight_layout()
    return fig


def fig_orcamento_parametros(dados: Dict[str, pd.DataFrame]):
    """
    A reversão: qual configuração vence depende do orçamento de treinamento.

    Linhas que se cruzam, e não barras agrupadas, porque o argumento é sobre uma
    INTERAÇÃO: o efeito dos parâmetros muda de sinal com a duração do treino, e
    é o cruzamento que precisa ficar visível.
    """
    aplicar_estilo()
    r = dados["resumo"]
    bruto = dados["bruto"]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    x = [0, 1]
    for cfg, cor, marca, rot in [
        ("inferidos", MUDO, "s", "inferidos (B=10, k=0,6)"),
        ("derivados", AZUL, "o", "derivados (B=0, k=1,0)"),
    ]:
        sub = r[r["config"] == cfg].set_index("orcamento")
        ys = [sub.loc[o, "media"] for o in ("300k", "550k")]
        es = [sub.loc[o, "desvio"] for o in ("300k", "550k")]
        ax.plot(x, ys, color=cor, marker=marca, markersize=8, lw=2.4,
                markeredgecolor="white", markeredgewidth=1.2, label=rot, zorder=3)
        ax.fill_between(x, [a - b for a, b in zip(ys, es)],
                        [a + b for a, b in zip(ys, es)], color=cor, alpha=0.13)
        for xi, yi in zip(x, ys):
            ax.text(xi + (0.045 if xi == 0 else -0.045), yi + 2.6, _virgula(yi),
                    ha="left" if xi == 0 else "right", fontsize=8.5,
                    fontweight="bold", color=TINTA)
        # sementes individuais
        for o, xi in zip(("300k", "550k"), x):
            v = bruto[(bruto["config"] == cfg) & (bruto["orcamento"] == o)]["conf_estreita"]
            ax.scatter([xi] * len(v), v, s=13, color=cor, alpha=0.55, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(["300 mil passos\n(orçamento da calibração)",
                        "550 mil passos\n(orçamento do protocolo)"], fontsize=8.5)
    ax.set_ylabel("Conforto na faixa estreita (%)")
    ax.set_ylim(35, 100)
    ax.set_xlim(-0.16, 1.16)
    _limpar(ax)
    # Sem nota dentro da área de plotagem: ela colidia com a legenda, e o
    # cruzamento das curvas já é autoexplicativo com a legenda da figura.
    ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=2)
    return fig


# ======================================================= BOPTEST (terceiros)

_ROTULO_PERIODO = {"peak_cool_day": "dia de pico de resfriamento",
                   "typical_cool_day": "dia típico de resfriamento"}


def fig_boptest_transferencia(df: pd.DataFrame):
    """
    O achado da seção em uma imagem: o veredito depende de quem sintonizou o quê.

    Barras horizontais agrupadas por período, com os dois PI destacados em cor e
    os agentes em cinza. A leitura pretendida é vertical: o PI com ganhos
    congelados cai ABAIXO do melhor agente, e o mesmo PI re-sintonizado sobe
    ACIMA dele. A faixa sombreada entre os dois marca o que a re-sintonia
    recupera — que é a grandeza em disputa.

    Barra horizontal, e não vertical, porque os rótulos dos controladores são
    longos: na vertical eles exigiriam rotação, que custa legibilidade sem
    contrapartida.
    """
    aplicar_estilo()
    periodos = [p for p in ("peak_cool_day", "typical_cool_day")
                if p in set(df["periodo"])]
    fig, axes = plt.subplots(1, len(periodos), figsize=(7.4, 3.4), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, per in zip(axes, periodos):
        sub = df[df["periodo"] == per].iloc[::-1]          # melhor no topo
        rot = [r.replace(" (", "\n(") for r in sub["controlador"]]
        vals = sub["comfort_wide_pct"].to_numpy()

        cores, hachura = [], []
        for c in sub["controlador"]:
            if c.startswith("PI re-sintonizado"):
                cores.append(AZUL); hachura.append("")
            elif c.startswith("PI ("):
                cores.append(AZUL); hachura.append("///")
            elif c.startswith("DQN"):
                cores.append(LARANJA); hachura.append("")
            else:
                cores.append(MUDO); hachura.append("")

        y = np.arange(len(vals))
        for yi, v, c, h in zip(y, vals, cores, hachura):
            ax.barh(yi, v, color=c, height=0.68, edgecolor="white",
                    linewidth=1.0, hatch=h)
        for yi, v in zip(y, vals):
            ax.text(v + 1.2, yi, f"{_virgula(v)}%", va="center", fontsize=8,
                    color=TINTA)

        # A distância entre os dois PI é o argumento; marcá-la evita que o
        # leitor precise subtrair duas barras de olho.
        try:
            i_cong = list(sub["controlador"]).index("PI (ganhos locais, congelados)")
            i_res = list(sub["controlador"]).index("PI re-sintonizado (emulador)")
            v_cong, v_res = vals[i_cong], vals[i_res]
            # Chave à DIREITA das barras: dentro delas o texto disputaria espaço
            # com a hachura, e entre elas cairia no vão branco. Aqui a diferença
            # entre os dois PI fica isolada, que é como ela precisa ser lida.
            xb = 118.0
            ax.plot([xb, xb], [i_cong, i_res], color=AZUL, linewidth=1.1)
            for yi in (i_cong, i_res):
                ax.plot([xb - 3.0, xb], [yi, yi], color=AZUL, linewidth=1.1)
            ax.text(xb + 3.0, (i_cong + i_res) / 2.0,
                    f"re-sintonia\n+{_virgula(v_res - v_cong)} pp",
                    ha="left", va="center", fontsize=7.5, color=AZUL,
                    fontweight="bold")
        except ValueError:
            pass

        ax.set_yticks(y)
        ax.set_yticklabels(rot, fontsize=7.5)
        ax.set_title(_ROTULO_PERIODO.get(per, per), fontsize=9)
        # Folga à direita para a chave da re-sintonia.
        ax.set_xlim(0, 152)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Conforto na faixa [22, 26] °C (%)")
        _limpar(ax, eixo="x")

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.tight_layout()
    return fig


def fig_boptest_sintonia(df: pd.DataFrame):
    """
    Superfície de sintonia do PI no emulador, com o ponto herdado marcado.

    O ponto do artigo (Kp = 1,3; Ki = 0,2) não está próximo do ótimo desta
    planta: é isso que a figura precisa mostrar, e por isso a grade inteira
    aparece, e não apenas o melhor valor. A escala de cor é sequencial porque a
    grandeza tem um extremo preferido e nenhum ponto neutro.
    """
    aplicar_estilo()
    tabela = df.pivot_table(index="ki", columns="kp", values="conf_larga")
    fig, ax = plt.subplots(figsize=(5.0, 3.1))

    im = ax.imshow(tabela.to_numpy(), aspect="auto", origin="lower",
                   cmap="Blues", vmin=float(df["conf_larga"].min()),
                   vmax=float(df["conf_larga"].max()))
    ax.set_xticks(range(len(tabela.columns)))
    ax.set_xticklabels([_virgula(c, 1) for c in tabela.columns])
    ax.set_yticks(range(len(tabela.index)))
    ax.set_yticklabels([_virgula(i, 2) for i in tabela.index])
    ax.set_xlabel("Ganho proporcional $K_p$")
    ax.set_ylabel("Ganho integral $K_i$")

    faixa = float(df["conf_larga"].max() - df["conf_larga"].min()) or 1.0
    for i, ki in enumerate(tabela.index):
        for j, kp in enumerate(tabela.columns):
            v = tabela.iloc[i, j]
            if np.isnan(v):
                continue
            claro = (v - df["conf_larga"].min()) / faixa > 0.55
            ax.text(j, i, _virgula(v), ha="center", va="center", fontsize=7.5,
                    color="white" if claro else TINTA)

    def _marcar(kp, ki, cor, rotulo, dy):
        if kp not in list(tabela.columns) or ki not in list(tabela.index):
            return
        j, i = list(tabela.columns).index(kp), list(tabela.index).index(ki)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=cor, linewidth=2.0))
        ax.annotate(rotulo, xy=(j, i), xytext=(j, i + dy), ha="center",
                    fontsize=7.5, color=cor, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=cor, linewidth=1.0))

    melhor = df.sort_values(["conf_larga", "conf_estreita"],
                            ascending=False).iloc[0]
    # As duas anotações saem FORA da grade, uma acima e outra abaixo: sobre as
    # células elas cobririam justamente os números que sustentam o argumento.
    ax.set_ylim(-1.95, len(tabela.index) - 0.5 + 0.95)
    _marcar(1.3, 0.2, LARANJA, "ganhos do artigo\n(planta local)", 0.85)
    _marcar(float(melhor["kp"]), float(melhor["ki"]), AQUA,
            "ótimo nesta planta", -1.75)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Conforto na faixa [22, 26] °C (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return fig
