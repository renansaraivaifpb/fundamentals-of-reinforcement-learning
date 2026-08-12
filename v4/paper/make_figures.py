# -*- coding: utf-8 -*-
"""
Gera as figuras do artigo a partir das MESMAS fontes que produzem as tabelas.

Motivação: figuras e tabelas divergentes são um defeito clássico de artigo — os
gráficos de `plots_paper/` e `plots_lab2/` vinham de execuções distintas das que
geraram os números reportados, e nada garantia correspondência. Aqui as Figuras 1
e 3 são RECOMPUTADAS nesta execução, e as Figuras 2 e 4 leem os mesmos CSVs
canônicos das Tabelas 5 e 8. O script imprime uma conferência dos valores usados
e grava `figuras_paper/valores_figuras.json` para auditoria.

    python make_figures.py
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import PIController, ThermostatAgent as ThermostatDB
from config import config_for_profile
from env import ClassroomACEnv
from metrics import evaluate_agent, is_controllable, run_episode
from model_io import load_agent
from scenarios import build_scenario_matrix
from stats_analysis import dwell_times
from wrappers import ActionRepeatWrapper, MinDwellWrapper

SAIDA = "figuras_paper"
os.makedirs(SAIDA, exist_ok=True)

# --- Paleta ---------------------------------------------------------------
# Slots 1-3 do tema categórico de referência; validados all-pairs em modo claro
# (pior par CVD ΔE 9,2; visão normal 24,0). O alívio exigido pelo aviso de
# contraste do aqua é atendido por rótulos diretos e pelas tabelas equivalentes.
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, MUDO = "#0b0b0b", "#52514e", "#8a8a85"
GRADE = "#e3e3e0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": MUDO,
    "axes.labelcolor": TINTA,
    "axes.titlesize": 9.5,
    "text.color": TINTA,
    "xtick.color": TINTA2,
    "ytick.color": TINTA2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _limpar(ax, *, eixo="y"):
    """Grade recessiva, sem molduras supérfluas."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=eixo, color=GRADE, linewidth=0.7)


CENARIOS = build_scenario_matrix()
registro: dict = {}


def fabrica(cfg, repeat=2, dwell=False):
    def f():
        env = ClassroomACEnv(config=cfg)
        if dwell:
            env = MinDwellWrapper(env)
        return ActionRepeatWrapper(env, repeat=repeat)
    return f


# ===================================================== FIGURA 1 — decomposição
def figura1() -> dict:
    """
    Cascata da decomposição da vantagem (Tabela 4).

    Forma: o dado é a decomposição de um total em contribuições atribuíveis —
    é o caso canônico de gráfico em cascata, e não de barras lado a lado, que
    perderiam a relação de acumulação.
    """
    print("\n[Figura 1] recomputando a decomposição...")
    cfg = config_for_profile("Equilibrado")
    fab = fabrica(cfg)

    etapas = []
    for rotulo, agente, info in [
        ("Termostato\n(zona morta = 0)", ThermostatDB(cfg, deadband=0.0), True),
        ("+ histerese\n(zona morta = 1 °C)", ThermostatDB(cfg, deadband=1.0), True),
        ("PI sintonizado\n(Kp=1,3; Ki=0,2)", PIController(cfg, kp=1.3, ki=0.2), True),
    ]:
        r = evaluate_agent(agente, fab, CENARIOS, cfg, seed=0, pass_info_to_agent=info)
        etapas.append((rotulo, float(r["summary"]["comfort_wide_pct"])))

    modelo, cfg_t, meta = load_agent("models_paper/DQN_Equilibrado_seed0.zip")
    fab_t = fabrica(cfg_t, meta.get("action_repeat", 2))
    r = evaluate_agent(modelo, fab_t, CENARIOS, cfg_t, seed=0)
    etapas.append(("DQN\n550k passos", float(r["summary"]["comfort_wide_pct"])))

    rotulos = [e[0] for e in etapas]
    vals = [e[1] for e in etapas]
    ganhos = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    base = 0.0
    # Cor por ATRIBUIÇÃO, não por posição: cinza = ponto de partida, laranja =
    # configuração do baseline, azul = controle clássico, aqua = aprendizado.
    cores = [MUDO, LARANJA, AZUL, AQUA]
    for i, (rot, g) in enumerate(zip(rotulos, ganhos)):
        ax.bar(i, g, bottom=base, color=cores[i], width=0.62,
               edgecolor="white", linewidth=1.2)
        if i > 0:
            ax.plot([i - 1 + 0.31, i - 0.31], [base, base],
                    color=MUDO, linewidth=0.8, linestyle=(0, (3, 3)))
        # Rótulo direto do incremento — seletivo, só o que carrega o argumento.
        if i == 0:
            txt = f"{vals[i]:.1f}%".replace(".", ",")
        elif abs(g) < 0.05:
            txt = "+0,0 pp"
        else:
            txt = f"+{g:.1f} pp".replace(".", ",")
        ax.text(i, base + g + 1.6, txt, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TINTA)
        # Incremento nulo não desenha barra: sem uma marca explícita o leitor lê
        # "dado ausente" em vez de "ganho zero", que é justamente o achado.
        if abs(g) < 0.05:
            ax.plot([i - 0.31, i + 0.31], [base, base], color=cores[i],
                    linewidth=3.0, solid_capstyle="butt")
        base += g

    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_ylabel("Conforto na faixa [22, 26] °C (%)")
    ax.set_ylim(0, 104)
    _limpar(ax)
    ax.annotate("aprendizado não\nacrescenta nada",
                xy=(3, vals[-1]), xytext=(2.55, 44),
                fontsize=8, color=TINTA2, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUDO, linewidth=0.9))
    fig.savefig(f"{SAIDA}/fig1_decomposicao.png")
    plt.close(fig)

    print("  " + " | ".join(f"{r.split(chr(10))[0]}: {v:.1f}%"
                            for r, v in zip(rotulos, vals)))
    return {"rotulos": [r.replace("\n", " ") for r in rotulos],
            "valores": vals, "ganhos": ganhos}


# ========================================================= FIGURA 2 — ablação
def figura2() -> dict:
    """
    Ablação: média por variante com IC95 e as três sementes sobrepostas.

    Mostrar as sementes individuais é obrigatório com n = 3 — uma barra de média
    esconde que `degraus_legado` tem uma semente em 44,0 e duas acima de 81.
    """
    print("\n[Figura 2] lendo results_ablation/ablation_raw.csv...")
    raw = pd.read_csv("results_ablation/ablation_raw.csv")
    nomes = {
        "completa": "completa (proposta)",
        "convencional": "convencional (quadrática pura)",
        "sem_short_cycling": "sem anti-short-cycling",
        "sem_penal_frio": "sem penalidade de frio",
        "quadratica_pura": "quadrática pura",
        "sem_penal_troca": "sem penalidade de troca",
        "degraus_legado": "degraus (legado)",
        "sem_gradiente": "sem gradiente interno",
    }
    g = (raw.groupby("variante")["conf_estreita_pct"]
            .agg(["mean", "min", "max"]).reindex(list(nomes)))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(g))[::-1]
    for i, (var, linha) in enumerate(g.iterrows()):
        yy = y[i]
        destaque = var == "completa"
        cor = AZUL if destaque else MUDO
        ax.barh(yy, linha["mean"], color=cor, height=0.52,
                alpha=1.0 if destaque else 0.55)
        ax.plot([linha["min"], linha["max"]], [yy, yy],
                color=TINTA, linewidth=1.4, solid_capstyle="butt")
        pts = raw.loc[raw["variante"] == var, "conf_estreita_pct"]
        ax.scatter(pts, [yy] * len(pts), s=17, color=TINTA, zorder=3,
                   edgecolor="white", linewidth=0.7)
        # Coluna de valores fixa, fora da área de dados: alinhada, ela nunca
        # colide com as sementes nem com a barra de amplitude.
        ax.text(104, yy, f"{linha['mean']:.1f}".replace(".", ","),
                va="center", ha="right", fontsize=8.5, color=TINTA,
                fontweight="bold" if destaque else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([nomes[v] for v in g.index], fontsize=8)
    ax.set_xlabel("Conforto na faixa estreita [23, 25] °C (%)")
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    _limpar(ax, eixo="x")
    ax.scatter([], [], s=17, color=TINTA, edgecolor="white",
               label="sementes individuais (n = 3)")
    ax.plot([], [], color=TINTA, linewidth=1.4, label="amplitude observada")
    # Legenda abaixo do eixo: dentro da área de dados ela cobria as duas
    # variantes de maior interesse.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, fontsize=7.5)
    fig.savefig(f"{SAIDA}/fig2_ablacao.png")
    plt.close(fig)

    print("  " + " | ".join(f"{v}: {g.loc[v,'mean']:.1f}" for v in g.index))
    return {v: float(g.loc[v, "mean"]) for v in g.index}


# ==================================================== FIGURA 3 — permanência
def figura3() -> dict:
    """
    Distribuição empírica acumulada dos tempos de permanência.

    A ECDF é a forma correta aqui: a afirmação é sobre a FRAÇÃO abaixo de um
    limiar (d_min), que a ECDF lê diretamente no eixo, ao contrário de um
    histograma. Duas séries, portanto legenda mais rótulos diretos.
    """
    print("\n[Figura 3] recomputando permanências...")
    modelo, cfg, meta = load_agent("models_paper/DQN_Equilibrado_seed0.zip")
    rep = meta.get("action_repeat", 2)

    def coletar(dwell: bool) -> np.ndarray:
        fab = fabrica(cfg, rep, dwell=dwell)
        dfs = [run_episode(modelo, fab(), c, seed=0)
               for c in CENARIOS if is_controllable(fab(), c, seed=0)]
        return np.concatenate([dwell_times(df, cfg) for df in dfs if len(df)])

    mole, dura = coletar(False), coletar(True)
    d_min = cfg.min_dwell_minutes

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for dados, cor, marca, rot in [
        (mole, LARANJA, "o", "Penalidade de recompensa (eq. 5)"),
        (dura, AZUL, "s", "Restrição dura (MinDwell)"),
    ]:
        x = np.sort(dados)
        yv = np.arange(1, len(x) + 1) / len(x) * 100.0
        ax.step(np.concatenate([[0], x]), np.concatenate([[0], yv]),
                where="post", color=cor, linewidth=2.0, label=rot)
        # Marcador esparso: identidade sobrevive à impressão em escala de cinza.
        idx = np.linspace(0, len(x) - 1, 7).astype(int)
        ax.scatter(x[idx], yv[idx], color=cor, marker=marca, s=22, zorder=3,
                   edgecolor="white", linewidth=0.8)
        viol = (dados < d_min).mean() * 100.0
        ax.scatter([d_min], [viol], color=cor, s=48, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{viol:.1f}%".replace(".", ","),
                    xy=(d_min, viol), xytext=(d_min + 14, viol + (5 if viol > 40 else 9)),
                    fontsize=9, fontweight="bold", color=TINTA,
                    arrowprops=dict(arrowstyle="-", color=MUDO, linewidth=0.8))

    ax.axvline(d_min, color=TINTA, linewidth=1.1, linestyle=(0, (4, 3)))
    # Rótulo acima da área de plotagem: à esquerda da linha ele colidia com o
    # eixo y e com o tique de 100 %.
    ax.text(d_min + 5, 112, f"requisito $d_{{min}}$ = {d_min:.0f} min",
            ha="left", va="center", fontsize=8, color=TINTA2)
    ax.set_xlabel("Tempo de permanência entre comutações (min)")
    ax.set_ylabel("Fração acumulada das comutações (%)")
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    _limpar(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(f"{SAIDA}/fig3_permanencia.png")
    plt.close(fig)

    out = {"violacoes_mole_pct": float((mole < d_min).mean() * 100),
           "violacoes_dura_pct": float((dura < d_min).mean() * 100),
           "mediana_mole_min": float(np.median(mole)),
           "mediana_dura_min": float(np.median(dura))}
    print("  " + " | ".join(f"{k}: {v:.1f}" for k, v in out.items()))
    return out


# ====================================================== FIGURA 4 — precisão
def figura4() -> dict:
    """
    Regime de precisão: custo × dispersão, com o tamanho do ponto indicando o
    tempo dentro da tolerância.

    Duas medidas de escalas distintas jamais em eixo duplo: aqui cada uma tem
    seu próprio eixo espacial, e a terceira entra como área. O canto inferior
    esquerdo é o ótimo (barato e estável).
    """
    print("\n[Figura 4] lendo results_lab2.csv...")
    d = pd.read_csv("results_lab2.csv")
    d = d[d["controlador"] != "Antecipatório (manual)"].copy()

    def familia(nome):
        if nome.startswith("PI"):
            return "PI (clássico)", AZUL, "D"
        if nome.startswith("TD3"):
            return "TD3", LARANJA, "o"
        return "SAC", AQUA, "^"

    from matplotlib.lines import Line2D

    # Deslocamentos por ponto: com seis rótulos numa área pequena, um offset
    # único garante colisão. Ajustados após inspeção visual da figura.
    desloc = {
        "PI bidirecional": (30, -4), "TD3 Lab_Equilibrado": (0, -14),
        "TD3 Lab_Precisao": (0, -14), "SAC Lab_Precisao": (0, 11),
        "SAC Lab_Equilibrado": (0, -14), "TD3 Lab_Economico": (0, 11),
    }
    alinha = {"PI bidirecional": "left"}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for _, r in d.iterrows():
        fam, cor, marca = familia(r["controlador"])
        ax.scatter(r["custo_dia"], r["sigma"], s=28 + (r["na_tol_%"] - 85) * 5.2,
                   color=cor, marker=marca, alpha=0.9, zorder=3,
                   edgecolor="white", linewidth=1.0)
        curto = (r["controlador"].replace("Lab_", "").replace(" bidirecional", "")
                 .replace("Equilibrado", "Equil.").replace("Precisao", "Precisão")
                 .replace("Economico", "Econôm."))
        ax.annotate(curto, (r["custo_dia"], r["sigma"]),
                    xytext=desloc[r["controlador"]], textcoords="offset points",
                    fontsize=7.5, color=TINTA2,
                    ha=alinha.get(r["controlador"], "center"), va="center")

    ax.set_xlabel("Custo diário (R$)")
    ax.set_ylabel("Dispersão da temperatura, σ (°C)")
    ax.set_xlim(14.0, 19.0)
    ax.set_ylim(0.045, 0.335)
    _limpar(ax)
    ax.grid(axis="x", color=GRADE, linewidth=0.7)
    # Sem seta: a anotação anterior cruzava o rótulo do PI. O canto ótimo é
    # nomeado por texto, que não colide com nada.
    ax.text(14.08, 0.055, "↙  melhor: mais barato e mais estável",
            fontsize=8, color=TINTA2, ha="left", va="center")
    # Handles de tamanho fixo: herdar o `s` do scatter faria a legenda codificar
    # o tempo na tolerância de um ponto arbitrário.
    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c,
                      markeredgecolor="white", markersize=7, label=n)
               for n, c, m in [("PI (clássico)", AZUL, "D"),
                               ("TD3", LARANJA, "o"), ("SAC", AQUA, "^")]]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.savefig(f"{SAIDA}/fig4_precisao.png")
    plt.close(fig)

    print("  " + " | ".join(f"{r['controlador']}: σ={r['sigma']:.3f}"
                            for _, r in d.iterrows()))
    return d.set_index("controlador")[["na_tol_%", "sigma", "custo_dia"]].to_dict("index")


if __name__ == "__main__":
    registro["figura1_decomposicao"] = figura1()
    registro["figura2_ablacao"] = figura2()
    registro["figura3_permanencia"] = figura3()
    registro["figura4_precisao"] = figura4()

    with open(f"{SAIDA}/valores_figuras.json", "w", encoding="utf-8") as fh:
        json.dump(registro, fh, ensure_ascii=False, indent=2)
    print(f"\nfiguras + valores_figuras.json gravados em {SAIDA}/")
