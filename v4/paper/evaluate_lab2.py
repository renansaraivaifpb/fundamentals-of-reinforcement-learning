# -*- coding: utf-8 -*-
"""
Avaliação do laboratório bidirecional (lab2) — roda sem supervisão.

Compara os agentes contínuos (TD3/SAC) contra dois baselines com a MESMA
autoridade de atuação:
  * PI bidirecional sintonizado — o adversário honesto
  * política antecipatória escrita à mão — o teto de economia no posto de ponta

Produz figuras em `plots_lab2/` e a tabela em `results_lab2.csv`.

    python evaluate_lab2.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def _numpy_shim() -> None:
    """Modelos serializados com numpy 2.x sob numpy 1.x instalado."""
    import numpy.core  # noqa: F401
    for sub in ("", ".numeric", ".multiarray", ".umath", "._multiarray_umath"):
        try:
            __import__("numpy.core" + sub)
            sys.modules["numpy._core" + sub] = sys.modules["numpy.core" + sub]
        except Exception:
            pass


_numpy_shim()

from ac_physics import ACState                       # noqa: E402
from baselines import PIController                   # noqa: E402
from config import LAB_PROFILES, config_for_lab2     # noqa: E402
from env import ClassroomACEnv                       # noqa: E402
from metrics import episode_metrics, run_episode     # noqa: E402
from wrappers import ActionRepeatWrapper             # noqa: E402

CORES = {
    "PI bidirecional": "#e05561", "Antecipatório (manual)": "#8c6bb1",
    "TD3 Lab_Precisao": "#2b7bba", "TD3 Lab_Equilibrado": "#4daf4a",
    "TD3 Lab_Economico": "#e8801a",
    "SAC Lab_Precisao": "#7fb3d5", "SAC Lab_Equilibrado": "#9ed19b",
    "SAC Lab_Economico": "#f5c18b",
}
CENARIO = {"start_temp": 24.0, "occupancy": 45, "hour": 12, "name": "lab_pico"}


class Antecipatorio:
    """
    Referência antecipatória bidirecional: aproxima-se da borda inferior da
    tolerância antes do posto de ponta e coasta durante ele.

    Não é um controlador a propor — é o TETO de economia no pico, para medir
    quanto da oportunidade cada agente aprendido capturou.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self) -> None:
        pass

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        c = self.cfg
        t = info["temperature"]
        h = info["hour_float"] % 24.0
        setp, tol = c.ideal_temp, c.lab_tolerance
        if 15.5 <= h < 17.5:                      # pré-resfria até a borda inferior
            alvo = setp - 0.8 * tol
        elif 17.5 <= h < 20.5:                    # coasta no posto de ponta
            alvo = setp + 0.8 * tol
        else:
            alvo = setp
        return np.array([float(np.clip((t - alvo) * 2.0, -1.0, 1.0))], np.float32), None


def fabrica(cfg):
    return lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)


def coletar(models_dir: str, seeds: List[int]) -> Dict[str, Dict]:
    """Trajetórias e configs por controlador."""
    base = config_for_lab2("Lab_Equilibrado")
    out: Dict[str, Dict] = {}

    for nome, agente in (
        ("PI bidirecional", PIController(base, kp=1.3, ki=0.2, discrete=False)),
        ("Antecipatório (manual)", Antecipatorio(base)),
    ):
        out[nome] = {"agente": agente, "cfg": base, "info": True}

    # A config vem do METADADO, não do default vigente: mudar um default do
    # módulo `config` não pode alterar silenciosamente como um modelo antigo é
    # avaliado. Ver model_io.py.
    from model_io import load_agent, read_metadata

    for path in sorted(glob.glob(os.path.join(models_dir, "*_lab2*.zip"))):
        meta = read_metadata(path)
        if meta.get("seed", 0) not in seeds:
            continue
        try:
            model, cfg_treino, meta = load_agent(path)
        except Exception as e:
            print(f"  ignorado {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue
        sufixo = "" if meta.get("observe_scaled_error", True) else " (obs7)"
        rotulo = f"{meta.get('algo')} {meta.get('profile','?')}{sufixo}"
        out[rotulo] = {"agente": model, "cfg": cfg_treino, "info": False}
    return out


def faixas(ax, cfg, h0, h1) -> None:
    for h in np.arange(np.floor(h0), np.ceil(h1), 0.5):
        cor = {"ponta": "#e05561", "intermediario": "#f0ad4e"}.get(cfg.tariff.posto(h))
        if cor:
            ax.axvspan(h, h + 0.5, color=cor, alpha=0.16, lw=0)


def eixo_horas(ax, h0, h1) -> None:
    ticks = np.arange(np.ceil(h0 / 3) * 3, h1 + 0.1, 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t) % 24:02d}h" for t in ticks])
    ax.set_xlim(h0, h1)


def main() -> None:
    p = argparse.ArgumentParser(description="Avaliação do laboratório bidirecional.")
    p.add_argument("--models_dir", type=str, default="models_lab2")
    p.add_argument("--save_dir", type=str, default="plots_lab2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

    ctrl = coletar(args.models_dir, args.seeds)
    if not ctrl:
        print("nenhum controlador encontrado"); return
    base = config_for_lab2("Lab_Equilibrado")
    SETP, TOL = base.ideal_temp, base.lab_tolerance
    print(f"tarifa: {base.tariff.name} | tolerância ±{TOL} °C")
    print(f"controladores: {list(ctrl)}")

    # --- trajetórias (semente de avaliação 0) e métricas (média das sementes) ---
    traj, linhas = {}, []
    for nome, d in ctrl.items():
        traj[nome] = run_episode(d["agente"], fabrica(d["cfg"])(), CENARIO,
                                 seed=0, pass_info_to_agent=d["info"])
        ms = [episode_metrics(run_episode(d["agente"], fabrica(d["cfg"])(), CENARIO,
                                         seed=s, pass_info_to_agent=d["info"]), d["cfg"])
              for s in args.eval_seeds]
        a = lambda k: float(np.mean([m[k] for m in ms]))
        linhas.append({
            "controlador": nome, "na_tol_%": a("in_tolerance_pct"),
            "sigma": a("temp_std"), "|T-24|": a("abs_dev_from_ideal"),
            "desv_max": a("max_abs_dev"), "custo_dia": a("cost_brl_day"),
            "custo_ponta": a("peak_cost_brl"), "kWh_dia": a("energy_kwh_day"),
            "ajustes_h": a("changes_per_hour"),
        })
    tab = pd.DataFrame(linhas).round(3).sort_values("na_tol_%", ascending=False)
    tab.to_csv("results_lab2.csv", index=False)
    print("\n" + tab.to_string(index=False))

    h0 = traj[list(traj)[0]]["hour_float"].min()
    h1 = traj[list(traj)[0]]["hour_float"].max()

    # --- fig1: trajetória sobre os postos ---
    fig, ax = plt.subplots(figsize=(13, 6))
    faixas(ax, base, h0, h1)
    ax.axhspan(SETP - TOL, SETP + TOL, color="#4daf4a", alpha=0.18, lw=0)
    ax.axhline(SETP, color="#2f6b2f", ls="--", lw=1.2)
    for nome, df in traj.items():
        ax.plot(df["hour_float"], df["temperature"], lw=1.7,
                ls="--" if "manual" in nome else "-", color=CORES.get(nome, "#888"), label=nome)
    eixo_horas(ax, h0, h1)
    ax.set_ylim(SETP - 1.5, SETP + 1.5)
    ax.set_xlabel("Hora do dia"); ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Laboratório bidirecional — trajetória sobre os postos tarifários", fontsize=13)
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="#e05561", alpha=.16), Patch(facecolor="#4daf4a", alpha=.18)]
    l += ["posto de ponta", f"tolerância ±{TOL} °C"]
    ax.legend(h, l, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(f"{args.save_dir}/fig1_trajetoria.png"); plt.close(fig)

    # --- fig2: carga com sinal (aquece vs resfria) ---
    n = len(traj)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.9 * n), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    axes = np.atleast_1d(axes)
    for ax, (nome, df) in zip(axes, traj.items()):
        faixas(ax, base, h0, h1)
        carga = df["load"] if "load" in df else df["action"] * 0.0
        ax.fill_between(df["hour_float"], 0, carga.clip(lower=0), step="post",
                        color="#3b6fb0", alpha=0.7, label="resfria")
        ax.fill_between(df["hour_float"], 0, carga.clip(upper=0), step="post",
                        color="#d1495b", alpha=0.7, label="aquece")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-1.1, 1.1); ax.set_ylabel("carga", fontsize=8)
        cp = df[df["posto"] == "ponta"]["cost_brl"].sum()
        ax.set_title(f"{nome} — pico R$ {cp:.2f}", fontsize=9, loc="left")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    eixo_horas(axes[-1], h0, h1); axes[-1].set_xlabel("Hora do dia")
    fig.suptitle("Carga com sinal: aquecimento vs. resfriamento", y=1.004, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{args.save_dir}/fig2_carga_bidirecional.png"); plt.close(fig)

    # --- fig3: custo acumulado ---
    fig, ax = plt.subplots(figsize=(13, 5))
    faixas(ax, base, h0, h1)
    for nome, df in traj.items():
        ax.plot(df["hour_float"], df["cost_brl"].cumsum(), lw=2.0,
                ls="--" if "manual" in nome else "-", color=CORES.get(nome, "#888"),
                label=f"{nome} (R$ {df['cost_brl'].sum():.2f})")
    eixo_horas(ax, h0, h1)
    ax.set_xlabel("Hora do dia"); ax.set_ylabel("Custo acumulado (R$)")
    ax.set_title("Custo acumulado — inclinação na faixa vermelha = custo do pico", fontsize=13)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{args.save_dir}/fig3_custo_acumulado.png"); plt.close(fig)

    # --- fig4: distribuição do erro ---
    nomes = list(traj)
    dados = [(traj[n]["temperature"] - SETP).to_numpy() for n in nomes]
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(nomes)), 5.5))
    ax.axhspan(-TOL, TOL, color="#4daf4a", alpha=0.18, lw=0, label=f"±{TOL} °C")
    ax.axhline(0, color="#2f6b2f", ls="--", lw=1.2)
    parts = ax.violinplot(dados, showextrema=False)
    for b, nm in zip(parts["bodies"], nomes):
        b.set_facecolor(CORES.get(nm, "#888")); b.set_alpha(0.6)
    ax.boxplot(dados, widths=0.12, showfliers=False, medianprops={"color": "black"})
    ax.set_xticks(range(1, len(nomes) + 1))
    ax.set_xticklabels([n.replace(" ", "\n") for n in nomes], fontsize=7)
    ax.set_ylabel("Desvio do setpoint (°C)")
    ax.set_title("Distribuição do erro de rastreamento", fontsize=13)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{args.save_dir}/fig4_distribuicao_erro.png"); plt.close(fig)

    # --- fig5: fronteira precisão × custo no pico ---
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for _, r in tab.iterrows():
        clas = ("PI" in r.controlador) or ("manual" in r.controlador)
        ax.scatter(r.custo_ponta, r["na_tol_%"], s=170, marker="X" if clas else "o",
                   color=CORES.get(r.controlador, "#888"), zorder=3)
        ax.annotate(r.controlador, (r.custo_ponta, r["na_tol_%"]),
                    textcoords="offset points", xytext=(9, 5), fontsize=8)
    ax.axhline(95, color="gray", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[1], 95.4, "requisito 95%", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("Custo no posto de ponta (R$) — menor é melhor")
    ax.set_ylabel("Tempo dentro de ±0,5 °C (%) — maior é melhor")
    ax.set_title("Fronteira precisão × custo no pico\n(X = controle clássico, ● = RL)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{args.save_dir}/fig5_fronteira.png"); plt.close(fig)

    print(f"\nfiguras em {args.save_dir}/ | tabela em results_lab2.csv")


if __name__ == "__main__":
    main()
