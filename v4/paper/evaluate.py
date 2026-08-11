# -*- coding: utf-8 -*-
"""
Avaliação: reproduz as Tabelas 4, 5 e 6 e as Figuras 1 a 4 do paper.

Uso:
    python evaluate.py                      # Tabelas 4 e 5 + figuras
    python evaluate.py --generalization     # inclui a Tabela 6
    python evaluate.py --figure1            # só a Figura 1 (não exige modelos)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ac_physics import ACState
from config import ROOM_VARIATIONS, ClassroomConfig, config_for_profile
from env import ClassroomACEnv
from metrics import evaluate_agent, format_table4, run_episode
from scenarios import ThermostatAgent, build_scenario_matrix
from wrappers import ActionRepeatWrapper, ContinuousActionWrapper

PLOTS_DIR = "plots_paper"
PROFILE_COLORS = {
    "Agressivo": "#2b7bba",
    "Equilibrado": "#4daf4a",
    "Passivo": "#8c6bb1",
    "Termostato": "#e05561",
    "SAC": "#ff8c1a",
}


# --------------------------------------------------------------- carregamento

def _load_sb3(path: str):
    """Carrega DQN ou SAC conforme o metadado, com shim de numpy 2.x."""
    import sys
    import numpy.core  # noqa: F401
    for sub in ("", ".numeric", ".multiarray", ".umath", "._multiarray_umath"):
        try:
            __import__("numpy.core" + sub)
            sys.modules["numpy._core" + sub] = sys.modules["numpy.core" + sub]
        except Exception:
            pass

    from stable_baselines3 import DQN, SAC
    meta_path = path.replace(".zip", "_config.json")
    algo = "DQN"
    meta: Dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        algo = meta.get("algo", "DQN")
    cls = SAC if algo == "SAC" else DQN
    return cls.load(path, device="cpu"), meta


def make_factory(cfg: ClassroomConfig, repeat: int, continuous: bool):
    """Fábrica de ambientes que respeita o CONTRATO do modelo treinado."""
    def factory():
        env = ClassroomACEnv(config=cfg)
        if continuous:
            env = ContinuousActionWrapper(env)
        return ActionRepeatWrapper(env, repeat=repeat)
    return factory


# ---------------------------------------------------------------- Tabelas 4/5

def evaluate_all(models_dir: str, seed: int = 0) -> Dict[str, Dict]:
    scenarios = build_scenario_matrix()
    results: Dict[str, Dict] = {}

    for path in sorted(glob.glob(os.path.join(models_dir, "*.zip"))):
        model, meta = _load_sb3(path)
        profile = meta.get("profile", os.path.basename(path))
        algo = meta.get("algo", "DQN")
        repeat = meta.get("action_repeat", 2)
        label = profile if algo == "DQN" else f"SAC ({profile})"
        if meta.get("seed", 0) != seed:
            continue

        cfg = config_for_profile(profile) if profile in ("Agressivo", "Equilibrado", "Passivo") else ClassroomConfig()
        factory = make_factory(cfg, repeat, continuous=(algo == "SAC"))
        results[label] = evaluate_agent(model, factory, scenarios, cfg, seed=seed)
        print(f"  avaliado: {label}")

    # Baseline termostático, sob o mesmo horizonte de decisão.
    cfg = ClassroomConfig()
    factory = make_factory(cfg, 2, continuous=False)
    results["Termostato"] = evaluate_agent(
        ThermostatAgent(cfg), factory, scenarios, cfg, seed=seed,
        pass_info_to_agent=True,
    )
    print("  avaliado: Termostato (baseline)")
    return results


# ----------------------------------------------------------------- Tabela 6

def evaluate_generalization(model, profile: str, seed: int = 0) -> pd.DataFrame:
    """Tabela 6: agente treinado em C_th=15, K=0,5 avaliado sem retreino."""
    scenarios = build_scenario_matrix()
    rows: List[Dict] = []

    for room, params in ROOM_VARIATIONS.items():
        cfg = replace(config_for_profile(profile), **params)
        factory = make_factory(cfg, 2, continuous=False)
        rl = evaluate_agent(model, factory, scenarios, cfg, seed=seed)
        th = evaluate_agent(
            ThermostatAgent(cfg), factory, scenarios, cfg, seed=seed,
            pass_info_to_agent=True,
        )
        c_rl = rl["summary"]["comfort_wide_pct"]
        c_th = th["summary"]["comfort_wide_pct"]
        rows.append({
            "sala": room,
            "conf_dqn_pct": c_rl,
            "conf_term_pct": c_th,
            "delta_pp": c_rl - c_th,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figuras

def plot_figure1(save_dir: str) -> None:
    """Figura 1: quadrática pura vs. platô plano vs. platô com gradiente."""
    cfg = ClassroomConfig()
    B, k, Bc = cfg.comfort_bonus, cfg.comfort_sensitivity, cfg.comfort_gradient
    t_min, t_max, ideal = cfg.temp_comfort_min, cfg.temp_comfort_max, cfg.ideal_temp
    temps = np.linspace(16, 32, 600)

    pure = B - k * (temps - ideal) ** 2

    def flat(t):
        if t_min <= t <= t_max:
            return B
        d = t - t_max if t > t_max else t_min - t
        return B - k * d ** 2

    def gradient(t):
        if t_min <= t <= t_max:
            return B + Bc * (1 - abs(t - ideal) / ((t_max - t_min) / 2))
        d = t - t_max if t > t_max else t_min - t
        return B - k * d ** 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(t_min, t_max, color="#4daf4a", alpha=0.15,
               label=f"Faixa de conforto [{t_min:.0f}–{t_max:.0f} °C]")
    ax.plot(temps, pure, ls=":", color="#555555", lw=1.6, label="Quadrática pura")
    ax.plot(temps, [flat(t) for t in temps], ls="--", color="#3b6fb0", lw=1.8,
            label="Platô plano (estaciona na borda)")
    ax.plot(temps, [gradient(t) for t in temps], color="#e8801a", lw=3.0,
            label="Platô + gradiente (proposta)")
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_title("Funções de recompensa de conforto", fontsize=13)
    ax.set_xlabel("Temperatura interna (°C)")
    ax.set_ylabel(r"Recompensa de conforto $R_{conforto}$")
    ax.legend(fontsize=9, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(save_dir, "figura1_recompensa_conforto.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def plot_figure2(results: Dict[str, Dict], save_dir: str) -> None:
    """Figura 2: distribuição da temperatura na janela ocupada (violino+box)."""
    cfg = ClassroomConfig()
    labels = [k for k in ("Agressivo", "Equilibrado", "Passivo", "Termostato")
              if k in results and results[k]["temperatures"].size]
    if not labels:
        return
    data = [results[k]["temperatures"] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhspan(cfg.temp_comfort_min, cfg.temp_comfort_max, color="#4daf4a",
               alpha=0.12, label="Faixa larga [22, 26]")
    ax.axhspan(cfg.temp_narrow_min, cfg.temp_narrow_max, color="#4daf4a",
               alpha=0.22, label="Faixa estreita [23, 25]")
    ax.axhline(cfg.ideal_temp, color="#2f6b2f", ls="--", lw=1.2, label="Ideal (24 °C)")

    parts = ax.violinplot(data, showextrema=False)
    for body, lab in zip(parts["bodies"], labels):
        body.set_facecolor(PROFILE_COLORS.get(lab, "#888888"))
        body.set_alpha(0.55)
    ax.boxplot(data, widths=0.12, showfliers=False,
               medianprops={"color": "black"})

    for i, lab in enumerate(labels, start=1):
        dev = results[lab]["summary"]["abs_dev_from_ideal"]
        ax.text(i, ax.get_ylim()[1], f"|T−24|\n{dev:.2f} °C",
                ha="center", va="top", fontsize=8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Temperatura na janela ocupada (°C)")
    ax.set_title("Distribuição de temperatura por perfil (cenários controláveis)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(save_dir, "figura2_distribuicao_temperatura.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def plot_figure3(results: Dict[str, Dict], save_dir: str) -> None:
    """Figura 3: fronteira conforto × custo."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, res in results.items():
        s = res["summary"]
        marker = "X" if label == "Termostato" else "o"
        ax.scatter(s["cost_brl_day"], s["comfort_wide_pct"], s=90, marker=marker,
                   color=PROFILE_COLORS.get(label.split(" ")[0], "#888888"),
                   zorder=3)
        ax.annotate(label, (s["cost_brl_day"], s["comfort_wide_pct"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel("Custo médio (R$/dia) — menor é melhor")
    ax.set_ylabel("Conforto médio (%) — maior é melhor")
    ax.set_title("Trade-off conforto × custo (média da matriz 3×3)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(save_dir, "figura3_tradeoff_conforto_custo.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def plot_figure4(models_dir: str, save_dir: str, seed: int = 0) -> None:
    """Figura 4: DQN discreto vs. SAC contínuo no cenário quente, 45 pessoas."""
    scenario = next(s for s in build_scenario_matrix() if s["id"] == "C9")
    series: Dict[str, pd.DataFrame] = {}

    for path in sorted(glob.glob(os.path.join(models_dir, "*.zip"))):
        model, meta = _load_sb3(path)
        algo, profile = meta.get("algo", "DQN"), meta.get("profile", "Equilibrado")
        if meta.get("seed", 0) != seed:
            continue
        if algo == "DQN" and profile != "Equilibrado":
            continue
        cfg = config_for_profile(profile)
        env = make_factory(cfg, meta.get("action_repeat", 2), algo == "SAC")()
        label = "DQN (discreto, 4 níveis)" if algo == "DQN" else "SAC (potência contínua)"
        series[label] = run_episode(model, env, scenario, seed=seed)

    if not series:
        print("  (figura 4 exige um DQN Equilibrado e/ou um SAC treinados)")
        return

    cfg = ClassroomConfig()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    ax1.axhspan(cfg.temp_comfort_min, cfg.temp_comfort_max, color="#4daf4a",
                alpha=0.15, label="Faixa de conforto")
    for label, df in series.items():
        style = "-" if label.startswith("DQN") else "--"
        color = PROFILE_COLORS["Equilibrado"] if label.startswith("DQN") else PROFILE_COLORS["SAC"]
        ax1.plot(df["hour_float"], df["temperature"], style, color=color, lw=1.6, label=label)
        power = (
            df["load"] * 100 if "load" in df
            else df["action"].map({int(s): cfg.physics.load_fraction[s] * 100 for s in ACState})
        )
        ax2.step(df["hour_float"], power, where="post", color=color, lw=1.3,
                 ls=style, label=label)

    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title("Controle discreto (DQN) vs. contínuo (SAC) — cenário quente, 45 pessoas")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Potência do AC (%)")
    ax2.set_xlabel("Hora do dia")
    ax2.set_yticks([0, 33, 67, 100])
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(save_dir, "figura4_discreto_vs_continuo.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Avaliação reprodutível do paper.")
    p.add_argument("--models_dir", type=str, default="models_paper")
    p.add_argument("--save_dir", type=str, default=PLOTS_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generalization", action="store_true")
    p.add_argument("--figure1", action="store_true", help="só a Figura 1")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.figure1:
        print("Figuras:")
        plot_figure1(args.save_dir)
        return

    print("Avaliando na matriz 3×3 (Tabela 3)...")
    results = evaluate_all(args.models_dir, seed=args.seed)

    print("\n" + "=" * 98)
    print("TABELA 4 — desempenho médio na matriz 3×3 (janela ocupada 7h–22h)")
    print("=" * 98)
    print(format_table4(results))

    n = next(iter(results.values()))["summary"]
    print(f"\ncenários controláveis: {n['n_controllable']}/{n['n_scenarios']}")

    print("\nFiguras:")
    plot_figure1(args.save_dir)
    plot_figure2(results, args.save_dir)
    plot_figure3(results, args.save_dir)
    plot_figure4(args.models_dir, args.save_dir, seed=args.seed)

    if args.generalization:
        eq = [p for p in glob.glob(os.path.join(args.models_dir, "*Equilibrado*.zip"))
              if "SAC" not in os.path.basename(p)]
        if eq:
            model, meta = _load_sb3(eq[0])
            print("\n" + "=" * 60)
            print("TABELA 6 — generalização sem retreino")
            print("=" * 60)
            print(evaluate_generalization(model, "Equilibrado", args.seed)
                  .to_string(index=False))

    # Persiste o resultado por cenário para auditoria.
    out = os.path.join(args.save_dir, "resultados_por_cenario.csv")
    pd.concat(
        [r["per_scenario"].assign(agent=name) for name, r in results.items()]
    ).to_csv(out, index=False)
    print(f"\nresultados por cenário: {out}")


if __name__ == "__main__":
    main()
