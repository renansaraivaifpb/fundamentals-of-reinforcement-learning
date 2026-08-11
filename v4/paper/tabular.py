# -*- coding: utf-8 -*-
"""
Q-Learning tabular — responde à dúvida do Revisor 1 sobre necessidade de RL
profundo, e testa empiricamente a tese do HNP (Zha et al. 2021).

    R1: "O espaço de estados, embora contínuo, possui baixa dimensionalidade.
         Assim, não é claro que a solução realmente precisa de abordagem
         profunda (e mesmo contínua). Tal fator também deveria ser melhor
         investigado."

A pergunta é legítima: 4 dimensões é pouco, e uma tabela deveria bastar. Mas há
um motivo TEÓRICO para não bastar, e ele vem de Zha et al. (2021):

    Em espaços contínuos com variáveis de dinâmica lenta, métodos tabulares com
    discretização falham porque a transição é INTRA-TILE — a ação aterrissa no
    mesmo tile de onde partiu. O valor nunca se propaga entre tiles, e o modelo
    trata a variável lenta como constante.

Este ambiente é o caso exemplar: a temperatura muda ~0,13 °C por passo. Com uma
grade de 1 °C, ~87 % das transições são intra-tile. `intra_tile_fraction` mede
isso empiricamente — é a evidência que conecta a dúvida do revisor à literatura.

O experimento tem dois desfechos, ambos informativos:
  - tabular COMPETE  -> o revisor está certo, RL profundo é desnecessário aqui;
  - tabular FALHA    -> a profundidade se justifica, e o HNP explica por quê.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ac_physics import ACState
from config import ClassroomConfig


@dataclass
class TabularConfig:
    """Discretização do espaço de estados."""
    temp_bins: int = 20        # 15–35 °C em passos de 1 °C
    occupancy_bins: int = 10
    hour_bins: int = 24

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5

    @property
    def n_states(self) -> int:
        return self.temp_bins * self.occupancy_bins * self.hour_bins


class TabularQAgent:
    """
    Q-Learning tabular com discretização uniforme.

    Interface `predict` compatível com os modelos SB3, para que a mesma função
    de avaliação sirva aos dois — condição para a comparação ser justa.
    """

    def __init__(self, config: ClassroomConfig, tab: Optional[TabularConfig] = None,
                 seed: int = 0):
        self.config = config
        self.tab = tab or TabularConfig()
        self.n_actions = len(ACState)
        self.q = np.zeros((self.tab.n_states, self.n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.epsilon = self.tab.epsilon_start
        self.visits = np.zeros(self.tab.n_states, dtype=np.int64)

    # ------------------------------------------------------------ discretização

    def _bins(self, obs: np.ndarray) -> Tuple[int, int, int]:
        obs = np.asarray(obs).reshape(-1)
        t_norm, o_norm = float(obs[0]), float(obs[1])
        # Reconstrói a hora do par (sin, cos): atan2 preserva o quadrante.
        angle = np.arctan2(float(obs[2]), float(obs[3]))
        hour = (angle / (2 * np.pi) * 24.0) % 24.0

        ti = min(int(t_norm * self.tab.temp_bins), self.tab.temp_bins - 1)
        oi = min(int(o_norm * self.tab.occupancy_bins), self.tab.occupancy_bins - 1)
        hi = min(int(hour / 24.0 * self.tab.hour_bins), self.tab.hour_bins - 1)
        return ti, oi, hi

    def state_index(self, obs: np.ndarray) -> int:
        ti, oi, hi = self._bins(obs)
        return (ti * self.tab.occupancy_bins + oi) * self.tab.hour_bins + hi

    # ------------------------------------------------------------------- política

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        s = self.state_index(obs)
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions)), None
        return int(np.argmax(self.q[s])), None

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------ treino

    def learn(self, env, total_timesteps: int, verbose: bool = True) -> Dict:
        decay_steps = max(1, int(total_timesteps * self.tab.epsilon_decay_fraction))
        obs, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        intra_tile = 0
        transitions = 0

        for step in range(total_timesteps):
            self.epsilon = max(
                self.tab.epsilon_end,
                self.tab.epsilon_start
                - (self.tab.epsilon_start - self.tab.epsilon_end) * step / decay_steps,
            )
            s = self.state_index(obs)
            self.visits[s] += 1
            action, _ = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            s_next = self.state_index(next_obs)

            # Mede a patologia descrita pelo HNP.
            transitions += 1
            if s == s_next:
                intra_tile += 1

            target = reward + (0.0 if terminated else self.tab.gamma * self.q[s_next].max())
            self.q[s, action] += self.tab.alpha * (target - self.q[s, action])

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

            if verbose and (step + 1) % 100_000 == 0:
                cov = (self.visits > 0).mean() * 100
                print(f"    {step+1:>7,} passos | eps={self.epsilon:.3f} | "
                      f"cobertura={cov:.1f}% | intra-tile={intra_tile/transitions*100:.1f}%")

        return {
            "intra_tile_fraction": intra_tile / max(1, transitions),
            "state_coverage": float((self.visits > 0).mean()),
            "n_states": self.tab.n_states,
            "states_visited": int((self.visits > 0).sum()),
        }


def intra_tile_fraction(config: ClassroomConfig, tab: Optional[TabularConfig] = None,
                        n_steps: int = 20_000, seed: int = 0) -> Dict[str, float]:
    """
    Mede a fração de transições que NÃO saem do tile de origem, sob política
    aleatória. É o diagnóstico do HNP, medido antes de treinar qualquer coisa.

    Se essa fração for alta, Q-Learning tabular não pode propagar valor: o
    backup atualiza um estado com o valor dele mesmo.
    """
    from env import ClassroomACEnv

    agent = TabularQAgent(config, tab, seed=seed)
    env = ClassroomACEnv(config=config)
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)
    intra = 0
    for _ in range(n_steps):
        s = agent.state_index(obs)
        obs, _, term, _, _ = env.step(int(rng.integers(len(ACState))))
        if agent.state_index(obs) == s:
            intra += 1
        if term:
            obs, _ = env.reset()

    dt_per_step = config.dt
    typical_dtemp = (
        config.max_occupancy * config.heat_gain_per_person / config.thermal_mass
    ) * dt_per_step
    bin_width = 20.0 / (tab or TabularConfig()).temp_bins

    return {
        "intra_tile_pct": intra / n_steps * 100.0,
        "typical_dtemp_per_step": typical_dtemp,
        "temp_bin_width": bin_width,
        "steps_to_cross_bin": bin_width / max(typical_dtemp, 1e-9),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Q-Learning tabular vs. DQN.")
    p.add_argument("--timesteps", type=int, default=550_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--profile", type=str, default="Equilibrado")
    args = p.parse_args()

    import pandas as pd

    from config import config_for_profile
    from env import ClassroomACEnv
    from metrics import evaluate_agent
    from scenarios import build_scenario_matrix
    from wrappers import ActionRepeatWrapper

    cfg = config_for_profile(args.profile)
    tab = TabularConfig()

    print("=" * 78)
    print("DIAGNÓSTICO HNP (Zha et al. 2021) — antes de treinar")
    print("=" * 78)
    diag = intra_tile_fraction(cfg, tab)
    print(f"  espaço discretizado: {tab.n_states:,} estados "
          f"({tab.temp_bins}×{tab.occupancy_bins}×{tab.hour_bins})")
    print(f"  largura do bin de temperatura: {diag['temp_bin_width']:.2f} °C")
    print(f"  ΔT típico por passo (sala cheia): {diag['typical_dtemp_per_step']:.4f} °C")
    print(f"  passos para cruzar um bin: {diag['steps_to_cross_bin']:.1f}")
    print(f"  -> TRANSIÇÕES INTRA-TILE: {diag['intra_tile_pct']:.1f}%")
    print("     (fração de transições em que o estado discreto não muda; alta "
          "=> valor não se propaga)")

    scenarios = build_scenario_matrix()
    factory = lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
    rows: List[Dict] = []

    for seed in args.seeds:
        print(f"\n  treinando tabular (semente {seed}, {args.timesteps:,} passos)...")
        agent = TabularQAgent(cfg, tab, seed=seed)
        stats = agent.learn(factory(), args.timesteps)
        res = evaluate_agent(agent, factory, scenarios, cfg, seed=seed)
        s = res["summary"]
        rows.append({
            "seed": seed,
            "conf_larga_pct": s["comfort_wide_pct"],
            "conf_estreita_pct": s["comfort_narrow_pct"],
            "desvio_ideal": s["abs_dev_from_ideal"],
            "energia_kwh": s["energy_kwh_day"],
            "cobertura_estados": stats["state_coverage"] * 100,
            "intra_tile_treino_pct": stats["intra_tile_fraction"] * 100,
        })
        print(f"    -> conf {s['comfort_wide_pct']:.1f}% | "
              f"|T-24| {s['abs_dev_from_ideal']:.2f} | "
              f"cobertura {stats['state_coverage']*100:.1f}%")

    df = pd.DataFrame(rows).round(2)
    print("\n" + df.to_string(index=False))
    df.to_csv("results_tabular.csv", index=False)
    print("\nresultados em results_tabular.csv")
    print("\nCompare com DQN (86,5 % conf, |T-24| 0,89) e PI (86,5 %, 0,72).")


if __name__ == "__main__":
    main()
