# Relatório de Análise de Agentes - results_20251007_113750

Análise do comportamento de agentes de RL para controle de climatização.

## 📊 Resumo Comparativo (Cenário de Estresse)

![Resumo Comparativo](analysis_comparative_summary.png)

---

## 🔎 Agente: `dqn_baseline`

<details>
<summary><strong>Clique para ver Parâmetros de Configuração</strong></summary>

#### Parâmetros do Agente
|               |      Valor |
|:--------------|-----------:|
| state_size    |      4     |
| action_size   |      4     |
| episodes      |   2000     |
| buffer_size   | 100000     |
| batch_size    |     64     |
| gamma         |      0.99  |
| alpha         |      0.001 |
| tau           |      0.001 |
| update_every  |      4     |
| epsilon_start |      1     |
| epsilon_decay |      0.995 |
| epsilon_min   |      0.01  |

#### Parâmetros do Ambiente
|                                | Valor                                                                                   |
|:-------------------------------|:----------------------------------------------------------------------------------------|
| length                         | 8.0                                                                                     |
| width                          | 6.0                                                                                     |
| height                         | 3.0                                                                                     |
| max_occupancy                  | 30                                                                                      |
| thermal_mass                   | 1000.0                                                                                  |
| heat_transfer_coeff            | 0.5                                                                                     |
| heat_gain_per_person           | 0.1                                                                                     |
| temp_comfort_min               | 22.0                                                                                    |
| temp_comfort_max               | 26.0                                                                                    |
| temp_very_cold                 | 18.0                                                                                    |
| temp_very_hot                  | 30.0                                                                                    |
| initial_temp                   | 24.0                                                                                    |
| season                         | summer                                                                                  |
| energy_cost_peak_hours         | [18, 21]                                                                                |
| energy_penalty_multiplier_peak | 1.5                                                                                     |
| ac_cooling_power               | {'ACState.OFF': 0.0, 'ACState.LOW': 2.0, 'ACState.MEDIUM': 4.0, 'ACState.HIGH': 6.0}    |
| ac_energy_consumption          | {'ACState.OFF': 0.0, 'ACState.LOW': 1.5, 'ACState.MEDIUM': 3.0, 'ACState.HIGH': 5.0}    |
| reward_structure               | {'VERY_COLD': -15.0, 'COLD': -5.0, 'COMFORTABLE': 1.0, 'WARM': -5.0, 'VERY_HOT': -15.0} |

</details>

### Resumo Quantitativo

| Cenário                                 |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kWh) |   Recompensa Média |
|:----------------------------------------|-------------------:|------------------------:|----------------------:|-------------------:|
| 1: ManhÃ£ Fria, Sala Vazia              |              18.37 |                    0    |                  18   |              -5.6  |
| 2: ManhÃ£ AgradÃ¡vel, Sala Enchendo     |              22.26 |                   71.67 |                   1.8 |              -0.82 |
| 3: Tarde Quente, Sala Cheia             |              28.43 |                    0    |                  18   |              -5.18 |
| 4: Tarde Quente, Sala Superlotada       |              28.53 |                    0    |                  18   |              -5.18 |
| 5: Fim de Tarde, Sala Esvaziando        |              25.45 |                  100    |                   0   |               1    |
| 6: Noite, Sala Vazia (HorÃ¡rio de Pico) |              23.84 |                  100    |                   0   |               1    |
| 7: Onda de Calor Extrema (Estresse)     |              31.53 |                    0    |                  18   |             -15.18 |
| 8: Inverno HipotÃ©tico (InaÃ§Ã£o)       |              11.85 |                    0    |                  18   |             -15.18 |

### Gráficos de Comportamento

![Gráfico de Análise](analysis_dqn_baseline_scenario_1.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_2.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_3.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_4.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_5.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_6.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_7.png)
![Gráfico de Análise](analysis_dqn_baseline_scenario_8.png)
---

## 🔎 Agente: `q-learning_baseline`

<details>
<summary><strong>Clique para ver Parâmetros de Configuração</strong></summary>

#### Parâmetros do Agente
|                          |      Valor |
|:-------------------------|-----------:|
| alpha                    |     0.1    |
| gamma                    |     0.95   |
| epsilon                  |     1      |
| epsilon_min              |     0.01   |
| epsilon_decay            |     0.9995 |
| episodes                 | 10000      |
| early_stopping_patience  |   500      |
| early_stopping_threshold |     0.01   |

#### Parâmetros do Ambiente
|                                | Valor                                                                                   |
|:-------------------------------|:----------------------------------------------------------------------------------------|
| length                         | 8.0                                                                                     |
| width                          | 6.0                                                                                     |
| height                         | 3.0                                                                                     |
| max_occupancy                  | 30                                                                                      |
| thermal_mass                   | 1000.0                                                                                  |
| heat_transfer_coeff            | 0.5                                                                                     |
| heat_gain_per_person           | 0.1                                                                                     |
| temp_comfort_min               | 22.0                                                                                    |
| temp_comfort_max               | 26.0                                                                                    |
| temp_very_cold                 | 18.0                                                                                    |
| temp_very_hot                  | 30.0                                                                                    |
| initial_temp                   | 24.0                                                                                    |
| season                         | summer                                                                                  |
| energy_cost_peak_hours         | [18, 21]                                                                                |
| energy_penalty_multiplier_peak | 1.5                                                                                     |
| ac_cooling_power               | {'ACState.OFF': 0.0, 'ACState.LOW': 2.0, 'ACState.MEDIUM': 4.0, 'ACState.HIGH': 6.0}    |
| ac_energy_consumption          | {'ACState.OFF': 0.0, 'ACState.LOW': 1.5, 'ACState.MEDIUM': 3.0, 'ACState.HIGH': 5.0}    |
| reward_structure               | {'VERY_COLD': -15.0, 'COLD': -5.0, 'COMFORTABLE': 1.0, 'WARM': -5.0, 'VERY_HOT': -15.0} |

</details>

### Resumo Quantitativo

| Cenário                                 |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kWh) |   Recompensa Média |
|:----------------------------------------|-------------------:|------------------------:|----------------------:|-------------------:|
| 1: ManhÃ£ Fria, Sala Vazia              |              18.99 |                       0 |                 17    |              -6.22 |
| 2: ManhÃ£ AgradÃ¡vel, Sala Enchendo     |              23.77 |                     100 |                  4.85 |               0.6  |
| 3: Tarde Quente, Sala Cheia             |              27.65 |                       0 |                 26.65 |              -6.61 |
| 4: Tarde Quente, Sala Superlotada       |              27.92 |                       0 |                 26.5  |              -6.54 |
| 5: Fim de Tarde, Sala Esvaziando        |              24.92 |                     100 |                 10.45 |               0.4  |
| 6: Noite, Sala Vazia (HorÃ¡rio de Pico) |              24.85 |                     100 |                 13.8  |               0.35 |
| 7: Onda de Calor Extrema (Estresse)     |              32.5  |                       0 |                  0    |             -15    |
| 8: Inverno HipotÃ©tico (InaÃ§Ã£o)       |              12.32 |                       0 |                  0    |             -15    |

### Gráficos de Comportamento

![Gráfico de Análise](analysis_q-learning_baseline_scenario_1.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_2.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_3.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_4.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_5.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_6.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_7.png)
![Gráfico de Análise](analysis_q-learning_baseline_scenario_8.png)
---

## 🔎 Agente: `q-learning_low_lr`

<details>
<summary><strong>Clique para ver Parâmetros de Configuração</strong></summary>

#### Parâmetros do Agente
|                          |      Valor |
|:-------------------------|-----------:|
| alpha                    |     0.01   |
| gamma                    |     0.95   |
| epsilon                  |     1      |
| epsilon_min              |     0.01   |
| epsilon_decay            |     0.9995 |
| episodes                 | 10000      |
| early_stopping_patience  |   500      |
| early_stopping_threshold |     0.01   |

#### Parâmetros do Ambiente
|                                | Valor                                                                                   |
|:-------------------------------|:----------------------------------------------------------------------------------------|
| length                         | 8.0                                                                                     |
| width                          | 6.0                                                                                     |
| height                         | 3.0                                                                                     |
| max_occupancy                  | 30                                                                                      |
| thermal_mass                   | 1000.0                                                                                  |
| heat_transfer_coeff            | 0.5                                                                                     |
| heat_gain_per_person           | 0.1                                                                                     |
| temp_comfort_min               | 22.0                                                                                    |
| temp_comfort_max               | 26.0                                                                                    |
| temp_very_cold                 | 18.0                                                                                    |
| temp_very_hot                  | 30.0                                                                                    |
| initial_temp                   | 24.0                                                                                    |
| season                         | summer                                                                                  |
| energy_cost_peak_hours         | [18, 21]                                                                                |
| energy_penalty_multiplier_peak | 1.5                                                                                     |
| ac_cooling_power               | {'ACState.OFF': 0.0, 'ACState.LOW': 2.0, 'ACState.MEDIUM': 4.0, 'ACState.HIGH': 6.0}    |
| ac_energy_consumption          | {'ACState.OFF': 0.0, 'ACState.LOW': 1.5, 'ACState.MEDIUM': 3.0, 'ACState.HIGH': 5.0}    |
| reward_structure               | {'VERY_COLD': -15.0, 'COLD': -5.0, 'COMFORTABLE': 1.0, 'WARM': -5.0, 'VERY_HOT': -15.0} |

</details>

### Resumo Quantitativo

| Cenário                                 |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kWh) |   Recompensa Média |
|:----------------------------------------|-------------------:|------------------------:|----------------------:|-------------------:|
| 1: ManhÃ£ Fria, Sala Vazia              |              19.03 |                    0    |                  3.3  |              -5.3  |
| 2: ManhÃ£ AgradÃ¡vel, Sala Enchendo     |              22.65 |                   88.33 |                  5.55 |              -0.25 |
| 3: Tarde Quente, Sala Cheia             |              28.41 |                    0    |                  5.65 |              -5.64 |
| 4: Tarde Quente, Sala Superlotada       |              29.09 |                    0    |                  1.8  |              -5.25 |
| 5: Fim de Tarde, Sala Esvaziando        |              24.75 |                  100    |                  0.45 |               0.9  |
| 6: Noite, Sala Vazia (HorÃ¡rio de Pico) |              23.94 |                  100    |                  0    |               1    |
| 7: Onda de Calor Extrema (Estresse)     |              32.49 |                    0    |                  0    |             -15    |
| 8: Inverno HipotÃ©tico (InaÃ§Ã£o)       |              12.66 |                    0    |                  0    |             -15    |

### Gráficos de Comportamento

![Gráfico de Análise](analysis_q-learning_low_lr_scenario_1.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_2.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_3.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_4.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_5.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_6.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_7.png)
![Gráfico de Análise](analysis_q-learning_low_lr_scenario_8.png)
