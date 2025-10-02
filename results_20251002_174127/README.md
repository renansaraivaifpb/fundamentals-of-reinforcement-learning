# Relatório de Análise de Agentes - results_20251002_174127

Este relatório documenta o comportamento de cada agente treinado sob um conjunto de 10 cenários de teste.

---

## 📊 Análise Comparativa Geral

A imagem abaixo compara o desempenho final de todos os agentes durante a fase de avaliação.

![Análise Comparativa Geral](experiment_comparison.png)

---

## 🔎 Análise Detalhada do Agente: `baseline`

### 1. Parâmetros de Configuração

#### Parâmetros do Agente
| Hiperparâmetro      |     Valor |
|:--------------------|----------:|
| alpha               |     0.1   |
| gamma               |     0.95  |
| epsilon             |     0.2   |
| epsilon_min         |     0.01  |
| epsilon_decay       |     0.995 |
| episodes            | 10000     |
| temperature_penalty |     0.5   |
| energy_penalty      |     0.1   |
| comfort_bonus       |     2     |
| use_epsilon_decay   |     1     |
| use_optimistic_init |     1     |
| optimistic_value    |     1     |


#### Parâmetros do Ambiente
| Parâmetro            |   Valor |
|:---------------------|--------:|
| length               |     8   |
| width                |     6   |
| height               |     3   |
| max_occupancy        |    30   |
| thermal_mass         |  1000   |
| heat_transfer_coeff  |     0.5 |
| heat_gain_per_person |     0.1 |
| temp_comfort_min     |    22   |
| temp_comfort_max     |    26   |
| temp_very_cold       |    18   |
| temp_very_hot        |    30   |
| initial_temp         |    24   |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.87 |                       0 |                206   |              -6.77 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.3  |                     100 |                 74.5 |               0.35 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              29.76 |                       0 |                120.5 |             -10.77 | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              28.3  |                       0 |                305.5 |              -6.72 | MED                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.76 |                     100 |                101   |               0.17 | OFF                 |
| 6: Noite, Sala Vazia                            |              23.78 |                     100 |                 16.5 |               0.75 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.77 |                     100 |                 43   |               0.58 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.85 |                       0 |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.19 |                     100 |                161   |              -0.18 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.62 |                       0 |                  0   |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para baseline](analysis_scenarios_baseline_part1.png)
![Gráfico de Análise para baseline](analysis_scenarios_baseline_part2.png)

---

## 🔎 Análise Detalhada do Agente: `high_discount_factor`

### 1. Parâmetros de Configuração

#### Parâmetros do Agente
| Hiperparâmetro      |     Valor |
|:--------------------|----------:|
| alpha               |     0.1   |
| gamma               |     0.99  |
| epsilon             |     0.2   |
| epsilon_min         |     0.01  |
| epsilon_decay       |     0.995 |
| episodes            | 10000     |
| temperature_penalty |     0.5   |
| energy_penalty      |     0.1   |
| comfort_bonus       |     2     |
| use_epsilon_decay   |     1     |
| use_optimistic_init |     1     |
| optimistic_value    |     1     |

#### Parâmetros do Ambiente
| Parâmetro            |   Valor |
|:---------------------|--------:|
| length               |     8   |
| width                |     6   |
| height               |     3   |
| max_occupancy        |    30   |
| thermal_mass         |  1000   |
| heat_transfer_coeff  |     0.5 |
| heat_gain_per_person |     0.1 |
| temp_comfort_min     |    22   |
| temp_comfort_max     |    26   |
| temp_very_cold       |    18   |
| temp_very_hot        |    30   |
| initial_temp         |    24   |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              20.41 |                    0    |                262.5 |              -6.72 | MED                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.02 |                  100    |                 37.5 |               0.45 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.74 |                    0    |                261   |              -7.1  | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              29.47 |                    0    |                175   |              -7.05 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.81 |                  100    |                 86.5 |               0.13 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.58 |                  100    |                 84.5 |               0.23 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.87 |                  100    |                 89   |               0.34 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.55 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              26.32 |                   31.67 |                237   |              -4.63 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.08 |                    0    |                  0   |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para high_discount_factor](analysis_scenarios_high_discount_factor_part1.png)
![Gráfico de Análise para high_discount_factor](analysis_scenarios_high_discount_factor_part2.png)

---

## 🔎 Análise Detalhada do Agente: `high_exploration`

### 1. Parâmetros de Configuração

#### Parâmetros do Agente
| Hiperparâmetro      |    Valor |
|:--------------------|---------:|
| alpha               |     0.1  |
| gamma               |     0.95 |
| epsilon             |     0.5  |
| epsilon_min         |     0.01 |
| epsilon_decay       |     0.99 |
| episodes            | 10000    |
| temperature_penalty |     0.5  |
| energy_penalty      |     0.1  |
| comfort_bonus       |     2    |
| use_epsilon_decay   |     1    |
| use_optimistic_init |     1    |
| optimistic_value    |     1    |

#### Parâmetros do Ambiente
| Parâmetro            |   Valor |
|:---------------------|--------:|
| length               |     8   |
| width                |     6   |
| height               |     3   |
| max_occupancy        |    30   |
| thermal_mass         |  1000   |
| heat_transfer_coeff  |     0.5 |
| heat_gain_per_person |     0.1 |
| temp_comfort_min     |    22   |
| temp_comfort_max     |    26   |
| temp_very_cold       |    18   |
| temp_very_hot        |    30   |
| initial_temp         |    24   |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              20.4  |                       0 |                225   |              -6.69 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.7  |                     100 |                 42.5 |               0.65 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.03 |                       0 |                275   |              -6.65 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              29.88 |                       0 |                158   |              -8.57 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.1  |                     100 |                197   |              -0.11 | OFF                 |
| 6: Noite, Sala Vazia                            |              23.64 |                     100 |                 41.5 |               0.57 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              23.94 |                     100 |                 34.5 |               0.64 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.99 |                       0 |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              24.88 |                     100 |                100   |               0.23 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.19 |                       0 |                  0   |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para high_exploration](analysis_scenarios_high_exploration_part1.png)
![Gráfico de Análise para high_exploration](analysis_scenarios_high_exploration_part2.png)

---

## 🔎 Análise Detalhada do Agente: `low_learning_rate`

### 1. Parâmetros de Configuração

#### Parâmetros do Agente
| Hiperparâmetro      |     Valor |
|:--------------------|----------:|
| alpha               |     0.05  |
| gamma               |     0.95  |
| epsilon             |     0.2   |
| epsilon_min         |     0.01  |
| epsilon_decay       |     0.995 |
| episodes            | 10000     |
| temperature_penalty |     0.5   |
| energy_penalty      |     0.1   |
| comfort_bonus       |     2     |
| use_epsilon_decay   |     1     |
| use_optimistic_init |     1     |
| optimistic_value    |     1     |

#### Parâmetros do Ambiente
| Parâmetro            |   Valor |
|:---------------------|--------:|
| length               |     8   |
| width                |     6   |
| height               |     3   |
| max_occupancy        |    30   |
| thermal_mass         |  1000   |
| heat_transfer_coeff  |     0.5 |
| heat_gain_per_person |     0.1 |
| temp_comfort_min     |    22   |
| temp_comfort_max     |    26   |
| temp_very_cold       |    18   |
| temp_very_hot        |    30   |
| initial_temp         |    24   |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.45 |                    0    |                  179 |              -8.23 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.11 |                  100    |                    0 |               1    | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.94 |                    0    |                  254 |              -6.56 | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              28.17 |                    0    |                  300 |              -6.72 | LOW                 |
| 5: Fim de Tarde, Sala Esvaziando                |              23.68 |                  100    |                    0 |               1    | OFF                 |
| 6: Noite, Sala Vazia                            |              23.59 |                  100    |                    0 |               1    | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              25.04 |                  100    |                    3 |               0.96 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.35 |                    0    |                    6 |             -15.07 | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              27.15 |                    4.17 |                  253 |              -6.39 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.17 |                    0    |                    0 |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part1.png)
![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part2.png)
