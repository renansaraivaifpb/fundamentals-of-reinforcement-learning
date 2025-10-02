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

#### Estrutura de Recompensa
| Nível de Conforto   |   Valor |
|:--------------------|--------:|
| VERY_COLD           |     -15 |
| COLD                |      -5 |
| COMFORTABLE         |       1 |
| WARM                |      -5 |
| VERY_HOT            |     -15 |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              19.8  |                    0    |                283.5 |              -6.67 | MED                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.22 |                  100    |                 32   |               0.62 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.42 |                    0    |                302.5 |              -6.77 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              28.83 |                    0    |                263.5 |              -6.64 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.42 |                  100    |                 55.5 |               0.47 | OFF                 |
| 6: Noite, Sala Vazia                            |              22.94 |                   76.67 |                132   |              -1.23 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.32 |                  100    |                 23   |               0.61 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.38 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              26.39 |                   27.5  |                259.5 |              -4.98 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.43 |                    0    |                  0   |             -15    | OFF                 |

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

#### Estrutura de Recompensa
| Nível de Conforto   |   Valor |
|:--------------------|--------:|
| VERY_COLD           |     -15 |
| COLD                |      -5 |
| COMFORTABLE         |       1 |
| WARM                |      -5 |
| VERY_HOT            |     -15 |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.95 |                    0    |                255.5 |              -6.75 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              22.55 |                   85.83 |                 64   |              -0.59 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.36 |                    0    |                290.5 |              -6.94 | HIGH                |
| 4: Tarde Quente, Sala Superlotada               |              28.61 |                    0    |                287.5 |              -6.66 | MED                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.08 |                  100    |                 35   |               0.69 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.41 |                  100    |                 35   |               0.65 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.21 |                  100    |                 56   |               0.45 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.86 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.18 |                  100    |                146   |              -0.17 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.97 |                    0    |                  0   |             -15    | OFF                 |

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

#### Estrutura de Recompensa
| Nível de Conforto   |   Valor |
|:--------------------|--------:|
| VERY_COLD           |     -15 |
| COLD                |      -5 |
| COMFORTABLE         |       1 |
| WARM                |      -5 |
| VERY_HOT            |     -15 |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              19.02 |                       0 |                257   |              -6.66 | LOW                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.15 |                     100 |                134.5 |               0.09 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.49 |                       0 |                308.5 |              -6.69 | HIGH                |
| 4: Tarde Quente, Sala Superlotada               |              29.53 |                       0 |                181.5 |              -6.9  | LOW                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.16 |                     100 |                213   |              -0.34 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.99 |                     100 |                175.5 |              -0.13 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.67 |                     100 |                 73.5 |               0.47 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              30.81 |                       0 |                 54.5 |             -13.5  | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              24.89 |                     100 |                134   |               0.17 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.79 |                       0 |                  0   |             -15    | OFF                 |

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

#### Estrutura de Recompensa
| Nível de Conforto   |   Valor |
|:--------------------|--------:|
| VERY_COLD           |     -15 |
| COLD                |      -5 |
| COMFORTABLE         |       1 |
| WARM                |      -5 |
| VERY_HOT            |     -15 |

### 2. Resumo Quantitativo de Desempenho

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.8  |                    0    |                246   |              -7.36 | MED                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.14 |                  100    |                  0   |               1    | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              26.66 |                   22.5  |                197   |              -5.01 | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              28.55 |                    0    |                235   |              -6.55 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.02 |                  100    |                  0   |               1    | OFF                 |
| 6: Noite, Sala Vazia                            |              24.63 |                  100    |                  3   |               0.96 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.63 |                  100    |                  0   |               1    | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.29 |                    0    |                 16.5 |             -15.21 | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              26.06 |                   35.83 |                180   |              -3.87 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.51 |                    0    |                  0   |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part1.png)
![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part2.png)
