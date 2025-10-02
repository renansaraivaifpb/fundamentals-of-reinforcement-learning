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
| 1: Manhã Fria, Sala Vazia                       |              18.16 |                    0    |                113.5 |              -9.78 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              22.12 |                   71.67 |                223   |              -2.27 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.13 |                    0    |                318.5 |              -6.83 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              29.28 |                    0    |                196   |              -7.11 | LOW                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.16 |                  100    |                157.5 |              -0.26 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.5  |                  100    |                 30   |               0.62 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              25.2  |                   77.5  |                137   |              -1.33 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.09 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.18 |                  100    |                165.5 |              -0.24 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.49 |                    0    |                  0   |             -15    | OFF                 |

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
| 1: Manhã Fria, Sala Vazia                       |              18.19 |                       0 |                126   |             -10.35 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              24.23 |                     100 |                 49   |               0.48 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.74 |                       0 |                272   |              -6.78 | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              29.37 |                       0 |                207.5 |              -6.67 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.82 |                     100 |                119   |              -0.02 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.03 |                     100 |                 39   |               0.53 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.3  |                     100 |                 21.5 |               0.75 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.31 |                       0 |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.37 |                      95 |                162.5 |              -0.67 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.58 |                       0 |                  0   |             -15    | OFF                 |

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
| 1: Manhã Fria, Sala Vazia                       |              19.33 |                    0    |                268.5 |              -6.62 | MED                 |
| 2: Manhã Agradável, Sala Enchendo               |              22.78 |                   95.83 |                156.5 |              -0.33 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.02 |                    0    |                286   |              -6.76 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              29.91 |                    0    |                125   |              -9.72 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              23.85 |                  100    |                 28.5 |               0.68 | OFF                 |
| 6: Noite, Sala Vazia                            |              23.29 |                  100    |                 98.5 |               0.18 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.08 |                  100    |                 17   |               0.82 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.6  |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.86 |                   66.67 |                279.5 |              -2.75 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12.61 |                    0    |                  0   |             -15    | OFF                 |

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
| 1: Manhã Fria, Sala Vazia                       |              18.15 |                    0    |                166   |             -10.54 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              22.23 |                   60    |                 90.5 |              -2.18 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              29    |                    0    |                177.5 |              -6.4  | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              29.14 |                    0    |                191   |              -6.16 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.79 |                  100    |                  0   |               1    | OFF                 |
| 6: Noite, Sala Vazia                            |              25.13 |                  100    |                  0   |               1    | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.84 |                  100    |                  0   |               1    | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.08 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.5  |                   71.67 |                 67   |              -1.11 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.57 |                    0    |                  0   |             -15    | OFF                 |

### 3. Gráficos de Comportamento em Cenários

![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part1.png)
![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part2.png)
