# Relatório de Análise de Agentes - results_20251002_174127

Este relatório documenta o comportamento de cada agente treinado sob um conjunto de 10 cenários de teste.

---

## 🔎 Análise do Agente: `baseline`

### Resumo Quantitativo

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              19.55 |                    0    |                327.5 |              -6.69 | MED                 |
| 2: Manhã Agradável, Sala Enchendo               |              21.89 |                   35.83 |                228   |              -4.46 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.3  |                    0    |                288   |              -6.64 | OFF                 |
| 4: Tarde Quente, Sala Superlotada               |              28.3  |                    0    |                256.5 |              -6.83 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              24.34 |                  100    |                 18   |               0.68 | OFF                 |
| 6: Noite, Sala Vazia                            |              24.26 |                  100    |                 49.5 |               0.46 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.05 |                  100    |                 19   |               0.78 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.4  |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.12 |                  100    |                145.5 |              -0.19 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.49 |                    0    |                  0   |             -15    | OFF                 |

### Gráficos de Comportamento

![Gráfico de Análise para baseline](analysis_scenarios_baseline_part1.png)
![Gráfico de Análise para baseline](analysis_scenarios_baseline_part2.png)

---

## 🔎 Análise do Agente: `high_discount_factor`

### Resumo Quantitativo

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.04 |                    0    |                109   |             -11.01 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.26 |                  100    |                 31   |               0.64 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.44 |                    0    |                266.5 |              -6.56 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              28.33 |                    0    |                223.5 |              -6.47 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.15 |                  100    |                131   |              -0.13 | OFF                 |
| 6: Noite, Sala Vazia                            |              22.95 |                  100    |                 32.5 |               0.57 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.6  |                  100    |                 51.5 |               0.56 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              32.33 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.13 |                   96.67 |                149   |              -0.27 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.44 |                    0    |                  0   |             -15    | OFF                 |

### Gráficos de Comportamento

![Gráfico de Análise para high_discount_factor](analysis_scenarios_high_discount_factor_part1.png)
![Gráfico de Análise para high_discount_factor](analysis_scenarios_high_discount_factor_part2.png)

---

## 🔎 Análise do Agente: `high_exploration`

### Resumo Quantitativo

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.28 |                     0   |                148.5 |              -9.96 | OFF                 |
| 2: Manhã Agradável, Sala Enchendo               |              23.45 |                   100   |                 67   |               0.33 | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              28.23 |                     0   |                263   |              -6.55 | MED                 |
| 4: Tarde Quente, Sala Superlotada               |              28.7  |                     0   |                272.5 |              -6.64 | MED                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.37 |                    92.5 |                199.5 |              -0.73 | OFF                 |
| 6: Noite, Sala Vazia                            |              23.49 |                   100   |                 64   |               0.38 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.12 |                   100   |                 13   |               0.86 | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              31.35 |                     0   |                  7.5 |             -15.14 | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.08 |                   100   |                161.5 |               0.05 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              12    |                     0   |                  0   |             -15    | OFF                 |

### Gráficos de Comportamento

![Gráfico de Análise para high_exploration](analysis_scenarios_high_exploration_part1.png)
![Gráfico de Análise para high_exploration](analysis_scenarios_high_exploration_part2.png)

---

## 🔎 Análise do Agente: `low_learning_rate`

### Resumo Quantitativo

| Cenário                                         |   Temp. Média (°C) |   Tempo em Conforto (%) |   Energia Total (kW) |   Recompensa Média | Ação Predominante   |
|:------------------------------------------------|-------------------:|------------------------:|---------------------:|-------------------:|:--------------------|
| 1: Manhã Fria, Sala Vazia                       |              18.44 |                    0    |                220.5 |              -7.05 | LOW                 |
| 2: Manhã Agradável, Sala Enchendo               |              22.33 |                   69.17 |                 80   |              -1.4  | OFF                 |
| 3: Tarde Quente, Sala Cheia                     |              27.5  |                    0    |                259.5 |              -6.87 | LOW                 |
| 4: Tarde Quente, Sala Superlotada               |              29.39 |                    0    |                190   |              -6.94 | OFF                 |
| 5: Fim de Tarde, Sala Esvaziando                |              25.56 |                   93.33 |                 15.5 |               0.42 | OFF                 |
| 6: Noite, Sala Vazia                            |              22.42 |                   59.17 |                 80.5 |              -2.12 | OFF                 |
| 7: Tarde Agradável, Sala Vazia (Manutenção)     |              24.66 |                  100    |                  0   |               1    | OFF                 |
| 8: Onda de Calor Extrema, Sala Cheia (Estresse) |              33.41 |                    0    |                  0   |             -15    | OFF                 |
| 9: Fim de Expediente (Eficiência Energética)    |              25.86 |                   61.67 |                104.5 |              -1.97 | OFF                 |
| 10: Inverno Hipotético (Caso de Borda/Inação)   |              11.44 |                    0    |                  0   |             -15    | OFF                 |

### Gráficos de Comportamento

![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part1.png)
![Gráfico de Análise para low_learning_rate](analysis_scenarios_low_learning_rate_part2.png)
