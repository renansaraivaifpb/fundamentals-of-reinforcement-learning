# analyzer_grid.py

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # Biblioteca para visualização estatística

# A função de análise é modular, facilitando a reutilização e o entendimento
def analyze_grid_results(base_models_dir: str, base_logs_dir: str):
    """
    Analisa os logs de treinamento de uma busca em grade e gera um heatmap
    de desempenho comparativo entre agentes e cenários.

    Args:
        base_models_dir (str): Diretório onde estão os arquivos de configuração .json.
        base_logs_dir (str): Diretório onde estão os arquivos de log monitor.csv.
    """
    print("Iniciando a análise dos resultados da busca em grade...")

    summary_data = []

    # Passo 1: Iterar sobre os arquivos de configuração
    for f in os.listdir(base_models_dir):
        # Processa apenas arquivos de configuração .json
        if not f.endswith("_config.json"):
            continue

        config_path = os.path.join(base_models_dir, f)

        try:
            with open(config_path, 'r', encoding='utf-8') as j:
                cfg = json.load(j)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"AVISO: Falha ao carregar o arquivo de configuração {f}. Erro: {e}")
            continue
        
        # Constrói o caminho para o arquivo de log monitor.csv
        # A lógica de nome do diretório deve corresponder à lógica do seu script de treinamento
        log_dir_name = f.replace('_config.json', '')
        log_path = os.path.join(base_logs_dir, log_dir_name, "monitor.csv")

        # Passo 2: Carregar os dados de log e calcular a recompensa média
        if not os.path.exists(log_path):
            print(f"AVISO: Log não encontrado para o experimento: {log_dir_name}. Pulando.")
            continue
        
        try:
            df = pd.read_csv(log_path, skiprows=1) # A maioria dos logs sb3 tem 1 linha de cabeçalho extra
            if 'r' not in df.columns:
                print(f"AVISO: Coluna 'r' (recompensa) não encontrada em {log_path}. Pulando.")
                continue
            
            mean_reward = df["r"].mean()
            
            # Adicione outras métricas, se necessário
            # mean_length = df["l"].mean()
            # ...

            # Coleta os dados em uma lista para facilitar a criação do DataFrame
            summary_data.append([
                cfg.get("ac_setup_name", "N/A"), # Usar .get() evita KeyError
                cfg.get("agent_personality_name", "N/A"),
                mean_reward
            ])

        except Exception as e:
            print(f"ERRO: Falha ao processar o log {log_path}. Erro: {e}")
            continue

    if not summary_data:
        print("Nenhum dado válido encontrado para análise. Verifique os diretórios e arquivos.")
        return

    # Passo 3: Criar um DataFrame e a Tabela Dinâmica (Pivot Table)
    df_summary = pd.DataFrame(
        summary_data, 
        columns=["AC", "Agente", "Recompensa Média"]
    )

    pivot_table = df_summary.pivot(index="AC", columns="Agente", values="Recompensa Média")

    # Passo 4: Gerar o Heatmap
    plt.figure(figsize=(12, 8))
    plt.title("Desempenho Médio (Recompensa) por Combinação de Agente e AC", fontsize=16)
    
    # seaborn.heatmap é ideal para isso
    sns.heatmap(
        pivot_table,
        annot=True,       # Exibe os valores na célula
        fmt=".2f",        # Formato de 2 casas decimais
        cmap="viridis",   # Escolha um mapa de cores agradável
        linewidths=.5     # Adiciona linhas entre as células para melhor visualização
    )
    
    plt.xlabel("Agente", fontsize=12)
    plt.ylabel("Configuração do AC", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Ajusta o layout para evitar que os rótulos se sobreponham
    plt.show()

if __name__ == '__main__':
    MODELS_DIR = "C:/Users/engre/Documents/reinforcement/air_conditioning_classroom/v4/models_v4"
    LOGS_DIR = "C:/Users/engre/Documents/reinforcement/air_conditioning_classroom/v4/sb3_logs_v4"
    
    # Verifica se os diretórios existem antes de iniciar
    if not os.path.exists(MODELS_DIR):
        print(f"ERRO: O diretório '{MODELS_DIR}' não foi encontrado.")
        exit()
    if not os.path.exists(LOGS_DIR):
        print(f"ERRO: O diretório '{LOGS_DIR}' não foi encontrado.")
        exit()

    analyze_grid_results(MODELS_DIR, LOGS_DIR)