import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Modelos de Fatores Externos (Inputs Dinâmicos) ---

def get_external_temperature(t_segundos):
    """
    Simula a temperatura externa ao longo do dia (ciclo de 24h).
    Usa uma onda senoidal para simular o dia/noite.
    """
    SEGUNDOS_POR_DIA = 24 * 3600
    # Média de 25°C, Amplitude de 5°C (varia de 20°C a 30°C)
    # Usamos -cos para que o mínimo seja de madrugada (perto de t=0)
    # e o máximo à tarde.
    t_dia = t_segundos % SEGUNDOS_POR_DIA
    temp_media = 25.0
    amplitude = 5.0
    # Shift de fase para o pico ser às 14h (14 * 3600s)
    fase_shift = 14 * 3600 
    
    T_ext = temp_media - amplitude * np.cos(2 * np.pi * (t_dia - fase_shift) / SEGUNDOS_POR_DIA)
    return T_ext

def get_internal_gain(t_segundos):
    """
    Simula a geração de calor interno (pessoas, computadores).
    Simula um "horário comercial" (8h às 18h).
    """
    hora_do_dia = (t_segundos / 3600) % 24
    
    if 8.0 <= hora_do_dia < 18.0:
        # Ex: 5 pessoas (5 * 100W) + 5 computadores (5 * 100W) = 1000W
        return 1000.0  # Watts
    else:
        # Fora do horário, apenas uma carga base (ex: standby)
        return 50.0  # Watts

# --- 2. O Simulador Principal ---

def simulate_room_physics(
    t_final_seg, 
    dt_seg, 
    T_inicial, 
    K_env, 
    K_int
):
    """
    Simula a temperatura da sala resolvendo a ODE com o Método de Euler.
    
    Args:
        t_final_seg (int): Tempo total da simulação em segundos.
        dt_seg (int): Passo de tempo da simulação (delta t) em segundos.
        T_inicial (float): Temperatura inicial da sala.
        K_env (float): Constante de acoplamento com o ambiente (1/s).
        K_int (float): Constante de ganho interno (K / W*s).
        
    Returns:
        tuple: (tempo, T_sala, T_externa, Q_interna)
    """
    print(f"Iniciando simulação de {t_final_seg / 3600:.1f} horas...")
    
    # Cria os vetores de tempo e resultados
    n_passos = int(t_final_seg / dt_seg)
    tempo = np.linspace(0, t_final_seg, n_passos + 1)
    
    # Inicializa os vetores de estado
    T_sala = np.zeros(n_passos + 1)
    T_externa = np.zeros(n_passos + 1)
    Q_interna = np.zeros(n_passos + 1)
    
    T_sala[0] = T_inicial
    
    # Loop da simulação
    for i in range(n_passos):
        t_atual = tempo[i]
        T_sala_atual = T_sala[i]
        
        # 1. Obter inputs do "mundo real"
        T_ext = get_external_temperature(t_atual)
        Q_int = get_internal_gain(t_atual)
        
        # 2. Calcular a derivada (nosso modelo físico)
        dTdt = K_env * (T_ext - T_sala_atual) + K_int * Q_int
        
        # 3. Integrar usando Euler
        T_sala_proximo = T_sala_atual + dTdt * dt_seg
        
        # 4. Salvar resultados
        T_sala[i+1] = T_sala_proximo
        T_externa[i] = T_ext  # Salva valores para plotagem
        Q_interna[i] = Q_int
        
    # Preenche o último valor dos inputs para o gráfico
    T_externa[-1] = get_external_temperature(tempo[-1])
    Q_interna[-1] = get_internal_gain(tempo[-1])
    
    print("Simulação concluída.")
    return tempo, T_sala, T_externa, Q_interna

# --- 3. Função de Visualização ---

def plot_simulation_results(tempo, T_sala, T_externa, Q_interna):
    """
    Cria um dashboard com os resultados da simulação.
    """
    print("Gerando visualização...")
    
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        sharex=True,  # Compartilha o eixo X
        figsize=(15, 10),
        gridspec_kw={'height_ratios': [3, 1]} # Dá mais espaço ao gráfico de T
    )
    
    fig.suptitle("Simulação de Comportamento Térmico da Sala (Sem AC)", fontsize=16)
    
    # --- Gráfico 1: Temperaturas ---
    ax1.plot(tempo / 3600, T_sala, label="T. Sala (Simulada)", color="red", linewidth=2)
    ax1.plot(tempo / 3600, T_externa, label="T. Externa", color="blue", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Temperatura (°C)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Adiciona linhas de "conforto" (objetivo futuro)
    ax1.axhline(22, color='green', linestyle=':', label='Conforto Min (22°C)', alpha=0.5)
    ax1.axhline(24, color='green', linestyle=':', label='Conforto Max (24°C)', alpha=0.5)
    ax1.legend()

    # --- Gráfico 2: Geração Interna ---
    ax2.plot(tempo / 3600, Q_interna, label="Geração Interna", color="orange", fillstyle="full")
    ax2.fill_between(tempo / 3600, Q_interna, color="orange", alpha=0.3)
    ax2.set_xlabel("Tempo (horas)")
    ax2.set_ylabel("Geração Interna (W)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajusta para o supertítulo
    plt.show()

# --- 4. Execução da Simulação ---

if __name__ == "__main__":
    
    # --- Parâmetros Físicos da Sala (Estimados) ---
    # Estes valores definem a "personalidade" da sala.
    # Altere-os para ver diferentes comportamentos.
    
    # Sala com isolamento "médio" e ~75m³ de ar
    # (Ref: C_th ≈ 90,000 J/K; R_th ≈ 0.05 K/W)
    
    # K_env = 1 / (R_th * C_th) ≈ 1 / (0.05 * 90000) ≈ 0.00022
    K_AMBIENTE = 0.00022  # Quão rápido a sala segue o exterior.
                         # (Valor alto = isolamento ruim, sala "vaza" calor)
    
    # K_int = 1 / C_th ≈ 1 / 90000 ≈ 0.000011
    K_INTERNO = 0.000011   # Quão rápido o calor interno (W) esquenta a sala.
                         # (Valor alto = sala pequena, esquenta rápido)

    # --- Parâmetros da Simulação ---
    TEMPO_SIMULACAO = 2 * 24 * 3600  # 48 horas (em segundos)
    PASSO_TEMPO = 60                 # Simular a cada 60 segundos (1 minuto)
    TEMP_INICIAL_SALA = 22.0         # Sala começa a 22°C
    
    # Executa a simulação
    tempo, T_sala, T_ext, Q_int = simulate_room_physics(
        t_final_seg=TEMPO_SIMULACAO,
        dt_seg=PASSO_TEMPO,
        T_inicial=TEMP_INICIAL_SALA,
        K_env=K_AMBIENTE,
        K_int=K_INTERNO
    )
    
    # Plota os resultados
    plot_simulation_results(tempo, T_sala, T_ext, Q_int)