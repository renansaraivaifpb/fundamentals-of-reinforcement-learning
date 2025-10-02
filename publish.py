# git_helper.py
# -*- coding: utf-8 -*-
"""
Script auxiliar para automatizar o fluxo de commit e push do Git.

Este script executa as seguintes ações:
1. Adiciona todos os arquivos modificados ao stage.
2. Pede ao usuário uma mensagem de commit descritiva.
3. Realiza o commit.
4. Detecta a branch atual e envia as alterações para o repositório remoto (origin).
"""

import subprocess
import os
import sys

# --- CONFIGURAÇÃO ---
# ALTERAÇÃO 1: O caminho do repositório é detectado automaticamente
# O script assume que ele está na pasta raiz do seu projeto Git.
REPO_PATH = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd: str, cwd: str) -> bool:
    """Executa um comando no terminal e imprime o output ou o erro."""
    print(f"⚡ Executando: {cmd}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, shell=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"❌ Erro:\n{result.stderr}")
        return False
    else:
        # Imprime a saída apenas se houver algo para mostrar
        if result.stdout:
            print(result.stdout)
        return True

def git_sync():
    """Orquestra o processo de add, commit e push."""
    print("="*50)
    print("🔄 Iniciando sincronização com o repositório Git...")
    print("="*50)

    # 1. Detectar a branch atual
    proc = subprocess.run("git rev-parse --abbrev-ref HEAD", cwd=REPO_PATH, text=True, capture_output=True, shell=True, encoding='utf-8')
    if proc.returncode != 0:
        print("❌ Erro: Não foi possível detectar a branch atual. Certifique-se de que você está em um repositório Git.")
        return
    current_branch = proc.stdout.strip()
    print(f"🎯 Branch atual detectada: {current_branch}")

    # 2. Adicionar todos os arquivos
    if not run_cmd("git add .", cwd=REPO_PATH):
        return

    # 3. ALTERAÇÃO 2: Pedir uma mensagem de commit interativa
    try:
        commit_message = input("\n💬 Digite a mensagem do commit (ex: feat: Adiciona análise de cenários)\n> ")
        if not commit_message:
            print("❌ Mensagem de commit vazia. Abortando.")
            return
    except KeyboardInterrupt:
        print("\n Abortado pelo usuário.")
        return

    # 4. Realizar o commit
    # Usamos aspas triplas para lidar com mensagens que contêm aspas simples ou duplas
    commit_cmd = f'''git commit -m "{commit_message}"'''
    if not run_cmd(commit_cmd, cwd=REPO_PATH):
        # Se o commit falhar (ex: nada para commitar), o Git já informa o usuário.
        print("ℹ️  Nenhuma alteração para commitar ou erro no commit.")
        return

    # 5. ALTERAÇÃO 3: Fazer o push para a branch detectada automaticamente
    push_cmd = f"git push origin {current_branch}"
    if not run_cmd(push_cmd, cwd=REPO_PATH):
        return
        
    print("\n🎉 Processo concluído com sucesso!")
    print("="*50)

if __name__ == "__main__":
    # Verifica se o caminho do repositório é um diretório Git válido
    if not os.path.isdir(os.path.join(REPO_PATH, '.git')):
        print(f"ERRO: O diretório '{REPO_PATH}' não parece ser um repositório Git.")
        sys.exit(1)
    
    git_sync()