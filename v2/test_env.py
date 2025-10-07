# test_env.py
import sys
print("Versão do Python:", sys.version)
print("-" * 20)

try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    print("Arquivo 'classroom_ac_env' importado com sucesso.")

    print("Tentando criar a configuração...")
    config = ClassroomConfig()
    print("Configuração criada.")

    print("Tentando criar o ambiente...")
    env = ClassroomACEnvironment(config) # A linha que dá o erro
    print("✅ SUCESSO! Ambiente criado sem erros.")

except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()