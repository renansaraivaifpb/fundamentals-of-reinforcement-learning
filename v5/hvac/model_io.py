# -*- coding: utf-8 -*-
"""
Carregamento de modelos com reconstrução fiel da configuração de treino.

Motivação: um `.zip` do SB3 é irreprodutível sozinho. A avaliação precisa
reconstruir o MESMO ambiente — mesma física, mesmos pesos de recompensa, mesmo
espaço de observação e mesmo horizonte de decisão. Usar o default atual do módulo
`config` funciona até alguém mudar o default, e então quebra em silêncio ou com
erro obscuro.

Este projeto já sofreu essa falha duas vezes:

1. O `analyzer_v4.1.py` original avaliava sem o `ActionRepeatWrapper` usado no
   treino, medindo uma política com horizonte de decisão diferente. Erro
   silencioso — os números saíam, só estavam errados.
2. Ao adicionar erro escalado e integral à observação, os modelos de 7-D
   passaram a ser incarregáveis no ambiente de 9-D. Erro ruidoso, mas só
   descoberto por acaso.

A regra que este módulo impõe: **a configuração de treino é parte do contrato do
modelo**, e vem do metadado, nunca do default vigente.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields, replace
from typing import Dict, Optional, Tuple


def numpy_shim() -> None:
    """
    Permite ler modelos serializados com numpy 2.x sob numpy 1.x instalado.
    numpy 2.x renomeou `numpy.core` -> `numpy._core`; o pickle guarda o caminho.
    """
    import numpy.core  # noqa: F401
    for sub in ("", ".numeric", ".multiarray", ".umath", "._multiarray_umath"):
        try:
            __import__("numpy.core" + sub)
            sys.modules["numpy._core" + sub] = sys.modules["numpy.core" + sub]
        except Exception:
            pass


def read_metadata(model_path: str) -> Dict:
    """Metadado gravado ao lado do modelo, ou {} se ausente."""
    meta_path = model_path.replace(".zip", "_config.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def config_from_metadata(meta: Dict):
    """
    Reconstrói a `ClassroomConfig` exata do treino a partir do metadado.

    Filtra apenas campos que existem na dataclass atual: se o projeto ganhou
    campos novos desde o treino, eles assumem o default — o que é correto, porque
    o modelo antigo foi treinado sem eles. Campos removidos são ignorados em vez
    de estourar.
    """
    from .config import ClassroomConfig

    validos = {f.name for f in fields(ClassroomConfig)}
    brutos = dict(meta.get("env_params") or {})

    # RENOMEAÇÕES: campo que virou property tem de ser remapeado, não descartado.
    # `demand_contracted_kw` deixou de ser campo livre e passou a ser DERIVADO da
    # condição de projeto (demand_sizing.py). Sem este mapeamento, um modelo
    # treinado sob contrato de 0,70 kW seria reavaliado sob o contrato derivado
    # (~1,32 kW) — silenciosamente, e violando a garantia que este módulo existe
    # para dar: a config de avaliação é a DO TREINO, não o default vigente.
    if "demand_contracted_kw" in brutos:
        brutos.setdefault("demand_contracted_kw_manual",
                          brutos.pop("demand_contracted_kw"))

    params = {k: v for k, v in brutos.items() if k in validos}

    cfg = replace(ClassroomConfig(), **params)

    # A física do equipamento não é serializada como dataclass; reconstrói do
    # bloco `ac_physics` quando presente.
    fis = meta.get("ac_physics") or {}
    if fis:
        from .physics import ACPhysicsModel, ACState

        campos = {}
        if "capacity_btu_per_hour" in fis:
            campos["capacity_btu_per_hour"] = fis["capacity_btu_per_hour"]
        if fis.get("load_fraction"):
            campos["load_fraction"] = {
                ACState[k]: v for k, v in fis["load_fraction"].items()
            }
        if fis.get("cop"):
            campos["cop"] = {ACState[k]: v for k, v in fis["cop"].items()}
        campos["heating_enabled"] = bool(cfg.heating_enabled)
        cfg.physics = replace(ACPhysicsModel(), **campos)

    return cfg


def load_agent(model_path: str, device: str = "cpu") -> Tuple[object, object, Dict]:
    """
    Carrega (modelo, config_de_treino, metadado).

    Levanta erro claro se o metadado estiver ausente — avaliar sem saber a
    configuração de treino é pior que falhar, porque produz números plausíveis e
    errados.
    """
    numpy_shim()
    meta = read_metadata(model_path)
    if not meta:
        raise FileNotFoundError(
            f"metadado ausente para {model_path}. Sem ele não é possível "
            "reconstruir o ambiente de treino, e avaliar no ambiente errado "
            "produz números plausíveis mas inválidos."
        )

    algo = meta.get("algo", "DQN")
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

    classes = {"DQN": DQN, "PPO": PPO, "A2C": A2C,
               "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    try:
        from sb3_contrib import TQC
        classes["TQC"] = TQC
    except ImportError:
        pass

    if algo not in classes:
        raise KeyError(f"algoritmo '{algo}' desconhecido em {model_path}")

    model = classes[algo].load(model_path, device=device)
    cfg = config_from_metadata(meta)

    from .env import ClassroomACEnv

    env = ClassroomACEnv(config=cfg)
    assert_schema_compatible(model, env, meta, os.path.basename(model_path))
    return model, cfg, meta


def assert_schema_compatible(model, env, meta: Dict, rotulo: str = "") -> None:
    """
    Verifica o contrato da observação: forma E semântica.

    POR QUE A FORMA NÃO BASTA. A v4 comparava apenas `observation_space.shape`.
    Isso pega o caso grosseiro (9 canais contra 10), mas é cego para o perigoso:
    duas configurações distintas podem produzir a MESMA dimensão com canais
    diferentes ou na ordem trocada. O modelo carrega, avalia, produz números
    plausíveis — e a política está lendo integral onde deveria ler derivada.

    Um caso concreto desta base: os modelos do laboratório foram gravados como
    `..._lab2_obs9.zip`. Ao habilitar a restrição de demanda, a observação passou
    a ter 10 canais e o sufixo do arquivo virou mentira, silenciosamente. A
    dimensão morava no NOME DO ARQUIVO por falta de lugar melhor.

    Aqui o schema — nomes ordenados dos canais — é gravado no metadado e
    conferido na carga. Modelos anteriores ao schema (metadado sem a chave) só
    podem ser verificados pela forma; o aviso torna isso explícito em vez de
    presumir compatibilidade.
    """
    esperado, obtido = model.observation_space.shape, env.observation_space.shape
    if esperado != obtido:
        raise ValueError(
            f"contrato violado em {rotulo}: modelo espera obs {esperado}, "
            f"ambiente reconstruído dá {obtido}. O metadado não descreve a "
            "configuração de treino por completo."
        )

    gravado = (meta.get("obs_schema") or {}).get("nomes")
    if not gravado:
        import warnings
        warnings.warn(
            f"{rotulo}: metadado sem `obs_schema` (modelo anterior à v5). "
            "Só foi possível conferir a FORMA da observação, não a ordem dos "
            "canais — uma troca de ordem passaria despercebida.",
            RuntimeWarning, stacklevel=2)
        return

    atual = env.obs_schema["nomes"]
    if list(gravado) != list(atual):
        raise ValueError(
            f"contrato violado em {rotulo}: a ordem/identidade dos canais mudou.\n"
            f"  treino:    {list(gravado)}\n"
            f"  ambiente:  {list(atual)}\n"
            "Avaliar assim faria a política ler cada canal com o significado "
            "errado, produzindo métricas plausíveis e inválidas."
        )


def make_env_factory(cfg, meta: Dict, extra_wrappers=()):
    """
    Fábrica de ambientes que respeita o contrato do modelo, incluindo o
    `action_repeat` — cuja omissão foi o bug original do `analyzer_v4.1.py`.
    """
    from .env import ClassroomACEnv
    from .wrappers import ActionRepeatWrapper

    repeat = int(meta.get("action_repeat", 1))

    def factory():
        env = ClassroomACEnv(config=cfg)
        for w in extra_wrappers:
            env = w(env)
        return ActionRepeatWrapper(env, repeat=repeat)

    return factory
