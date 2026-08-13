# -*- coding: utf-8 -*-
"""
Ponte para o BOPTEST (IBPSA Project 1).

Responde à limitação declarada como a mais séria do artigo: todos os resultados
vinham de um simulador de autoria própria, que por construção não exibe
descasamento de modelo. Aqui os mesmos controladores enfrentam um emulador de
terceiros, mantido pelo IBPSA e usado como referência na literatura de building
performance simulation.

Serviço local:

    cd project1-boptest && docker compose up -d web worker provision
    export BOPTEST_URL=http://127.0.0.1:8000
"""
from .avaliacao import avaliar, rodar_episodio
from .calibracao import Autoridade, medir_autoridade
from .client import BoptestClient, BoptestError
from .env import BoptestClassroomEnv, ConfigBoptest

__all__ = ["BoptestClient", "BoptestError", "BoptestClassroomEnv",
           "ConfigBoptest", "avaliar", "rodar_episodio",
           "Autoridade", "medir_autoridade"]
