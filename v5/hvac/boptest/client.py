# -*- coding: utf-8 -*-
"""
Cliente da API REST do BOPTEST.

POR QUE UM CLIENTE PRÓPRIO, E NÃO O `boptest-gym`
-------------------------------------------------
O `boptest-gym` oficial já embala o BOPTEST como ambiente Gymnasium, mas embala
junto as *decisões* que este trabalho precisa controlar: quais canais entram na
observação, em que ordem, e como são normalizados. Nossa observação é declarada
uma única vez em `hvac.features`, e o schema resultante é gravado no metadado do
modelo e conferido no carregamento — é essa verificação que permite avaliar um
agente treinado no ambiente local dentro do BOPTEST sem risco de ler os canais
com o significado errado.

Adotar o wrapper oficial significaria manter duas declarações de observação em
paralelo, exatamente o defeito que `hvac.features` existe para eliminar. Um
cliente REST fino tem menos código do que a camada de compatibilidade que a
alternativa exigiria, e mantém o contrato sob nosso controle.

CICLO DE VIDA DE UM TESTE
-------------------------
    selecionar(caso)  ->  testid
    cenario(...)      ->  período e preço de energia
    inicializar(...)  ->  estado inicial, com warmup
    passo(segundos)   ->  intervalo de controle
    avancar({...})    ->  aplica ação, devolve medições
    kpis()            ->  KPIs padronizados do framework
    resultados([...]) ->  séries temporais para métricas próprias
    encerrar()        ->  libera o worker

A API é versionada; `verificar_versao` registra a versão do servidor no
resultado, porque um KPI comparado entre versões diferentes do framework não é
comparável e essa é justamente a classe de erro que este artigo audita.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import requests


class BoptestError(RuntimeError):
    """Falha de comunicação ou de contrato com o serviço BOPTEST."""


# O serviço responde em 8000 na composição padrão do repositório oficial.
URL_PADRAO = os.environ.get("BOPTEST_URL", "http://127.0.0.1:8000")

# Credenciais de teste da composição local (`.env` do project1-boptest). Em uma
# implantação real vêm do painel; aqui são fixas e públicas de propósito.
CHAVE_PADRAO = os.environ.get("BOPTEST_API_KEY", "")


@dataclass
class BoptestClient:
    """
    Cliente síncrono, um teste por instância.

    `timeout` é generoso porque `advance` executa um passo de simulação de um
    modelo Modelica compilado: no `bestest_air` um passo de uma hora leva
    tipicamente algumas centenas de milissegundos, mas a primeira chamada após o
    `initialize` inclui o warmup e pode levar muito mais.
    """

    url: str = URL_PADRAO
    api_key: str = CHAVE_PADRAO
    timeout: float = 300.0
    tentativas: int = 3
    testid: Optional[str] = None
    versao: Optional[str] = None
    caso: Optional[str] = None
    _sessao: requests.Session = field(default_factory=requests.Session, repr=False)

    # ------------------------------------------------------------ transporte

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key} if self.api_key else {}

    def _pedido(self, metodo: str, rota: str, *, json=None, exigir_json=True):
        """
        Uma chamada, com repetição apenas em falha de transporte.

        Erro HTTP com corpo (4xx/5xx) não é repetido: significa contrato
        violado, e repetir esconderia o defeito atrás de um timeout.
        """
        alvo = f"{self.url.rstrip('/')}/{rota.lstrip('/')}"
        ultimo: Optional[Exception] = None
        for tentativa in range(self.tentativas):
            try:
                r = self._sessao.request(metodo, alvo, json=json,
                                         headers=self._headers(),
                                         timeout=self.timeout)
            except requests.RequestException as e:      # rede, não contrato
                ultimo = e
                time.sleep(2.0 * (tentativa + 1))
                continue
            if r.status_code >= 400:
                raise BoptestError(f"{metodo} {rota} -> {r.status_code}: "
                                   f"{r.text[:400]}")
            if not exigir_json:
                return r.text
            try:
                corpo = r.json()
            except ValueError as e:
                raise BoptestError(f"{metodo} {rota}: resposta não é JSON "
                                   f"({r.text[:200]})") from e
            # A API encapsula o resultado em {"status":..., "payload":...} nas
            # versões de serviço; versões antigas devolvem o payload cru.
            if isinstance(corpo, dict) and "payload" in corpo:
                return corpo["payload"]
            return corpo
        raise BoptestError(f"{metodo} {rota}: falha de transporte após "
                           f"{self.tentativas} tentativas ({ultimo})")

    # -------------------------------------------------------------- catálogo

    def verificar_versao(self) -> str:
        p = self._pedido("GET", "version")
        self.versao = p.get("version") if isinstance(p, dict) else str(p)
        return self.versao

    def casos(self) -> List[str]:
        p = self._pedido("GET", "testcases")
        if isinstance(p, list):
            return [c.get("testcaseid", c) if isinstance(c, dict) else c
                    for c in p]
        return list(p)

    # -------------------------------------------------------- ciclo de teste

    def selecionar(self, caso: str) -> str:
        p = self._pedido("POST", f"testcases/{caso}/select")
        tid = p.get("testid") if isinstance(p, dict) else p
        if not tid:
            raise BoptestError(f"seleção de '{caso}' não devolveu testid: {p}")
        self.testid, self.caso = str(tid), caso
        return self.testid

    def _tid(self) -> str:
        if not self.testid:
            raise BoptestError("nenhum teste selecionado: chame `selecionar`")
        return self.testid

    def cenario(self, *, periodo: Optional[str] = None,
                preco: Optional[str] = None) -> Dict:
        """
        Define período e estrutura tarifária.

        `periodo` é um dos rótulos do caso (`peak_cool_day`, `typical_cool_day`,
        ...). Ele já inclui o warmup definido pelos autores do caso, o que é
        preferível a escolher um instante arbitrário: o estado inicial passa a
        ser o mesmo que qualquer outro trabalho que use o mesmo rótulo, e a
        comparação entre publicações fica possível.
        """
        corpo: Dict[str, str] = {}
        if periodo:
            corpo["time_period"] = periodo
        if preco:
            corpo["electricity_price"] = preco
        return self._pedido("PUT", f"scenario/{self._tid()}", json=corpo)

    def inicializar(self, *, inicio_s: float, warmup_s: float = 0.0) -> Dict:
        return self._pedido("PUT", f"initialize/{self._tid()}",
                            json={"start_time": float(inicio_s),
                                  "warmup_period": float(warmup_s)})

    def passo(self, segundos: float) -> Dict:
        return self._pedido("PUT", f"step/{self._tid()}",
                            json={"step": float(segundos)})

    def avancar(self, acao: Optional[Dict[str, float]] = None) -> Dict:
        """Aplica os sobrescritos e avança um intervalo de controle."""
        return self._pedido("POST", f"advance/{self._tid()}", json=acao or {})

    # ----------------------------------------------------------- introspecção

    def entradas(self) -> Dict:
        return self._pedido("GET", f"inputs/{self._tid()}")

    def medicoes(self) -> Dict:
        return self._pedido("GET", f"measurements/{self._tid()}")

    def kpis(self) -> Dict:
        return self._pedido("GET", f"kpi/{self._tid()}")

    def resultados(self, pontos: Sequence[str], inicio_s: float,
                   fim_s: float) -> Dict[str, List[float]]:
        return self._pedido("PUT", f"results/{self._tid()}",
                            json={"point_names": list(pontos),
                                  "start_time": float(inicio_s),
                                  "final_time": float(fim_s)})

    def pontos_de_previsao(self) -> Dict:
        return self._pedido("GET", f"forecast_points/{self._tid()}")

    def previsao(self, pontos: Sequence[str], horizonte_s: float,
                 intervalo_s: float) -> Dict[str, List[float]]:
        return self._pedido("PUT", f"forecast/{self._tid()}",
                            json={"point_names": list(pontos),
                                  "horizon": float(horizonte_s),
                                  "interval": float(intervalo_s)})

    # ---------------------------------------------------------------- término

    def encerrar(self) -> None:
        """
        Libera o worker.

        Silencia falhas de propósito: encerrar é limpeza, e um teste que já
        expirou por inatividade devolve 404 — perder o resultado do experimento
        por causa disso seria absurdo.
        """
        if not self.testid:
            return
        for rota in (f"stop/{self.testid}", f"tests/{self.testid}"):
            try:
                self._pedido("PUT" if rota.startswith("stop") else "DELETE",
                             rota, exigir_json=False)
                break
            except BoptestError:
                continue
        self.testid = None

    def __enter__(self) -> "BoptestClient":
        return self

    def __exit__(self, *exc) -> None:
        self.encerrar()
