# -*- coding: utf-8 -*-
"""
Testes da camada inglês usada pelo manuscrito da Energy & Buildings.

O QUE ESTES TESTES PROTEGEM. A tradução das figuras é feita depois do desenho,
substituindo texto por texto. Esse desenho só é seguro porque `t()` FALHA quando
não conhece uma string — se alguém trocar isso por um `dict.get(s, s)` para
"deixar de dar erro", o manuscrito volta a poder sair com rótulo em português e
ninguém percebe. O primeiro teste é o que impede essa troca.

Não se testa aqui a geração das figuras: ela exige carregar os modelos
treinados, e um teste de segundos não deve depender de arquivos de centenas de
megabytes. A cobertura equivalente é o próprio `gerar_paper_eb.py`, que percorre
todas as figuras e interrompe na primeira ausência.
"""
from __future__ import annotations

import pytest

from hvac import i18n_en as I


def test_texto_desconhecido_levanta():
    """A falha é explícita, e não silenciosa — ver a nota do módulo."""
    with pytest.raises(I.TraducaoAusente):
        I.t("uma frase que não está no glossário")


def test_glossario_tem_lacunas_pareadas():
    """Chave e modelo com contagem de '#' diferente truncariam o rótulo."""
    for chave, modelo in I.GLOSSARIO.items():
        assert chave.count("#") == modelo.count("#"), chave


def test_numeros_sao_reinjetados_com_ponto_decimal():
    assert I.t("contrato derivado = 1,32 kW") == "derived contract = 1.32 kW"
    assert I.t("±0,5 °C") == "±0.5 °C"
    assert I.t("7,55 kWh/dia") == "7.55 kWh/day"


def test_chave_ignora_o_valor_do_numero():
    """
    O mesmo rótulo com outro número precisa traduzir sem nova entrada: os
    valores mudam a cada reexecução, o glossário não pode mudar com eles.
    """
    assert I.t("derived contract = 9,99 kW".replace(
        "derived contract", "contrato derivado")) == "derived contract = 9.99 kW"


def test_rotulos_de_dados_e_de_figura_usam_o_mesmo_glossario():
    """
    Tabela e figura leem o mesmo rótulo do mesmo DataFrame; traduzi-los por
    caminhos distintos permitiria que discordassem entre si.
    """
    assert I.t("PI sintonizado") == "Tuned PI"
    assert I.t("DQN Equilibrado (550k)") == "DQN Balanced (550k)"
    assert I.t("sem gradiente interno") == "no inner gradient"


def test_texto_vazio_atravessa_intacto():
    """Eixos sem rótulo são comuns; não devem virar erro."""
    assert I.t("") == ""
    assert I.t("   ") == "   "
