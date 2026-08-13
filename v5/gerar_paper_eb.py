# -*- coding: utf-8 -*-
"""
Gera o manuscrito em inglês no formato de submissão da Energy & Buildings
(Elsevier).

MESMA FONTE DE DADOS QUE `gerar_paper.py` E QUE OS CADERNOS. Números vêm de
`hvac.results`, gráficos de `hvac.figures`. Nenhum valor é digitado à mão, de
modo que a versão em inglês não pode divergir da versão em português nem do
código: as duas leem o mesmo objeto. O que este arquivo acrescenta sobre o
gerador em português é (a) o texto em inglês, (b) o layout exigido pela
Elsevier, (c) a camada `hvac.i18n_en`, que traduz os rótulos das figuras e os
rótulos vindos de DataFrame.

TRÊS DIFERENÇAS DE FORMATO EM RELAÇÃO AO GERADOR DE CONGRESSO, todas exigidas
pelo fluxo de submissão da Elsevier e nenhuma cosmética:

  1. Coluna única, espaçamento duplo, numeração de linhas contínua e numeração
     de páginas, é o que o parecerista usa para referenciar trechos.
  2. Referências numeradas por ordem de aparição, no estilo numerado da
     Elsevier, e não autor-data. O número é DERIVADO do registro de citação,
     nunca escrito: ver `cite`. Isso torna impossível o defeito presente na
     versão de congresso, em que Watkins e Dayan (1992) é citado no texto sem
     constar da lista, e a ASHRAE consta da lista sem ser citada.
  3. Abstract dentro do limite editorial, Highlights e Nomenclature, que a
     Energy & Buildings exige e um artigo de congresso não tem.

    python gerar_paper_eb.py
"""
import os
import sys
import re
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # antes de hvac.figures: script headless

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

warnings.filterwarnings("ignore")

from hvac import figures as F
from hvac import i18n_en as I
from hvac import tex_backend as TB
from hvac import results as R

# Duas saídas, um único conteúdo. O corpo deste arquivo — as ~1800 linhas de
# chamadas a p(), rico(), tabela(), figura() — é escrito uma vez; os helpers
# despacham para o backend .docx (padrão) ou para o backend LaTeX da classe
# `cas-sc` da Elsevier (`--tex`). Duplicar o texto em dois geradores seria o
# mesmo defeito que este projeto evita entre tabela e figura.
TEX = "--tex" in sys.argv

RAIZ = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(RAIZ, "submissao_eb")
# O pacote LaTeX vive num diretório próprio porque a submissão da Elsevier em
# LaTeX é um conjunto: .tex, a classe, e as figuras em caminho relativo.
DESTINO_TEX = os.path.join(DESTINO, "cas")
FIGS = os.path.join(DESTINO_TEX if TEX else DESTINO, "figures")
os.makedirs(FIGS, exist_ok=True)
SAIDA = (os.path.join(DESTINO_TEX, "manuscript.tex") if TEX
         else os.path.join(DESTINO, "manuscript.docx"))

# Identificação do autor. Único bloco do manuscrito que não vem dos dados nem
# do código; fica isolado no topo para que alterá-lo não exija procurar.
AUTOR = "Renan Saraiva dos Santos"
AFILIACAO = "Instituto Federal da Paraíba, Campus Cajazeiras, Cajazeiras, Paraíba, Brazil"
EMAIL = "dossaraiva@gmail.com"
REPO = "https://github.com/renansaraivaifpb/fundamentals-of-reinforcement-learning"
ORCID = "0009-0006-7307-8361"

# --------------------------------------------- numeração derivada da composição
#
# Mesma disciplina do gerador em português: a ordem é declarada aqui, e o número
# é derivado dela. Cortar uma seção não quebra referência cruzada nenhuma.
_TABS = ["relacionados", "fisica", "correspondencia", "hiperparametros",
         "generalizacao", "status", "comparacao", "decomposicao",
         "ablacao", "permanencia", "hnp", "precisao", "faixa", "niveis",
         "qvalues", "aleatorios", "transitorio", "coerencia", "derivados",
         "combinacao", "orcamento", "boptest", "boptest_sintonia"]
_FIGS = ["ciclo", "decomposicao", "ablacao", "permanencia", "precisao",
         "demanda", "faixa", "curva", "niveis", "consumo", "divergencia",
         "pareto", "transitorio", "derivados", "orcamento",
         "boptest", "boptest_sintonia"]
_EQS = ["bellman", "fisica", "observacao", "recompensa",
        "conforto_dentro", "conforto_fora", "ciclo"]

_NT = {k: i + 1 for i, k in enumerate(_TABS)}
_NF = {k: i + 1 for i, k in enumerate(_FIGS)}
_NE = {k: i + 1 for i, k in enumerate(_EQS)}


def T(chave: str) -> int:
    return _NT[chave]


def Fg(chave: str) -> int:
    return _NF[chave]


def E(chave: str) -> int:
    return _NE[chave]


# ------------------------------------------------------- registro de citações
#
# `cite("mnih")` devolve "[n]", com n atribuído na primeira aparição. A lista de
# referências é emitida ao fim NA ORDEM DE APARIÇÃO, que é o que o estilo
# numerado da Elsevier exige. Consequência de projeto: uma referência não citada
# nunca entra na lista, e uma citação sem entrada na bibliografia interrompe a
# geração — os dois defeitos que a versão em português carrega.

_ORDEM_CITACAO = []

BIBLIO = {
 "alsayed": "K. Al Sayed, A. Boodi, R. Sadeghian Broujeny, K. Beddiar, "
            "Reinforcement learning for HVAC control in intelligent buildings: "
            "a technical and conceptual review, Journal of Building "
            "Engineering 95 (2024) 110085.",
 "blum": "D. Blum, J. Arroyo, S. Huang, J. Drgona, F. Jorissen, H.T. Walnum, "
         "Y. Chen, K. Benne, D. Vrabie, M. Wetter, L. Helsen, Building "
         "optimization testing framework (BOPTEST) for simulation-based "
         "benchmarking of control strategies in buildings, Journal of Building "
         "Performance Simulation 14 (2021) 586–610.",
 "judkoff": "R. Judkoff, J. Neymark, International Energy Agency Building "
            "Energy Simulation Test (BESTEST) and Diagnostic Method, Technical "
            "Report NREL/TP-472-6231, National Renewable Energy Laboratory, "
            "Golden, CO, 1995.",
 "xu": "S. Xu, Y. Fu, Y. Wang, Z. Yang, C. Huang, Z. O'Neill, Z. Wang, Q. Zhu, "
       "Efficient and assured reinforcement learning-based building HVAC "
       "control with heterogeneous expert-guided training, Scientific Reports "
       "15 (2025) 7414.",
 "sutton": "R.S. Sutton, A.G. Barto, Reinforcement Learning: An Introduction, "
           "second ed., MIT Press, Cambridge, MA, 2018.",
 "bellman": "R. Bellman, Dynamic Programming, Princeton University Press, "
            "Princeton, NJ, 1957.",
 "watkins": "C.J.C.H. Watkins, P. Dayan, Q-learning, Machine Learning 8 (1992) "
            "279–292.",
 "mnih": "V. Mnih, K. Kavukcuoglu, D. Silver, A.A. Rusu, J. Veness, "
         "M.G. Bellemare, et al., Human-level control through deep "
         "reinforcement learning, Nature 518 (2015) 529–533.",
 "haarnoja": "T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, Soft actor-critic: "
             "off-policy maximum entropy deep reinforcement learning with a "
             "stochastic actor, in: Proceedings of the 35th International "
             "Conference on Machine Learning (ICML), 2018, pp. 1861–1870.",
 "fujimoto": "S. Fujimoto, H. van Hoof, D. Meger, Addressing function "
             "approximation error in actor-critic methods, in: Proceedings of "
             "the 35th International Conference on Machine Learning (ICML), "
             "2018, pp. 1587–1596.",
 "wei": "T. Wei, Y. Wang, Q. Zhu, Deep reinforcement learning for building "
        "HVAC control, in: Proceedings of the 54th Annual Design Automation "
        "Conference (DAC), 2017, pp. 1–6.",
 "yuan": "X. Yuan, Y. Pan, J. Yang, W. Wang, Z. Huang, Study on the "
         "application of reinforcement learning in the operation optimization "
         "of HVAC system, Building Simulation 13 (2020) 1117–1132.",
 "boutahri": "Y. Boutahri, A. Tilioua, Reinforcement learning for HVAC control "
             "and energy efficiency in residential buildings with BOPTEST "
             "simulations and real-case validation, Discover Computing 28 "
             "(2025) 44.",
 "dai": "X. Dai, R. Chen, S. Guan, W.-T. Li, C. Yuen, BuildingGym: an "
        "open-source toolbox for AI-based building energy management using "
        "reinforcement learning, arXiv:2509.11922 (2025).",
 "zha": "V. Zha, I. Chiu, A. Guilbault, J. Tatis, Hyperspace neighbor "
        "penetration approach to dynamic programming for model-based "
        "reinforcement learning problems with slowly changing variables in a "
        "continuous state space, arXiv:2106.05497 (2021).",
 "henderson": "P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, "
              "D. Meger, Deep reinforcement learning that matters, in: "
              "Proceedings of the AAAI Conference on Artificial Intelligence, "
              "vol. 32, 2018, pp. 3207–3214.",
 "agarwal": "R. Agarwal, M. Schwarzer, P.S. Castro, A. Courville, "
            "M.G. Bellemare, Deep reinforcement learning at the edge of the "
            "statistical precipice, in: Advances in Neural Information "
            "Processing Systems (NeurIPS), 2021.",
 "towers": "M. Towers, A. Kwiatkowski, J. Terry, J.U. Balis, G. De Cola, "
           "T. Deleu, et al., Gymnasium: a standard interface for "
           "reinforcement learning environments, arXiv:2407.17032 (2024).",
 "raffin": "A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, "
           "N. Dormann, Stable-Baselines3: reliable reinforcement learning "
           "implementations, Journal of Machine Learning Research 22 (2021) "
           "1–8.",
 "astrom": "K.J. Åström, T. Hägglund, Advanced PID Control, ISA, The "
           "Instrumentation, Systems and Automation Society, Research Triangle "
           "Park, NC, 2006.",
 "cliff": "N. Cliff, Dominance statistics: ordinal analyses to answer ordinal "
          "questions, Psychological Bulletin 114 (1993) 494–509.",
 "ashrae": "ANSI/ASHRAE Standard 55-2020: Thermal Environmental Conditions for "
           "Human Occupancy, American Society of Heating, Refrigerating and "
           "Air-Conditioning Engineers, Atlanta, GA, 2020.",
}


def cite(*chaves) -> str:
    """Marcador de citação numerado, no estilo numerado da Elsevier."""
    numeros = []
    for chave in chaves:
        if chave not in BIBLIO:
            raise KeyError(f"citação '{chave}' sem entrada em BIBLIO")
        if chave not in _ORDEM_CITACAO:
            _ORDEM_CITACAO.append(chave)
        numeros.append(_ORDEM_CITACAO.index(chave) + 1)
    numeros.sort()
    # Sequências de três ou mais viram intervalo, convenção do estilo.
    partes, i = [], 0
    while i < len(numeros):
        j = i
        while j + 1 < len(numeros) and numeros[j + 1] == numeros[j] + 1:
            j += 1
        partes.append(str(numeros[i]) if j == i
                      else f"{numeros[i]},{numeros[j]}" if j == i + 1
                      else f"{numeros[i]}–{numeros[j]}")
        i = j + 1
    return "[" + ",".join(partes) + "]"


# ------------------------------------------------------------------- dados
print("collecting results (recomputing what is cheap, reading what is costly)...")
DEC   = R.decomposicao_da_vantagem()
COMP  = R.comparacao_controladores()
ABL   = R.ablacao()
ABLS  = R.ablacao_por_semente()
PERM  = R.permanencia()
PERMA = R.permanencia_por_agente()
LAB   = R.laboratorio_precisao()
TAB   = R.tabular()
DEM   = R.dimensionamento_demanda()
FAIXA = R.faixa_estreita()
CURVA = R.curva_aprendizado()
NIVEIS = R.uso_dos_niveis()
ALEAT = R.cenarios_aleatorios_cache()
QVAL  = R.analise_q_values()
TRANS = R.resposta_transitoria()
FIS   = R.correspondencia_fisica()
PARAM = R.analise_parametros_recompensa()
CALIB = R.calibracao_parametros()
COMB  = R.combinacao_derivados()
ORC   = R.orcamento_e_parametros()
HIPER = R.hiperparametros()
DECOMP = R.consumo_decomposto()
BOPT  = R.boptest_transferencia()
BOPTS = R.boptest_sintonia()
BRES  = R.boptest_resumo()
COP   = {k.name: val for k, val in
         __import__("hvac.config", fromlist=["x"]).config_for_profile(
             "Equilibrado").physics.cop.items()}


def v(x, casas=1) -> str:
    """Número no padrão decimal inglês."""
    return f"{x:.{casas}f}"


def _linha(df, coluna, valor):
    return df[df[coluna] == valor].iloc[0]


# Valores citados no abstract e no texto corrido, lidos dos mesmos objetos que
# alimentam as tabelas — escrevê-los à mão é como a v4 divergia de si mesma.
_PI = _linha(COMP, "controlador", "PI sintonizado")
_DQN = COMP[COMP["controlador"].str.startswith("DQN")]
DEV_PI = v(_PI["abs_dev_from_ideal"], 2)
DEV_DQN = f"{v(_DQN['abs_dev_from_ideal'].min(), 2)}–{v(_DQN['abs_dev_from_ideal'].max(), 2)}"
CUSTO_PI = v(_PI["cost_brl_day"], 2)
CUSTO_DQN = f"{v(_DQN['cost_brl_day'].min(), 2)}–{v(_DQN['cost_brl_day'].max(), 2)}"
ENER_PI = v(_PI["energy_kwh_day"], 2)
ENER_DQN = f"{v(_DQN['energy_kwh_day'].min(), 2)}–{v(_DQN['energy_kwh_day'].max(), 2)}"
CONF_PI = v(_PI["comfort_wide_pct"])
GANHO = {r["atribuivel_a"]: r["ganho_pp"] for _, r in DEC.iterrows()}

TEXDOC = TB.DocumentoCAS() if TEX else None
doc = Document()

# ------------------------------------------------------------------ formatação
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.line_spacing = 2.0    # espaçamento duplo exigido


def _numerar_linhas(secao, contar_de=1, distancia_twips=360) -> None:
    """
    Numeração de linhas contínua no corpo da página.

    Não é enfeite: o parecerista da Elsevier referencia o texto por número de
    linha, e um manuscrito sem isso costuma voltar da triagem técnica.
    """
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), str(contar_de))
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), str(distancia_twips))
    secao._sectPr.append(ln)


def _numerar_paginas(secao) -> None:
    """Campo PAGE centralizado no rodapé."""
    par = secao.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.line_spacing = 1.0
    for tipo, texto in (("begin", None), (None, "PAGE"), ("end", None)):
        run = par.add_run()
        if tipo is None:
            campo = OxmlElement("w:instrText")
            campo.set(qn("xml:space"), "preserve")
            campo.text = f" {texto} "
        else:
            campo = OxmlElement("w:fldChar")
            campo.set(qn("w:fldCharType"), tipo)
        run._r.append(campo)
        run.font.size = Pt(11)


_numerar_linhas(sec)
_numerar_paginas(sec)

# ------------------------------------------------------- tipografia do texto
# Marcação inline apenas: **negrito** e *itálico*. O gerador em português
# mantém um glossário de estrangeirismos para italizar; num texto em inglês
# esses termos são a própria língua, e a máquina inteira desaparece.
_RX_MARCA = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*", re.DOTALL)


def _segmentos(texto, *, bold=False, italic=False):
    saida, pos = [], 0
    for m in _RX_MARCA.finditer(texto):
        if m.start() > pos:
            saida.append((texto[pos:m.start()], bold, italic))
        if m.group(1) is not None:
            saida.append((m.group(1), True, italic))
        else:
            saida.append((m.group(2), bold, not italic))
        pos = m.end()
    if pos < len(texto):
        saida.append((texto[pos:], bold, italic))
    return saida



# Variáveis com índice ("K_p", "d_min", "C_th") escritas em ASCII no fonte. Sem
# tratamento elas saem com o sublinhado literal; aqui viram subscrito de
# verdade. A base é de UMA letra de propósito: identificadores de código como
# `tdis_tot` devem permanecer literais.
_RX_INDICE = re.compile(r"(?<![\w\\])([A-Za-z])_([A-Za-z]{1,6}|\d{1,2})\b")


def _runs_com_indice(par, texto, *, size, bold, italic):
    """Emite `texto` quebrando os índices em runs subscritos."""
    pos = 0
    for m in _RX_INDICE.finditer(texto):
        if m.start() > pos:
            r = par.add_run(texto[pos:m.start()])
            r.font.size = Pt(size); r.bold = bold; r.italic = italic
        base = par.add_run(m.group(1))
        base.font.size = Pt(size); base.bold = bold; base.italic = True
        ind = par.add_run(m.group(2))
        ind.font.size = Pt(size); ind.bold = bold
        ind.font.subscript = True
        pos = m.end()
    if pos < len(texto):
        r = par.add_run(texto[pos:])
        r.font.size = Pt(size); r.bold = bold; r.italic = italic


def _escreve(par, texto, *, size, bold=False, italic=False):
    for trecho, b, i in _segmentos(texto, bold=bold, italic=italic):
        _runs_com_indice(par, trecho, size=size, bold=b, italic=i)
    return par


def p(texto="", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, bold=False,
      italic=False, space_before=0, space_after=0, first=None, indent=None,
      duplo=True):
    if TEX:
        return TEXDOC.paragrafo(texto, bold=bold, italic=italic)
    par = doc.add_paragraph()
    par.alignment = align
    pf = par.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 2.0 if duplo else 1.15
    if first is not None:
        pf.first_line_indent = Cm(first)
    if indent is not None:
        pf.left_indent = Cm(indent)
    _escreve(par, texto, size=size, bold=bold, italic=italic)
    return par


def rico(partes, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, first=None,
         space_before=0, space_after=0, duplo=True):
    """Parágrafo com trechos em (texto, negrito, itálico)."""
    if TEX:
        return TEXDOC.paragrafo_rico(partes)
    par = doc.add_paragraph()
    par.alignment = align
    pf = par.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 2.0 if duplo else 1.15
    if first is not None:
        pf.first_line_indent = Cm(first)
    for texto, b, i in partes:
        _escreve(par, texto, size=size, bold=bool(b), italic=bool(i))
    return par


# Cabeçalhos que a classe CAS trata fora da numeração: uns porque vivem no
# preâmbulo (o backend os consome dos dados), outros porque são back matter.
_SEM_NUMERO = {"Highlights", "Abstract", "Nomenclature", "References",
               "CRediT authorship contribution statement",
               "Declaration of competing interest", "Data availability"}


def h(texto, nivel=1):
    if TEX:
        if texto in _SEM_NUMERO:
            # `Highlights`, `Abstract` e `References` são estruturais na classe
            # e não podem virar seção: a montagem final os emite no lugar certo.
            if texto in {"Highlights", "Abstract", "References"}:
                return None
            return TEXDOC.secao_sem_numero(texto)
        return TEXDOC.secao(texto, nivel)
    return p(texto, align=WD_ALIGN_PARAGRAPH.LEFT, size=12, bold=True,
             space_before=12 if nivel == 1 else 10, space_after=4)


def item(texto, *, marcador="• "):
    if TEX:
        return TEXDOC.item(texto)
    return p(marcador + texto, first=0.0, indent=0.6, space_after=0)


def legenda(texto, *, acima=False):
    """
    Legenda em espaçamento simples e corpo menor.

    Espaçamento duplo vale para o corpo do texto; aplicá-lo à legenda de uma
    tabela de sete colunas empurraria a tabela para a página seguinte sem
    ganho de legibilidade para o parecerista.
    """
    if TEX:
        return TEXDOC.legenda(texto, acima=acima)
    return p(texto, align=WD_ALIGN_PARAGRAPH.LEFT, size=10,
             space_before=10 if acima else 4,
             space_after=4 if acima else 10, duplo=False)


def quebra_pagina():
    # No LaTeX a paginação é da classe; forçar quebras aqui brigaria com ela.
    if TEX:
        return None
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def figura(chave, fig, largura_cm=14.5):
    """Traduz, grava em 300 dpi e insere centralizada."""
    caminho = os.path.join(FIGS, f"fig{Fg(chave):02d}_{chave}.png")
    I.traduzir_figura(fig).savefig(caminho, dpi=300, bbox_inches="tight")
    if TEX:
        return TEXDOC.figura(caminho)
    doc.add_picture(caminho, width=Cm(largura_cm))
    par = doc.paragraphs[-1]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(10)
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.line_spacing = 1.0


def _sombrear(celula, cor="D9D9D9"):
    tc = celula._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), cor)
    tc.append(shd)


def tabela(cabecalho, linhas, *, negrito_linhas=(), size=9.5, larguras=None):
    if TEX:
        return TEXDOC.tabela(cabecalho, linhas,
                             negrito_linhas=negrito_linhas,
                             larguras=larguras)
    t = doc.add_table(rows=1, cols=len(cabecalho))
    t.style = "Table Grid"
    for j, texto in enumerate(cabecalho):
        cel = t.rows[0].cells[j]
        cel.text = ""
        par = cel.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(2)
        par.paragraph_format.line_spacing = 1.0
        run = par.add_run(str(texto))
        run.bold = True
        run.font.size = Pt(size)
        _sombrear(cel)
    for i, linha in enumerate(linhas):
        cells = t.add_row().cells
        for j, val in enumerate(linha):
            cells[j].text = ""
            par = cells[j].paragraphs[0]
            par.alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                             else WD_ALIGN_PARAGRAPH.CENTER)
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.line_spacing = 1.0
            run = par.add_run(str(val))
            run.font.size = Pt(size)
            run.bold = i in negrito_linhas
    if larguras:
        for row in t.rows:
            for j, w in enumerate(larguras):
                row.cells[j].width = Cm(w)
    return t


def nota(texto):
    """Nota de rodapé de tabela."""
    if TEX:
        return TEXDOC.nota(texto)
    return p(texto, align=WD_ALIGN_PARAGRAPH.LEFT, size=9,
             space_before=2, space_after=8, duplo=False)


# --------------------------------------------------------------------- equações
#
# Os construtores devolvem uma ÁRVORE, e não XML: o backend .docx a converte em
# OMML e o backend LaTeX a converte em fórmula. Fosse cada um com sua notação, as
# sete equações do artigo precisariam ser escritas duas vezes, e a segunda cópia
# divergiria da primeira na revisão seguinte.
_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _op(texto):
    """Texto romano: operador, número ou pontuação."""
    return ("op", texto)


def _sub(base, indice):
    return ("sub", base, indice)


def _sup(base, expoente):
    return ("sup", base, expoente)


def _frac(num, den):
    return ("frac", num, den)


def _m(tag: str):
    return OxmlElement(f"m:{tag}")


def _run(texto: str, italico: bool = True):
    r = _m("r")
    if not italico:
        pr, sty = _m("rPr"), _m("sty")
        sty.set(qn("m:val"), "p")
        pr.append(sty)
        r.append(pr)
    t = _m("t")
    t.set(qn("xml:space"), "preserve")
    t.text = texto
    r.append(t)
    return r


def _omml(no):
    """Converte um nó da árvore em elemento OMML."""
    if isinstance(no, str):
        return _run(no)
    tipo = no[0] if isinstance(no, tuple) and no else None
    if tipo == "op":
        return _run(no[1], italico=False)
    if tipo in {"sub", "sup"}:
        n = _m("sSub" if tipo == "sub" else "sSup")
        e = _omml_grupo(no[1], _m("e"))
        idx = _omml_grupo(no[2], _m("sub" if tipo == "sub" else "sup"))
        n.append(e), n.append(idx)
        return n
    if tipo == "frac":
        n = _m("f")
        n.append(_omml_grupo(no[1], _m("num")))
        n.append(_omml_grupo(no[2], _m("den")))
        return n
    raise TypeError(f"nó de equação desconhecido: {no!r}")


def _omml_grupo(parte, alvo):
    """Anexa a `alvo` um nó ou uma sequência de nós."""
    for x in (parte if isinstance(parte, list) else [parte]):
        alvo.append(_omml(x))
    return alvo


_EQ_EMITIDAS = []


def equacao(partes, rotulo: Optional[str] = None):
    numero = None
    if rotulo is not None:
        numero = E(rotulo)
        _EQ_EMITIDAS.append(rotulo)
        esperado = _EQS[len(_EQ_EMITIDAS) - 1]
        if rotulo != esperado:
            raise AssertionError(
                f"equation '{rotulo}' emitted where '{esperado}' was declared: "
                "the order of `_EQS` does not match the order of the text")
    if TEX:
        return TEXDOC.equacao(partes, numero)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = par.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.0
    math = _m("oMath")
    for x in partes:
        math.append(_omml(x))
    par._p.append(math)
    if numero is not None:
        run = par.add_run(f"\t\t({numero})")
        run.font.size = Pt(12)
    return par


# ============================================================== PÁGINA DE TÍTULO
TITULO = ("When does deep reinforcement learning improve HVAC control? "
          "A reproducibility audit against competently tuned baselines")
TITULO_CURTO = "Auditing deep RL for HVAC control"
PALAVRAS_CHAVE = ["Reinforcement learning", "HVAC control", "Reproducibility",
                  "Baseline tuning", "Reward shaping",
                  "Building energy simulation"]
CREDITO = ("Conceptualization, Methodology, Software, Validation, Formal "
           "analysis, Investigation, Data curation, Writing -- original draft, "
           "Writing -- review & editing, Visualization")

# No LaTeX o front matter vive no preâmbulo, montado ao final por `TB.montar`;
# a página de título do .docx não tem equivalente e é simplesmente omitida.
if not TEX:
    p(TITULO, align=WD_ALIGN_PARAGRAPH.LEFT, size=16, bold=True,
      space_after=18, duplo=False)

    p(AUTOR, align=WD_ALIGN_PARAGRAPH.LEFT, size=12, space_after=4,
      duplo=False)
    p(AFILIACAO, align=WD_ALIGN_PARAGRAPH.LEFT, size=11, italic=True,
      space_after=2, duplo=False)
    p(f"ORCID: https://orcid.org/{ORCID}", align=WD_ALIGN_PARAGRAPH.LEFT,
      size=11, space_after=14, duplo=False)

    p("Corresponding author.", align=WD_ALIGN_PARAGRAPH.LEFT, size=11,
      bold=True, space_after=2, duplo=False)
    p(f"{AUTOR}, {AFILIACAO}", align=WD_ALIGN_PARAGRAPH.LEFT, size=11,
      space_after=2, duplo=False)
    p(f"E-mail address: {EMAIL}", align=WD_ALIGN_PARAGRAPH.LEFT, size=11,
      space_after=2, duplo=False)

    quebra_pagina()

# =================================================================== HIGHLIGHTS
h("Highlights")

_DESTAQUES = [
    "A tuned PI matches a DQN on comfort at lower deviation and lower cost.",
    "Decomposing a claimed +32 pp advantage attributes +0.0 pp to learning.",
    "A deterministic argmax policy discards the highest-COP power level.",
    "The anti-short-cycling reward term fails; a hard dwell constraint works.",
    "In BOPTEST, agents lead a PI with frozen gains and trail it once retuned.",
]

# Regras editoriais da Elsevier: de 3 a 5 destaques, cada um com no máximo 85
# caracteres. O excesso é devolvido na triagem, antes da revisão por pares, de
# modo que conferir aqui é mais barato que descobrir depois. A contagem entra na
# guarda junto do comprimento porque acrescentar um destaque sem remover outro é
# o erro fácil de cometer ao incorporar um resultado novo.
if not 3 <= len(_DESTAQUES) <= 5:
    raise AssertionError(f"{len(_DESTAQUES)} destaques (a revista aceita 3 a 5)")
for destaque in _DESTAQUES:
    if len(destaque) > 85:
        raise AssertionError(
            f"highlight com {len(destaque)} caracteres (limite 85): {destaque}")
    if not TEX:
        item(destaque)

quebra_pagina()

# ====================================================================== ABSTRACT
h("Abstract")

_ABSTRACT = (
  "Deep reinforcement learning (DRL) is widely reported to outperform "
  "conventional HVAC controllers, yet the reference controllers it beats are "
  "rarely configured with the care devoted to the agent: of six works surveyed "
  "here, only two include a tuned classical controller, and there the margin "
  "shrinks. How much of the claimed advantage survives a competently "
  "configured baseline? A DQN classroom HVAC controller is implemented from "
  "its specification and audited against a thermostat with hysteresis and a "
  "grid-tuned "
  "proportional-integral (PI) controller given identical actuation authority "
  "and decision horizon. The reproduction is faithful in comfort but not in "
  "energy or switching, bounding each claim. The advantage does not survive: "
  f"the PI, with two constants, matches the DQN on binary comfort ({CONF_PI} % "
  f"both) at lower deviation ({DEV_PI} vs {DEV_DQN} °C) and lower energy use "
  f"({ENER_PI} vs {ENER_DQN} kWh/day); decomposing the claimed +32 pp "
  f"attributes +{v(GANHO['configuração do baseline'])} pp to adding hysteresis "
  f"to the baseline, +{v(GANHO['controle clássico'])} pp to classical control "
  f"and +{v(GANHO['aprendizado'])} pp to learning. The excess energy is "
  "explained: a deterministic argmax policy converts a ~1 % action-value "
  "margin into 0 % usage of the highest-COP level. Transferred zero-shot to a "
  "third-party BOPTEST emulator, the agents overtake a PI carrying gains "
  "fitted to the original plant and lose that lead once the PI is retuned, "
  "the same defect, committed against our own baseline. What generalizes is "
  "that claims of learned advantage require baselines tuned and reported with "
  "the agent's own rigour.")

# 250 palavras é o teto editorial usual da Elsevier; ultrapassá-lo é motivo
# frequente de devolução antes da revisão por pares.
_N_PALAVRAS = len(_ABSTRACT.split())
if _N_PALAVRAS > 250:
    raise AssertionError(f"abstract com {_N_PALAVRAS} palavras (limite 250)")

if not TEX:
    p(_ABSTRACT, space_after=10)

if not TEX:
    rico([("Keywords: ", 1, 0), ("; ".join(PALAVRAS_CHAVE), 0, 0)],
         space_after=6)

quebra_pagina()

# =================================================================== NOMENCLATURE
h("Nomenclature")

for simbolo, sentido in [
    ("B", "base comfort bonus (reward units)"),
    ("B_c", "inner comfort gradient (reward units)"),
    ("C_th", "lumped thermal capacitance (simulation units, u·h/°C)"),
    ("COP", "coefficient of performance (–)"),
    ("d, d_min", "dwell time and minimum dwell requirement (min)"),
    ("K", "envelope heat transfer coefficient (u/°C)"),
    ("k", "curvature of the out-of-band comfort penalty (–)"),
    ("K_p, K_i", "proportional and integral gains of the PI controller (–)"),
    ("N, N_max", "occupancy and design occupancy (persons)"),
    ("Q(s, a)", "action-value function"),
    ("Q_ac", "cooling power delivered by the equipment (u)"),
    ("q_p", "sensible heat gain per occupant (u)"),
    ("R", "reward; subscripts comf, energy, switch, cold, cycle denote its terms"),
    ("s, a, r", "state, action and reward"),
    ("T, T_ext", "indoor and outdoor air temperature (°C)"),
    ("γ", "discount factor (–)"),
    ("δ", "Cliff's delta, non-parametric effect size (–)"),
    ("Δt", "control time step (h)"),
    ("ε", "process noise; also the exploration rate of the ε-greedy policy"),
    ("ρ", "weight of the anti-short-cycling penalty (reward units)"),
    ("σ", "standard deviation of indoor temperature (°C)"),
    ("τ", "plant time constant, C_th/K (h)"),
]:
    if TEX:
        # Tabulação não existe no LaTeX; a lista de descrição é a estrutura
        # correta e alinha os símbolos sem depender de espaçamento manual.
        TEXDOC.descricao(simbolo, sentido)
    else:
        rico([(simbolo, 1, 0), ("\t", 0, 0), (sentido, 0, 0)],
             size=11, space_after=0, duplo=False)

p("", space_after=6, duplo=False)
p("Abbreviations: BRL, Brazilian real; DQN, deep Q-network; DRL, deep "
  "reinforcement learning; HVAC, heating, ventilation and air conditioning; "
  "MDP, Markov decision process; MPC, model predictive control; pp, percentage "
  "points; SAC, soft actor-critic; TD3, twin delayed deep deterministic policy "
  "gradient.", size=11, duplo=False)

quebra_pagina()

# =============================================================== 1. INTRODUCTION
h("1. Introduction")

p("Heating, ventilation and air conditioning (HVAC) systems account for a "
  "dominant share of building energy use, and their operation directly "
  f"determines occupant comfort and productivity {cite('xu')}. The recent "
  "literature is abundant in proposals for deep reinforcement learning (DRL) "
  "control in this domain, almost always reporting substantial gains over "
  f"conventional strategies {cite('alsayed')}.")

p("One recurring methodological pattern in that literature deserves scrutiny: "
  "the magnitude of the reported gain depends critically on how the reference "
  "controller was configured. An on-off thermostat without hysteresis, with a "
  "zero deadband, is an artificially incapable opponent, and comparisons "
  "against it produce advantages that do not survive a competently tuned "
  "classical controller.", first=0.5)

rico([("This work carries out a ", 0, 0),
      ("reproducibility audit", 0, 0),
      (" of a DRL controller for classroom air conditioning, taken as a case "
       "study of that pattern.", 0, 0)], first=0.5)

p("The general objective is to analyse to what extent the performance reported "
  "for reinforcement learning HVAC controllers follows from learning rather "
  "than from the configuration of the reference controller adopted for "
  "comparison. The specific objectives are:", first=0.5)

for objetivo in [
    "(i) to assess to what extent the claimed advantage over the reference "
    "controller is attributable to learning or to the configuration of the "
    "opponent;",
    "(ii) to quantify, by controlled ablation over eight variants and three "
    "seeds, the contribution of each term of the proposed reward function, "
    "comparing it with conventional formulations from the literature;",
    "(iii) to verify the effectiveness of the anti-short-cycling penalty "
    "declared as a contribution, contrasting it with an equivalent structural "
    "constraint;",
    "(iv) to establish, theoretically and empirically, the adequacy of tabular "
    "methods to this domain, through the fraction of intra-tile transitions;",
    "(v) to examine whether extensions of the problem, precision tracking, "
    "tariff anticipation and contracted demand, constitute regimes favourable "
    "to learning;",
    "(vi) to determine the effect of narrowing the target band on relative "
    "performance, with both controllers re-prepared for each specification, "
    "and to verify the influence of the training budget on that effect;",
    "(vii) to identify the mechanisms that explain the differences in energy "
    "use observed between policies.",
]:
    item(objetivo, marcador="")

p("The aggregate result is negative as to the necessity of RL in this problem "
  "formulation, and positive as to the rigour required to assert it. "
  "Well-controlled negative results are scarce in this literature and, it is "
  "argued here, necessary to calibrate it.", first=0.5, space_before=6)

# ==================================================== 2. BACKGROUND / RELATED WORK
h("2. Background and related work")

h("2.1. Reinforcement learning", 2)

p("Reinforcement learning addresses the problem of deciding sequentially under "
  "uncertainty. An agent observes the state of the environment, chooses an "
  "action, receives a numerical reward and observes the new state; by "
  "repeating this cycle it adjusts its behaviour so as to maximise the reward "
  f"accumulated over time {cite('sutton')}. Fig. {Fg('ciclo')} illustrates the "
  "interaction.", first=0.5)

figura("ciclo", F.fig_ciclo_rl(), largura_cm=12.6)
legenda(f"Fig. {Fg('ciclo')}. Interaction cycle between agent and environment. "
        "At each decision interval the agent commands a power level and "
        "observes the resulting temperature and the corresponding reward.")

p("Formally the problem is modelled as a Markov decision process, defined by "
  "the quadruple (S, A, P, R): the state set S, the action set A, the "
  "transition function P, which describes the probability of reaching each "
  f"successor state, and the reward function R {cite('bellman')}. The Markov "
  "property requires the observed state to contain all information relevant to "
  "the decision, so that the past may be discarded. The agent's behaviour is "
  "described by the policy π(a | s), which assigns a choice of action to each "
  "state.", first=0.5)

p("The objective is to maximise the return, that is, the sum of future rewards "
  "discounted by a factor γ ∈ [0, 1) expressing how much the agent values "
  "distant rewards relative to immediate ones. The action-value function "
  "Q(s, a) is the expected return from taking action a in state s and "
  "following the policy thereafter, and satisfies the Bellman equation:",
  first=0.5)

equacao([_op("Q"), _op("(s, a) = 𝔼["), "r", _op(" + "), "γ",
         _op(" max"), _sub(_op(""), "a′"), _op(" Q(s′, a′)]")], rotulo="bellman")

p("The Q-learning algorithm approximates this solution by iterating over "
  "interaction samples, without requiring prior knowledge of the transition "
  f"function {cite('watkins')}. Its tabular formulation stores one value per "
  "state-action pair, which restricts it to small, discrete spaces. For "
  "continuous states the alternative is to approximate Q with a neural "
  f"network, the approach that characterises the deep Q-network {cite('mnih')}; "
  "stable training depends on two mechanisms, a replay buffer of past "
  "transitions, which decorrelates the samples, and a delayed target network, "
  "which stabilises the regression target.", first=0.5)

p("Methods of this family operate on discrete action spaces, since they "
  "require maximising Q over all actions at every step. For continuous actions "
  "one uses actor-critic architectures, in which one network proposes the "
  "action and another estimates its value. Two variants are used in this work: "
  "soft actor-critic (SAC), which jointly optimises return and policy entropy, "
  f"favouring exploration {cite('haarnoja')}, and twin delayed deep "
  "deterministic policy gradient (TD3), which mitigates value overestimation "
  f"through two critics and delayed actor updates {cite('fujimoto')}.",
  first=0.5)

rico([("One distinction matters for interpreting the results of this work: the "
       "policy learned by DQN is ", 0, 0),
      ("deterministic and obtained by maximisation", 0, 0),
      (", that is, in each state the action of highest estimated value is "
       "chosen. An action whose value is only slightly below the maximum, even "
       "if nearly equivalent, is never executed. Section 4.8 shows that this "
       "property has a direct consequence for energy use.", 0, 0)], first=0.5)

h("2.2. Related work", 2)

p("The literature on reinforcement learning applied to HVAC is abundant in "
  "reports of gains, but heterogeneous as to the opponent against which the "
  "gain is measured. Since the reported magnitude depends critically on that "
  "choice, the review below is organised by the quality of the reference "
  "controller, and not by algorithm or application.", first=0.5)

legenda(f"Table {T('relacionados')}. Reinforcement learning works for HVAC, "
        "classified by the reference controller adopted.", acima=True)
tabela(["Work", "Environment", "Reference for comparison", "Reported gain"],
       [[f"Wei et al. {cite('wei')}", "EnergyPlus", "rule-based",
         "20–70 % of cost"],
        [f"Yuan et al. {cite('yuan')}", "TRNSYS", "rule-based and PID",
         "7.7 % / 4.7 % of energy"],
        [f"Boutahri and Tilioua {cite('boutahri')}",
         "BOPTEST + real building", "PI and rule-based",
         "26.3 % / 8.8 % of energy"],
        [f"Dai et al. {cite('dai')}", "EnergyPlus",
         "setpoint schedule (RBC)", "framework"],
        [f"Xu et al. {cite('xu')}", "in-house model",
         "DDQN (another RL agent)", "8.8× in speed"],
        [f"Zha et al. {cite('zha')}", "—", "methods paper", "—"]],
       larguras=[3.6, 3.4, 4.4, 3.6])

p("Two patterns emerge. First, most works compare against rule-based control "
  "or against another RL agent; only two include a tuned classical feedback "
  "controller. Second, where such a controller is present the margin shrinks: "
  "Yuan et al. report 7.7 % savings over the rule-based controller and only "
  "4.7 % over the PID, and observe that in a multi-zone configuration the "
  "agent surpasses the references only after two years of exploration plus two "
  "years of stabilisation, reaching best performance in the seventh year, a "
  "sample cost that is itself a decision factor.", first=0.5, space_before=6)

p(f"Al Sayed et al. {cite('alsayed')} review the field and catalogue gains of "
  "27–30 % over rule-based controllers, up to 39.6 % over standard controllers "
  "and 23 % peak reduction over manual systems, all therefore over opponents "
  "of the same family. As for predictive control, the review describes it as "
  "the reference of the field and records that the best RL case in the "
  "literature *matches* the performance of model predictive control (MPC) "
  "without surpassing it; surpassing MPC appears as a future objective, "
  "conditional on reducing computational cost.", first=0.5)

rico([("One conditional from the same review deserves emphasis, because it "
       "bounds the scope of validity of the whole literature: the advantage of "
       "RL over traditional methods and over MPC is asserted ", 0, 0),
      ("\"when the simulation training environment accurately replicates "
       "real-world scenarios\"", 0, 1),
      (". The advantage is therefore conditional on simulator fidelity, a "
       "point to which Section 5.1 returns.", 0, 0)], first=0.5)

p(f"Wei et al. {cite('wei')} are the first work to apply deep RL to this "
  "domain and formulate the central difficulty of the multi-zone case: with z "
  "zones and m actuation levels the action space has m^z elements, which "
  "degrades training. The authors propose a multi-level control heuristic to "
  f"circumvent it. Zha et al. {cite('zha')} identify why tabular methods fail "
  "in continuous spaces with slowly varying variables, the transition remains "
  "within the same hyper-tile and value never propagates, a diagnosis used in "
  f"Section 4.5. Xu et al. {cite('xu')} attack training cost through "
  "distillation of expert knowledge and introduce run-time shielding, an idea "
  "applicable to this work's finding on anti-short-cycling.", first=0.5)

p(f"Methodologically, Henderson et al. {cite('henderson')} and Agarwal et al. "
  f"{cite('agarwal')} document the fragility of comparisons in deep RL with "
  "few seeds, motivating the use of non-parametric effect sizes rather than "
  "isolated p-values, the practice adopted here.", first=0.5)

p("This work measures the step that the literature mostly omits: the "
  "performance of a competently tuned proportional-integral controller, "
  "situated between the rule-based controller (which RL beats) and MPC (which "
  "RL aspires to match).", first=0.5)

# ======================================================= 3. MATERIALS AND METHODS
h("3. Materials and methods")

h("3.1. Simulation environment", 2)

p("The environment models a classroom with a capacity of 45 occupants, "
  f"implemented with the Gymnasium interface {cite('towers')}. The thermal "
  "dynamics are lumped, treating the room as a single node:", first=0.5)

equacao([_sub("T", "t+1"), _op(" = "), _sub("T", "t"), _op(" + "),
         _frac([_op("Δ"), "t"], _sub("C", "th")),
         _op(" ["), "N", _op("·"), _sub("q", "p"), _op(" + "), "K",
         _op(" ("), _sub("T", "ext"), _op(" − "), _sub("T", "t"), _op(") − "),
         _sub("Q", "ac"), _op("(a)] + "), "ε"], rotulo="fisica")

p("where C_th is the thermal capacitance (15.0 in the training room), N the "
  "number of occupants, q_p = 0.3 u the heat gain per person, K = 0.5 the heat "
  "transfer coefficient to the exterior, Q_ac the cooling power and ε Gaussian "
  "process noise (σ = 0.01). The step is Δt = 0.1 h (6 min) and the episode "
  "spans 24 h (240 steps). Outdoor temperature follows a daily sinusoid "
  "peaking at 14:00, with a base of 28 °C and an amplitude of 8 °C.",
  first=0.5)

rico([(f"Table {T('fisica')} of the audited manuscript is ", 0, 0),
      ("derived", 0, 0),
      (", rather than tabulated, from two catalogue parameters: a capacity of "
       "30,000 BTU/h and a coefficient of performance (COP) per level. That "
       "choice makes the model auditable and allows the equipment to be "
       "replaced without rewriting the table. The derivation reproduces the "
       "published values to better than 0.5 %.", 0, 0)], first=0.5)

legenda(f"Table {T('fisica')}. Equipment model, derived from rated capacity "
        "and COP.", acima=True)
tabela(["Level", "Load fraction", "COP", "Cooling (u)",
        "Electrical (kW), derived", "Published"],
       [["OFF", "0.00", "—", "0", "0.000", "—"],
        ["LOW", "0.25", "3.45", "10", "0.637", "0.64"],
        ["MEDIUM", "0.55", "3.59", "22", "1.347", "1.35"],
        ["HIGH", "1.00", "3.00", "40", "2.931", "2.93"]],
       larguras=[2.0, 2.6, 1.6, 2.6, 3.6, 2.2])

p("The COP peak at part load (MEDIUM) reproduces the behaviour of inverter "
  "equipment and is what creates the relevant trade-off: operating at MEDIUM "
  "is energetically cheap, which favours the thermostat on cost even where it "
  "loses on comfort.", first=0.5, space_before=6)

p("The model operates in dimensionless units, which prevents judging what "
  "physical room the environment corresponds to, a limitation raised in peer "
  "review of the audited manuscript. Equipment capacity anchors the scale, "
  "since 40 units correspond to 8.79 kW thermal, and from that anchor the "
  "corresponding physical quantities are derived.", first=0.5)

legenda(f"Table {T('correspondencia')}. Correspondence between simulation "
        "units and physical quantities.", acima=True)
tabela(["Quantity", "Value in simulation", "Physical equivalent"],
       [["Unit of thermal power", "1 u", f"{v(FIS['kw_por_unidade'], 4)} kW"],
        ["Thermal capacitance C_th", "15 u·h/°C",
         f"{v(FIS['capacidade_kwh_por_c'], 2)} kWh/°C "
         f"({FIS['capacidade_kj_por_c']:,.0f} kJ/°C)"],
        ["Equivalent air volume", "—",
         f"{FIS['volume_ar_equivalente_m3']:,.0f} m³"],
        ["Conductance K", "0.5 u/°C",
         f"{v(FIS['condutancia_kw_por_c'], 3)} kW/°C"],
        ["Time constant τ = C/K", "30 u·h/u",
         f"{v(FIS['constante_tempo_h'], 1)} h"],
        ["Heat gain per occupant", "0.3 u",
         f"{FIS['ganho_por_pessoa_w']:.0f} W"],
        ["Load at design occupancy", "13.5 u",
         f"{v(FIS['carga_ocupacao_plena_kw'], 2)} kW"]],
       larguras=[5.4, 3.4, 5.2])

rico([("Two readings follow from the table. The first is favourable: the heat "
       "gain per occupant, 66 W sensible, lies in the usual range for "
       "sedentary activity, and equipment capacity exceeds the design "
       "occupancy load by almost threefold, which is plausible sizing. The "
       "second is unfavourable and consequential: ", 0, 0),
      ("the time constant of the environment is 30 h, longer than the 24 h "
       "duration of the episode itself", 0, 0),
      (". The thermal mass is equivalent to roughly 10,000 m³ of air, about "
       "seventy times the air volume of a conventional classroom, a value that "
       "would be justified only by very strong coupling to the building "
       "structure.", 0, 0)], first=0.5, space_before=6)

p("The consequence is structural and cuts across several results of this work. "
  "An environment whose time constant exceeds the episode horizon strongly "
  "damps disturbances, which makes the control problem easier than it would be "
  "in a real room and reduces the surface available for differentiating "
  "controllers. That same sluggishness is the cause of the diagnosis in "
  "Section 4.5: the typical per-step variation, 0.090 °C, is eleven times "
  "smaller than the discretisation interval, which is why transitions remain "
  "intra-tile. High inertia and the failure of the tabular method are not "
  "independent findings; they are the same property of the environment "
  "observed through two different instruments.", first=0.5)

h("3.2. Markov decision process formulation", 2)

p(f"The observed state is normalised as in Eq. {E('observacao')}, with four "
  "components: temperature, occupancy and a circular encoding of time of day.",
  first=0.5)

equacao(["s", _op(" = ["), _op("clip"), _op("("), _frac(["T", _op(" − 15")], "20"),
         _op(", 0, 1), "), _frac("N", _sub("N", "max")), _op(", sin"),
         _op("("), _frac([_op("2π"), "h"], "24"), _op("), cos"),
         _op("("), _frac([_op("2π"), "h"], "24"), _op(")]")], rotulo="observacao")

p("The action space is discrete with four levels (OFF, LOW, MEDIUM, HIGH) for "
  "the DQN agents, and continuous in [0, 1] for SAC. The reward aggregates "
  "five terms:", first=0.5)

equacao(["R", _op(" = "), _sub("R", "comf"), _op(" + "), _sub("R", "energy"),
         _op(" + "), _sub("R", "switch"), _op(" + "), _sub("R", "cold"),
         _op(" + "), _sub("R", "cycle")], rotulo="recompensa")

p("The comfort term adopts the topology called quadratic plateau with inner "
  f"gradient (Eq. {E('conforto_dentro')}), where B is the base bonus, B_c the "
  "inner gradient, k the curvature outside the band and [T_min, T_max] = "
  f"[22, 26] °C, a band consistent with the acceptability limits of "
  f"ANSI/ASHRAE Standard 55 {cite('ashrae')}:", first=0.5)

equacao([_sub("R", "comf"), _op(" = "), "B", _op(" + "), _sub("B", "c"),
         _op(" (1 − "), _frac([_op("|"), "T", _op(" − 24|")], "2"), _op("),"),
         _op("  if 22 ≤ "), "T", _op(" ≤ 26")], rotulo="conforto_dentro")
equacao([_sub("R", "comf"), _op(" = "), "B", _op(" − "), "k",
         _op(" "), _sup([_op("("), "T", _op(" − "), _sub("T", "lim"), _op(")")], "2"),
         _op(",  otherwise")], rotulo="conforto_fora")

p(f"The anti-short-cycling penalty (Eq. {E('ciclo')}) punishes switching before "
  "the minimum dwell time d_min = 36 min (6 steps), in proportion to how early "
  "the switch occurs:", first=0.5)

equacao([_sub("R", "cycle"), _op(" = "), "ρ", _op(" "),
         _frac([_sub("d", "min"), _op(" − "), "d"], _sub("d", "min")),
         _op(",  if a switch occurred and "), "d", _op(" < "),
         _sub("d", "min")], rotulo="ciclo")

rico([("Inferred parameters. ", 1, 0),
      ("The audited manuscript specifies B_c, the energy penalty and the "
       "switching penalty numerically, but does not publish B, k, ρ or the "
       "cold penalty. Those values were inferred from the manuscript's reward "
       "figure and are explicitly flagged in the code: B = 10.0 (plateau edges "
       "at ≈ +10); k = 0.6 (the curve reaches ≈ −11 at 32 °C); ρ = −5.0 (not "
       "observable, chosen on the order of magnitude of comfort); cold penalty "
       "= −2.0 (the manuscript's conclusion warns that a severe value induced "
       "avoidance behaviour). Numerical divergences should be attributed to "
       "these four parameters before the implementation is suspected.",
       0, 0)], first=0.5, space_before=4)

h("3.3. Reference controllers", 2)

p("Three reference controllers were implemented with the same actuation "
  "authority and the same decision horizon (action repeat 2) as the learned "
  "agents, so that the comparison isolates the policy and not the sampling "
  "interval:", first=0.5)

for controlador in [
    "Thermostat with zero deadband, the baseline of the audited manuscript. It "
    "switches to MEDIUM above 26 °C and to HIGH above 28 °C, turning off on "
    "return to the band.",
    "Thermostat with 1 °C hysteresis, the same logic with a deadband, "
    "isolating the effect of baseline configuration.",
    "PI controller with anti-windup, tuned by grid search over the scenario "
    "matrix (K_p = 1.3; K_i = 0.2), with its output discretised onto the same "
    f"action space as the agents {cite('astrom')}.",
]:
    item(controlador)

h("3.4. Experimental protocol and metrics", 2)

p("The evaluation uses a 3×3 factorial matrix crossing initial thermal "
  "condition (17 °C, 24 °C, 30 °C) with occupancy level (few, medium, many), "
  "for nine scenarios C1–C9. A controllability filter is applied: scenarios in "
  "which the trivial reference policy (always off) already satisfies the "
  "requirement are excluded, since they do not discriminate between "
  "controllers. Eight of the nine scenarios are controllable, C1 (cold + few "
  "people) being the excluded one, a result identical to that of the audited "
  "manuscript.", first=0.5)

p("Metrics are computed exclusively over the occupied window (07:00–22:00): "
  "percentage of time in the wide band [22, 26] °C and in the narrow band "
  "[23, 25] °C; mean absolute deviation from the ideal, |T − 24|; percentage "
  "of overheating; energy use (kWh/day); cost (BRL/day); and switches per "
  "hour. In addition, the distribution of dwell times between switches is "
  "reported, which is the correct metric for assessing protection against "
  "short cycling.", first=0.5)

p(f"The agents were trained for 550,000 steps with Stable-Baselines3 "
  f"{cite('raffin')}, learning rate 5·10⁻⁵, batch size 64 and action repeat 2. "
  f"The algorithms covered are DQN {cite('mnih')}, SAC {cite('haarnoja')} and "
  f"TD3 {cite('fujimoto')}, plus tabular Q-learning {cite('watkins')} for the "
  "diagnosis of Section 4.5.", first=0.5)

legenda(f"Table {T('hiperparametros')}. Effective hyperparameters of the DQN "
        "agent, extracted from the trained object.", acima=True)
tabela(["Parameter", "Value"],
       [[I.t(_r["parametro"]), str(_r["valor"])] for _, _r in HIPER.iterrows()],
       larguras=[8.4, 5.6])

p("The values above were extracted from the trained object, and not from what "
  "the code declares. Framework defaults that never appear explicitly (replay "
  "buffer capacity, discount factor, target network update interval, gradient "
  "clipping) are as necessary to reproduction as the hand-written parameters, "
  "and their omission was raised in peer review of the audited manuscript. The "
  "policy is a multilayer perceptron with two hidden layers of 64 units and "
  "9,480 trainable parameters; exploration is ε-greedy with linear decay from "
  "1.0 to 0.05 over the first 10 % of training.", first=0.5, space_before=6)

h("3.5. Statistical analysis", 2)

p("With three seeds per condition, the smallest attainable two-sided p-value "
  "in a permutation test is 2/C(6,3) = 0.10; therefore p ≈ 0.101 is the floor, "
  f"not a marginal result. Cliff's δ {cite('cliff')}, a non-parametric effect "
  "size, is therefore reported together with percentile bootstrap confidence "
  "intervals. A value δ = −1.00 indicates complete separation between groups, "
  "every seed of the ablated condition is worse than every seed of the "
  f"reference. This choice follows the recommendations of Henderson et al. "
  f"{cite('henderson')} and Agarwal et al. {cite('agarwal')}.", first=0.5)

h("3.6. Reproducibility safeguards", 2)

p("Three mechanisms were implemented so that the results reported here are "
  "verifiable and cannot diverge silently from the code.", first=0.5)

for salvaguarda in [
    "Verified observation contract. The observation space is declared once, and "
    "from it derive both the space bounds and the vector emitted at each step "
    "and the ordered list of channels. That list is persisted alongside the "
    "model and checked on loading: two configurations may produce the same "
    "dimensionality with different channels, or with channels in a different "
    "order, in which case the policy would read each channel with the wrong "
    "meaning and produce plausible, invalid metrics. The check turns that "
    "silent failure into a loading error.",
    "Multi-seed evaluation. All controllers are evaluated on the same seeds, "
    "disjoint from the training seeds, which makes the comparison paired: "
    "environment noise is common to the arms and cancels in the difference.",
    "Single-source tables and figures. Each result is produced by a single "
    "function, consumed both by the analysis notebooks and by the generator of "
    "this document. Results of low computational cost are recomputed at each "
    "generation; those requiring training are read from canonical files. "
    "Divergence between a table and the figure beside it is impossible by "
    "construction.",
]:
    item(salvaguarda)

# ==================================================================== 4. RESULTS
h("4. Results")

h("4.1. Fidelity of the reproduction", 2)

p("Before any comparison, it is necessary to establish in which dimensions the "
  "implementation reproduces the described behaviour, and in which it does "
  "not. The comfort metrics of the thermostatic baseline fall within about "
  "3 percentage points of the published values (54.0 % against 50.9 % in the "
  "wide band; 2.10 °C against 2.21 °C of absolute deviation), which supports "
  "the physics being correct. Energy and switching, however, fall appreciably "
  "below the published figures (−31 % and −61 %, respectively), because the "
  "manuscript does not publish the deadband thresholds; adjusting them to "
  "match the results would constitute circular reproduction and was not done.",
  first=0.5)

p(f"The generalisation test without retraining (Table {T('generalizacao')}) "
  "reproduced the qualitative signature of the manuscript: the advantage over "
  "the thermostat stays in the range +24.8 to +35.0 pp, against +21.7 to "
  "+38.5 pp published, with the same ordering, worst degradation in the "
  "high-inertia room and smallest advantage in the well-insulated room.",
  first=0.5)

legenda(f"Table {T('generalizacao')}. Generalisation without retraining: "
        "comfort in the wide band (%). Δ is the advantage of the DQN over the "
        "thermostat.", acima=True)
tabela(["Room variation", "DQN (repro.)", "DQN (publ.)", "Therm. (repro.)",
        "Therm. (publ.)", "Δ repro.", "Δ publ."],
       [["C_th = 10 (light)", "87.3", "88.6", "53.9", "51.3", "+33.3", "+37.3"],
        ["C_th = 15 (training)", "86.5", "83.2", "54.0", "51.0", "+32.5", "+32.2"],
        ["C_th = 20", "87.2", "77.6", "54.5", "49.4", "+32.8", "+28.2"],
        ["C_th = 30 (heavy)", "81.1", "67.5", "56.4", "45.8", "+24.8", "+21.7"],
        ["K = 0.3 (well insulated)", "85.5", "80.3", "57.0", "51.9", "+28.5", "+28.4"],
        ["K = 0.8 (poorly insulated)", "84.0", "85.3", "49.0", "46.7", "+35.0", "+38.5"]],
       larguras=[4.2, 2.1, 2.1, 2.2, 2.1, 1.9, 1.9])

rico([("This asymmetry must be taken seriously, and not recorded as a "
       "footnote caveat: ", 0, 0),
      ("fidelity is established in comfort, the metric of smallest divergence, "
       "and not in energy and switching, which are the two quantities on which "
       "part of the conclusions of this paper rest", 0, 0),
      (". The claim that the PI is cheaper than the DQN, and the analysis of "
       "dwell times in Section 4.4, both fall on quantities that diverge from "
       "the published ones. Two observations bound the damage. First, the "
       "comparison between controllers is ", 0, 0),
      ("internal", 0, 1),
      (": PI, thermostat and agents operate in the same environment and in the "
       "same run, so the divergence is common to the arms and does not explain "
       "the difference between them. Second, in the case of switching the "
       "divergence is conservative: this reproduction switches less than the "
       "manuscript, so the dwell violation documented in Section 4.4 would be "
       "worse, not better, in the audited setting.", 0, 0)], first=0.5)

p("Since the findings of this paper do not all depend on the same premises, "
  f"Table {T('status')} states, for each of them, what it depends on and what "
  "that authorises one to assert. This is the recommended reading for anyone "
  "wishing to cite an isolated result.", first=0.5)

legenda(f"Table {T('status')}. Epistemic status of the findings: what each one "
        "depends on and what it authorises one to assert.", acima=True)
tabela(["Finding", "Depends on", "Robustness"],
       [["79.8 % intra-tile transitions (Section 4.5)",
         "only on C_th, Δt and the gain per occupant, all published",
         "Maximal: recomputable from the manuscript, independent of this "
         "reproduction"],
        ["The anti-short-cycling penalty does not protect (Section 4.4)",
         "on the learned policy; the arithmetic argument uses only |ρ| and "
         "B + B_c, both known",
         "High and conservative: this reproduction switches less, so in the "
         "original the violation would be worse"],
        ["The highest-COP level is discarded (Section 4.8)",
         "on the learned policy and on the COP table, which is published",
         "High: the control is internal, with SAC under an identical reward as "
         "contrast"],
        ["The PI matches or beats the DQN (Section 4.2)",
         "on the fidelity of this environment to the audited one",
         "Inference, not direct measurement: internally valid, but its "
         "transfer to the audited setting is not demonstrated"]],
       larguras=[4.6, 5.0, 5.4])

p("With the reach of the fidelity established, we turn to the audit of the "
  "central claim.", first=0.5, space_before=6)

h("4.2. The claimed advantage does not survive a competent baseline", 2)

p(f"Table {T('comparacao')} compares the three reward profiles of the "
  "manuscript against the reference controllers, all evaluated under identical "
  "conditions.", first=0.5)

legenda(f"Table {T('comparacao')}. Comparison under identical conditions (3×3 "
        "matrix, occupied window). Best values in bold.", acima=True)
_ordem = ["PI sintonizado", "DQN Agressivo (550k)", "DQN Equilibrado (550k)",
          "DQN Passivo (550k)", "SAC Equilibrado (550k)",
          "Termostato zona morta = 1 °C", "Termostato zona morta = 0"]
_extra = {"PI sintonizado": " (Kp=1.3; Ki=0.2)",
          "SAC Equilibrado (550k)": ", continuous",
          "Termostato zona morta = 0": " (baseline)"}
_linhas = []
for _n in _ordem:
    _sel = COMP[COMP["controlador"] == _n]
    if _sel.empty:
        continue
    _r = _sel.iloc[0]
    _linhas.append([I.t(_n) + _extra.get(_n, ""),
                    v(_r["comfort_wide_pct"]), v(_r["comfort_narrow_pct"]),
                    v(_r["abs_dev_from_ideal"], 2), v(_r["overheat_pct"]),
                    v(_r["energy_kwh_day"], 2), v(_r["cost_brl_day"], 2),
                    v(_r["changes_per_hour"], 2)])
_linhas.append(["Tabular Q-learning (550k)", v(TAB["conf_larga_pct"].mean()),
                v(TAB["conf_estreita_pct"].mean()),
                v(TAB["desvio_ideal"].mean(), 2), "—",
                v(TAB["energia_kwh"].mean(), 2), "—", "—"])
tabela(["Controller", "Comfort [22,26] %", "Comfort [23,25] %", "|T−24| °C",
        "Overheat %", "Energy kWh/day", "Cost BRL/day", "Switches/h"],
       _linhas, negrito_linhas=(0,),
       larguras=[5.2, 2.0, 2.0, 1.7, 1.6, 1.9, 1.6, 1.4])
nota("Costs are expressed in Brazilian reais (BRL) under the Brazilian "
     "time-of-use residential tariff, whose peak-to-off-peak ratio is 2.21×. "
     "Energy in kWh/day is reported alongside so that the comparison remains "
     "currency-independent.")

p("The SAC agent, with continuous action, is included because it corresponds "
  "to the discrete-versus-continuous action space comparison of the audited "
  "manuscript. The reproduction confirms the ordering reported there, the DQN "
  "beats SAC on comfort, but with the PI controller present the reading "
  "changes: the difference between action spaces, between reward profiles and "
  "between learning algorithms all lies below what two classical constants "
  "deliver.", first=0.5, space_before=6)

rico([("The PI controller dominated the three DQN profiles on every quality "
       "and cost metric simultaneously: the same binary comfort, a lower "
       "absolute deviation from the ideal and lower energy use. ", 0, 0),
      ("Two tuned constants match or beat 550,000 training steps.", 0, 0)],
     first=0.5, space_before=6)

p(f"Table {T('decomposicao')} decomposes the claimed advantage, attributing "
  "each increment to its cause.", first=0.5)

legenda(f"Table {T('decomposicao')}. Decomposition of the claimed +32 pp "
        "advantage.", acima=True)
_l = []
for _i, _r in DEC.iterrows():
    _g = "—" if _i == 0 else f"+{v(_r['ganho_pp'])} pp"
    _l.append([I.t(_r["etapa"]), v(_r["conforto_larga_pct"]), _g,
               I.t(_r["atribuivel_a"])])
tabela(["Step", "Comfort, wide band (%)", "Gain", "Attributable to"], _l,
       negrito_linhas=(len(_l) - 1,), larguras=[6.4, 3.2, 2.2, 4.2])

figura("decomposicao", F.fig_decomposicao(DEC))
legenda(f"Fig. {Fg('decomposicao')}. Decomposition of the claimed advantage. "
        "Each column adds a single factor to the previous one. The last step, "
        "learning, is null.")

p("Approximately 100 % of the claimed advantage is attributable to inadequate "
  f"configuration of the opponent (Fig. {Fg('decomposicao')}). The claim of "
  "82.9 % against 50.9 % is technically true and substantively misleading: the "
  "comparison is against an artificially incapable controller.", first=0.5,
  space_before=6)

h("4.3. Ablation of the reward function", 2)

p("The thesis that reward shaping matters more than the algorithm requires a "
  "controlled ablation. Eight variants were trained, each disabling exactly "
  "one mechanism, over three seeds, 24 training runs of 550,000 steps. "
  f"Table {T('ablacao')} reports comfort in the narrow band, the metric that "
  "discriminates between regimes.", first=0.5)

legenda(f"Table {T('ablacao')}. Reward ablation: comfort in the narrow "
        "[23, 25] °C band.", acima=True)
_l = []
for _, _r in ABL.iterrows():
    _sem = ABLS.loc[ABLS["variante"] == _r["variante"], "conf_estreita_pct"]
    _d = "—" if _r["variante"] == "completa" else v(_r["cliffs_delta"], 2)
    _l.append([I.t(_r["rotulo"]), v(_r["media"]),
               f"[{v(_r['minimo'])}; {v(_r['maximo'])}]",
               " / ".join(v(x) for x in _sem), _d, I.t(_r["magnitude"])])
tabela(["Variant", "Mean", "Range", "Per seed", "Cliff's δ", "Magnitude"], _l,
       negrito_linhas=(0, len(_l) - 1),
       larguras=[4.6, 1.5, 2.5, 3.7, 1.9, 2.0])

figura("ablacao", F.fig_ablacao(ABL, ABLS))
legenda(f"Fig. {Fg('ablacao')}. Reward ablation. The three seeds of each "
        "variant are shown individually: with n = 3, the mean alone would hide "
        "that the legacy step reward has one seed at 44.0 and two above 81.")

p(f"Three conclusions emerge (Fig. {Fg('ablacao')}). First, the flat-plateau "
  "hypothesis is confirmed: removing the inner gradient degrades narrow-band "
  "comfort from 83.7 % to 47.7 % (−36 pp), raises the deviation from 0.85 to "
  "1.46 °C and overheating from 5.2 % to 13.8 %, with complete separation "
  "between seeds (δ = −1.00). A genuinely flat plateau does not reward "
  "entering the band, and the policy parks at its edge.", first=0.5,
  space_before=6)

rico([("Second, and contrary to the thesis of the manuscript: the proposed "
       "formulation does ", 0, 0), ("not beat", 0, 0),
      (" the conventional one. The variant using a pure quadratic, no inner "
       "gradient, no switching penalty, no anti-short-cycling and no cold "
       "penalty, reaches 83.1 % against 83.7 % for the proposal (δ = −0.11, "
       "negligible). The reason is conceptual: a pure quadratic is a parabola "
       "with its maximum at 24 °C and therefore already has a gradient "
       "everywhere by construction. It is the flat plateau that destroys it; "
       "the proposal merely restores it. The contribution, correctly framed, "
       "is the correction of a defect introduced by the authors themselves, a "
       "negative finding, not a positive one.", 0, 0)], first=0.5)

p("Third, three of the five terms are inert: anti-short-cycling has exactly "
  "null effect (δ = 0.00), the switching penalty is negligible (δ = −0.11) and "
  "the cold penalty has small magnitude. An additional indication, with "
  "overlapping intervals and therefore suggestive: removing anti-short-cycling "
  "reduces energy use by 5 % (10.53 against 11.09 kWh/day) at no comfort cost, "
  "which suggests the term is actively harmful.", first=0.5)

h("4.4. The anti-short-cycling penalty does not meet its objective", 2)

p("Frequent compressor switching shortens equipment life; the manuscript "
  f"addresses this through Eq. {E('ciclo')}, with d_min = 36 min. The "
  "aggregate switches-per-hour metric, however, cannot verify the protection. "
  f"The distribution of dwell times can (Table {T('permanencia')} and Fig. "
  f"{Fg('permanencia')}).", first=0.5)

legenda(f"Table {T('permanencia')}. Distribution of dwell times between "
        "switches (requirement: d_min = 36 min).", acima=True)
tabela(["Agent", "Median", "Mean", "Minimum", "Violations of d_min"],
       [[I.t(_r["agente"]), f"{_r['mediana_min']:.0f} min",
         f"{v(_r['media_min'])} min", f"{_r['min_min']:.0f} min",
         f"{v(_r['violacoes_pct'])} %"] for _, _r in PERMA.iterrows()],
       larguras=[4.4, 2.4, 2.4, 2.2, 3.4])

figura("permanencia", F.fig_permanencia(PERM))
legenda(f"Fig. {Fg('permanencia')}. Cumulative distribution of dwell times for "
        "the Balanced DQN. The highlighted point on the dashed line marks the "
        "fraction of switches below the requirement: 71.4 % under the reward "
        "penalty, 6.4 % under the hard constraint.")

p("The median of 12 minutes corresponds to a single decision. The mean of 36 "
  "to 60 minutes is inflated by long periods with the equipment off and masks "
  "the problem completely, a textbook case of an aggregate hiding the "
  "distribution.", first=0.5, space_before=6)

rico([(f"The cause is arithmetic: the penalty of Eq. {E('ciclo')} is worth at "
       "most |ρ| = 5, against comfort gains of up to B + B_c = 14. ", 0, 0),
      ("A reward term that can be outbid by another reward term is not a "
       "protection; it is a suggestion.", 0, 0)], first=0.5)

p("Applying d_min as a hard constraint, along the lines of the shielding of Xu "
  f"et al. {cite('xu')}, on the already trained agent and without retraining, "
  "violations fall from 71.4 % to 6.4 % and the median rises from 12 to 48 "
  "minutes, at a cost of +0.08 °C in absolute deviation and +3.8 % in energy, "
  "leaving binary comfort unchanged at 86.5 %. The lesson is transferable: "
  "safety and hardware-integrity properties should be imposed structurally, "
  "not negotiated through the reward function.", first=0.5)

h("4.5. If RL is used here, it has to be deep", 2)

p("The low dimensionality of the state space (four variables) suggests that a "
  f"tabular method would suffice. Zha et al. {cite('zha')} give the "
  "theoretical reason why it does not: in continuous spaces with slowly "
  "varying variables the transition is intra-tile and value never propagates "
  "between cells. That fraction was measured directly, without training any "
  f"agent (Table {T('hnp')}).", first=0.5)

legenda(f"Table {T('hnp')}. Discretisation diagnosis.", acima=True)
tabela(["Quantity", "Value"],
       [["Discretised space", "4,800 states (20 × 10 × 24)"],
        ["Temperature bin width", "1.00 °C"],
        ["Typical ΔT per step (full room)", "0.090 °C"],
        ["Steps to cross one bin", "11.1"],
        ["Intra-tile transitions (random policy)", "79.8 %"],
        ["Intra-tile transitions (after training)", "59.8 %"],
        ["State coverage (ceiling reached)", "76.7 %"]],
       larguras=[8.0, 7.0])

p("About 80 % of the backups update a state with its own value. Tabular "
  "Q-learning trained for 550,000 steps over three seeds reached 42.1 % "
  "comfort in the wide band (31.5 / 38.7 / 56.0 per seed, standard deviation "
  "12.6), worse than the thermostat with a deadband and highly unstable. The "
  "drop from 80 % to 60 % intra-tile transitions after training occurs because "
  "the policy learns to use HIGH, which changes temperature fast enough to "
  "cross bins; even so, six of every ten updates remain inert, and about 1,100 "
  "of the 4,800 states are never visited.", first=0.5, space_before=6)

p("The obstacle is therefore not dimensionality, but the ratio between the "
  "step of the dynamics (0.090 °C) and the resolution of the discretisation "
  "(1.00 °C). This is the only finding of this work that is entirely derivable "
  "from published parameters, and therefore the most robust.", first=0.5)

h("4.6. Three attempts to construct a regime favourable to RL", 2)

p("Having established that classical control dominates in the original "
  "formulation, we investigated whether any extension of the problem would "
  "reverse the result. Three regimes were constructed and evaluated.",
  first=0.5)

rico([("Precision tracking. ", 1, 0),
      ("A laboratory with a ±0.5 °C tolerance around the setpoint, with "
       "reversible equipment, continuous action and an observation enriched "
       "with scaled error, leaky integral and derivative. TD3 and SAC agents "
       "were trained under three cost profiles. The bidirectional PI remains "
       f"superior (Table {T('precisao')} and Fig. {Fg('precisao')}).", 0, 0)],
     first=0.5)

legenda(f"Table {T('precisao')}. Precision regime: percentage of time within "
        "the ±0.5 °C tolerance at steady state.", acima=True)
tabela(["Controller", "Within tolerance %", "σ (°C)", "|T−24| °C",
        "Cost BRL/day", "kWh/day", "Adjustments/h"],
       [[I.t(_r["controlador"]), v(_r["na_tol_%"]), v(_r["sigma"], 3),
         v(_r["|T-24|"], 3), v(_r["custo_dia"], 2), v(_r["kWh_dia"], 2),
         v(_r["ajustes_h"], 2)]
        for _, _r in LAB[LAB["controlador"] != "Antecipatório (manual)"].iterrows()],
       negrito_linhas=(0,), larguras=[4.6, 2.4, 1.7, 1.9, 2.2, 1.8, 1.9])

figura("precisao", F.fig_precisao(LAB))
legenda(f"Fig. {Fg('precisao')}. Precision regime: daily cost against "
        "temperature dispersion; marker area is proportional to time within "
        "tolerance. The PI occupies the optimal corner alone.")

rico([("Tariff anticipation. ", 1, 0),
      ("Under a real time-of-use tariff, with a peak period ratio of 2.21×, "
       "the opportunity for pre-cooling is limited by the tolerance itself: "
       "with ±0.5 °C, the available thermal storage covers only about 29 % of "
       "the duration of the peak period. The margin for anticipation is "
       "therefore structurally small.", 0, 0)], first=0.5, space_before=6)

rico([("Contracted demand. ", 1, 0),
      ("A constraint on the maximum of the 15-minute integrated average, and "
       "not on the mean, a quantity that purely reactive feedback does not "
       "represent, and meeting which would require anticipation. A preliminary "
       "version of this experiment fixed contracted demand at 0.70 kW, a value "
       "chosen by sweep as the point of maximum tension between comfort and "
       "constraint. Auditing that value under the same requirement applied to "
       "the manuscript, it was found to be ", 0, 0), ("infeasible", 0, 0),
      (": the worst-case steady state requires 1.204 kW, so 0.70 kW is 72 % "
       "below what is needed and supports only about 17 of the 45 occupants. "
       "No policy satisfies such a constraint, because anticipation does not "
       "create steady state.", 0, 0)], first=0.5)

p("The arbitrated value was replaced by sizing derived from the design "
  "condition, design occupancy, outdoor temperature at the daily peak, "
  "setpoint maintained, plus a 10 % contracting margin, giving 1.324 kW. "
  "Under that contract none of the nine scenarios exceeds the limit (maximum "
  "observed peak 1.213 kW), and the naive PI becomes indistinguishable from "
  "the constraint-aware PI.", first=0.5)

figura("demanda", F.fig_demanda(DEM))
legenda(f"Fig. {Fg('demanda')}. Origin of the demand pressure. Only scenarios "
        "starting away from the setpoint exceed the derived contract, and in "
        "those the violation occurs at the initial step, where anticipation is "
        "impossible by definition.")

p(f"The reason is a structural contradiction (Fig. {Fg('demanda')}): the only "
  "scenarios that exceed the derived contract are those starting away from the "
  "setpoint, in which the violation occurs at the initial step, the instant at "
  "which anticipation is impossible by definition. Building the scenario "
  "matrix so that it starts at the setpoint, in order to eliminate that "
  "impossible violation, simultaneously eliminates the only source of demand "
  "above steady state. The set on which the hypothesis can be tested is "
  "empty.", first=0.5, space_before=6)

h("4.7. Required precision: does a narrow band favour RL?", 2)

p("The hypothesis is natural: if the specification demands more precision, "
  "learning would have more to offer than a fixed law. To test it, both "
  "controllers are prepared for *each* band width, the DQN is retrained with "
  "the corresponding band and the PI is re-tuned by grid search. Freezing the "
  "PI gains and then demanding greater precision would reproduce, within this "
  "experiment, the very defect that Section 4.2 identifies in the audited "
  "manuscript.", first=0.5)

_l = []
for _t in sorted(FAIXA["tolerancia"].unique(), reverse=True):
    _s = FAIXA[FAIXA["tolerancia"] == _t]
    _pi = _s[_s["controlador"] == "PI sintonizado"]["na_tolerancia_pct"].iloc[0]
    _dq = _s[_s["controlador"] == "DQN"]["na_tolerancia_pct"]
    _te = _s[_s["controlador"] == "Termostato (zm=1 °C)"]["na_tolerancia_pct"]
    _l.append([f"±{v(_t)} °C", v(_pi), v(_dq.mean()),
               " / ".join(v(x) for x in _dq), v(_dq.std()),
               v(_te.iloc[0]) if len(_te) else "—", f"+{v(_pi - _dq.mean())}"])
legenda(f"Table {T('faixa')}. Performance by target band width, with the DQN "
        "retrained and the PI re-tuned for each.", acima=True)
tabela(["Target band", "PI", "DQN (mean)", "DQN per seed", "Std. dev.",
        "Thermostat", "PI advantage"], _l, negrito_linhas=(2,),
       larguras=[2.2, 1.6, 2.2, 3.4, 1.6, 2.2, 2.6])

figura("faixa", F.fig_faixa_estreita(FAIXA))
legenda(f"Fig. {Fg('faixa')}. Performance by band width, with both controllers "
        "prepared for each width. The DQN seeds are shown individually.")

p(f"The effect was the opposite of the prediction (Fig. {Fg('faixa')}): "
  "narrowing the band *widens* the advantage of classical control, "
  "monotonically in the mean, +0.0, +3.0 and +8.3 percentage points. A second "
  "effect accompanies the first: the standard deviation between DQN seeds grew "
  "from 0.00 to 5.20 and 4.04. When the specification tightens, training not "
  "only delivered less, it became unstable.", first=0.5, space_before=6)

# A leitura por semente é derivada dos dados: com n = 3, a diferença de médias
# pode ser menor que a dispersão, caso em que reportar só a média afirmaria
# mais do que os dados sustentam.
_por_faixa = {}
for _t in sorted(FAIXA["tolerancia"].unique(), reverse=True):
    _s = FAIXA[FAIXA["tolerancia"] == _t]
    _pi_v = _s[_s["controlador"] == "PI sintonizado"]["na_tolerancia_pct"].iloc[0]
    _dq_v = _s[_s["controlador"] == "DQN"]["na_tolerancia_pct"]
    _por_faixa[_t] = (int((_dq_v < _pi_v - 0.05).sum()), len(_dq_v),
                      _dq_v.std(), _pi_v - _dq_v.mean())

_EXT = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
_ab1, _n1, _sd1, _dif1 = _por_faixa[1.0]
_ab05, _n05 = _por_faixa[0.5][0], _por_faixa[0.5][1]

rico([("The monotonicity of the mean, however, should not be read as a "
       "systematic deficit growing uniformly, and the distinction matters "
       f"because the between-seed dispersion at ±1.0 °C ({v(_sd1, 2)}) ", 0, 0),
      ("exceeds the difference between the means itself", 0, 0),
      (f" ({v(_dif1)} points). At that width, {_EXT[_n1 - _ab1]} of the "
       f"{_EXT[_n1]} seeds tie with the PI and {_EXT[_ab1]} "
       f"{'collapses' if _ab1 == 1 else 'collapse'}, so what the mean records "
       "is training instability, not a uniform loss. At ±0.5 °C the picture is "
       f"different and stronger: {_EXT[_ab05]} of the {_EXT[_n05]} seeds fall "
       "below the PI, which characterises separation and supports the claim. "
       "The defensible reading is therefore that the advantage of classical "
       "control is null at wide band, indistinguishable from seed noise at "
       "intermediate band, and consistent at narrow band.", 0, 0)], first=0.5)

p(f"The obvious objection remains: had the agent trained enough? Fig. "
  f"{Fg('curva')} answers by evaluating the policy periodically during "
  "training, against the PI line, which is horizontal by construction since it "
  "does not learn.", first=0.5)

figura("curva", F.fig_curva_aprendizado(CURVA))
legenda(f"Fig. {Fg('curva')}. Performance against training budget, for two "
        "band widths. Faint lines are individual seeds.")

p("At ±2.0 °C the agent crossed quickly and saturated exactly on the PI line: "
  "the tie of Section 4.2 corresponds to the convergence plateau, and is not "
  "an artefact of the stopping point. At ±0.5 °C the curve rose from 4 % to "
  "about 63 % and remained 16.9 points below the PI at the end of the budget, "
  "with strong deceleration.", first=0.5, space_before=6)

rico([("Necessary caveat: ", 1, 0),
      ("the curve decelerates sharply but is not flat, the trend over the "
       "last 150,000 steps is still +2.9 points. Maintaining that rate, an "
       "optimistic assumption that the flattening itself contradicts, about "
       "870,000 additional steps would be needed to close the gap. What the "
       "data support, therefore, is a high sample cost that grows with the "
       "precision demanded, and not the impossibility of convergence. It is "
       f"the same order of difficulty that Yuan et al. {cite('yuan')} report "
       "when observing that the agent beats the PID only after years of "
       "operation.", 0, 0)], first=0.5)

h("4.8. Why the agent spends more: the most efficient level is discarded", 2)

p("The DQN agents deliver the same comfort as the PI while consuming more "
  "energy. Decomposing energy use by the commanded power level (Fig. "
  f"{Fg('consumo')}), along the lines of the per-item analysis of Yuan et al. "
  f"{cite('yuan')}, identifies the cause.", first=0.5)

legenda(f"Table {T('niveis')}. Fraction of time at each commanded power level, "
        "by controller.", acima=True)
tabela(["Controller", "OFF %", f"LOW % (COP {COP['LOW']:.2f})",
        f"MEDIUM % (COP {COP['MEDIUM']:.2f})", f"HIGH % (COP {COP['HIGH']:.2f})"],
       [[I.t(_r["controlador"]), v(_r["OFF"]), v(_r["LOW"]), v(_r["MEDIUM"]),
         v(_r["HIGH"])] for _, _r in NIVEIS.iterrows()],
       larguras=[4.6, 2.2, 2.9, 3.1, 2.6])

figura("niveis", F.fig_uso_dos_niveis(NIVEIS, cop=COP))
legenda(f"Fig. {Fg('niveis')}. Fraction of time at each power level. The "
        "highest-COP level is absent from all three DQN profiles.")

p("The three profiles of the manuscript commanded MEDIUM 0.0 % of the time "
  f"(Fig. {Fg('niveis')}), even though their reward weights vary by factors of "
  "three to four, the comfort gradient runs from 7.0 to 3.0, the energy "
  "penalty from 0.03 to 0.12. The Aggressive profile discards the LOW level as "
  "well, operating in a purely on-off regime. And MEDIUM is precisely the "
  "level of highest coefficient of performance of the equipment, since it "
  "reproduces the part-load efficiency peak characteristic of inverter "
  "machines.", first=0.5, space_before=6)

figura("consumo", F.fig_consumo_decomposto(DECOMP, cop=COP))
legenda(f"Fig. {Fg('consumo')}. Daily energy decomposed by commanded level. "
        "The learned agent's excess energy does not come from operating longer, "
        "but from operating at the less efficient levels.")

rico([("The cause is not the reward function, and there is a clean "
       "experimental control for that: the SAC agent, trained with the *same* "
       "reward as the Balanced profile, commands MEDIUM 13.3 % of the time, "
       "practically the same as the PI (12.0 %). Same reward, opposite "
       "behaviours.", 0, 0)], first=0.5, space_before=6)

p("Analysis of the action values clarifies the mechanism. MEDIUM ranked third "
  "in all profiles, but the magnitude must be read on the correct scale: "
  "absolute values are on the order of 980, the entire range across the four "
  "actions is worth about 1.9 % of that value, and the deficit of MEDIUM "
  "relative to the best action is about 1.2 %. The four actions are therefore "
  "nearly equivalent.", first=0.5)

legenda(f"Table {T('qvalues')}. Action values: rank of MEDIUM and the scale of "
        "the deficit.", acima=True)
tabela(["Controller", "Rank of MEDIUM", "Absolute deficit", "Mean value",
        "Range across actions", "Relative deficit", "Times best"],
       [[I.t(_r["controlador"]), v(_r["posicao_medium"], 2),
         v(_r["deficit_medium"], 2), v(_r["q_absoluto"], 0),
         v(_r["faixa_pct_do_valor"], 2) + " %",
         v(_r["deficit_pct_do_valor"], 2) + " %",
         v(_r["argmax_medium_pct"], 1) + " %"] for _, _r in QVAL.iterrows()],
       larguras=[3.4, 2.6, 2.4, 2.2, 2.6, 2.4, 2.4])

rico([("The mechanism is therefore ", 0, 0), ("winner-take-all", 0, 1),
      (": a deterministic policy obtained by maximisation converts a margin of "
       "about 1 % into 0 % usage. Under that decision regime there is no such "
       "thing as \"commanding MEDIUM occasionally\", either the action is best "
       "in some state, or it never appears. The practical consequence is "
       "direct: reweighting reward terms is the wrong path, since the three "
       "profiles already span wide variation and collapse in the same way. "
       "What restores use of the intermediate levels is changing the policy "
       "class, to continuous or stochastic, or granting the agent finer "
       "actuation authority.", 0, 0)], first=0.5, space_before=6)

h("4.9. Evaluation on a broad, independent set", 2)

p("The factorial matrix has nine points and was used to tune the PI "
  "controller, which is the methodological caveat recorded in Section 4.2. "
  "Evaluation was therefore also carried out on 150 scenarios sampled at "
  "random over the continuous space, initial temperature 16 to 33 °C, "
  "occupancy 0 to 45 people, start hour 0 to 23, with a fixed seed distinct "
  "from the training ones. All controllers receive exactly the same scenarios, "
  "which makes the comparison paired. For the PI this is an out-of-sample "
  "test.", first=0.5)

_g = ALEAT.groupby("controlador").agg(
    larga=("conf_larga_pct", "mean"), estreita=("conf_estreita_pct", "mean"),
    tol=("na_tolerancia_pct", "mean"), desvio=("desvio_ideal", "mean"),
    kwh=("energia_kwh", "mean")).sort_values("tol", ascending=False)
legenda(f"Table {T('aleatorios')}. Performance on 150 independent random "
        "scenarios.", acima=True)
tabela(["Controller", "Comfort [22,26] %", "Comfort [23,25] %",
        "Within ±0.5 °C %", "|T−24| °C", "Energy kWh/day"],
       [[I.t(_n), v(_r["larga"]), v(_r["estreita"]), v(_r["tol"]),
         v(_r["desvio"], 2), v(_r["kwh"], 2)] for _n, _r in _g.iterrows()],
       larguras=[4.4, 2.4, 2.4, 2.8, 2.0, 2.4])

p("Three readings. First, in the wide band the four best controllers produced "
  "an identical value: the metric was saturated, and the number is determined "
  "by the start-up transient, limited by equipment physics and not by the "
  "policy. Second, only the narrow tolerance discriminates, and there the "
  "Aggressive profile matched the PI, a result more favourable to learning "
  "than the factorial matrix suggested. Third, the PI held its performance "
  "out of sample, which mitigates, without eliminating, the caveat about its "
  "tuning.", first=0.5, space_before=6)

_tol = ALEAT.groupby("controlador")["na_tolerancia_pct"].mean()
_amp = _tol.max() - _tol.min()

rico([("The saturation deserves more attention than a metric defect would, "
       "because it suggests a fundamental objection to the whole paper: ", 0, 0),
      ("an environment in which four different controllers produce the same "
       "number may simply lack the resolution to discriminate between policies", 0, 0),
      (", in which case the absence of a learning advantage would be a "
       "consequence of the environment's parameterisation and not a property "
       "of the air-conditioning problem. The 30 h time constant discussed in "
       "Section 3.1 gives substance to the objection, and it is acknowledged "
       "as the most serious threat to the external validity of these results. "
       "Three elements, however, prevent it from explaining the whole picture. "
       "The saturation is specific to the wide-band metrics and to the "
       "transient, and disappears under the ±0.5 °C tolerance, where the "
       f"controllers separate by {v(_amp)} percentage points. The separation "
       "that appears there is not noise, since it concentrates in an "
       "identifiable condition, nearly empty rooms, and has a mechanism "
       "measured in Section 4.8. And energy use, which saturates in none of "
       "the conditions evaluated, orders the controllers consistently. What "
       "high inertia compromises, therefore, is the transferable magnitude of "
       "the differences, not the existence of the mechanism that produces "
       "them.", 0, 0)], first=0.5)

figura("divergencia", F.fig_divergencia_por_condicao(ALEAT))
legenda(f"Fig. {Fg('divergencia')}. Difference relative to the PI by occupancy "
        "band. The deficit of the Balanced and Passive profiles concentrates "
        "in empty rooms.")

p(f"The divergence is not diffuse but concentrated (Fig. {Fg('divergencia')}): "
  "the Balanced and Passive profiles lost 21.7 and 18.2 percentage points in "
  "scenarios with up to ten occupants, against 4.4 and 3.1 points with a full "
  "room. The worst cases share a signature, few occupants, small hours, in "
  "which the PI remained entirely within tolerance while the agent fell to the "
  "37 to 58 % range.", first=0.5, space_before=6)

p("The mechanism is consistent with Section 4.8: with minimal thermal load, "
  "holding ±0.5 °C requires very low and finely dosed power, and the smallest "
  "non-zero level available already represents 25 % of rated capacity. The "
  "proportional controller obtains an effective mean below any isolated level "
  "by alternating with the appropriate duty cycle; the discrete policy "
  "oscillates between overcooling and allowing the temperature to rise. The "
  "Aggressive profile does not suffer from this effect, since it operates "
  "on-off and commands maximum power without hesitation, at the cost of energy "
  "use.", first=0.5)

figura("pareto", F.fig_pareto_aleatorios(ALEAT))
legenda(f"Fig. {Fg('pareto')}. Precision against energy in the random "
        "scenarios. No learned profile occupies the optimal corner.")

p(f"The consolidated picture (Fig. {Fg('pareto')}) is one of agents lying "
  "below the Pareto front: to match the PI on precision, the Aggressive "
  "profile consumed 12.9 % more energy; to match its energy use, the Balanced "
  "and Passive profiles give up between six and seven points of precision. "
  "Since the three profiles were obtained by varying reward weights alone, "
  "the central thesis of the audited manuscript, the result indicates that "
  "this variation traces a curve lying entirely inside the front, without "
  "touching it.", first=0.5, space_before=6)

h("4.10. Speed of response", 2)

p("The preceding metrics assess steady-state quality and do not measure speed "
  "of response, which is an independent operational requirement: a room that "
  "takes hours to become usable is a problem even if it is adequate "
  "thereafter. This section measures the transient over the random scenarios "
  "that start above the band, reporting three distinct quantities, the "
  "instant of first entry into the band, the instant after which there is no "
  "further exit, and the fraction of the transient in which the actuator "
  "operates saturated.", first=0.5)

_g = TRANS.groupby("controlador").agg(
    ent=("entrada_h", "mean"), sd=("entrada_h", "std"),
    aco=("acomodacao_h", "mean"), tax=("taxa_c_por_h", "mean"),
    sat=("saturacao_pct", "mean")).sort_values("ent")
legenda(f"Table {T('transitorio')}. Transient response on scenarios starting "
        "above the band.", acima=True)
tabela(["Controller", "Entry (h)", "Std. dev.", "Settling (h)", "Rate (°C/h)",
        "Saturation %"],
       [[I.t(_n), v(_r["ent"], 2), v(_r["sd"], 2), v(_r["aco"], 2),
         v(_r["tax"], 2), v(_r["sat"], 0)] for _n, _r in _g.iterrows()],
       negrito_linhas=(0,), larguras=[4.6, 2.2, 1.8, 2.6, 2.2, 2.2])

figura("transitorio", F.fig_transitorio(TRANS))
legenda(f"Fig. {Fg('transitorio')}. Time to enter the band and time to stop "
        "leaving it. The distance between the two bars reveals the oscillatory "
        "behaviour of the thermostat.")

p(f"The PI controller and the three DQN profiles showed identical times (Fig. "
  f"{Fg('transitorio')}), differing in zero of the forty-eight scenarios "
  "evaluated. The explanation is in the last column: all operate at maximum "
  "power throughout the transient. During initial cooling there is no decision "
  "to make, the only sensible action is full power, and speed is determined "
  "by equipment physics, not by the control policy. This result confirms by "
  "measurement the explanation offered in Section 4.9 for the identical "
  f"comfort observed in Table {T('aleatorios')}, until now an inference.",
  first=0.5, space_before=6)

p("The thermostat revealed its characteristic defect in the distance between "
  "the two quantities: it entered the band at 1.99 h, but took about twenty "
  "hours to stop leaving it, since it parks at the ceiling and oscillates "
  "around it. A metric considering only first entry would favour it unduly. "
  "The SAC agent, in turn, was 3.1 times slower than the PI, and the cause is "
  "the same column: it saturated the actuator in only 31 % of the transient.",
  first=0.5)

rico([("One observation about the metric set is worth recording. This is the "
       "third quantity on which PI and DQN produce indistinguishable values, "
       "alongside wide-band comfort and comfort on random scenarios. The "
       "pattern is not statistical coincidence: ", 0, 0),
      ("outside steady state the optimal policy is trivial", 0, 0),
      (", maximum power, and any competent controller finds it. "
       "Differentiation between controllers exists only in steady state, and "
       "there the decisive factor is actuation resolution, for the reasons of "
       "Section 4.8.", 0, 0)], first=0.5)

h("4.11. Can the inferred parameters be derived?", 2)

p("Four constants of the reward function are not published by the audited "
  "manuscript and were inferred from its figure, which is the principal "
  "reproducibility caveat of this work. It is worth asking whether some of "
  "them can be obtained by argument rather than by reading a plot.",
  first=0.5)

rico([("The base bonus B admits a purely analytical answer. It appears in both "
       f"branches of the comfort term (Eqs. {E('conforto_dentro')} and "
       f"{E('conforto_fora')}), and the episode has fixed duration with no "
       "early termination; adding a constant at every step adds the same value "
       "to the return of any policy and therefore ", 0, 0),
      ("does not change the ordering between them", 0, 0),
      (". The optimal policy is independent of B. A numerical effect remains, "
       "however: B inflates the scale of the action values without carrying "
       "information, which compresses the relative difference between actions "
       ", precisely the condition identified in Section 4.8.", 0, 0)],
     first=0.5)

p("The curvature k, in turn, proves incoherent by simple inspection of the "
  "cost scale that the formulation produces.", first=0.5)

legenda(f"Table {T('coerencia')}. Reward cost of each departure, under the "
        "inferred parameters.", acima=True)
tabela(["Movement over the temperature range", "Cost in reward"],
       [["From centre to edge, from inside (24 to 26 °C)",
         v(PARAM["custo_centro_ate_borda"], 2)],
        ["Leaving 0.5 °C beyond the edge", v(PARAM["custo_fora_por_delta"][0.5], 2)],
        ["Leaving 1 °C beyond the edge", v(PARAM["custo_fora_por_delta"][1], 2)],
        ["Leaving 2 °C beyond the edge", v(PARAM["custo_fora_por_delta"][2], 2)],
        ["Slope next to the edge, from inside",
         v(PARAM["inclinacao_interna_por_c"], 2) + " per °C"],
        ["Slope 0.1 °C from the edge, from outside",
         v(PARAM["inclinacao_externa_em"][0.1], 2) + " per °C"]],
       larguras=[8.6, 5.4])

p("Remaining 2 °C outside the band costs less than traversing the half-band "
  "from inside it, and the marginal penalty drops from 2.00 to 0.12 per degree "
  "on crossing the boundary, so that leaving becomes a relief at the margin. "
  "Imposing that leaving by δ degrees cost the same as traversing the "
  "half-band gives k = 4 for δ = 1 °C and k = 1 for δ = 2 °C, against the 0.6 "
  "inferred. It should be noted that derivative continuity is unattainable "
  "with a pure quadratic penalty, since its slope is zero at the boundary; "
  "achieving it would require a linear term.", first=0.5, space_before=6)

_ordem_calib = ["B = 10.0", "B = 0.0", "k = 0.6", "k = 1.0", "k = 4.0",
                "rho = -5.0", "rho = -20.0", "frio = -2.0", "frio = -10.0"]
_l = []
for _c in _ordem_calib:
    _s = CALIB[CALIB["variante"] == _c]
    if _s.empty:
        continue
    _l.append([I.t(_c), v(_s["conf_larga"].mean()), v(_s["conf_estreita"].mean()),
               v(_s["conf_estreita"].std(), 2), v(_s["desvio"].mean(), 2),
               v(_s["violacoes_dmin_pct"].mean(), 0) + " %"])
legenda(f"Table {T('derivados')}. Sweep over the four parameters, two seeds "
        "per configuration.", acima=True)
tabela(["Configuration", "Wide comfort %", "Narrow comfort %",
        "Std. dev. between seeds", "|T−24| °C", "Violations of d_min"], _l,
       larguras=[2.8, 2.4, 2.6, 2.8, 2.0, 2.4])

figura("derivados", F.fig_parametros_derivados(COMB, CALIB))
legenda(f"Fig. {Fg('derivados')}. Effect of deriving B and k. Left, each "
        "parameter in isolation; right, the combination. Points are individual "
        "seeds.")

p(f"The sweep (Fig. {Fg('derivados')}) confirms the two analytical predictions "
  "and refutes the two remaining ones. Zeroing the base bonus preserves "
  "wide-band comfort, as the analysis required, and raises narrow-band comfort "
  "from 70.2 % to 82.7 %, with the scale of the action values reduced from 356 "
  "to 145. Adopting k = 1 raises it to 79.8 % and reduces minimum-dwell "
  "violations. By contrast k = 4 adds no performance and destabilises "
  "training, with a between-seed standard deviation of 13.2.", first=0.5,
  space_before=6)

rico([("As for ρ, quadrupling the penalty did ", 0, 0), ("not", 0, 0),
      (" reduce violations, which went from 82.6 % to 83.4 %, and it "
       "destabilised training. The result corroborates the analysis of the "
       f"form of Eq. {E('ciclo')}: being proportional to how early the switch "
       "occurs, the penalty is worth only 0.83 when the switch happens at five "
       "of the six required steps, that is, it practically vanishes where the "
       "violation is most likely. No value of ρ corrects a penalty that "
       "vanishes precisely where it should act; the correction is structural, "
       "as in Section 4.4.", 0, 0)], first=0.5)

p("The cold penalty, finally, proves correctly chosen. Raising it from −2 to "
  "−10 collapses the policy, with wide-band comfort falling from 86.5 % to "
  "41.4 %, which literally reproduces the warning recorded in the conclusion "
  "of the audited manuscript regarding the induction of avoidance behaviour. "
  "One precision is due: the term is not incurred at any step under the "
  "trained policy, but it is incurred at 63.5 % of steps under a random "
  "policy, which is the regime at the start of training, accounting for about "
  "9 % of the return. It is therefore not an inert term, but an effective one "
  "whose success consists exactly in ceasing to be triggered.", first=0.5)

legenda(f"Table {T('combinacao')}. Derived configuration against the inferred "
        "one, under the reduced budget of the calibration (300,000 steps, "
        "three seeds).", acima=True)
tabela(["Configuration", "Wide comfort %", "Narrow comfort %", "Per seed",
        "Std. dev.", "|T−24| °C", "kWh/day"],
       [[I.t(_c), v(COMB[COMB["config"] == _c]["conf_larga"].mean()),
         v(COMB[COMB["config"] == _c]["conf_estreita"].mean()),
         " / ".join(v(x) for x in COMB[COMB["config"] == _c]["conf_estreita"]),
         v(COMB[COMB["config"] == _c]["conf_estreita"].std(), 2),
         v(COMB[COMB["config"] == _c]["desvio"].mean(), 2),
         v(COMB[COMB["config"] == _c]["energia"].mean(), 2)]
        for _c in COMB["config"].unique()],
       negrito_linhas=(1,), larguras=[3.6, 2.2, 2.4, 3.2, 1.6, 1.8, 1.8])

p("Under 300,000 steps the two derived parameters raise narrow-band comfort "
  "from 64.5 % to 81.2 %, with complete separation between seeds and the "
  "standard deviation falling from 10.14 to 2.51. The result appeared to "
  "authorise replacing the inferred values with the derived ones.", first=0.5,
  space_before=6)

rico([("That conclusion, however, does not survive the protocol's budget. "
       "Retraining the three profiles with 550,000 steps and three seeds, ",
       0, 0), ("the effect reverses sign", 0, 0), (".", 0, 0)], first=0.5)

_r = ORC["resumo"].set_index(["orcamento", "config"])
_l = []
for _o in ("300k", "550k"):
    for _c in ("inferidos", "derivados"):
        if (_o, _c) not in _r.index:
            continue
        _x = _r.loc[(_o, _c)]
        _l.append([f"{_o[:3]},000 steps", I.t(_c), v(_x["media"]),
                   v(_x["desvio"], 2), str(int(_x["n"]))])
legenda(f"Table {T('orcamento')}. Narrow-band comfort by training budget and "
        "reward configuration.", acima=True)
tabela(["Budget", "Configuration", "Narrow comfort %", "Std. dev.", "n"], _l,
       larguras=[3.4, 3.4, 3.0, 2.2, 1.6])

figura("orcamento", F.fig_orcamento_parametros(ORC))
legenda(f"Fig. {Fg('orcamento')}. Interaction between reward parameters and "
        "training budget. The curves cross: the derived configuration "
        "converges faster and then degrades, while the inferred one learns "
        "slowly and keeps improving.")

p(f"The derived configuration converges faster and then degrades (Fig. "
  f"{Fg('orcamento')}); the inferred one learns slowly and keeps improving. "
  "The curves cross between the two budgets. The behaviour is consistent with "
  "instability of temporal-difference learning: raising the curvature widens "
  "the dynamic range of the regression targets outside the comfort band, and "
  "removing the base bonus reduces the scale of the values, a combination that "
  "favours initial convergence and harms prolonged stability. The between-seed "
  "dispersion of the derived configuration at 550,000 steps, 9.95 against 0.10 "
  "for the inferred one, reinforces this reading.", first=0.5, space_before=6)

rico([("Three consequences. The first concerns the parameters: the inferred "
       "values, incoherent though they are by the scale analysis of "
       f"Table {T('coerencia')}, are the ones that work under the budget "
       "adopted, and they remain in use throughout the other sections of this "
       "paper. The second is methodological and transcends this case: ", 0, 0),
      ("calibrating reward parameters under a reduced budget and extrapolating "
       "to the full budget is invalid", 0, 0),
      (", since the interaction between formulation and training duration is "
       "not monotonic. The third concerns the reproducibility caveat: the "
       "analysis of B and k remains valid as a criticism of the formulation, "
       "but does not authorise replacing the values, so the caveat about the "
       "four unpublished parameters stands.", 0, 0)], first=0.5)

# ============================================ 4.12. TRANSFERÊNCIA (BOPTEST)
if BOPT is not None and BRES is not None:
    h("4.12. Transfer to a third-party emulator", 2)

    _pk, _tp = "peak_cool_day", "typical_cool_day"
    _b = BRES

    # Todo número citado no texto desta seção sai daqui, e não da leitura da
    # tabela: é o que impede que o parágrafo e a tabela ao lado divirjam.
    def _celula(periodo, controlador, coluna):
        _s = BOPT[(BOPT["periodo"] == periodo)
                  & (BOPT["controlador"] == controlador)][coluna]
        return float(_s.iloc[0])

    _PI_R = "PI re-sintonizado (emulador)"
    _narrow_pi_pk = _celula(_pk, _PI_R, "comfort_narrow_pct")
    _narrow_ag_pk = _celula(_pk, _b[_pk]["melhor_agente"], "comfort_narrow_pct")
    _kwh_pi_pk = _celula(_pk, _PI_R, "energy_kwh_day")
    _kwh_ag_pk = _celula(_pk, _b[_pk]["melhor_agente"], "energy_kwh_day")
    _tdis_pi_pk = _celula(_pk, _PI_R, "kpi_tdis_tot")
    _tdis_ag_pk = _celula(_pk, _b[_pk]["melhor_agente"], "kpi_tdis_tot")
    _tdis_pi_tp = _celula(_tp, _PI_R, "kpi_tdis_tot")
    _tdis_ag_tp = _celula(_tp, _b[_tp]["melhor_agente"], "kpi_tdis_tot")

    p("Section 5.1 argues that the niche of reinforcement learning is model "
      "mismatch, and that a self-authored simulator cannot exhibit it by "
      "construction. That argument bounds every result reported so far, and it "
      "is testable: the same controllers can be run against an emulator written "
      "by someone else. This section does so, using the BOPTEST framework "
      f"{cite('blum')}, the reference benchmark for building control, and its "
      f"`bestest_air` case, a single zone derived from BESTEST Case 900 "
      f"{cite('judkoff')} with a fan-coil unit, measured weather for Denver, "
      "and its own internal and solar gains.", first=0.5)

    rico([("What does not change is the policy. The agents are loaded from the "
           "same files evaluated throughout this paper and executed ", 0, 0),
          ("without retraining", 0, 0),
          (", deterministically, and the observation is assembled by the same "
           "single declaration described in Section 3.6, so that every channel "
           "reaches the policy with the meaning it was trained with. What "
           "changes is the plant. Two of the case's cooling periods are used, "
           f"in which the zone without conditioning reaches {v(35.5)} °C and "
           f"{v(30.0)} °C respectively.", 0, 0)], first=0.5)

    rico([("One transfer decision must be reported before the results, because "
           "it is itself a finding. The protocol of this paper decides every "
           "12 min, which is benign in a plant whose full load moves the room "
           "0.09 °C per step. In the emulator full load moves the zone ", 0, 0),
          ("8.8 °C in the same 12 min", 0, 0),
          (", more than twice the entire width of the comfort band. Keeping the "
           "12 min would not preserve the protocol; it would turn the problem "
           "into pure on-off for every controller and the comparison would stop "
           "discriminating. The control interval is therefore derived from a "
           "design condition stated before measuring, full load must not cross "
           "more than half the comfort band in one decision interval, which "
           "yields 180 s. This is the discipline Section 5.2 recommends, and "
           "the number is a measurement, from the outside, of the threat "
           "declared in Section 5.3: the plant of this paper is slow, and its "
           "equipment small, relative to a real zone.", 0, 0)], first=0.5)

    legenda(f"Table {T('boptest')}. Transfer to the BOPTEST `bestest_air` case. "
            "Agents run without retraining; the PI controller appears twice, "
            "with the gains tuned on the local plant and with gains retuned "
            "inside the emulator. Best value per period in bold.", acima=True)

    _rot = {"peak_cool_day": "Peak cooling day",
            "typical_cool_day": "Typical cooling day"}
    _lb, _neg = [], []
    for _per in (_pk, _tp):
        _sub = BOPT[BOPT["periodo"] == _per]
        _melhor = _sub["comfort_wide_pct"].max()
        for _i, (_, _r) in enumerate(_sub.iterrows()):
            if abs(_r["comfort_wide_pct"] - _melhor) < 1e-9:
                _neg.append(len(_lb))
            _lb.append([_rot[_per] if _i == 0 else "",
                        I.t(_r["controlador"]),
                        v(_r["comfort_wide_pct"]),
                        v(_r["comfort_narrow_pct"]),
                        v(_r["abs_dev_from_ideal"], 2),
                        v(_r["energy_kwh_day"], 2),
                        v(_r["kpi_tdis_tot"], 2)])
    tabela(["Period", "Controller", "Comfort [22, 26] %", "Comfort [23, 25] %",
            "|T−24| °C", "kWh/day", "tdis_tot (Kh)"],
           _lb, negrito_linhas=tuple(_neg),
           larguras=[2.6, 4.4, 2.3, 2.3, 1.8, 1.6, 2.1])

    figura("boptest", F.fig_boptest_transferencia(BOPT))
    legenda(f"Fig. {Fg('boptest')}. Comfort under transfer. The hatched bar is "
            "the PI controller carrying the gains tuned on the local plant; the "
            "solid one is the same controller retuned inside the emulator. The "
            "bracket marks what retuning recovers.")

    rico([("Read with the frozen gains alone, the table reverses the central "
           "result of this paper: the best agent reaches ", 0, 0),
          (f"{v(_b[_pk]['melhor_agente_conf'])} % against "
           f"{v(_b[_pk]['pi_congelado'])} % on the peak cooling day and "
           f"{v(_b[_tp]['melhor_agente_conf'])} % against "
           f"{v(_b[_tp]['pi_congelado'])} % on the typical one", 0, 0),
          (f", advantages of {v(_b[_pk]['vantagem_agente_sobre_pi_congelado_pp'])}"
           f" and {v(_b[_tp]['vantagem_agente_sobre_pi_congelado_pp'])} "
           "percentage points for learning. Reporting that number alone would "
           "be the finding of this section, and it would be wrong.", 0, 0)],
         first=0.5, space_before=6)

    rico([("The reason is the defect this paper audits, committed here against "
           "our own baseline. The PI gains were tuned on a plant whose time "
           "constant is 30 h; the emulator responds several degrees in the same "
           "interval. Charging a classical controller for performance under a "
           "dynamics its gains were never fitted to is exactly what Section 4.2 "
           "identifies in the audited manuscript, displaced from one environment "
           "to another. Section 4.7 already fixed the rule: ", 0, 0),
          ("when the specification changes, both controllers are re-prepared", 0, 0),
          (". Here the plant changes, and the same rule applies.", 0, 0)],
         first=0.5)

    figura("boptest_sintonia", F.fig_boptest_sintonia(BOPTS))
    legenda(f"Fig. {Fg('boptest_sintonia')}. Grid search for the PI gains "
            "inside the emulator, on the peak cooling day. The gains inherited "
            "from the local plant sit far from the optimum of this one.")

    _best = BOPTS.sort_values(["conf_larga", "conf_estreita"],
                              ascending=False).iloc[0]
    rico([(f"A grid search over {len(BOPTS)} combinations (Fig. "
           f"{Fg('boptest_sintonia')}) recovers "
           f"{v(_b[_pk]['ganho_resintonia_pp'])} percentage points on the "
           f"tuning period, moving the PI controller from "
           f"{v(_b[_pk]['pi_congelado'])} % to {v(_b[_pk]['pi_resintonizado'])} "
           f"% with Kp = {v(_best['kp'], 1)} and Ki = {v(_best['ki'], 2)}. "
           "With both sides re-prepared, ", 0, 0),
          ("the ordering of Section 4.2 is restored", 0, 0),
          (f": the PI controller leads by {v(_b[_pk]['vantagem_pi_resintonizado_pp'])}"
           f" and {v(_b[_tp]['vantagem_pi_resintonizado_pp'])} percentage points, "
           "and its margin on the narrow band is far larger, "
           f"{v(_narrow_pi_pk)} % against {v(_narrow_ag_pk)} % on the peak day, "
           f"at {v(_kwh_pi_pk, 2)} kWh/day against {v(_kwh_ag_pk, 2)}. The "
           "typical cooling day is out of sample for the tuning, which was "
           "performed on the peak day alone, and there the retuned controller "
           f"reaches {v(_b[_tp]['pi_resintonizado'])} % of the occupied window "
           "inside the band.", 0, 0)], first=0.5, space_before=6)

    rico([("One divergence must be reported rather than smoothed over. Under "
           "the framework's own thermal discomfort indicator, ", 0, 0),
          ("the agents lead", 0, 0),
          (f" ({v(_tdis_ag_pk, 2)} against {v(_tdis_pi_pk, 2)} Kh on the peak "
           f"day and {v(_tdis_ag_tp, 2)} against {v(_tdis_pi_tp, 2)} Kh on the "
           "typical one), because that indicator is computed against the "
           "setpoints of the case, which include a night setback, and not "
           "against the [22, 26] °C band that this paper controls for. The two "
           "families of metric therefore reward different objectives, and the "
           "winner changes with the choice. That is the same lesson as the rest "
           "of this paper, with the metric in the place of the baseline.",
           0, 0)], first=0.5)

    p("Two limits of this experiment are worth stating. The fan-coil of the "
      "emulator does not reproduce the part-load COP peak of the local "
      "equipment, so the mechanism of Section 4.8 is not testable here and none "
      "of these numbers speak to it. And the emulator publishes no occupant "
      "count: the occupancy channel is nominal, following the occupied window "
      "of the configuration, which is the semantics the policies were trained "
      "with. What the section supports is narrow and, for that reason, solid, "
      "the central result survives a plant this work did not write, provided "
      "both controllers are given the same preparation.", first=0.5)


# ================================================================= 5. DISCUSSION
h("5. Discussion")

h("5.1. Why classical control dominates in this formulation", 2)

p("The result does not follow from fortunate tuning nor from deficient "
  "implementation of the agents. It follows from the class of the problem. The "
  "modelled plant is single-input single-output, first order, essentially "
  "linear over the operating range, with a known two-parameter model and a "
  "measurable disturbance, and the objective is to track a setpoint or hold a "
  "band. This is the canonical case in which proportional-integral control is "
  "optimal or near-optimal.", first=0.5)

p("Reinforcement learning has an established advantage in regimes that this "
  "formulation does not exhibit: strong non-linearity, high-dimensional "
  "coupling, objectives not expressible as a quadratic cost, combinatorial "
  "allocation decisions, or an unknown or time-varying model. The three "
  "extensions evaluated in Section 4.6 introduced difficulty but did not "
  "change the class of the problem: a harder tracking problem is still a "
  "tracking problem, and responds to more aggressive tuning.", first=0.5)

rico([("There is also a methodological limitation considered the most relevant "
       "of this work: ", 0, 0),
      ("the niche of RL is model mismatch, and a self-authored simulator "
       "exhibits no mismatch by construction", 0, 0),
      (". Any comparison conducted entirely inside one's own environment "
       "structurally favours methods that exploit the model, MPC included. "
       "Testing the hypothesis of an RL advantage requires "
       "simulation-to-reality transfer or a third-party benchmark, as in "
       f"Boutahri and Tilioua {cite('boutahri')} and Dai et al. {cite('dai')}. "
       "Section 4.12 takes the second route, and its outcome refines this "
       "argument rather than merely confirming it: against an emulator written "
       "by others, the learned policies do overtake the PI controller carrying "
       "gains fitted to the local plant, and lose that lead once the PI is "
       "retuned for the new dynamics. Model mismatch, in this instance, "
       "penalised the frozen classical controller more than it rewarded the "
       "learned one.", 0, 0)], first=0.5)

p("There is a second mechanism, specific to the discrete action space and "
  "identified in Section 4.8. A deterministic policy obtained by maximisation "
  "selects a single action in each state; when several actions have nearly "
  "equivalent value, as here, where the range across the four options is "
  "worth about 1.9 % of the absolute value, the one that is never maximal "
  "disappears entirely. A proportional controller, by contrast, naturally "
  "traverses all levels as it sweeps the error range. The consequence is "
  "economic: the level of best coefficient of performance ceases to be used, "
  "and the agent compensates with the least efficient level.", first=0.5)

p("This finding has a design implication that transcends the case studied. In "
  "systems whose actuator has few discrete levels and whose efficiency is "
  "non-monotonic in load, which describes a large share of inverter-equipped "
  "air-conditioning equipment, the discretisation of the action space "
  "interacts with the structure of the policy so as to penalise precisely the "
  "most efficient operating regime. Granting finer actuation authority, "
  "through continuous action or duty-cycle modulation, is a more promising "
  "intervention than reweighting reward terms.", first=0.5)

h("5.2. Methodological implications", 2)

p("Four recommendations follow directly from the findings. First, the baseline "
  "configuration should be reported in the same detail as the agent "
  "architecture, and preferably tuned with the same computational effort; "
  f"Table {T('decomposicao')} shows that the difference between a configured "
  "and an unconfigured baseline can exceed the entire claimed contribution. "
  "Second, safety properties should be verified through their distribution and "
  "not through aggregates: the same policy that exhibits 2.5 switches per hour "
  "has a median dwell time of 12 minutes against a requirement of 36. Third, "
  "constants that define constraints should be derived from design conditions, "
  "not arbitrated, a constraint calibrated to produce the desired result is "
  "the same defect this work identifies in the audited baseline, displaced to "
  "another component. Fourth, saturated metrics should be detected and "
  "discarded: in the independent set of 150 scenarios, four different "
  "controllers produce an identical value of wide-band comfort, because the "
  "number is determined by the start-up transient and not by the policy. A "
  "metric that does not discriminate should not be the headline metric of a "
  "paper.", first=0.5)

h("5.3. Threats to validity", 2)

for ameaca in [
    "Parameters not published by the audited specification. Four constants "
    "(B, k, ρ and the cold penalty) had to be inferred. Section 4.11 partially "
    "mitigates this caveat: B and k become derivable from an explicit "
    "criterion, ρ proves insensitive to its value, and the cold penalty has "
    "its choice confirmed by measurement. The finding of Section 4.2 is "
    "internally valid, both controllers operate in the same environment, but "
    "its transfer to the audited setting is inference, not direct measurement. "
    f"Table {T('status')} distinguishes the findings by that criterion: those "
    "of Sections 4.4 and 4.5, depending only on published parameters, do not "
    "carry this caveat.",
    "Scope of the audited claim. The audit is a case study: the pattern "
    "documented in Section 2.2 for published work is measured here end to end, "
    "with full access to the environment, the agents and the baselines, which "
    "an audit conducted from the outside would not have. The recommendations "
    "of Section 5.2 are accordingly formulated about the literature, and not "
    "about a single manuscript.",
    "Tuning and evaluation on the same set. The PI controller was tuned over "
    "the same 3×3 matrix used for evaluation. The result should be read as "
    "equivalence under the conditions in which both were adjusted. Separation "
    "between calibration and test partitions is implemented and running it is "
    "immediate future work.",
    "Number of seeds. Three seeds per condition impose a floor of p ≈ 0.101 in "
    "a permutation test, which is why effect size is reported. Conclusions "
    "supported by δ = −1.00 are robust; those supported by small magnitudes "
    "are suggestive.",
    "High thermal inertia, the most serious threat. The time constant of the "
    "environment, 30 h, exceeds the episode duration and corresponds to about "
    "seventy times the air volume of a conventional classroom. Disturbances "
    "are strongly damped, which makes control easier and compresses the "
    "differences between controllers; the metric saturation reported in "
    "Sections 4.9 and 4.10 is the direct manifestation of this. The "
    "conservative reading is that the magnitude of the differences measured "
    "here does not transfer to a real room, and that the tie between PI and "
    "DQN in the wide band is in part a property of the parameterisation and "
    "not of the air-conditioning problem. Re-running the protocol under a "
    "physically plausible thermal capacitance is therefore the pending "
    "experiment with the highest return, ahead of increasing the number of "
    "seeds: measuring more precisely a tie that the plant already favours does "
    "not change the conclusion. Section 4.12 quantifies this threat from the "
    "outside: in a third-party zone, full load moves the temperature in one "
    "12 min step by more than twice the width of the comfort band, against "
    "0.09 °C per step here.",
    "Scope of the transfer. Section 4.12 uses one test case and two cooling "
    "periods of a single climate, with the agents transferred zero-shot. It is "
    "enough to show that the ordering survives when both controllers are "
    "re-prepared, and not enough to establish it across building types, "
    "climates or system topologies. The emulator's fan-coil also lacks the "
    "part-load COP peak of the local equipment, so the mechanism of Section "
    "4.8 remains untested outside this work.",
    "Absence of hardware validation and of MPC as a reference. All results are "
    "from simulation, with a lumped thermal balance, without spatial "
    "gradients, humidity or CO₂, and with an instantaneous actuator. Model "
    "predictive control, the natural opponent in allocation regimes, was not "
    "implemented.",
]:
    item(ameaca)

# ================================================================ 6. CONCLUSIONS
h("6. Conclusions")

p("A deep reinforcement learning HVAC controller was implemented from its "
  "text and its central claim audited under competently configured baselines. "
  "The reproduction proved faithful in physics, comfort and generalisation, "
  "and divergent in energy and switching, an asymmetry that bounds every "
  "subsequent claim. The claimed advantage does not survive: a PI controller "
  "with two constants matches the DQN on comfort and beats it on deviation "
  "from setpoint and on cost. The decomposition attributes approximately the "
  "whole advantage to inadequate configuration of the opponent.", first=0.5)

p("The ablation shows that three of the five terms of the proposed reward are "
  "inert and that the formulation does not beat a conventional quadratic; the "
  "contribution, correctly framed, is the correction of a defect introduced by "
  "adopting the flat plateau. The anti-short-cycling penalty does not meet its "
  "objective, and a structural constraint meets it at practically zero cost. "
  "Conversely, tabular methods are inadequate in this environment for a "
  "measured theoretical reason, 79.8 % intra-tile transitions, so that if RL "
  "is to be used here, it has to be deep. The criterion that generalises is "
  "not the number but the ratio between the typical step of the dynamics and "
  "the resolution of the discretisation, computable before training any "
  "agent.", first=0.5)

p("We further investigated whether a greater precision requirement would "
  "reverse the picture, and the measured effect is the opposite: with both "
  "controllers re-prepared for each band width, the mean advantage of the PI "
  "grows as the specification tightens, being null at wide band, of the order "
  "of seed noise at intermediate band, and consistent across seeds at narrow "
  "band. Evaluating performance against the training budget shows that this is "
  "not explained by insufficient training at the wide band, where the agent "
  "converges exactly onto the reference, even though at the narrow band the "
  "curve is still rising slowly at the end of the budget, which characterises "
  "a high sample cost rather than impossibility.", first=0.5)

p("The mechanism of the excess energy use was also identified: the three "
  "reward profiles discard entirely the power level of best coefficient of "
  "performance. The cause lies not in the weights, which vary by factors of "
  "three to four between profiles without changing the behaviour, but in the "
  "deterministic nature of the policy, which converts a value difference of "
  "about 1 % into complete absence of use. A continuous agent trained with the "
  "identical reward uses that level in the same proportion as the classical "
  "controller.", first=0.5)

p("We further examined whether the unpublished reward constants could be "
  "obtained by argument instead of graphical inference. Two admit it "
  "analytically: the base bonus, being an additive constant under an episode "
  "of fixed duration, does not change the optimal policy, and the curvature "
  "can be fixed by imposing coherence between the cost of leaving the band and "
  "that of traversing it internally. Empirical verification, however, produced "
  "a result that contradicts expectation and deserves recording: under a "
  "reduced budget the derived values raise narrow-band comfort by 16.7 "
  "percentage points, but under the protocol's budget the effect reverses, "
  "with a fall of 17.6 points. The derived configuration converges faster and "
  "then degrades, while the inferred one learns slowly and keeps improving. "
  "From this follows a methodological recommendation of general reach: "
  "calibrating reward parameters under a reduced budget and extrapolating to "
  "the full budget is an invalid procedure. The two remaining constants proved "
  "not improvable by value: the switching penalty is insensitive to magnitude, "
  "since the defect lies in its proportional form, and the cold penalty was "
  "already correctly chosen, since severe values collapse the policy.",
  first=0.5)

p("Finally, the failure of three attempts to construct a regime favourable to "
  "RL was documented and their common cause identified: introducing difficulty "
  "does not change the class of the problem. We conclude that, for "
  "single-zone air conditioning with a known model and a tracking objective, "
  "reinforcement learning is not the indicated tool.", first=0.5)

rico([("The reach of that conclusion must be stated without ambiguity. ", 0, 0),
      ("These results are not evidence that deep reinforcement learning cannot "
       "beat classical control in HVAC", 0, 0),
      (". They are evidence that, in this formulation and in this environment, "
       "it does not, and that claims of superiority require evaluation against "
       "competent baselines and in environments exhibiting uncertainty and "
       "model mismatch to a relevant degree. Section 4.12 addresses the first "
       "of the two central caveats stated in Section 5.3, by running the same "
       "controllers against an emulator this work did not write; the second, "
       "that the time constant adopted makes the plant more benign than a real "
       "room, stands, and was in fact measured from the outside there. What "
       "this work supports, therefore, is a methodological requirement, and "
       "not a verdict on the family of methods.", 0, 0)], first=0.5)

p("From this follows the agenda. Section 4.12 has already discharged one item "
  "of it: the controllers were run against a third-party emulator, and the "
  "central result survived, provided both sides were given the same "
  "preparation. What remains, in order of priority, is to broaden that "
  "transfer, further BOPTEST cases and further periods, and a comparison "
  "against each case's own baseline controller, to re-run the protocol under "
  "a physically plausible thermal capacitance, and to raise the number of "
  "seeds in the main comparisons. Next, the investigation of multi-zone "
  "regimes with shared capacity, in which the allocation decision admits no "
  "local control law, with model predictive control as a mandatory reference.",
  first=0.5)

# ================================================================= DECLARAÇÕES
h("CRediT authorship contribution statement")
p(f"**{AUTOR}:** Conceptualization, Methodology, Software, Validation, Formal "
  "analysis, Investigation, Data curation, Writing – original draft, Writing – "
  "review & editing, Visualization.", first=0.0)

h("Declaration of competing interest")
# A declaração de conflito é o lugar formal da informação, e não o corpo do
# texto: é o campo que a editora indexa e o que um editor procura. Mantida
# autocontida, sem remeter a seções que não a repetem.
p("The author declares that he has no known competing financial interests or "
  "personal relationships that could have appeared to influence the work "
  "reported in this paper. The controller audited in this study derives from "
  "earlier, unpublished work by the same author. The limits this places on the "
  "findings are stated in Section 5.3.", first=0.0)

h("Data availability")
p("The complete material (environment, agents, baselines, regression suite "
  "and the full record of declared and effective hyperparameters) is "
  f"available for replication at {REPO}. Every number reported is produced by "
  "a single "
  "function, consumed both by the analysis notebooks and by the generator of "
  "this document: figures depending only on the environment and the models are "
  "recomputed in the same run that produces the tables, and the remaining ones "
  "read the same result files that feed those tables, so that divergence "
  "between a figure and the table beside it is impossible by construction. "
  "Table, figure and equation numbering is derived from the composition rather "
  "than written by hand. The values actually plotted are written to a "
  "dedicated file for auditing.", first=0.0)

# ================================================================== REFERÊNCIAS
h("References")

_NAO_CITADAS = [k for k in BIBLIO if k not in _ORDEM_CITACAO]
if _NAO_CITADAS:
    raise AssertionError(
        "entradas de BIBLIO que nunca são citadas no texto, o estilo numerado "
        f"não admite referência sem chamada: {_NAO_CITADAS}")

for i, chave in enumerate([] if TEX else _ORDEM_CITACAO, start=1):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = par.paragraph_format
    pf.left_indent = Cm(1.0)
    pf.first_line_indent = Cm(-1.0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    run = par.add_run(f"[{i}]\t{BIBLIO[chave]}")
    run.font.size = Pt(11)

if TEX:
    # A afiliação é uma linha única no .docx; a classe CAS a quer em campos
    # separados, que é o que o sistema de submissão indexa.
    _AFIL = {"organization": "Instituto Federal da Paraíba",
             "addressline": "Campus Cajazeiras", "city": "Cajazeiras",
             "postcode": "", "state": "Paraíba", "country": "Brazil"}
    with open(SAIDA, "w", encoding="utf-8") as fh:
        fh.write(TEXDOC.montar(
            titulo=TITULO, titulo_curto=TITULO_CURTO, autor=AUTOR,
            afiliacao=_AFIL, email=EMAIL, orcid=ORCID,
            credito=CREDITO,
            resumo=_ABSTRACT, destaques=_DESTAQUES,
            palavras_chave=PALAVRAS_CHAVE,
            bibliografia=[(k, BIBLIO[k]) for k in _ORDEM_CITACAO]))
    _CLASSE = os.path.join(RAIZ, "..", "els-cas-templates")
    _copiados = TB.instalar_classe(os.path.normpath(_CLASSE), DESTINO_TEX)
    print(f"generated: {SAIDA}")
    print(f"figures:   {FIGS}")
    print(f"class:     {', '.join(_copiados) or 'NÃO ENCONTRADA'}")
    print(f"abstract:  {_N_PALAVRAS} words | references: {len(_ORDEM_CITACAO)}")
else:
    doc.save(SAIDA)
    print(f"generated: {SAIDA}")
    print(f"figures:   {FIGS}")
    print(f"abstract:  {_N_PALAVRAS} words | references: {len(_ORDEM_CITACAO)}")
