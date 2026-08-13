# -*- coding: utf-8 -*-
"""
Gera o artigo em .docx no formato de congresso (padrão SBC/ENIAC).

TABELAS E FIGURAS VÊM DA MESMA FONTE QUE OS NOTEBOOKS: `hvac.results` para os
números, `hvac.figures` para os gráficos. Nenhum valor é digitado à mão, de modo
que divergência entre o artigo, os notebooks e o código é impossível por
construção, era o defeito da v4, cujos PNGs vinham de execuções diferentes das
que produziram os números reportados.

    python gerar_paper.py
"""
import os
import re
import sys
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # antes de importar hvac.figures: script headless

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

warnings.filterwarnings("ignore")

from hvac import figures as F
from hvac import results as R

# --- modo de composição -----------------------------------------------------
# A versão completa reúne todos os experimentos; a curta cabe no limite típico de
# páginas de um artigo de congresso, preservando a linha argumentativa central.
# A numeração de tabelas e figuras é DERIVADA da composição, e não escrita à mão:
# cortar uma seção sem renumerar tudo à mão é a origem clássica de referências
# quebradas num artigo.
CURTO = "--curto" in sys.argv

_TABS = ["relacionados", "fisica", "correspondencia", "hiperparametros",
         "generalizacao", "status", "comparacao", "decomposicao",
         "ablacao", "permanencia", "hnp", "precisao", "faixa", "niveis",
         "qvalues", "aleatorios", "transitorio", "coerencia", "derivados",
         "combinacao", "orcamento", "boptest", "boptest_sintonia"]
_FIGS = ["ciclo", "decomposicao", "ablacao", "permanencia", "precisao", "demanda",
         "faixa", "curva", "niveis", "consumo", "divergencia", "pareto",
         "transitorio", "derivados", "orcamento",
         "boptest", "boptest_sintonia"]

# O que sai na versão curta: a generalização reforça fidelidade mas não sustenta
# achado central; a Seção 4.6 é a mais longa e a menos ligada à tese; as duas
# figuras removidas duplicam tabelas vizinhas.
_SEM_TAB = {"generalizacao", "precisao"} if CURTO else set()
_SEM_FIG = {"precisao", "demanda", "consumo", "pareto"} if CURTO else set()

# As equações seguem a mesma disciplina das tabelas e figuras: a ordem é
# declarada aqui, na sequência em que aparecem no texto, e o número é derivado
# dela. Antes esses números eram literais no corpo do arquivo, e a equação de
# Bellman — a primeira do artigo — carregava o rótulo (7) porque havia sido
# escrita depois das demais. Referenciar por chave torna esse erro impossível.
_EQS = ["bellman", "fisica", "observacao", "recompensa",
        "conforto_dentro", "conforto_fora", "ciclo"]

_NT = {k: i + 1 for i, k in enumerate(t for t in _TABS if t not in _SEM_TAB)}
_NF = {k: i + 1 for i, k in enumerate(f for f in _FIGS if f not in _SEM_FIG)}
_NE = {k: i + 1 for i, k in enumerate(_EQS)}


def T(chave: str) -> int:
    """Número da tabela na composição vigente."""
    return _NT[chave]


def E(chave: str) -> int:
    """Número da equação, derivado da ordem declarada em `_EQS`."""
    return _NE[chave]


def Fg(chave: str) -> int:
    """Número da figura na composição vigente."""
    return _NF[chave]


def tem_tab(chave: str) -> bool:
    return chave in _NT


def tem_fig(chave: str) -> bool:
    return chave in _NF


SAIDA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "paper_auditoria_hvac_rl_curto.docx" if CURTO
    else "paper_auditoria_hvac_rl.docx")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras_v5")
os.makedirs(FIGS, exist_ok=True)

print("coletando resultados (recomputa o que é barato, lê o que é caro)...")
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


def salvar_fig(fig, nome):
    caminho = os.path.join(FIGS, nome)
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    return caminho


def v(x, casas=1):
    """Número no padrão decimal brasileiro."""
    return f"{x:.{casas}f}".replace(".", ",")


# Contagens pequenas vão por extenso no corpo do texto, por convenção editorial.
_EXT = {0: "nenhuma", 1: "uma", 2: "duas", 3: "três", 4: "quatro", 5: "cinco",
        6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez"}

REPO = "https://github.com/renansaraivaifpb/fundamentals-of-reinforcement-learning"

doc = Document()

# ---------------------------------------------------------------- formatação
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.top_margin, sec.bottom_margin = Cm(3.5), Cm(2.5)
sec.left_margin, sec.right_margin = Cm(3.0), Cm(3.0)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.0
# Garante a fonte também para o conjunto de caracteres complexos.
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


# --------------------------------------------------- tipografia do corpo do texto
#
# Duas convenções editoriais valem para todo o documento e por isso são resolvidas
# em um único ponto, e não marcadas à mão em cada parágrafo:
#
#   1. termo estrangeiro vai em itálico;
#   2. caixa alta não é recurso de ênfase no corpo do texto — os níveis de
#      potência aparecem em minúsculas e em itálico nos parágrafos, e a caixa alta
#      fica reservada às tabelas e às figuras, onde é rótulo de categoria.
#
# Centralizar isso garante que o mesmo termo receba sempre o mesmo tratamento; a
# alternativa, marcar ocorrência por ocorrência ao longo de mil e oitocentas
# linhas, divergiria na primeira revisão.

_ESTRANGEIRISMOS = [
    "anti-short-cycling", "short-cycling", "rule-based", "action repeat",
    "replay buffer", "winner-take-all", "intra-tile", "hiper-tile",
    "baselines", "baseline", "setpoints", "setpoint", "benchmarks", "benchmark",
    "shielding", "lumped", "inverter", "deadband", "defaults", "argmax",
    "backups", "bins", "bin", "batch", "ε-greedy", "fancoil",
    "bestest_air", "zero-shot",
]

# Níveis de potência: o texto do código usa caixa alta, o documento imprime
# minúscula em itálico.
_NIVEIS = {"OFF": "off", "LOW": "low", "MEDIUM": "medium", "HIGH": "high"}

_SUBST = {}
for _t in _ESTRANGEIRISMOS:
    _SUBST[_t] = _t
    _SUBST[_t[0].upper() + _t[1:]] = _t[0].upper() + _t[1:]
_SUBST.update(_NIVEIS)

# Alternativas mais longas primeiro, para que "anti-short-cycling" case inteiro em
# vez de casar apenas o sufixo. O lookaround recusa hífen e alfanumérico nas
# bordas, o que evita capturar "Baselines" dentro de "Stable-Baselines3".
_RX_TERMOS = re.compile(
    r"(?<![\w\-])(" + "|".join(re.escape(k) for k in
                               sorted(_SUBST, key=len, reverse=True)) + r")(?![\w\-])")

# Marcação inline: **negrito** e *itálico*.
_RX_MARCA = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*", re.DOTALL)


def _aplica_termos(texto, bold, italic):
    """Quebra `texto` nos termos do glossário, alternando o itálico."""
    saida, pos = [], 0
    for m in _RX_TERMOS.finditer(texto):
        if m.start() > pos:
            saida.append((texto[pos:m.start()], bold, italic))
        # Itálico dentro de trecho já italizado volta ao romano, que é a
        # convenção para destaque aninhado.
        saida.append((_SUBST[m.group(1)], bold, not italic))
        pos = m.end()
    if pos < len(texto):
        saida.append((texto[pos:], bold, italic))
    return saida


def _segmentos(texto, *, bold=False, italic=False, auto=True):
    """Converte texto com marcação em uma lista de (trecho, negrito, itálico)."""
    brutos, pos = [], 0
    for m in _RX_MARCA.finditer(texto):
        if m.start() > pos:
            brutos.append((texto[pos:m.start()], bold, italic))
        if m.group(1) is not None:
            brutos.append((m.group(1), True, italic))
        else:
            brutos.append((m.group(2), bold, not italic))
        pos = m.end()
    if pos < len(texto):
        brutos.append((texto[pos:], bold, italic))
    if not auto:
        return brutos
    return [s for t, b, i in brutos for s in _aplica_termos(t, b, i)]



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


def _escreve(par, texto, *, size, bold=False, italic=False, auto=True):
    for trecho, b, i in _segmentos(texto, bold=bold, italic=italic, auto=auto):
        _runs_com_indice(par, trecho, size=size, bold=b, italic=i)
    return par


def p(texto="", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, bold=False,
      italic=False, space_before=0, space_after=6, indent=None, first=None,
      auto=True):
    par = doc.add_paragraph()
    par.alignment = align
    pf = par.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if indent is not None:
        pf.left_indent = Cm(indent)
        pf.right_indent = Cm(indent)
    if first is not None:
        pf.first_line_indent = Cm(first)
    _escreve(par, texto, size=size, bold=bold, italic=italic, auto=auto)
    return par


def rico(partes, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, first=None,
         space_before=0, space_after=6):
    """Parágrafo com trechos em (texto, negrito, itálico)."""
    par = doc.add_paragraph()
    par.alignment = align
    par.paragraph_format.space_before = Pt(space_before)
    par.paragraph_format.space_after = Pt(space_after)
    if first is not None:
        par.paragraph_format.first_line_indent = Cm(first)
    for texto, b, i in partes:
        _escreve(par, texto, size=size, bold=bool(b), italic=bool(i))
    return par


def h(texto, nivel=1):
    tam = {1: 13, 2: 12, 3: 12}[nivel]
    return p(texto, align=WD_ALIGN_PARAGRAPH.LEFT, size=tam, bold=True,
             space_before=12 if nivel == 1 else 10, space_after=6)


def legenda(texto, *, acima=False):
    return p(texto, align=WD_ALIGN_PARAGRAPH.CENTER, size=10,
             space_before=8 if acima else 4, space_after=8 if not acima else 4)


def figura(caminho, largura_cm=14.4):
    """Insere a figura centralizada. Ver make_figures.py, mesmas fontes das tabelas."""
    doc.add_picture(caminho, width=Cm(largura_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(8)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)



# ------------------------------------------------------------------ equações
#
# As equações são emitidas como OMML (Office Math Markup Language), o formato
# nativo de equação do Word, e não como texto em itálico. A diferença é
# substantiva: subscritos, expoentes e frações passam a ser estruturais, o
# editor as reconhece como objetos matemáticos e a tipografia (espaçamento,
# itálico de variáveis, altura de frações) segue a convenção matemática.

_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _m(tag: str):
    return OxmlElement(f"m:{tag}")


def _run(texto: str, italico: bool = True):
    """Run matemático. Variáveis em itálico; operadores e números, romanos."""
    r = _m("r")
    if not italico:
        pr = _m("rPr")
        sty = _m("sty")
        sty.set(qn("m:val"), "p")          # 'plain': sem itálico
        pr.append(sty)
        r.append(pr)
    t = _m("t")
    t.set(qn("xml:space"), "preserve")
    t.text = texto
    r.append(t)
    return r


def _preenche(alvo, parte, italico=True):
    """Anexa a `alvo` uma string, um elemento OMML ou uma lista de ambos."""
    for x in (parte if isinstance(parte, (list, tuple)) else [parte]):
        alvo.append(_run(x, italico=italico) if isinstance(x, str) else x)
    return alvo


def _sub(base, indice):
    """Subscrito: base com índice inferior (índice em romano, por convenção)."""
    n = _m("sSub")
    e, sb = _m("e"), _m("sub")
    _preenche(e, base)
    _preenche(sb, indice, italico=False)
    n.append(e); n.append(sb)
    return n


def _sup(base, expoente):
    """Sobrescrito: base elevada a expoente."""
    n = _m("sSup")
    e, sp = _m("e"), _m("sup")
    _preenche(e, base)
    _preenche(sp, expoente, italico=False)
    n.append(e); n.append(sp)
    return n


def _frac(num, den):
    """Fração empilhada."""
    n = _m("f")
    nu, de = _m("num"), _m("den")
    _preenche(nu, num)
    _preenche(de, den)
    n.append(nu); n.append(de)
    return n


_EQ_EMITIDAS = []


def equacao(partes, rotulo: Optional[str] = None):
    """
    Equação centralizada, numerada à direita.

    `partes` é uma lista de elementos OMML ou strings; strings viram runs em
    itálico, a convenção para variáveis. O número vem de `_EQS`, e a ordem de
    emissão é conferida contra a ordem declarada: numeração e texto não podem
    divergir sem que a geração falhe.
    """
    numero = None
    if rotulo is not None:
        numero = E(rotulo)
        _EQ_EMITIDAS.append(rotulo)
        esperado = _EQS[len(_EQ_EMITIDAS) - 1]
        if rotulo != esperado:
            raise AssertionError(
                f"equação '{rotulo}' emitida na posição de '{esperado}': "
                "a ordem de `_EQS` não corresponde à ordem do texto")

    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(8)
    par.paragraph_format.space_after = Pt(8)

    math = _m("oMath")
    for x in partes:
        math.append(_run(x) if isinstance(x, str) else x)
    par._p.append(math)

    if numero is not None:
        r = par.add_run(f"\t\t({numero})")
        r.font.size = Pt(12)
    return par


def _op(texto):
    """Operador ou texto romano dentro da equação."""
    return _run(texto, italico=False)


def _sombrear(celula, cor="D9D9D9"):
    tc = celula._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), cor)
    tc.append(shd)


def tabela(cabecalho, linhas, *, negrito_linhas=(), size=9.5, larguras=None):
    t = doc.add_table(rows=1, cols=len(cabecalho))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for j, texto in enumerate(cabecalho):
        hdr[j].text = ""
        par = hdr[j].paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(2)
        r = par.add_run(str(texto))
        r.bold = True
        r.font.size = Pt(size)
        _sombrear(hdr[j])
    for i, linha in enumerate(linhas):
        cells = t.add_row().cells
        for j, val in enumerate(linha):
            cells[j].text = ""
            par = cells[j].paragraphs[0]
            par.alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                             else WD_ALIGN_PARAGRAPH.CENTER)
            par.paragraph_format.space_after = Pt(2)
            r = par.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = i in negrito_linhas
    if larguras:
        for row in t.rows:
            for j, w in enumerate(larguras):
                row.cells[j].width = Cm(w)
    return t


# =========================================================== TÍTULO E AUTORIA
p("Quando o Aprendizado por Reforço Profundo Melhora o Controle de HVAC? "
  "Uma Auditoria de Reprodutibilidade com Baselines Competentes",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, space_after=14)

p("Renan Saraiva dos Santos¹", align=WD_ALIGN_PARAGRAPH.CENTER, size=12,
  space_after=2)
p("¹Instituto Federal da Paraíba, Campus Cajazeiras, Cajazeiras, PB, Brasil",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=2)
p("dossaraiva@gmail.com", align=WD_ALIGN_PARAGRAPH.CENTER, size=11,
  space_after=14)

# ==================================================================== ABSTRACT
p("Abstract.", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, bold=True,
  space_after=0, indent=1.0)
p("Deep reinforcement learning (DRL) is widely reported to outperform "
  "conventional controllers in HVAC applications. Those gains, however, are "
  "measured against reference controllers whose configuration is rarely "
  "reported with the care devoted to the agent's architecture: of six works "
  "surveyed here, only two include a tuned classical feedback controller, and "
  "where one is present the margin shrinks. This paper asks how much of the "
  "reported advantage survives a competently configured baseline. A DQN "
  "classroom HVAC controller is implemented from its specification and "
  "audited against "
  "a thermostat with hysteresis and a grid-tuned PI controller given the same "
  "actuation authority and decision horizon. The reproduction is faithful in "
  "physics and comfort but not in energy or switching, which bounds what may be "
  "claimed and is stated per finding. The claimed advantage does not survive: "
  "the PI controller, with two constants, matches the DQN on binary comfort "
  "(86.5% both) at a lower deviation from setpoint (0.72 °C vs 0.77–0.89 °C) "
  "and lower cost (R$ 9.24 vs 9.59–10.44/day); decomposing the claimed +32 pp "
  "attributes +23.7 pp to adding hysteresis to the baseline, +8.8 pp to "
  "classical control and +0.0 pp to learning. The mechanism behind the agents' "
  "extra energy is identified: all three reward profiles discard the "
  "highest-COP power level entirely, because a deterministic argmax policy "
  "turns a ~1% value margin into 0% usage, while a SAC agent trained on the "
  "identical reward uses that level as the PI does, a design consequence for "
  "any actuator whose efficiency is non-monotonic in load. Three further "
  "findings are methodological: an ablation over 8 variants × 3 seeds shows "
  "three of five reward terms to be inert and the proposed formulation not to "
  "beat a plain quadratic (Cliff's δ = −0.11); the anti-short-cycling penalty "
  "fails its own objective (71–83% dwell violations against a 36 min "
  "requirement) whereas a hard constraint reduces them to 6.4% at zero comfort "
  "cost; and reward constants calibrated under a 300k-step budget reverse sign "
  "at the protocol's 550k steps, so calibrating under a reduced budget and "
  "extrapolating is invalid. Conversely, tabular Q-learning collapses (42.1% "
  "comfort) because 79.8% of transitions are intra-tile, the same slow "
  "dynamics, with a 30 h time constant exceeding the 24 h episode, that make "
  "this plant unusually forgiving. The conclusion is therefore bounded: under "
  "this formulation and in this environment no DRL advantage is observed, and a "
  "self-authored simulator cannot, in principle, exhibit the model-mismatch "
  "regime where RL is expected to hold one. What generalizes is the "
  "requirement: claims of learned advantage need classical baselines tuned and "
  "reported with the same effort as the agent.",
  align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, italic=True, indent=1.0,
  space_after=10, auto=False)

p("Resumo.", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, bold=True,
  space_after=0, indent=1.0)
p("O aprendizado por reforço profundo (DRL) é amplamente reportado como "
  "superior a controladores convencionais em aplicações de HVAC. Esses ganhos, "
  "porém, são medidos contra controladores de referência cuja configuração "
  "raramente é reportada com o cuidado dedicado à arquitetura do agente: de "
  "seis trabalhos aqui levantados, apenas dois incluem um controlador clássico "
  "com realimentação sintonizado, e onde ele está presente a margem encolhe. "
  "Este artigo pergunta quanto da vantagem reportada sobrevive a um baseline "
  "competentemente configurado. Um controlador DQN para climatização de salas "
  "de aula é implementado a partir de sua especificação e auditado "
  "contra um termostato com histerese e um controlador PI sintonizado por busca "
  "em grade, ambos com a mesma autoridade de atuação e o mesmo horizonte de "
  "decisão. A reprodução mostrou-se fiel na física e no conforto, mas não em "
  "energia e comutações, o que limita o que pode ser afirmado e é declarado "
  "achado por achado. A vantagem reivindicada não sobrevive: o controlador PI, "
  "com duas constantes, iguala o DQN em conforto binário (86,5 % para ambos) "
  "com menor desvio do setpoint (0,72 °C contra 0,77–0,89 °C) e menor custo "
  "(R$ 9,24 contra 9,59–10,44/dia); a decomposição dos +32 pp reivindicados "
  "atribui +23,7 pp à inclusão de histerese no baseline, +8,8 pp ao controle "
  "clássico e +0,0 pp ao aprendizado. Identificou-se o mecanismo do consumo "
  "excedente dos agentes: os três perfis de recompensa descartam integralmente "
  "o nível de potência de maior COP, porque uma política determinística por "
  "maximização converte uma margem de valor de cerca de 1 % em uso de 0 %, "
  "enquanto um agente SAC treinado com a recompensa idêntica utiliza esse nível "
  "como o PI o faz, consequência de projeto para qualquer atuador cuja "
  "eficiência não é monotônica na carga. Três achados adicionais são "
  "metodológicos: uma ablação sobre 8 variantes × 3 sementes mostra três dos "
  "cinco termos da recompensa inertes e a formulação proposta incapaz de "
  "superar uma quadrática simples (δ de Cliff = −0,11); a penalidade "
  "anti-short-cycling falha em seu próprio objetivo (71–83 % de violações "
  "contra requisito de 36 min), ao passo que uma restrição dura as reduz a "
  "6,4 % sem custo de conforto; e parâmetros de recompensa calibrados sob "
  "orçamento de 300 mil passos invertem de sinal sob os 550 mil do protocolo, "
  "de modo que calibrar sob orçamento reduzido e extrapolar é inválido. Em "
  "contrapartida, o Q-Learning tabular colapsa (42,1 % de conforto) porque "
  "79,8 % das transições são intra-tile, efeito da mesma dinâmica lenta, com "
  "constante de tempo de 30 h superior ao episódio de 24 h, que torna esta "
  "planta pouco exigente. A conclusão é, portanto, delimitada: nesta "
  "formulação e neste ambiente não se observa vantagem do DRL, e um simulador "
  "de autoria própria não pode, em princípio, exibir o regime de descasamento "
  "de modelo em que se espera que ela exista. O que generaliza é a exigência: "
  "afirmar vantagem do aprendizado requer baselines clássicos sintonizados e "
  "reportados com o mesmo esforço dedicado ao agente.",
  align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, italic=True, indent=1.0,
  space_after=12)

# ================================================================= 1. INTRO
h("1. Introdução")

p("Sistemas de aquecimento, ventilação e ar-condicionado (HVAC) respondem por "
  "parcela dominante do consumo energético de edificações, e sua operação "
  "determina diretamente o conforto e a produtividade dos ocupantes [Xu et al. "
  "2025]. A literatura recente é abundante em propostas de controle por "
  "aprendizado por reforço profundo (DRL) para esse domínio, quase sempre "
  "reportando ganhos expressivos sobre estratégias convencionais [Al Sayed et "
  "al. 2024].", first=1.25)

p("Um padrão metodológico recorrente nessa literatura, porém, merece escrutínio: "
  "a magnitude do ganho reportado depende criticamente de como o controlador de "
  "referência foi configurado. Um termostato liga-desliga sem histerese, com "
  "zona morta nula, é um adversário artificialmente incapaz, e comparações "
  "contra ele produzem vantagens que não se sustentam frente a um controlador "
  "clássico competentemente sintonizado.", first=1.25)

rico([("Este trabalho realiza uma ", 0, 0),
      ("auditoria de reprodutibilidade", 0, 0),
      (" de um controlador DRL para climatização de salas de aula, tomado como "
       "estudo de caso desse padrão.", 0, 0)],
     first=1.25)

p("O objetivo geral consiste em analisar em que medida o desempenho reportado "
  "para controladores HVAC baseados em aprendizado por reforço decorre do "
  "aprendizado, e não da configuração do controlador de referência adotado como "
  "comparação. Como objetivos específicos, delimitam-se:", space_before=4,
  first=1.25)

for item in [
    "(i) avaliar em que medida a vantagem reivindicada sobre o controlador de "
    "referência é atribuível ao aprendizado ou à configuração do adversário;",
    "(ii) quantificar, por ablação controlada sobre oito variantes e três "
    "sementes, a contribuição de cada termo da função de recompensa proposta, "
    "comparando-a a formulações convencionais da literatura;",
    "(iii) verificar a efetividade da penalidade anti-short-cycling declarada "
    "como contribuição, contrastando-a com uma restrição estrutural equivalente;",
    "(iv) fundamentar teórica e empiricamente a adequação de métodos tabulares "
    "a este domínio, por meio da fração de transições intra-tile;",
    "(v) examinar se extensões do problema, rastreamento de precisão, "
    "antecipação tarifária e demanda contratada, constituem regimes favoráveis "
    "ao aprendizado;",
    "(vi) determinar o efeito do estreitamento da faixa alvo sobre o desempenho "
    "relativo, com ambos os controladores repreparados para cada especificação, "
    "e verificar a influência do orçamento de treinamento sobre esse efeito;",
    "(vii) identificar os mecanismos que explicam as diferenças de consumo "
    "energético observadas entre as políticas.",
]:
    p(item, first=0.75, space_after=3)

p("O resultado agregado é negativo quanto à necessidade de RL nesta formulação "
  "do problema, e positivo quanto ao rigor exigido para afirmá-la. Resultados "
  "negativos bem controlados são escassos nesta literatura e, argumenta-se, "
  "necessários para calibrá-la.", space_before=4, first=1.25)

# ====================================================== 2. TRABALHOS RELACIONADOS
h("2. Fundamentação teórica e trabalhos relacionados")

h("2.1. Aprendizado por reforço", 2)

p("O aprendizado por reforço trata do problema de decidir sequencialmente sob "
  "incerteza. Um agente observa o estado do ambiente, escolhe uma ação, recebe "
  "uma recompensa numérica e observa o novo estado; repetindo esse ciclo, "
  "ajusta seu comportamento de modo a maximizar a recompensa acumulada ao longo "
  f"do tempo [Sutton e Barto 2018]. A Figura {Fg('ciclo')} ilustra a "
  "interação.", first=1.25)

figura(salvar_fig(F.fig_ciclo_rl(), "fig0_ciclo_rl.png"), largura_cm=12.6)
legenda(f"Figura {Fg('ciclo')}. Ciclo de interação entre agente e ambiente. A "
        "cada intervalo de decisão o agente aciona um nível de potência e "
        "observa a temperatura resultante e a recompensa correspondente.")

p("Formalmente, o problema é modelado como um processo de decisão markoviano, "
  "definido pela quádrupla (S, A, P, R): o conjunto de estados S, o conjunto de "
  "ações A, a função de transição P, que descreve a probabilidade de alcançar "
  "cada estado seguinte, e a função de recompensa R [Bellman 1957]. A "
  "propriedade de Markov exige que o estado observado contenha toda a "
  "informação relevante para decidir, de modo que o passado possa ser "
  "descartado. O comportamento do agente é descrito pela política π(a | s), que "
  "associa a cada estado uma escolha de ação.", space_before=8, first=1.25)

p("O objetivo é maximizar o retorno, isto é, a soma das recompensas futuras "
  "descontadas por um fator γ ∈ [0, 1), que expressa quanto o agente valoriza "
  "recompensas distantes frente às imediatas. A função de valor de ação "
  "Q(s, a) representa o retorno esperado ao executar a ação a no estado s e "
  "seguir a política a partir daí, e satisfaz a equação de Bellman:", first=1.25)

equacao([_op("Q"), _op("(s, a) = 𝔼["), "r", _op(" + "), "γ",
         _op(" max"), _sub(_op(""), "a′"), _op(" Q(s′, a′)]")], rotulo="bellman")

p("O algoritmo Q-Learning aproxima essa solução por iteração sobre amostras de "
  "interação, sem exigir conhecimento prévio da função de transição [Watkins e "
  "Dayan 1992]. Sua formulação tabular armazena um valor para cada par "
  "estado-ação, o que restringe sua aplicação a espaços pequenos e discretos. "
  "Para estados contínuos, a alternativa é aproximar Q por uma rede neural, "
  "abordagem que caracteriza o Deep Q-Network [Mnih et al. 2015]; o treinamento "
  "estável depende de dois mecanismos, o repositório de transições passadas, "
  "que descorrelaciona as amostras, e uma rede-alvo atualizada com atraso, que "
  "estabiliza o alvo de regressão.", first=1.25)

p("Métodos dessa família operam sobre espaços de ação discretos, pois exigem "
  "maximizar Q sobre todas as ações a cada passo. Para ações contínuas "
  "empregam-se arquiteturas ator-crítico, nas quais uma rede propõe a ação e "
  "outra estima seu valor. Duas variantes são utilizadas neste trabalho: o Soft "
  "Actor-Critic, que otimiza simultaneamente retorno e entropia da política, "
  "favorecendo a exploração [Haarnoja et al. 2018], e o Twin Delayed DDPG, que "
  "mitiga a superestimação de valor por meio de dois críticos e da atualização "
  "atrasada do ator [Fujimoto et al. 2018].", first=1.25)

rico([("Uma distinção importa para a interpretação dos resultados deste "
       "trabalho: a política aprendida por DQN é ", 0, 0),
      ("determinística e obtida por maximização", 0, 0),
      (", isto é, em cada estado escolhe-se a ação de maior valor estimado. Uma "
       "ação cujo valor seja apenas ligeiramente inferior ao máximo, ainda que "
       "quase equivalente, não é executada nunca. A Seção 4.8 mostra que essa "
       "propriedade tem consequência energética direta.", 0, 0)], first=1.25)

h("2.2. Trabalhos relacionados", 2)



p("A literatura de aprendizado por reforço aplicado a HVAC é abundante em "
  "relatos de ganho, mas heterogênea quanto ao adversário contra o qual esse "
  "ganho é medido. Como a magnitude reportada depende criticamente dessa "
  "escolha, organizamos a revisão pela qualidade do controlador de "
  "referência, e não por algoritmo ou aplicação.", first=1.25)

legenda(f"Tabela {T('relacionados')}. Trabalhos de RL para HVAC, classificados pelo controlador de "
        "referência adotado.", acima=True)
tabela(["Trabalho", "Ambiente", "Referência de comparação", "Ganho reportado"],
       [["Wei et al. (2017)", "EnergyPlus", "rule-based", "20–70 % de custo"],
        ["Yuan et al. (2020)", "TRNSYS", "rule-based e PID", "7,7 % / 4,7 % de energia"],
        ["Boutahri e Tilioua (2025)", "BOPTEST + prédio real", "PI e rule-based",
         "26,3 % / 8,8 % de energia"],
        ["Dai et al. (2025)", "EnergyPlus", "agenda de setpoint (RBC)", "arcabouço"],
        ["Xu et al. (2025)", "modelo próprio", "DDQN (outro agente de RL)",
         "8,8× em velocidade"],
        ["Zha et al. (2021)", "—", "trabalho de método", "—"]],
       larguras=[3.6, 3.4, 4.4, 3.6])

p("Dois padrões emergem. Primeiro, a maioria compara contra controle baseado em "
  "regras ou contra outro agente de RL; apenas dois trabalhos incluem um "
  "controlador clássico com realimentação sintonizado. Segundo, quando esse "
  "controlador está presente, a margem encolhe: Yuan et al. reportam 7,7 % de "
  "economia sobre o rule-based e apenas 4,7 % sobre o PID, e observam que em "
  "configuração multi-zona o agente só supera as referências após dois anos de "
  "exploração somados a dois anos de estabilização, atingindo o melhor "
  "desempenho no sétimo ano, um custo de amostra que é, por si, um fator de "
  "decisão.", space_before=8, first=1.25)

p("Al Sayed et al. [2024] revisam o campo e catalogam ganhos de 27–30 % sobre "
  "controladores baseados em regras, até 39,6 % sobre controladores padrão e "
  "23 % de redução de pico sobre sistemas manuais, todos, portanto, sobre "
  "adversários da mesma família. Quanto ao controle preditivo, a revisão o "
  "descreve como referência do campo e registra que o melhor caso do RL na "
  "literatura *iguala* o desempenho do MPC, sem superá-lo; superar o MPC aparece "
  "como objetivo futuro, condicionado à redução do custo computacional.",
  first=1.25)

rico([("Uma condicional da mesma revisão merece destaque, porque delimita o "
       "escopo de validade de toda a literatura: a vantagem do RL sobre métodos "
       "tradicionais e sobre MPC é afirmada ", 0, 0),
      ("\"quando o ambiente de treinamento em simulação replica com precisão "
       "cenários do mundo real\"", 0, 1),
      (". A vantagem é, portanto, condicional à fidelidade do simulador, ponto "
       "ao qual retornamos na Seção 5.1.", 0, 0)], first=1.25)

p("Wei et al. [2017] são o primeiro trabalho a aplicar RL profundo ao domínio e "
  "formulam a dificuldade central do caso multi-zona: com z zonas e m níveis de "
  "atuação, o espaço de ação tem m^z elementos, o que degrada o treinamento. Os "
  "autores propõem uma heurística de controle multinível para contorná-la. "
  "Zha et al. [2021] identificam por que métodos tabulares falham em espaços "
  "contínuos com variáveis de dinâmica lenta, a transição permanece no mesmo "
  "hiper-tile e o valor não se propaga, diagnóstico utilizado na "
  "Seção 4.5. Xu et al. [2025] atacam o custo de treinamento por destilação de "
  "conhecimento especialista e introduzem shielding em tempo de execução, ideia "
  "aplicável ao achado deste trabalho sobre anti-short-cycling.", first=1.25)

p("Do ponto de vista metodológico, Henderson et al. [2018] e Agarwal et al. "
  "[2021] documentam a fragilidade de comparações em RL profundo com poucas "
  "sementes, motivando o uso de tamanho de efeito não paramétrico em vez de "
  "valores-p isolados, prática aqui adotada.", first=1.25)

p("Este trabalho mede o degrau que a literatura majoritariamente omite: o "
  "desempenho de um controlador proporcional-integral competentemente "
  "sintonizado, situado entre o rule-based (onde o RL vence) e o MPC (que o RL "
  "aspira a igualar).", first=1.25)

h("3. Materiais e Métodos")

h("3.1. Ambiente de simulação", 2)

p("O ambiente modela uma sala de aula com capacidade para 45 ocupantes, "
  "implementado com a interface Gymnasium [Towers et al. 2024]. A dinâmica "
  "térmica é agregada (lumped), tratando a sala como nó único:", first=1.25)

equacao([_sub("T", "t+1"), _op(" = "), _sub("T", "t"), _op(" + "),
         _frac([_op("Δ"), "t"], _sub("C", "th")),
         _op(" ["), "N", _op("·"), _sub("q", "p"), _op(" + "), "K",
         _op(" ("), _sub("T", "ext"), _op(" − "), _sub("T", "t"), _op(") − "),
         _sub("Q", "ac"), _op("(a)] + "), "ε"], rotulo="fisica")

p("em que C_th é a capacidade térmica (15,0 na sala de treino), N o número de "
  "ocupantes, q_p = 0,3 u o ganho térmico por pessoa, K = 0,5 o coeficiente de "
  "transferência com o exterior, Q_ac a potência de resfriamento e ε ruído "
  "gaussiano de processo (σ = 0,01). O passo é Δt = 0,1 h (6 min) e o episódio "
  "cobre 24 h (240 passos). A temperatura externa segue senoide diária com pico "
  "às 14 h, base 28 °C e amplitude 8 °C.", first=1.25)

rico([(f"A Tabela {T('fisica')} do manuscrito auditado é ", 0, 0), ("derivada", 0, 0),
      (", e não tabelada, a partir de dois parâmetros de catálogo: capacidade "
       "de 30.000 BTU/h e coeficiente de performance (COP) por nível. Essa "
       "escolha torna o modelo auditável e permite trocar de equipamento sem "
       "reescrever a tabela. A derivação reproduz os valores publicados com "
       "erro inferior a 0,5%.", 0, 0)], first=1.25)

legenda(f"Tabela {T('fisica')}. Modelo do equipamento, derivado de capacidade nominal e COP.",
        acima=True)
tabela(["Nível", "Fração de carga", "COP", "Resfriamento (u)",
        "Elétrica (kW), derivada", "Publicada"],
       [["OFF", "0,00", "—", "0", "0,000", "—"],
        ["LOW", "0,25", "3,45", "10", "0,637", "0,64"],
        ["MEDIUM", "0,55", "3,59", "22", "1,347", "1,35"],
        ["HIGH", "1,00", "3,00", "40", "2,931", "2,93"]],
       larguras=[2.0, 2.6, 1.6, 2.6, 3.6, 2.2])

p("O pico de COP em carga parcial (MEDIUM) reproduz comportamento de "
  "equipamento inverter e é o que cria o compromisso relevante: operar em "
  "MEDIUM é energeticamente barato, o que favorece o termostato em custo mesmo "
  "quando ele perde em conforto.", space_before=8, first=1.25)

p("O modelo opera em unidades adimensionais, o que impede julgar a que sala "
  "física o ambiente corresponde, limitação apontada em avaliação por pares do "
  "manuscrito auditado. A capacidade do equipamento ancora a escala, pois "
  "40 unidades equivalem a 8,79 kW térmicos, e dessa âncora derivam-se as "
  "grandezas físicas correspondentes.", first=1.25)

legenda(f"Tabela {T('correspondencia')}. Correspondência entre as unidades de "
        "simulação e grandezas físicas.", acima=True)
tabela(["Grandeza", "Valor em simulação", "Equivalente físico"],
       [["Unidade de potência térmica", "1 u",
         f"{v(FIS['kw_por_unidade'], 4)} kW"],
        ["Capacidade térmica C_th", "15 u·h/°C",
         f"{v(FIS['capacidade_kwh_por_c'], 2)} kWh/°C "
         f"({FIS['capacidade_kj_por_c']:,.0f} kJ/°C)".replace(",", ".")],
        ["Volume de ar equivalente", "—",
         f"{FIS['volume_ar_equivalente_m3']:,.0f} m³".replace(",", ".")],
        ["Condutância K", "0,5 u/°C",
         f"{v(FIS['condutancia_kw_por_c'], 3)} kW/°C"],
        ["Constante de tempo τ = C/K", "30 u·h/u",
         f"{v(FIS['constante_tempo_h'], 1)} h"],
        ["Ganho térmico por ocupante", "0,3 u",
         f"{FIS['ganho_por_pessoa_w']:.0f} W"],
        ["Carga com lotação máxima", "13,5 u",
         f"{v(FIS['carga_ocupacao_plena_kw'], 2)} kW"]],
       larguras=[5.4, 3.4, 5.2])

rico([("Duas leituras decorrem da tabela. A primeira é favorável: o ganho "
       "térmico por ocupante, 66 W sensíveis, situa-se na faixa usual para "
       "atividade sedentária, e a capacidade do equipamento supera em quase três "
       "vezes a carga de ocupação plena, o que é dimensionamento plausível. A "
       "segunda é desfavorável e relevante: ", 0, 0),
      ("a constante de tempo do ambiente é de 30 h, superior à própria duração "
       "do episódio de 24 h", 0, 0),
      (". A massa térmica equivale a cerca de 10.000 m³ de ar, cerca de setenta "
       "vezes o volume de ar de uma sala de aula convencional, valor que só se "
       "justificaria por acoplamento muito forte com a estrutura da edificação.",
       0, 0)], space_before=8, first=1.25)

p("A consequência é estrutural e atravessa vários resultados deste trabalho. Um "
  "ambiente cuja constante de tempo excede o horizonte do episódio amortece "
  "fortemente as perturbações, o que torna o problema de controle mais fácil do "
  "que seria numa sala real e reduz a superfície disponível para diferenciação "
  "entre controladores. Essa mesma lentidão é a causa do diagnóstico da "
  f"Seção 4.5: a variação típica por passo, de 0,090 °C, é onze vezes menor que "
  "o intervalo de discretização, e é por isso que as transições permanecem "
  "intra-tile. Inércia elevada e falha do método tabular não são achados "
  "independentes, são a mesma propriedade do ambiente observada por dois "
  "instrumentos distintos.", first=1.25)

h("3.2. Formulação como processo de decisão markoviano", 2)

p(f"O estado observado é normalizado conforme a Equação {E('observacao')}, com quatro "
  "componentes: temperatura, ocupação e codificação circular da hora do dia.",
  first=1.25)

equacao(["s", _op(" = ["), _op("clip"), _op("("), _frac([ "T", _op(" − 15")], "20"),
         _op(", 0, 1), "), _frac("N", _sub("N", "max")), _op(", sen"),
         _op("("), _frac([_op("2π"), "h"], "24"), _op("), cos"),
         _op("("), _frac([_op("2π"), "h"], "24"), _op(")]")], rotulo="observacao")

p("O espaço de ação é discreto com quatro níveis (OFF, LOW, MEDIUM, HIGH) para "
  "os agentes DQN, e contínuo em [0,1] para SAC. A recompensa agrega cinco "
  "termos:", first=1.25)

equacao(["R", _op(" = "), _sub("R", "conf"), _op(" + "), _sub("R", "energ"),
         _op(" + "), _sub("R", "troca"), _op(" + "), _sub("R", "frio"),
         _op(" + "), _sub("R", "ciclo")], rotulo="recompensa")

p("O termo de conforto adota a topologia denominada Platô Quadrático com "
  f"gradiente interno (Equação {E('conforto_dentro')}), em que B é o bônus base, B_c o gradiente "
  "interno, k a curvatura fora da faixa e [T_min, T_max] = [22, 26] °C:",
  first=1.25)

equacao([_sub("R", "conf"), _op(" = "), "B", _op(" + "), _sub("B", "c"),
         _op(" (1 − "), _frac([_op("|"), "T", _op(" − 24|")], "2"), _op("),"),
         _op("  se 22 ≤ "), "T", _op(" ≤ 26")], rotulo="conforto_dentro")
equacao([_sub("R", "conf"), _op(" = "), "B", _op(" − "), "k",
         _op(" "), _sup([_op("("), "T", _op(" − "), _sub("T", "lim"), _op(")")], "2"),
         _op(",  caso contrário")], rotulo="conforto_fora")

p(f"A penalidade anti-short-cycling (Equação {E('ciclo')}) pune comutações antes do tempo "
  "mínimo de permanência d_min = 36 min (6 passos), proporcionalmente à "
  "antecipação:", first=1.25)

equacao([_sub("R", "ciclo"), _op(" = "), "ρ", _op(" "),
         _frac([_sub("d", "min"), _op(" − "), "d"], _sub("d", "min")),
         _op(",  se houve troca e "), "d", _op(" < "), _sub("d", "min")], rotulo="ciclo")

rico([("Parâmetros inferidos. ", 1, 0),
      ("O manuscrito auditado especifica numericamente B_c, a penalidade de "
       "energia e a de troca, mas não publica B, k, ρ nem a penalidade de "
       "frio. Os valores foram inferidos da figura de recompensa do manuscrito "
       "e estão explicitamente marcados no código: B = 10,0 (bordas do platô em "
       "≈ +10); k = 0,6 (a curva atinge ≈ −11 em 32 °C); ρ = −5,0 (não "
       "observável, escolhido na ordem de grandeza do conforto); penalidade de "
       "frio = −2,0 (a conclusão do manuscrito alerta que valor severo induziu "
       "comportamento de evitação). Divergências numéricas devem ser atribuídas "
       "a esses quatro parâmetros antes de se suspeitar da implementação.",
       0, 0)], first=1.25, space_before=4)

h("3.3. Controladores de referência", 2)

p("Foram implementados três controladores de referência com a mesma autoridade "
  "de atuação e o mesmo horizonte de decisão (action repeat 2) dos agentes "
  "aprendidos, de modo que a comparação isole a política e não o intervalo de "
  "amostragem:", first=1.25)

for item in [
    "Termostato com zona morta nula, o baseline do manuscrito auditado. Liga "
    "em MEDIUM acima de 26 °C e em HIGH acima de 28 °C, desligando ao retornar "
    "à faixa.",
    "Termostato com histerese de 1 °C, a mesma lógica, com zona morta, "
    "isolando o efeito da configuração do baseline.",
    "Controlador PI com anti-windup, sintonizado por busca em grade sobre a "
    "matriz de cenários (K_p = 1,3; K_i = 0,2), com saída discretizada no mesmo "
    "espaço de ação dos agentes [Åström e Hägglund 2006].",
]:
    p("• " + item, first=0.5, space_after=3)

h("3.4. Protocolo experimental e métricas", 2)

p("A avaliação usa matriz fatorial 3×3 cruzando condição térmica inicial "
  "(17 °C, 24 °C, 30 °C) com nível de ocupação (poucas, médias, muitas), "
  "totalizando nove cenários C1–C9. Aplica-se filtro de controlabilidade: "
  "cenários em que a política de referência trivial (sempre desligado) já "
  "satisfaz o requisito são excluídos, pois não discriminam controladores. "
  "Oito dos nove cenários são controláveis, sendo C1 (frio + poucas pessoas) o "
  "excluído, resultado idêntico ao do manuscrito auditado.", first=1.25)

p("As métricas são computadas exclusivamente sobre a janela ocupada (7h–22h): "
  "percentual de tempo na faixa larga [22, 26] °C e na faixa estreita "
  "[23, 25] °C; desvio absoluto médio do ideal |T − 24|; percentual de "
  "sobreaquecimento; consumo (kWh/dia); custo (R$/dia); e comutações por hora. "
  "Adicionalmente, reporta-se a distribuição dos tempos de permanência entre "
  "comutações, que é a métrica correta para avaliar proteção contra "
  "short-cycling.", first=1.25)

p("Os agentes foram treinados por 550.000 passos com Stable-Baselines3 [Raffin "
  "et al. 2021], taxa de aprendizado 5·10⁻⁵, batch 64 e action repeat 2. Os "
  "algoritmos cobertos são DQN [Mnih et al. 2015], SAC [Haarnoja et al. 2018] e "
  "TD3 [Fujimoto et al. 2018], além de Q-Learning tabular [Sutton e Barto 2018] "
  "para o diagnóstico da Seção 4.5.", first=1.25)

legenda(f"Tabela {T('hiperparametros')}. Hiperparâmetros efetivos do agente "
        "DQN, extraídos do objeto treinado.", acima=True)
tabela(["Parâmetro", "Valor"],
       [[_r["parametro"], str(_r["valor"])] for _, _r in HIPER.iterrows()],
       larguras=[8.4, 5.6])

p("Os valores acima foram extraídos do objeto treinado, e não do que foi "
  "declarado em código. Defaults do arcabouço que nunca aparecem "
  "explicitamente (capacidade do replay buffer, fator de desconto, intervalo "
  "de atualização da rede-alvo, recorte de gradiente) são tão necessários à "
  "reprodução quanto os "
  "parâmetros escritos à mão, e sua omissão foi apontada em avaliação por pares "
  "do manuscrito auditado. A política é um perceptron multicamadas com duas "
  "camadas ocultas de 64 unidades e 9.480 parâmetros treináveis; a exploração é "
  "ε-greedy com decaimento linear de 1,0 a 0,05 ao longo dos primeiros 10 % do "
  "treinamento.", space_before=8, first=1.25)

h("3.5. Análise estatística", 2)

p("Com três sementes por condição, o menor valor-p bicaudal alcançável em teste "
  "de permutação é 2/C(6,3) = 0,10; portanto p ≈ 0,101 é o piso, e não um "
  "resultado marginal. Reporta-se por isso o δ de Cliff [Cliff 1993], tamanho de "
  "efeito não paramétrico, junto de intervalos de confiança bootstrap "
  "percentílicos. Um valor δ = −1,00 indica separação completa entre grupos, "
  "toda semente da condição ablacionada é pior que toda semente da referência. "
  "Essa escolha segue as recomendações de Henderson et al. [2018] e Agarwal et "
  "al. [2021].", first=1.25)

# ============================================================== 4. RESULTADOS
h("3.6. Garantias de reprodutibilidade", 2)

p("Três mecanismos foram implementados para que os resultados aqui reportados "
  "sejam verificáveis e não possam divergir silenciosamente do código.", first=1.25)

for item in [
    "Contrato de observação verificado. O espaço de observação é declarado uma "
    "única vez, e dele derivam tanto os limites do espaço quanto o vetor emitido "
    "a cada passo e a lista ordenada dos canais. Essa lista é persistida junto do "
    "modelo e conferida no carregamento: duas configurações podem produzir a "
    "mesma dimensionalidade com canais distintos ou em ordem trocada, caso em que "
    "a política leria cada canal com o significado errado e produziria métricas "
    "plausíveis e inválidas. A verificação transforma essa falha silenciosa em "
    "erro de carregamento.",
    "Avaliação com múltiplas sementes. Todos os controladores são avaliados nas "
    "mesmas sementes, disjuntas das de treinamento, o que torna a comparação "
    "pareada, o ruído do ambiente é comum aos braços e cancela na diferença.",
    "Tabelas e figuras de fonte única. Cada resultado é produzido por uma única "
    "função, consumida tanto pelos cadernos de análise quanto pelo gerador deste "
    "documento. Os resultados de custo computacional baixo são recomputados a "
    "cada geração; os que exigem treinamento são lidos dos arquivos canônicos. "
    "Divergência entre uma tabela e a figura ao lado é impossível por construção.",
]:
    p("\u2022 " + item, first=0.5, space_after=4)

h("4. Resultados")

h("4.1. Fidelidade da reprodução", 2)

p("Antes de qualquer comparação, é necessário estabelecer em que dimensões a "
  "implementação reproduz o comportamento descrito, e em que dimensões não "
  "reproduz. As métricas de conforto do baseline termostático fecham dentro de "
  "aproximadamente 3 pontos percentuais dos valores publicados (54,0% contra "
  "50,9% na faixa larga; 2,10 °C contra 2,21 °C de desvio absoluto), o que "
  "sustenta que a física está correta. O consumo e as comutações, porém, ficam "
  "sensivelmente abaixo dos publicados (−31% e −61%, respectivamente), porque o "
  "manuscrito não publica os limiares da zona morta; ajustá-los aos resultados "
  "constituiria reprodução circular e não foi feito.", first=1.25)

if tem_tab('generalizacao'):
    p(f"O teste de generalização sem retreino (Tabela {T('generalizacao')}) reproduziu a assinatura "
      "qualitativa do manuscrito: a vantagem sobre o termostato mantém-se na faixa "
      "de +24,8 a +35,0 pp, contra +21,7 a +38,5 pp publicados, com a mesma "
      "ordenação, pior degradação em sala de alta inércia e menor vantagem em sala "
      "bem isolada.", first=1.25)

    legenda(f"Tabela {T('generalizacao')}. Generalização sem retreino: conforto na faixa larga (%). "
            "Δ é a vantagem do DQN sobre o termostato.", acima=True)
    tabela(["Variação da sala", "DQN (repro)", "DQN (publ.)", "Term. (repro)",
            "Term. (publ.)", "Δ repro", "Δ publ."],
           [["C_th = 10 (leve)", "87,3", "88,6", "53,9", "51,3", "+33,3", "+37,3"],
            ["C_th = 15 (treino)", "86,5", "83,2", "54,0", "51,0", "+32,5", "+32,2"],
            ["C_th = 20", "87,2", "77,6", "54,5", "49,4", "+32,8", "+28,2"],
            ["C_th = 30 (pesada)", "81,1", "67,5", "56,4", "45,8", "+24,8", "+21,7"],
            ["K = 0,3 (bem isolada)", "85,5", "80,3", "57,0", "51,9", "+28,5", "+28,4"],
            ["K = 0,8 (mal isolada)", "84,0", "85,3", "49,0", "46,7", "+35,0", "+38,5"]],
           larguras=[4.2, 2.1, 2.1, 2.2, 2.1, 1.9, 1.9])

rico([("Essa assimetria precisa ser levada a sério, e não registrada como "
       "ressalva de rodapé: ", 0, 0),
      ("a fidelidade está estabelecida em conforto, que é a métrica de menor "
       "divergência, e não em energia e comutações, que são as duas grandezas "
       "sobre as quais se apoiam parte das conclusões deste artigo", 0, 0),
      (". A afirmação de que o PI é mais barato que o DQN e a análise da Seção "
       "4.4 sobre permanência entre comutações caem, uma e outra, sobre "
       "grandezas que divergem do publicado. Duas observações delimitam o dano. "
       "Primeira, a comparação entre controladores é ", 0, 0),
      ("interna", 0, 1),
      (": PI, termostato e agentes operam no mesmo ambiente e na mesma "
       "execução, de modo que a divergência é comum aos braços e não explica a "
       "diferença entre eles. Segunda, no caso das comutações a divergência é "
       "conservadora: esta reprodução comuta menos que o manuscrito, de modo "
       "que a violação de permanência mínima documentada na Seção 4.4 seria "
       "pior, e não melhor, no trabalho original.", 0, 0)],
     first=1.25)

p("Como os achados deste artigo não dependem todos das mesmas premissas, "
  f"a Tabela {T('status')} declara, para cada um, de que ele depende e o que "
  "isso autoriza a afirmar. Essa é a leitura recomendada para quem quiser citar "
  "um resultado isolado.", first=1.25)

legenda(f"Tabela {T('status')}. Status epistêmico dos achados: do que cada um "
        "depende e o que autoriza afirmar.", acima=True)
tabela(["Achado", "Depende de", "Robustez"],
       [["79,8 % de transições intra-tile (Seção 4.5)",
         "apenas de C_th, Δt e do ganho por ocupante, todos publicados",
         "Máxima: recalculável a partir do manuscrito, independe desta "
         "reprodução"],
        ["A penalidade anti-short-cycling não protege (Seção 4.4)",
         "da política aprendida; o argumento aritmético usa só |ρ| e B + B_c, "
         "ambos conhecidos",
         "Alta e conservadora: esta reprodução comuta menos, logo no original a "
         "violação seria pior"],
        ["O nível de maior COP é descartado (Seção 4.8)",
         "da política aprendida e da tabela de COP, publicada",
         "Alta: o controle é interno, com SAC sob recompensa idêntica como "
         "contraste"],
        ["O PI iguala ou supera o DQN (Seção 4.2)",
         "da fidelidade deste ambiente ao original",
         "Inferência, não medição direta: internamente válida, mas sua "
         "transferência ao trabalho original não é demonstrada"]],
       larguras=[4.6, 5.0, 5.4])

p("Estabelecido o alcance da fidelidade, passamos à auditoria da afirmação "
  "central.", space_before=8, first=1.25)

h("4.2. A vantagem reivindicada não sobrevive a um baseline competente", 2)

p(f"A Tabela {T('comparacao')} compara os três perfis de recompensa do manuscrito contra os "
  "controladores de referência, todos avaliados sob condições idênticas.",
  first=1.25)

legenda(f"Tabela {T('comparacao')}. Comparação sob condições idênticas (matriz 3×3, janela "
        "ocupada). Melhores valores em negrito.", acima=True)
_ordem_t3 = ["PI sintonizado", "DQN Agressivo (550k)", "DQN Equilibrado (550k)",
             "DQN Passivo (550k)", "SAC Equilibrado (550k)",
             "Termostato zona morta = 1 \u00b0C", "Termostato zona morta = 0"]
_rot_t3 = {"PI sintonizado": "PI sintonizado (Kp=1,3; Ki=0,2)",
           "SAC Equilibrado (550k)": "SAC Equilibrado (550k, contínuo)",
           "Termostato zona morta = 0": "Termostato, zona morta = 0 (baseline)",
           "Termostato zona morta = 1 \u00b0C": "Termostato, zona morta = 1 \u00b0C"}
_l3 = []
for _n in _ordem_t3:
    _r = COMP[COMP["controlador"] == _n]
    if _r.empty:
        continue
    _r = _r.iloc[0]
    _l3.append([_rot_t3.get(_n, _n), v(_r["comfort_wide_pct"]), v(_r["comfort_narrow_pct"]),
                v(_r["abs_dev_from_ideal"], 2), v(_r["overheat_pct"]),
                v(_r["energy_kwh_day"], 2), v(_r["cost_brl_day"], 2),
                v(_r["changes_per_hour"], 2)])
_l3.append(["Q-Learning tabular (550k)", v(TAB["conf_larga_pct"].mean()),
            v(TAB["conf_estreita_pct"].mean()), v(TAB["desvio_ideal"].mean(), 2),
            "\u2014", v(TAB["energia_kwh"].mean(), 2), "\u2014", "\u2014"])

tabela(["Controlador", "Conf. [22,26] %", "Conf. [23,25] %", "|T\u221224| \u00b0C",
        "Sobreaq. %", "Energia kWh/dia", "Custo R$/dia", "Trocas/h"],
       _l3, negrito_linhas=(0,),
       larguras=[5.2, 2.0, 2.0, 1.7, 1.6, 1.9, 1.6, 1.4])

p("O agente SAC, de ação contínua, é incluído por corresponder à comparação "
  "entre espaços de ação discreto e contínuo do manuscrito auditado. A "
  "reprodução confirma a ordenação ali reportada, o DQN supera o SAC em "
  "conforto, mas com o controlador PI presente a leitura muda: a diferença "
  "entre espaços de ação, entre perfis de recompensa e entre algoritmos de "
  "aprendizado situa-se toda abaixo do que duas constantes clássicas entregam.",
  space_before=8, first=1.25)

rico([("O controlador PI dominou os três perfis DQN em todas as métricas de "
       "qualidade e de custo simultaneamente: mesmo conforto binário, menor "
       "desvio absoluto do ideal e menor consumo. ", 0, 0),
      ("Duas constantes sintonizadas igualam ou superam 550.000 passos de "
       "treinamento.", 0, 0)],
     space_before=8, first=1.25)

p(f"A Tabela {T('decomposicao')} decompõe a vantagem reivindicada, atribuindo cada incremento à "
  "sua causa.", first=1.25)

legenda(f"Tabela {T('decomposicao')}. Decomposição da vantagem de +32 pp reivindicada.", acima=True)
_l4 = []
for _i, _r in DEC.iterrows():
    _g = "\u2014" if _i == 0 else f"+{v(_r['ganho_pp'])} pp"
    _l4.append([_r["etapa"], v(_r["conforto_larga_pct"]), _g, _r["atribuivel_a"]])
tabela(["Etapa", "Conforto faixa larga (%)", "Ganho", "Atribuível a"], _l4,
       negrito_linhas=(len(_l4) - 1,), larguras=[6.4, 3.2, 2.2, 4.2])

figura(salvar_fig(F.fig_decomposicao(DEC), "fig1_decomposicao.png"))
legenda(f"Figura {Fg('decomposicao')}. Decomposição da vantagem reivindicada. Cada coluna acrescenta "
        "um único fator ao anterior. O último degrau, o aprendizado, é nulo.")

p("Aproximadamente 100% da vantagem reivindicada é atribuível à configuração "
  f"inadequada do adversário (Figura {Fg('decomposicao')}). A afirmação de 82,9% contra 50,9% é "
  "tecnicamente verdadeira e substantivamente enganosa: compara-se contra um "
  "controlador artificialmente incapaz.", space_before=8, first=1.25)

h("4.3. Ablação da função de recompensa", 2)

p("A tese de que a modelagem da recompensa é mais determinante que o algoritmo "
  "exige ablação controlada. Foram treinadas oito variantes, cada uma "
  "desativando exatamente um mecanismo, sobre três sementes, 24 treinamentos "
  f"de 550.000 passos. A Tabela {T('ablacao')} reporta o conforto na faixa estreita, métrica "
  "que discrimina os regimes.", first=1.25)

legenda(f"Tabela {T('ablacao')}. Ablação da recompensa: conforto na faixa estreita [23,25] °C.",
        acima=True)
_l5 = []
for _, _r in ABL.iterrows():
    _sem = ABLS.loc[ABLS["variante"] == _r["variante"], "conf_estreita_pct"]
    _d = "\u2014" if _r["variante"] == "completa" else v(_r["cliffs_delta"], 2)
    _l5.append([_r["rotulo"], v(_r["media"]),
                f"[{v(_r['minimo'])}; {v(_r['maximo'])}]",
                " / ".join(v(x) for x in _sem), _d, _r["magnitude"]])
tabela(["Variante", "Média", "Amplitude", "Por semente", "\u03b4 de Cliff", "Magnitude"],
       _l5, negrito_linhas=(0, len(_l5) - 1),
       larguras=[4.6, 1.5, 2.5, 3.7, 1.9, 2.0])

figura(salvar_fig(F.fig_ablacao(ABL, ABLS), "fig2_ablacao.png"))
legenda(f"Figura {Fg('ablacao')}. Ablação da recompensa. As três sementes de cada variante são "
        "exibidas individualmente: com n = 3, a média isolada esconderia que "
        "degraus (legado) tem uma semente em 44,0 e duas acima de 81.")

p(f"Três conclusões emergem (Figura {Fg('ablacao')}). Primeira, a hipótese do platô plano é confirmada: "
  "remover o gradiente interno degrada o conforto estreito de 83,7% para 47,7% "
  "(−36 pp), eleva o desvio de 0,85 para 1,46 °C e o sobreaquecimento de 5,2% "
  "para 13,8%, com separação completa entre sementes (δ = −1,00). Um platô "
  "verdadeiramente plano não recompensa entrar na faixa, e a política estaciona "
  "junto à borda.", space_before=8, first=1.25)

rico([("Segunda, e contrariando a tese do manuscrito: a formulação proposta ",
       0, 0), ("não supera", 0, 0),
      (" a convencional. A variante que usa quadrática pura, sem gradiente "
       "interno, sem penalidade de troca, sem anti-short-cycling e sem "
       "penalidade de frio, atinge 83,1% contra 83,7% da proposta "
       "(δ = −0,11, desprezível). A razão é conceitual: uma quadrática pura é "
       "uma parábola com máximo em 24 °C e, portanto, já possui gradiente em "
       "todo o domínio por construção. O platô plano é que o destrói; a "
       "proposta apenas o restaura. A contribuição, corretamente enquadrada, "
       "é a correção de uma falha introduzida pelos próprios autores, um "
       "achado negativo, não positivo.", 0, 0)], first=1.25)

p("Terceira, três dos cinco termos são inertes: o anti-short-cycling tem efeito "
  "exatamente nulo (δ = 0,00), a penalidade de troca é desprezível (δ = −0,11) "
  "e a penalidade de frio tem magnitude pequena. Indício adicional, com "
  "intervalos sobrepostos e portanto sugestivo: remover o anti-short-cycling "
  "reduz o consumo em 5% (10,53 contra 11,09 kWh/dia) sem custo de conforto, o "
  "que sugere que o termo é ativamente prejudicial.", first=1.25)

h("4.4. A penalidade anti-short-cycling não cumpre seu objetivo", 2)

p("Comutações frequentes do compressor reduzem a vida útil do equipamento; o "
  f"manuscrito endereça isso via a Equação {E('ciclo')}, com d_min = 36 min. A métrica "
  "agregada de comutações por hora, contudo, não permite verificar a proteção. "
  f"A distribuição dos tempos de permanência, sim (Tabela {T('permanencia')} e Figura {Fg('permanencia')}).",
  first=1.25)

legenda(f"Tabela {T('permanencia')}. Distribuição dos tempos de permanência entre comutações "
        "(requisito: d_min = 36 min).", acima=True)
_l6 = [[_r["agente"], f"{_r['mediana_min']:.0f} min", f"{v(_r['media_min'])} min",
        f"{_r['min_min']:.0f} min", f"{v(_r['violacoes_pct'])}%"]
       for _, _r in PERMA.iterrows()]
tabela(["Agente", "Mediana", "Média", "Mínimo", "Violações de d_min"], _l6,
       larguras=[4.4, 2.4, 2.4, 2.2, 3.4])

figura(salvar_fig(F.fig_permanencia(PERM), "fig3_permanencia.png"))
legenda(f"Figura {Fg('permanencia')}. Distribuição acumulada dos tempos de permanência do DQN "
        "Equilibrado. O ponto destacado sobre a linha tracejada marca a fração "
        "de comutações abaixo do requisito: 71,4% sob a penalidade de "
        "recompensa, 6,4% sob restrição dura.")

p("A mediana de 12 minutos corresponde a uma única decisão. A média de 36 a 60 "
  "minutos é inflada por longos períodos com o equipamento desligado e mascara "
  "completamente o problema, caso didático de agregado que oculta a "
  "distribuição.", space_before=8, first=1.25)

rico([(f"A causa é aritmética: a penalidade da Equação {E('ciclo')} vale no máximo "
       "|ρ| = 5, contra ganhos de conforto de até B + B_c = 14. ", 0, 0),
      ("Um termo de recompensa que pode ser superado por outro termo não é uma "
       "proteção; é uma sugestão.", 0, 0)], first=1.25)

p("Aplicando d_min como restrição dura, na linha do shielding de Xu et al. "
  "[2025], sobre o agente já treinado e sem retreinamento, as violações caem de "
  "71,4% para 6,4% e a mediana sobe de 12 para 48 minutos, ao custo de +0,08 °C "
  "no desvio absoluto e +3,8% de energia, mantendo o conforto binário "
  "inalterado em 86,5%. A lição é transferível: propriedades de segurança e de "
  "integridade de hardware devem ser impostas estruturalmente, não negociadas "
  "via função de recompensa.", first=1.25)

h("4.5. Se RL for utilizado, ele precisa ser profundo", 2)

p("A baixa dimensionalidade do espaço de estados (quatro variáveis) sugere que "
  "um método tabular bastaria. Zha et al. [2021] fornecem a razão teórica pela "
  "qual não basta: em espaços contínuos com variáveis de dinâmica lenta, a "
  "transição é intra-tile e o valor nunca se propaga entre células. Mediu-se "
  f"diretamente essa fração, sem treinar nenhum agente (Tabela {T('hnp')}).", first=1.25)

legenda(f"Tabela {T('hnp')}. Diagnóstico de discretização.", acima=True)
tabela(["Grandeza", "Valor"],
       [["Espaço discretizado", "4.800 estados (20 × 10 × 24)"],
        ["Largura do bin de temperatura", "1,00 °C"],
        ["ΔT típico por passo (sala cheia)", "0,090 °C"],
        ["Passos para cruzar um bin", "11,1"],
        ["Transições intra-tile (política aleatória)", "79,8%"],
        ["Transições intra-tile (após treinamento)", "59,8%"],
        ["Cobertura de estados (teto atingido)", "76,7%"]],
       larguras=[8.0, 7.0])

p("Cerca de 80% dos backups atualizam um estado com o próprio valor. O "
  "Q-Learning tabular treinado por 550.000 passos e três sementes atingiu 42,1% "
  "de conforto na faixa larga (31,5 / 38,7 / 56,0 por semente, desvio 12,6), "
  "pior que o termostato com zona morta e altamente instável. A queda de 80% "
  "para 60% de transições intra-tile após o treinamento ocorre porque a "
  "política aprende a usar HIGH, que altera a temperatura rápido o bastante "
  "para cruzar bins; ainda assim, seis de cada dez atualizações permanecem "
  "inócuas, e cerca de 1.100 dos 4.800 estados nunca são visitados.",
  space_before=8, first=1.25)

p("O obstáculo, portanto, não é a dimensionalidade, e sim a razão entre o passo "
  "da dinâmica (0,090 °C) e a resolução da discretização (1,00 °C). Este é o "
  "único achado deste trabalho que é integralmente derivável de parâmetros "
  "publicados, sendo, portanto, o de maior robustez.", first=1.25)

if not CURTO:
    h("4.6. Três tentativas de construir um regime favorável ao RL", 2)

    p("Estabelecido que o controle clássico domina na formulação original, "
      "investigou-se se alguma extensão do problema reverteria o resultado. Três "
      "regimes foram construídos e avaliados.", first=1.25)

    rico([("Rastreamento de precisão. ", 1, 0),
          ("Um laboratório com tolerância de ±0,5 °C em torno do setpoint, com "
           "equipamento reversível, ação contínua e observação enriquecida com "
           "erro escalado, integral com fuga e derivada. Agentes TD3 e SAC foram "
           "treinados sob três perfis de custo. O PI bidirecional permanece "
           f"superior (Tabela {T('precisao')} e Figura {Fg('precisao')}).", 0, 0)], first=1.25)

    legenda(f"Tabela {T('precisao')}. Regime de precisão: percentual do tempo dentro da tolerância "
            "de ±0,5 °C em regime permanente.", acima=True)
    _l8 = [[_r["controlador"].replace("Lab_", "").replace("Precisao", "Precisão")
             .replace("Economico", "Econômico"),
            v(_r["na_tol_%"]), v(_r["sigma"], 3), v(_r["|T-24|"], 3),
            v(_r["custo_dia"], 2), v(_r["kWh_dia"], 2), v(_r["ajustes_h"], 2)]
           for _, _r in LAB[LAB["controlador"] != "Antecipatório (manual)"].iterrows()]
    tabela(["Controlador", "Na tolerância %", "\u03c3 (\u00b0C)", "|T\u221224| \u00b0C",
            "Custo R$/dia", "kWh/dia", "Ajustes/h"], _l8, negrito_linhas=(0,),
           larguras=[4.6, 2.4, 1.7, 1.9, 2.2, 1.8, 1.9])

    figura(salvar_fig(F.fig_precisao(LAB), "fig4_precisao.png"))
    legenda(f"Figura {Fg('precisao')}. Regime de precisão: custo diário contra dispersão da "
            "temperatura; a área do marcador é proporcional ao tempo dentro da "
            "tolerância. O PI ocupa sozinho o canto ótimo.")

    rico([("Antecipação tarifária. ", 1, 0),
          ("Sob tarifa branca real, com posto de ponta de razão 2,21×, a "
           "oportunidade de pré-resfriamento é limitada pela própria tolerância: "
           "com ±0,5 °C, o armazenamento térmico disponível cobre apenas cerca de "
           "29% da duração do posto de ponta. A margem para antecipação é, "
           "portanto, estruturalmente pequena.", 0, 0)],
         space_before=8, first=1.25)

    rico([("Demanda contratada. ", 1, 0),
          ("Restrição sobre o máximo da média integrada em 15 minutos, e não sobre "
           "a média, grandeza que realimentação puramente reativa não representa, "
           "e cujo atendimento exigiria antecipação. Uma versão preliminar deste "
           "experimento fixava a demanda contratada em 0,70 kW, valor escolhido por "
           "varredura como ponto de máxima tensão entre conforto e restrição. "
           "Auditando esse valor sob a mesma exigência aplicada ao manuscrito, "
           "constatou-se que ele era ", 0, 0), ("infactível", 0, 0),
          (": o regime permanente de pior caso exige 1,204 kW, de modo que 0,70 kW "
           "está 72% abaixo do necessário e sustenta apenas cerca de 17 dos 45 "
           "ocupantes. Nenhuma política respeita tal restrição, pois antecipação "
           "não cria regime permanente.", 0, 0)], first=1.25)

    p("Substituiu-se o valor arbitrado por dimensionamento derivado da condição de "
      "projeto, ocupação máxima, externa no pico diário, mantendo o setpoint, "
      "acrescido de margem de contratação de 10%, resultando em 1,324 kW. Sob esse "
      "contrato, nenhum dos nove cenários excede o limite (pico máximo observado de "
      "1,213 kW), e o PI ingênuo torna-se indistinguível do PI ciente da restrição.",
      first=1.25)

    figura(salvar_fig(F.fig_demanda(DEM), "fig5_demanda.png"))
    legenda(f"Figura {Fg('demanda')}. Origem da pressão de demanda. Apenas os cenários que partem "
            "fora do setpoint excedem o contrato derivado, e neles a violação ocorre "
            "no passo inicial, onde antecipar é impossível por definição.")

    p(f"A razão é uma contradição estrutural (Figura {Fg('demanda')}): os únicos cenários que "
      "excedem o contrato derivado são os que partem fora do setpoint, nos quais a violação "
      "ocorre no passo inicial, instante em que antecipação é impossível por "
      "definição. Ao construir a matriz de cenários partindo do setpoint para "
      "eliminar essa violação impossível, elimina-se simultaneamente a única fonte "
      "de demanda acima do regime permanente. O conjunto no qual a hipótese pode "
      "ser testada é vazio.", first=1.25)

    # ============================================================== 5. DISCUSSÃO

h("4.7. Precisão exigida: a faixa estreita favorece o RL?", 2)

p("A hipótese é natural: se a especificação exigir mais precisão, o aprendizado "
  "teria mais a oferecer que uma lei fixa. Para testá-la, ambos os controladores "
  "são preparados para *cada* largura de faixa, o DQN é retreinado com a faixa "
  "correspondente e o PI é re-sintonizado por busca em grade. Congelar os ganhos "
  "do PI e cobrar precisão maior reproduziria, dentro deste experimento, o mesmo "
  "defeito que a Seção 4.2 identifica no manuscrito auditado.", first=1.25)

_l10 = []
for _t in sorted(FAIXA["tolerancia"].unique(), reverse=True):
    _sub = FAIXA[FAIXA["tolerancia"] == _t]
    _pi = _sub[_sub["controlador"] == "PI sintonizado"]["na_tolerancia_pct"].iloc[0]
    _dq = _sub[_sub["controlador"] == "DQN"]["na_tolerancia_pct"]
    _te = _sub[_sub["controlador"] == "Termostato (zm=1 °C)"]["na_tolerancia_pct"]
    _l10.append([f"\u00b1{v(_t)} \u00b0C", v(_pi), v(_dq.mean()),
                 " / ".join(v(x) for x in _dq), v(_dq.std()),
                 v(_te.iloc[0]) if len(_te) else "\u2014",
                 f"+{v(_pi - _dq.mean())}"])
legenda(f"Tabela {T('faixa')}. Desempenho por largura de faixa alvo, com o DQN retreinado "
        "e o PI re-sintonizado para cada uma.", acima=True)
tabela(["Faixa alvo", "PI", "DQN (média)", "DQN por semente", "Desvio",
        "Termostato", "Vantagem do PI"], _l10, negrito_linhas=(2,),
       larguras=[2.2, 1.6, 2.2, 3.4, 1.6, 2.2, 2.6])

figura(salvar_fig(F.fig_faixa_estreita(FAIXA), "fig6_faixa_estreita.png"))
legenda(f"Figura {Fg('faixa')}. Desempenho por largura de faixa, com ambos os controladores "
        "preparados para cada largura. As sementes do DQN aparecem "
        "individualmente.")

p(f"O efeito foi o inverso do previsto (Figura {Fg('faixa')}): estreitar a "
  "faixa *amplia* a vantagem do controle clássico, de forma monotônica na "
  "média, +0,0, +3,0 e +8,3 pontos percentuais. Um segundo efeito acompanha o "
  "primeiro: o desvio entre sementes do DQN cresceu de 0,00 para 5,20 e 4,04. "
  "Quando a especificação aperta, o treinamento não apenas entregou menos, como "
  "se tornou instável.", space_before=8, first=1.25)

# A leitura por semente é derivada dos dados, e não escrita à mão: com n = 3, a
# diferença de médias pode ser menor que a dispersão, caso em que reportar
# apenas a média afirmaria mais do que os dados sustentam.
_sem_por_faixa = {}
for _t in sorted(FAIXA["tolerancia"].unique(), reverse=True):
    _s = FAIXA[FAIXA["tolerancia"] == _t]
    _pi_v = _s[_s["controlador"] == "PI sintonizado"]["na_tolerancia_pct"].iloc[0]
    _dq_v = _s[_s["controlador"] == "DQN"]["na_tolerancia_pct"]
    _sem_por_faixa[_t] = (_pi_v, int((_dq_v < _pi_v - 0.05).sum()), len(_dq_v),
                          _dq_v.std(), _pi_v - _dq_v.mean())

_n_ab_1, _n_1, _sd_1, _dif_1 = _sem_por_faixa[1.0][1:]
_n_ab_05, _n_05 = _sem_por_faixa[0.5][1], _sem_por_faixa[0.5][2]

rico([("A monotonicidade da média, porém, não deve ser lida como um déficit "
       "sistemático que cresce de modo uniforme, e a distinção importa porque a "
       f"dispersão entre sementes em ±1,0 °C ({v(_sd_1, 2)}) ", 0, 0),
      ("excede a própria diferença entre as médias", 0, 0),
      (f" ({v(_dif_1)} pontos). Nessa largura, {_EXT[_n_1 - _n_ab_1]} das "
       f"{_EXT[_n_1]} sementes empatam com o PI e "
       f"{_EXT[_n_ab_1]} {'colapsa' if _n_ab_1 == 1 else 'colapsam'}, de modo "
       "que o que a média registra é instabilidade de treinamento, e não perda "
       f"uniforme. Em ±0,5 °C o quadro é distinto e mais forte: "
       f"{_EXT[_n_ab_05]} das {_EXT[_n_05]} "
       "sementes ficam abaixo do PI, o que caracteriza separação e sustenta a "
       "afirmação. A leitura defensável é, portanto, que a vantagem do controle "
       "clássico é nula em faixa larga, indistinguível do ruído de semente em "
       "faixa intermediária e consistente em faixa estreita.", 0, 0)],
     first=1.25)

rico([("Resta a objeção óbvia: o agente teria treinado o suficiente? A "
       f"Figura {Fg('curva')} responde avaliando a política periodicamente "
       "durante o treino, contra a linha do PI, que é horizontal por "
       "construção, pois não aprende.", 0, 0)], first=1.25)

figura(salvar_fig(F.fig_curva_aprendizado(CURVA), "fig7_curva.png"))
legenda(f"Figura {Fg('curva')}. Desempenho contra orçamento de treinamento, para duas "
        "larguras de faixa. Linhas tênues são sementes individuais.")

p("Em ±2,0 °C o agente cruzou rapidamente e saturou exatamente sobre a linha do "
  "PI: o empate da Seção 4.2 corresponde ao patamar de convergência, e não um artefato do "
  "ponto de parada. Em ±0,5 °C a curva subiu de 4 % para cerca de 63 % e permaneceu "
  "16,9 pontos abaixo do PI ao fim do orçamento, com forte desaceleração.",
  space_before=8, first=1.25)

rico([("Ressalva necessária: ", 1, 0),
      ("a curva desacelera acentuadamente, mas não é plana, a tendência nos "
       "últimos 150 mil passos ainda é de +2,9 pontos. Mantida essa taxa, "
       "hipótese otimista que o próprio achatamento contradiz, seriam "
       "necessários cerca de 870 mil passos adicionais para fechar a diferença. "
       "O que os dados sustentam, portanto, é um custo de amostra elevado e "
       "crescente com a exigência de precisão, e não a impossibilidade de "
       "convergência. É a mesma ordem de dificuldade que Yuan et al. [2020] "
       "reportam ao observar que o agente supera o PID apenas após anos de "
       "operação.", 0, 0)], first=1.25)

h("4.8. Por que o agente gasta mais: o nível de melhor eficiência é descartado", 2)

_ref_consumo = f" (Figura {Fg('consumo')})" if tem_fig('consumo') else ""
p("Os agentes DQN entregam o mesmo conforto do PI consumindo mais energia. A "
  f"decomposição do consumo por nível de potência acionado{_ref_consumo}, na "
  "linha da análise por item de Yuan et al. [2020], identifica a causa.",
  first=1.25)

_l11 = []
for _, _r in NIVEIS.iterrows():
    _l11.append([_r["controlador"], v(_r["OFF"]), v(_r["LOW"]), v(_r["MEDIUM"]),
                 v(_r["HIGH"])])
legenda(f"Tabela {T('niveis')}. Fração do tempo em cada nível de potência "
        "acionado, por controlador.", acima=True)
tabela(["Controlador", "OFF %", f"LOW % (COP {COP['LOW']:.2f})",
        f"MEDIUM % (COP {COP['MEDIUM']:.2f})", f"HIGH % (COP {COP['HIGH']:.2f})"],
       _l11, larguras=[4.6, 2.2, 2.9, 3.1, 2.6])

figura(salvar_fig(F.fig_uso_dos_niveis(NIVEIS, cop=COP), "fig8_niveis.png"))
legenda(f"Figura {Fg('niveis')}. Fração do tempo em cada nível de potência. O nível de melhor "
        "COP está ausente nos três perfis aprendidos por DQN.")

p(f"Os três perfis do manuscrito acionaram MEDIUM em 0,0 % do tempo (Figura {Fg('niveis')}), embora seus "
  "pesos de recompensa variem por fatores de três a quatro, o gradiente de "
  "conforto vai de 7,0 a 3,0, a penalidade de energia de 0,03 a 0,12. O perfil "
  "Agressivo descarta também o nível LOW, operando em regime puramente "
  "liga-desliga. E MEDIUM é justamente o nível de maior coeficiente de "
  "performance do equipamento, por reproduzir o pico de eficiência em carga "
  "parcial característico de máquinas inverter.", space_before=8, first=1.25)

if tem_fig('consumo'):
    figura(salvar_fig(F.fig_consumo_decomposto(DECOMP, cop=COP), "fig9_consumo.png"))
    legenda(f"Figura {Fg('consumo')}. Consumo diário decomposto por nível "
            "acionado. A energia excedente do agente aprendido não vem de "
            "operar mais tempo, e sim de operar nos níveis menos eficientes.")

rico([("A causa não é a função de recompensa, e há um controle experimental "
       "limpo para isso: o agente SAC, treinado com a *mesma* recompensa do perfil "
       "Equilibrado, aciona MEDIUM em 13,3 % do tempo, praticamente o mesmo que "
       "o PI (12,0 %). Mesma recompensa, comportamentos opostos.", 0, 0)],
     space_before=8, first=1.25)

p("A análise dos valores de ação esclarece o mecanismo. MEDIUM ocupou a terceira "
  "posição no ordenamento em todos os perfis, mas a magnitude precisa ser lida "
  "na escala correta: os valores absolutos são da ordem de 980, a faixa inteira "
  "entre as quatro ações vale cerca de 1,9 % desse valor, e o déficit de MEDIUM "
  "para a melhor ação é de aproximadamente 1,2 %. As quatro ações são, portanto, "
  "quase equivalentes.", first=1.25)

_l_q = [[_r["controlador"], v(_r["posicao_medium"], 2),
         v(_r["deficit_medium"], 2), v(_r["q_absoluto"], 0),
         v(_r["faixa_pct_do_valor"], 2) + " %",
         v(_r["deficit_pct_do_valor"], 2) + " %", v(_r["argmax_medium_pct"], 1) + " %"]
        for _, _r in QVAL.iterrows()]
legenda(f"Tabela {T('qvalues')}. Valores de ação: posição de MEDIUM e a escala do déficit.",
        acima=True)
tabela(["Controlador", "Posição de MEDIUM", "Déficit absoluto", "Valor médio",
        "Faixa entre ações", "Déficit relativo", "Vezes como melhor"],
       _l_q, larguras=[3.4, 2.6, 2.4, 2.2, 2.6, 2.4, 2.4])

rico([("O mecanismo é, portanto, de ", 0, 0), ("winner-take-all", 0, 1),
      (": uma política determinística por maximização converte uma margem de "
       "cerca de 1 % em uso de 0 %. Sob esse regime de decisão não existe "
       "\"acionar MEDIUM ocasionalmente\", ou a ação é a melhor em algum "
       "estado, ou nunca aparece. A consequência prática é direta: reponderar "
       "termos da recompensa é o caminho errado, pois os três perfis já cobrem "
       "ampla variação e colapsam do mesmo modo. O que restaura o uso dos níveis "
       "intermediários é mudar a classe de política, contínua ou estocástica, "
       "ou conceder ao agente autoridade de atuação mais fina.", 0, 0)],
     space_before=8, first=1.25)

h("4.9. Avaliação em conjunto amplo e independente", 2)

p("A matriz fatorial tem nove pontos e foi utilizada para sintonizar o "
  "controlador PI, o que constitui a ressalva metodológica registrada na "
  "Seção 4.2. Avaliou-se, então, em 150 cenários amostrados aleatoriamente sobre o "
  "espaço contínuo, temperatura inicial de 16 a 33 °C, ocupação de 0 a 45 "
  "pessoas, hora de início de 0 a 23, com semente fixa distinta das de "
  "treinamento. Todos os controladores recebem exatamente os mesmos cenários, o "
  "que torna a comparação emparelhada. Para o PI, trata-se de teste fora da "
  "amostra.", first=1.25)

_g = ALEAT.groupby("controlador").agg(
    larga=("conf_larga_pct", "mean"), estreita=("conf_estreita_pct", "mean"),
    tol=("na_tolerancia_pct", "mean"), desvio=("desvio_ideal", "mean"),
    kwh=("energia_kwh", "mean")).sort_values("tol", ascending=False)
_l12 = [[_n, v(_r["larga"]), v(_r["estreita"]), v(_r["tol"]),
         v(_r["desvio"], 2), v(_r["kwh"], 2)] for _n, _r in _g.iterrows()]
legenda(f"Tabela {T('aleatorios')}. Desempenho em 150 cenários aleatórios independentes.",
        acima=True)
tabela(["Controlador", "Conf. [22,26] %", "Conf. [23,25] %", "Dentro de \u00b10,5 \u00b0C %",
        "|T\u221224| \u00b0C", "Energia kWh/dia"], _l12,
       larguras=[4.4, 2.4, 2.4, 2.8, 2.0, 2.4])

p("Três leituras. Primeira, na faixa larga os quatro melhores controladores "
  "produziram valor idêntico: a métrica estava saturada, e o número é determinado "
  "pelo transitório de partida, limitado pela física do equipamento e não pela "
  "política. Segunda, apenas a tolerância estreita discrimina, e nela o perfil "
  "Agressivo alcançou o PI, resultado mais favorável ao aprendizado do que a "
  "matriz fatorial sugeria. Terceira, o PI manteve o desempenho fora da amostra, "
  "o que atenua, sem eliminar, a ressalva sobre sua sintonia.",
  space_before=8, first=1.25)

_tol_g = ALEAT.groupby("controlador")["na_tolerancia_pct"].mean()
_amp_tol = _tol_g.max() - _tol_g.min()

rico([("A saturação merece atenção maior do que a de um defeito de métrica, "
       "porque sugere uma objeção de fundo a todo o artigo: ", 0, 0),
      ("um ambiente em que quatro controladores distintos produzem o mesmo "
       "número pode simplesmente não ter resolução suficiente para discriminar "
       "políticas", 0, 0),
      (", e nesse caso a ausência de vantagem do aprendizado seria consequência "
       "da parametrização do ambiente e não uma propriedade do problema de "
       "climatização. A constante de tempo de 30 h discutida na Seção 3.1 dá "
       "substância à objeção, e ela é reconhecida como a ameaça mais séria à "
       "validade externa destes resultados. Três elementos, contudo, impedem "
       "que ela explique o quadro por completo. A saturação é específica das "
       "métricas de faixa larga e do transitório, e desaparece sob a tolerância "
       f"de ±0,5 °C, onde os controladores se separam por {v(_amp_tol)} pontos "
       "percentuais. A separação que aparece ali não é ruído, pois se concentra "
       "em uma condição identificável, salas quase vazias, e tem mecanismo "
       "medido na Seção 4.8. E o consumo energético, que não satura em nenhuma "
       "das condições avaliadas, ordena os controladores de forma consistente. "
       "O que a inércia elevada compromete, portanto, é a magnitude "
       "transferível das diferenças, e não a existência do mecanismo que as "
       "produz.", 0, 0)], first=1.25)

figura(salvar_fig(F.fig_divergencia_por_condicao(ALEAT), "fig10_divergencia.png"))
legenda(f"Figura {Fg('divergencia')}. Diferença em relação ao PI por faixa de ocupação. O déficit "
        "dos perfis Equilibrado e Passivo concentra-se em salas vazias.")

p(f"A divergência não é difusa, e sim concentrada (Figura {Fg('divergencia')}): os perfis Equilibrado e "
  "Passivo perderam 21,7 e 18,2 pontos percentuais em cenários com até dez "
  "ocupantes, contra 4,4 e 3,1 pontos com a sala cheia. Os piores casos "
  "compartilham assinatura, poucos ocupantes, madrugada, nos quais o PI "
  "permaneceu integralmente dentro da tolerância enquanto o agente caiu para a "
  "faixa de 37 a 58 %.", space_before=8, first=1.25)

p("O mecanismo é coerente com a Seção 4.8: com carga térmica mínima, manter "
  "±0,5 °C exige potência muito baixa e finamente dosada, e o menor nível "
  "não-nulo disponível já representa 25 % da capacidade nominal. O controlador "
  "proporcional obtém média efetiva inferior a qualquer nível isolado alternando "
  "com o ciclo adequado; a política discreta oscila entre resfriar em excesso e "
  "permitir a elevação. O perfil Agressivo não sofre desse efeito, por operar em "
  "liga-desliga e acionar potência máxima sem hesitação, ao custo do consumo.",
  first=1.25)

if tem_fig('pareto'):
    figura(salvar_fig(F.fig_pareto_aleatorios(ALEAT), "fig11_pareto.png"))
    legenda(f"Figura {Fg('pareto')}. Precisão contra energia nos cenários "
            "aleatórios. Nenhum perfil aprendido ocupa o canto ótimo.")

_ref_pareto = f" (Figura {Fg('pareto')})" if tem_fig('pareto') else ""
p(f"O quadro consolidado{_ref_pareto} é o de agentes situados "
  "abaixo da fronteira de Pareto: "
  "para igualar o PI em precisão, o perfil Agressivo consumiu 12,9 % mais "
  "energia; para igualar o consumo, os perfis Equilibrado e Passivo perdem entre "
  "seis e sete pontos de precisão. Como os três perfis foram obtidos variando "
  "exclusivamente pesos da recompensa, a tese central do manuscrito auditado, "
  "o resultado indica que essa variação percorre uma curva situada inteiramente "
  "no interior da fronteira, sem tangenciá-la.", space_before=8, first=1.25)

h("4.10. Velocidade de resposta", 2)

p("As métricas anteriores avaliam qualidade em regime permanente e não medem "
  "velocidade de resposta, que é requisito operacional independente: um ambiente "
  "que leva horas para tornar-se utilizável constitui problema ainda que depois "
  "se mantenha adequado. Esta seção mede o transitório sobre os cenários "
  "aleatórios que iniciam acima da faixa, reportando três grandezas distintas, "
  "o instante da primeira entrada na faixa, o instante a partir do qual não há "
  "mais saída, e a fração do transitório em que o atuador opera saturado.",
  first=1.25)

_g = TRANS.groupby("controlador").agg(
    ent=("entrada_h", "mean"), sd=("entrada_h", "std"),
    aco=("acomodacao_h", "mean"), tax=("taxa_c_por_h", "mean"),
    sat=("saturacao_pct", "mean")).sort_values("ent")
_l14 = [[_n, v(_r["ent"], 2), v(_r["sd"], 2), v(_r["aco"], 2), v(_r["tax"], 2),
         v(_r["sat"], 0)] for _n, _r in _g.iterrows()]
legenda(f"Tabela {T('transitorio')}. Resposta transitória nos cenários que "
        "iniciam acima da faixa.", acima=True)
tabela(["Controlador", "Entrada (h)", "Desvio", "Acomodação (h)",
        "Taxa (°C/h)", "Saturação %"], _l14, negrito_linhas=(0,),
       larguras=[4.6, 2.2, 1.8, 2.6, 2.2, 2.2])

figura(salvar_fig(F.fig_transitorio(TRANS), "fig12_transitorio.png"))
legenda(f"Figura {Fg('transitorio')}. Tempo para entrar na faixa e tempo para "
        "deixar de sair dela. A distância entre as duas barras revela o "
        "comportamento oscilatório do termostato.")

p(f"O controlador PI e os três perfis DQN apresentaram tempo idêntico (Figura "
  f"{Fg('transitorio')}), divergindo em zero dos quarenta e oito cenários "
  "avaliados. A explicação está na última "
  "coluna: todos operam em potência máxima durante a totalidade do transitório. "
  "Durante o resfriamento inicial não existe decisão a ser tomada, a única ação "
  "sensata é acionar potência total, e a velocidade fica determinada pela "
  "física do equipamento, não pela política de controle. Esse resultado confirma "
  "por medição a explicação oferecida na Seção 4.9 para o conforto idêntico "
  f"observado na Tabela {T('aleatorios')}, até então uma inferência.",
  space_before=8, first=1.25)

p("O termostato revelou seu defeito característico na distância entre as duas "
  "grandezas: entrou na faixa em 1,99 h, mas levou aproximadamente vinte horas "
  "para deixar de sair dela, pois estaciona no teto e oscila em torno dele. Uma "
  "métrica que considerasse apenas a primeira entrada o favoreceria "
  "indevidamente. O agente SAC, por sua vez, foi 3,1 vezes mais lento que o PI, e "
  "a causa é a mesma coluna: saturou o atuador em apenas 31 % do transitório.",
  first=1.25)

rico([("Registre-se uma observação sobre o conjunto de métricas. Esta é a "
       "terceira grandeza em que PI e DQN produzem valores indistinguíveis, ao "
       "lado do conforto na faixa larga e do conforto em cenários aleatórios. O "
       "padrão não é coincidência estatística: ", 0, 0),
      ("fora do regime permanente a política ótima é trivial", 0, 0),
      (", potência máxima, e qualquer controlador competente a encontra. A "
       "diferenciação entre controladores existe apenas em regime, e ali o fator "
       "decisivo é a resolução da atuação, pelas razões da Seção 4.8.", 0, 0)],
     first=1.25)

h("4.11. Os parâmetros inferidos podem ser derivados?", 2)

p("Quatro constantes da função de recompensa não são publicadas pelo manuscrito "
  "auditado e foram inferidas de sua figura, o que constitui a principal "
  "ressalva de reprodutibilidade deste trabalho. Cabe perguntar se algumas delas "
  "podem ser obtidas por argumento, em vez de leitura de gráfico.", first=1.25)

rico([("O bônus base B admite resposta puramente analítica. Ele comparece a "
       f"ambos os ramos do termo de conforto (Equações {E('conforto_dentro')} e "
       f"{E('conforto_fora')}), e o episódio tem duração fixa, sem terminação "
       "antecipada; somar uma constante a cada passo acrescenta o mesmo valor ao "
       "retorno de qualquer política e, portanto, ", 0, 0),
      ("não altera a ordenação entre elas", 0, 0),
      (". A política ótima independe de B. Resta, contudo, um efeito numérico: "
       "B infla a escala dos valores de ação sem carregar informação, o que "
       "comprime a diferença relativa entre ações, precisamente a condição "
       "identificada na Seção 4.8.", 0, 0)], first=1.25)

p("A curvatura k, por sua vez, revela-se incoerente por simples inspeção da "
  "escala de custos que a formulação produz.", first=1.25)

legenda(f"Tabela {T('coerencia')}. Custo em recompensa de cada desvio, sob os "
        "parâmetros inferidos.", acima=True)
tabela(["Movimento na faixa de temperatura", "Custo em recompensa"],
       [["Do centro à borda, por dentro (24 a 26 °C)",
         v(PARAM["custo_centro_ate_borda"], 2)],
        ["Sair 0,5 °C além da borda", v(PARAM["custo_fora_por_delta"][0.5], 2)],
        ["Sair 1 °C além da borda", v(PARAM["custo_fora_por_delta"][1], 2)],
        ["Sair 2 °C além da borda", v(PARAM["custo_fora_por_delta"][2], 2)],
        ["Inclinação junto à borda, por dentro",
         v(PARAM["inclinacao_interna_por_c"], 2) + " por °C"],
        ["Inclinação a 0,1 °C da borda, por fora",
         v(PARAM["inclinacao_externa_em"][0.1], 2) + " por °C"]],
       larguras=[8.6, 5.4])

p("Permanecer 2 °C fora da faixa custa menos que percorrer a meia-faixa por "
  "dentro dela, e a penalidade marginal cai de 2,00 para 0,12 por grau ao cruzar "
  "a fronteira, de modo que sair passa a ser um alívio na margem. Impondo que "
  "sair δ graus custe o mesmo que atravessar a meia-faixa, obtém-se k = 4 para "
  "δ = 1 °C e k = 1 para δ = 2 °C, contra os 0,6 inferidos. Registre-se que "
  "continuidade de derivada é inatingível com penalidade quadrática pura, pois "
  "sua inclinação é nula na fronteira; alcançá-la exigiria um termo linear.",
  space_before=8, first=1.25)

_ord = ["B = 10.0", "B = 0.0", "k = 0.6", "k = 1.0", "k = 4.0",
        "rho = -5.0", "rho = -20.0", "frio = -2.0", "frio = -10.0"]
_lc = []
for _v in _ord:
    _s = CALIB[CALIB["variante"] == _v]
    if _s.empty:
        continue
    _lc.append([_v.replace(".0", "").replace("rho", "ρ"),
                v(_s["conf_larga"].mean()), v(_s["conf_estreita"].mean()),
                v(_s["conf_estreita"].std(), 2), v(_s["desvio"].mean(), 2),
                v(_s["violacoes_dmin_pct"].mean(), 0) + " %"])
legenda(f"Tabela {T('derivados')}. Varredura dos quatro parâmetros, duas "
        "sementes por configuração.", acima=True)
tabela(["Configuração", "Conf. larga %", "Conf. estreita %", "Desvio entre sementes",
        "|T−24| °C", "Violações de d_min"], _lc, larguras=[2.8, 2.4, 2.6, 2.8, 2.0, 2.4])

figura(salvar_fig(F.fig_parametros_derivados(COMB, CALIB), "fig13_derivados.png"))
legenda(f"Figura {Fg('derivados')}. Efeito de derivar B e k. À esquerda, cada "
        "parâmetro isoladamente; à direita, a combinação. Os pontos são as "
        "sementes individuais.")

p(f"A varredura (Figura {Fg('derivados')}) confirma as duas previsões "
  "analíticas e refuta as duas restantes. "
  "Zerar o bônus base preserva o conforto na faixa larga, como a análise exigia, "
  "e eleva o conforto na faixa estreita de 70,2 % para 82,7 %, com escala dos "
  "valores de ação reduzida de 356 para 145. Adotar k = 1 eleva-o a 79,8 % e "
  "reduz as violações de permanência mínima. Já k = 4 não acrescenta desempenho "
  "e desestabiliza o treinamento, com desvio entre sementes de 13,2.",
  space_before=8, first=1.25)

rico([("Quanto a ρ, quadruplicar a penalidade ", 0, 0), ("não", 0, 0),
      (" reduziu as violações, que passaram de 82,6 % para 83,4 %, e "
       "desestabilizou o treinamento. O resultado corrobora a análise da forma "
       f"da Equação {E('ciclo')}: sendo a penalidade proporcional à antecipação, ela vale apenas "
       "0,83 quando a comutação ocorre a cinco dos seis passos exigidos, isto é, "
       "praticamente desaparece onde a violação é mais provável. Nenhum valor de "
       "ρ corrige uma penalidade que se anula justamente onde deveria atuar; a "
       "correção é estrutural, conforme a Seção 4.4.", 0, 0)], first=1.25)

p("A penalidade de frio, por fim, mostra-se corretamente escolhida. Elevá-la de "
  "−2 para −10 colapsa a política, com o conforto na faixa larga caindo de "
  "86,5 % para 41,4 %, o que reproduz literalmente a advertência registrada na "
  "conclusão do manuscrito auditado quanto à indução de comportamento de "
  "evitação. Cabe uma precisão: o termo não incide em nenhum passo sob a "
  "política treinada, mas incide em 63,5 % dos passos sob política aleatória, "
  "que é o regime do início do treinamento, respondendo por cerca de 9 % do "
  "retorno. Não se trata, portanto, de termo inerte, e sim de termo eficaz, cujo "
  "sucesso consiste exatamente em deixar de ser acionado.", first=1.25)

legenda(f"Tabela {T('combinacao')}. Configuração derivada contra a inferida, "
        "sob o orçamento reduzido da calibração (300 mil passos, três sementes).",
        acima=True)
tabela(["Configuração", "Conf. larga %", "Conf. estreita %", "Por semente",
        "Desvio", "|T−24| °C", "kWh/dia"],
       [[_c, v(COMB[COMB["config"] == _c]["conf_larga"].mean()),
         v(COMB[COMB["config"] == _c]["conf_estreita"].mean()),
         " / ".join(v(x) for x in COMB[COMB["config"] == _c]["conf_estreita"]),
         v(COMB[COMB["config"] == _c]["conf_estreita"].std(), 2),
         v(COMB[COMB["config"] == _c]["desvio"].mean(), 2),
         v(COMB[COMB["config"] == _c]["energia"].mean(), 2)]
        for _c in COMB["config"].unique()],
       negrito_linhas=(1,), larguras=[3.6, 2.2, 2.4, 3.2, 1.6, 1.8, 1.8])

p("Sob 300 mil passos, os dois parâmetros derivados elevam o conforto na faixa "
  "estreita de 64,5 % para 81,2 %, com separação completa entre sementes e queda "
  "do desvio de 10,14 para 2,51. O resultado parecia autorizar a substituição "
  "dos valores inferidos pelos derivados.", space_before=8, first=1.25)

rico([("Essa conclusão, porém, não sobrevive ao orçamento do protocolo. "
       "Retreinando os três perfis com 550 mil passos e três sementes, ", 0, 0),
      ("o efeito inverte de sinal", 0, 0), (".", 0, 0)], first=1.25)

_r = ORC["resumo"].set_index(["orcamento", "config"])
_lo = []
for _o in ("300k", "550k"):
    for _c in ("inferidos", "derivados"):
        if (_o, _c) not in _r.index:
            continue
        _x = _r.loc[(_o, _c)]
        _lo.append([f"{_o[:3]} mil passos", _c,
                    v(_x["media"]), v(_x["desvio"], 2), str(int(_x["n"]))])
legenda(f"Tabela {T('orcamento')}. Conforto na faixa estreita segundo o "
        "orçamento de treinamento e a configuração de recompensa.", acima=True)
tabela(["Orçamento", "Configuração", "Conf. estreita %", "Desvio", "n"],
       _lo, larguras=[3.4, 3.4, 3.0, 2.2, 1.6])

figura(salvar_fig(F.fig_orcamento_parametros(ORC), "fig14_orcamento.png"))
legenda(f"Figura {Fg('orcamento')}. Interação entre parâmetros de recompensa e "
        "orçamento de treinamento. As curvas se cruzam: a configuração derivada "
        "converge mais rápido e depois degrada, enquanto a inferida aprende "
        "devagar e continua melhorando.")

p(f"A configuração derivada converge mais rapidamente e em seguida degrada "
  f"(Figura {Fg('orcamento')}); a "
  "inferida aprende devagar e continua a melhorar. As curvas se cruzam entre os "
  "dois orçamentos. O comportamento é compatível com instabilidade do "
  "aprendizado por diferenças temporais: elevar a curvatura amplia a faixa "
  "dinâmica dos alvos de regressão fora da faixa de conforto, e remover o bônus "
  "base reduz a escala dos valores, combinação que favorece a convergência "
  "inicial e prejudica a estabilidade prolongada. A dispersão entre sementes da "
  "configuração derivada sob 550 mil passos, de 9,95, contra 0,10 da inferida, "
  "reforça essa leitura.", space_before=8, first=1.25)

rico([("Três consequências. A primeira é sobre os parâmetros: os valores "
       "inferidos, ainda que incoerentes segundo a análise de escala da "
       f"Tabela {T('coerencia')}, são os que funcionam sob o orçamento adotado, "
       "e permanecem em uso em todas as demais seções deste artigo. A segunda é "
       "metodológica e transcende este caso: ", 0, 0),
      ("calibrar parâmetros de recompensa sob orçamento reduzido e extrapolar "
       "para o orçamento completo é inválido", 0, 0),
      (", pois a interação entre formulação e duração do treinamento não é "
       "monotônica. A terceira diz respeito à ressalva de reprodutibilidade: a "
       "análise de B e k continua válida como crítica à formulação, mas não "
       "autoriza substituir os valores, de modo que a ressalva sobre os quatro "
       "parâmetros não publicados permanece.", 0, 0)], first=1.25)

# ============================================ 4.12. TRANSFERÊNCIA (BOPTEST)
if BOPT is not None and BRES is not None:
    h("4.12. Transferência para um emulador de terceiros", 2)

    _pk, _tp = "peak_cool_day", "typical_cool_day"
    _b = BRES

    # Todo número citado no texto sai daqui, não da leitura da tabela.
    def _celula(periodo, controlador, coluna):
        _s = BOPT[(BOPT["periodo"] == periodo)
                  & (BOPT["controlador"] == controlador)][coluna]
        return float(_s.iloc[0])

    _PI_R = "PI re-sintonizado (emulador)"
    _narrow_pi = _celula(_pk, _PI_R, "comfort_narrow_pct")
    _narrow_ag = _celula(_pk, _b[_pk]["melhor_agente"], "comfort_narrow_pct")
    _kwh_pi = _celula(_pk, _PI_R, "energy_kwh_day")
    _kwh_ag = _celula(_pk, _b[_pk]["melhor_agente"], "energy_kwh_day")
    _tdis_pi_pk = _celula(_pk, _PI_R, "kpi_tdis_tot")
    _tdis_ag_pk = _celula(_pk, _b[_pk]["melhor_agente"], "kpi_tdis_tot")
    _tdis_pi_tp = _celula(_tp, _PI_R, "kpi_tdis_tot")
    _tdis_ag_tp = _celula(_tp, _b[_tp]["melhor_agente"], "kpi_tdis_tot")

    p("A Seção 5.1 argumenta que o nicho do aprendizado por reforço é o "
      "descasamento de modelo, e que um simulador de autoria própria não o "
      "exibe por construção. Esse argumento delimita todos os resultados "
      "reportados até aqui, e é testável: os mesmos controladores podem ser "
      "executados contra um emulador escrito por outra pessoa. Esta seção o "
      "faz, usando o arcabouço BOPTEST [Blum et al. 2021], o benchmark de "
      "referência para avaliação de controle predial, e seu caso bestest_air, "
      "uma zona única derivada do Caso 900 do BESTEST [Judkoff e Neymark 1995] "
      "com unidade fancoil, clima medido de Denver e ganhos internos e solares "
      "próprios.", first=1.25)

    rico([("O que não muda é a política. Os agentes são carregados dos mesmos "
           "arquivos avaliados em todo este artigo e executados ", 0, 0),
          ("sem retreinamento", 0, 0),
          (", de forma determinística, e a observação é montada pela mesma "
           "declaração única descrita na Seção 3.6, de modo que cada canal "
           "chega à política com o significado com que foi treinado. O que muda "
           "é a planta. Empregam-se dois períodos de resfriamento do caso, nos "
           "quais a zona sem climatização atinge 35,5 °C e 30,0 °C, "
           "respectivamente.", 0, 0)], first=1.25)

    rico([("Uma decisão de transferência precisa ser reportada antes dos "
           "resultados, porque ela é, por si, um achado. O protocolo deste "
           "artigo decide a cada 12 min, o que é benigno numa planta cuja plena "
           "carga move a sala 0,090 °C por passo. No emulador a plena carga "
           "move a zona ", 0, 0),
          ("8,8 °C nos mesmos 12 min", 0, 0),
          (", mais que o dobro da largura inteira da faixa de conforto. Manter "
           "os 12 min não preservaria o protocolo; transformaria o problema em "
           "liga-desliga puro para todos os controladores, e a comparação "
           "deixaria de discriminar. O intervalo de controle é, por isso, "
           "derivado de uma condição de projeto declarada antes de medir: a "
           "plena carga não deve atravessar mais que a meia-faixa de conforto "
           "em um intervalo de decisão, o que resulta em 180 s. É a mesma "
           "disciplina que a Seção 5.2 recomenda, e o número é uma medição, "
           "feita de fora, da ameaça declarada na Seção 5.3: a planta deste "
           "artigo é lenta, e seu equipamento pequeno, frente a uma zona real.",
           0, 0)], first=1.25)

    legenda(f"Tabela {T('boptest')}. Transferência para o caso bestest_air do "
            "BOPTEST. Agentes executados sem retreinamento; o controlador PI "
            "aparece duas vezes, com os ganhos sintonizados na planta local e "
            "com ganhos re-sintonizados dentro do emulador. Melhor valor por "
            "período em negrito.", acima=True)

    _rot = {"peak_cool_day": "Dia de pico de resfriamento",
            "typical_cool_day": "Dia típico de resfriamento"}
    _lb, _neg = [], []
    for _per in (_pk, _tp):
        _sub = BOPT[BOPT["periodo"] == _per]
        _melhor = _sub["comfort_wide_pct"].max()
        for _i, (_, _r) in enumerate(_sub.iterrows()):
            if abs(_r["comfort_wide_pct"] - _melhor) < 1e-9:
                _neg.append(len(_lb))
            _lb.append([_rot[_per] if _i == 0 else "", _r["controlador"],
                        v(_r["comfort_wide_pct"]), v(_r["comfort_narrow_pct"]),
                        v(_r["abs_dev_from_ideal"], 2),
                        v(_r["energy_kwh_day"], 2), v(_r["kpi_tdis_tot"], 2)])
    tabela(["Período", "Controlador", "Conf. [22,26] %", "Conf. [23,25] %",
            "|T−24| °C", "kWh/dia", "tdis_tot (Kh)"],
           _lb, negrito_linhas=tuple(_neg),
           larguras=[2.7, 4.3, 2.2, 2.2, 1.7, 1.5, 2.0])

    figura(salvar_fig(F.fig_boptest_transferencia(BOPT), "fig16_boptest.png"))
    legenda(f"Figura {Fg('boptest')}. Conforto sob transferência. A barra "
            "hachurada é o controlador PI carregando os ganhos sintonizados na "
            "planta local; a sólida é o mesmo controlador re-sintonizado dentro "
            "do emulador. A chave marca o que a re-sintonia recupera.")

    rico([("Lida apenas com os ganhos congelados, a tabela inverte o resultado "
           "central deste artigo: o melhor agente atinge ", 0, 0),
          (f"{v(_b[_pk]['melhor_agente_conf'])} % contra "
           f"{v(_b[_pk]['pi_congelado'])} % no dia de pico e "
           f"{v(_b[_tp]['melhor_agente_conf'])} % contra "
           f"{v(_b[_tp]['pi_congelado'])} % no dia típico", 0, 0),
          (f", vantagens de {v(_b[_pk]['vantagem_agente_sobre_pi_congelado_pp'])}"
           f" e {v(_b[_tp]['vantagem_agente_sobre_pi_congelado_pp'])} pontos "
           "percentuais para o aprendizado. Reportar esse número isolado seria "
           "o achado desta seção, e seria errado.", 0, 0)],
         space_before=8, first=1.25)

    rico([("A razão é o defeito que este artigo audita, cometido aqui contra o "
           "nosso próprio baseline. Os ganhos do PI foram sintonizados numa "
           "planta cuja constante de tempo é de 30 h; o emulador responde "
           "vários graus no mesmo intervalo. Cobrar de um controlador clássico "
           "desempenho sob uma dinâmica para a qual seus ganhos nunca foram "
           "ajustados é exatamente o que a Seção 4.2 identifica no manuscrito "
           "auditado, deslocado de um ambiente para outro. A Seção 4.7 já "
           "fixou a regra correta: ", 0, 0),
          ("quando a especificação muda, ambos os controladores são "
           "repreparados", 0, 0),
          (". Aqui muda a planta, e a mesma regra se aplica.", 0, 0)],
         first=1.25)

    figura(salvar_fig(F.fig_boptest_sintonia(BOPTS), "fig17_boptest_sintonia.png"))
    legenda(f"Figura {Fg('boptest_sintonia')}. Busca em grade dos ganhos do PI "
            "dentro do emulador, no dia de pico de resfriamento. Os ganhos "
            "herdados da planta local situam-se longe do ótimo desta.")

    _best = BOPTS.sort_values(["conf_larga", "conf_estreita"],
                              ascending=False).iloc[0]
    rico([(f"Uma busca em grade sobre {len(BOPTS)} combinações (Figura "
           f"{Fg('boptest_sintonia')}) recupera "
           f"{v(_b[_pk]['ganho_resintonia_pp'])} pontos percentuais no período "
           f"de sintonia, levando o controlador PI de "
           f"{v(_b[_pk]['pi_congelado'])} % a {v(_b[_pk]['pi_resintonizado'])} % "
           f"com Kp = {v(_best['kp'], 1)} e Ki = {v(_best['ki'], 2)}. Com ambos "
           "os lados repreparados, ", 0, 0),
          ("a ordenação da Seção 4.2 é restaurada", 0, 0),
          (f": o controlador PI lidera por {v(_b[_pk]['vantagem_pi_resintonizado_pp'])}"
           f" e {v(_b[_tp]['vantagem_pi_resintonizado_pp'])} pontos percentuais, "
           "e sua margem na faixa estreita é muito maior, "
           f"{v(_narrow_pi)} % contra {v(_narrow_ag)} % no dia de pico, com "
           f"{v(_kwh_pi, 2)} contra {v(_kwh_ag, 2)} kWh/dia. O dia típico de "
           "resfriamento é fora da amostra da sintonia, que foi feita apenas no "
           "dia de pico, e nele o controlador re-sintonizado atinge "
           f"{v(_b[_tp]['pi_resintonizado'])} % da janela ocupada dentro da "
           "faixa.", 0, 0)], space_before=8, first=1.25)

    rico([("Uma divergência precisa ser reportada, e não suavizada. Sob o "
           "indicador de desconforto térmico do próprio arcabouço, ", 0, 0),
          ("os agentes lideram", 0, 0),
          (f" ({v(_tdis_ag_pk, 2)} contra {v(_tdis_pi_pk, 2)} Kh no dia de pico "
           f"e {v(_tdis_ag_tp, 2)} contra {v(_tdis_pi_tp, 2)} Kh no dia "
           "típico), porque esse indicador é calculado contra os setpoints do "
           "caso, que incluem recuo noturno, e não contra a faixa de "
           "[22, 26] °C que este artigo controla. As duas famílias de métrica "
           "premiam, portanto, objetivos distintos, e o vencedor muda com a "
           "escolha. É a mesma lição do restante deste artigo, com a métrica no "
           "lugar do baseline.", 0, 0)], first=1.25)

    p("Cabem dois limites deste experimento. O fancoil do emulador não "
      "reproduz o pico de COP em carga parcial do equipamento local, de modo "
      "que o mecanismo da Seção 4.8 não é testável aqui e nenhum destes "
      "números fala sobre ele. E o emulador não publica contagem de ocupantes: "
      "o canal de ocupação é nominal, seguindo a janela ocupada da "
      "configuração, que é a semântica com que as políticas foram treinadas. O "
      "que a seção sustenta é estreito e, por isso, sólido, o resultado "
      "central sobrevive a uma planta que este trabalho não escreveu, desde "
      "que ambos os controladores recebam a mesma preparação.", first=1.25)


h("5. Discussão")

h("5.1. Por que o controle clássico domina nesta formulação", 2)

p("O resultado não decorre de sintonia fortuita nem de implementação deficiente "
  "dos agentes. Ele decorre da classe do problema. A planta modelada é "
  "monovariável, de primeira ordem, essencialmente linear na faixa de operação, "
  "com modelo conhecido de dois parâmetros e perturbação mensurável, e o "
  "objetivo é rastrear um setpoint ou manter uma faixa. Esse é o caso canônico "
  "em que controle proporcional-integral é ótimo ou quase-ótimo.", first=1.25)

p("O aprendizado por reforço possui vantagem estabelecida em regimes que essa "
  "formulação não exibe: não-linearidade forte, acoplamento de alta "
  "dimensionalidade, objetivos não expressáveis como custo quadrático, decisões "
  "combinatórias de alocação, ou modelo desconhecido ou variável. As três "
  "extensões avaliadas na Seção 4.6 introduziram dificuldade, mas não alteraram "
  "a classe do problema, um problema de rastreamento mais difícil continua "
  "sendo um problema de rastreamento, e responde a uma sintonia mais agressiva.",
  first=1.25)

rico([("Há ainda uma limitação de método considerada a mais relevante "
       "deste trabalho: ", 0, 0),
      ("o nicho do RL é o descasamento de modelo, e um simulador de autoria "
       "própria não exibe descasamento por construção", 0, 0),
      (". Qualquer comparação conduzida integralmente dentro do próprio "
       "ambiente favorece estruturalmente métodos que exploram o modelo, "
       "incluindo controle preditivo. Testar a hipótese de vantagem do RL exige "
       "transferência simulação-realidade ou benchmark de terceiros, como em "
       "Boutahri e Tilioua [2025] e Dai et al. [2025].", 0, 0)], first=1.25)

p("Há ainda um segundo mecanismo, específico ao espaço de ação discreto e "
  "identificado na Seção 4.8. Uma política determinística por maximização "
  "seleciona, em cada estado, uma única ação; quando várias ações têm valor "
  "quase equivalente, como aqui, em que a faixa entre as quatro opções vale "
  "cerca de 1,9 % do valor absoluto, aquela que nunca é máxima desaparece por "
  "completo. Um controlador proporcional, ao contrário, percorre naturalmente "
  "todos os níveis ao varrer a faixa de erro. A consequência é econômica: o "
  "nível de melhor coeficiente de performance deixa de ser utilizado, e o agente "
  "compensa com o nível menos eficiente.", first=1.25)

p("Esse achado tem implicação de projeto que transcende o caso estudado. Em "
  "sistemas cujo atuador possui poucos níveis discretos e cuja eficiência não é "
  "monotônica na carga, o que descreve boa parte dos equipamentos de "
  "climatização com inversor, a discretização do espaço de ação interage com a "
  "estrutura da política de modo a penalizar justamente o regime de operação "
  "mais eficiente. Conceder autoridade de atuação mais fina, por ação contínua "
  "ou por modulação de ciclo de trabalho, é intervenção mais promissora do que "
  "reponderar termos da função de recompensa.", first=1.25)

h("5.2. Implicações metodológicas", 2)

p("Três recomendações decorrem diretamente dos achados. Primeira, a "
  "configuração do baseline deve ser reportada com o mesmo detalhe da "
  "arquitetura do agente, e preferencialmente sintonizada com o mesmo esforço "
  f"computacional; a Tabela {T('decomposicao')} mostra que a diferença entre um baseline "
  "configurado e um não configurado pode exceder toda a contribuição "
  "reivindicada. Segunda, propriedades de segurança devem ser verificadas por "
  "sua distribuição, e não por agregados: a mesma política que exibe 2,5 "
  "comutações por hora apresenta mediana de permanência de 12 minutos contra "
  "requisito de 36. Terceira, constantes que definem restrições devem ser "
  "derivadas de condições de projeto, não arbitradas, uma restrição calibrada "
  "para produzir o resultado desejado é o mesmo defeito que este trabalho "
  "identifica no baseline auditado, deslocado para outro componente. Quarta, "
  "métricas saturadas devem ser detectadas e descartadas: no conjunto "
  "independente de 150 cenários, quatro controladores distintos produzem valor "
  "idêntico de conforto na faixa larga, porque o número é determinado pelo "
  "transitório de partida e não pela política. Uma métrica que não discrimina "
  "não deve ser a métrica principal de um artigo.",
  first=1.25)

h("5.3. Ameaças à validade", 2)

for item in [
    "Parâmetros não publicados pela especificação auditada. Quatro constantes "
    "(B, k, ρ e a penalidade de frio) tiveram de ser inferidas. A Seção 4.11 "
    "mitiga parcialmente essa ressalva: B e k passam a ser derivados de "
    "critério explícito, ρ mostra-se insensível ao valor, e a penalidade de "
    "frio tem sua escolha confirmada por medição. O achado da Seção 4.2 é "
    "internamente válido, ambos os controladores operam no mesmo ambiente, mas "
    f"sua transferência ao caso auditado é inferência, e não medição direta. A "
    f"Tabela {T('status')} distingue os achados por esse critério: os das "
    "Seções 4.4 e 4.5, por dependerem apenas de parâmetros publicados, não têm "
    "essa ressalva.",
    "Alcance da alegação auditada. A auditoria é um estudo de caso: o padrão "
    "documentado na Seção 2.2 para trabalhos publicados é medido aqui de ponta "
    "a ponta, com acesso integral ao ambiente, aos agentes e aos baselines, "
    "que uma auditoria conduzida de fora não teria. As recomendações da Seção "
    "5.2 são, por isso, formuladas sobre a literatura, e não sobre um "
    "manuscrito isolado.",
    "Sintonia e avaliação no mesmo conjunto. O controlador PI foi sintonizado "
    "sobre a mesma matriz 3×3 usada na avaliação. O resultado deve ser lido "
    "como equivalência nas condições em que ambos foram ajustados. Separação "
    "entre partição de calibração e de teste está implementada e sua execução "
    "constitui trabalho imediato.",
    "Número de sementes. Três sementes por condição impõem piso de p ≈ 0,101 em "
    "teste de permutação, razão pela qual se reporta o tamanho de efeito. "
    "Conclusões apoiadas em δ = −1,00 são robustas; as apoiadas em magnitudes "
    "pequenas são sugestivas.",
    "Inércia térmica elevada, a ameaça mais séria. A constante de tempo do "
    "ambiente, 30 h, excede a duração do episódio e equivale a cerca de setenta "
    "vezes o volume de ar de uma sala de aula convencional. Perturbações são "
    "fortemente amortecidas, o que facilita o controle e comprime as diferenças "
    "entre controladores; a saturação de métricas relatada nas Seções 4.9 e "
    "4.10 é a manifestação direta disso. A leitura conservadora é que a "
    "magnitude das diferenças aqui medidas não é transferível para uma sala "
    "real, e que o empate entre PI e DQN em faixa larga é, em parte, "
    "propriedade da parametrização e não do problema de climatização. Reexecutar "
    "o protocolo sob capacidade térmica fisicamente plausível é, por isso, o "
    "experimento de maior retorno entre os pendentes, à frente do aumento do "
    "número de sementes: medir com mais precisão um empate que a planta já "
    "favorece não altera a conclusão.",
    "Ausência de validação em hardware e de MPC como referência. Todos os "
    "resultados são de simulação, com balanço térmico agregado, sem gradientes "
    "espaciais, umidade ou CO₂, e com atuador instantâneo. Controle preditivo "
    "baseado em modelo, o adversário natural em regimes de alocação, não foi "
    "implementado.",
]:
    p("• " + item, first=0.5, space_after=4)

# ============================================================== 6. CONCLUSÃO
h("6. Conclusão")

p("Implementou-se, a partir de sua especificação, um controlador HVAC baseado "
  "em aprendizado por reforço profundo, auditando-se sua afirmação central sob "
  "baselines competentemente configurados. A reprodução mostrou-se fiel em "
  "física, conforto e generalização, e divergente em energia e comutações, "
  "assimetria que delimita cada afirmação subsequente. A vantagem reivindicada "
  "não sobrevive: um controlador PI com duas constantes iguala o DQN em conforto "
  "e o supera em desvio do setpoint e em custo. A decomposição atribui "
  "aproximadamente toda a vantagem à configuração inadequada do adversário.",
  first=1.25)

p("A ablação mostra que três dos cinco termos da recompensa proposta são "
  "inertes e que a formulação não supera uma quadrática convencional; a "
  "contribuição, corretamente enquadrada, é a correção de uma falha introduzida "
  "pela adoção do platô plano. A penalidade anti-short-cycling não cumpre seu "
  "objetivo, e uma restrição estrutural o cumpre a custo praticamente nulo. Em "
  "contrapartida, métodos tabulares são inadequados neste ambiente por razão "
  "teórica medida, 79,8% de transições intra-tile, de modo que, se RL for "
  "empregado aqui, ele precisa ser profundo. O critério que generaliza não é "
  "o número, e sim a razão entre o passo típico da dinâmica e a resolução da "
  "discretização, calculável antes de treinar qualquer agente.", first=1.25)

p("Investigou-se ainda se maior exigência de precisão reverteria o quadro, e o "
  "efeito medido é o oposto: com ambos os controladores repreparados para cada "
  "largura de faixa, a vantagem média do PI cresce conforme a especificação "
  "aperta, sendo nula em faixa larga, da ordem do ruído entre sementes em "
  "faixa intermediária e consistente entre sementes em faixa estreita. A "
  "avaliação do desempenho contra o orçamento de "
  "treinamento mostra que isso não se explica por treino insuficiente na faixa "
  "larga, onde o agente converge exatamente sobre a referência, ainda que na "
  "faixa estreita a curva permaneça em ascensão lenta ao fim do orçamento, o que "
  "caracteriza custo de amostra elevado em vez de impossibilidade.", first=1.25)

p("Identificou-se também o mecanismo do consumo excedente: os três perfis de "
  "recompensa descartam integralmente o nível de potência de melhor coeficiente "
  "de performance. A causa não está nos pesos, que variam por fatores de três a "
  "quatro entre os perfis sem alterar o comportamento, e sim na natureza "
  "determinística da política, que converte uma diferença de valor de cerca de "
  "1 % em ausência completa de uso. Um agente contínuo treinado com a recompensa "
  "idêntica utiliza aquele nível na mesma proporção que o controlador clássico.",
  first=1.25)

p("Examinou-se ainda se as constantes não publicadas da recompensa poderiam ser "
  "obtidas por argumento em lugar de inferência gráfica. Duas o admitem em "
  "análise: o bônus base, por ser constante aditiva sob episódio de duração "
  "fixa, não altera a política ótima, e a curvatura pode ser fixada impondo "
  "coerência entre o custo de sair da faixa e o de percorrê-la internamente. A "
  "verificação empírica, contudo, produziu resultado que contraria a expectativa "
  "e merece registro: sob orçamento reduzido os valores derivados elevam o "
  "conforto estreito em 16,7 pontos percentuais, mas sob o orçamento do "
  "protocolo o efeito inverte, com queda de 17,6 pontos. A configuração derivada "
  "converge mais rápido e depois degrada, enquanto a inferida aprende devagar e "
  "continua a melhorar. Disso decorre uma recomendação metodológica de alcance "
  "geral: calibrar parâmetros de recompensa sob orçamento reduzido e extrapolar "
  "para o orçamento completo é procedimento inválido. As duas constantes "
  "restantes não se mostraram melhoráveis por valor: a penalidade de comutação é "
  "insensível à magnitude, pois o defeito está na forma proporcional, e a "
  "penalidade de frio já se encontrava corretamente escolhida, uma vez que "
  "valores severos colapsam a política.", first=1.25)

p("Finalmente, documentou-se o insucesso de três tentativas de construir um "
  "regime favorável ao RL e identificou-se sua causa comum: introduzir "
  "dificuldade não altera a classe do problema. Conclui-se que, para "
  "climatização de ambiente único com modelo conhecido e objetivo de "
  "rastreamento, o aprendizado por reforço não é a ferramenta indicada.",
  first=1.25)

rico([("O alcance dessa conclusão precisa ser declarado sem ambiguidade. ", 0, 0),
      ("Estes resultados não constituem evidência de que o aprendizado por "
       "reforço profundo não possa superar o controle clássico em HVAC", 0, 0),
      (". Constituem evidência de que, nesta formulação e neste ambiente, ele "
       "não supera, e de que afirmações de superioridade exigem avaliação contra "
       "baselines competentes e em ambientes que exibam incerteza e descasamento "
       "de modelo em grau relevante. As duas ressalvas centrais operam na mesma "
       "direção e são declaradas na Seção 5.3: um simulador de autoria própria "
       "não exibe descasamento por construção, e a constante de tempo adotada "
       "torna a planta mais benigna do que uma sala real. O que este trabalho "
       "sustenta, portanto, é uma exigência metodológica, e não um veredito "
       "sobre a família de métodos.", 0, 0)], first=1.25)

p("Daí decorre a agenda. Como trabalho futuro imediato, apontamos a reexecução "
  "do protocolo sob capacidade térmica fisicamente plausível, a migração para um "
  "benchmark público de terceiros, BOPTEST ou BuildingGym, e a elevação do "
  "número de sementes nas comparações principais, nessa ordem de prioridade. A "
  "seguir, a investigação de regimes multi-zona com capacidade compartilhada, "
  "nos quais a decisão de alocação não admite lei de controle local, tendo, "
  "porém, o controle preditivo como referência obrigatória.", first=1.25)

p("Todo o material (ambiente, agentes, baselines, suíte de regressão e "
  f"registro completo de hiperparâmetros declarados e efetivos) está "
  f"disponível em {REPO} para "
  "replicação. Cada número reportado é produzido por uma única função, consumida "
  "tanto pelos cadernos de análise quanto pelo gerador deste documento: as "
  "figuras que dependem apenas do ambiente e dos modelos são recomputadas na "
  "mesma execução que produz as tabelas, e as demais leem os mesmos arquivos de "
  "resultados que as alimentam, de modo que divergência entre uma figura e a "
  "tabela ao lado é impossível por construção. A numeração de tabelas, figuras e "
  "equações é derivada da composição, e não escrita à mão. Os valores "
  "efetivamente plotados são gravados em arquivo próprio para auditoria.",
  first=1.25)

# ============================================================== REFERÊNCIAS
h("Referências")

refs = [
 "Agarwal, R. Schwarzer, M. Castro, P. S. Courville, A. e Bellemare, M. G. "
 "(2021). Deep Reinforcement Learning at the Edge of the Statistical "
 "Precipice. In Advances in Neural Information Processing Systems (NeurIPS).",

 "Al Sayed, K. Boodi, A. Sadeghian Broujeny, R. e Beddiar, K. (2024). "
 "Reinforcement learning for HVAC control in intelligent buildings: A "
 "technical and conceptual review. Journal of Building Engineering, 95:110085.",

 "ASHRAE (2020). ANSI/ASHRAE Standard 55-2020: Thermal Environmental "
 "Conditions for Human Occupancy. American Society of Heating, Refrigerating "
 "and Air-Conditioning Engineers, Atlanta.",

 "Åström, K. J. e Hägglund, T. (2006). Advanced PID Control. ISA, The "
 "Instrumentation, Systems and Automation Society, Research Triangle Park.",

 "Bellman, R. (1957). Dynamic Programming. Princeton University Press, "
 "Princeton.",

 "Blum, D. Arroyo, J. Huang, S. Drgona, J. Jorissen, F. Walnum, H. T. Chen, "
 "Y. Benne, K. Vrabie, D. Wetter, M. e Helsen, L. (2021). Building "
 "optimization testing framework (BOPTEST) for simulation-based benchmarking "
 "of control strategies in buildings. Journal of Building Performance "
 "Simulation, 14(5):586–610.",

 "Boutahri, Y. e Tilioua, A. (2025). Reinforcement learning for HVAC control "
 "and energy efficiency in residential buildings with BOPTEST simulations and "
 "real-case validation. Discover Computing, 28:44.",

 "Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal "
 "questions. Psychological Bulletin, 114(3):494–509.",

 "Dai, X. Chen, R. Guan, S. Li, W.-T. e Yuen, C. (2025). BuildingGym: An "
 "open-source toolbox for AI-based building energy management using "
 "reinforcement learning. arXiv:2509.11922.",

 "Fujimoto, S. van Hoof, H. e Meger, D. (2018). Addressing Function "
 "Approximation Error in Actor-Critic Methods. In Proceedings of the 35th "
 "International Conference on Machine Learning (ICML), pages 1587–1596.",

 "Judkoff, R. e Neymark, J. (1995). International Energy Agency Building "
 "Energy Simulation Test (BESTEST) and Diagnostic Method. Relatório técnico "
 "NREL/TP-472-6231, National Renewable Energy Laboratory, Golden.",

 "Haarnoja, T. Zhou, A. Abbeel, P. e Levine, S. (2018). Soft Actor-Critic: "
 "Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic "
 "Actor. In Proceedings of the 35th International Conference on Machine "
 "Learning (ICML), pages 1861–1870.",

 "Henderson, P. Islam, R. Bachman, P. Pineau, J. Precup, D. e Meger, D. "
 "(2018). Deep Reinforcement Learning that Matters. In Proceedings of the "
 "AAAI Conference on Artificial Intelligence, volume 32.",

 "Mnih, V. Kavukcuoglu, K. Silver, D. et al. (2015). Human-level control "
 "through deep reinforcement learning. Nature, 518(7540):529–533.",

 "Raffin, A. Hill, A. Gleave, A. Kanervisto, A. Ernestus, M. e Dormann, N. "
 "(2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. "
 "Journal of Machine Learning Research, 22(268):1–8.",

 "Sutton, R. S. e Barto, A. G. (2018). Reinforcement Learning: An "
 "Introduction. 2ª edição. MIT Press, Cambridge.",

 "Towers, M. Kwiatkowski, A. Terry, J. et al. (2024). Gymnasium: A Standard "
 "Interface for Reinforcement Learning Environments. arXiv:2407.17032.",

 "Xu, S. Fu, Y. Wang, Y. Yang, Z. Huang, C. O'Neill, Z. Wang, Z. e Zhu, "
 "Q. (2025). Efficient and assured reinforcement learning-based building HVAC "
 "control with heterogeneous expert-guided training. Scientific Reports, "
 "15:7414.",

 "Wei, T. Wang, Y. e Zhu, Q. (2017). Deep Reinforcement Learning for "
 "Building HVAC Control. In Proceedings of the 54th Annual Design Automation "
 "Conference (DAC), pages 1-6.",

 "Yuan, X. Pan, Y. Yang, J. Wang, W. e Huang, Z. (2020). Study on the "
 "application of reinforcement learning in the operation optimization of HVAC "
 "system. Building Simulation, 13.",

 "Zha, V. Chiu, I. Guilbault, A. e Tatis, J. (2021). Hyperspace Neighbor "
 "Penetration Approach to Dynamic Programming for Model-Based Reinforcement "
 "Learning Problems with Slowly Changing Variables in a Continuous State "
 "Space. arXiv:2106.05497.",
]
for r in refs:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.left_indent = Cm(0.75)
    par.paragraph_format.first_line_indent = Cm(-0.75)
    par.paragraph_format.space_after = Pt(5)
    run = par.add_run(r)
    run.font.size = Pt(11)

doc.save(SAIDA)
print("gerado:", SAIDA)
