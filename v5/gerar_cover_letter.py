# -*- coding: utf-8 -*-
"""
Gera a cover letter de submissão à Energy & Buildings.

POR QUE A CARTA TAMBÉM É GERADA. Ela cita números do artigo. Digitá-los à mão
cria a possibilidade de a carta afirmar 86,4 % onde a Tabela 7 diz 86,5 %, e o
editor lê os dois documentos lado a lado. Os valores vêm dos mesmos objetos que
alimentam o manuscrito, de modo que divergência entre carta e artigo é
impossível por construção, pela mesma razão que vale entre tabela e figura.

    python gerar_cover_letter.py
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

warnings.filterwarnings("ignore")

from hvac import results as R

RAIZ = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(RAIZ, "submissao_eb")
os.makedirs(DESTINO, exist_ok=True)
SAIDA = os.path.join(DESTINO, "cover_letter.docx")

# Mesma identificação do manuscrito, e pela mesma razão: é o único conteúdo
# que não vem nem dos dados nem do código.
AUTOR = "Renan Saraiva dos Santos"
AFILIACAO = "Instituto Federal da Paraíba, Campus Cajazeiras, Cajazeiras, Paraíba, Brazil"
EMAIL = "dossaraiva@gmail.com"
REPO = "https://github.com/renansaraivaifpb/fundamentals-of-reinforcement-learning"
ORCID = "0009-0006-7307-8361"
TITULO = ("When does deep reinforcement learning improve HVAC control? "
          "A reproducibility audit against competently tuned baselines")

# --------------------------------------------------------------------- dados
COMP = R.comparacao_controladores()
DEC = R.decomposicao_da_vantagem()
PERMA = R.permanencia_por_agente()
BRES = R.boptest_resumo()

_PI = COMP[COMP["controlador"] == "PI sintonizado"].iloc[0]
_DQN = COMP[COMP["controlador"].str.startswith("DQN")]
CONF = f"{_PI['comfort_wide_pct']:.1f}"
DEV_PI = f"{_PI['abs_dev_from_ideal']:.2f}"
DEV_DQN = (f"{_DQN['abs_dev_from_ideal'].min():.2f}–"
           f"{_DQN['abs_dev_from_ideal'].max():.2f}")
ENER_PI = f"{_PI['energy_kwh_day']:.2f}"
ENER_DQN = (f"{_DQN['energy_kwh_day'].min():.2f}–"
            f"{_DQN['energy_kwh_day'].max():.2f}")
GANHO = {r["atribuivel_a"]: f"{r['ganho_pp']:.1f}" for _, r in DEC.iterrows()}

# Números do experimento de transferência. Ausentes se o BOPTEST ainda não tiver
# sido executado, caso em que a carta simplesmente não menciona a validação
# externa — melhor omitir do que citar um número que o manuscrito não contém.
_PK = (BRES or {}).get("peak_cool_day")
_TP = (BRES or {}).get("typical_cool_day")

doc = Document()

sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15


def p(texto="", *, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
      size=11, space_after=8):
    par = doc.add_paragraph()
    par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    par.paragraph_format.line_spacing = 1.15
    run = par.add_run(texto)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return par


def item(texto):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.left_indent = Cm(0.7)
    par.paragraph_format.space_after = Pt(6)
    par.paragraph_format.line_spacing = 1.15
    # Marcador de lista. Já foi travessão, e a varredura que trocou
    # travessões por vírgulas no texto corrido o transformou numa vírgula
    # solta no início de cada item.
    run = par.add_run("\u2022  " + texto)
    run.font.size = Pt(11)
    return par


# ------------------------------------------------------------------- cabeçalho
p(AUTOR, bold=True, space_after=2)
p(AFILIACAO, space_after=2)
p(EMAIL, space_after=2)
p(f"ORCID: https://orcid.org/{ORCID}", space_after=16)

# A Elsevier aceita a carta sem data; deixá-la em branco evita que uma data
# antiga vá junto se a submissão for adiada.
p("Editor-in-Chief", bold=True, space_after=2)
p("Energy & Buildings", space_after=16)

p("Dear Editor,", space_after=10)

p(f"I wish to submit for consideration as an Original Research Article the "
  f"manuscript entitled \u201c{TITULO}\u201d.")

p("Reported gains of deep reinforcement learning (DRL) over conventional HVAC "
  "control depend critically on how the reference controller was configured, "
  "and that configuration is rarely reported with the care devoted to the "
  "agent. Of six works surveyed in the manuscript, only two include a tuned "
  "classical feedback controller, and where one is present the reported margin "
  "shrinks. This paper measures the step the literature mostly omits: a "
  "competently tuned proportional-integral (PI) controller, which sits between "
  "the rule-based baselines that DRL beats and the model predictive control it "
  "aspires to match.")

p("The main contributions are:", space_after=6)

for contribuicao in [
    f"A controlled decomposition of a claimed +32 percentage-point advantage, "
    f"attributing +{GANHO['configuração do baseline']} pp to adding hysteresis "
    f"to the baseline, +{GANHO['controle clássico']} pp to classical control "
    f"and +{GANHO['aprendizado']} pp to learning. A PI controller with two "
    f"constants matches the DQN on comfort ({CONF} % for both) at a lower "
    f"deviation from setpoint ({DEV_PI} against {DEV_DQN} °C) and lower energy "
    f"use ({ENER_PI} against {ENER_DQN} kWh/day).",

    "A mechanism, with a clean experimental control, for the excess energy of "
    "the discrete agents: a deterministic argmax policy turns an action-value "
    "margin of about 1 % into 0 % usage of the highest-COP power level, while "
    "a soft actor-critic agent on the identical reward uses that level as the "
    "PI does. The implication extends to any actuator whose efficiency is "
    "non-monotonic in load, which describes much inverter-equipped equipment.",

    *([f"External validation in BOPTEST. Transferred zero-shot to the "
       f"bestest_air case, the agents overtake a PI carrying the gains fitted "
       f"to the original plant by "
       f"{_PK['vantagem_agente_sobre_pi_congelado_pp']:.1f} and "
       f"{_TP['vantagem_agente_sobre_pi_congelado_pp']:.1f} percentage points, "
       f"and lose that lead once the PI is retuned "
       f"({_PK['pi_congelado']:.1f} % to {_PK['pi_resintonizado']:.1f} %, with "
       f"the second period out of sample). The ordering survives an "
       f"environment I did not write, and the experiment reproduced, against "
       f"my own baseline, the defect the paper audits."] if BRES else []),

    "Three methodological findings that transfer beyond the case studied: "
    "three of five reward terms are inert; an anti-short-cycling reward "
    "penalty fails its own objective where a hard dwell constraint meets it at "
    "no comfort cost; and reward constants reverse sign between a reduced and "
    "a full training budget, so that calibrating cheaply and extrapolating is "
    "invalid.",
]:
    item(contribuicao)

p("The principal result is negative: under this formulation, DRL offers no "
  "advantage over competently tuned classical control. The manuscript is "
  "explicit that this is not evidence that DRL cannot beat classical control "
  "in HVAC, and Section 5.3 states the caveats that operate against the "
  "finding. Well-controlled negative results are scarce in this literature "
  "and, I argue, necessary to calibrate it: the transferable claim is a "
  "requirement on method, not a verdict on a family of methods.")

p("The work fits the scope of Energy & Buildings in its concern with the "
  "energy performance of building control systems and with the rigour of the "
  "evidence behind control recommendations: it reports energy and cost, "
  "identifies a mechanism bearing on equipment efficiency at part load, and "
  "validates externally in the benchmark this community maintains for the "
  "purpose.")

p("The manuscript is original, has not been published previously and is not "
  "under consideration elsewhere. There is a single author, who has approved "
  "the submission, and no competing financial or personal interests to "
  "declare. The complete material (environment, agents, baselines and the "
  f"full record of hyperparameters) is available at {REPO}: every number "
  "and "
  "figure is produced by a single set of functions, so that tables, figures "
  "and analysis notebooks cannot disagree.")

p("Thank you for your consideration. I look forward to the reviewers' "
  "comments.", space_after=18)

p("Yours sincerely,", space_after=16)
p(AUTOR, bold=True, space_after=2)
p(AFILIACAO, space_after=2)
p(EMAIL, space_after=2)
p(f"ORCID: https://orcid.org/{ORCID}", space_after=2)

doc.save(SAIDA)
print(f"generated: {SAIDA}")
