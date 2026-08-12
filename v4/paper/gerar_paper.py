# -*- coding: utf-8 -*-
"""
Gera o paper em .docx no formato de congresso (padrão SBC/ENIAC).

Todos os números vêm das medições registradas no repositório:
README.md, REVISAO.md, results_ablation/, results_lab2.csv, results_tabular.csv
e a derivação de demanda_sizing executada nesta sessão.
"""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SAIDA = "/home/renan/Downloads/air_conditioning_classroom/v4/paper/paper_auditoria_hvac_rl.docx"

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


def p(texto="", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, bold=False,
      italic=False, space_before=0, space_after=6, indent=None, first=None):
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
    run = par.add_run(texto)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
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
        r = par.add_run(texto)
        r.font.size = Pt(size)
        r.bold = b
        r.italic = i
    return par


def h(texto, nivel=1):
    tam = {1: 13, 2: 12, 3: 12}[nivel]
    return p(texto, align=WD_ALIGN_PARAGRAPH.LEFT, size=tam, bold=True,
             space_before=12 if nivel == 1 else 10, space_after=6)


def eq(texto, numero=None):
    """Equação centralizada com numeração à direita."""
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(6)
    r = par.add_run(texto)
    r.font.size = Pt(12)
    r.italic = True
    if numero:
        rn = par.add_run(f"\t\t({numero})")
        rn.font.size = Pt(12)
    return par


def legenda(texto, *, acima=False):
    return p(texto, align=WD_ALIGN_PARAGRAPH.CENTER, size=10,
             space_before=8 if acima else 4, space_after=8 if not acima else 4)


def figura(caminho, largura_cm=14.4):
    """Insere a figura centralizada. Ver make_figures.py — mesmas fontes das tabelas."""
    doc.add_picture(caminho, width=Cm(largura_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(8)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)


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
p("Controle de HVAC por Aprendizado por Reforço Profundo: uma Auditoria "
  "de Reprodutibilidade com Baselines Competentes",
  align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, space_after=14)

p("Renan Saraiva¹", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=2)
p("¹Instituição — a preencher", align=WD_ALIGN_PARAGRAPH.CENTER, size=11,
  space_after=2)
p("rsaraiva@prysmo.com", align=WD_ALIGN_PARAGRAPH.CENTER, size=11,
  space_after=14)

# ==================================================================== ABSTRACT
p("Abstract.", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, bold=True,
  space_after=0, indent=1.0)
p("Deep reinforcement learning (DRL) is widely reported to outperform "
  "conventional controllers in HVAC applications. We conduct an independent "
  "reproduction of a DRL classroom HVAC controller and audit its central claim "
  "under competently configured baselines. The reproduction is faithful in "
  "physics and comfort metrics, yet the reported advantage does not survive: a "
  "grid-tuned PI controller with two constants matches the DQN on binary "
  "comfort (86.5% both), attains a lower mean absolute deviation from setpoint "
  "(0.72 °C vs 0.77–0.89 °C) and lower cost (R$ 9.47 vs 9.83–10.72/day). "
  "Decomposing the claimed +32 pp advantage attributes +23.7 pp to adding "
  "hysteresis to the baseline, +8.8 pp to classical control, and +0.0 pp to "
  "learning. A reward ablation over 8 variants × 3 seeds shows three of five "
  "reward terms are inert, and that the proposed formulation does not "
  "outperform a plain quadratic (Cliff's δ = −0.11, negligible). The "
  "anti-short-cycling penalty fails its own objective (71–83% dwell violations; "
  "median dwell of 12 min against a 36 min requirement), whereas a hard "
  "constraint reduces violations to 6.4% at zero comfort cost. Conversely, "
  "tabular Q-learning collapses (42.1% comfort, below a deadband thermostat) "
  "because 79.8% of transitions are intra-tile — so if RL is used, it must be "
  "deep. We further show that three attempts to construct a regime favouring "
  "RL — precision tracking, tariff anticipation and contracted demand — fail "
  "for structurally distinct reasons, and argue that a self-authored simulator "
  "cannot, in principle, exhibit the model-mismatch regime where RL holds an "
  "advantage.",
  align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, italic=True, indent=1.0,
  space_after=10)

p("Resumo.", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, bold=True,
  space_after=0, indent=1.0)
p("O aprendizado por reforço profundo (DRL) é amplamente reportado como "
  "superior a controladores convencionais em aplicações de HVAC. Realizamos uma "
  "reprodução independente de um controlador DRL para climatização de salas de "
  "aula e auditamos sua afirmação central sob baselines competentemente "
  "configurados. A reprodução é fiel na física e nas métricas de conforto, mas "
  "a vantagem reportada não sobrevive: um controlador PI sintonizado por busca "
  "em grade, com duas constantes, iguala o DQN em conforto binário (86,5% para "
  "ambos), obtém menor desvio absoluto do setpoint (0,72 °C contra 0,77–0,89 °C) "
  "e menor custo (R$ 9,47 contra 9,83–10,72/dia). A decomposição dos +32 pp "
  "reivindicados atribui +23,7 pp à inclusão de histerese no baseline, +8,8 pp "
  "ao controle clássico e +0,0 pp ao aprendizado. Uma ablação da recompensa "
  "sobre 8 variantes × 3 sementes mostra que três dos cinco termos são inertes e "
  "que a formulação proposta não supera uma quadrática simples (δ de Cliff = "
  "−0,11, desprezível). A penalidade anti-short-cycling falha em seu próprio "
  "objetivo (71–83% de violações; mediana de permanência de 12 min contra "
  "requisito de 36 min), enquanto uma restrição dura reduz as violações a 6,4% "
  "sem custo de conforto. Em contrapartida, o Q-Learning tabular colapsa (42,1% "
  "de conforto, abaixo de um termostato com zona morta) porque 79,8% das "
  "transições são intra-tile — ou seja, se RL for usado, ele precisa ser "
  "profundo. Mostramos ainda que três tentativas de construir um regime "
  "favorável ao RL falham por razões estruturalmente distintas, e argumentamos "
  "que um simulador de autoria própria não pode, em princípio, exibir o regime "
  "de descasamento de modelo em que o RL possui vantagem.",
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
  "zona morta nula, é um adversário artificialmente incapaz — e comparações "
  "contra ele produzem vantagens que não se sustentam frente a um controlador "
  "clássico competentemente sintonizado.", first=1.25)

rico([("Este trabalho realiza uma ", 0, 0),
      ("auditoria de reprodutibilidade", 1, 0),
      (" de um controlador DRL para climatização de salas de aula. O código "
       "original do trabalho auditado foi perdido; a reimplementação foi "
       "construída a partir do texto do manuscrito, o que constitui ao mesmo "
       "tempo a principal limitação e a principal contribuição metodológica "
       "deste artigo — obriga a explicitar cada parâmetro e a derivar, de "
       "primeiros princípios, tudo o que o texto não publica.", 0, 0)],
     first=1.25)

p("As contribuições são as seguintes:", space_before=4)

for item in [
    "(i) demonstramos que a vantagem reivindicada de +32 pontos percentuais "
    "sobre o baseline é praticamente toda atribuível à configuração inadequada "
    "do adversário, e não ao aprendizado;",
    "(ii) quantificamos, por ablação controlada sobre 8 variantes e 3 sementes, "
    "que três dos cinco termos da função de recompensa proposta são inertes, e "
    "que a formulação não supera uma quadrática convencional;",
    "(iii) mostramos que a penalidade anti-short-cycling — destacada como "
    "contribuição — não cumpre seu objetivo, e que uma restrição estrutural o "
    "cumpre a custo praticamente nulo;",
    "(iv) fornecemos justificativa teórica e medida para o abandono de métodos "
    "tabulares neste domínio, via a fração de transições intra-tile;",
    "(v) reportamos o insucesso de três tentativas sucessivas de construir um "
    "regime experimental favorável ao RL, e analisamos a razão estrutural comum "
    "a elas.",
]:
    p(item, first=0.75, space_after=3)

p("O resultado agregado é negativo quanto à necessidade de RL nesta formulação "
  "do problema, e positivo quanto ao rigor exigido para afirmá-la. Resultados "
  "negativos bem controlados são escassos nesta literatura e, argumentamos, "
  "necessários para calibrá-la.", space_before=4, first=1.25)

# ====================================================== 2. TRABALHOS RELACIONADOS
h("2. Trabalhos Relacionados")

p("Al Sayed et al. [2024] apresentam revisão técnica e conceitual de RL para "
  "controle de HVAC em edificações inteligentes, sistematizando formulações de "
  "estado, ação e recompensa. A revisão evidencia a heterogeneidade dos "
  "protocolos de avaliação e a ausência de baselines padronizados — precisamente "
  "a lacuna que este trabalho explora.", first=1.25)

p("Boutahri e Tilioua [2025] utilizam a plataforma BOPTEST para validação "
  "sistemática e reportam até 26,3% de economia energética sobre controladores "
  "PI, com validação experimental complementar em edificação residencial real. "
  "O trabalho é relevante por duas razões: adota um benchmark padronizado, o que "
  "elimina o grau de liberdade de construir o próprio ambiente, e realiza a "
  "transferência simulação-realidade, regime em que o RL possui vantagem "
  "plausível sobre métodos baseados em modelo.", first=1.25)

p("Xu et al. [2025] atacam o custo de treinamento com destilação de "
  "conhecimento especialista, reportando aceleração de até 8,8× e introduzindo "
  "um arcabouço de shielding em tempo de execução para reduzir a taxa de "
  "violação de temperatura. A separação entre o que é otimizado por recompensa e "
  "o que é imposto estruturalmente é diretamente aplicável ao nosso achado sobre "
  "anti-short-cycling.", first=1.25)

p("Zha et al. [2021] identificam a falha de métodos tabulares com discretização "
  "em espaços contínuos com variáveis de dinâmica lenta: a transição permanece "
  "no mesmo hiper-tile de origem, de modo que o valor não se propaga entre "
  "tiles. Os autores propõem o Hyperspace Neighbor Penetration (HNP). "
  "Utilizamos o diagnóstico — não o método — para justificar quantitativamente a "
  "inadequação do Q-Learning tabular neste domínio.", first=1.25)

p("Dai et al. [2025] disponibilizam o BuildingGym, arcabouço aberto com "
  "EnergyPlus como simulador central, reforçando a tendência de padronização de "
  "ambientes. Do ponto de vista metodológico, Henderson et al. [2018] e Agarwal "
  "et al. [2021] documentam a fragilidade estatística de comparações em RL "
  "profundo com poucas sementes, motivando o uso de tamanho de efeito não "
  "paramétrico em vez de valores-p isolados, prática que adotamos.", first=1.25)

# ============================================================ 3. METODOLOGIA
h("3. Materiais e Métodos")

h("3.1. Ambiente de simulação", 2)

p("O ambiente modela uma sala de aula com capacidade para 45 ocupantes, "
  "implementado com a interface Gymnasium [Towers et al. 2024]. A dinâmica "
  "térmica é agregada (lumped), tratando a sala como nó único:", first=1.25)

eq("T(t+Δt) = T(t) + (Δt / C_th) · [ N·q_p + K·(T_ext(t) − T(t)) − Q_ac(a) ] + ε",
   numero=1)

p("em que C_th é a capacidade térmica (15,0 na sala de treino), N o número de "
  "ocupantes, q_p = 0,3 u o ganho térmico por pessoa, K = 0,5 o coeficiente de "
  "transferência com o exterior, Q_ac a potência de resfriamento e ε ruído "
  "gaussiano de processo (σ = 0,01). O passo é Δt = 0,1 h (6 min) e o episódio "
  "cobre 24 h (240 passos). A temperatura externa segue senoide diária com pico "
  "às 14 h, base 28 °C e amplitude 8 °C.", first=1.25)

rico([("A Tabela 1 do manuscrito auditado é ", 0, 0), ("derivada", 1, 0),
      (", e não tabelada, a partir de dois parâmetros de catálogo: capacidade "
       "de 30.000 BTU/h e coeficiente de performance (COP) por nível. Essa "
       "escolha torna o modelo auditável e permite trocar de equipamento sem "
       "reescrever a tabela. A derivação reproduz os valores publicados com "
       "erro inferior a 0,5%.", 0, 0)], first=1.25)

legenda("Tabela 1. Modelo do equipamento, derivado de capacidade nominal e COP.",
        acima=True)
tabela(["Nível", "Fração de carga", "COP", "Resfriamento (u)",
        "Elétrica (kW) — derivada", "Publicada"],
       [["OFF", "0,00", "—", "0", "0,000", "—"],
        ["LOW", "0,25", "3,45", "10", "0,637", "0,64"],
        ["MEDIUM", "0,55", "3,59", "22", "1,347", "1,35"],
        ["HIGH", "1,00", "3,00", "40", "2,931", "2,93"]],
       larguras=[2.0, 2.6, 1.6, 2.6, 3.6, 2.2])

p("O pico de COP em carga parcial (MEDIUM) reproduz comportamento de "
  "equipamento inverter e é o que cria o compromisso relevante: operar em "
  "MEDIUM é energeticamente barato, o que favorece o termostato em custo mesmo "
  "quando ele perde em conforto.", space_before=8, first=1.25)

h("3.2. Formulação como processo de decisão markoviano", 2)

p("O estado observado é normalizado conforme a Equação 2, com quatro "
  "componentes: temperatura, ocupação e codificação circular da hora do dia.",
  first=1.25)

eq("s = [ clip((T−15)/20, 0, 1),  N/N_max,  sen(2πh/24),  cos(2πh/24) ]",
   numero=2)

p("O espaço de ação é discreto com quatro níveis (OFF, LOW, MEDIUM, HIGH) para "
  "os agentes DQN, e contínuo em [0,1] para SAC. A recompensa agrega cinco "
  "termos:", first=1.25)

eq("R = R_conforto + R_energia + R_mudança + R_frio + R_ciclo", numero=3)

p("O termo de conforto adota a topologia denominada Platô Quadrático com "
  "gradiente interno (Equação 4), em que B é o bônus base, B_c o gradiente "
  "interno, k a curvatura fora da faixa e [T_min, T_max] = [22, 26] °C:",
  first=1.25)

eq("R_conforto = B + B_c·(1 − |T − 24|/2),  se 22 ≤ T ≤ 26;  "
   "B − k·(T − T_lim)²,  caso contrário", numero=4)

p("A penalidade anti-short-cycling (Equação 5) pune comutações antes do tempo "
  "mínimo de permanência d_min = 36 min (6 passos), proporcionalmente à "
  "antecipação:", first=1.25)

eq("R_ciclo = ρ · (d_min − d) / d_min,  se houve troca e d < d_min", numero=5)

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
    "Termostato com zona morta nula — o baseline do manuscrito auditado. Liga "
    "em MEDIUM acima de 26 °C e em HIGH acima de 28 °C, desligando ao retornar "
    "à faixa.",
    "Termostato com histerese de 1 °C — a mesma lógica, com zona morta, "
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
  "excluído — resultado idêntico ao do manuscrito auditado.", first=1.25)

p("As métricas são computadas exclusivamente sobre a janela ocupada (7h–22h): "
  "percentual de tempo na faixa larga [22, 26] °C e na faixa estreita "
  "[23, 25] °C; desvio absoluto médio do ideal |T − 24|; percentual de "
  "sobreaquecimento; consumo (kWh/dia); custo (R$/dia); e comutações por hora. "
  "Adicionalmente, reportamos a distribuição dos tempos de permanência entre "
  "comutações, que é a métrica correta para avaliar proteção contra "
  "short-cycling.", first=1.25)

p("Os agentes foram treinados por 550.000 passos com Stable-Baselines3 [Raffin "
  "et al. 2021], taxa de aprendizado 5·10⁻⁵, batch 64 e action repeat 2. Os "
  "algoritmos cobertos são DQN [Mnih et al. 2015], SAC [Haarnoja et al. 2018] e "
  "TD3 [Fujimoto et al. 2018], além de Q-Learning tabular [Sutton e Barto 2018] "
  "para o diagnóstico da Seção 4.5.", first=1.25)

h("3.5. Análise estatística", 2)

p("Com três sementes por condição, o menor valor-p bicaudal alcançável em teste "
  "de permutação é 2/C(6,3) = 0,10; portanto p ≈ 0,101 é o piso, e não um "
  "resultado marginal. Reportamos por isso o δ de Cliff [Cliff 1993], tamanho de "
  "efeito não paramétrico, junto de intervalos de confiança bootstrap "
  "percentílicos. Um valor δ = −1,00 indica separação completa entre grupos — "
  "toda semente da condição ablacionada é pior que toda semente da referência. "
  "Essa escolha segue as recomendações de Henderson et al. [2018] e Agarwal et "
  "al. [2021].", first=1.25)

# ============================================================== 4. RESULTADOS
h("4. Resultados")

h("4.1. Fidelidade da reprodução", 2)

p("Antes de qualquer comparação, é necessário estabelecer que a reimplementação "
  "reproduz o comportamento descrito. As métricas de conforto do baseline "
  "termostático fecham dentro de aproximadamente 3 pontos percentuais dos "
  "valores publicados (54,0% contra 50,9% na faixa larga; 2,10 °C contra "
  "2,21 °C de desvio absoluto), o que sustenta que a física está correta. O "
  "consumo e as comutações ficam abaixo dos publicados (−31% e −61%, "
  "respectivamente) porque o manuscrito não publica os limiares da zona morta; "
  "ajustá-los aos resultados constituiria reprodução circular e não foi feito.",
  first=1.25)

p("O teste de generalização sem retreino (Tabela 2) reproduz a assinatura "
  "qualitativa do manuscrito: a vantagem sobre o termostato mantém-se na faixa "
  "de +24,8 a +35,0 pp, contra +21,7 a +38,5 pp publicados, com a mesma "
  "ordenação — pior degradação em sala de alta inércia e menor vantagem em sala "
  "bem isolada.", first=1.25)

legenda("Tabela 2. Generalização sem retreino: conforto na faixa larga (%). "
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

p("Estabelecida a fidelidade, passamos à auditoria da afirmação central.",
  space_before=8, first=1.25)

h("4.2. A vantagem reivindicada não sobrevive a um baseline competente", 2)

p("A Tabela 3 compara os três perfis de recompensa do manuscrito contra os "
  "controladores de referência, todos avaliados sob condições idênticas.",
  first=1.25)

legenda("Tabela 3. Comparação sob condições idênticas (matriz 3×3, janela "
        "ocupada). Melhores valores em negrito.", acima=True)
tabela(["Controlador", "Conf. [22,26] %", "Conf. [23,25] %", "|T−24| °C",
        "Sobreaq. %", "Energia kWh/dia", "Custo R$/dia", "Trocas/h"],
       [["PI sintonizado (K_p=1,3; K_i=0,2)", "86,5", "83,8", "0,72", "5,2",
         "10,39", "9,47", "1,04"],
        ["DQN Agressivo (550k)", "86,5", "83,8", "0,77", "5,2", "11,66",
         "10,72", "1,22"],
        ["DQN Equilibrado (550k)", "86,5", "83,8", "0,89", "5,2", "10,90",
         "9,90", "0,78"],
        ["DQN Passivo (550k)", "86,5", "83,7", "0,87", "5,2", "10,79",
         "9,83", "0,79"],
        ["Termostato, zona morta = 1 °C", "77,7", "20,5", "1,78", "14,0",
         "8,12", "7,63", "0,20"],
        ["Termostato, zona morta = 0 (baseline)", "54,0", "12,2", "2,10",
         "37,7", "7,55", "7,11", "0,97"],
        ["Q-Learning tabular (550k)", "42,1", "27,3", "3,23", "—", "—",
         "—", "—"]],
       negrito_linhas=(0,),
       larguras=[5.2, 2.0, 2.0, 1.7, 1.6, 1.9, 1.6, 1.4])

rico([("O controlador PI domina os três perfis DQN em todas as métricas de "
       "qualidade e de custo simultaneamente: mesmo conforto binário, menor "
       "desvio absoluto do ideal e menor consumo. ", 0, 0),
      ("Duas constantes sintonizadas igualam ou superam 550.000 passos de "
       "treinamento.", 1, 0)],
     space_before=8, first=1.25)

p("A Tabela 4 decompõe a vantagem reivindicada, atribuindo cada incremento à "
  "sua causa.", first=1.25)

legenda("Tabela 4. Decomposição da vantagem de +32 pp reivindicada.", acima=True)
tabela(["Etapa", "Conforto faixa larga (%)", "Ganho", "Atribuível a"],
       [["Termostato do manuscrito (zona morta = 0)", "54,0", "—", "—"],
        ["Adicionar apenas histerese", "77,7", "+23,7 pp",
         "configuração do baseline"],
        ["PI sintonizado", "86,5", "+8,8 pp", "controle clássico"],
        ["DQN, 550k passos", "86,5", "+0,0 pp", "aprendizado"]],
       negrito_linhas=(3,),
       larguras=[6.4, 3.2, 2.2, 4.2])

figura("figuras_paper/fig1_decomposicao.png")
legenda("Figura 1. Decomposição da vantagem reivindicada. Cada coluna acrescenta "
        "um único fator ao anterior. O último degrau — o aprendizado — é nulo.")

p("Aproximadamente 100% da vantagem reivindicada é atribuível à configuração "
  "inadequada do adversário (Figura 1). A afirmação de 82,9% contra 50,9% é "
  "tecnicamente verdadeira e substantivamente enganosa: compara-se contra um "
  "controlador artificialmente incapaz.", space_before=8, first=1.25)

h("4.3. Ablação da função de recompensa", 2)

p("A tese de que a modelagem da recompensa é mais determinante que o algoritmo "
  "exige ablação controlada. Foram treinadas oito variantes, cada uma "
  "desativando exatamente um mecanismo, sobre três sementes — 24 treinamentos "
  "de 550.000 passos. A Tabela 5 reporta o conforto na faixa estreita, métrica "
  "que discrimina os regimes.", first=1.25)

legenda("Tabela 5. Ablação da recompensa: conforto na faixa estreita [23,25] °C.",
        acima=True)
tabela(["Variante", "Média", "IC 95%", "Por semente", "δ de Cliff", "Magnitude"],
       [["completa (proposta)", "83,7", "[83,3; 83,8]", "83,8 / 83,3 / 83,8",
         "—", "referência"],
        ["convencional (quadrática pura)", "83,1", "[81,7; 83,8]",
         "83,8 / 81,7 / 83,8", "−0,11", "desprezível"],
        ["sem anti-short-cycling", "83,7", "[83,3; 83,8]", "83,8 / 83,3 / 83,8",
         "0,00", "desprezível"],
        ["sem penalidade de frio", "82,6", "[81,8; 83,8]", "81,8 / 82,0 / 83,8",
         "−0,56", "pequena"],
        ["sem penalidade de troca", "82,1", "[78,5; 83,8]", "83,8 / 78,5 / 83,8",
         "−0,11", "desprezível"],
        ["quadrática pura", "82,3", "[80,7; 83,8]", "83,8 / 80,7 / 82,5",
         "−0,56", "pequena"],
        ["degraus (legado)", "69,1", "[44,0; 82,2]", "81,2 / 82,2 / 44,0",
         "−1,00", "grande"],
        ["sem gradiente interno", "47,7", "[34,0; 61,3]", "61,3 / 34,0 / 47,8",
         "−1,00", "grande"]],
       negrito_linhas=(0, 7),
       larguras=[4.6, 1.5, 2.5, 3.7, 1.9, 2.0])

figura("figuras_paper/fig2_ablacao.png")
legenda("Figura 2. Ablação da recompensa. As três sementes de cada variante são "
        "exibidas individualmente: com n = 3, a média isolada esconderia que "
        "degraus (legado) tem uma semente em 44,0 e duas acima de 81.")

p("Três conclusões emergem (Figura 2). Primeira, a hipótese do platô plano é confirmada: "
  "remover o gradiente interno degrada o conforto estreito de 83,7% para 47,7% "
  "(−36 pp), eleva o desvio de 0,85 para 1,46 °C e o sobreaquecimento de 5,2% "
  "para 13,8%, com separação completa entre sementes (δ = −1,00). Um platô "
  "verdadeiramente plano não recompensa entrar na faixa, e a política estaciona "
  "junto à borda.", space_before=8, first=1.25)

rico([("Segunda, e contrariando a tese do manuscrito: a formulação proposta ",
       0, 0), ("não supera", 1, 0),
      (" a convencional. A variante que usa quadrática pura, sem gradiente "
       "interno, sem penalidade de troca, sem anti-short-cycling e sem "
       "penalidade de frio, atinge 83,1% contra 83,7% da proposta "
       "(δ = −0,11, desprezível). A razão é conceitual: uma quadrática pura é "
       "uma parábola com máximo em 24 °C e, portanto, já possui gradiente em "
       "todo o domínio por construção. O platô plano é que o destrói; a "
       "proposta apenas o restaura. A contribuição, corretamente enquadrada, "
       "é a correção de uma falha introduzida pelos próprios autores — um "
       "achado negativo, não positivo.", 0, 0)], first=1.25)

p("Terceira, três dos cinco termos são inertes: o anti-short-cycling tem efeito "
  "exatamente nulo (δ = 0,00), a penalidade de troca é desprezível (δ = −0,11) "
  "e a penalidade de frio tem magnitude pequena. Indício adicional, com "
  "intervalos sobrepostos e portanto sugestivo: remover o anti-short-cycling "
  "reduz o consumo em 5% (10,53 contra 11,09 kWh/dia) sem custo de conforto, o "
  "que sugere que o termo é ativamente prejudicial.", first=1.25)

h("4.4. A penalidade anti-short-cycling não cumpre seu objetivo", 2)

p("Comutações frequentes do compressor reduzem a vida útil do equipamento; o "
  "manuscrito endereça isso via a Equação 5, com d_min = 36 min. A métrica "
  "agregada de comutações por hora, contudo, não permite verificar a proteção. "
  "A distribuição dos tempos de permanência, sim (Tabela 6 e Figura 3).",
  first=1.25)

legenda("Tabela 6. Distribuição dos tempos de permanência entre comutações "
        "(requisito: d_min = 36 min).", acima=True)
tabela(["Agente", "Mediana", "Média", "Mínimo", "Violações de d_min"],
       [["DQN Agressivo", "12 min", "36,0 min", "12 min", "83,3%"],
        ["DQN Equilibrado", "12 min", "59,8 min", "12 min", "71,4%"],
        ["DQN Passivo", "12 min", "57,7 min", "12 min", "72,9%"],
        ["Termostato", "12 min", "46,2 min", "12 min", "79,0%"]],
       larguras=[4.4, 2.4, 2.4, 2.2, 3.4])

figura("figuras_paper/fig3_permanencia.png")
legenda("Figura 3. Distribuição acumulada dos tempos de permanência do DQN "
        "Equilibrado. O ponto destacado sobre a linha tracejada marca a fração "
        "de comutações abaixo do requisito: 71,4% sob a penalidade de "
        "recompensa, 6,4% sob restrição dura.")

p("A mediana de 12 minutos corresponde a uma única decisão. A média de 36 a 60 "
  "minutos é inflada por longos períodos com o equipamento desligado e mascara "
  "completamente o problema — caso didático de agregado que oculta a "
  "distribuição.", space_before=8, first=1.25)

rico([("A causa é aritmética: a penalidade da Equação 5 vale no máximo "
       "|ρ| = 5, contra ganhos de conforto de até B + B_c = 14. ", 0, 0),
      ("Um termo de recompensa que pode ser superado por outro termo não é uma "
       "proteção; é uma sugestão.", 1, 0)], first=1.25)

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
  "transição é intra-tile e o valor nunca se propaga entre células. Medimos "
  "diretamente essa fração, sem treinar nenhum agente (Tabela 7).", first=1.25)

legenda("Tabela 7. Diagnóstico de discretização.", acima=True)
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
  "Q-Learning tabular treinado por 550.000 passos e três sementes atinge 42,1% "
  "de conforto na faixa larga (31,5 / 38,7 / 56,0 por semente, desvio 12,6) — "
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

h("4.6. Três tentativas de construir um regime favorável ao RL", 2)

p("Estabelecido que o controle clássico domina na formulação original, "
  "investigamos se alguma extensão do problema reverteria o resultado. Três "
  "regimes foram construídos e avaliados.", first=1.25)

rico([("Rastreamento de precisão. ", 1, 0),
      ("Um laboratório com tolerância de ±0,5 °C em torno do setpoint, com "
       "equipamento reversível, ação contínua e observação enriquecida com "
       "erro escalado, integral com fuga e derivada. Agentes TD3 e SAC foram "
       "treinados sob três perfis de custo. O PI bidirecional permanece "
       "superior (Tabela 8 e Figura 4).", 0, 0)], first=1.25)

legenda("Tabela 8. Regime de precisão: percentual do tempo dentro da tolerância "
        "de ±0,5 °C em regime permanente.", acima=True)
tabela(["Controlador", "Na tolerância %", "σ (°C)", "|T−24| °C",
        "Custo R$/dia", "kWh/dia", "Ajustes/h"],
       [["PI bidirecional", "100,0", "0,088", "0,069", "14,76", "17,17", "0,56"],
        ["TD3 Lab_Equilibrado", "100,0", "0,198", "0,163", "16,44", "19,26", "2,24"],
        ["TD3 Lab_Precisão", "97,3", "0,186", "0,138", "18,14", "21,37", "1,21"],
        ["SAC Lab_Precisão", "94,2", "0,268", "0,229", "16,40", "18,79", "1,86"],
        ["SAC Lab_Equilibrado", "92,0", "0,225", "0,230", "16,90", "20,18", "2,31"],
        ["TD3 Lab_Econômico", "86,7", "0,293", "0,222", "15,50", "18,59", "1,69"]],
       negrito_linhas=(0,),
       larguras=[4.6, 2.4, 1.7, 1.9, 2.2, 1.8, 1.9])

figura("figuras_paper/fig4_precisao.png")
legenda("Figura 4. Regime de precisão: custo diário contra dispersão da "
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
       "a média — grandeza que realimentação puramente reativa não representa, "
       "e cujo atendimento exigiria antecipação. Uma versão preliminar deste "
       "experimento fixava a demanda contratada em 0,70 kW, valor escolhido por "
       "varredura como ponto de máxima tensão entre conforto e restrição. "
       "Auditando esse valor sob a mesma exigência aplicada ao manuscrito, "
       "constatou-se que ele era ", 0, 0), ("infactível", 1, 0),
      (": o regime permanente de pior caso exige 1,204 kW, de modo que 0,70 kW "
       "está 72% abaixo do necessário e sustenta apenas cerca de 17 dos 45 "
       "ocupantes. Nenhuma política respeita tal restrição, pois antecipação "
       "não cria regime permanente.", 0, 0)], first=1.25)

p("Substituímos o valor arbitrado por dimensionamento derivado da condição de "
  "projeto — ocupação máxima, externa no pico diário, mantendo o setpoint — "
  "acrescido de margem de contratação de 10%, resultando em 1,324 kW. Sob esse "
  "contrato, nenhum dos nove cenários excede o limite (pico máximo observado de "
  "1,213 kW), e o PI ingênuo torna-se indistinguível do PI ciente da restrição.",
  first=1.25)

p("A razão é uma contradição estrutural: os únicos cenários que excedem o "
  "contrato derivado são os que partem fora do setpoint, nos quais a violação "
  "ocorre no passo inicial — instante em que antecipação é impossível por "
  "definição. Ao construir a matriz de cenários partindo do setpoint para "
  "eliminar essa violação impossível, elimina-se simultaneamente a única fonte "
  "de demanda acima do regime permanente. O conjunto no qual a hipótese pode "
  "ser testada é vazio.", first=1.25)

# ============================================================== 5. DISCUSSÃO
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
  "a classe do problema — um problema de rastreamento mais difícil continua "
  "sendo um problema de rastreamento, e responde a uma sintonia mais agressiva.",
  first=1.25)

rico([("Há ainda uma limitação de método que consideramos a mais relevante "
       "deste trabalho: ", 0, 0),
      ("o nicho do RL é o descasamento de modelo, e um simulador de autoria "
       "própria não exibe descasamento por construção", 1, 0),
      (". Qualquer comparação conduzida integralmente dentro do próprio "
       "ambiente favorece estruturalmente métodos que exploram o modelo — "
       "incluindo controle preditivo. Testar a hipótese de vantagem do RL exige "
       "transferência simulação-realidade ou benchmark de terceiros, como em "
       "Boutahri e Tilioua [2025] e Dai et al. [2025].", 0, 0)], first=1.25)

h("5.2. Implicações metodológicas", 2)

p("Três recomendações decorrem diretamente dos achados. Primeira, a "
  "configuração do baseline deve ser reportada com o mesmo detalhe da "
  "arquitetura do agente, e preferencialmente sintonizada com o mesmo esforço "
  "computacional; a Tabela 4 mostra que a diferença entre um baseline "
  "configurado e um não configurado pode exceder toda a contribuição "
  "reivindicada. Segunda, propriedades de segurança devem ser verificadas por "
  "sua distribuição, e não por agregados: a mesma política que exibe 2,5 "
  "comutações por hora apresenta mediana de permanência de 12 minutos contra "
  "requisito de 36. Terceira, constantes que definem restrições devem ser "
  "derivadas de condições de projeto, não arbitradas — uma restrição calibrada "
  "para produzir o resultado desejado é o mesmo defeito que este trabalho "
  "identifica no baseline auditado, deslocado para outro componente.",
  first=1.25)

h("5.3. Ameaças à validade", 2)

for item in [
    "Reimplementação a partir do texto. O código original foi perdido; quatro "
    "parâmetros (B, k, ρ e a penalidade de frio) foram inferidos. O achado da "
    "Seção 4.2 é internamente válido — ambos os controladores operam no mesmo "
    "ambiente — mas sua transferência ao trabalho original é inferência, e não "
    "medição direta. Os achados das Seções 4.4 e 4.5, por dependerem apenas de "
    "parâmetros publicados, não têm essa ressalva.",
    "Sintonia e avaliação no mesmo conjunto. O controlador PI foi sintonizado "
    "sobre a mesma matriz 3×3 usada na avaliação. O resultado deve ser lido "
    "como equivalência nas condições em que ambos foram ajustados. Separação "
    "entre partição de calibração e de teste está implementada e sua execução "
    "constitui trabalho imediato.",
    "Número de sementes. Três sementes por condição impõem piso de p ≈ 0,101 em "
    "teste de permutação, razão pela qual reportamos tamanho de efeito. "
    "Conclusões apoiadas em δ = −1,00 são robustas; as apoiadas em magnitudes "
    "pequenas são sugestivas.",
    "Ausência de validação em hardware e de MPC como referência. Todos os "
    "resultados são de simulação, com balanço térmico agregado, sem gradientes "
    "espaciais, umidade ou CO₂, e com atuador instantâneo. Controle preditivo "
    "baseado em modelo, o adversário natural em regimes de alocação, não foi "
    "implementado.",
]:
    p("• " + item, first=0.5, space_after=4)

# ============================================================== 6. CONCLUSÃO
h("6. Conclusão")

p("Reproduzimos de forma independente um controlador HVAC baseado em "
  "aprendizado por reforço profundo e auditamos sua afirmação central sob "
  "baselines competentemente configurados. A reprodução é fiel em física, "
  "conforto e generalização, mas a vantagem reivindicada não sobrevive: um "
  "controlador PI com duas constantes iguala o DQN em conforto e o supera em "
  "desvio do setpoint e em custo. A decomposição atribui aproximadamente toda a "
  "vantagem à configuração inadequada do adversário.", first=1.25)

p("A ablação mostra que três dos cinco termos da recompensa proposta são "
  "inertes e que a formulação não supera uma quadrática convencional; a "
  "contribuição, corretamente enquadrada, é a correção de uma falha introduzida "
  "pela adoção do platô plano. A penalidade anti-short-cycling não cumpre seu "
  "objetivo, e uma restrição estrutural o cumpre a custo praticamente nulo. Em "
  "contrapartida, métodos tabulares são inadequados neste domínio por razão "
  "teórica medida — 79,8% de transições intra-tile —, de modo que, se RL for "
  "empregado, ele precisa ser profundo.", first=1.25)

p("Finalmente, documentamos o insucesso de três tentativas de construir um "
  "regime favorável ao RL e identificamos sua causa comum: introduzir "
  "dificuldade não altera a classe do problema. Concluímos que, para "
  "climatização de ambiente único com modelo conhecido e objetivo de "
  "rastreamento, o aprendizado por reforço não é a ferramenta indicada, e que "
  "sua avaliação honesta requer benchmarks de terceiros e transferência "
  "simulação-realidade. Como trabalho futuro, apontamos a migração para BOPTEST "
  "ou BuildingGym e a investigação de regimes multi-zona com capacidade "
  "compartilhada, nos quais a decisão de alocação não admite lei de controle "
  "local — tendo, porém, o controle preditivo como referência obrigatória.",
  first=1.25)

p("Todo o material — ambiente, agentes, baselines, suíte de regressão com 79 "
  "testes e registro completo de hiperparâmetros declarados e efetivos — está "
  "disponível para replicação. As Figuras 1 e 3 são recomputadas a partir dos "
  "modelos e do ambiente na mesma execução que produz as Tabelas 3 a 7, e as "
  "Figuras 2 e 4 leem os mesmos arquivos de resultados das Tabelas 5 e 8, de "
  "modo que divergência entre figura e tabela é impossível por construção; os "
  "valores efetivamente plotados são gravados em arquivo próprio para "
  "auditoria.", first=1.25)

# ============================================================== REFERÊNCIAS
h("Referências")

refs = [
 "Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. e Bellemare, M. G. "
 "(2021). Deep Reinforcement Learning at the Edge of the Statistical "
 "Precipice. In Advances in Neural Information Processing Systems (NeurIPS).",

 "Al Sayed, K., Boodi, A., Sadeghian Broujeny, R. e Beddiar, K. (2024). "
 "Reinforcement learning for HVAC control in intelligent buildings: A "
 "technical and conceptual review. Journal of Building Engineering, 95:110085.",

 "ASHRAE (2020). ANSI/ASHRAE Standard 55-2020: Thermal Environmental "
 "Conditions for Human Occupancy. American Society of Heating, Refrigerating "
 "and Air-Conditioning Engineers, Atlanta.",

 "Åström, K. J. e Hägglund, T. (2006). Advanced PID Control. ISA — The "
 "Instrumentation, Systems and Automation Society, Research Triangle Park.",

 "Boutahri, Y. e Tilioua, A. (2025). Reinforcement learning for HVAC control "
 "and energy efficiency in residential buildings with BOPTEST simulations and "
 "real-case validation. Discover Computing, 28:44.",

 "Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal "
 "questions. Psychological Bulletin, 114(3):494–509.",

 "Dai, X., Chen, R., Guan, S., Li, W.-T. e Yuen, C. (2025). BuildingGym: An "
 "open-source toolbox for AI-based building energy management using "
 "reinforcement learning. arXiv:2509.11922.",

 "Fujimoto, S., van Hoof, H. e Meger, D. (2018). Addressing Function "
 "Approximation Error in Actor-Critic Methods. In Proceedings of the 35th "
 "International Conference on Machine Learning (ICML), pages 1587–1596.",

 "Haarnoja, T., Zhou, A., Abbeel, P. e Levine, S. (2018). Soft Actor-Critic: "
 "Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic "
 "Actor. In Proceedings of the 35th International Conference on Machine "
 "Learning (ICML), pages 1861–1870.",

 "Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. e Meger, D. "
 "(2018). Deep Reinforcement Learning that Matters. In Proceedings of the "
 "AAAI Conference on Artificial Intelligence, volume 32.",

 "Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015). Human-level control "
 "through deep reinforcement learning. Nature, 518(7540):529–533.",

 "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. e Dormann, N. "
 "(2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. "
 "Journal of Machine Learning Research, 22(268):1–8.",

 "Sutton, R. S. e Barto, A. G. (2018). Reinforcement Learning: An "
 "Introduction. 2ª edição. MIT Press, Cambridge.",

 "Towers, M., Kwiatkowski, A., Terry, J. et al. (2024). Gymnasium: A Standard "
 "Interface for Reinforcement Learning Environments. arXiv:2407.17032.",

 "Xu, S., Fu, Y., Wang, Y., Yang, Z., Huang, C., O'Neill, Z., Wang, Z. e Zhu, "
 "Q. (2025). Efficient and assured reinforcement learning-based building HVAC "
 "control with heterogeneous expert-guided training. Scientific Reports, "
 "15:7414.",

 "Zha, V., Chiu, I., Guilbault, A. e Tatis, J. (2021). Hyperspace Neighbor "
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
