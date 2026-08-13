# -*- coding: utf-8 -*-
"""
Backend LaTeX para a classe `cas-sc` da Elsevier.

POR QUE ESTE MÓDULO EXISTE
--------------------------
A Elsevier distribui, para submissão em LaTeX, o pacote CAS (`cas-sc.cls` em
coluna única, `cas-dc.cls` em coluna dupla). O manuscrito precisa sair nesse
formato, e o texto do artigo não pode ser escrito duas vezes: um gerador para
.docx e outro para .tex divergiriam na primeira revisão, que é exatamente o
defeito que este projeto elimina entre tabela e figura.

A solução é a mesma adotada para os números: uma fonte, dois consumidores. O
corpo de `gerar_paper_eb.py` chama `p()`, `rico()`, `tabela()`, `figura()`,
`legenda()`, `h()`, `item()` e `equacao()`; este módulo fornece a implementação
LaTeX dessas mesmas operações, e o gerador escolhe o backend por linha de
comando. O texto do artigo permanece escrito uma única vez.

O QUE O FORMATO CAS EXIGE, E QUE O .DOCX NÃO EXIGIA
---------------------------------------------------
1. Numeração de seções, tabelas, figuras e equações é do LaTeX, não do texto.
   As legendas que chegam aqui trazem "Table 7." ou "Fig. 3." no início porque
   o .docx precisa delas; o prefixo é removido e o `\\caption` renumera. As duas
   numerações coincidem porque a ordem de emissão é a mesma, e é derivada da
   composição em ambos os casos.
2. O `front matter` (título, autor, afiliação, `abstract`, `highlights`,
   `keywords`) vive no preâmbulo, antes de `\\maketitle`, e não na sequência em
   que o .docx o imprime.
3. As citações passam a ser `\\cite{chave}`: a numeração por ordem de aparição
   deixa de ser calculada em Python e passa a ser responsabilidade do LaTeX.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------- escape

# Caracteres que o LaTeX interpreta. A barra invertida vem primeiro, pois
# substituí-la depois corromperia as sequências introduzidas pelas demais.
_ESPECIAIS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
              ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
              ("}", r"\}"), ("~", r"\textasciitilde{}"),
              ("^", r"\textasciicircum{}")]

# Unicode que aparece no manuscrito. Manter o mapa explícito, em vez de confiar
# no UTF-8 do pdfTeX, evita o modo de falha mais chato do LaTeX: o caractere
# some silenciosamente do PDF em vez de gerar erro.
_UNICODE = {
    "—": "---", "–": "--", "−": "$-$", "·": "$\\cdot$", "×": "$\\times$",
    "±": "$\\pm$", "≤": "$\\le$", "≥": "$\\ge$", "≈": "$\\approx$",
    "∈": "$\\in$", "°": "$^\\circ$", "•": "$\\bullet$",
    "δ": "$\\delta$", "ρ": "$\\rho$", "ε": "$\\varepsilon$",
    "γ": "$\\gamma$", "σ": "$\\sigma$", "τ": "$\\tau$", "π": "$\\pi$",
    "Δ": "$\\Delta$", "²": "$^2$", "³": "$^3$", "⁻": "$^-$", "⁵": "$^5$",
    "₂": "$_2$", "Å": "\\AA{}", "ö": '\\"o', "ä": '\\"a', "’": "'",
    "“": "``", "”": "''", "…": "\\dots{}", " ": "\\,", " ": "~",
}


# Variáveis com índice escritas em ASCII no fonte ("K_p", "d_min", "C_th").
# Sem tratamento elas saem no PDF com o sublinhado literal, que é o defeito que
# o revisor vê primeiro. A base é de UMA letra, de propósito: identificadores de
# código como `tdis_tot` ou `comfort_wide_pct` devem permanecer literais.
_RX_INDICE = re.compile(r"(?<![\w\\])([A-Za-z])_([A-Za-z]{1,6}|\d{1,2})\b")
_MARCA_IND = "\x00%d\x00"


def escapar(texto: str) -> str:
    """Texto puro para modo texto do LaTeX."""
    # Os índices viram fórmula ANTES do escape, senão o `_` seria neutralizado.
    guardados = []

    def _guarda(m):
        guardados.append("$%s_{\\mathrm{%s}}$" % (m.group(1), m.group(2)))
        return _MARCA_IND % (len(guardados) - 1)

    texto = _RX_INDICE.sub(_guarda, texto)
    for alvo, troca in _ESPECIAIS:
        texto = texto.replace(alvo, troca)
    for alvo, troca in _UNICODE.items():
        texto = texto.replace(alvo, troca)
    for i, fórmula in enumerate(guardados):
        texto = texto.replace(_MARCA_IND % i, fórmula)
    return texto


# ------------------------------------------------------ marcação inline

_RX_MARCA = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*", re.DOTALL)


def inline(texto: str, *, bold: bool = False, italic: bool = False) -> str:
    """Converte `**negrito**` e `*itálico*` em comandos, escapando o resto."""
    saida, pos = [], 0
    for m in _RX_MARCA.finditer(texto):
        if m.start() > pos:
            saida.append(escapar(texto[pos:m.start()]))
        if m.group(1) is not None:
            saida.append(r"\textbf{" + escapar(m.group(1)) + "}")
        else:
            saida.append(r"\emph{" + escapar(m.group(2)) + "}")
        pos = m.end()
    if pos < len(texto):
        saida.append(escapar(texto[pos:]))
    corpo = "".join(saida)
    if bold:
        corpo = r"\textbf{" + corpo + "}"
    if italic:
        corpo = r"\emph{" + corpo + "}"
    return corpo


# --------------------------------------------------------- equações

# As equações são descritas por uma árvore mínima, construída pelos mesmos
# `_op/_sub/_sup/_frac` que o backend .docx usa. Cada backend a percorre à sua
# maneira: um emite OMML, o outro, LaTeX. Sem essa árvore, as sete equações do
# artigo precisariam ser escritas duas vezes.

_MATH_UNICODE = {
    "−": "-", "·": r"\cdot ", "×": r"\times ", "≤": r"\le ", "≥": r"\ge ",
    "∈": r"\in ", "γ": r"\gamma ", "ρ": r"\rho ", "ε": r"\varepsilon ",
    "σ": r"\sigma ", "τ": r"\tau ", "π": r"\pi ", "Δ": r"\Delta ",
    "δ": r"\delta ", "𝔼": r"\mathbb{E}", "′": "'", "°": r"^\circ ",
}


def _uni_math(texto: str) -> str:
    """Só a troca de unicode por comando; sem mexer em espaços."""
    for alvo, troca in _MATH_UNICODE.items():
        texto = texto.replace(alvo, troca)
    return texto


def _math_txt(texto: str) -> str:
    # Em modo matemático o espaço é ignorado; preservá-lo exige `\ `. A troca
    # acontece uma única vez: encadeá-la com outra substituição de espaço
    # duplicava as barras e imprimia `\\` no lugar do espaço.
    return _uni_math(texto).replace("  ", r"\ \ ")


# Trecho em PROSA dentro da fórmula ("if a switch occurred and"): duas ou mais
# letras seguidas e ao menos um espaço.
_RX_PROSA = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]")     # duas palavras: prosa
_RX_PALAVRA = re.compile(r"[A-Za-z]{2,}")               # uma palavra: função


def tex_math(no) -> str:
    """Renderiza a árvore de uma equação em LaTeX."""
    if isinstance(no, str):                       # variável: itálico natural
        return _math_txt(no)
    if isinstance(no, (list, tuple)) and no and no[0] in {
            "op", "sub", "sup", "frac"}:
        tipo = no[0]
        if tipo == "op":                          # texto romano
            bruto = no[1]
            if not bruto.strip():
                # Espaço puro entre termos: em modo matemático ele é ignorado,
                # e é preciso pedi-lo explicitamente.
                return r"\;" if bruto else ""
            if _RX_PROSA.search(bruto):
                # Prosa dentro da fórmula. `\mathrm` NÃO preserva espaços, foi
                # o que colou "if a switch occurred and" numa palavra só;
                # `\text` volta ao modo texto, onde o espaço existe, e os
                # símbolos restantes voltam ao matemático pelo mapa de escape.
                return r"\text{" + escapar(bruto) + "}"
            if _RX_PALAVRA.search(bruto):
                # Nome de função (clip, sin, cos, max): romano, com o espaço
                # pedido explicitamente.
                return r"\mathrm{%s}" % _uni_math(bruto).replace(" ", r"\ ")
            # Operadores e pontuação vão CRUS. Envolvê-los em `\mathrm` os
            # transforma em átomos comuns e o LaTeX deixa de aplicar o
            # espaçamento de relação e de operação binária, que foi o que
            # comprimiu "T=T+" na equação da física.
            return _math_txt(bruto)
        if tipo == "sub":
            return "{" + tex_math(no[1]) + "}_{" + tex_math(no[2]) + "}"
        if tipo == "sup":
            return "{" + tex_math(no[1]) + "}^{" + tex_math(no[2]) + "}"
        return (r"\frac{" + tex_math(no[1]) + "}{" + tex_math(no[2]) + "}")
    if isinstance(no, (list, tuple)):             # sequência
        return "".join(tex_math(x) for x in no)
    return str(no)


# ------------------------------------------------------------ documento

_RX_PREFIXO_LEG = re.compile(r"^(Table|Fig\.|Figure)\s*\d+\.\s*")
_RX_NUM_SECAO = re.compile(r"^\d+(\.\d+)*\.?\s+")


class DocumentoCAS:
    """
    Acumula o corpo do manuscrito em LaTeX e o encerra no formato CAS.

    Mantém apenas o estado que o LaTeX exige e o .docx não: a legenda pendente
    de uma tabela (que no .docx é impressa antes, e aqui vira `\\caption` dentro
    do flutuante), a figura aberta à espera da legenda seguinte, e a lista de
    itens em aberto.
    """

    def __init__(self, *, figs_rel: str = "figures"):
        self.linhas: List[str] = []
        self.figs_rel = figs_rel
        self._legenda_pendente: Optional[str] = None
        self._figura_aberta: Optional[Tuple[str, float]] = None
        self._itens: List[str] = []
        self._descricoes: List[Tuple[str, str]] = []
        self._n_tab = 0
        self._fim_tabela: Optional[int] = None
        self._n_fig = 0

    # ---------------------------------------------------------- infraestrutura

    # Há duas listas em aberto possíveis, e elas não podem se fechar uma à
    # outra: `descricao()` fecha só a de marcadores, `item()` fecha só a de
    # descrição, e qualquer outro bloco fecha ambas. Sem essa separação cada
    # símbolo da Nomenclatura abria e fechava seu próprio ambiente.
    def _fechar_itens(self) -> None:
        if not self._itens:
            return
        self.linhas.append(r"\begin{itemize}")
        self.linhas.extend(r"  \item " + x for x in self._itens)
        self.linhas.append(r"\end{itemize}")
        self.linhas.append("")
        self._itens = []

    def _fechar_listas(self) -> None:
        self._fechar_itens()
        self._fechar_descricoes()

    def bruto(self, texto: str) -> None:
        self._fechar_listas()
        self.linhas.append(texto)

    # ----------------------------------------------------------------- blocos

    def paragrafo(self, texto: str, *, bold=False, italic=False) -> None:
        if not texto.strip():
            return
        self._fechar_listas()
        self.linhas.append(inline(texto, bold=bold, italic=italic))
        self.linhas.append("")

    def paragrafo_rico(self, partes: Iterable[Tuple[str, object, object]]) -> None:
        self._fechar_listas()
        self.linhas.append("".join(
            inline(t, bold=bool(b), italic=bool(i)) for t, b, i in partes))
        self.linhas.append("")

    def secao(self, texto: str, nivel: int = 1) -> None:
        self._fechar_listas()
        # A numeração é do LaTeX: o prefixo que o .docx carrega sairia duplicado.
        titulo = _RX_NUM_SECAO.sub("", texto).strip()
        cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[nivel]
        self.linhas.append("\\%s{%s}" % (cmd, escapar(titulo)))
        self.linhas.append("")

    def secao_sem_numero(self, texto: str) -> None:
        self._fechar_listas()
        self.linhas.append(r"\section*{%s}" % escapar(texto))
        self.linhas.append("")

    def item(self, texto: str) -> None:
        self._itens.append(inline(texto))

    def descricao(self, termo: str, sentido: str) -> None:
        """Item de lista de descrição, para a Nomenclatura."""
        self._fechar_itens()
        self._descricoes.append((termo, sentido))

    def item(self, texto: str) -> None:
        self._fechar_descricoes()
        self._itens.append(inline(texto))

    def _fechar_descricoes(self) -> None:
        if not self._descricoes:
            return
        self.linhas.append(r"\begin{description}[leftmargin=3.2em,"
                           r"style=sameline,labelwidth=2.6em]")
        for termo, sentido in self._descricoes:
            self.linhas.append(r"  \item[%s] %s"
                               % (inline(termo), inline(sentido)))
        self.linhas.append(r"\end{description}")
        self.linhas.append("")
        self._descricoes = []

    def nota(self, texto: str) -> None:
        """
        Nota de tabela, colocada DENTRO do flutuante.

        No .docx a nota é um parágrafo logo abaixo da tabela, e ali isso basta,
        porque a tabela não se move. No LaTeX a tabela é um flutuante: emitida
        no fluxo do texto, a nota fica órfã, em corpo reduzido, no meio de uma
        seção da qual a tabela já saiu. Foi o que apareceu logo abaixo do
        parágrafo de abertura da Seção 4.2.

        A nota é, portanto, inserida antes do `\end{table}` da tabela mais
        recente. O `minipage` garante quebra de linha e alinhamento à esquerda
        dentro do ambiente centralizado do flutuante.
        """
        if self._fim_tabela is None:
            # Sem tabela recente, degrada para parágrafo: melhor que descartar
            # a nota silenciosamente.
            self.paragrafo(texto)
            return
        bloco = [r"\vspace{3pt}",
                 r"\begin{minipage}{\columnwidth}\footnotesize %s\end{minipage}"
                 % inline(texto)]
        self.linhas[self._fim_tabela:self._fim_tabela] = bloco
        self._fim_tabela = None

    def legenda(self, texto: str, *, acima: bool = False) -> None:
        """
        No LaTeX a legenda pertence ao flutuante, não ao fluxo do texto.

        Chamada antes de uma tabela, fica pendente; chamada depois de uma
        figura, fecha o flutuante aberto.
        """
        limpo = _RX_PREFIXO_LEG.sub("", texto).strip()
        if self._figura_aberta is not None:
            caminho, largura = self._figura_aberta
            self._figura_aberta = None
            self._n_fig += 1
            self.linhas.append(r"  \caption{%s}" % inline(limpo))
            self.linhas.append(r"  \label{fig:%d}" % self._n_fig)
            self.linhas.append(r"\end{figure}")
            self.linhas.append("")
            return
        self._legenda_pendente = limpo

    def figura(self, caminho_png: str, largura_frac: float = 0.86) -> None:
        self._fechar_listas()
        nome = os.path.basename(caminho_png)
        self.linhas.append(r"\begin{figure}[htbp]")
        self.linhas.append(r"  \centering")
        self.linhas.append(
            r"  \includegraphics[width=%.2f\linewidth]{%s/%s}"
            % (largura_frac, self.figs_rel, nome))
        self._figura_aberta = (nome, largura_frac)

    def tabela(self, cabecalho: Sequence[str], linhas: Sequence[Sequence[str]],
               *, negrito_linhas: Sequence[int] = (),
               larguras: Optional[Sequence[float]] = None) -> None:
        """
        Tabela de largura EXATA, com colunas que quebram linha.

        Duas armadilhas foram encontradas por medição, e ambas motivam a forma
        abaixo. A primeira: as colunas `L`/`C` da classe não quebram linha, de
        modo que uma tabela de sete colunas com rótulos longos estoura a margem
        (441 pt no primeiro teste). A segunda: `tabular*` distribui a folga em
        `\extracolsep`, e sem essa cola ele briga com colunas de largura fixa,
        gerando um aviso por linha.

        A forma correta para colunas medidas é `tabular` simples com `p{}`, e a
        largura de cada uma descontando o próprio espaçamento:

            p{\dimexpr F\columnwidth - 2\tabcolsep\relax}

        Como cada coluna ocupa `p + 2·tabcolsep` e as frações somam 1, a tabela
        mede exatamente uma coluna de texto, qualquer que seja o número de
        colunas. As larguras em centímetros já declaradas para o .docx são a
        fonte das frações: a mesma especificação serve aos dois formatos.
        """
        self._fechar_listas()
        self._n_tab += 1
        n = len(cabecalho)

        if larguras and len(larguras) == n:
            total = float(sum(larguras)) or 1.0
            fracoes = [w / total for w in larguras]
        else:
            fracoes = [1.0 / n] * n

        col = []
        for j, f in enumerate(fracoes):
            alinha = (r">{\raggedright\arraybackslash}" if j == 0
                      else r">{\centering\arraybackslash}")
            col.append(r"%sp{\dimexpr %.4f\columnwidth-2\tabcolsep\relax}"
                       % (alinha, f))
        spec = "@{}" + "".join(col) + "@{}"

        # Acima de seis colunas o corpo reduzido é o que mantém a tabela legível
        # sem recorrer a rotação de página, que o formato de submissão evita.
        corpo = r"\scriptsize" if n >= 7 else r"\footnotesize"

        self.linhas.append(r"\begin{table}[htbp]")
        if self._legenda_pendente:
            self.linhas.append(r"\caption{%s}" % inline(self._legenda_pendente))
            self._legenda_pendente = None
        self.linhas.append(r"\label{tbl:%d}" % self._n_tab)
        self.linhas.append(r"\centering%s\setlength{\tabcolsep}{3pt}" % corpo)
        self.linhas.append(r"\begin{tabular}{%s}" % spec)
        self.linhas.append(r"\toprule")
        self.linhas.append(" & ".join(r"\textbf{%s}" % escapar(str(c))
                                      for c in cabecalho) + r" \\")
        self.linhas.append(r"\midrule")
        for i, linha in enumerate(linhas):
            celulas = [escapar(str(x)) for x in linha]
            if i in negrito_linhas:
                celulas = [r"\textbf{%s}" % c if c.strip() else c
                           for c in celulas]
            self.linhas.append(" & ".join(celulas) + r" \\")
        self.linhas.append(r"\bottomrule")
        self.linhas.append(r"\end{tabular}")
        # Onde o flutuante fecha: é aqui que uma nota de tabela deve entrar.
        self._fim_tabela = len(self.linhas)
        self.linhas.append(r"\end{table}")
        self.linhas.append("")

    def equacao(self, arvore, numero: Optional[int] = None) -> None:
        self._fechar_listas()
        corpo = tex_math(arvore)
        if numero is None:
            self.linhas.append(r"\begin{equation*}")
            self.linhas.append("  " + corpo)
            self.linhas.append(r"\end{equation*}")
        else:
            self.linhas.append(r"\begin{equation}")
            self.linhas.append("  " + corpo)
            self.linhas.append(r"\end{equation}")
        self.linhas.append("")

    # ------------------------------------------------------------- montagem

    def montar(self, *, titulo: str, titulo_curto: str, autor: str,
               afiliacao: Dict[str, str], email: str, credito: str,
               orcid: str = "",
               resumo: str, destaques: Sequence[str],
               palavras_chave: Sequence[str],
               bibliografia: Sequence[Tuple[str, str]]) -> str:
        """Devolve o documento completo, do `\\documentclass` ao `\\end`."""
        self._fechar_listas()
        L: List[str] = []
        a = L.append

        a("% !TEX program = pdflatex")
        a("%")
        a("% Manuscrito gerado por gerar_paper_eb.py --tex")
        a("% NÃO EDITAR À MÃO: a próxima execução do gerador sobrescreve este")
        a("% arquivo. Alterações de texto vão no gerador.")
        a("%")
        a(r"\documentclass[a4paper,fleqn]{cas-sc}")
        a(r"\usepackage[numbers]{natbib}")
        a(r"\usepackage{graphicx}")
        a(r"\usepackage{amsmath}")
        a(r"\usepackage{amssymb}")
        a(r"\usepackage{array}")
        a(r"\usepackage{enumitem}")
        a(r"\usepackage{placeins}")
        a("")
        a(r"\begin{document}")
        a(r"\let\WriteBookmarks\relax")
        # Parâmetros de flutuação. Com os padrões do LaTeX e figuras da largura
        # da coluna, a primeira figura não coube na página em que é citada e,
        # pela regra de ordem dos flutuantes, ARRASTOU todas as seguintes para o
        # fim do documento — 17 figuras empilhadas depois das referências.
        # Afrouxar as frações devolve cada figura para perto do seu texto.
        a(r"\renewcommand{\topfraction}{0.9}")
        a(r"\renewcommand{\bottomfraction}{0.8}")
        a(r"\renewcommand{\textfraction}{0.07}")
        a(r"\renewcommand{\floatpagefraction}{0.7}")
        a(r"\setcounter{topnumber}{3}")
        a(r"\setcounter{bottomnumber}{2}")
        a(r"\setcounter{totalnumber}{5}")
        # Sem barreira, a primeira figura que não coube arrastou as dezesseis
        # seguintes para depois das referências. A barreira por SEÇÃO devolve
        # cada figura à sua seção; por subseção ela também funcionava, mas
        # forçava quebras de página. Optou-se pela barreira por SUBSEÇÃO: a
        # proximidade entre figura e texto vale mais para quem revisa do que a
        # contagem de páginas, que a revista não limita, e a largura reduzida
        # das figuras recupera parte do espaço.
        a(r"\makeatletter")
        a(r"\let\@Oldsubsection\subsection")
        a(r"\renewcommand{\subsection}{\FloatBarrier\@Oldsubsection}")
        a(r"\let\@Oldsection\section")
        a(r"\renewcommand{\section}{\FloatBarrier\@Oldsection}")
        a(r"\makeatother")
        a("")
        a(r"\shorttitle{%s}" % escapar(titulo_curto))
        a(r"\shortauthors{%s}" % escapar(autor))
        a("")
        a(r"\title[mode=title]{%s}" % escapar(titulo))
        a("")
        a(r"\author[1]{%s}[orcid=%s]" % (escapar(autor), escapar(orcid)))
        a(r"\cormark[1]")
        a(r"\ead{%s}" % escapar(email))
        a(r"\credit{%s}" % escapar(credito))
        a("")
        a(r"\affiliation[1]{organization={%s},"
          % escapar(afiliacao.get("organization", "")))
        a(r"            addressline={%s},"
          % escapar(afiliacao.get("addressline", "")))
        a(r"            city={%s}," % escapar(afiliacao.get("city", "")))
        a(r"            postcode={%s}," % escapar(afiliacao.get("postcode", "")))
        a(r"            state={%s}," % escapar(afiliacao.get("state", "")))
        a(r"            country={%s}}" % escapar(afiliacao.get("country", "")))
        a("")
        a(r"\cortext[1]{Corresponding author}")
        a("")
        a(r"\begin{abstract}")
        a(inline(resumo))
        a(r"\end{abstract}")
        a("")
        a(r"\begin{highlights}")
        for d in destaques:
            a(r"\item " + inline(d))
        a(r"\end{highlights}")
        a("")
        a(r"\begin{keywords}")
        a(" \\sep ".join(escapar(k) for k in palavras_chave))
        a(r"\end{keywords}")
        a("")
        a(r"\maketitle")
        a("")
        L.extend(self.linhas)
        a("")
        a(r"\printcredits")
        a("")
        # `thebibliography` em vez de .bib: as entradas já estão formatadas no
        # estilo da editora, e reconstruí-las em BibTeX só acrescentaria uma
        # oportunidade de divergir da versão .docx.
        a(r"\begin{thebibliography}{%d}" % len(bibliografia))
        for chave, texto in bibliografia:
            a(r"\bibitem{%s} %s" % (chave, escapar(texto)))
        a(r"\end{thebibliography}")
        a("")
        a(r"\end{document}")
        return "\n".join(L) + "\n"


def instalar_classe(origem: str, destino: str) -> List[str]:
    """
    Copia a classe CAS e seus arquivos de apoio para junto do manuscrito.

    A Elsevier pede que a submissão em LaTeX seja autocontida; sem estes
    arquivos o .tex não compila na máquina do editor.
    """
    necessarios = ["cas-sc.cls", "cas-common.sty", "cas-model2-names.bst",
                   "cas-dc.cls"]
    copiados = []
    for nome in necessarios:
        caminho = os.path.join(origem, nome)
        if os.path.exists(caminho):
            shutil.copy2(caminho, os.path.join(destino, nome))
            copiados.append(nome)
    return copiados
