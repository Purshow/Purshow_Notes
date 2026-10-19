#!/usr/bin/env python3
"""Build either language edition from its combined Markdown source using XeLaTeX.

The five-column Markdown table is the editable content source. Styling follows
the Muon and Video_VAE tables in Purshow_Notes. Only Python's standard library,
XeLaTeX, the TeX Gyre fonts, and macOS Songti/Heiti fonts are required.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from urllib.parse import unquote


BASE = Path(__file__).resolve().parent
LANGUAGE = sys.argv[1] if len(sys.argv) == 2 else "CN"
if LANGUAGE not in {"CN", "EN"}:
    raise SystemExit("Usage: python3 build_pdf.py [CN|EN]")
ROOT = BASE / LANGUAGE
STEM = f"Vision_Encoder_{LANGUAGE}"
SOURCE = ROOT / f"{STEM}.md"
OUTPUT = ROOT / f"{STEM}.pdf"


def escape_tex(value: str, break_words: bool = True) -> str:
    escapes = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
        "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "→": r"\(\rightarrow\)", "×": r"\(\times\)",
        "θ": r"\(\theta\)", "≤": r"\(\leq\)", "–": "-", "—": r"\textemdash{}", "‑": "-",
        "①": r"{\fontspec[Path=/System/Library/Fonts/Supplemental/,FontIndex=6]{Songti.ttc}①}",
        "②": r"{\fontspec[Path=/System/Library/Fonts/Supplemental/,FontIndex=6]{Songti.ttc}②}",
        "③": r"{\fontspec[Path=/System/Library/Fonts/Supplemental/,FontIndex=6]{Songti.ttc}③}",
        "④": r"{\fontspec[Path=/System/Library/Fonts/Supplemental/,FontIndex=6]{Songti.ttc}④}",
    }
    result = []
    for char in html.unescape(value):
        result.append(escapes.get(char, char))
        if break_words and char in "/_=-.":
            result.append(r"\allowbreak{}")
    return "".join(result)


TOKENS = re.compile(
    r"(?P<strong>\*\*(?P<strong_text>.+?)\*\*)"
    r"|(?P<code>`(?P<code_text>[^`]+)`)"
    r"|(?P<math>\$(?P<math_text>[^$]+)\$)"
    r"|(?P<link>\[(?P<link_text>[^\]]+)\]\((?P<link_url>https?://[^\s)]+)\))"
    r"|(?P<reference>\[(?P<reference_text>[^\]]+)\]\[(?P<reference_id>\d+)\])"
    r"|(?P<br><br\s*/?>)",
    re.DOTALL,
)


def inline(value: str, references: dict[str, str]) -> str:
    result, end = [], 0
    for match in TOKENS.finditer(value):
        result.append(escape_tex(value[end:match.start()]))
        if match.group("strong"):
            result.append(r"{\bfseries " + inline(match.group("strong_text"), references) + "}")
        elif match.group("code"):
            result.append(r"\texttt{" + escape_tex(match.group("code_text")) + "}")
        elif match.group("math"):
            formula = match.group("math_text")
            for char, command in {"×": r"\times ", "→": r"\rightarrow ", "θ": r"\theta ", "≤": r"\leq ", "⌈": r"\lceil ", "⌉": r"\rceil ", "–": "-"}.items():
                formula = formula.replace(char, command)
            result.append(r"\(" + formula + r"\)")
        elif match.group("link"):
            result.append(r"\href{\detokenize{" + match.group("link_url") + "}}{" + inline(match.group("link_text"), references) + "}")
        elif match.group("reference"):
            reference_id = match.group("reference_id")
            url = references[reference_id]
            result.append(r"\href{\detokenize{" + url + "}}{" + escape_tex(match.group("reference_text")) + "}")
        else:
            result.append(r"\par\vspace{1pt}")
        end = match.end()
    result.append(escape_tex(value[end:]))
    return "".join(result)


PREAMBLE = r"""\input{pdf-header.tex}
\begin{document}
{\sffamily\bfseries\color{notesblue}\fontsize{23}{28}\selectfont @@TITLE@@\par}
\vspace{1mm}{\color{notesblue}\rule{\linewidth}{0.8pt}}\par\vspace{1mm}
@@INTRO@@
\begingroup
\fontsize{9.5pt}{12pt}\selectfont
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.02}
\setlength{\extrarowheight}{2pt}
\setlength{\LTpre}{5pt}
\setlength{\LTpost}{6pt}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\begin{longtable}{@{}P{0.130\dimexpr\textwidth-40pt\relax}P{0.235\dimexpr\textwidth-40pt\relax}P{0.300\dimexpr\textwidth-40pt\relax}P{0.245\dimexpr\textwidth-40pt\relax}P{0.090\dimexpr\textwidth-40pt\relax}@{}}
\toprule
"""


DETAIL_LAYOUT = r"""
\clearpage
\newgeometry{layoutsize={210mm,297mm},margin=18mm,headheight=15pt,headsep=9pt,footskip=22pt}
\setlength{\paperwidth}{210mm}
\setlength{\paperheight}{297mm}
\setlength{\pdfpagewidth}{210mm}
\setlength{\pdfpageheight}{297mm}
\fancyhfoffset{0pt}
\fancyhead[L]{\small\sffamily\color{notesgray}Vision Encoder / ViT Details}
\ctexset{section={format=\Large\bfseries\sffamily\color{notesblue},beforeskip=1.5ex,afterskip=1ex},subsection={format=\large\bfseries\sffamily\color{notesblue},beforeskip=1.5ex,afterskip=1ex},subsubsection={format=\normalsize\bfseries\sffamily\color{notesblue},beforeskip=1.5ex,afterskip=0.5ex}}
\linespread{1.10}\normalsize
\setlength{\parskip}{0.26em}
\setlength{\emergencystretch}{2em}
\setlist[itemize]{leftmargin=1.7em,itemsep=0.08em,parsep=0.08em,topsep=0.22em}
\setlength{\abovedisplayskip}{7pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{7pt plus 2pt minus 2pt}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
"""


def render_details(source: str) -> str:
    """Render the headings, lists, equations and local figures in the ViT notes."""
    layout = DETAIL_LAYOUT
    if LANGUAGE == "EN":
        layout = layout.replace(r"\linespread{1.10}\normalsize", r"\linespread{1.05}\fontsize{10.5pt}{12pt}\selectfont")
    output = [layout]
    paragraphs: list[str] = []
    list_indents: list[int] = []
    lines = source.splitlines()
    section_count = 0

    def flush_paragraph() -> None:
        if paragraphs:
            output.append(inline(" ".join(paragraphs), {}) + r"\par")
            paragraphs.clear()

    def close_lists() -> None:
        while list_indents:
            output.append(r"\end{itemize}")
            list_indents.pop()

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            flush_paragraph()
            continue
        bullet = re.match(r"^(\s*)- (.+)$", line)
        if bullet:
            flush_paragraph()
            indent = len(bullet.group(1))
            while list_indents and list_indents[-1] > indent:
                output.append(r"\end{itemize}")
                list_indents.pop()
            if not list_indents or list_indents[-1] < indent:
                output.append(r"\begin{itemize}")
                list_indents.append(indent)
            following = next((item for item in lines[index:] if item.strip()), "")
            child = re.match(r"^(\s*)- ", following)
            if child and len(child.group(1)) > indent:
                output.append(r"\Needspace{6\baselineskip}")
            output.append(r"\item " + inline(bullet.group(2), {}))
            continue
        close_lists()
        heading = re.match(r"^(#{1,4}) (.+)$", line)
        figure = re.fullmatch(r"!\[[^\]]*\]\((.+)\)", line.strip())
        if line.strip() == "<!-- pagebreak -->":
            flush_paragraph()
            output.append(r"\clearpage")
        elif heading:
            flush_paragraph()
            level, title = len(heading.group(1)), heading.group(2)
            if level == 1:
                output.append(r"{\sffamily\bfseries\color{notesblue}\fontsize{23}{28}\selectfont " + inline(title, {}) + r"\par}\vspace{1mm}{\color{notesblue}\rule{\linewidth}{0.8pt}}\par\vspace{1mm}")
            else:
                if level == 2:
                    if section_count:
                        output.append(r"\clearpage")
                    section_count += 1
                    output.append(r"\fancyhead[L]{\small\sffamily\color{notesgray}Vision Encoder / " + inline(title, {}) + "}")
                output.append(r"\Needspace{5\baselineskip}")
                command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                output.append("\\" + command + "{" + inline(title, {}) + "}")
        elif figure:
            flush_paragraph()
            path = (ROOT / unquote(figure.group(1))).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            output.append(r"\begin{center}\includegraphics[width=\linewidth,height=0.30\textheight,keepaspectratio]{\detokenize{" + path.relative_to(ROOT).as_posix() + r"}}\end{center}")
        elif line.strip() == "$$":
            flush_paragraph()
            formula = []
            while index < len(lines) and lines[index].strip() != "$$":
                formula.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed display equation in ViT notes")
            index += 1
            output.append("\\[\n" + "\n".join(formula) + "\n\\]")
        else:
            paragraphs.append(line.strip())
    flush_paragraph()
    close_lists()
    return "\n".join(output)


def render(source: str, details: str) -> str:
    references = dict(re.findall(r'^\[(\d+)\]:\s+(https?://\S+)', source, flags=re.MULTILINE))
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in source.splitlines() if line.startswith("|")
    ]
    if len(rows) < 3 or any(len(row) != 5 for row in rows):
        raise ValueError("Expected a five-column table with at least one model row")
    header = " & ".join(r"\textbf{" + inline(cell, references) + "}" for cell in rows[0]) + r" \\"
    intro_paragraphs = [line for line in source.splitlines()[:next(i for i,line in enumerate(source.splitlines()) if line.startswith("|"))] if line.strip() and not line.startswith("#")]
    introduction = "\n".join(r"{\small " + inline(line, references) + r"}\par\vspace{2pt}" for line in intro_paragraphs)
    title = next(line[2:] for line in source.splitlines() if line.startswith("# "))
    preamble = PREAMBLE.replace("@@INTRO@@", introduction).replace("@@TITLE@@", inline(title, {}))
    if LANGUAGE == "EN":
        # English prose needs more space in the infrastructure column.
        for before, after in [("0.130", "0.100"), ("0.235", "0.230"),
                              ("0.300", "0.290"), ("0.245", "0.240"),
                              ("0.090", "0.140")]:
            preamble = preamble.replace(before, after)
    output = [preamble, header, r"\midrule\endfirsthead", r"\toprule", header,
              r"\midrule\endhead", r"\midrule", r"\multicolumn{5}{r}{\footnotesize\color{notesgray}续下页}\endfoot",
              r"\bottomrule\endlastfoot"]
    for index, row in enumerate(rows[2:]):
        if index:
            output.append(r"\addlinespace[6pt]")
        output.append(" & ".join(inline(cell, references) for cell in row) + r" \\")
    output.extend([r"\end{longtable}", r"\endgroup", render_details(details), r"\end{document}"])
    return ("\n".join(output) + "\n").replace("续下页", "Continued on next page" if LANGUAGE == "EN" else "续下页")


def main() -> None:
    engine = shutil.which("xelatex") or "/Library/TeX/texbin/xelatex"
    source, details = SOURCE.read_text(encoding="utf-8").split("<!-- vit-details -->", 1)
    tex = render(source, details)
    tex_path = ROOT / f"{STEM}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="vision-encoder-") as temporary:
        build = Path(temporary)
        for _ in range(2):
            result = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", str(tex_path)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode:
                print(result.stdout)
                raise SystemExit(result.returncode)
        log = (build / f"{STEM}.log").read_text(errors="replace")
        warnings = re.findall(r"(?:Overfull[^\n]*|Missing character[^\n]*)", log)
        if warnings:
            raise RuntimeError("PDF layout needs correction:\n" + "\n".join(warnings))
        shutil.copy2(build / f"{STEM}.pdf", OUTPUT)
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
