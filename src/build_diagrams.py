#!/usr/bin/env python3
"""
Write the TikZ sources for F1 (encrypted-inference pipeline) and F2 (leaf-aligned packing
architecture) -- the two conceptual/architecture diagrams the guide's Sec 23 lists alongside the
six data figures F3-F8. Unlike F3-F8 (src/generate_figures.py), these do NOT go through
src/paper_data.py's Evidence accessor: they are schematics of the pipeline/packing STRUCTURE
described in paper/sections/05_methods_ckks_circuit.tex and 06_methods_packing.tex, not a plot of
measured numbers, so there is nothing here for verify_paper_assets.py to trace to a results/*.json
artifact. Where a real number would matter (ensemble size, leaf count), the manuscript CAPTION
(paper/floats/f1_pipeline.tex, f2_packing.tex) cites the real \\MOne... macro instead -- the
diagrams themselves are deliberately schematic (a small illustrative tree count/depth, not the
actual 25-model, \\MOneMaxLeaves{}-leaf configuration) and say so in their captions.

This script only WRITES the .tex sources (paper_assets/figures_src/f1_pipeline.tex,
f2_packing.tex) -- pure text generation, no TeX toolchain needed, so it can run from either the
WSL venv or Windows Python. Compiling them to PDF requires pdflatex (Windows-side MiKTeX, same
cross-OS split as paper/build_paper.ps1), done by the sibling paper/build_diagrams.ps1, which is
therefore also NOT an orchestrator stage in experiments/run_reproducible_m1.py, for the same
reason paper/build_paper.ps1 is not.

Output: paper_assets/figures_src/f1_pipeline.tex, f2_packing.tex
Usage: python build_diagrams.py
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "paper_assets", "figures_src")

PREAMBLE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,decorations.pathreplacing}
\begin{document}
"""
POSTAMBLE = r"""\end{document}
"""

# ---------------------------------------------------------------------------------------------
# F1 -- encrypted-inference pipeline. Stages per paper/sections/05_methods_ckks_circuit.tex line
# 6-8: "encrypted features -> polynomial comparison -> leaf/path products -> packed tree-level
# aggregation -> encrypted score -> decryption in margin space", framed with the client/server
# trust boundary (the server never holds the decryption key) and the phase names
# 10_results_latency.tex actually measures (client preprocessing, encode/encrypt, polynomial
# evaluation, path product, final rotate-and-add sum, decrypt/decode).
# ---------------------------------------------------------------------------------------------
F1_BODY = r"""
\begin{tikzpicture}[
  font=\scriptsize,
  stage/.style={draw, rounded corners, align=center, minimum height=1.1cm, minimum width=2.05cm,
                fill=blue!6, inner sep=3pt},
  client/.style={draw, rounded corners, align=center, minimum height=1.1cm, minimum width=1.9cm,
                 fill=green!8, inner sep=3pt},
  arr/.style={-{Stealth[length=2.2mm]}, thick},
  every node/.style={font=\scriptsize}
]
  \node[client] (feat) {Plaintext\\ patient features $x$};
  \node[client, right=1.0cm of feat] (enc) {Encode \&\\ encrypt};
  \node[stage, right=1.0cm of enc] (cmp) {Polynomial\\ comparison\\ (degree $k$, per split)};
  \node[stage, right=1.0cm of cmp] (path) {Balanced path\\ product (per leaf,\\ zero rotations)};
  \node[stage, right=1.0cm of path] (agg) {Leaf-aligned\\ aggregation\\ (rotate-and-add)};
  \node[client, right=1.0cm of agg] (dec) {Decrypt \&\\ decode};
  \node[client, right=1.0cm of dec] (score) {Margin-space\\ score};

  \draw[arr] (feat) -- (enc);
  \draw[arr] (enc) -- (cmp);
  \draw[arr] (cmp) -- (path);
  \draw[arr] (path) -- (agg);
  \draw[arr] (agg) -- (dec);
  \draw[arr] (dec) -- (score);

  \begin{scope}[on background layer]
    \node[draw=red!60, dashed, thick, rounded corners, fit=(cmp)(path)(agg),
          inner sep=8pt, label={[red!70, font=\scriptsize]above:Server -- encrypted domain, no decryption key}]
      (serverbox) {};
  \end{scope}
  \node[below=0.15cm of feat, text width=1.9cm, align=center, font=\scriptsize, gray] {Client};
  \node[below=0.15cm of score, text width=1.9cm, align=center, font=\scriptsize, gray] {Client};

  \node[below=1.15cm of cmp, text width=2.05cm, align=center, font=\scriptsize, blue!60]
    {$k\in\{3,5,7,9\}$ locked family};
  \node[below=1.15cm of path, text width=2.05cm, align=center, font=\scriptsize, blue!60]
    {depth $\lceil\log_2 D\rceil$, folds leaf coeff.};
  \node[below=1.15cm of agg, text width=2.05cm, align=center, font=\scriptsize, blue!60]
    {real, measured cost (latency results)};
\end{tikzpicture}
"""

# ---------------------------------------------------------------------------------------------
# F2 -- (redrawn 2026-10-09 as a column-wise mechanism picture) leaf-aligned packing architecture. Three panels per 06_methods_packing.tex: (i) per-level
# ciphertext packing, one slot per leaf, padded with the constant 1 for shallower trees, zero
# rotations; (ii) the balanced binary-tree path product per leaf, zero additional rotations, leaf
# value folded into the last level's plaintext coefficients; (iii) the FINAL ensemble aggregation,
# which IS a real rotate-and-add reduction over the leaf slots -- drawn explicitly per the
# correction already sent to the supervisor (docs/guide_verification_note.md): "zero rotations"
# is true only of (i) and (ii), never of (iii). Schematic tree count (4) and depth (3) for
# legibility, NOT the real 25-model/\MOneMaxLeaves{}-leaf configuration (see the manuscript
# caption for the real numbers via Table 3).
# ---------------------------------------------------------------------------------------------
F2_BODY = r"""
\definecolor{ink}{HTML}{1A1A1A}
\definecolor{inkb}{HTML}{52514E}
\definecolor{mut}{HTML}{898781}
\definecolor{blu}{HTML}{2A78D6}
\definecolor{blul}{HTML}{DCE9F8}
\definecolor{blum}{HTML}{9EC5F4}
\definecolor{padg}{HTML}{E6E5DF}
\hyphenpenalty=10000 \exhyphenpenalty=10000
\begin{tikzpicture}[
  x=1cm, y=1cm, font=\small, text=ink,
  cell/.style={draw=inkb, line width=0.5pt, minimum width=1.3cm, minimum height=0.78cm, inner sep=1pt,
               fill=blul},
  padc/.style={cell, fill=padg, text=inkb},
  rowlab/.style={anchor=east, align=right, text width=3.0cm, font=\small},
  note/.style={anchor=west, align=left, text width=3.7cm, font=\footnotesize, text=inkb},
  flow/.style={line width=1.6pt, line cap=round},
]
  % slot c is centred at x = 4.25 + 1.4 (c-1); cells are 1.3 cm wide
  \draw[decorate, decoration={brace, amplitude=3pt}, inkb] (3.6, 5.45) -- (9.1, 5.45)
    node[midway, above=3pt, text=ink] {tree 1 (four leaves)};
  \draw[decorate, decoration={brace, amplitude=3pt}, inkb] (9.2, 5.45) -- (11.9, 5.45)
    node[midway, above=3pt, text=ink] {tree 2 (two leaves)};
  \node[rowlab, text=inkb] at (3.45, 4.85) {slot (one per leaf)};
  \foreach \c in {1,...,6} {\node[text=inkb, font=\footnotesize] at ({4.25+1.4*(\c-1)}, 4.85) {\c};}

  \node[rowlab] at (3.45, 3.85) {level-1 ciphertext};
  \foreach \c/\t in {1/a, 2/a, 3/a, 4/a, 5/d, 6/d} {
    \node[cell] at ({4.25+1.4*(\c-1)}, 3.85) {$z_{\t}$};
  }
  \node[rowlab] at (3.45, 2.85) {level-2 ciphertext};
  \foreach \c/\t in {1/b, 2/b, 3/c, 4/c} {
    \node[cell] at ({4.25+1.4*(\c-1)}, 2.85) {$z_{\t}$};
  }
  \foreach \c in {5,6} {\node[padc] at ({4.25+1.4*(\c-1)}, 2.85) {\footnotesize pad};}
  \node[note] at (12.3, 3.35) {\textbf{Client.} One margin $z$ per slot and level; a padded slot
    holds the constant $1$.};

  \draw[-{Stealth[length=2.2mm]}, flow, blu] (7.75, 2.2) -- (7.75, 1.5);
  \node[anchor=west, font=\footnotesize, text=inkb, align=left, text width=4.0cm] at (8.0, 1.85)
    {server: comparison polynomial $b_k(z)$ on all slots at once};

  \node[rowlab] at (3.45, 0.85) {path product $\times$ leaf value};
  \foreach \c in {1,...,6} {
    \node[cell, fill=blum] at ({4.25+1.4*(\c-1)}, 0.85) {$p_{\c}v_{\c}$};
  }
  \node[note] at (12.3, 0.85) {\textbf{Server, no rotations.} Balanced product over the levels
    ($\lceil\log_2 D\rceil$ deep); the leaf value $v_\ell$ is folded into the last level's coefficients.};

  \draw[decorate, decoration={brace, mirror, amplitude=4pt}, inkb] (3.6, 0.3) -- (11.9, 0.3);
  \draw[-{Stealth[length=2.2mm]}, flow, blu] (7.75, -0.2) -- (7.75, -0.75);
  \node[draw=blu, line width=1.3pt, fill=blum, minimum width=5.4cm, minimum height=0.8cm,
        rounded corners=2pt] at (7.75, -1.3) {encrypted score $\sum_\ell p_\ell v_\ell$};
  \node[note] at (12.3, -1.3) {\textbf{Server, rotations.} Rotate-and-add over the slots: the only
    step that rotates.};
\end{tikzpicture}
"""


def write(name: str, body: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + body + POSTAMBLE)
    print(f"wrote {path}")
    return path


def main() -> int:
    write("f1_pipeline.tex", F1_BODY)
    write("f2_packing.tex", F2_BODY)
    print("Note: these are TikZ SOURCES, not compiled figures. Compile with "
          "paper/build_diagrams.ps1 (Windows-side pdflatex, same cross-OS split as "
          "paper/build_paper.ps1) to produce paper_assets/figures/f1_pipeline.pdf and "
          "f2_packing.pdf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
