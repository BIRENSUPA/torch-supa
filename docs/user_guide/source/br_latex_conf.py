# Shared LaTeX/PDF configuration for BR-style Sphinx documents.
# Import these variables from conf.py to reuse the same PDF style across projects.

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGAL_NOTICE_MD = _REPO_ROOT / 'docs/user_guide/source/_static/legalnotice.md'


def _latex_escape(text):
    return (text.replace('\\', r'\textbackslash{}')
            .replace('#', r'\#')
            .replace('$', r'\$')
            .replace('%', r'\%')
            .replace('&', r'\&')
            .replace('_', r'\_')
            .replace('{', r'\{')
            .replace('}', r'\}')
            .replace('~', r'\textasciitilde{}')
            .replace('^', r'\textasciicircum{}'))


def _build_legal_notice_latex():
    try:
        legal_notice = _LEGAL_NOTICE_MD.read_text(encoding='utf-8').lstrip('﻿').strip()
    except OSError:
        return ''

    if not legal_notice:
        return ''

    lines = [r'\clearpage', r'\phantomsection']
    title_added = False
    for line in legal_notice.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append('')
            continue
        if stripped.startswith('## '):
            title = _latex_escape(stripped[3:].strip())
            lines.extend([
                rf'\chapter*{{{title}}}',
                rf'\addcontentsline{{toc}}{{chapter}}{{{title}}}',
                rf'\markboth{{{title}}}{{}}',
            ])
            title_added = True
            continue
        if stripped.startswith('**') and stripped.endswith('**') and len(stripped) > 4:
            subtitle = _latex_escape(stripped[2:-2].strip())
            lines.append(rf'\noindent{{\bfseries\textcolor{{black}}{{{subtitle}}}}}\par')
            continue
        lines.append(_latex_escape(stripped) + r'\par')

    if not title_added:
        lines.insert(2, r'\chapter*{法律声明}')
        lines.insert(3, r'\addcontentsline{toc}{chapter}{法律声明}')
        lines.insert(4, r'\markboth{法律声明}{}')
    return '\n'.join(lines) + '\n'


latex_engine = 'xelatex'
latex_toc_depth = 7
latex_secnum_depth = 7

latex_additional_files = [
    '_static/logo.png',
    '_static/main.png',
]

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'passoptionstopackages': r'\PassOptionsToPackage{table}{xcolor}',
    'babel': r'\usepackage[english]{babel}\usepackage[UTF8,fontset=none]{ctex}',
    'polyglossia': '',
    'fontpkg': r'''
        \IfFontExistsTF{Microsoft YaHei}{
            \setmainfont{Microsoft YaHei}[
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
            \setsansfont{Microsoft YaHei}[
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
            \setmonofont{Microsoft YaHei}[
              Scale          = 0.9,
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
            \setCJKmainfont{Microsoft YaHei}[
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
            \setCJKsansfont{Microsoft YaHei}[
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
            \setCJKmonofont{Microsoft YaHei}[
              Scale          = 0.9,
              BoldFont       = Microsoft YaHei Bold,
              ItalicFont     = Microsoft YaHei,
              BoldItalicFont = Microsoft YaHei Bold,
              AutoFakeBold   = 2.5,
              AutoFakeSlant  = 0.2
            ]
        }{
            \PackageError{brlatex}{Microsoft YaHei font is required}{Install Microsoft YaHei and rebuild the PDF.}
        }
    ''',
    'maketitle': r'''
        \renewcommand{\suasdoctitle}{壁仞™ Torch SUPA 用户指南}
        \markboth{\suasdoctitle}{}
        \begin{titlepage}
        \thispagestyle{empty}
        \begin{flushleft}
        \IfFileExists{logo.png}{
            \includegraphics[width=0.25\textwidth]{logo.png}
        }{}
        \end{flushleft}

        \vspace*{4.6cm}
        \begin{center}
        {\fontsize{26pt}{32pt}\selectfont\bfseries\textcolor{black}{壁仞™ Torch SUPA 用户指南}\par}
        \vspace{0.9cm}
        {\Large\textcolor{black}{文档版本：01}\par}
        \vspace{0.35cm}
        {\Large\textcolor{black}{发布日期：\today}\par}
        \end{center}

        \vfill
        \begin{center}
        {\Large\textcolor{black}{壁仞科技}}
        \end{center}
        \end{titlepage}
        \clearpage
        \pagenumbering{roman}
        \setcounter{page}{1}
    ''',
    'preamble': r'''
        \usepackage{graphicx}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{multirow}
        \usepackage{colortbl}
        \usepackage{float}
        \usepackage{tocloft}
        \usepackage[most]{tcolorbox}
        \usepackage{fancyhdr}
        \usepackage{enumitem}
        \usepackage{url}
        \usepackage{caption}
        \usepackage{titlesec}
        \usepackage{indentfirst}
        \usepackage{fancyvrb}
        \usepackage{upquote}
        \usepackage{etoolbox}

        \geometry{left=1.65cm,right=1.65cm,top=2.2cm,bottom=2.2cm}
        \setlistdepth{15}
        \setlength{\headheight}{15pt}
        \addtolength{\topmargin}{-2.63403pt}
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{9pt plus 2pt minus 1pt}
        \setlength{\emergencystretch}{6em}
        \linespread{1.2}
        \urlstyle{same}
        \def\UrlBreaks{\do\/\do-\do_}
        \setcounter{tocdepth}{7}
        \setcounter{secnumdepth}{7}
        \let\cleardoublepage\clearpage

        \definecolor{titleblue}{RGB}{0,102,204}
        \definecolor{textbody}{RGB}{51,51,51}
        \definecolor{inlinecodebg}{gray}{0.93}
        \definecolor{codebg}{gray}{0.94}
        \definecolor{codeframe}{gray}{0.72}
        \definecolor{tabheadgray}{gray}{0.80}
        \definecolor{tabrowodd}{gray}{0.945}
        \definecolor{tabroweven}{gray}{1}
        \definecolor{quotebg}{gray}{0.95}
        \definecolor{quotebd}{gray}{0.60}
        \definecolor{calloutnotebg}{HTML}{E6F0FA}
        \definecolor{calloutnotebd}{HTML}{1E6F9F}
        \definecolor{calloutwarnbg}{HTML}{FFF4D6}
        \definecolor{calloutwarnbd}{HTML}{D97904}
        \definecolor{calloutimpbg}{HTML}{F2EAFE}
        \definecolor{calloutimpbd}{HTML}{6A1B9A}

        \sphinxsetup{
            TitleColor={RGB}{0,102,204},
            InnerLinkColor={RGB}{0,102,204},
            OuterLinkColor={RGB}{0,102,204},
            VerbatimColor={gray}{0.94},
            VerbatimBorderColor={gray}{0.72},
            verbatimwithframe=true,
            verbatimborder=0.45pt,
            verbatimvisiblespace={ },
            pre_border-radius=0pt,
            pre_padding=5pt,
            pre_box-shadow=none,
            TableRowColorHeader={gray}{0.80},
            TableRowColorOdd={gray}{0.945},
            TableRowColorEven={gray}{1},
            booktabscolorgaps=false,
            noteBgColor={HTML}{E6F0FA},
            noteBorderColor={HTML}{1E6F9F},
            warningBgColor={HTML}{FFF4D6},
            warningBorderColor={HTML}{D97904},
            cautionBgColor={HTML}{FFF4D6},
            cautionBorderColor={HTML}{D97904},
            attentionBgColor={HTML}{FDECEC},
            attentionBorderColor={HTML}{B3261E},
            hintBgColor={HTML}{E6F7E6},
            hintBorderColor={HTML}{2E7D32},
            importantBgColor={HTML}{F2EAFE},
            importantBorderColor={HTML}{6A1B9A}
        }

        \renewcommand{\thechapter}{\arabic{chapter}.}
        \renewcommand{\thesection}{\arabic{chapter}.\arabic{section}.}
        \renewcommand{\thesubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.}
        \renewcommand{\thesubsubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.}
        \renewcommand{\theparagraph}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.\arabic{paragraph}.}
        \renewcommand{\thesubparagraph}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.\arabic{paragraph}.\arabic{subparagraph}.}

        \titleformat{\chapter}[hang]
            {\normalfont\huge\bfseries\color{titleblue}}
            {\thechapter}
            {0.5em}
            {\Huge}
        \titleformat{\section}{\color{titleblue}\Large\bfseries}{\thesection}{0.5em}{}
        \titleformat{\subsection}{\color{titleblue}\large\bfseries}{\thesubsection}{0.5em}{}
        \titleformat{\subsubsection}{\color{titleblue}\normalsize\bfseries}{\thesubsubsection}{0.5em}{}
        \titleformat{\paragraph}[block]{\color{titleblue}\normalsize\bfseries}{\theparagraph}{0.5em}{}
        \titleformat{\subparagraph}[block]{\color{titleblue}\normalsize\bfseries}{\thesubparagraph}{0.5em}{}

        \makeatletter
        \newcounter{subsubparagraph}[subparagraph]
        \renewcommand{\thesubsubparagraph}{\thesubparagraph\arabic{subsubparagraph}.}
        \newcommand{\subsubparagraph}[1]{%
          \refstepcounter{subsubparagraph}%
          \addcontentsline{toc}{subsubparagraph}{\protect\numberline{\thesubsubparagraph}#1}%
          \par\smallskip
          {\color{titleblue}\normalsize\bfseries\thesubsubparagraph\hspace{0.5em}#1}\par\nobreak\smallskip
        }
        \newcommand{\l@subsubparagraph}{\@dottedtocline{6}{15em}{5em}}
        \newcounter{subsubsubparagraph}[subsubparagraph]
        \renewcommand{\thesubsubsubparagraph}{\thesubsubparagraph\arabic{subsubsubparagraph}.}
        \newcommand{\subsubsubparagraph}[1]{%
          \refstepcounter{subsubsubparagraph}%
          \addcontentsline{toc}{subsubsubparagraph}{\protect\numberline{\thesubsubsubparagraph}#1}%
          \par\smallskip
          {\color{titleblue}\normalsize\bfseries\thesubsubsubparagraph\hspace{0.5em}#1}\par\nobreak\smallskip
        }
        \newcommand{\l@subsubsubparagraph}{\@dottedtocline{7}{18.5em}{6em}}
        \providecommand*{\toclevel@section}{1}
        \providecommand*{\toclevel@subsection}{2}
        \providecommand*{\toclevel@subsubsection}{3}
        \providecommand*{\toclevel@paragraph}{4}
        \providecommand*{\toclevel@subparagraph}{5}
        \providecommand*{\toclevel@subsubparagraph}{6}
        \providecommand*{\toclevel@subsubsubparagraph}{7}
        \renewcommand{\sphinxtableofcontentshook}{%
          \renewcommand{\contentsname}{目录}%
          \setlength{\cftbeforesecskip}{2pt}%
          \setlength{\cftbeforesubsecskip}{2pt}%
          \setlength{\cftbeforesubsubsecskip}{1.5pt}%
          \renewcommand*\l@section{\@dottedtocline{1}{1.5em}{2.6em}}%
          \renewcommand*\l@subsection{\@dottedtocline{2}{4.1em}{3.5em}}%
          \renewcommand*\l@subsubsection{\@dottedtocline{3}{7.6em}{4.2em}}%
          \renewcommand*\l@paragraph{\@dottedtocline{4}{10.8em}{4.6em}}%
          \renewcommand*\l@subparagraph{\@dottedtocline{5}{13.2em}{5em}}%
        }
        \renewcommand{\sphinxtableofcontents}{%
          \pagenumbering{roman}%
          \begingroup
            \parskip \z@skip
            \sphinxtableofcontentshook
            \tableofcontents
          \endgroup
          \clearpage
          \pagenumbering{arabic}%
          \setcounter{page}{1}%
          \markboth{\suasdoctitle}{}%
        }
        \makeatother

        \renewenvironment{quote}{%
          \begin{tcolorbox}[
            enhanced, breakable, boxrule=0pt, arc=1.5mm,
            colback=quotebg,
            borderline west={2.5pt}{0pt}{quotebd},
            left=8pt, right=8pt, top=6pt, bottom=6pt,
            before upper={},
            after upper={},
          ]%
        }{%
          \end{tcolorbox}%
        }

        \newenvironment{suasbrcallout}[3]{%
          \begin{tcolorbox}[
            enhanced, breakable, boxrule=0pt, arc=2mm,
            colback=#1,
            borderline west={2.2pt}{0pt}{#2},
            left=8pt, right=8pt, top=7pt, bottom=7pt,
            before upper={\sphinxstrong{#3}\par\smallskip},
          ]%
        }{%
          \end{tcolorbox}%
        }
        \renewenvironment{sphinxnote}[1]
          {\begin{suasbrcallout}{calloutnotebg}{calloutnotebd}{#1}}
          {\end{suasbrcallout}}
        \renewenvironment{sphinxwarning}[1]
          {\begin{suasbrcallout}{calloutwarnbg}{calloutwarnbd}{#1}}
          {\end{suasbrcallout}}
        \renewenvironment{sphinximportant}[1]
          {\begin{suasbrcallout}{calloutimpbg}{calloutimpbd}{#1}}
          {\end{suasbrcallout}}

        \newcommand{\suasdoctitle}{壁仞™ Torch SUPA 用户指南}
        \makeatletter
        \renewcommand*\sectionmark[1]{\markboth{#1}{}}
        \renewcommand*\subsectionmark[1]{}
        \makeatother
        \fancyhf{}
        \fancyhead[R]{\suasdoctitle}
        \fancyfoot[L]{\leftmark}
        \fancyfoot[C]{\thepage}
        \renewcommand{\headrulewidth}{0.4pt}
        \renewcommand{\footrulewidth}{0.4pt}
        \fancypagestyle{plain}{%
          \fancyhf{}%
          \fancyhead[R]{\suasdoctitle}%
          \fancyfoot[L]{\leftmark}%
          \fancyfoot[C]{\thepage}%
          \renewcommand{\headrulewidth}{0.4pt}%
          \renewcommand{\footrulewidth}{0.4pt}%
        }
        \fancypagestyle{normal}{%
          \fancyhf{}%
          \fancyhead[R]{\suasdoctitle}%
          \fancyfoot[L]{\leftmark}%
          \fancyfoot[C]{\thepage}%
          \renewcommand{\headrulewidth}{0.4pt}%
          \renewcommand{\footrulewidth}{0.4pt}%
        }

        \setlist[itemize]{itemsep=6pt, topsep=6pt, parsep=2pt, leftmargin=2em}
        \setlist[enumerate]{itemsep=6pt, topsep=6pt, parsep=2pt, leftmargin=2em}
        \setlength{\tabcolsep}{8pt}
        \renewcommand{\arraystretch}{1.35}
        \setlength{\LTleft}{3pt}
        \setlength{\LTright}{3pt}
        \setlength{\tymin}{4em}
        \setlength{\tymax}{0.92\linewidth}
        \renewcommand{\sphinxcolorpanelextraoverhang}{0pt}
        \renewcommand{\sphinxbooktabscolorgapsoverhang}{0pt}
        \makeatletter
        \def\sphinxbottomrule{\noalign{\sphinxrowcolorOFF}\bottomrule}
        \def\sphinxbooktabsbottomrule{\noalign{\sphinxrowcolorOFF}\bottomrule}
        \newcolumntype{\X}[2]{>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr
              (\linewidth-6pt-\spx@arrayrulewidth)*#1/#2-\tw@\tabcolsep-\spx@arrayrulewidth\relax}}
        \newcolumntype{\Y}[1]{>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr
              #1\dimexpr\linewidth-6pt-\spx@arrayrulewidth\relax-\tw@\tabcolsep-\spx@arrayrulewidth\relax}}
        \makeatother
        \AtBeginEnvironment{longtable}{%
          \setlength{\LTleft}{3pt}%
          \setlength{\LTright}{3pt}%
          \small%
          \sloppy%
        }
        \AtBeginEnvironment{tabular}{%
          \small%
          \sloppy%
        }
        \AtEndEnvironment{longtable}{\fussy}
        \AtEndEnvironment{tabular}{\fussy}

        \makeatletter
        \def\maxwidth{\ifdim\Gin@nat@width>0.7\linewidth 0.7\linewidth\else\Gin@nat@width\fi}
        \def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
        \makeatother
        \setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
        \floatplacement{figure}{H}

        \fvset{
            fontsize=\footnotesize,
            formatcom=\color{black},
            frame=none,
            numbers=none,
            xleftmargin=0pt,
            xrightmargin=0pt,
            commandchars=\\\{\}
        }
        \makeatletter
        \AtBeginEnvironment{sphinxVerbatim}{%
          \advance\linewidth\dimexpr-\spx@pre@padding@left-\spx@pre@padding@right-\spx@pre@border@left-\spx@pre@border@right\relax
        }
        \long\def\spx@verb@FrameCommand #1#2#3{%
          \hskip\@totalleftmargin
          \spx@verb@fcolorbox {#1}{#2}{#3}%
          \hskip-\dimexpr\linewidth+\spx@boxes@border@left+\spx@boxes@padding@left+\spx@boxes@padding@right+\spx@boxes@border@right\relax
          \hskip-\@totalleftmargin \hskip\columnwidth
        }%
        \define@key{FV}{numbers}{\def\FV@Numbers{none}}
        \makeatother

        \captionsetup[table]{name=表}
        \captionsetup[figure]{name=图}
        \renewcommand\contentsname{目录}
        \renewcommand\listtablename{表格目录}
        \renewcommand\listfigurename{插图目录}
        \renewcommand\tablename{表}
        \renewcommand\figurename{图}

        \AtBeginDocument{%
          \color{textbody}%
          \hypersetup{bookmarksnumbered=true}%
          \fancyhf{}%
          \fancyhead[R]{\suasdoctitle}%
          \fancyfoot[L]{\leftmark}%
          \fancyfoot[C]{\thepage}%
          \pagestyle{normal}%
        }
    ''',
    'atendofbody': _build_legal_notice_latex(),
}
