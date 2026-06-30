# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

autosummary_generate = True
autosummary_generate_overwrite = True
# 添加 PyTorch 源码路径，使 autodoc 能够导入
# sys.path.insert(0, os.path.abspath('../../../torch_supa'))
sys.path.insert(0, os.path.abspath('../../../'))
project = 'BRPYTORCH'
copyright = '2025, develop'
author = 'develop'
release = '1.0'


master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

latex_use_xindy = False
language = 'en'

extensions = [
    'sphinx.ext.autodoc',        # 自动提取 Python API（原生支持）
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',       # 支持 Google 风格 docstrings
    'sphinx_rtd_theme',          # 美观的主题
    'exhale',   # 关键：自动组织 API 文档结构
    'breathe',                   # 连接 Doxygen 和 Sphinx
    'sphinx.ext.imgmath',  # 或 'sphinx.ext.mathjax'
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
]
primary_domain = 'cpp'
breathe_default_domain = 'cpp'


breathe_domain_by_extension = {
    "h": "cpp",
    "hpp": "cpp",
    "c": "cpp",
    "cpp": "cpp",
    "cc": "cpp",
}

breathe_projects = {'BRPYTORCH': '../../doxygen/xml'}
breathe_default_project = 'BRPYTORCH'
latex_engine = 'xelatex'  # 必须添加这一行
latex_toc_depth = 2
latex_secnum_depth = 2

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
        % 允许代码块自动换行（核心包）
        \usepackage{listings}
        \usepackage{courier}  % 代码字体（保持一致性）

        % 配置 listings 包：让代码自动换行
        \lstset{
            breaklines=true,        % 启用自动换行
            breakatwhitespace=false, % 允许在非空格处换行（适配长标识符）
            postbreak=\raisebox{0ex}[0ex][0ex]{\ensuremath{\hookrightarrow\space}} % 换行后添加箭头标记
        }

        \usepackage{times}
        \usepackage{graphicx}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{multirow}
        \usepackage{xcolor}
        \usepackage{float}
        \usepackage{ctex}

        % 修复 fancyhdr 警告
        \usepackage{fancyhdr}
        \setlength{\headheight}{15pt}
        \addtolength{\topmargin}{-2.63403pt}

        % 使用 enumitem 包的正确命令
        \usepackage{enumitem}
        \setlistdepth{15}  % 使用 setlistdepth 而不是 setcounter{maxlistdepth}

        % 配置页眉页脚
        \pagestyle{fancy}
        \fancyhf{}
        \fancyhead[LE,RO]{\thepage}
        \fancyhead[RE]{\nouppercase{\leftmark}}
        \fancyhead[LO]{\nouppercase{\rightmark}}

        % 章节页样式
        \fancypagestyle{plain}{
            \fancyhf{}
            \fancyfoot[C]{\thepage}
            \renewcommand{\headrulewidth}{0pt}
        }

        % 改进代码和 URL 的断行
        \usepackage{url}
        \urlstyle{same}
        \def\UrlBreaks{\do\/\do-}

        % 限制目录深度
        \setcounter{tocdepth}{2}
        \setcounter{secnumdepth}{2}

        % 其他配置
        \usepackage{fancyvrb}
        \usepackage{upquote}
    ''',
    # ... 其他配置
}

exhale_args = {
    "containmentFolder": "./api",  # 生成的 API 文档放在 docs/source/api/ 目录
    "rootFileName": "api_index.rst",  # 生成的 API 首页文件名
    "rootFileTitle": "BR PYTORCH API",  # API 首页的标题（会显示在 PDF 中）
    "doxygenStripFromPath": "../../../",  # 去掉代码路径中的冗余部分（和 Doxyfile 的 INPUT 对应）
    # 添加这些关键配置
    "verboseBuild": False,  # 设为False减少输出噪音
    "generateBreatheFileDirectives": True,

    # 指定要处理的文件类型
    "exhaleExecutesDoxygen": False,  # 假设您已单独运行Doxygen
    "exhaleDoxygenStdin": "",  # 如果有自定义Doxygen配置

    # C++ 特定配置
    "listingExclude": [r".*\.md$"],  # 排除markdown文件
    "fullApiSubSectionTitle": "Full API",

    # 处理宏定义问题
    "createTreeView": True,  # 简化树视图

}


# 自定义配置
latex_documents = [
    (
        "index",  # 自定义入口 .rst 文件（仅包含 API 内容）
        "torch.tex",  # 中间 LaTeX 文件名 → 最终 PDF 名
        "BRPYTORCH",  # PDF 标题
        "BRPyTorch Team",  # 作者
        "manual"  # 文档类别
    ),
]

autodoc_typehints = 'description'
add_module_names = False
