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
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath('../../../'))
project = u'Torch SUPA 用户指南'
copyright = u'2026, 壁仞科技'
author = u'壁仞科技'
release = '1.0'


master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

latex_use_xindy = False
language = 'zh_CN'

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    # 'exhale',
    # 'breathe',
    'sphinx.ext.imgmath',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx_rtd_theme'
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

breathe_projects = {'TORCH_SUPA': '../../doxygen/xml'}
breathe_default_project = 'TORCH_SUPA'
from br_latex_conf import (  # noqa: F401
    latex_additional_files,
    latex_elements,
    latex_engine,
    latex_secnum_depth,
    latex_toc_depth,
)

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_with_keys': True,
    'sticky_navigation': True,
}
html_static_path = ['_static']
# Add navigation bar localtoc to the right
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
        'localtoc.html',
    ]
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
        "壁仞_Torch_SUPA_用户指南.tex",  # 中间 LaTeX 文件名 → 最终 PDF 名
        "壁仞™ Torch SUPA 用户指南",  # PDF 标题
        "壁仞科技",  # 作者
        "manual",  # 文档类别
        True
    ),
]

autodoc_typehints = 'description'
add_module_names = False
