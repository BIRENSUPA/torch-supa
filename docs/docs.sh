#!/bin/bash
# docs.sh - 文档生成脚本
# 用法: ./docs.sh [install|all|help]

set -e  

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

show_help() {
    echo -e "${BLUE}文档生成脚本使用方法:${NC}"
    echo "  ./docs.sh install    安装必要的依赖环境"
    echo "  ./docs.sh all        安装环境并生成PDF文档"
    echo "  ./docs.sh            仅生成PDF文档"
    echo "  ./docs.sh help       显示此帮助信息"
    echo ""
    echo "注意：无参数运行将仅生成PDF文档"
}

install_dependencies() {
    echo "请在有root权限的用户下执行此操作！！！"
    apt-get update
    apt-get install doxygen-latex doxygen-doc doxygen-gui graphviz wget unzip fontconfig -y
    apt install latexmk -y
    apt install texlive-full -y
    apt install xindy -y
    pip install sphinx breathe rst2pdf sphinx_rtd_theme exhale

    echo "安装思源雅黑字体（Source Han Sans SC）"
    font_dir="/usr/local/share/fonts/source-han-sans-sc"
    tmp_dir="$(mktemp -d)"
    mkdir -p "${font_dir}"
    wget -O "${tmp_dir}/SourceHanSansSC.zip" "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip"
    unzip -o "${tmp_dir}/SourceHanSansSC.zip" -d "${tmp_dir}/SourceHanSansSC"
    find "${tmp_dir}/SourceHanSansSC" -name "*.otf" -exec cp {} "${font_dir}/" \;
    fc-cache -fv
    rm -rf "${tmp_dir}"
}

generate_pdf(){
    cd ${DIR}/doxygen
    doxygen Doxygen
    cd ${DIR}/sphinx
    make latexpdf
    cd ${DIR}/user_guide
    make html
}

clean_docs() {
    echo "清理所有生成的文档"
    cd ${DIR}/doxygen
    rm -rf xml 
    cd ${DIR}/sphinx
    rm -rf build source/api
    cd ${DIR}/user_guide
    rm -rf build
}

main() {
    case "$1" in
        "install")
            install_dependencies
            ;;
        "all")
            install_dependencies
            clean_docs
            generate_pdf
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "")
            clean_docs
            generate_pdf
            ;;
    esac
}
main "$@"
