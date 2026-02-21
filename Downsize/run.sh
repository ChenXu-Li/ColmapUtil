#!/bin/bash
# ============================================================
# 图片下采样脚本
# Image Downsampling Script
# ============================================================

set -e  # 遇到错误时退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 打印函数
print_header() {
    echo -e "\n${BOLD}${BLUE}============================================================${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}============================================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config FILE    配置文件路径 (默认: config.yaml)"
    echo "  -i, --input DIR      输入目录"
    echo "  -o, --output DIR     输出目录"
    echo "  -f, --factor N       下采样因子 (2, 4, 8...)"
    echo "  -w, --workers N      并行线程数"
    echo "  -r, --recursive      递归处理子目录"
    echo "  --overwrite          覆盖已存在的文件"
    echo "  -v, --verbose        显示详细信息"
    echo "  -q, --quiet          静默模式"
    echo "  -h, --help           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置文件"
    echo "  $0 -c myconfig.yaml                   # 使用指定配置文件"
    echo "  $0 -i ./images -o ./images_2x -f 2   # 命令行模式"
    echo "  $0 -f 4 --overwrite                   # 4x下采样，覆盖已存在文件"
    echo ""
}

# 检查 Python 环境
check_python() {
    print_info "检查 Python 环境..."
    
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            print_error "未找到 Python，请安装 Python 3.7+"
            exit 1
        fi
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"
    fi
    
    # 检查 Python 版本
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python 版本: $PYTHON_VERSION"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    local missing_deps=()
    
    # 检查 opencv
    if ! $PYTHON_CMD -c "import cv2" 2>/dev/null; then
        missing_deps+=("opencv-python")
    fi
    
    # 检查 yaml
    if ! $PYTHON_CMD -c "import yaml" 2>/dev/null; then
        missing_deps+=("pyyaml")
    fi
    
    # 检查 tqdm
    if ! $PYTHON_CMD -c "import tqdm" 2>/dev/null; then
        missing_deps+=("tqdm")
    fi
    
    # 检查 numpy
    if ! $PYTHON_CMD -c "import numpy" 2>/dev/null; then
        missing_deps+=("numpy")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warn "缺少以下依赖: ${missing_deps[*]}"
        echo ""
        read -p "是否自动安装? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装依赖..."
            pip install ${missing_deps[*]}
        else
            print_error "请手动安装依赖: pip install ${missing_deps[*]}"
            exit 1
        fi
    else
        print_info "所有依赖已安装"
    fi
}

# 主函数
main() {
    print_header "📸 图片下采样工具"
    
    # 解析参数
    ARGS=""
    CONFIG_FILE="config.yaml"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                ARGS="$ARGS --config $2"
                shift 2
                ;;
            -i|--input)
                ARGS="$ARGS --input $2"
                shift 2
                ;;
            -o|--output)
                ARGS="$ARGS --output $2"
                shift 2
                ;;
            -f|--factor)
                ARGS="$ARGS --factor $2"
                shift 2
                ;;
            -w|--workers)
                ARGS="$ARGS --workers $2"
                shift 2
                ;;
            -r|--recursive)
                ARGS="$ARGS --recursive"
                shift
                ;;
            --overwrite)
                ARGS="$ARGS --overwrite"
                shift
                ;;
            -v|--verbose)
                ARGS="$ARGS --verbose"
                shift
                ;;
            -q|--quiet)
                ARGS="$ARGS --quiet"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查环境
    check_python
    check_dependencies
    
    # 检查配置文件
    if [[ ! -f "$CONFIG_FILE" && -z "$ARGS" ]]; then
        print_warn "配置文件不存在: $CONFIG_FILE"
        print_info "使用默认配置或命令行参数"
    fi
    
    # 运行 Python 脚本
    echo ""
    print_info "启动下采样处理..."
    echo ""
    
    $PYTHON_CMD downsize.py $ARGS
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "处理完成！"
    else
        echo ""
        print_error "处理失败，退出码: $EXIT_CODE"
        exit $EXIT_CODE
    fi
}

# 运行
main "$@"
