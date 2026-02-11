#!/bin/bash
#
# 普通图像（针孔相机）重建启动脚本
# 适用于标准透视相机图像
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../configs/config_pinhole.yaml"

# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 显示使用说明
show_usage() {
    cat << EOF
普通图像（针孔相机）重建脚本

用法: $0 [选项]

选项:
  -i, --input PATH        输入图像目录（如果未指定，将从配置文件读取）
  -o, --output PATH       输出路径（如果未指定，将从配置文件读取）
  -c, --config PATH       配置文件路径（默认: configs/config_pinhole.yaml）
  -m, --matcher TYPE      匹配器类型 (exhaustive|sequential|spatial|vocabtree, 默认: exhaustive)
  --sparse-only           仅执行稀疏重建
  --dense-only            仅执行稠密重建（需要已有稀疏重建结果）
  -h, --help              显示此帮助信息

示例:
  # 使用配置文件中的路径（推荐）
  $0 --sparse-only

  # 完整重建流程（命令行指定路径）
  $0 -i /path/to/images -o /path/to/output

  # 仅稀疏重建（命令行指定路径）
  $0 -i /path/to/images -o /path/to/output --sparse-only

  # 仅稠密重建（从配置文件读取输出路径）
  $0 --dense-only

EOF
}

# 解析参数
INPUT_PATH=""
OUTPUT_PATH=""
CONFIG_PATH="$CONFIG_FILE"
MATCHER="exhaustive"
SPARSE_ONLY=false
DENSE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--matcher)
            MATCHER="$2"
            shift 2
            ;;
        --sparse-only)
            SPARSE_ONLY=true
            shift
            ;;
        --dense-only)
            DENSE_ONLY=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 从配置文件读取路径（如果命令行未指定）
if [ -z "$OUTPUT_PATH" ] || [ "$DENSE_ONLY" = false -a -z "$INPUT_PATH" ]; then
    # 使用Python读取配置文件中的路径
    PYTHON_READ_CONFIG="$PROJECT_ROOT/scripts/utils/read_config.py"
    if [ ! -f "$PYTHON_READ_CONFIG" ]; then
        # 如果读取脚本不存在，创建临时脚本
        cat > /tmp/read_config.py << 'PYEOF'
import yaml
import sys
config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
paths = config.get('paths', {})
print(f"INPUT:{paths.get('input_images', '')}")
print(f"OUTPUT:{paths.get('output_path', '')}")
PYEOF
        PYTHON_READ_CONFIG="/tmp/read_config.py"
    fi
    
    CONFIG_PATHS=$(python "$PYTHON_READ_CONFIG" "$CONFIG_PATH" 2>/dev/null || echo "")
    if [ -n "$CONFIG_PATHS" ]; then
        CONFIG_INPUT=$(echo "$CONFIG_PATHS" | grep "^INPUT:" | cut -d: -f2-)
        CONFIG_OUTPUT=$(echo "$CONFIG_PATHS" | grep "^OUTPUT:" | cut -d: -f2-)
        
        if [ -z "$INPUT_PATH" ] && [ -n "$CONFIG_INPUT" ]; then
            INPUT_PATH="$CONFIG_INPUT"
            echo "从配置文件读取输入路径: $INPUT_PATH"
        fi
        
        if [ -z "$OUTPUT_PATH" ] && [ -n "$CONFIG_OUTPUT" ]; then
            OUTPUT_PATH="$CONFIG_OUTPUT"
            echo "从配置文件读取输出路径: $OUTPUT_PATH"
        fi
    fi
fi

# 检查必需参数
if [ -z "$OUTPUT_PATH" ]; then
    echo "错误: 必须指定输出路径 (-o 或在配置文件的 paths.output_path 中指定)"
    show_usage
    exit 1
fi

if [ "$DENSE_ONLY" = false ] && [ -z "$INPUT_PATH" ]; then
    echo "错误: 稀疏重建需要指定输入图像目录 (-i 或在配置文件的 paths.input_images 中指定)"
    show_usage
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查Python脚本
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/reconstruction/pinhole_reconstruction.py"
DENSE_SCRIPT="$PROJECT_ROOT/scripts/reconstruction/dense_reconstruction_cli.py"

if [ "$DENSE_ONLY" = false ]; then
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
        echo "请确保已创建普通图像重建脚本"
        exit 1
    fi
else
    if [ ! -f "$DENSE_SCRIPT" ]; then
        echo "错误: 稠密重建脚本不存在: $DENSE_SCRIPT"
        exit 1
    fi
fi

echo "=========================================="
echo "普通图像（针孔相机）重建"
echo "=========================================="
echo "输入图像: $INPUT_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "配置文件: $CONFIG_PATH"
echo "匹配器: $MATCHER"
echo "=========================================="
echo ""

# 执行重建
if [ "$DENSE_ONLY" = true ]; then
    echo "执行稠密重建..."
    # 从配置文件读取稠密重建参数
    QUALITY=$(python -c "import yaml; f=open('$CONFIG_PATH'); c=yaml.safe_load(f); print(c.get('dense_reconstruction', {}).get('patch_match', {}).get('quality', 'medium'))" 2>/dev/null || echo "medium")
    MAX_IMAGE_SIZE=$(python -c "import yaml; f=open('$CONFIG_PATH'); c=yaml.safe_load(f); d=c.get('dense_reconstruction', {}); print(d.get('undistortion', {}).get('max_image_size') or d.get('patch_match', {}).get('max_image_size', 3200))" 2>/dev/null || echo "3200")
    
    python "$DENSE_SCRIPT" \
        --workspace_path "$OUTPUT_PATH" \
        --quality "$QUALITY" \
        --max_image_size "$MAX_IMAGE_SIZE"
elif [ "$SPARSE_ONLY" = true ]; then
    echo "执行稀疏重建..."
    python "$PYTHON_SCRIPT" \
        --input_image_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --config "$CONFIG_PATH" \
        --matcher "$MATCHER" \
        --sparse-only
else
    echo "执行完整重建流程（稀疏+稠密）..."
    python "$PYTHON_SCRIPT" \
        --input_image_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --config "$CONFIG_PATH" \
        --matcher "$MATCHER"
fi

echo ""
echo "=========================================="
echo "重建完成！"
echo "=========================================="
