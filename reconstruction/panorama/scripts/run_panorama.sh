#!/bin/bash
#
# 全景图重建启动脚本
# 适用于360°全景图像，使用虚拟相机组（Rig）
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../configs/config_panorama.yaml"

# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 显示使用说明
show_usage() {
    cat << EOF
全景图重建脚本

用法: $0 [选项]

选项:
  -i, --input PATH        输入全景图目录（必需）
  -o, --output PATH       输出路径（必需）
  -c, --config PATH       配置文件路径（默认: configs/config_panorama.yaml）
  -m, --matcher TYPE      匹配器类型 (sequential|exhaustive|vocabtree|spatial, 默认: sequential)
  -t, --render-type TYPE  渲染类型 (overlapping|non-overlapping, 默认: overlapping)
  --sparse-only           仅执行稀疏重建
  --dense-only            仅执行稠密重建（需要已有稀疏重建结果）
  -h, --help              显示此帮助信息

示例:
  # 完整重建流程
  $0 -i /path/to/panoramas -o /path/to/output

  # 仅稀疏重建
  $0 -i /path/to/panoramas -o /path/to/output --sparse-only

  # 仅稠密重建
  $0 -o /path/to/output --dense-only

注意:
  - 输入图像必须是360°全景图（宽高比2:1）
  - 输出格式包含Rig配置，与普通图像重建不同

EOF
}

# 解析参数
INPUT_PATH=""
OUTPUT_PATH=""
CONFIG_PATH="$CONFIG_FILE"
MATCHER="sequential"
RENDER_TYPE="overlapping"
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
        -t|--render-type)
            RENDER_TYPE="$2"
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

# 检查必需参数
if [ -z "$OUTPUT_PATH" ]; then
    echo "错误: 必须指定输出路径 (-o)"
    show_usage
    exit 1
fi

if [ "$DENSE_ONLY" = false ] && [ -z "$INPUT_PATH" ]; then
    echo "错误: 稀疏重建需要指定输入全景图目录 (-i)"
    show_usage
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查Python脚本
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/reconstruction/sparse_reconstruction.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "全景图重建"
echo "=========================================="
echo "输入全景图: $INPUT_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "配置文件: $CONFIG_PATH"
echo "匹配器: $MATCHER"
echo "渲染类型: $RENDER_TYPE"
echo "=========================================="
echo ""

# 执行重建
if [ "$DENSE_ONLY" = true ]; then
    echo "执行稠密重建..."
    python "$PROJECT_ROOT/scripts/reconstruction/dense_reconstruction.py" \
        --workspace_path "$OUTPUT_PATH" \
        --config "$CONFIG_PATH" \
        --dense-only
elif [ "$SPARSE_ONLY" = true ]; then
    echo "执行稀疏重建..."
    python "$PYTHON_SCRIPT" \
        --input_image_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --matcher "$MATCHER" \
        --pano_render_type "$RENDER_TYPE"
else
    echo "执行完整重建流程（稀疏+稠密）..."
    # 先执行稀疏重建
    python "$PYTHON_SCRIPT" \
        --input_image_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --matcher "$MATCHER" \
        --pano_render_type "$RENDER_TYPE"
    
    # 再执行稠密重建
    python "$PROJECT_ROOT/scripts/reconstruction/dense_reconstruction.py" \
        --workspace_path "$OUTPUT_PATH" \
        --config "$CONFIG_PATH"
fi

echo ""
echo "=========================================="
echo "重建完成！"
echo "=========================================="
echo "注意: 输出包含Rig配置，格式与普通图像重建不同"
