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
  -i, --input PATH        输入全景图目录（必需，或从配置文件读取）
  -o, --output PATH       输出路径（必需，或从配置文件读取）
  -c, --config PATH       配置文件路径（默认: configs/config_panorama.yaml）
  -m, --matcher TYPE      匹配器类型 (sequential|exhaustive|vocabtree|spatial, 默认: sequential)
  -t, --render-type TYPE  渲染类型 (overlapping|non-overlapping, 默认: overlapping)
  --sparse-only           仅执行稀疏重建
  --dense-only            仅执行稠密重建（需要已有稀疏重建结果）
  -h, --help              显示此帮助信息

示例:
  # 完整重建流程（使用配置文件中的默认值）
  $0

  # 完整重建流程（指定路径）
  $0 -i /path/to/panoramas -o /path/to/output

  # 仅稀疏重建
  $0 -i /path/to/panoramas -o /path/to/output --sparse-only

  # 仅稠密重建
  $0 -o /path/to/output --dense-only

注意:
  - 输入图像必须是360°全景图（宽高比2:1）
  - 输出格式包含Rig配置，与普通图像重建不同
  - 如果不提供 -i 和 -o 参数，将从配置文件读取默认值

EOF
}

# 保存原始命令行参数
ORIGINAL_ARGS=("$@")

# 从配置文件读取默认参数（如果配置文件存在）
# 配置文件的值会被放在命令行参数前面，这样命令行参数可以覆盖配置值
if [[ -f "$CONFIG_FILE" ]]; then
    export PANO_CONFIG_FILE="$CONFIG_FILE"
    DEFAULT_ARGS_STR=$(python - << 'PY'
import os, sys
from pathlib import Path

cfg_path = Path(os.environ.get("PANO_CONFIG_FILE", ""))
if not cfg_path.is_file():
    sys.exit(0)

try:
    import yaml
except ImportError:
    print("错误: 需要 PyYAML 支持以从配置文件读取参数，请先执行 `pip install pyyaml`。", file=sys.stderr)
    sys.exit(1)

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

args = []

# input_path (从 paths.input_panoramas)
paths = cfg.get("paths", {}) or {}
input_panoramas = paths.get("input_panoramas")
if input_panoramas:
    args.extend(["-i", str(input_panoramas)])

# output_path (从 paths.output_path)
output_path = paths.get("output_path")
if output_path:
    args.extend(["-o", str(output_path)])

# matcher (从 feature_matching.matcher_type)
feature_matching = cfg.get("feature_matching", {}) or {}
matcher_type = feature_matching.get("matcher_type")
if matcher_type:
    args.extend(["-m", str(matcher_type)])

# render_type (从 panorama_rendering.render_type)
panorama_rendering = cfg.get("panorama_rendering", {}) or {}
render_type = panorama_rendering.get("render_type")
if render_type:
    args.extend(["-t", str(render_type)])

def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

print(" ".join(shell_quote(a) for a in args))
PY
)

    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # 将配置文件的默认值和原始命令行参数合并
    # 配置值在前，命令行参数在后（后面的参数会覆盖前面的）
    if [[ -n "$DEFAULT_ARGS_STR" ]]; then
        eval "set -- $DEFAULT_ARGS_STR"
        # 追加原始命令行参数
        set -- "$@" "${ORIGINAL_ARGS[@]}"
    fi
fi

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
    echo "提示: 可以在配置文件中设置 paths.output_path，然后不带参数运行脚本"
    show_usage
    exit 1
fi

if [ "$DENSE_ONLY" = false ] && [ -z "$INPUT_PATH" ]; then
    echo "错误: 稀疏重建需要指定输入全景图目录 (-i)"
    echo "提示: 可以在配置文件中设置 paths.input_panoramas，然后不带参数运行脚本"
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
DENSE_SCRIPT="$SCRIPT_DIR/dense_reconstruction_cli.py"

if [ "$DENSE_ONLY" = true ]; then
    echo "执行稠密重建..."
    python "$DENSE_SCRIPT" \
        --workspace_path "$OUTPUT_PATH"
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
    python "$DENSE_SCRIPT" \
        --workspace_path "$OUTPUT_PATH"
fi

echo ""
echo "=========================================="
echo "重建完成！"
echo "=========================================="
echo "注意: 输出包含Rig配置，格式与普通图像重建不同"
