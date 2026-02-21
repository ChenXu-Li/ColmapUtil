#!/bin/bash
#
# 可视化 COLMAP rig 相机组的位置和旋转
# 用法: ./visualize_rig.sh [选项]
# 示例: ./visualize_rig.sh --scene BridgeB --colmap_dir /root/autodl-tmp/data/colmap_STAGE1_4x

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
# Python 脚本已迁移到当前目录
PYTHON_SCRIPT="$PROJECT_ROOT/visualize_rig.py"
# 本脚本专用配置
CONFIG_FILE="$SCRIPT_DIR/rig_config.yaml"

# 显示使用说明
show_usage() {
    cat << EOF
COLMAP rig 相机组可视化脚本

用法: $0 [选项]

选项:
  --scene SCENE         场景名称（如 BridgeB, RoofTop, BridgeA 等，默认: BridgeB）
  --colmap_dir DIR      colmap_STAGE数据集根目录（默认: /root/autodl-tmp/data/colmap_STAGE1_4x）
  -p, --port PORT       Viser服务器端口（默认: 8080）
  --axis-length LENGTH  坐标轴长度（默认: 0.3米）
  --axis-width WIDTH    坐标轴线条宽度（默认: 3.0）
  --camera-scale SCALE  相机frustum的缩放比例（默认: 0.05）
  --dense-ply PATH      稠密点云PLY文件路径（可选，默认自动查找 fused.ply）
  --dense-point-size SIZE 稠密点云点的大小（默认: 0.005）
  --hide-points         隐藏COLMAP稀疏点云
  --hide-cameras        隐藏相机位置
  --hide-dense-points   隐藏稠密点云
  -h, --help            显示此帮助信息

示例:
  # 基本用法（无命令行参数时从配置文件 rig_config.yaml 加载）
  $0

  # 指定场景和端口（覆盖配置文件中的值）
  $0 --scene BridgeB --port 8081

  # 指定完整路径
  $0 --scene RoofTop --colmap_dir /root/autodl-tmp/data/colmap_STAGE1_4x

  # 调整坐标轴大小
  $0 --scene BridgeB --axis-length 0.5 --axis-width 5.0

  # 隐藏稀疏点云和相机
  $0 --scene BridgeB --hide-points --hide-cameras

EOF
}

# 如果没有传入命令行参数，尝试从配置文件加载默认参数
if [[ $# -eq 0 ]] && [[ -f "$CONFIG_FILE" ]]; then
    export VIZ_CONFIG_FILE="$CONFIG_FILE"
    DEFAULT_ARGS_STR=$(python - << 'PY'
import os, sys
from pathlib import Path

cfg_path = Path(os.environ.get("VIZ_CONFIG_FILE", ""))
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

# scene
scene = cfg.get("scene")
if scene:
    args.extend(["--scene", str(scene)])

# colmap_dir
colmap_dir = cfg.get("colmap_dir")
if colmap_dir:
    args.extend(["--colmap_dir", str(colmap_dir)])

# port
port = cfg.get("port", None)
if port is not None:
    args.extend(["-p", str(port)])

# axis_length
axis_length = cfg.get("axis_length", None)
if axis_length is not None:
    args.extend(["--axis-length", str(axis_length)])

# axis_width
axis_width = cfg.get("axis_width", None)
if axis_width is not None:
    args.extend(["--axis-width", str(axis_width)])

# camera_scale
camera_scale = cfg.get("camera_scale", None)
if camera_scale is not None:
    args.extend(["--camera-scale", str(camera_scale)])

# dense_ply
dense_ply = cfg.get("dense_ply")
if dense_ply:
    args.extend(["--dense-ply", str(dense_ply)])

# dense_point_size
dense_point_size = cfg.get("dense_point_size", None)
if dense_point_size is not None:
    args.extend(["--dense-point-size", str(dense_point_size)])

# 布尔开关映射为 --hide-xxx
if bool(cfg.get("hide_points", False)):
    args.append("--hide-points")
if bool(cfg.get("hide_cameras", False)):
    args.append("--hide-cameras")
if bool(cfg.get("hide_dense_points", False)):
    args.append("--hide-dense-points")

def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

print(" ".join(shell_quote(a) for a in args))
PY
)

    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    if [[ -n "$DEFAULT_ARGS_STR" ]]; then
        eval "set -- $DEFAULT_ARGS_STR"
    fi
fi

# 解析参数
SCENE=""
COLMAP_DIR=""
PORT=""
AXIS_LENGTH=""
AXIS_WIDTH=""
CAMERA_SCALE=""
DENSE_PLY=""
DENSE_POINT_SIZE=""
HIDE_POINTS=""
HIDE_CAMERAS=""
HIDE_DENSE_POINTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --colmap_dir)
            COLMAP_DIR="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --axis-length)
            AXIS_LENGTH="$2"
            shift 2
            ;;
        --axis-width)
            AXIS_WIDTH="$2"
            shift 2
            ;;
        --camera-scale)
            CAMERA_SCALE="$2"
            shift 2
            ;;
        --dense-ply)
            DENSE_PLY="$2"
            shift 2
            ;;
        --dense-point-size)
            DENSE_POINT_SIZE="$2"
            shift 2
            ;;
        --hide-points)
            HIDE_POINTS="--hide_points"
            shift
            ;;
        --hide-cameras)
            HIDE_CAMERAS="--hide_cameras"
            shift
            ;;
        --hide-dense-points)
            HIDE_DENSE_POINTS="--hide_dense_points"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "错误: 未知选项: $1"
            show_usage
            exit 1
            ;;
        *)
            echo "错误: 未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 构建Python命令参数
PYTHON_ARGS=()

# 添加可选参数
if [ -n "$SCENE" ]; then
    PYTHON_ARGS+=("--scene" "$SCENE")
fi

if [ -n "$COLMAP_DIR" ]; then
    PYTHON_ARGS+=("--colmap_dir" "$COLMAP_DIR")
fi

if [ -n "$PORT" ]; then
    PYTHON_ARGS+=("--port" "$PORT")
fi

if [ -n "$AXIS_LENGTH" ]; then
    PYTHON_ARGS+=("--axis_length" "$AXIS_LENGTH")
fi

if [ -n "$AXIS_WIDTH" ]; then
    PYTHON_ARGS+=("--axis_width" "$AXIS_WIDTH")
fi

if [ -n "$CAMERA_SCALE" ]; then
    PYTHON_ARGS+=("--camera_scale" "$CAMERA_SCALE")
fi

if [ -n "$DENSE_PLY" ]; then
    PYTHON_ARGS+=("--dense_ply" "$DENSE_PLY")
fi

if [ -n "$DENSE_POINT_SIZE" ]; then
    PYTHON_ARGS+=("--dense_point_size" "$DENSE_POINT_SIZE")
fi

if [ -n "$HIDE_POINTS" ]; then
    PYTHON_ARGS+=("$HIDE_POINTS")
fi

if [ -n "$HIDE_CAMERAS" ]; then
    PYTHON_ARGS+=("$HIDE_CAMERAS")
fi

if [ -n "$HIDE_DENSE_POINTS" ]; then
    PYTHON_ARGS+=("$HIDE_DENSE_POINTS")
fi

echo "=========================================="
echo "COLMAP Rig 相机组可视化"
echo "=========================================="
if [ -n "$SCENE" ]; then
    echo "场景: $SCENE"
fi
if [ -n "$COLMAP_DIR" ]; then
    echo "COLMAP目录: $COLMAP_DIR"
fi
if [ -n "$PORT" ]; then
    echo "端口: $PORT"
fi
echo "=========================================="
echo ""

# 执行可视化
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
