#!/bin/bash
#
# 可视化 pinhole 模型重建结果
# 用法: ./visualize_pinhole.sh <workspace_path> [选项]
# 示例: ./visualize_pinhole.sh /root/autodl-tmp/data/MobilePhone/BridgeB/colmap

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
# Python 脚本已迁移到当前目录
PYTHON_SCRIPT="$PROJECT_ROOT/visualize_pinhole.py"
# 本脚本专用配置
CONFIG_FILE="$SCRIPT_DIR/pinhole_config.yaml"

# 显示使用说明
show_usage() {
    cat << EOF
Pinhole 模型重建结果可视化脚本

用法: $0 <workspace_path> [选项]

参数:
  workspace_path        COLMAP工作目录路径（包含sparse/0和可选的fused.ply）

选项:
  --sparse-path PATH   稀疏重建结果路径（默认: workspace_path/sparse/0）
  --dense-ply PATH     稠密点云PLY文件路径（默认: 自动查找 workspace_path/fused.ply）
  -p, --port PORT      Viser服务器端口（默认: 8080）
  --sparse-point-size SIZE  稀疏点云点的大小（默认: 0.01）
  --dense-point-size SIZE   稠密点云点的大小（默认: 0.005）
  --camera-scale SCALE 相机frustum的缩放比例（默认: 0.05）
  --hide-sparse-points 隐藏稀疏点云
  --hide-cameras       隐藏相机位置
  --hide-dense-points  隐藏稠密点云
  -h, --help           显示此帮助信息

示例:
  # 基本用法（自动查找稠密点云）
  $0 /root/autodl-tmp/data/MobilePhone/BridgeA/colmap

  # 指定端口和点大小
  $0 /root/autodl-tmp/data/MobilePhone/BridgeA/colmap -p 8081 --dense-point-size 0.01

  # 仅显示稀疏重建结果（隐藏稠密点云）
  $0 /root/autodl-tmp/data/MobilePhone/BridgeA/colmap --hide-dense-points

  # 指定稠密点云路径
  $0 /root/autodl-tmp/data/MobilePhone/BridgeA/colmap --dense-ply /path/to/fused.ply

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

# 独立配置文件，顶层就是本脚本的配置
workspace = cfg.get("workspace_path")
if not workspace:
    # 如果没有配置 workspace_path，则不输出任何默认参数
    sys.exit(0)

args = [str(workspace)]

sp = cfg.get("sparse_path")
if sp:
    args.extend(["--sparse-path", str(sp)])

dp = cfg.get("dense_ply")
if dp:
    args.extend(["--dense-ply", str(dp)])

port = cfg.get("port", None)
if port is not None:
    args.extend(["-p", str(port)])

sps = cfg.get("sparse_point_size", None)
if sps is not None:
    args.extend(["--sparse-point-size", str(sps)])

dps = cfg.get("dense_point_size", None)
if dps is not None:
    args.extend(["--dense-point-size", str(dps)])

cs = cfg.get("camera_scale", None)
if cs is not None:
    args.extend(["--camera-scale", str(cs)])

# 布尔开关映射为 --hide-xxx
if bool(cfg.get("hide_sparse_points", False)):
    args.append("--hide-sparse-points")
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
WORKSPACE_PATH=""
SPARSE_PATH=""
DENSE_PLY=""
PORT=""
SPARSE_POINT_SIZE=""
DENSE_POINT_SIZE=""
CAMERA_SCALE=""
HIDE_SPARSE_POINTS=""
HIDE_CAMERAS=""
HIDE_DENSE_POINTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sparse-path)
            SPARSE_PATH="$2"
            shift 2
            ;;
        --dense-ply)
            DENSE_PLY="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --sparse-point-size)
            SPARSE_POINT_SIZE="$2"
            shift 2
            ;;
        --dense-point-size)
            DENSE_POINT_SIZE="$2"
            shift 2
            ;;
        --camera-scale)
            CAMERA_SCALE="$2"
            shift 2
            ;;
        --hide-sparse-points)
            HIDE_SPARSE_POINTS="--hide-sparse-points"
            shift
            ;;
        --hide-cameras)
            HIDE_CAMERAS="--hide-cameras"
            shift
            ;;
        --hide-dense-points)
            HIDE_DENSE_POINTS="--hide-dense-points"
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
            if [ -z "$WORKSPACE_PATH" ]; then
                WORKSPACE_PATH="$1"
            else
                echo "错误: 只能指定一个工作目录路径"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查是否提供了工作目录路径
if [ -z "$WORKSPACE_PATH" ]; then
    echo "错误: 必须指定工作目录路径"
    show_usage
    exit 1
fi

# 检查工作目录是否存在
if [ ! -d "$WORKSPACE_PATH" ]; then
    echo "错误: 工作目录不存在: $WORKSPACE_PATH"
    exit 1
fi

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 构建Python命令参数
PYTHON_ARGS=(
    "--workspace_path" "$WORKSPACE_PATH"
)

# 添加可选参数
if [ -n "$SPARSE_PATH" ]; then
    PYTHON_ARGS+=("--sparse_path" "$SPARSE_PATH")
fi

if [ -n "$DENSE_PLY" ]; then
    PYTHON_ARGS+=("--dense_ply" "$DENSE_PLY")
fi

if [ -n "$PORT" ]; then
    PYTHON_ARGS+=("--port" "$PORT")
fi

if [ -n "$SPARSE_POINT_SIZE" ]; then
    PYTHON_ARGS+=("--sparse_point_size" "$SPARSE_POINT_SIZE")
fi

if [ -n "$DENSE_POINT_SIZE" ]; then
    PYTHON_ARGS+=("--dense_point_size" "$DENSE_POINT_SIZE")
fi

if [ -n "$CAMERA_SCALE" ]; then
    PYTHON_ARGS+=("--camera_scale" "$CAMERA_SCALE")
fi

if [ -n "$HIDE_SPARSE_POINTS" ]; then
    PYTHON_ARGS+=("$HIDE_SPARSE_POINTS")
fi

if [ -n "$HIDE_CAMERAS" ]; then
    PYTHON_ARGS+=("$HIDE_CAMERAS")
fi

if [ -n "$HIDE_DENSE_POINTS" ]; then
    PYTHON_ARGS+=("$HIDE_DENSE_POINTS")
fi

echo "=========================================="
echo "Pinhole 模型重建结果可视化"
echo "=========================================="
echo "工作目录: $WORKSPACE_PATH"
if [ -n "$SPARSE_PATH" ]; then
    echo "稀疏重建: $SPARSE_PATH"
fi
if [ -n "$DENSE_PLY" ]; then
    echo "稠密点云: $DENSE_PLY"
fi
if [ -n "$PORT" ]; then
    echo "端口: $PORT"
fi
echo "=========================================="
echo ""

# 执行可视化
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
