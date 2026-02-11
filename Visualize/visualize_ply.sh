#!/bin/bash
#
# 简单的PLY点云可视化脚本
# 用法: ./visualize_ply.sh <ply_file> [ply_file2] [ply_file3] ...
# 示例: ./visualize_ply.sh /root/autodl-tmp/data/STAGE1_4x/BridgeB/mutil_refined/raw.ply
# 示例命令（如不需要可删除或保留为注释）
# bash ./visualize_ply.sh /root/autodl-tmp/data/colmap_STAGE1_4x/BridgeB/fused.ply /root/autodl-tmp/data/MobilePhone/BridgeB/colmap/fused.ply
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
# Python 脚本已迁移到当前目录
PYTHON_SCRIPT="$PROJECT_ROOT/visualize_ply.py"
# 本脚本专用配置
CONFIG_FILE="$SCRIPT_DIR/ply_config.yaml"

# 显示使用说明
show_usage() {
    cat << EOF
PLY点云可视化脚本

用法: $0 <ply_file> [ply_file2] [ply_file3] ... [选项]

参数:
  ply_file               PLY点云文件路径（必需，可指定多个）

选项:
  -p, --port PORT       Viser服务器端口（默认: 8080）
  -s, --point-size SIZE 点云点的大小（默认: 0.005）
  -h, --help            显示此帮助信息

示例:
  # 显示单个点云
  $0 /path/to/pointcloud.ply

  # 显示多个点云
  $0 /path/to/pointcloud1.ply /path/to/pointcloud2.ply

  # 指定端口和点大小
  $0 /path/to/pointcloud.ply -p 8081 -s 0.01

EOF
}

# 如果没有传入命令行参数，尝试从配置文件加载默认参数
if [[ $# -eq 0 ]] && [[ -f "$CONFIG_FILE" ]]; then
    # 使用 Python + PyYAML 从 config.yaml 中读取 ply 配置并转成命令行参数
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
files = cfg.get("files") or []
if isinstance(files, str):
    files = [files]

args = []
for p in files:
    if not p:
        continue
    args.append(p)

port = cfg.get("port", None)
if port is not None:
    args.extend(["-p", str(port)])

point_size = cfg.get("point_size", None)
if point_size is not None:
    args.extend(["-s", str(point_size)])

if not args:
    # 没有任何可用配置，不输出
    sys.exit(0)

# 生成安全的 shell 参数字符串
def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

print(" ".join(shell_quote(a) for a in args))
PY
)

    # 如果 Python 解析失败（返回码非 0），则退出
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # 如果解析到了默认参数，则将其注入到 $@ 中
    if [[ -n "$DEFAULT_ARGS_STR" ]]; then
        eval "set -- $DEFAULT_ARGS_STR"
    fi
fi

# 解析参数
PLY_FILES=()
PORT=""
POINT_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -s|--point-size)
            POINT_SIZE="$2"
            shift 2
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
            # 检查文件是否存在
            if [ ! -f "$1" ]; then
                echo "警告: 文件不存在: $1"
            fi
            PLY_FILES+=("$1")
            shift
            ;;
    esac
done

# 检查是否提供了PLY文件
if [ ${#PLY_FILES[@]} -eq 0 ]; then
    echo "错误: 必须指定至少一个PLY文件"
    show_usage
    exit 1
fi

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 构建Python命令参数
PYTHON_ARGS=()

# 添加所有PLY文件
for ply_file in "${PLY_FILES[@]}"; do
    PYTHON_ARGS+=("--ply" "$ply_file")
done

# 添加端口（如果指定）
if [ -n "$PORT" ]; then
    PYTHON_ARGS+=("--port" "$PORT")
fi

# 添加点大小（如果指定）
if [ -n "$POINT_SIZE" ]; then
    PYTHON_ARGS+=("--point_size" "$POINT_SIZE")
fi

echo "=========================================="
echo "PLY点云可视化"
echo "=========================================="
echo "点云文件:"
for ply_file in "${PLY_FILES[@]}"; do
    echo "  - $ply_file"
done
if [ -n "$PORT" ]; then
    echo "端口: $PORT"
fi
if [ -n "$POINT_SIZE" ]; then
    echo "点大小: $POINT_SIZE"
fi
echo "=========================================="
echo ""

# 执行可视化
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
