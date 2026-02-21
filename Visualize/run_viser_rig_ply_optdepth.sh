#!/bin/bash

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Viser Rig PLY Optimized Depth Visualization"
echo "=========================================="

echo "[Setup] Activate environment (optional)"
# 按需启用你的环境，例如：
# source ~/miniconda3/bin/activate your_env
# conda activate your_env

echo "[Deps] Install Python packages if missing"
# 检查并安装必要的依赖
pip install -q numpy viser pycolmap pyyaml plyfile || {
    echo "⚠️  某些依赖安装失败，请检查网络连接或手动安装"
    echo "   需要的包: numpy, viser, pycolmap, pyyaml, plyfile"
}

# 检查配置文件是否存在
CONFIG_FILE="viser_rig_ply_optdepth_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  配置文件不存在: $CONFIG_FILE"
    echo "   将使用默认参数或命令行参数"
fi

echo "[Run] viser_rig_ply_optdepth.py"
echo "=========================================="
echo ""

# 传递所有命令行参数给 Python 脚本
python viser_rig_ply_optdepth.py "$@"
