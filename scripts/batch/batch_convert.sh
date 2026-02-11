#!/bin/bash

################################################################################
# 脚本名称: dense2colmap_points.sh
# 功能: 将 COLMAP 稠密重建的点云转换为 COLMAP points3D 格式
#       用于替换稀疏重建的 points3D，以便在 Gaussian Splatting 训练中使用稠密点云
#
# 使用方法:
#   1. 单个场景转换:
#      bash dense2colmap_points.sh \
#          --dense_ply /path/to/dense/fused.ply \
#          --colmap_sparse_dir /path/to/sparse/0
#
#   2. 批量转换多个场景:
#      bash dense2colmap_points.sh --batch \
#          --base_dir /root/autodl-tmp/dataset/ColmapDataset/colmap_STAGE1_4x
#
#   3. 不备份原始文件:
#      bash dense2colmap_points.sh \
#          --dense_ply /path/to/dense/fused.ply \
#          --colmap_sparse_dir /path/to/sparse/0 \
#          --no_backup
#
# 参数说明:
#   --dense_ply: 稠密重建的 PLY 文件路径（如 dense/fused.ply）
#   --colmap_sparse_dir: COLMAP 稀疏重建目录（如 sparse/0 或 sparse）
#   --no_backup: 不备份原始的 points3D 文件（可选）
#   --batch: 批量处理模式，处理 base_dir 下所有场景
#   --base_dir: 批量处理时的基础目录，包含多个场景目录
#
# 示例:
#   # 转换单个场景
#   bash dense2colmap_points.sh \
#       --dense_ply /root/autodl-tmp/data/STAGE1_4x/BridgeB/cut_dense_merge.ply \
#       --/root/autodl-tmp/data/colmap_STAGE1_4x/BridgeB/sparse/0
#
#   # 批量转换所有场景
#   bash dense2colmap_points.sh --batch --base_dir /root/autodl-tmp/data/colmap_STAGE1_4x
#
################################################################################

# set -e  # 注释掉，允许批量处理时继续处理下一个场景

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/dense2colmap_points.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python 脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 显示使用说明
show_usage() {
    cat << EOF
用法: $0 [选项]

选项:
  --dense_ply PATH           稠密重建的 PLY 文件路径
  --colmap_sparse_dir PATH   COLMAP 稀疏重建目录
  --no_backup                不备份原始的 points3D 文件
  --batch                    批量处理模式
  --base_dir PATH            批量处理时的基础目录
  -h, --help                 显示此帮助信息

示例:
  # 单个场景转换
  $0 --dense_ply /path/to/fused.ply --colmap_sparse_dir /path/to/sparse/0

  # 批量转换
  $0 --batch --base_dir /path/to/colmap_STAGE1_4x

EOF
}

# 处理单个场景
process_single() {
    local dense_ply="$1"
    local sparse_dir="$2"
    local no_backup="$3"
    
    echo "=========================================="
    echo "处理场景:"
    echo "  稠密点云: $dense_ply"
    echo "  稀疏目录: $sparse_dir"
    echo "=========================================="
    
    # 构建命令参数数组（更安全的方式，避免 eval）
    local cmd_args=("python" "$PYTHON_SCRIPT" "--dense_ply" "$dense_ply" "--colmap_sparse_dir" "$sparse_dir")
    if [ "$no_backup" = "true" ]; then
        cmd_args+=("--no_backup")
    fi
    
    # 执行命令（不因错误退出，允许批量处理继续）
    if "${cmd_args[@]}"; then
        echo "✓ 成功处理: $sparse_dir"
        echo ""
        return 0
    else
        echo "✗ 处理失败: $sparse_dir"
        echo ""
        return 1
    fi
}

# 批量处理
process_batch() {
    local base_dir="$1"
    local no_backup="$2"
    
    if [ ! -d "$base_dir" ]; then
        echo "Error: 基础目录不存在: $base_dir"
        exit 1
    fi
    
    echo "=========================================="
    echo "批量处理模式"
    echo "基础目录: $base_dir"
    echo "=========================================="
    echo ""
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    
    # 遍历所有场景目录
    for scene_dir in "$base_dir"/*; do
        if [ ! -d "$scene_dir" ]; then
            continue
        fi
        
        scene_name=$(basename "$scene_dir")
        dense_ply="$scene_dir/fused.ply"
        sparse_dir="$scene_dir/sparse/0"
        
        # 检查必要文件是否存在
        if [ ! -f "$dense_ply" ]; then
            echo "⚠ 跳过 $scene_name: 未找到 fused.ply (路径: $dense_ply)"
            ((skip_count++))
            continue
        fi
        
        if [ ! -d "$sparse_dir" ]; then
            echo "⚠ 跳过 $scene_name: 未找到 sparse/0 目录 (路径: $sparse_dir)"
            ((skip_count++))
            continue
        fi
        
        # 处理场景
        if process_single "$dense_ply" "$sparse_dir" "$no_backup"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "批量处理完成"
    echo "成功: $success_count"
    echo "失败: $fail_count"
    echo "跳过: $skip_count"
    echo "=========================================="
}

# 解析参数
DENSE_PLY=""
SPARSE_DIR=""
NO_BACKUP="false"
BATCH_MODE="false"
BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dense_ply)
            DENSE_PLY="$2"
            shift 2
            ;;
        --colmap_sparse_dir)
            SPARSE_DIR="$2"
            shift 2
            ;;
        --no_backup)
            NO_BACKUP="true"
            shift
            ;;
        --batch)
            BATCH_MODE="true"
            shift
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: 未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查 Python 是否可用
if ! command -v python &> /dev/null; then
    echo "Error: 未找到 python 命令"
    exit 1
fi

# 检查 plyfile 是否安装
if ! python -c "import plyfile" 2>/dev/null; then
    echo "Error: 未安装 plyfile 模块，请运行: pip install plyfile"
    exit 1
fi

# 执行相应模式
if [ "$BATCH_MODE" = "true" ]; then
    if [ -z "$BASE_DIR" ]; then
        echo "Error: 批量模式需要指定 --base_dir"
        show_usage
        exit 1
    fi
    process_batch "$BASE_DIR" "$NO_BACKUP"
else
    if [ -z "$DENSE_PLY" ] || [ -z "$SPARSE_DIR" ]; then
        echo "Error: 单个模式需要指定 --dense_ply 和 --colmap_sparse_dir"
        show_usage
        exit 1
    fi
    process_single "$DENSE_PLY" "$SPARSE_DIR" "$NO_BACKUP"
fi

