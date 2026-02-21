#!/bin/bash
# ============================================================
# Combine Point Clouds Script
# ============================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
SCENE="BridgeB"
COLMAP_DIR="/root/autodl-tmp/data/colmap_STAGE1_4x"
INPUT_DIR="/root/autodl-tmp/data/STAGE1_4x/BridgeB/elastic_refined"
OUTPUT="output/merged.ply"
CAMERA_NAME="pano_camera12"
NO_TRANSFORM=false
VOXEL_SIZE=""
GENERATE_COLMAP_POINTS3D=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --colmap_dir)
            COLMAP_DIR="$2"
            shift 2
            ;;
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --camera_name)
            CAMERA_NAME="$2"
            shift 2
            ;;
        --no_transform)
            NO_TRANSFORM=true
            shift
            ;;
        --voxel_size)
            VOXEL_SIZE="$2"
            shift 2
            ;;
        --generate_colmap_points3d)
            GENERATE_COLMAP_POINTS3D=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config PATH        配置文件路径（默认: config.yaml）"
            echo "  --scene NAME         场景名称（覆盖配置文件）"
            echo "  --colmap_dir PATH    COLMAP数据集根目录（覆盖配置文件）"
            echo "  --input_dir PATH     输入点云目录（覆盖配置文件）"
            echo "  --output PATH        输出PLY文件路径（覆盖配置文件）"
            echo "  --camera_name NAME   相机名称（覆盖配置文件）"
            echo "  --no_transform       不对点云应用坐标变换（覆盖配置文件）"
            echo "  --voxel_size SIZE    体素下采样大小（米），0表示不下采样（覆盖配置文件）"
            echo "  --generate_colmap_points3d  生成 COLMAP points3D 格式文件到 output 目录（覆盖配置文件）"
            echo "  -h, --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0"
            echo "  $0 --scene BridgeB --input_dir /path/to/plys --output merged.ply"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python3 combine_plys.py --config \"$CONFIG_FILE\""

if [ -n "$SCENE" ]; then
    CMD="$CMD --scene \"$SCENE\""
fi

if [ -n "$COLMAP_DIR" ]; then
    CMD="$CMD --colmap_dir \"$COLMAP_DIR\""
fi

if [ -n "$INPUT_DIR" ]; then
    CMD="$CMD --input_dir \"$INPUT_DIR\""
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output \"$OUTPUT\""
fi

if [ -n "$CAMERA_NAME" ]; then
    CMD="$CMD --camera_name \"$CAMERA_NAME\""
fi

if [ "$NO_TRANSFORM" = true ]; then
    CMD="$CMD --no_transform"
fi

if [ -n "$VOXEL_SIZE" ]; then
    CMD="$CMD --voxel_size \"$VOXEL_SIZE\""
fi

if [ "$GENERATE_COLMAP_POINTS3D" = true ]; then
    CMD="$CMD --generate_colmap_points3d"
fi

# 执行命令
echo "======================================="
echo "合并点云文件"
echo "======================================="
echo "执行命令: $CMD"
echo "======================================="
echo ""

eval $CMD

echo ""
echo "======================================="
echo "完成"
echo "======================================="
