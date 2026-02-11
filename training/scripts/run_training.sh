#!/bin/bash
#
# 训练工具启动脚本
# 用于准备训练数据和启动训练流程
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../configs/config_training.yaml"

# 显示使用说明
show_usage() {
    cat << EOF
COLMAP训练工具启动脚本

用法: $0 <command> [选项]

命令:
  convert <colmap_dir> <output_dir>  转换COLMAP数据为训练格式
  prepare <colmap_dir> <output_dir>  准备训练数据（包含转换）
  train <config>                     启动训练（需要具体训练框架）

选项:
  -c, --config PATH                  配置文件路径（默认: configs/config_training.yaml）
  -f, --format FORMAT              输出格式 (gaussian_splatting|nerfstudio|instant_ngp)
  --dense-ply PATH                   稠密点云PLY文件路径
  --convert-dense                    将稠密点云转换为COLMAP格式
  -h, --help                         显示此帮助信息

示例:
  # 转换COLMAP数据为Gaussian Splatting格式
  $0 convert /path/to/colmap/output /path/to/training/data -f gaussian_splatting

  # 准备训练数据（包含稠密点云转换）
  $0 prepare /path/to/colmap/output /path/to/training/data \\
      --dense-ply /path/to/fused.ply --convert-dense

注意:
  - 训练命令需要根据具体训练框架实现
  - 当前脚本主要提供数据准备功能

EOF
}

# 解析参数
COMMAND=""
CONFIG_PATH="$CONFIG_FILE"
OUTPUT_FORMAT="gaussian_splatting"
DENSE_PLY=""
CONVERT_DENSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --dense-ply)
            DENSE_PLY="$2"
            shift 2
            ;;
        --convert-dense)
            CONVERT_DENSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        convert|prepare|train)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            echo "错误: 未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    echo "错误: 必须指定命令"
    show_usage
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "警告: 配置文件不存在: $CONFIG_PATH，使用默认配置"
fi

echo "=========================================="
echo "训练工具: $COMMAND"
echo "=========================================="

case "$COMMAND" in
    convert)
        if [ $# -lt 2 ]; then
            echo "错误: convert命令需要2个参数: <colmap_dir> <output_dir>"
            exit 1
        fi
        COLMAP_DIR="$1"
        OUTPUT_DIR="$2"
        
        echo "转换COLMAP数据为训练格式..."
        echo "COLMAP目录: $COLMAP_DIR"
        echo "输出目录: $OUTPUT_DIR"
        echo "格式: $OUTPUT_FORMAT"
        
        # 这里应该调用实际的转换脚本
        echo "注意: 转换脚本需要根据具体格式实现"
        ;;
    
    prepare)
        if [ $# -lt 2 ]; then
            echo "错误: prepare命令需要2个参数: <colmap_dir> <output_dir>"
            exit 1
        fi
        COLMAP_DIR="$1"
        OUTPUT_DIR="$2"
        
        echo "准备训练数据..."
        echo "COLMAP目录: $COLMAP_DIR"
        echo "输出目录: $OUTPUT_DIR"
        
        # 如果需要转换稠密点云
        if [ "$CONVERT_DENSE" = true ] && [ -n "$DENSE_PLY" ]; then
            echo "转换稠密点云为COLMAP格式..."
            SPARSE_DIR="$COLMAP_DIR/sparse/0"
            if [ -d "$SPARSE_DIR" ]; then
                python "$PROJECT_ROOT/scripts/conversion/dense_to_colmap.py" \
                    --dense_ply "$DENSE_PLY" \
                    --colmap_sparse_dir "$SPARSE_DIR"
            else
                echo "警告: 稀疏重建目录不存在: $SPARSE_DIR"
            fi
        fi
        
        # 转换数据格式
        echo "转换数据格式为: $OUTPUT_FORMAT"
        echo "注意: 转换脚本需要根据具体格式实现"
        ;;
    
    train)
        echo "启动训练..."
        echo "注意: 训练命令需要根据具体训练框架实现"
        echo "请参考相应训练框架的文档"
        ;;
    
    *)
        echo "错误: 未知命令: $COMMAND"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
