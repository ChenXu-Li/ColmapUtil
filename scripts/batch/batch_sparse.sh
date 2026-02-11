#!/bin/bash
#
# 批量稀疏重建脚本
# 用法: ./batch_sparse.sh [input_base] [output_base]
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/reconstruction/sparse_reconstruction.py"

# 加载配置
if [ -f "$PROJECT_ROOT/config.yaml" ]; then
    INPUT_BASE=$(python3 -c "import yaml; f=open('$PROJECT_ROOT/config.yaml'); d=yaml.safe_load(f); print(d['paths']['input_base']); f.close()" 2>/dev/null || echo "/root/autodl-tmp/data/STAGE1_4x")
    OUTPUT_BASE=$(python3 -c "import yaml; f=open('$PROJECT_ROOT/config.yaml'); d=yaml.safe_load(f); print(d['paths']['colmap_base']); f.close()" 2>/dev/null || echo "/root/autodl-tmp/data/colmap_STAGE1_4x")
    MATCHER=$(python3 -c "import yaml; f=open('$PROJECT_ROOT/config.yaml'); d=yaml.safe_load(f); print(d['sparse_reconstruction']['matcher']); f.close()" 2>/dev/null || echo "exhaustive")
else
    INPUT_BASE="${1:-/root/autodl-tmp/data/STAGE1_4x}"
    OUTPUT_BASE="${2:-/root/autodl-tmp/data/colmap_STAGE1_4x}"
    MATCHER="exhaustive"
fi

# 如果命令行提供了参数，则覆盖配置
if [ $# -ge 1 ]; then
    INPUT_BASE="$1"
fi
if [ $# -ge 2 ]; then
    OUTPUT_BASE="$2"
fi

# 设置环境变量（避免OpenBLAS段错误）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 记录脚本开始时间
script_start_time=$(date +%s)

# 用于存储所有场景的处理时间
declare -a scene_times
declare -a scene_names

echo "=========================================="
echo "批量稀疏重建"
echo "=========================================="
echo "输入目录: $INPUT_BASE"
echo "输出目录: $OUTPUT_BASE"
echo "匹配器: $MATCHER"
echo "=========================================="
echo ""

# 遍历输入目录下的每个场景
for scene_dir in "$INPUT_BASE"/*; do
    # 检查是否为目录
    if [ ! -d "$scene_dir" ]; then
        continue
    fi
    
    # 获取场景名称（目录名）
    scene_name=$(basename "$scene_dir")
    
    # 检查是否存在backgrounds文件夹
    backgrounds_path="$scene_dir/backgrounds"
    if [ ! -d "$backgrounds_path" ]; then
        echo "⚠️  警告: $scene_name 场景下没有找到 backgrounds 文件夹，跳过..."
        continue
    fi
    
    # 设置输出路径
    output_path="$OUTPUT_BASE/${scene_name}"
    
    # 记录场景开始时间
    scene_start_time=$(date +%s)
    scene_start_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "========================================="
    echo "开始处理场景: $scene_name"
    echo "开始时间: $scene_start_time_readable"
    echo "输入路径: $backgrounds_path"
    echo "输出路径: $output_path"
    echo "========================================="
    
    # 运行COLMAP稀疏重建
    python "$PYTHON_SCRIPT" \
        --input_image_path "$backgrounds_path" \
        --output_path "$output_path" \
        --matcher "$MATCHER"
    
    # 记录场景结束时间
    scene_end_time=$(date +%s)
    scene_end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
    scene_duration=$((scene_end_time - scene_start_time))
    
    # 格式化时间显示
    hours=$((scene_duration / 3600))
    minutes=$(((scene_duration % 3600) / 60))
    seconds=$((scene_duration % 60))
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 场景 $scene_name 处理完成"
        echo "结束时间: $scene_end_time_readable"
        if [ $hours -gt 0 ]; then
            printf "耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "耗时: %d分钟 %d秒 (%d秒)\n" $minutes $seconds $scene_duration
        else
            printf "耗时: %d秒\n" $scene_duration
        fi
        
        # 保存场景处理时间和名称
        scene_times+=($scene_duration)
        scene_names+=("$scene_name")
    else
        echo "✗ 场景 $scene_name 处理失败"
        echo "结束时间: $scene_end_time_readable"
        if [ $hours -gt 0 ]; then
            printf "耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "耗时: %d分钟 %d秒 (%d秒)\n" $minutes $seconds $scene_duration
        else
            printf "耗时: %d秒\n" $scene_duration
        fi
    fi
    
    echo ""
done

# 计算总时间
script_end_time=$(date +%s)
total_duration=$((script_end_time - script_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

echo "========================================="
echo "所有场景处理完成！"
echo "========================================="
echo "总统计信息:"
echo "  处理场景数: ${#scene_names[@]}"
if [ $total_hours -gt 0 ]; then
    printf "  总耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $total_hours $total_minutes $total_seconds $total_duration
elif [ $total_minutes -gt 0 ]; then
    printf "  总耗时: %d分钟 %d秒 (%d秒)\n" $total_minutes $total_seconds $total_duration
else
    printf "  总耗时: %d秒\n" $total_duration
fi

# 显示每个场景的详细时间统计
if [ ${#scene_names[@]} -gt 0 ]; then
    echo ""
    echo "各场景耗时统计:"
    for i in "${!scene_names[@]}"; do
        scene_name="${scene_names[$i]}"
        scene_duration="${scene_times[$i]}"
        hours=$((scene_duration / 3600))
        minutes=$(((scene_duration % 3600) / 60))
        seconds=$((scene_duration % 60))
        
        if [ $hours -gt 0 ]; then
            printf "  %-20s: %d小时 %d分钟 %d秒 (%d秒)\n" "$scene_name" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "  %-20s: %d分钟 %d秒 (%d秒)\n" "$scene_name" $minutes $seconds $scene_duration
        else
            printf "  %-20s: %d秒\n" "$scene_name" $scene_duration
        fi
    done
    
    # 计算平均时间
    total_scene_time=0
    for duration in "${scene_times[@]}"; do
        total_scene_time=$((total_scene_time + duration))
    done
    if [ ${#scene_times[@]} -gt 0 ]; then
        avg_time=$((total_scene_time / ${#scene_times[@]}))
        avg_hours=$((avg_time / 3600))
        avg_minutes=$(((avg_time % 3600) / 60))
        avg_seconds=$((avg_time % 60))
        
        echo ""
        echo "平均耗时:"
        if [ $avg_hours -gt 0 ]; then
            printf "  %d小时 %d分钟 %d秒 (%d秒)\n" $avg_hours $avg_minutes $avg_seconds $avg_time
        elif [ $avg_minutes -gt 0 ]; then
            printf "  %d分钟 %d秒 (%d秒)\n" $avg_minutes $avg_seconds $avg_time
        else
            printf "  %d秒\n" $avg_time
        fi
    fi
fi

echo "========================================="
