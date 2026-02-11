#!/bin/bash

# =============================================
# COLMAP稠密重建批量处理脚本
# 
# 用法：./batch_denseline.sh [GPU_ID]
# 示例：./batch_denseline.sh                    # 使用默认GPU (0)
#       ./batch_denseline.sh 0                  # 指定GPU ID
# =============================================

# 设置线程环境变量以避免OpenBLAS段错误
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 颜色定义（用于终端输出）
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印带颜色的消息函数
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_time() { echo -e "${CYAN}[TIME]${NC} $1"; }
print_step() { echo -e "\n${BOLD}${BLUE}════════════════════════════════════════${NC}"; echo -e "${BOLD}${BLUE}$1${NC}"; echo -e "${BOLD}${BLUE}════════════════════════════════════════${NC}\n"; }

# COLMAP数据集根目录
COLMAP_BASE_DIR="/root/autodl-tmp/data/colmap_STAGE2_4x"
DENSELINE_SCRIPT="/root/autodl-tmp/code/colmap_util/denseline.sh"

# GPU ID（默认0）
GPU_INDEX=0
if [ $# -ge 1 ]; then
    GPU_INDEX="$1"
fi

# 检查COLMAP根目录是否存在
if [ ! -d "$COLMAP_BASE_DIR" ]; then
    print_error "COLMAP数据集根目录不存在: $COLMAP_BASE_DIR"
    exit 1
fi

# 检查denseline.sh脚本是否存在
if [ ! -f "$DENSELINE_SCRIPT" ]; then
    print_error "denseline.sh脚本不存在: $DENSELINE_SCRIPT"
    exit 1
fi

# 确保denseline.sh可执行
chmod +x "$DENSELINE_SCRIPT"

# 记录脚本开始时间
script_start_time=$(date +%s)
script_start_time_readable=$(date '+%Y-%m-%d %H:%M:%S')

# 用于存储所有场景的处理时间
declare -a scene_times
declare -a scene_names
declare -a scene_status

print_step "COLMAP稠密重建批量处理"
print_info "COLMAP数据集根目录: $COLMAP_BASE_DIR"
print_info "GPU索引: $GPU_INDEX"
print_time "开始时间: $script_start_time_readable"
print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 遍历COLMAP根目录下的每个场景
scene_count=0
for scene_dir in "$COLMAP_BASE_DIR"/*; do
    # 检查是否为目录
    if [ ! -d "$scene_dir" ]; then
        continue
    fi
    
    # 获取场景名称（目录名）
    scene_name=$(basename "$scene_dir")
    
    # 检查是否存在sparse/0目录（稀疏重建结果）
    sparse_path="$scene_dir/sparse/0"
    if [ ! -d "$sparse_path" ]; then
        print_warning "场景 $scene_name 没有稀疏重建结果（sparse/0 不存在），跳过..."
        continue
    fi
    
    # 检查必要的稀疏文件
    missing_files=0
    for file in cameras.bin images.bin points3D.bin; do
        if [ ! -f "$sparse_path/$file" ]; then
            print_warning "场景 $scene_name 缺少稀疏重建文件: $file，跳过..."
            missing_files=1
            break
        fi
    done
    
    if [ $missing_files -eq 1 ]; then
        continue
    fi
    
    # 检查是否存在images目录
    images_path="$scene_dir/images"
    if [ ! -d "$images_path" ]; then
        print_warning "场景 $scene_name 没有images目录，跳过..."
        continue
    fi
    
    # 记录场景开始时间
    scene_start_time=$(date +%s)
    scene_start_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
    
    scene_count=$((scene_count + 1))
    
    print_step "处理场景 $scene_count: $scene_name"
    print_info "场景路径: $scene_dir"
    print_time "开始时间: $scene_start_time_readable"
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 运行denseline.sh脚本
    "$DENSELINE_SCRIPT" "$scene_dir" "$GPU_INDEX"
    
    # 记录场景结束时间
    scene_end_time=$(date +%s)
    scene_end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
    scene_duration=$((scene_end_time - scene_start_time))
    
    # 格式化时间显示（小时:分钟:秒）
    hours=$((scene_duration / 3600))
    minutes=$(((scene_duration % 3600) / 60))
    seconds=$((scene_duration % 60))
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        print_success "场景 $scene_name 处理完成"
        print_time "结束时间: $scene_end_time_readable"
        if [ $hours -gt 0 ]; then
            printf "${CYAN}[TIME]${NC} 耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "${CYAN}[TIME]${NC} 耗时: %d分钟 %d秒 (%d秒)\n" $minutes $seconds $scene_duration
        else
            printf "${CYAN}[TIME]${NC} 耗时: %d秒\n" $scene_duration
        fi
        
        # 保存场景处理时间和名称
        scene_times+=($scene_duration)
        scene_names+=("$scene_name")
        scene_status+=("成功")
    else
        print_error "场景 $scene_name 处理失败"
        print_time "结束时间: $scene_end_time_readable"
        if [ $hours -gt 0 ]; then
            printf "${CYAN}[TIME]${NC} 耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "${CYAN}[TIME]${NC} 耗时: %d分钟 %d秒 (%d秒)\n" $minutes $seconds $scene_duration
        else
            printf "${CYAN}[TIME]${NC} 耗时: %d秒\n" $scene_duration
        fi
        
        # 保存场景处理时间和名称（即使失败也记录）
        scene_times+=($scene_duration)
        scene_names+=("$scene_name")
        scene_status+=("失败")
    fi
    
    print_info ""
done

# 计算总时间
script_end_time=$(date +%s)
script_end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')
total_duration=$((script_end_time - script_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

print_step "批量处理完成"
print_success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_info "总统计信息:"
print_info "  处理场景数: ${#scene_names[@]}"
print_time "结束时间: $script_end_time_readable"

if [ $total_hours -gt 0 ]; then
    printf "${CYAN}[TIME]${NC} 总耗时: %d小时 %d分钟 %d秒 (%d秒)\n" $total_hours $total_minutes $total_seconds $total_duration
elif [ $total_minutes -gt 0 ]; then
    printf "${CYAN}[TIME]${NC} 总耗时: %d分钟 %d秒 (%d秒)\n" $total_minutes $total_seconds $total_duration
else
    printf "${CYAN}[TIME]${NC} 总耗时: %d秒\n" $total_duration
fi

# 显示每个场景的详细时间统计
if [ ${#scene_names[@]} -gt 0 ]; then
    print_info ""
    print_info "各场景耗时统计:"
    success_count=0
    fail_count=0
    
    for i in "${!scene_names[@]}"; do
        scene_name="${scene_names[$i]}"
        scene_duration="${scene_times[$i]}"
        status="${scene_status[$i]}"
        hours=$((scene_duration / 3600))
        minutes=$(((scene_duration % 3600) / 60))
        seconds=$((scene_duration % 60))
        
        if [ "$status" = "成功" ]; then
            success_count=$((success_count + 1))
            status_color="${GREEN}✓${NC}"
        else
            fail_count=$((fail_count + 1))
            status_color="${RED}✗${NC}"
        fi
        
        if [ $hours -gt 0 ]; then
            printf "  ${status_color} %-30s: %d小时 %d分钟 %d秒 (%d秒)\n" "$scene_name" $hours $minutes $seconds $scene_duration
        elif [ $minutes -gt 0 ]; then
            printf "  ${status_color} %-30s: %d分钟 %d秒 (%d秒)\n" "$scene_name" $minutes $seconds $scene_duration
        else
            printf "  ${status_color} %-30s: %d秒\n" "$scene_name" $scene_duration
        fi
    done
    
    print_info ""
    print_info "处理结果:"
    printf "  ${GREEN}成功: %d${NC} 个场景\n" $success_count
    if [ $fail_count -gt 0 ]; then
        printf "  ${RED}失败: %d${NC} 个场景\n" $fail_count
    fi
    
    # 计算平均时间（只计算成功的场景）
    if [ $success_count -gt 0 ]; then
        total_scene_time=0
        success_scene_count=0
        for i in "${!scene_names[@]}"; do
            if [ "${scene_status[$i]}" = "成功" ]; then
                total_scene_time=$((total_scene_time + ${scene_times[$i]}))
                success_scene_count=$((success_scene_count + 1))
            fi
        done
        
        if [ $success_scene_count -gt 0 ]; then
            avg_time=$((total_scene_time / success_scene_count))
            avg_hours=$((avg_time / 3600))
            avg_minutes=$(((avg_time % 3600) / 60))
            avg_seconds=$((avg_time % 60))
            
            print_info ""
            print_info "平均耗时（仅成功场景）:"
            if [ $avg_hours -gt 0 ]; then
                printf "  %d小时 %d分钟 %d秒 (%d秒)\n" $avg_hours $avg_minutes $avg_seconds $avg_time
            elif [ $avg_minutes -gt 0 ]; then
                printf "  %d分钟 %d秒 (%d秒)\n" $avg_minutes $avg_seconds $avg_time
            else
                printf "  %d秒\n" $avg_time
            fi
        fi
    fi
else
    print_warning "没有成功处理任何场景"
fi

print_success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_info ""
print_info "稠密重建结果位置:"
print_info "  每个场景的 fused.ply 文件位于:"
print_info "    $COLMAP_BASE_DIR/<场景名>/fused.ply"
print_info ""
print_info "可以使用以下工具查看点云:"
print_info "  - Meshlab: meshlab <fused.ply>"
print_info "  - CloudCompare: cloudcompare <fused.ply>"
print_info "  - Python脚本: python viser_colmap.py --dense_ply <fused.ply>"
print_success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
