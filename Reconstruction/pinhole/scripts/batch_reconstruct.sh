#!/bin/bash
#
# 批量重建脚本
# 使用 nohup 脱离终端运行多个场景的重建
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_PINHOLE_SCRIPT="$SCRIPT_DIR/run_pinhole.sh"

# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 显示使用说明
show_usage() {
    cat << EOF
批量重建脚本

用法: $0 [选项]

选项:
  --scenes SCENE1 SCENE2 ...  要重建的场景列表（默认: BridgeB Rooftop）
  --base-dir PATH             数据基础目录（默认: /root/autodl-tmp/data/MobilePhone）
  --config PATH               配置文件路径（默认: configs/config_pinhole.yaml）
  --matcher TYPE              匹配器类型 (exhaustive|sequential|spatial|vocabtree, 默认: exhaustive)
  --sparse-only               仅执行稀疏重建
  --dense-only                仅执行稠密重建（需要已有稀疏重建结果）
  --log-dir PATH              日志输出目录（默认: 脚本目录/logs）
  -h, --help                  显示此帮助信息

示例:
  # 默认重建 BridgeB 和 Rooftop
  $0

  # 指定场景列表
  $0 --scenes BridgeB Rooftop BridgeA

  # 仅稀疏重建
  $0 --scenes BridgeB Rooftop --sparse-only

  # 仅稠密重建
  $0 --scenes BridgeB Rooftop --dense-only

EOF
}

# 默认参数
SCENES=("BridgeB" "Road")
BASE_DIR="/root/autodl-tmp/data/MobilePhone"
CONFIG_PATH="$SCRIPT_DIR/../configs/config_pinhole.yaml"
MATCHER="exhaustive"
SPARSE_ONLY=false
DENSE_ONLY=false
LOG_DIR="$SCRIPT_DIR/logs"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes)
            SCENES=()
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SCENES+=("$1")
                shift
            done
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --matcher)
            MATCHER="$2"
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
        --log-dir)
            LOG_DIR="$2"
            shift 2
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

# 检查 run_pinhole.sh 脚本
if [ ! -f "$RUN_PINHOLE_SCRIPT" ]; then
    echo "错误: run_pinhole.sh 脚本不存在: $RUN_PINHOLE_SCRIPT"
    exit 1
fi

# 检查基础目录
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 基础目录不存在: $BASE_DIR"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/batch_reconstruct_${TIMESTAMP}.log"

# 记录开始信息
echo "=========================================="
echo "批量重建任务"
echo "=========================================="
echo "场景列表: ${SCENES[*]}"
echo "基础目录: $BASE_DIR"
echo "配置文件: $CONFIG_PATH"
echo "匹配器: $MATCHER"
echo "日志目录: $LOG_DIR"
echo "主日志文件: $MAIN_LOG"
if [ "$SPARSE_ONLY" = true ]; then
    echo "模式: 仅稀疏重建"
elif [ "$DENSE_ONLY" = true ]; then
    echo "模式: 仅稠密重建"
else
    echo "模式: 完整重建（稀疏+稠密）"
fi
echo "=========================================="
echo ""

# 验证场景目录
echo "验证场景目录..."
for scene in "${SCENES[@]}"; do
    scene_dir="$BASE_DIR/$scene"
    if [ ! -d "$scene_dir" ]; then
        echo "⚠️  警告: 场景目录不存在: $scene_dir"
    else
        echo "✅ 场景目录存在: $scene_dir"
    fi
done
echo ""

# 创建临时脚本文件（更可靠的方式）
TEMP_SCRIPT="$LOG_DIR/batch_reconstruct_${TIMESTAMP}.sh"
cat > "$TEMP_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
# 批量重建临时脚本
# 由 batch_reconstruct.sh 自动生成

set -e

# 从环境变量读取参数
BASE_DIR="${BATCH_BASE_DIR}"
CONFIG_PATH="${BATCH_CONFIG_PATH}"
MATCHER="${BATCH_MATCHER}"
SPARSE_ONLY="${BATCH_SPARSE_ONLY}"
DENSE_ONLY="${BATCH_DENSE_ONLY}"
RUN_PINHOLE_SCRIPT="${BATCH_RUN_PINHOLE_SCRIPT}"
MAIN_LOG="${BATCH_MAIN_LOG}"
LOG_DIR="${BATCH_LOG_DIR}"
TIMESTAMP="${BATCH_TIMESTAMP}"

# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 场景列表（从环境变量读取，空格分隔）
SCENES_STR="${BATCH_SCENES}"
read -ra SCENES <<< "$SCENES_STR"

SCRIPT_EOF

# 添加场景重建命令到临时脚本
for scene in "${SCENES[@]}"; do
    scene_dir="$BASE_DIR/$scene"
    input_path="$scene_dir/images"
    output_path="$scene_dir/colmap"
    scene_log="$LOG_DIR/${scene}_${TIMESTAMP}.log"
    
    echo "准备重建场景: $scene"
    echo "  输入路径: $input_path"
    echo "  输出路径: $output_path"
    echo "  场景日志: $scene_log"
    
    # 添加到临时脚本
    cat >> "$TEMP_SCRIPT" << SCRIPT_EOF

# 重建场景: $scene
echo "==========================================" >> "$MAIN_LOG"
echo "开始重建场景: $scene (\$(date '+%Y-%m-%d %H:%M:%S'))" >> "$MAIN_LOG"
echo "==========================================" >> "$MAIN_LOG"
bash "$RUN_PINHOLE_SCRIPT" \\
SCRIPT_EOF
    
    if [ "$DENSE_ONLY" = false ]; then
        echo "    -i \"$input_path\" \\" >> "$TEMP_SCRIPT"
    fi
    echo "    -o \"$output_path\" \\" >> "$TEMP_SCRIPT"
    echo "    -c \"$CONFIG_PATH\" \\" >> "$TEMP_SCRIPT"
    echo "    -m \"$MATCHER\" \\" >> "$TEMP_SCRIPT"
    
    if [ "$SPARSE_ONLY" = true ]; then
        echo "    --sparse-only \\" >> "$TEMP_SCRIPT"
    elif [ "$DENSE_ONLY" = true ]; then
        echo "    --dense-only \\" >> "$TEMP_SCRIPT"
    fi
    
    cat >> "$TEMP_SCRIPT" << SCRIPT_EOF
    >> "$scene_log" 2>&1

EXIT_CODE=\$?
echo "场景 $scene 重建完成，退出码: \$EXIT_CODE (\$(date '+%Y-%m-%d %H:%M:%S'))" >> "$MAIN_LOG"
if [ \$EXIT_CODE -ne 0 ]; then
    echo "❌ 场景 $scene 重建失败" >> "$MAIN_LOG"
else
    echo "✅ 场景 $scene 重建成功" >> "$MAIN_LOG"
fi
echo "" >> "$MAIN_LOG"
SCRIPT_EOF
done

# 添加完成信息
cat >> "$TEMP_SCRIPT" << 'SCRIPT_EOF'

echo "==========================================" >> "$MAIN_LOG"
echo "所有场景重建完成 ($(date '+%Y-%m-%d %H:%M:%S'))" >> "$MAIN_LOG"
echo "==========================================" >> "$MAIN_LOG"
SCRIPT_EOF

chmod +x "$TEMP_SCRIPT"

# 显示执行信息
echo "=========================================="
echo "启动批量重建任务（后台运行）"
echo "=========================================="
echo "主日志文件: $MAIN_LOG"
echo "场景日志文件:"
for scene in "${SCENES[@]}"; do
    echo "  - $LOG_DIR/${scene}_${TIMESTAMP}.log"
done
echo "临时脚本: $TEMP_SCRIPT"
echo ""
echo "使用以下命令查看日志:"
echo "  tail -f $MAIN_LOG"
echo "  或"
for scene in "${SCENES[@]}"; do
    echo "  tail -f $LOG_DIR/${scene}_${TIMESTAMP}.log"
done
echo ""
echo "使用以下命令查看进程:"
echo "  ps aux | grep run_pinhole"
echo ""
echo "正在启动..."
echo ""

# 设置环境变量供临时脚本使用
export BATCH_BASE_DIR="$BASE_DIR"
export BATCH_CONFIG_PATH="$CONFIG_PATH"
export BATCH_MATCHER="$MATCHER"
export BATCH_SPARSE_ONLY="$SPARSE_ONLY"
export BATCH_DENSE_ONLY="$DENSE_ONLY"
export BATCH_RUN_PINHOLE_SCRIPT="$RUN_PINHOLE_SCRIPT"
export BATCH_MAIN_LOG="$MAIN_LOG"
export BATCH_LOG_DIR="$LOG_DIR"
export BATCH_TIMESTAMP="$TIMESTAMP"
export BATCH_SCENES="${SCENES[*]}"

# 使用 nohup 执行临时脚本
nohup bash "$TEMP_SCRIPT" > "$LOG_DIR/nohup_${TIMESTAMP}.log" 2>&1 &

# 获取进程ID
PID=$!
echo "✅ 批量重建任务已启动"
echo "   进程ID: $PID"
echo "   主日志: $MAIN_LOG"
echo "   nohup日志: $LOG_DIR/nohup_${TIMESTAMP}.log"
echo ""
echo "提示: 任务已在后台运行，您可以安全关闭终端"
echo "      使用 'tail -f $MAIN_LOG' 查看实时日志"
echo "      使用 'kill $PID' 可以终止任务（如果需要）"