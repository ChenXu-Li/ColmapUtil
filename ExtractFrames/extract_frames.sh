#!/bin/bash
#
# 从视频文件中均匀提取帧的便捷脚本
# 用法: ./extract_frames.sh <input> <output> <num_frames> [options]
#./extract_frames.sh /root/autodl-tmp/data/MobilePhone/BridgeA/Videos/ /root/autodl-tmp/data/MobilePhone/BridgeA/images 50 --distribute --prefix-mode none

set -e
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/scripts/extract_frames.py"

# 显示使用说明
show_usage() {
    cat << EOF
从视频文件中均匀提取帧

用法: $0 [input] [output] [num_frames] [选项]

参数（如果未指定，将从配置文件读取）:
  input         输入视频文件或文件夹路径
  output        输出图像文件夹路径
  num_frames    要提取的帧数
                - 单个视频：提取的帧数
                - 文件夹：总帧数（如果使用--distribute）或每视频帧数（默认）

注意: 如果使用配置文件，可以省略 input/output/num_frames 参数
      例如: $0 --config config.yaml

选项:
  --format FORMAT          图像格式 (jpg|png, 默认: jpg)
  --start-frame N          起始帧（默认: 0）
  --end-frame N            结束帧（默认: 到视频末尾）
  --prefix-mode MODE       前缀模式 (video_name|sequential|none, 默认: video_name)
  --distribute             按视频时长比例分配总帧数（文件夹模式）
  --equal-frames           每个视频提取相同数量的帧（默认，文件夹模式）
  -c, --config PATH        配置文件路径（默认: ExtractFrames/config.yaml）
  --max-width N            输出图像最大宽度（像素），覆盖配置文件设置
  --max-height N           输出图像最大高度（像素），覆盖配置文件设置
  --max-edge N             输出图像最大边长（像素），覆盖配置文件设置
  --resize-mode MODE       缩放模式 (fit|crop|stretch)，覆盖配置文件设置
  --jpeg-quality N         JPEG 质量 (1-100)，覆盖配置文件设置
  -h, --help               显示此帮助信息

示例:
  # 从单个视频提取30帧
  $0 video.mp4 output/ 30

  # 从文件夹中的所有视频总共提取100帧（按时长比例分配）
  $0 videos/ frames/ 100 --distribute

  # 从文件夹中的所有视频各提取50帧
  $0 videos/ frames/ 50

  # 指定图像格式
  $0 videos/ frames/ 30 --format png

  # 指定帧范围
  $0 videos/ frames/ 30 --start-frame 100 --end-frame 1000

EOF
}

# 解析参数
# 检查第一个参数是否是选项（以 - 开头）
if [ $# -gt 0 ] && [[ "$1" == -* ]]; then
    # 第一个参数是选项，说明使用配置文件模式（路径从配置文件读取）
    INPUT=""
    OUTPUT=""
    NUM_FRAMES=""
else
    # 传统模式：需要至少3个位置参数
    if [ $# -lt 3 ]; then
        echo "错误: 参数不足（需要至少3个位置参数，或使用 --config 参数）"
        show_usage
        exit 1
    fi
    INPUT="$1"
    OUTPUT="$2"
    NUM_FRAMES="$3"
    shift 3
fi

# 解析可选参数
CONFIG_PATH=""

# 解析可选参数
FORMAT="jpg"
START_FRAME="0"
END_FRAME=""
PREFIX_MODE=""  # 空字符串表示未指定，将从配置文件读取
DISTRIBUTE=""
CONFIG_PATH=""
MAX_WIDTH=""
MAX_HEIGHT=""
MAX_EDGE=""
RESIZE_MODE=""
JPEG_QUALITY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --start-frame)
            START_FRAME="$2"
            shift 2
            ;;
        --end-frame)
            END_FRAME="$2"
            shift 2
            ;;
        --prefix-mode)
            PREFIX_MODE="$2"
            shift 2
            ;;
        --distribute)
            DISTRIBUTE="--distribute-by-duration"
            shift
            ;;
        --equal-frames)
            DISTRIBUTE="--equal-frames"
            shift
            ;;
        -c|--config)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                # 如果没有提供路径，使用默认配置文件
                CONFIG_PATH="$SCRIPT_DIR/config.yaml"
                shift
            else
                CONFIG_PATH="$2"
                shift 2
            fi
            ;;
        --max-width)
            MAX_WIDTH="$2"
            shift 2
            ;;
        --max-height)
            MAX_HEIGHT="$2"
            shift 2
            ;;
        --max-edge)
            MAX_EDGE="$2"
            shift 2
            ;;
        --resize-mode)
            RESIZE_MODE="$2"
            shift 2
            ;;
        --jpeg-quality)
            JPEG_QUALITY="$2"
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

# 检查Python脚本
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 构建Python命令
# 注意：只有当位置参数非空时才添加，否则让 Python 脚本从配置文件读取
PYTHON_ARGS=()

if [ -n "$INPUT" ]; then
    PYTHON_ARGS+=("--input" "$INPUT")
fi

if [ -n "$OUTPUT" ]; then
    PYTHON_ARGS+=("--output" "$OUTPUT")
fi

if [ -n "$NUM_FRAMES" ]; then
    PYTHON_ARGS+=("--num-frames" "$NUM_FRAMES")
fi

PYTHON_ARGS+=(
    "--format" "$FORMAT"
    "--start-frame" "$START_FRAME"
)

# 只在显式指定了 prefix_mode 时才添加参数
if [ -n "$PREFIX_MODE" ]; then
    PYTHON_ARGS+=("--prefix-mode" "$PREFIX_MODE")
fi

if [ -n "$END_FRAME" ]; then
    PYTHON_ARGS+=("--end-frame" "$END_FRAME")
fi

if [ -n "$DISTRIBUTE" ]; then
    PYTHON_ARGS+=("$DISTRIBUTE")
fi

if [ -n "$CONFIG_PATH" ]; then
    PYTHON_ARGS+=("--config" "$CONFIG_PATH")
fi

if [ -n "$MAX_WIDTH" ]; then
    PYTHON_ARGS+=("--max-width" "$MAX_WIDTH")
fi

if [ -n "$MAX_HEIGHT" ]; then
    PYTHON_ARGS+=("--max-height" "$MAX_HEIGHT")
fi

if [ -n "$MAX_EDGE" ]; then
    PYTHON_ARGS+=("--max-edge" "$MAX_EDGE")
fi

if [ -n "$RESIZE_MODE" ]; then
    PYTHON_ARGS+=("--resize-mode" "$RESIZE_MODE")
fi

if [ -n "$JPEG_QUALITY" ]; then
    PYTHON_ARGS+=("--jpeg-quality" "$JPEG_QUALITY")
fi

# 如果没有指定配置文件路径，但使用了配置文件模式（没有提供位置参数），使用默认配置文件
if [ -z "$CONFIG_PATH" ] && [ -z "$INPUT" ] && [ -z "$OUTPUT" ] && [ -z "$NUM_FRAMES" ]; then
    DEFAULT_CONFIG="$SCRIPT_DIR/config.yaml"
    if [ -f "$DEFAULT_CONFIG" ]; then
        CONFIG_PATH="$DEFAULT_CONFIG"
    fi
fi

echo "=========================================="
echo "视频帧提取"
echo "=========================================="
if [ -n "$INPUT" ]; then
    echo "输入: $INPUT"
else
    echo "输入: （从配置文件读取）"
fi
if [ -n "$OUTPUT" ]; then
    echo "输出: $OUTPUT"
else
    echo "输出: （从配置文件读取）"
fi
if [ -n "$NUM_FRAMES" ]; then
    if [ -n "$DISTRIBUTE" ] && [ "$DISTRIBUTE" = "--distribute-by-duration" ]; then
        echo "总帧数: $NUM_FRAMES（按视频时长比例分配）"
    else
        echo "每视频帧数: $NUM_FRAMES"
    fi
else
    echo "帧数: （从配置文件读取）"
fi
echo "图像格式: $FORMAT"
if [ -n "$CONFIG_PATH" ]; then
    echo "配置文件: $CONFIG_PATH"
fi
echo "=========================================="
echo ""

# 执行提取
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
