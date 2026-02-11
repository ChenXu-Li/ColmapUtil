#!/usr/bin/env python3
"""
从视频文件中均匀提取帧
支持从文件夹中的多个视频文件提取帧
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None，尝试从默认位置加载
    
    Returns:
        配置字典
    """
    default_config = {
        "paths": {
            "input": None,
            "output": None,
            "num_frames": None
        },
        "output": {
            "max_resolution": {
                "width": None,
                "height": None
            },
            "max_edge": None,
            "resize_mode": "fit",
            "format": "jpg",
            "jpeg_quality": 95
        },
        "extraction": {
            "prefix_mode": "video_name",
            "video_extensions": ["mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v"],
            "start_frame": 0,
            "end_frame": None,
            "distribute_mode": "equal"
        },
        "logging": {
            "level": "INFO",
            "show_resize_info": True
        }
    }
    
    if config_path is None:
        # 尝试从脚本目录的父目录查找 config.yaml
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config.yaml"
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 递归合并配置
                def merge_dict(base, user):
                    for key, value in user.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            merge_dict(base[key], value)
                        else:
                            base[key] = value
                merge_dict(default_config, user_config)
                logging.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            logging.warning(f"无法加载配置文件 {config_path}: {e}，使用默认配置")
    else:
        logging.info("未找到配置文件，使用默认配置")
    
    return default_config


def resize_image(
    image: cv2.Mat,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_edge: Optional[int] = None,
    resize_mode: str = "fit"
) -> Tuple[cv2.Mat, Tuple[int, int], Tuple[int, int]]:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        max_width: 最大宽度（像素），None 表示不限制
        max_height: 最大高度（像素），None 表示不限制
        max_edge: 最大边长（宽度和高度中的较大值），None 表示不限制
        resize_mode: 缩放模式
            - "fit": 保持宽高比，缩放到适合最大分辨率（默认）
            - "crop": 保持宽高比，裁剪到最大分辨率
            - "stretch": 不保持宽高比，拉伸到最大分辨率
    
    Returns:
        (调整后的图像, 原始尺寸 (w, h), 调整后尺寸 (w, h))
    """
    h, w = image.shape[:2]
    original_size = (w, h)
    
    # 确定目标尺寸
    if max_edge is not None:
        # 使用最大边长模式
        if max(w, h) <= max_edge:
            return image, original_size, original_size
        if w > h:
            target_width = max_edge
            target_height = int(h * max_edge / w)
        else:
            target_height = max_edge
            target_width = int(w * max_edge / h)
    elif max_width is not None or max_height is not None:
        # 使用宽度和高度限制
        max_w = max_width if max_width is not None else w
        max_h = max_height if max_height is not None else h
        
        if w <= max_w and h <= max_h:
            return image, original_size, original_size
        
        if resize_mode == "fit":
            # 保持宽高比，缩放到适合最大分辨率
            scale_w = max_w / w
            scale_h = max_h / h
            scale = min(scale_w, scale_h)
            target_width = int(w * scale)
            target_height = int(h * scale)
        elif resize_mode == "crop":
            # 保持宽高比，裁剪到最大分辨率
            scale_w = max_w / w
            scale_h = max_h / h
            scale = max(scale_w, scale_h)
            target_width = int(w * scale)
            target_height = int(h * scale)
            # 裁剪到目标尺寸
            crop_x = (target_width - max_w) // 2
            crop_y = (target_height - max_h) // 2
            resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            cropped = resized[crop_y:crop_y+max_h, crop_x:crop_x+max_w]
            return cropped, original_size, (max_w, max_h)
        else:  # stretch
            # 不保持宽高比，拉伸到最大分辨率
            target_width = max_w
            target_height = max_h
    else:
        # 没有限制
        return image, original_size, original_size
    
    # 执行缩放
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized, original_size, (target_width, target_height)


def get_video_info(video_path: Path) -> Tuple[int, float, int, int]:
    """
    获取视频信息
    
    Returns:
        total_frames: 总帧数
        fps: 帧率
        width: 视频宽度
        height: 视频高度
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return total_frames, fps, width, height


def extract_frames_uniform(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    prefix: str = "",
    image_format: str = "jpg",
    start_frame: int = 0,
    end_frame: int = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_edge: Optional[int] = None,
    resize_mode: str = "fit",
    jpeg_quality: int = 95,
    show_resize_info: bool = True
) -> List[Path]:
    """
    从视频中均匀提取指定数量的帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        num_frames: 要提取的帧数
        prefix: 输出文件名前缀
        image_format: 图像格式 (jpg, png)
        start_frame: 起始帧（可选）
        end_frame: 结束帧（可选，None表示到视频末尾）
        max_width: 最大宽度（像素），None 表示不限制
        max_height: 最大高度（像素），None 表示不限制
        max_edge: 最大边长（像素），None 表示不限制
        resize_mode: 缩放模式 (fit|crop|stretch)
        jpeg_quality: JPEG 质量（1-100）
        show_resize_info: 是否显示缩放信息
    
    Returns:
        保存的图像文件路径列表
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 确定实际提取范围
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    start_frame = max(0, min(start_frame, total_frames - 1))
    available_frames = end_frame - start_frame
    
    if available_frames <= 0:
        cap.release()
        raise ValueError(f"无效的帧范围: start={start_frame}, end={end_frame}, total={total_frames}")
    
    # 计算要提取的帧索引（均匀分布）
    if num_frames >= available_frames:
        # 如果请求的帧数大于等于可用帧数，提取所有帧
        frame_indices = list(range(start_frame, end_frame))
        logging.warning(f"请求 {num_frames} 帧，但只有 {available_frames} 帧可用，将提取所有帧")
    else:
        # 均匀分布
        step = available_frames / num_frames
        frame_indices = [start_frame + int(i * step) for i in range(num_frames)]
        # 确保最后一个帧索引不超过范围
        frame_indices[-1] = min(frame_indices[-1], end_frame - 1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）作为默认前缀
    if not prefix:
        prefix = video_path.stem
    
    saved_paths = []
    frame_count = 0
    resize_count = 0
    
    logging.info(f"从视频提取 {len(frame_indices)} 帧 (总帧数: {total_frames}, FPS: {fps:.2f})")
    
    for target_frame_idx in frame_indices:
        # 跳转到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logging.warning(f"无法读取第 {target_frame_idx} 帧，跳过")
            continue
        
        # 调整图像大小（如果需要）
        original_size = (frame.shape[1], frame.shape[0])
        if max_width is not None or max_height is not None or max_edge is not None:
            frame, orig_size, new_size = resize_image(
                frame,
                max_width=max_width,
                max_height=max_height,
                max_edge=max_edge,
                resize_mode=resize_mode
            )
            if orig_size != new_size:
                resize_count += 1
                if show_resize_info and frame_count == 0:
                    logging.info(f"图像缩放: {orig_size[0]}x{orig_size[1]} -> {new_size[0]}x{new_size[1]}")
        
        # 生成输出文件名
        output_filename = f"{prefix}_frame_{target_frame_idx:06d}.{image_format}"
        output_path = output_dir / output_filename
        
        # 保存帧
        save_params = []
        if image_format == "jpg":
            save_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        elif image_format == "png":
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        
        cv2.imwrite(str(output_path), frame, save_params)
        saved_paths.append(output_path)
        frame_count += 1
        
        if frame_count % 10 == 0:
            logging.info(f"已提取 {frame_count}/{len(frame_indices)} 帧")
    
    if resize_count > 0 and show_resize_info:
        logging.info(f"共缩放 {resize_count} 帧图像")
    
    cap.release()
    logging.info(f"✅ 成功提取 {frame_count} 帧到 {output_dir}")
    
    return saved_paths


def process_video_folder(
    input_dir: Path,
    output_dir: Path,
    num_frames_total: int,
    image_format: str = "jpg",
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"),
    start_frame: int = 0,
    end_frame: int = None,
    prefix_mode: str = "video_name",  # "video_name", "sequential", "none"
    distribute_by_duration: bool = True,  # True: 按时长比例分配, False: 每个视频相同数量
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    max_edge: Optional[int] = None,
    resize_mode: str = "fit",
    jpeg_quality: int = 95,
    show_resize_info: bool = True
) -> dict:
    """
    处理文件夹中的所有视频文件
    
    Args:
        input_dir: 输入视频文件夹
        output_dir: 输出图像文件夹
        num_frames_total: 总共要提取的帧数（如果distribute_by_duration=True）
                         或每个视频提取的帧数（如果distribute_by_duration=False）
        image_format: 图像格式
        video_extensions: 支持的视频扩展名
        start_frame: 起始帧（每个视频）
        end_frame: 结束帧（每个视频，None表示到末尾）
        prefix_mode: 前缀模式
            - "video_name": 使用视频文件名作为前缀
            - "sequential": 使用序号作为前缀 (video_001, video_002, ...)
            - "none": 不使用前缀
        distribute_by_duration: 如果True，按视频时长比例分配总帧数
                                如果False，每个视频提取相同数量的帧
    
    Returns:
        处理结果统计
    """
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    video_files = sorted(video_files)
    
    if len(video_files) == 0:
        logging.warning(f"在 {input_dir} 中未找到视频文件")
        return {"total_videos": 0, "total_frames": 0, "success": 0, "failed": 0}
    
    logging.info(f"找到 {len(video_files)} 个视频文件")
    
    # 计算每个视频应该提取的帧数
    frames_per_video = []
    
    if distribute_by_duration:
        # 按视频时长比例分配总帧数
        logging.info("按视频时长比例分配帧数...")
        video_durations = []  # 存储每个视频的可用帧数
        
        for video_path in video_files:
            try:
                total_frames, fps, width, height = get_video_info(video_path)
                # 计算可用帧数
                if end_frame is None:
                    available_frames = total_frames - start_frame
                else:
                    available_frames = min(end_frame, total_frames) - start_frame
                available_frames = max(0, available_frames)
                video_durations.append(available_frames)
                logging.info(f"  {video_path.name}: {available_frames} 帧 (总帧数: {total_frames}, FPS: {fps:.2f})")
            except Exception as e:
                logging.warning(f"  无法获取 {video_path.name} 的信息: {e}")
                video_durations.append(0)
        
        total_duration = sum(video_durations)
        
        if total_duration == 0:
            logging.error("所有视频的可用帧数为0，无法分配")
            return {"total_videos": len(video_files), "total_frames": 0, "success": 0, "failed": len(video_files), "failed_videos": [v.name for v in video_files]}
        
        # 按比例分配帧数
        allocated_total = 0
        for idx, duration in enumerate(video_durations):
            if duration > 0:
                # 按比例分配，最后一个视频分配剩余的所有帧
                if idx == len(video_durations) - 1:
                    frames = num_frames_total - allocated_total
                else:
                    frames = int(num_frames_total * duration / total_duration)
                frames = max(1, frames)  # 至少分配1帧
                frames_per_video.append(frames)
                allocated_total += frames
            else:
                frames_per_video.append(0)
        
        logging.info("帧数分配:")
        for idx, (video_path, frames) in enumerate(zip(video_files, frames_per_video), 1):
            if frames > 0:
                logging.info(f"  [{idx}] {video_path.name}: {frames} 帧")
        
        if allocated_total != num_frames_total:
            logging.warning(f"分配的帧数 ({allocated_total}) 与请求的帧数 ({num_frames_total}) 不完全匹配（由于取整）")
    else:
        # 每个视频提取相同数量的帧
        frames_per_video = [num_frames_total] * len(video_files)
        logging.info(f"每个视频提取 {num_frames_total} 帧")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "total_videos": len(video_files),
        "total_frames": 0,
        "success": 0,
        "failed": 0,
        "failed_videos": []
    }
    
    for idx, video_path in enumerate(video_files, 1):
        logging.info("=" * 60)
        logging.info(f"处理视频 [{idx}/{len(video_files)}]: {video_path.name}")
        logging.info("=" * 60)
        
        try:
            # 确定前缀
            if prefix_mode == "video_name":
                prefix = video_path.stem
            elif prefix_mode == "sequential":
                prefix = f"video_{idx:03d}"
            else:
                prefix = ""
            
            # 获取该视频应该提取的帧数
            num_frames_this_video = frames_per_video[idx - 1]
            
            if num_frames_this_video == 0:
                logging.warning(f"跳过 {video_path.name}（分配的帧数为0）")
                continue
            
            # 提取帧
            saved_paths = extract_frames_uniform(
                video_path=video_path,
                output_dir=output_dir,
                num_frames=num_frames_this_video,
                prefix=prefix,
                image_format=image_format,
                start_frame=start_frame,
                end_frame=end_frame,
                max_width=max_width,
                max_height=max_height,
                max_edge=max_edge,
                resize_mode=resize_mode,
                jpeg_quality=jpeg_quality,
                show_resize_info=show_resize_info
            )
            
            results["total_frames"] += len(saved_paths)
            results["success"] += 1
            
        except Exception as e:
            logging.error(f"处理视频失败 {video_path.name}: {e}")
            results["failed"] += 1
            results["failed_videos"].append(video_path.name)
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="从视频文件中均匀提取帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从单个视频提取30帧
  python extract_frames.py -i video.mp4 -o output/ -n 30

  # 从文件夹中的所有视频各提取50帧
  python extract_frames.py -i videos/ -o frames/ -n 50

  # 指定图像格式和帧范围
  python extract_frames.py -i videos/ -o frames/ -n 30 --format png --start-frame 100 --end-frame 1000

  # 使用序号前缀
  python extract_frames.py -i videos/ -o frames/ -n 30 --prefix-mode sequential
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="输入视频文件或文件夹路径（如果未指定，将从配置文件读取）"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出图像文件夹路径（如果未指定，将从配置文件读取）"
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=None,
        help="要提取的帧数（如果未指定，将从配置文件读取；如果--distribute-by-duration，则为总帧数；否则为每个视频的帧数）"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="输出图像格式 (默认: jpg)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="起始帧（每个视频，默认: 0）"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="结束帧（每个视频，None表示到视频末尾，默认: None）"
    )
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default=None,
        choices=["video_name", "sequential", "none"],
        help="文件名前缀模式（如果未指定，将从配置文件读取）"
    )
    parser.add_argument(
        "--video-extensions",
        type=str,
        nargs="+",
        default=["mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v"],
        help="支持的视频扩展名 (默认: mp4 avi mov mkv flv wmv m4v)"
    )
    parser.add_argument(
        "--distribute-by-duration",
        action="store_true",
        help="按视频时长比例分配总帧数（仅对文件夹模式有效）"
    )
    parser.add_argument(
        "--equal-frames",
        action="store_true",
        help="每个视频提取相同数量的帧（默认行为，与--distribute-by-duration互斥）"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="配置文件路径（默认: ExtractFrames/config.yaml）"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="输出图像最大宽度（像素），覆盖配置文件设置"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="输出图像最大高度（像素），覆盖配置文件设置"
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=None,
        help="输出图像最大边长（像素），覆盖配置文件设置"
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default=None,
        choices=["fit", "crop", "stretch"],
        help="缩放模式 (fit|crop|stretch)，覆盖配置文件设置"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG 质量 (1-100)，覆盖配置文件设置"
    )
    
    args = parser.parse_args()
    
    # 规范化图像格式
    if args.format == "jpeg":
        args.format = "jpg"
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 从配置文件或命令行参数获取路径设置（命令行参数优先）
    paths_config = config.get("paths", {})
    input_path = args.input if args.input is not None else (Path(paths_config.get("input")) if paths_config.get("input") else None)
    output_path = args.output if args.output is not None else (Path(paths_config.get("output")) if paths_config.get("output") else None)
    num_frames = args.num_frames if args.num_frames is not None else paths_config.get("num_frames")
    
    # 检查必需参数
    if input_path is None:
        logging.error("必须指定输入路径（通过 -i/--input 参数或在配置文件的 paths.input 中指定）")
        sys.exit(1)
    if output_path is None:
        logging.error("必须指定输出路径（通过 -o/--output 参数或在配置文件的 paths.output 中指定）")
        sys.exit(1)
    if num_frames is None:
        logging.error("必须指定帧数（通过 -n/--num-frames 参数或在配置文件的 paths.num_frames 中指定）")
        sys.exit(1)
    
    # 从配置文件或命令行参数获取设置（命令行参数优先）
    output_config = config.get("output", {})
    extraction_config = config.get("extraction", {})
    max_resolution = output_config.get("max_resolution", {})
    
    # 输出配置
    max_width = args.max_width if args.max_width is not None else max_resolution.get("width")
    max_height = args.max_height if args.max_height is not None else max_resolution.get("height")
    max_edge = args.max_edge if args.max_edge is not None else output_config.get("max_edge")
    resize_mode = args.resize_mode if args.resize_mode is not None else output_config.get("resize_mode", "fit")
    jpeg_quality = args.jpeg_quality if args.jpeg_quality is not None else output_config.get("jpeg_quality", 95)
    image_format = args.format if args.format else output_config.get("format", "jpg")
    
    # 提取配置
    prefix_mode = args.prefix_mode if args.prefix_mode is not None else extraction_config.get("prefix_mode", "video_name")
    video_extensions = args.video_extensions if args.video_extensions else extraction_config.get("video_extensions", ["mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v"])
    # 处理 start_frame 和 end_frame
    # 注意：由于 argparse 的 default=0，我们无法区分用户是否显式指定了 --start-frame
    # 策略：如果命令行参数使用了默认值，则尝试从配置文件读取；否则使用命令行值
    # 对于 start_frame：如果值为 0 且配置文件中有设置，使用配置文件值；否则使用命令行值
    config_start_frame = extraction_config.get("start_frame", 0)
    # 如果命令行是默认值 0，且配置文件中有非 0 值，使用配置文件值
    # 否则使用命令行值（包括显式指定的 0）
    start_frame = config_start_frame if args.start_frame == 0 and config_start_frame != 0 else args.start_frame
    
    # 处理 end_frame：命令行参数优先，如果未指定（None）则使用配置文件值
    end_frame = args.end_frame if args.end_frame is not None else extraction_config.get("end_frame")
    
    # 日志配置
    show_resize_info = config.get("logging", {}).get("show_resize_info", True)
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    logging.info("=" * 60)
    logging.info("视频帧提取工具")
    logging.info("=" * 60)
    
    # 确定分配模式（命令行参数优先）
    if args.equal_frames and args.distribute_by_duration:
        logging.error("--distribute-by-duration 和 --equal-frames 不能同时使用")
        sys.exit(1)
    
    # 从配置文件读取分配模式（如果命令行未指定）
    if input_path.is_dir():
        if args.distribute_by_duration:
            distribute_mode = True
        elif args.equal_frames:
            distribute_mode = False
        else:
            # 从配置文件读取
            config_distribute_mode = extraction_config.get("distribute_mode", "equal")
            distribute_mode = (config_distribute_mode == "distribute")
    else:
        distribute_mode = False
    
    logging.info(f"输入: {input_path}")
    logging.info(f"输出: {output_path}")
    if distribute_mode:
        logging.info(f"总帧数: {num_frames}（按视频时长比例分配）")
    else:
        logging.info(f"每视频帧数: {num_frames}")
    logging.info(f"图像格式: {image_format}")
    logging.info(f"前缀模式: {prefix_mode}")
    if start_frame > 0 or end_frame is not None:
        logging.info(f"帧范围: {start_frame} - {end_frame}")
    if max_width is not None or max_height is not None or max_edge is not None:
        if max_edge is not None:
            logging.info(f"最大分辨率: {max_edge}px (最大边长)")
        else:
            logging.info(f"最大分辨率: {max_width or '无限制'}x{max_height or '无限制'}")
        logging.info(f"缩放模式: {resize_mode}")
    if image_format == "jpg":
        logging.info(f"JPEG 质量: {jpeg_quality}")
    logging.info("=" * 60)
    
    # 检查输入路径
    if not input_path.exists():
        logging.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 判断是文件还是文件夹
    if input_path.is_file():
        # 单个视频文件
        logging.info("处理单个视频文件...")
        try:
            saved_paths = extract_frames_uniform(
                video_path=input_path,
                output_dir=output_path,
                num_frames=num_frames,
                prefix=input_path.stem if prefix_mode == "video_name" else "",
                image_format=image_format,
                start_frame=start_frame,
                end_frame=end_frame,
                max_width=max_width,
                max_height=max_height,
                max_edge=max_edge,
                resize_mode=resize_mode,
                jpeg_quality=jpeg_quality,
                show_resize_info=show_resize_info
            )
            logging.info(f"✅ 成功提取 {len(saved_paths)} 帧")
        except Exception as e:
            logging.error(f"❌ 处理失败: {e}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # 视频文件夹
        logging.info("处理视频文件夹...")
        results = process_video_folder(
            input_dir=input_path,
            output_dir=output_path,
            num_frames_total=num_frames,
            image_format=image_format,
            video_extensions=tuple(f".{ext.lstrip('.')}" for ext in video_extensions),
            start_frame=start_frame,
            end_frame=end_frame,
            prefix_mode=prefix_mode,
            distribute_by_duration=distribute_mode,
            max_width=max_width,
            max_height=max_height,
            max_edge=max_edge,
            resize_mode=resize_mode,
            jpeg_quality=jpeg_quality,
            show_resize_info=show_resize_info
        )
        
        logging.info("=" * 60)
        logging.info("处理完成！")
        logging.info("=" * 60)
        logging.info(f"总视频数: {results['total_videos']}")
        logging.info(f"成功: {results['success']}")
        logging.info(f"失败: {results['failed']}")
        logging.info(f"总提取帧数: {results['total_frames']}")
        
        if results['failed'] > 0:
            logging.warning(f"失败的视频: {', '.join(results['failed_videos'])}")
            sys.exit(1)
    else:
        logging.error(f"输入路径既不是文件也不是文件夹: {input_path}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("✅ 所有处理完成！")
    logging.info(f"输出目录: {output_path}")


if __name__ == "__main__":
    main()
