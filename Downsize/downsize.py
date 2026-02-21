#!/usr/bin/env python3
"""
图片批量下采样工具
Image Batch Downsampling Tool

支持多种下采样方式、并行处理、进度显示等功能
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ============================================================
# 配置类
# ============================================================

@dataclass
class DownsizeConfig:
    """下采样配置"""
    # 路径
    input_dir: Path
    output_dir: Path
    
    # 下采样参数
    factor: int = 2
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    keep_aspect_ratio: bool = True
    
    # 处理参数
    interpolation: str = "AREA"
    jpeg_quality: int = 95
    png_compression: int = 3
    output_format: str = "auto"
    
    # 过滤
    extensions: List[str] = None
    recursive: bool = False
    exclude_patterns: List[str] = None
    
    # 并行
    num_workers: int = 0
    batch_size: int = 10
    
    # 输出
    overwrite: bool = False
    preserve_structure: bool = True
    suffix: str = ""
    
    # 日志
    log_level: str = "INFO"
    show_progress: bool = True
    show_stats: bool = True
    
    def __post_init__(self):
        if self.extensions is None:
            self.extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        if self.num_workers == 0:
            self.num_workers = os.cpu_count() or 4


def load_config(config_path: Path) -> DownsizeConfig:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    paths = cfg.get('paths', {})
    downsampling = cfg.get('downsampling', {})
    processing = cfg.get('processing', {})
    filter_cfg = cfg.get('filter', {})
    parallel = cfg.get('parallel', {})
    output = cfg.get('output', {})
    logging = cfg.get('logging', {})
    
    return DownsizeConfig(
        input_dir=Path(paths.get('input_dir', '.')),
        output_dir=Path(paths.get('output_dir', './output')),
        factor=downsampling.get('factor', 2),
        target_width=downsampling.get('target_width'),
        target_height=downsampling.get('target_height'),
        keep_aspect_ratio=downsampling.get('keep_aspect_ratio', True),
        interpolation=processing.get('interpolation', 'AREA'),
        jpeg_quality=processing.get('jpeg_quality', 95),
        png_compression=processing.get('png_compression', 3),
        output_format=processing.get('output_format', 'auto'),
        extensions=filter_cfg.get('extensions', [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]),
        recursive=filter_cfg.get('recursive', False),
        exclude_patterns=filter_cfg.get('exclude_patterns'),
        num_workers=parallel.get('num_workers', 0),
        batch_size=parallel.get('batch_size', 10),
        overwrite=output.get('overwrite', False),
        preserve_structure=output.get('preserve_structure', True),
        suffix=output.get('suffix', ''),
        log_level=logging.get('level', 'INFO'),
        show_progress=logging.get('show_progress', True),
        show_stats=logging.get('show_stats', True),
    )


# ============================================================
# 日志工具
# ============================================================

class Logger:
    """带颜色的日志输出"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }
    
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'PROGRESS': '📊',
        'FILE': '📄',
        'FOLDER': '📁',
        'TIME': '⏱️ ',
        'SUCCESS': '🎉',
        'START': '🚀',
    }
    
    def __init__(self, level: str = 'INFO'):
        self.level = level
        self.level_order = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
    
    def _should_log(self, level: str) -> bool:
        return self.level_order.get(level, 1) >= self.level_order.get(self.level, 1)
    
    def _format(self, level: str, msg: str, icon: str = None) -> str:
        color = self.COLORS.get(level, '')
        reset = self.COLORS['RESET']
        icon_str = icon or self.ICONS.get(level, '')
        return f"{color}{icon_str} {msg}{reset}"
    
    def debug(self, msg: str):
        if self._should_log('DEBUG'):
            print(self._format('DEBUG', msg))
    
    def info(self, msg: str, icon: str = None):
        if self._should_log('INFO'):
            print(self._format('INFO', msg, icon or self.ICONS['INFO']))
    
    def warning(self, msg: str):
        if self._should_log('WARNING'):
            print(self._format('WARNING', msg))
    
    def error(self, msg: str):
        if self._should_log('ERROR'):
            print(self._format('ERROR', msg))
    
    def header(self, msg: str):
        """打印标题"""
        print(f"\n{self.COLORS['BOLD']}{'='*60}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{msg}{self.COLORS['RESET']}")
        print(f"{self.COLORS['BOLD']}{'='*60}{self.COLORS['RESET']}\n")
    
    def section(self, msg: str):
        """打印分节标题"""
        print(f"\n{self.COLORS['BOLD']}--- {msg} ---{self.COLORS['RESET']}")


# ============================================================
# 图片处理
# ============================================================

INTERPOLATION_MAP = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'AREA': cv2.INTER_AREA,
    'CUBIC': cv2.INTER_CUBIC,
    'LANCZOS4': cv2.INTER_LANCZOS4,
}


def get_image_files(config: DownsizeConfig, logger: Logger) -> List[Tuple[Path, Path]]:
    """
    获取所有待处理的图片文件
    返回: [(input_path, output_path), ...]
    """
    input_dir = config.input_dir
    output_dir = config.output_dir
    
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return []
    
    # 获取所有图片文件
    files = []
    extensions = set(ext.lower() for ext in config.extensions)
    
    if config.recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for path in input_dir.glob(pattern):
        if not path.is_file():
            continue
        
        if path.suffix.lower() not in extensions:
            continue
        
        # 检查排除模式
        if config.exclude_patterns:
            excluded = False
            for pattern in config.exclude_patterns:
                if re.search(pattern, path.name):
                    excluded = True
                    break
            if excluded:
                logger.debug(f"跳过排除文件: {path.name}")
                continue
        
        # 计算输出路径
        if config.recursive and config.preserve_structure:
            rel_path = path.relative_to(input_dir)
            out_path = output_dir / rel_path
        else:
            out_path = output_dir / path.name
        
        # 添加后缀
        if config.suffix:
            out_path = out_path.with_stem(out_path.stem + config.suffix)
        
        # 修改输出格式
        if config.output_format != 'auto':
            fmt = config.output_format.lower()
            if fmt == 'jpg':
                out_path = out_path.with_suffix('.jpg')
            elif fmt == 'png':
                out_path = out_path.with_suffix('.png')
            elif fmt == 'webp':
                out_path = out_path.with_suffix('.webp')
        
        # 检查是否需要覆盖
        if out_path.exists() and not config.overwrite:
            logger.debug(f"跳过已存在文件: {out_path.name}")
            continue
        
        files.append((path, out_path))
    
    return files


def calculate_new_size(
    original_size: Tuple[int, int],
    config: DownsizeConfig
) -> Tuple[int, int]:
    """计算新的图片尺寸"""
    orig_w, orig_h = original_size
    
    if config.target_width or config.target_height:
        # 使用目标尺寸
        if config.keep_aspect_ratio:
            if config.target_width and config.target_height:
                # 两者都指定时，选择较小的缩放因子
                scale_w = config.target_width / orig_w
                scale_h = config.target_height / orig_h
                scale = min(scale_w, scale_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
            elif config.target_width:
                scale = config.target_width / orig_w
                new_w = config.target_width
                new_h = int(orig_h * scale)
            else:
                scale = config.target_height / orig_h
                new_w = int(orig_w * scale)
                new_h = config.target_height
        else:
            new_w = config.target_width or orig_w
            new_h = config.target_height or orig_h
    else:
        # 使用缩放因子
        new_w = orig_w // config.factor
        new_h = orig_h // config.factor
    
    # 确保至少 1 像素
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    return new_w, new_h


def process_single_image(
    input_path: Path,
    output_path: Path,
    config: DownsizeConfig,
    logger: Logger
) -> Tuple[bool, str, dict]:
    """
    处理单张图片
    返回: (success, message, stats)
    """
    stats = {
        'input_size': (0, 0),
        'output_size': (0, 0),
        'input_bytes': 0,
        'output_bytes': 0,
        'time_ms': 0,
    }
    
    try:
        start_time = time.time()
        
        # 读取图片
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, f"无法读取图片: {input_path.name}", stats
        
        stats['input_size'] = (img.shape[1], img.shape[0])
        stats['input_bytes'] = input_path.stat().st_size
        
        # 计算新尺寸
        new_w, new_h = calculate_new_size(stats['input_size'], config)
        stats['output_size'] = (new_w, new_h)
        
        # 获取插值方法
        interp = INTERPOLATION_MAP.get(config.interpolation.upper(), cv2.INTER_AREA)
        
        # 缩放图片
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图片
        ext = output_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression]
        elif ext == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, config.jpeg_quality]
        else:
            params = []
        
        success = cv2.imwrite(str(output_path), resized, params)
        
        if not success:
            return False, f"保存失败: {output_path.name}", stats
        
        stats['output_bytes'] = output_path.stat().st_size
        stats['time_ms'] = (time.time() - start_time) * 1000
        
        return True, f"完成: {input_path.name}", stats
        
    except Exception as e:
        return False, f"处理出错 {input_path.name}: {str(e)}", stats


def process_images(config: DownsizeConfig, logger: Logger) -> dict:
    """
    批量处理图片
    返回统计信息
    """
    # 获取文件列表
    logger.section("扫描图片文件")
    files = get_image_files(config, logger)
    
    if not files:
        logger.warning("没有找到需要处理的图片文件")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    logger.info(f"找到 {len(files)} 张图片待处理", Logger.ICONS['FILE'])
    logger.info(f"输入目录: {config.input_dir}", Logger.ICONS['FOLDER'])
    logger.info(f"输出目录: {config.output_dir}", Logger.ICONS['FOLDER'])
    
    # 显示下采样参数
    logger.section("下采样参数")
    if config.target_width or config.target_height:
        target = f"{config.target_width or 'auto'}x{config.target_height or 'auto'}"
        logger.info(f"目标尺寸: {target}")
    else:
        logger.info(f"下采样因子: {config.factor}x")
    logger.info(f"插值方法: {config.interpolation}")
    logger.info(f"并行线程: {config.num_workers}")
    
    # 创建输出目录
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 开始处理
    logger.section("开始处理")
    start_time = time.time()
    
    results = {
        'total': len(files),
        'success': 0,
        'failed': 0,
        'total_input_bytes': 0,
        'total_output_bytes': 0,
        'total_time_ms': 0,
        'failed_files': [],
        'input_size': None,   # 原始图片尺寸 (w, h)
        'output_size': None,  # 输出图片尺寸 (w, h)
    }
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_single_image, inp, out, config, logger): (inp, out)
            for inp, out in files
        }
        
        # 进度条
        if config.show_progress:
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="处理进度",
                unit="张",
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            iterator = as_completed(futures)
        
        for future in iterator:
            inp, out = futures[future]
            try:
                success, msg, stats = future.result()
                
                if success:
                    results['success'] += 1
                    results['total_input_bytes'] += stats['input_bytes']
                    results['total_output_bytes'] += stats['output_bytes']
                    results['total_time_ms'] += stats['time_ms']
                    # 记录第一张图片的尺寸（用于显示）
                    if results['input_size'] is None:
                        results['input_size'] = stats['input_size']
                        results['output_size'] = stats['output_size']
                    logger.debug(msg)
                else:
                    results['failed'] += 1
                    results['failed_files'].append((inp, msg))
                    logger.warning(msg)
                    
            except Exception as e:
                results['failed'] += 1
                results['failed_files'].append((inp, str(e)))
                logger.error(f"处理异常 {inp.name}: {e}")
    
    results['wall_time'] = time.time() - start_time
    
    return results


def print_stats(results: dict, logger: Logger):
    """打印统计信息"""
    logger.section("处理统计")
    
    total = results['total']
    success = results['success']
    failed = results['failed']
    
    # 成功率
    if total > 0:
        success_rate = success / total * 100
        logger.info(f"处理结果: {success}/{total} 成功 ({success_rate:.1f}%)", Logger.ICONS['PROGRESS'])
    
    # 图片尺寸
    if results.get('input_size') and results.get('output_size'):
        in_w, in_h = results['input_size']
        out_w, out_h = results['output_size']
        logger.info(f"原始尺寸: {in_w} x {in_h} 像素")
        logger.info(f"输出尺寸: {out_w} x {out_h} 像素")
    
    if failed > 0:
        logger.warning(f"失败数量: {failed}")
        for path, msg in results['failed_files'][:5]:  # 只显示前5个
            logger.error(f"  - {path.name}: {msg}")
        if len(results['failed_files']) > 5:
            logger.warning(f"  ... 还有 {len(results['failed_files']) - 5} 个失败")
    
    # 大小统计
    if success > 0:
        input_mb = results['total_input_bytes'] / 1024 / 1024
        output_mb = results['total_output_bytes'] / 1024 / 1024
        ratio = results['total_output_bytes'] / results['total_input_bytes'] * 100 if results['total_input_bytes'] > 0 else 0
        
        logger.info(f"输入总大小: {input_mb:.2f} MB")
        logger.info(f"输出总大小: {output_mb:.2f} MB ({ratio:.1f}%)")
        logger.info(f"节省空间: {input_mb - output_mb:.2f} MB")
    
    # 时间统计
    wall_time = results.get('wall_time', 0)
    if wall_time > 0:
        logger.info(f"总耗时: {wall_time:.2f} 秒", Logger.ICONS['TIME'])
        if success > 0:
            avg_time = results['total_time_ms'] / success
            logger.info(f"平均处理时间: {avg_time:.1f} ms/张")
            throughput = success / wall_time
            logger.info(f"处理速度: {throughput:.1f} 张/秒")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="图片批量下采样工具 - Image Batch Downsampling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件
  python downsize.py --config config.yaml
  
  # 命令行指定参数（覆盖配置文件）
  python downsize.py --config config.yaml --factor 4
  
  # 完全命令行模式
  python downsize.py --input ./images --output ./images_2x --factor 2
  
  # 指定目标尺寸
  python downsize.py --input ./images --output ./resized --target-width 1920
        """
    )
    
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                       help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--input", "-i", type=Path,
                       help="输入目录 (覆盖配置文件)")
    parser.add_argument("--output", "-o", type=Path,
                       help="输出目录 (覆盖配置文件)")
    parser.add_argument("--factor", "-f", type=int,
                       help="下采样因子 (覆盖配置文件)")
    parser.add_argument("--target-width", type=int,
                       help="目标宽度 (覆盖配置文件)")
    parser.add_argument("--target-height", type=int,
                       help="目标高度 (覆盖配置文件)")
    parser.add_argument("--interpolation", choices=['NEAREST', 'LINEAR', 'AREA', 'CUBIC', 'LANCZOS4'],
                       help="插值方法 (覆盖配置文件)")
    parser.add_argument("--workers", "-w", type=int,
                       help="并行线程数 (覆盖配置文件)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="递归处理子目录")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已存在的文件")
    parser.add_argument("--quality", "-q", type=int,
                       help="JPEG/WebP 质量 (0-100)")
    parser.add_argument("--format", choices=['auto', 'jpg', 'png', 'webp'],
                       help="输出格式")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="显示详细信息 (DEBUG 级别)")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式 (只显示错误)")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config.exists():
        config = load_config(args.config)
    else:
        if not args.input or not args.output:
            print("❌ 错误: 配置文件不存在，请使用 --input 和 --output 指定路径")
            print(f"   配置文件路径: {args.config}")
            sys.exit(1)
        config = DownsizeConfig(
            input_dir=args.input,
            output_dir=args.output,
        )
    
    # 命令行参数覆盖配置
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.factor:
        config.factor = args.factor
        config.target_width = None
        config.target_height = None
    if args.target_width:
        config.target_width = args.target_width
    if args.target_height:
        config.target_height = args.target_height
    if args.interpolation:
        config.interpolation = args.interpolation
    if args.workers:
        config.num_workers = args.workers
    if args.recursive:
        config.recursive = True
    if args.overwrite:
        config.overwrite = True
    if args.quality:
        config.jpeg_quality = args.quality
    if args.format:
        config.output_format = args.format
    if args.verbose:
        config.log_level = 'DEBUG'
    if args.quiet:
        config.log_level = 'ERROR'
        config.show_progress = False
        config.show_stats = False
    
    # 初始化日志
    logger = Logger(config.log_level)
    
    # 打印标题
    logger.header("📸 图片批量下采样工具")
    logger.info("开始处理...", Logger.ICONS['START'])
    
    # 处理图片
    results = process_images(config, logger)
    
    # 打印统计
    if config.show_stats:
        print_stats(results, logger)
    
    # 完成
    if results['failed'] == 0 and results['success'] > 0:
        logger.info("所有图片处理完成！", Logger.ICONS['SUCCESS'])
    elif results['success'] > 0:
        logger.warning(f"处理完成，但有 {results['failed']} 张图片失败")
    else:
        logger.error("没有成功处理任何图片")
        sys.exit(1)


if __name__ == "__main__":
    main()
