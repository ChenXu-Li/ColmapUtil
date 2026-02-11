#!/usr/bin/env python3
"""
普通图像（针孔相机）重建脚本
适用于标准透视相机图像，不使用Rig配置
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import yaml

import pycolmap
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_config(config_path: Path):
    """加载配置文件"""
    if not config_path.exists():
        logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_environment(config: dict):
    """设置环境变量"""
    env_vars = config.get('environment', {})
    for key, value in env_vars.items():
        os.environ[key] = str(value)


def run_sparse_reconstruction(
    input_image_path: Path,
    output_path: Path,
    config: dict,
    matcher: str = "exhaustive"
):
    """
    执行稀疏重建
    
    Args:
        input_image_path: 输入图像目录
        output_path: 输出路径
        config: 配置字典
        matcher: 匹配器类型
    """
    logging.info("=" * 60)
    logging.info("开始稀疏重建（普通图像）")
    logging.info("=" * 60)
    logging.info(f"输入图像: {input_image_path}")
    logging.info(f"输出路径: {output_path}")
    logging.info(f"匹配器: {matcher}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir = output_path / "images"
    database_path = output_path / "database.db"
    rec_path = output_path / "sparse"
    
    # 复制图像到输出目录（COLMAP标准格式）
    # 检查是否需要复制图像
    need_copy = False
    
    if not image_dir.exists():
        need_copy = True
        logging.info("图像目录不存在，需要复制图像")
    else:
        # 获取输入和输出目录中的图像文件
        input_images = {
            f.name: f 
            for f in input_image_path.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()
        }
        output_images = {
            f.name: f
            for f in image_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()
        }
        
        # 检查图像文件是否一致
        if len(input_images) == 0:
            logging.warning("输入目录中没有找到图像文件")
            need_copy = False
        elif len(input_images) != len(output_images):
            need_copy = True
            logging.info(f"图像数量不匹配（输入: {len(input_images)}, 输出: {len(output_images)}），需要重新复制")
        else:
            # 检查文件名是否一致，以及文件是否需要更新
            missing_files = set(input_images.keys()) - set(output_images.keys())
            if missing_files:
                need_copy = True
                logging.info(f"发现缺失的图像文件: {len(missing_files)} 个，需要重新复制")
            else:
                # 检查文件修改时间，如果输入文件更新则重新复制
                import shutil
                files_need_update = []
                for img_name in input_images.keys():
                    input_file = input_images[img_name]
                    output_file = output_images[img_name]
                    # 如果输入文件比输出文件新，需要更新
                    if input_file.stat().st_mtime > output_file.stat().st_mtime:
                        files_need_update.append(img_name)
                
                if files_need_update:
                    need_copy = True
                    logging.info(f"发现 {len(files_need_update)} 个图像文件需要更新，需要重新复制")
                else:
                    logging.info(f"图像目录已存在且图像文件一致（{len(input_images)} 个文件），跳过复制")
    
    if need_copy:
        logging.info("复制图像到输出目录...")
        import shutil
        image_dir.mkdir(parents=True, exist_ok=True)
        copied_count = 0
        for img_file in input_image_path.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png'] and img_file.is_file():
                shutil.copy2(img_file, image_dir / img_file.name)
                copied_count += 1
        logging.info(f"已复制 {copied_count} 个图像文件")
    
    # 删除旧数据库
    if database_path.exists():
        database_path.unlink()
    
    rec_path.mkdir(parents=True, exist_ok=True)
    
    # 特征提取配置
    feat_config = config.get('feature_extraction', {})
    feature_opts = pycolmap.FeatureExtractionOptions()
    feature_opts.use_gpu = feat_config.get('use_gpu', True)
    feature_opts.gpu_index = feat_config.get('gpu_index', '0')
    feature_opts.num_threads = feat_config.get('num_threads', 4)
    feature_opts.max_image_size = feat_config.get('max_image_size', 3200)
    # max_num_features 在 sift 选项中
    feature_opts.sift.max_num_features = feat_config.get('max_num_features', 8192)
    
    logging.info("提取特征...")
    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        camera_mode=pycolmap.CameraMode.AUTO,
        extraction_options=feature_opts,
    )
    logging.info("特征提取完成")
    
    # 特征匹配配置
    match_config = config.get('feature_matching', {})
    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.use_gpu = match_config.get('use_gpu', True)
    matching_opts.gpu_index = match_config.get('gpu_index', '0')
    matching_opts.num_threads = match_config.get('num_threads', 4)
    
    logging.info(f"特征匹配（{matcher}）...")
    if matcher == "sequential":
        seq_config = match_config.get('sequential', {})
        pairing_opts = pycolmap.SequentialPairingOptions()
        pairing_opts.overlap = seq_config.get('overlap', 10)
        pairing_opts.loop_detection = seq_config.get('loop_detection', True)
        pairing_opts.loop_detection_period = seq_config.get('loop_detection_period', 10)
        pycolmap.match_sequential(
            str(database_path),
            pairing_options=pairing_opts,
            matching_options=matching_opts,
        )
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(str(database_path), matching_options=matching_opts)
    elif matcher == "spatial":
        spatial_config = match_config.get('spatial', {})
        pairing_opts = pycolmap.SpatialPairingOptions()
        pairing_opts.is_gps = spatial_config.get('is_gps', False)
        pairing_opts.ignore_z = spatial_config.get('ignore_z', False)
        pairing_opts.max_num_neighbors = spatial_config.get('max_num_neighbors', 50)
        pairing_opts.max_distance = spatial_config.get('max_distance', 100.0)
        pycolmap.match_spatial(str(database_path), matching_options=matching_opts)
    elif matcher == "vocabtree":
        pycolmap.match_vocabtree(str(database_path), matching_options=matching_opts)
    else:
        raise ValueError(f"未知的匹配器类型: {matcher}")
    
    logging.info("特征匹配完成")
    
    # 稀疏重建配置
    sparse_config = config.get('sparse_reconstruction', {})
    mapper_opts = pycolmap.IncrementalMapperOptions()
    mapper_opts.init_min_num_inliers = sparse_config.get('init_min_num_inliers', 100)
    mapper_opts.init_max_error = sparse_config.get('init_max_error', 4.0)
    mapper_opts.init_min_tri_angle = sparse_config.get('init_min_tri_angle', 16.0)
    mapper_opts.init_max_reg_trials = sparse_config.get('init_max_reg_trials', 2)
    
    mapper_opts.abs_pose_min_num_inliers = sparse_config.get('abs_pose_min_num_inliers', 30)
    mapper_opts.abs_pose_min_inlier_ratio = sparse_config.get('abs_pose_min_inlier_ratio', 0.25)
    mapper_opts.abs_pose_max_error = sparse_config.get('abs_pose_max_error', 4.0)
    
    mapper_opts.filter_min_tri_angle = sparse_config.get('filter_min_tri_angle', 1.5)
    mapper_opts.filter_max_reproj_error = sparse_config.get('filter_max_reproj_error', 4.0)
    
    ba_config = sparse_config
    pipeline_opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_focal_length=ba_config.get('ba_refine_focal_length', True),
        ba_refine_principal_point=ba_config.get('ba_refine_principal_point', False),
        ba_refine_extra_params=ba_config.get('ba_refine_extra_params', True),
        mapper=mapper_opts,
    )
    
    logging.info("开始增量重建...")
    recs = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(rec_path),
        pipeline_opts,
    )
    
    logging.info(f"重建完成，生成了 {len(recs)} 个模型")
    for idx, rec in recs.items():
        try:
            summary = rec.summary()
            logging.info(f"模型 #{idx}: {summary}")
        except Exception as e:
            logging.warning(f"模型 #{idx} 摘要获取失败: {e}")
            logging.info(f"模型 #{idx}: {len(rec.images)} 图像, {len(rec.points3D)} 点")


def main():
    parser = argparse.ArgumentParser(
        description="普通图像（针孔相机）重建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_image_path",
        type=Path,
        default=None,
        help="输入图像目录路径（如果未指定，将从配置文件读取）",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="输出路径（如果未指定，将从配置文件读取）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="配置文件路径",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "sequential", "spatial", "vocabtree"],
        help="匹配器类型",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="仅执行稀疏重建",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
        config_path = args.config
    else:
        # 使用默认配置
        default_config_path = Path(__file__).parent.parent.parent / "reconstruction" / "pinhole" / "configs" / "config_pinhole.yaml"
        config = load_config(default_config_path)
        config_path = default_config_path
    
    # 设置环境变量
    setup_environment(config)
    
    # 从配置文件读取路径（如果命令行未指定）
    paths_config = config.get('paths', {})
    input_image_path = args.input_image_path
    output_path = args.output_path
    
    if input_image_path is None:
        config_input = paths_config.get('input_images', '')
        if config_input:
            input_image_path = Path(config_input)
            logging.info(f"从配置文件读取输入路径: {input_image_path}")
        else:
            logging.error("未指定输入图像目录，请在命令行使用 --input_image_path 或在配置文件的 paths.input_images 中指定")
            sys.exit(1)
    
    if output_path is None:
        config_output = paths_config.get('output_path', '')
        if config_output:
            output_path = Path(config_output)
            logging.info(f"从配置文件读取输出路径: {output_path}")
        else:
            logging.error("未指定输出路径，请在命令行使用 --output_path 或在配置文件的 paths.output_path 中指定")
            sys.exit(1)
    
    # 执行稀疏重建
    run_sparse_reconstruction(
        input_image_path,
        output_path,
        config,
        args.matcher,
    )
    
    # 如果不需要仅稀疏重建，继续执行稠密重建
    if not args.sparse_only:
        logging.info("=" * 60)
        logging.info("开始稠密重建...")
        logging.info("=" * 60)
        
        # 从配置文件读取稠密重建参数
        dense_config = config.get('dense_reconstruction', {})
        quality = dense_config.get('patch_match', {}).get('quality', 'medium')
        max_image_size = dense_config.get('undistortion', {}).get('max_image_size', 3200)
        if not max_image_size:
            max_image_size = dense_config.get('patch_match', {}).get('max_image_size', 3200)
        
        # 导入稠密重建脚本（使用命令行工具版本）
        dense_script = Path(__file__).parent.parent.parent / "scripts" / "reconstruction" / "dense_reconstruction_cli.py"
        if dense_script.exists():
            import subprocess
            dense_cmd = [
                sys.executable,
                str(dense_script),
                "--workspace_path", str(output_path),
                "--quality", quality,
                "--max_image_size", str(max_image_size),
            ]
            logging.info(f"执行命令: {' '.join(dense_cmd)}")
            subprocess.run(dense_cmd, check=False)
        else:
            logging.warning("稠密重建脚本不存在，跳过稠密重建")


if __name__ == "__main__":
    main()
