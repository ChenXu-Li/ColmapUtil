#!/usr/bin/env python3
"""
COLMAP稠密重建脚本（使用命令行工具）
从稀疏重建结果生成稠密点云
参考: denseline.sh
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def find_colmap():
    """查找COLMAP可执行文件"""
    # 常见的COLMAP安装路径
    possible_paths = [
        'colmap',  # 在PATH中
        '/root/miniconda3/bin/colmap',
        '/usr/local/bin/colmap',
        '/usr/bin/colmap',
        '/opt/colmap/bin/colmap',
        os.path.expanduser('~/miniconda3/bin/colmap'),
        os.path.expanduser('~/anaconda3/bin/colmap'),
    ]
    
    for colmap_path in possible_paths:
        try:
            # 对于特定路径，先检查文件是否存在
            if colmap_path != 'colmap':
                if not os.path.exists(colmap_path) or not os.access(colmap_path, os.X_OK):
                    continue
            
            # 尝试运行 help 命令来验证 COLMAP 是否可用
            test_cmd = [colmap_path, 'help'] if colmap_path != 'colmap' else ['colmap', 'help']
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                env=os.environ.copy()
            )
            
            # COLMAP 使用 'help' 命令，如果能执行说明找到了
            # 尝试从输出中提取版本信息
            version_info = result.stdout.strip().split('\n')[0] if result.stdout else "未知版本"
            logging.info(f"找到COLMAP: {colmap_path}")
            if "COLMAP" in version_info:
                logging.info(f"COLMAP信息: {version_info}")
            return colmap_path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    
    return None


def check_colmap_installed():
    """检查COLMAP是否已安装"""
    colmap_path = find_colmap()
    if colmap_path:
        return colmap_path
    else:
        logging.error("COLMAP未找到，请确保已安装并加入PATH")
        logging.error("常见安装位置:")
        logging.error("  - /root/miniconda3/bin/colmap")
        logging.error("  - /usr/local/bin/colmap")
        logging.error("  - /usr/bin/colmap")
        logging.error("安装方法: https://colmap.github.io/install.html")
        return None


def check_cuda():
    """检查CUDA是否可用"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            logging.info(f"检测到NVIDIA GPU: {gpu_name}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("未检测到nvidia-smi，将以CPU模式运行（非常慢）")
    return False


def run_dense_reconstruction_cli(
    workspace_path: Path,
    sparse_path: Path = None,
    image_path: Path = None,
    quality: str = "medium",
    max_image_size: int = 3200,
    gpu_index: int = 0,
    skip_undistortion: bool = False,
    skip_patch_match: bool = False,
    skip_fusion: bool = False,
    colmap_path: str = "colmap",
):
    """
    使用COLMAP命令行工具执行稠密重建
    
    Args:
        workspace_path: 工作目录路径（包含database.db和sparse重建结果）
        sparse_path: 稀疏重建结果路径（sparse/0），默认: workspace_path/sparse/0
        image_path: 图像目录路径，默认: workspace_path/images
        quality: 重建质量 ("low", "medium", "high", "extreme")
        max_image_size: 最大图像尺寸
        gpu_index: GPU索引
        skip_undistortion: 跳过图像去畸变步骤
        skip_patch_match: 跳过Patch Match步骤
        skip_fusion: 跳过点云融合步骤
    """
    workspace_path = Path(workspace_path)
    
    # 设置默认路径
    if sparse_path is None:
        sparse_path = workspace_path / "sparse" / "0"
    else:
        sparse_path = Path(sparse_path)
    
    if image_path is None:
        image_path = workspace_path / "images"
    else:
        image_path = Path(image_path)
    
    # 检查路径
    if not sparse_path.exists():
        raise FileNotFoundError(f"稀疏重建结果不存在: {sparse_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_path}")
    
    # 检查必要的稀疏文件
    for file in ["cameras.bin", "images.bin", "points3D.bin"]:
        if not (sparse_path / file).exists():
            raise FileNotFoundError(f"稀疏重建文件不完整: 缺少 {file}")
    
    # 创建输出目录结构
    stereo_path = workspace_path / "stereo"
    depth_maps_path = stereo_path / "depth_maps"
    normal_maps_path = stereo_path / "normal_maps"
    consistency_graphs_path = stereo_path / "consistency_graphs"
    fused_ply = workspace_path / "fused.ply"
    
    # 质量设置
    quality_settings = {
        "low": {
            "num_samples": 10,
            "window_radius": 5,
            "window_step": 2,
            "num_iterations": 3,
            "geom_consistency": "false",
            "filter_min_ncc": 0.1,
        },
        "medium": {
            "num_samples": 15,
            "window_radius": 7,
            "window_step": 1,
            "num_iterations": 5,
            "geom_consistency": "true",
            "filter_min_ncc": 0.1,
        },
        "high": {
            "num_samples": 20,
            "window_radius": 9,
            "window_step": 1,
            "num_iterations": 7,
            "geom_consistency": "true",
            "filter_min_ncc": 0.1,
        },
        "extreme": {
            "num_samples": 25,
            "window_radius": 11,
            "window_step": 1,
            "num_iterations": 10,
            "geom_consistency": "true",
            "filter_min_ncc": 0.1,
        },
    }
    
    if quality not in quality_settings:
        raise ValueError(f"未知的质量设置: {quality}，可选: {list(quality_settings.keys())}")
    
    qs = quality_settings[quality]
    
    # 检查工作空间（用于后续步骤）
    workspace_dense = workspace_path / ".dense_workspace"
    undistorted_images_path = workspace_dense / "images" if workspace_dense.exists() else None
    
    # 步骤1: 图像去畸变
    if not skip_undistortion:
        logging.info("=" * 60)
        logging.info("Step 1: 图像去畸变 (Undistortion)")
        logging.info("=" * 60)
        
        # 检查去畸变是否已完成
        # 需要检查：1) 配置文件存在 2) 去畸变图像目录存在 3) 图像数量匹配
        stereo_config = stereo_path / "patch-match.cfg"
        fusion_config = stereo_path / "fusion.cfg"
        
        # 检查去畸变是否完整
        undistortion_complete = False
        if stereo_config.exists() and fusion_config.exists():
            # 检查去畸变图像目录
            if undistorted_images_path and undistorted_images_path.exists():
                # 统计输入图像和去畸变图像数量
                input_image_count = len([f for f in image_path.glob("*") 
                                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()])
                undistorted_image_count = len([f for f in undistorted_images_path.glob("*")
                                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()])
                
                if input_image_count > 0 and undistorted_image_count == input_image_count:
                    # 检查稀疏重建文件是否匹配
                    dense_sparse_path = workspace_dense / "sparse" / "0"
                    if dense_sparse_path.exists():
                        # 检查关键文件是否存在
                        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
                        if all((dense_sparse_path / f).exists() for f in required_files):
                            # 比较稀疏重建的时间戳（如果输入稀疏重建更新，需要重新去畸变）
                            input_sparse_time = max((sparse_path / f).stat().st_mtime for f in required_files 
                                                   if (sparse_path / f).exists())
                            dense_sparse_time = max((dense_sparse_path / f).stat().st_mtime for f in required_files)
                            
                            if dense_sparse_time >= input_sparse_time:
                                undistortion_complete = True
                                logging.info(f"✅ 去畸变已完成且完整")
                                logging.info(f"   配置文件: {stereo_config}")
                                logging.info(f"   去畸变图像: {undistorted_image_count} 个（匹配输入图像数量）")
                            else:
                                logging.warning(f"⚠️  输入稀疏重建已更新，需要重新去畸变")
                                logging.info(f"   输入稀疏重建时间: {input_sparse_time}")
                                logging.info(f"   去畸变稀疏重建时间: {dense_sparse_time}")
                        else:
                            logging.warning(f"⚠️  去畸变稀疏重建文件不完整，需要重新去畸变")
                    else:
                        logging.warning(f"⚠️  去畸变稀疏重建目录不存在，需要重新去畸变")
                else:
                    logging.warning(f"⚠️  去畸变图像数量不匹配（输入: {input_image_count}, 去畸变: {undistorted_image_count}），需要重新去畸变")
            else:
                logging.warning(f"⚠️  去畸变图像目录不存在，需要重新去畸变")
        else:
            logging.info("配置文件不存在，需要执行去畸变")
        
        if undistortion_complete:
            logging.info("跳过图像去畸变步骤")
        else:
            # 创建临时dense目录（COLMAP image_undistorter会创建dense目录）
            temp_dense_path = workspace_path / ".dense_temp"
            if temp_dense_path.exists():
                shutil.rmtree(temp_dense_path)
            
            logging.info(f"输入图像: {image_path}")
            logging.info(f"输入模型: {sparse_path}")
            logging.info(f"输出路径: {temp_dense_path}")
            
            # 运行 image_undistorter
            cmd = [
                colmap_path, "image_undistorter",
                "--image_path", str(image_path),
                "--input_path", str(sparse_path),
                "--output_path", str(temp_dense_path),
                "--output_type", "COLMAP",
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            
            # 迁移stereo目录到标准格式
            if (temp_dense_path / "stereo").exists():
                if stereo_path.exists():
                    shutil.rmtree(stereo_path)
                shutil.move(str(temp_dense_path / "stereo"), str(stereo_path))
                logging.info(f"✅ 已创建stereo目录: {stereo_path}")
            
            # 保留临时dense目录中的images和sparse，供后续步骤使用
            workspace_dense = workspace_path / ".dense_workspace"
            if workspace_dense.exists():
                shutil.rmtree(workspace_dense)
            if (temp_dense_path / "images").exists() and (temp_dense_path / "sparse").exists():
                shutil.move(str(temp_dense_path), str(workspace_dense))
                logging.info(f"✅ 已创建工作空间: {workspace_dense}")
            else:
                logging.warning("临时dense目录结构不完整")
            
            logging.info("✅ 图像去畸变完成")
    else:
        logging.info("跳过图像去畸变步骤")
    
    # 检查工作空间（如果去畸变已完成，应该存在）
    if not workspace_dense.exists():
        raise FileNotFoundError(f"工作空间目录不存在: {workspace_dense}，请先运行图像去畸变")
    
    # 确保stereo链接存在
    workspace_stereo = workspace_dense / "stereo"
    if not workspace_stereo.exists():
        if workspace_stereo.is_symlink():
            os.remove(workspace_stereo)
        os.symlink(str(stereo_path.absolute()), str(workspace_stereo))
        logging.info(f"✅ 已创建stereo符号链接")
    
    # 步骤2: 深度图估计（Patch Match）
    if not skip_patch_match:
        logging.info("=" * 60)
        logging.info("Step 2: 深度图估计 (Patch Match Stereo)")
        logging.info("=" * 60)
        
        # 检查深度图是否完整
        # 需要检查：1) 深度图数量是否匹配去畸变图像数量 2) 深度图文件是否完整
        patch_match_complete = False
        
        if depth_maps_path.exists():
            # 统计深度图文件
            geometric_depth_files = list(depth_maps_path.glob("*.geometric.bin"))
            photometric_depth_files = list(depth_maps_path.glob("*.photometric.bin"))
            num_depth_files = len(geometric_depth_files) + len(photometric_depth_files)
            
            # 获取去畸变图像数量（应该等于深度图数量）
            if undistorted_images_path and undistorted_images_path.exists():
                expected_depth_count = len([f for f in undistorted_images_path.glob("*")
                                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()])
                
                if num_depth_files > 0:
                    # 检查深度图数量是否合理（至少应该有一定数量）
                    if num_depth_files >= expected_depth_count * 0.8:  # 允许20%的容差
                        # 检查深度图文件大小（避免损坏的文件）
                        valid_depth_files = 0
                        for depth_file in geometric_depth_files + photometric_depth_files:
                            if depth_file.stat().st_size > 1024:  # 至少1KB
                                valid_depth_files += 1
                        
                        if valid_depth_files >= num_depth_files * 0.9:  # 90%的文件有效
                            patch_match_complete = True
                            logging.info(f"✅ Patch Match已完成且完整")
                            logging.info(f"   深度图数量: {num_depth_files} (期望: {expected_depth_count})")
                            logging.info(f"   有效深度图: {valid_depth_files}")
                        else:
                            logging.warning(f"⚠️  深度图文件可能损坏（有效: {valid_depth_files}/{num_depth_files}），需要重新计算")
                    else:
                        logging.warning(f"⚠️  深度图数量不足（找到: {num_depth_files}, 期望: {expected_depth_count}），需要重新计算")
                else:
                    logging.info("深度图目录存在但为空，需要计算深度图")
            else:
                logging.warning(f"⚠️  无法确定期望的深度图数量（去畸变图像目录不存在），需要重新计算")
        else:
            logging.info("深度图目录不存在，需要计算深度图")
        
        if patch_match_complete:
            logging.info("跳过Patch Match步骤")
        else:
            # 确保depth_maps目录存在
            depth_maps_path.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"工作空间: {workspace_dense}")
            logging.info(f"质量设置: {quality}")
            logging.info(f"GPU索引: {gpu_index}")
            logging.info(f"几何一致性: {qs['geom_consistency']}")
            logging.info(f"采样数: {qs['num_samples']}")
            
            # 运行 patch_match_stereo
            cmd = [
                colmap_path, "patch_match_stereo",
                "--workspace_path", str(workspace_dense),
                "--PatchMatchStereo.gpu_index", str(gpu_index),
                "--PatchMatchStereo.geom_consistency", qs["geom_consistency"],
                "--PatchMatchStereo.num_samples", str(qs["num_samples"]),
                "--PatchMatchStereo.filter_min_ncc", str(qs["filter_min_ncc"]),
                "--PatchMatchStereo.filter", "1",
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            logging.info("注意: 此步骤可能需要较长时间，请耐心等待...")
            
            result = subprocess.run(cmd, check=True)
            
            # 验证深度图
            num_depth_files_after = len(list(depth_maps_path.glob("*.geometric.bin"))) + \
                                   len(list(depth_maps_path.glob("*.photometric.bin")))
            
            if num_depth_files_after == 0:
                raise RuntimeError("深度图估计完成，但未生成任何深度图文件")
            
            logging.info(f"✅ 深度图估计完成，生成 {num_depth_files_after} 个深度图")
    else:
        logging.info("跳过Patch Match步骤")
    
    # 步骤3: 点云融合
    if not skip_fusion:
        logging.info("=" * 60)
        logging.info("Step 3: 点云融合 (Stereo Fusion)")
        logging.info("=" * 60)
        
        if fused_ply.exists():
            file_size = fused_ply.stat().st_size / (1024 * 1024)  # MB
            logging.warning(f"点云文件已存在: {fused_ply} (大小: {file_size:.2f} MB)")
            response = input("是否重新生成？(y/n): ").strip().lower()
            if response != 'y':
                logging.info("使用现有点云文件")
                return
        
        logging.info(f"工作空间: {workspace_dense}")
        logging.info(f"输出文件: {fused_ply}")
        
        # 运行 stereo_fusion
        cmd = [
            colmap_path, "stereo_fusion",
            "--workspace_path", str(workspace_dense),
            "--output_path", str(fused_ply),
            "--StereoFusion.check_num_images", "20",
        ]
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True)
        
        if fused_ply.exists():
            file_size = fused_ply.stat().st_size / (1024 * 1024)  # MB
            logging.info(f"✅ 点云融合完成")
            logging.info(f"输出文件: {fused_ply}")
            logging.info(f"文件大小: {file_size:.2f} MB")
        else:
            raise RuntimeError("点云文件未生成")
    else:
        logging.info("跳过头云融合步骤")
    
    logging.info("=" * 60)
    logging.info("✅ 稠密重建完成")
    logging.info("=" * 60)
    logging.info(f"最终点云: {fused_ply}")
    logging.info(f"可以使用以下工具查看点云:")
    logging.info(f"  - Meshlab: meshlab {fused_ply}")
    logging.info(f"  - CloudCompare: cloudcompare {fused_ply}")
    logging.info(f"  - Python脚本: ./visualize_ply.sh {fused_ply}")


def main():
    parser = argparse.ArgumentParser(
        description="COLMAP稠密重建（使用命令行工具）：从稀疏重建结果生成稠密点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace_path",
        type=Path,
        required=True,
        help="工作目录路径（包含database.db和sparse重建结果）",
    )
    parser.add_argument(
        "--sparse_path",
        type=Path,
        help="稀疏重建结果路径（sparse/0），默认: workspace_path/sparse/0",
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        help="图像目录路径，默认: workspace_path/images",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "extreme"],
        help="重建质量 (default: medium)",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=3200,
        help="最大图像尺寸 (default: 3200)",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU索引 (default: 0)",
    )
    parser.add_argument(
        "--skip_undistortion",
        action="store_true",
        help="跳过图像去畸变步骤（如果已存在配置文件）",
    )
    parser.add_argument(
        "--skip_patch_match",
        action="store_true",
        help="跳过Patch Match步骤（如果已存在深度图）",
    )
    parser.add_argument(
        "--skip_fusion",
        action="store_true",
        help="跳过点云融合步骤（如果已存在点云文件）",
    )
    
    args = parser.parse_args()
    
    # 检查COLMAP是否安装
    colmap_path = check_colmap_installed()
    if not colmap_path:
        sys.exit(1)
    
    # 检查CUDA（可选）
    check_cuda()
    
    # 验证路径
    workspace_path = Path(args.workspace_path)
    if not workspace_path.exists():
        logging.error(f"工作目录不存在: {workspace_path}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("COLMAP 稠密重建（命令行工具）")
    logging.info("=" * 60)
    logging.info(f"工作目录: {workspace_path}")
    logging.info(f"稀疏重建: {args.sparse_path or workspace_path / 'sparse' / '0'}")
    logging.info(f"图像目录: {args.image_path or workspace_path / 'images'}")
    logging.info(f"质量设置: {args.quality}")
    logging.info(f"GPU索引: {args.gpu_index}")
    logging.info("=" * 60)
    
    try:
        run_dense_reconstruction_cli(
            workspace_path=workspace_path,
            sparse_path=args.sparse_path,
            image_path=args.image_path,
            quality=args.quality,
            max_image_size=args.max_image_size,
            gpu_index=args.gpu_index,
            skip_undistortion=args.skip_undistortion,
            skip_patch_match=args.skip_patch_match,
            skip_fusion=args.skip_fusion,
            colmap_path=colmap_path,
        )
    except KeyboardInterrupt:
        logging.info("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ 稠密重建失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
