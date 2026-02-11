"""
COLMAP稠密重建脚本
从稀疏重建结果生成稠密点云

参考: https://github.com/colmap/colmap/tree/main/python/examples
"""

import argparse
import os
import sys
from pathlib import Path
import pycolmap
import logging
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def export_sparse_pointcloud(reconstruction: pycolmap.Reconstruction, output_path: Path) -> None:
    """
    导出稀疏重建的点云为PLY格式
    
    Args:
        reconstruction: COLMAP重建对象
        output_path: 输出PLY文件路径
    """
    import numpy as np
    
    points = []
    colors = []
    
    for point3D in reconstruction.points3D.values():
        xyz = np.array(point3D.xyz)
        if xyz.shape != (3,):
            xyz = xyz.flatten()[:3]
        points.append(xyz)
        
        color = np.array(point3D.color)
        if color.shape != (3,):
            color = color.flatten()[:3]
        colors.append(color)
    
    if len(points) == 0:
        raise ValueError("稀疏重建中没有3D点")
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    # 写入PLY文件
    with open(output_path, 'wb') as f:
        # PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        # 写入点云数据
        for i in range(len(points)):
            f.write(points[i].tobytes())
            f.write(colors[i].tobytes())
    
    logging.info(f"导出了 {len(points)} 个稀疏点")


def run_dense_reconstruction(
    workspace_path: Path,
    sparse_path: Path,
    image_path: Path,
    quality: str = "medium",
    max_image_size: int = 3200,
):
    """
    执行COLMAP稠密重建
    
    Args:
        workspace_path: 工作目录路径（包含database.db和sparse重建结果）
        sparse_path: 稀疏重建结果路径（sparse/0）
        image_path: 图像目录路径
        quality: 重建质量 ("low", "medium", "high", "extreme")
        max_image_size: 最大图像尺寸
    """
    workspace_path = Path(workspace_path)
    sparse_path = Path(sparse_path)
    image_path = Path(image_path)
    
    # 检查路径是否存在
    if not sparse_path.exists():
        raise FileNotFoundError(
            f"稀疏重建结果不存在: {sparse_path}\n"
            f"请确保已运行稀疏重建（panorama.py）并生成了 sparse/0 目录"
        )
    if not image_path.exists():
        raise FileNotFoundError(
            f"图像目录不存在: {image_path}\n"
            f"请检查图像路径是否正确"
        )
    
    # 检查稀疏重建是否有效
    try:
        reconstruction = pycolmap.Reconstruction(sparse_path)
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)
        logging.info(f"加载稀疏重建: {num_images} 张图像, {num_points} 个3D点")
        if num_images == 0:
            raise ValueError("稀疏重建中没有注册的图像！")
    except Exception as e:
        raise RuntimeError(f"无法加载稀疏重建结果: {e}")
    
    # 创建稠密重建输出目录
    dense_path = workspace_path / "dense"
    dense_path.mkdir(exist_ok=True, parents=True)
    
    # 1. 图像去畸变（Undistortion）
    logging.info("=" * 60)
    logging.info("Step 1: 图像去畸变 (Undistortion)")
    logging.info("=" * 60)
    
    undistorted_image_path = dense_path / "images"
    
    # 如果去畸变图像已存在，询问是否跳过
    if undistorted_image_path.exists() and any(undistorted_image_path.iterdir()):
        logging.warning(f"去畸变图像目录已存在: {undistorted_image_path}")
        logging.info("跳过图像去畸变步骤（如需重新处理，请删除 dense/images 目录）")
    else:
        undistorted_image_path.mkdir(exist_ok=True, parents=True)
        
        # 去畸变选项
        undistortion_options = pycolmap.UndistortCameraOptions()
        undistortion_options.max_image_size = max_image_size
        
        logging.info(f"输入图像路径: {image_path}")
        logging.info(f"稀疏重建路径: {sparse_path}")
        logging.info(f"输出路径: {undistorted_image_path}")
        
        # 执行去畸变
        try:
            pycolmap.undistort_images(
                str(undistorted_image_path),
                str(sparse_path),
                str(image_path),
                undistort_options=undistortion_options,
            )
            logging.info(f"✅ 去畸变完成，图像保存到: {undistorted_image_path}")
        except Exception as e:
            logging.error(f"❌ 去畸变失败: {e}")
            raise
    
    # 去畸变后，稀疏重建文件在 dense/images/sparse/，需要复制到 dense/sparse/
    # stereo配置文件在 dense/images/stereo/，需要复制到 dense/stereo/
    undistorted_sparse_path = undistorted_image_path / "sparse"
    dense_sparse_path = dense_path / "sparse"
    
    if undistorted_sparse_path.exists():
        if dense_sparse_path.exists():
            logging.info(f"稀疏重建文件已存在: {dense_sparse_path}，跳过复制")
        else:
            shutil.copytree(undistorted_sparse_path, dense_sparse_path)
            logging.info(f"✅ 已复制稀疏重建文件到: {dense_sparse_path}")
    else:
        logging.warning(f"未找到去畸变后的稀疏重建文件: {undistorted_sparse_path}")
    
    undistorted_stereo_path = undistorted_image_path / "stereo"
    dense_stereo_path = dense_path / "stereo"
    
    if undistorted_stereo_path.exists():
        if dense_stereo_path.exists():
            logging.info(f"stereo配置文件已存在: {dense_stereo_path}，跳过复制")
        else:
            shutil.copytree(undistorted_stereo_path, dense_stereo_path)
            logging.info(f"✅ 已复制stereo配置文件到: {dense_stereo_path}")
    else:
        logging.warning(f"未找到stereo配置文件: {undistorted_stereo_path}")
    
    # 2. 稠密重建（Patch Match MVS）
    logging.info("=" * 60)
    logging.info("Step 2: 稠密重建 (Patch Match MVS)")
    logging.info("=" * 60)
    
    # 质量设置
    quality_settings = {
        "low": {
            "max_image_size": 3200,
            "window_radius": 5,
            "window_step": 2,
            "num_iterations": 3,
            "geom_consistency": False,
        },
        "medium": {
            "max_image_size": 3200,
            "window_radius": 7,
            "window_step": 1,
            "num_iterations": 5,
            "geom_consistency": True,
        },
        "high": {
            "max_image_size": 3200,
            "window_radius": 9,
            "window_step": 1,
            "num_iterations": 7,
            "geom_consistency": True,
        },
        "extreme": {
            "max_image_size": 3200,
            "window_radius": 11,
            "window_step": 1,
            "num_iterations": 10,
            "geom_consistency": True,
        },
    }
    
    if quality not in quality_settings:
        raise ValueError(f"未知的质量设置: {quality}，可选: {list(quality_settings.keys())}")
    
    settings = quality_settings[quality]
    
    # Patch Match选项
    patch_match_options = pycolmap.PatchMatchOptions()
    patch_match_options.max_image_size = settings["max_image_size"]
    patch_match_options.window_radius = settings["window_radius"]
    patch_match_options.window_step = settings["window_step"]
    patch_match_options.num_iterations = settings["num_iterations"]
    patch_match_options.geom_consistency = settings["geom_consistency"]
    # 启用filter（必须启用才能进行点云融合）
    patch_match_options.filter = True
    # Filter参数设置（放宽一些限制以获得更多点）
    patch_match_options.filter_min_ncc = 0.1
    patch_match_options.filter_min_num_consistent = 2
    patch_match_options.filter_min_triangulation_angle = 1.0
    patch_match_options.filter_geom_consistency_max_cost = 2.0
    # 设置GPU索引（字符串类型，0表示使用第一个GPU）
    patch_match_options.gpu_index = "0"
    
    # 检查是否已有深度图
    depth_maps_path = dense_path / "stereo" / "depth_maps"
    if depth_maps_path.exists() and any(depth_maps_path.glob("*.geometric.bin")):
        logging.info(f"深度图已存在: {depth_maps_path}")
        logging.info("跳过Patch Match步骤（如需重新计算，请删除 dense/stereo/depth_maps 目录）")
    else:
        # 执行Patch Match
        logging.info(f"工作目录: {dense_path}")
        logging.info(f"质量设置: {quality}")
        logging.info(f"窗口半径: {patch_match_options.window_radius}")
        logging.info(f"迭代次数: {patch_match_options.num_iterations}")
        logging.info(f"几何一致性: {patch_match_options.geom_consistency}")
        
        try:
            pycolmap.patch_match_stereo(
                str(dense_path),
                options=patch_match_options,
            )
            logging.info(f"✅ Patch Match完成")
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg.lower() or "compiled with" in error_msg.lower():
                logging.error("=" * 60)
                logging.error("❌ 错误：当前pycolmap安装不支持CUDA")
                logging.error("=" * 60)
                logging.error("COLMAP的PatchMatch算法需要CUDA支持才能运行。")
                logging.error("")
                logging.error("解决方案有以下几种：")
                logging.error("")
                logging.error("方案1：从源码编译支持CUDA的COLMAP（推荐）")
                logging.error("  # 安装CUDA工具包（如果还没有）")
                logging.error("  # 然后编译COLMAP:")
                logging.error("  git clone https://github.com/colmap/colmap.git")
                logging.error("  cd colmap")
                logging.error("  mkdir build && cd build")
                logging.error("  cmake .. -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=native")
                logging.error("  make -j$(nproc)")
                logging.error("  cd ../scripts/python")
                logging.error("  pip install -e .")
                logging.error("")
                logging.error("方案2：使用Docker镜像（如果可用）")
                logging.error("  docker pull colmap/colmap:latest")
                logging.error("")
                logging.error("方案3：导出稀疏点云为PLY格式（当前可用）")
                logging.error("  将尝试导出稀疏点云...")
                logging.error("")
                logging.error("注意：如果没有NVIDIA GPU，无法进行稠密重建。")
                logging.error("=" * 60)
                
                # 尝试导出稀疏点云
                try:
                    sparse_ply_path = dense_path / "sparse_points.ply"
                    # 重新加载重建结果以导出点云
                    sparse_recon = pycolmap.Reconstruction(sparse_path)
                    export_sparse_pointcloud(sparse_recon, sparse_ply_path)
                    logging.info(f"✅ 已导出稀疏点云到: {sparse_ply_path}")
                    logging.info("   可以使用visualizer.py或MeshLab等工具查看")
                except Exception as export_error:
                    logging.warning(f"⚠️ 导出稀疏点云失败: {export_error}")
                
                raise RuntimeError(
                    "稠密重建需要CUDA支持。请按照上述方案安装支持CUDA的COLMAP，"
                    "或使用稀疏点云进行可视化。"
                )
            raise
        except Exception as e:
            logging.error(f"❌ Patch Match失败: {e}")
            raise
    
    # 3. 融合点云（Stereo Fusion）
    logging.info("=" * 60)
    logging.info("Step 3: 融合点云 (Stereo Fusion)")
    logging.info("=" * 60)
    
    # 融合选项
    fusion_options = pycolmap.StereoFusionOptions()
    fusion_options.max_image_size = settings["max_image_size"]
    fusion_options.min_num_pixels = 5
    fusion_options.max_num_pixels = 10000
    fusion_options.max_traversal_depth = 100
    fusion_options.max_reproj_error = 2.0
    fusion_options.max_depth_error = 0.01
    fusion_options.max_normal_error = 0.1
    fusion_options.check_num_images = 50
    fusion_options.cache_size = 32
    fusion_options.num_threads = min(8, os.cpu_count() or 4)
    
    # 输出点云路径
    fused_ply_path = dense_path / "fused.ply"
    
    # 如果点云已存在，询问是否跳过
    if fused_ply_path.exists() and fused_ply_path.stat().st_size > 1000:
        logging.info(f"点云文件已存在: {fused_ply_path}")
        logging.info("跳过点云融合步骤（如需重新生成，请删除该文件）")
    else:
        fused_ply_path.unlink(missing_ok=True)  # 删除旧文件
        
        logging.info(f"融合参数:")
        logging.info(f"  - 最大图像尺寸: {fusion_options.max_image_size}")
        logging.info(f"  - 最小像素数: {fusion_options.min_num_pixels}")
        logging.info(f"  - 最大像素数: {fusion_options.max_num_pixels}")
        logging.info(f"  - 最大重投影误差: {fusion_options.max_reproj_error}")
        
        # 执行融合
        try:
            pycolmap.stereo_fusion(
                str(fused_ply_path),
                str(dense_path),
                options=fusion_options,
            )
            if fused_ply_path.exists():
                file_size_mb = fused_ply_path.stat().st_size / (1024 * 1024)
                logging.info(f"✅ 点云融合完成，保存到: {fused_ply_path} ({file_size_mb:.2f} MB)")
            else:
                logging.warning("⚠️ 点云文件未生成，可能没有足够的匹配点")
        except Exception as e:
            logging.error(f"❌ 点云融合失败: {e}")
            raise
    
    # 4. 可选：生成泊松重建（Poisson Reconstruction）
    poisson_ply_path = dense_path / "poisson.ply"
    # 只有当点云文件存在且不为空时才进行泊松重建
    if fused_ply_path.exists() and fused_ply_path.stat().st_size > 1000:  # 至少1KB
        logging.info("=" * 60)
        logging.info("Step 4: 泊松重建 (Poisson Reconstruction)")
        logging.info("=" * 60)
        
        poisson_ply_path.unlink(missing_ok=True)
        
        poisson_options = pycolmap.PoissonMeshingOptions()
        poisson_options.trim = 10
        poisson_options.point_weight = 1.0
        poisson_options.depth = 9
        poisson_options.color = 32
        poisson_options.num_threads = min(8, os.cpu_count() or 4)
        
        try:
            pycolmap.poisson_meshing(
                str(fused_ply_path),
                str(poisson_ply_path),
                options=poisson_options,
            )
            logging.info(f"✅ 泊松重建完成，保存到: {poisson_ply_path}")
        except Exception as e:
            logging.warning(f"⚠️ 泊松重建失败: {e}")
            logging.info("   这通常不影响主要结果，可以忽略")
    else:
        logging.warning("⚠️ 跳过泊松重建：点云文件为空或不存在")
    
    logging.info("=" * 60)
    logging.info("🎉 稠密重建完成！")
    logging.info("=" * 60)
    logging.info(f"输出文件:")
    logging.info(f"  - 稠密点云: {fused_ply_path}")
    if poisson_ply_path.exists():
        logging.info(f"  - 泊松网格: {poisson_ply_path}")
    logging.info(f"  - 去畸变图像: {undistorted_image_path}")
    logging.info(f"  - 深度图: {dense_path / 'stereo' / 'depth_maps'}")


def main():
    parser = argparse.ArgumentParser(
        description="COLMAP稠密重建：从稀疏重建结果生成稠密点云\n"
                    "参考: https://github.com/colmap/colmap/tree/main/python/examples",
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
        "--skip_undistortion",
        action="store_true",
        help="跳过图像去畸变步骤（如果已存在去畸变图像）",
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
    parser.add_argument(
        "--skip_poisson",
        action="store_true",
        help="跳过泊松重建步骤",
    )
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.sparse_path is None:
        args.sparse_path = args.workspace_path / "sparse" / "0"
    if args.image_path is None:
        args.image_path = args.workspace_path / "images"
    
    # 验证路径
    workspace_path = Path(args.workspace_path)
    if not workspace_path.exists():
        logging.error(f"工作目录不存在: {workspace_path}")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("COLMAP 稠密重建")
    logging.info("=" * 60)
    logging.info(f"工作目录: {workspace_path}")
    logging.info(f"稀疏重建: {args.sparse_path}")
    logging.info(f"图像目录: {args.image_path}")
    logging.info(f"质量设置: {args.quality}")
    logging.info("=" * 60)
    
    try:
        run_dense_reconstruction(
            workspace_path=workspace_path,
            sparse_path=args.sparse_path,
            image_path=args.image_path,
            quality=args.quality,
            max_image_size=args.max_image_size,
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

