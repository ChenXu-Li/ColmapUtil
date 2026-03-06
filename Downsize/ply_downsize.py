"""
PLY 文件体素化下采样脚本
对单个PLY文件进行体素化下采样和随机下采样
参考 merge_and_downsample_ply.py
"""

import os
import numpy as np
import yaml
import argparse
from pathlib import Path
import trimesh

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[Warning] Open3D not available. Voxel downsampling will use fallback method.")


def load_ply_file(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    加载PLY文件
    
    Args:
        ply_path: PLY文件路径
    
    Returns:
        points: (N, 3) numpy array，点云坐标
        colors: (N, 3) numpy array，点云颜色 (RGB, 0-255)
    """
    try:
        # 使用 trimesh 加载
        mesh = trimesh.load(ply_path)
        
        if isinstance(mesh, trimesh.PointCloud):
            points = mesh.vertices.astype(np.float32)
            if hasattr(mesh, 'colors') and mesh.colors is not None:
                colors = mesh.colors
                # 处理颜色值范围 [0, 1] 或 [0, 255]
                if colors.max() <= 1.0:
                    colors = (colors * 255.0).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)
                # 如果颜色是4通道（RGBA），只取前3个通道（RGB）
                if colors.shape[1] == 4:
                    colors = colors[:, :3]
                elif colors.shape[1] != 3:
                    raise ValueError(f"Unsupported color channels: {colors.shape[1]}, expected 3 or 4")
            else:
                # 如果没有颜色，使用白色
                colors = np.ones((len(points), 3), dtype=np.uint8) * 255
            return points, colors
        elif isinstance(mesh, trimesh.Trimesh):
            # 如果是网格，提取顶点
            points = mesh.vertices.astype(np.float32)
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors
                # 处理颜色值范围 [0, 1] 或 [0, 255]
                if colors.max() <= 1.0:
                    colors = (colors * 255.0).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)
                # 如果颜色是4通道（RGBA），只取前3个通道（RGB）
                if colors.shape[1] == 4:
                    colors = colors[:, :3]
                elif colors.shape[1] != 3:
                    raise ValueError(f"Unsupported color channels: {colors.shape[1]}, expected 3 or 4")
            else:
                colors = np.ones((len(points), 3), dtype=np.uint8) * 255
            return points, colors
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        raise


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    method: str = "open3d",
) -> tuple[np.ndarray, np.ndarray]:
    """
    对点云进行体素化下采样
    
    Args:
        points: (N, 3) numpy array，点云坐标
        colors: (N, 3) numpy array，点云颜色 (RGB, 0-255)
        voxel_size: 体素大小（单位与点云坐标系一致）
        method: 下采样方法，"open3d" 或 "average"
    
    Returns:
        downsampled_points: (M, 3) numpy array
        downsampled_colors: (M, 3) numpy array
    """
    if voxel_size <= 0 or len(points) == 0:
        return points, colors
    
    if method == "open3d" and HAS_OPEN3D:
        # 使用 Open3D 的体素化下采样（推荐）
        # 确保点数和颜色数匹配
        if len(points) != len(colors):
            raise ValueError(f"Points and colors length mismatch: {len(points)} vs {len(colors)}")
        
        # 确保颜色数组形状正确
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"Colors shape should be (N, 3), got {colors.shape}")
        
        # 确保颜色值在有效范围内 [0, 255]
        colors_clipped = np.clip(colors, 0, 255).astype(np.float64) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_clipped)
        
        # 执行体素下采样
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        
        # 转换回numpy数组
        downsampled_points = np.asarray(pcd_downsampled.points).astype(np.float32)
        downsampled_colors = (np.asarray(pcd_downsampled.colors) * 255.0).astype(np.uint8)
        
        return downsampled_points, downsampled_colors
    
    elif method == "average":
        # 使用平均值方法（不依赖 Open3D）
        # 将点分配到体素网格
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # 使用字典存储每个体素的点和颜色
        voxel_dict = {}
        for i in range(len(points)):
            voxel_key = tuple(voxel_indices[i])
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points': [],
                    'colors': []
                }
            voxel_dict[voxel_key]['points'].append(points[i])
            voxel_dict[voxel_key]['colors'].append(colors[i])
        
        # 计算每个体素的平均值
        downsampled_points = []
        downsampled_colors = []
        for voxel_key, data in voxel_dict.items():
            # 点取体素中心（体素索引 * 体素大小 + 体素大小/2）
            voxel_center = np.array(voxel_key) * voxel_size + voxel_size / 2.0
            # 颜色取平均
            avg_color = np.mean(data['colors'], axis=0).astype(np.uint8)
            
            downsampled_points.append(voxel_center)
            downsampled_colors.append(avg_color)
        
        return np.array(downsampled_points, dtype=np.float32), np.array(downsampled_colors, dtype=np.uint8)
    
    else:
        # 如果方法不支持，返回原始点云
        print(f"[Warning] Unknown voxel downsampling method: {method}, skipping downsampling")
        return points, colors


def random_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    num_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    随机下采样点云
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array
        num_max: 最大点数
    
    Returns:
        downsampled_points: (M, 3) numpy array
        downsampled_colors: (M, 3) numpy array
    """
    if num_max <= 0 or len(points) <= num_max:
        return points, colors
    
    # 随机选择索引
    indices = np.random.choice(len(points), num_max, replace=False)
    return points[indices], colors[indices]


def remove_invalid_points(
    points: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    移除无效点（非有限值）
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array
    
    Returns:
        valid_points: (M, 3) numpy array
        valid_colors: (M, 3) numpy array
    """
    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask], colors[valid_mask]


def remove_duplicate_points(
    points: np.ndarray,
    colors: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    移除重复点（基于距离阈值）
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array
        threshold: 距离阈值
    
    Returns:
        unique_points: (M, 3) numpy array
        unique_colors: (M, 3) numpy array
    """
    if threshold <= 0 or len(points) == 0:
        return points, colors
    
    if HAS_OPEN3D:
        # 使用 Open3D 的 remove_duplicated_points
        # 确保点数和颜色数匹配
        if len(points) != len(colors):
            raise ValueError(f"Points and colors length mismatch: {len(points)} vs {len(colors)}")
        
        # 确保颜色数组形状正确
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"Colors shape should be (N, 3), got {colors.shape}")
        
        # 确保颜色值在有效范围内 [0, 255]
        colors_clipped = np.clip(colors, 0, 255).astype(np.float64) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_clipped)
        
        pcd, indices = pcd.remove_duplicated_points(threshold)
        
        unique_points = np.asarray(pcd.points).astype(np.float32)
        unique_colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
        
        return unique_points, unique_colors
    else:
        # 简单的基于体素的去重（使用体素大小作为阈值）
        return voxel_downsample(points, colors, threshold, method="average")


def apply_bbox_filter(
    points: np.ndarray,
    colors: np.ndarray,
    bbox: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    应用边界框过滤
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array
        bbox: [min_x, max_x, min_y, max_y, min_z, max_z]
    
    Returns:
        filtered_points: (M, 3) numpy array
        filtered_colors: (M, 3) numpy array
    """
    if bbox is None or len(bbox) != 6:
        return points, colors
    
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    
    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    
    return points[mask], colors[mask]


def downsize_ply(config_path: str):
    """
    主函数：对单个PLY文件进行体素化下采样
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("PLY 文件体素化下采样配置")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 提取配置参数
    input_file = config["input_file"]
    output_file = config["output_file"]
    
    voxel_size = config.get("voxel_size", 0.0)
    voxel_downsample_method = config.get("voxel_downsample_method", "open3d")
    maxpoints = config.get("maxpoints", 0)  # 最大点数上限
    
    remove_invalid = config.get("remove_invalid_points", True)
    remove_duplicate_threshold = config.get("remove_duplicate_threshold", 0.0)
    bbox_filter = config.get("bbox_filter")
    
    show_progress = config.get("show_progress", True)
    
    # 1. 加载PLY文件
    print(f"\n加载PLY文件: {input_file}")
    if not os.path.exists(input_file):
        raise ValueError(f"Input file does not exist: {input_file}")
    
    try:
        points, colors = load_ply_file(input_file)
        print(f"  ✓ 加载成功: {len(points):,} 点")
        print(f"  点云范围:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    except Exception as e:
        raise RuntimeError(f"Failed to load PLY file: {e}")
    
    # 2. 应用过滤
    # 移除无效点
    if remove_invalid:
        print(f"\n移除无效点...")
        original_count = len(points)
        points, colors = remove_invalid_points(points, colors)
        print(f"  ✓ 从 {original_count:,} 点减少到 {len(points):,} 点")
    
    # 边界框过滤
    if bbox_filter is not None:
        print(f"\n应用边界框过滤: {bbox_filter}")
        original_count = len(points)
        points, colors = apply_bbox_filter(points, colors, bbox_filter)
        print(f"  ✓ 从 {original_count:,} 点减少到 {len(points):,} 点")
    
    # 移除重复点
    if remove_duplicate_threshold > 0:
        print(f"\n移除重复点 (阈值: {remove_duplicate_threshold})...")
        original_count = len(points)
        points, colors = remove_duplicate_points(points, colors, remove_duplicate_threshold)
        print(f"  ✓ 从 {original_count:,} 点减少到 {len(points):,} 点")
    
    # 3. 下采样
    # 先进行体素化下采样
    if voxel_size > 0:
        print(f"\n进行体素化下采样 (体素大小: {voxel_size}, 方法: {voxel_downsample_method})...")
        original_count = len(points)
        points, colors = voxel_downsample(points, colors, voxel_size, voxel_downsample_method)
        print(f"  ✓ 从 {original_count:,} 点减少到 {len(points):,} 点 (减少 {100*(1-len(points)/original_count):.1f}%)")
    
    # 再进行随机采样（如果超过 maxpoints）
    if maxpoints > 0 and len(points) > maxpoints:
        print(f"\n进行随机下采样 (最大点数: {maxpoints:,})...")
        original_count = len(points)
        points, colors = random_downsample(points, colors, maxpoints)
        print(f"  ✓ 从 {original_count:,} 点减少到 {len(points):,} 点 (减少 {100*(1-len(points)/original_count):.1f}%)")
    elif maxpoints > 0:
        print(f"\n点云数量 ({len(points):,}) 未超过最大点数限制 ({maxpoints:,})，跳过随机下采样")
    
    # 4. 保存结果
    print(f"\n保存结果: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if len(points) > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        pc.export(output_file)
        print(f"  ✓ 已保存 {len(points):,} 点")
    else:
        # Create empty point cloud file
        pc = trimesh.points.PointCloud(vertices=np.zeros((0, 3)), colors=np.zeros((0, 3), dtype=np.uint8))
        pc.export(output_file)
        print(f"  ⚠️  警告: 没有有效点，已创建空的 PLY 文件")
    
    # 5. 输出统计信息
    print(f"\n{'=' * 60}")
    print("处理完成！")
    print(f"{'=' * 60}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"最终点云数量: {len(points):,}")
    if voxel_size > 0:
        print(f"体素大小: {voxel_size} (注意：基于点云坐标系，可能是相对尺度)")
    if maxpoints > 0:
        print(f"最大点数限制: {maxpoints:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对单个PLY文件进行体素化下采样")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ply_downsize_config.yaml"),
        help="配置文件路径",
    )
    
    args = parser.parse_args()
    
    downsize_ply(args.config)
