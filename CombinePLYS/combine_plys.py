#!/usr/bin/env python3
"""
合并多个优化后的点云文件为一个PLY文件
参考 viser_rig_ply_optdepth.py 的点云对齐方式
"""

import numpy as np
import pycolmap
import argparse
import sys
import yaml
import open3d as o3d
import struct
from pathlib import Path
from plyfile import PlyData, PlyElement

def load_config(config_path: Path) -> dict:
    """加载YAML配置文件"""
    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}，将使用默认值")
        return {}
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    return config

def load_ply(ply_path):
    """
    加载PLY点云文件
    Returns:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
    """
    try:
        ply = PlyData.read(ply_path)
        vertex = ply["vertex"]
        
        points = np.stack([
            vertex["x"],
            vertex["y"],
            vertex["z"]
        ], axis=1).astype(np.float32)
        
        colors = np.stack([
            vertex["red"],
            vertex["green"],
            vertex["blue"]
        ], axis=1).astype(np.uint8)
        
        return points, colors
    except Exception as e:
        raise RuntimeError(f"无法读取PLY文件 {ply_path}: {e}")

def transform_points(points, rig_from_world, cam_from_rig=None, camera_coord_correction=None):
    """
    将点云从局部坐标系转换到全局坐标系（世界坐标系）
    Args:
        points: (N, 3) 局部坐标系点云
            - 如果cam_from_rig为None：点云在rig坐标系中（默认情况）
            - 如果cam_from_rig不为None：点云在camera坐标系中（如DAP生成的点云）
        rig_from_world: pycolmap.Rigid3d 变换（rig_from_world，表示从世界坐标系到rig坐标系的变换）
        cam_from_rig: 可选的pycolmap.Rigid3d变换（cam_from_rig，表示从rig坐标系到camera坐标系的变换）
            如果提供，会先将点云从camera坐标系转换到rig坐标系
        camera_coord_correction: (3, 3) 可选的相机坐标系修正矩阵，用于在camera坐标系中修正点云坐标轴
            例如：DAP点云的x轴与camera12的-z对齐，y与x对齐，z与-y对齐
    """
    # 如果提供了camera_coord_correction，先在camera坐标系中应用修正
    # 这个修正应该在转换到rig坐标系之前应用
    if camera_coord_correction is not None:
        points_T = points.T
        points_T = camera_coord_correction @ points_T  # 在camera坐标系中修正
        points = points_T.T
    
    # 如果提供了cam_from_rig，需要先应用它的逆变换
    # 将点云从camera坐标系转换到rig坐标系
    if cam_from_rig is not None:
        # camera坐标系 -> rig坐标系
        rig_from_cam = cam_from_rig.inverse()
        R_rig_cam = rig_from_cam.rotation.matrix()
        t_rig_cam = rig_from_cam.translation
        
        points_T = points.T
        points_rig_T = R_rig_cam @ points_T + t_rig_cam[:, None]
        points = points_rig_T.T
    # 否则，点云已经在rig坐标系中
    
    # rig坐标系 -> 世界坐标系
    # rig_from_world 表示从世界坐标系到rig坐标系的变换
    # 我们需要 world_from_rig 来将点云从rig坐标系转换到世界坐标系
    # world_from_rig = (rig_from_world)^(-1)
    
    # 使用pycolmap的inverse方法（更可靠）
    if hasattr(rig_from_world, 'inverse'):
        world_from_rig = rig_from_world.inverse()
        R_world_rig = world_from_rig.rotation.matrix()  # (3, 3)
        t_world_rig = world_from_rig.translation  # (3,)
    else:
        # 手动计算inverse（备用方法）
        R = rig_from_world.rotation.matrix()  # (3, 3)
        t = rig_from_world.translation  # (3,)
        R_world_rig = R.T  # 旋转矩阵的转置
        t_world_rig = -R.T @ t  # 平移
    
    # 应用变换：点云以相机为原点，直接变换即可
    # world_point = R_world_rig @ rig_point + t_world_rig
    points_T = points.T  # (3, N)
    transformed_T = R_world_rig @ points_T + t_world_rig[:, None]  # (3, N)
    transformed_points = transformed_T.T  # (N, 3)
    
    return transformed_points

def build_pano_to_frame_mapping(recon):
    """
    建立全景图名称到frame的映射关系
    Args:
        recon: pycolmap.Reconstruction对象
    Returns:
        pano_to_frame: dict, {pano_name: frame_id}
    """
    pano_to_frame = {}
    
    # 遍历所有图像，提取pano_name和对应的frame_id
    for img_id, img in recon.images.items():
        if img.frame_id not in recon.frames:
            continue
        
        # 图像名称格式: pano_camera{idx}/{pano_name}.png
        # 例如: pano_camera0/point2_median.png
        img_name = img.name
        if '/' in img_name:
            pano_name = img_name.split('/')[-1]  # 获取文件名
            pano_name = Path(pano_name).stem  # 去掉扩展名
            
            # 如果这个pano还没有映射，或者当前frame有pose而之前的没有，则更新
            if pano_name not in pano_to_frame:
                pano_to_frame[pano_name] = img.frame_id
            else:
                # 优先选择有pose的frame
                current_frame = recon.frames[img.frame_id]
                existing_frame = recon.frames[pano_to_frame[pano_name]]
                if current_frame.has_pose() and not existing_frame.has_pose():
                    pano_to_frame[pano_name] = img.frame_id
    
    return pano_to_frame

def downsample_pointcloud(points, colors, voxel_size):
    """
    对点云进行体素下采样
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
        voxel_size: 体素大小（米）
    Returns:
        downsampled_points: (M, 3) numpy array
        downsampled_colors: (M, 3) numpy array
    """
    if voxel_size <= 0:
        return points, colors
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    
    # 执行体素下采样
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    
    # 转换回numpy数组
    downsampled_points = np.asarray(pcd_downsampled.points).astype(np.float32)
    downsampled_colors = (np.asarray(pcd_downsampled.colors) * 255.0).astype(np.uint8)
    
    return downsampled_points, downsampled_colors

def write_points3D_bin(points, colors, output_file):
    """
    写入 COLMAP points3D.bin 文件（二进制格式）
    参考: dense2colmap_points.py
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
        output_file: 输出文件路径
    """
    def write_uint64(fid, value):
        fid.write(struct.pack('<Q', value))  # '<' 表示小端序
    
    def write_uint32(fid, value):
        fid.write(struct.pack('<I', value))
    
    def write_double(fid, value):
        fid.write(struct.pack('<d', value))
    
    def write_uint8(fid, value):
        fid.write(struct.pack('B', value))
    
    num_points = len(points)
    
    with open(output_file, 'wb') as fid:
        # 写入点数
        write_uint64(fid, num_points)
        
        # 按 ID 顺序写入（从 1 开始）
        for i in range(num_points):
            point_id = i + 1
            xyz = points[i].astype(np.float64)
            color = colors[i].astype(np.uint8) if colors is not None else np.array([255, 255, 255], dtype=np.uint8)
            error = 0.0  # 稠密点云没有重投影误差
            
            # 写入点 ID
            write_uint64(fid, point_id)
            
            # 写入 3D 坐标 (3 * double)
            write_double(fid, float(xyz[0]))
            write_double(fid, float(xyz[1]))
            write_double(fid, float(xyz[2]))
            
            # 写入颜色 (3 * uint8)
            write_uint8(fid, int(color[0]))
            write_uint8(fid, int(color[1]))
            write_uint8(fid, int(color[2]))
            
            # 写入误差 (double)
            write_double(fid, float(error))
            
            # 写入 track 长度（稠密点云没有 track，所以为 0）
            track_length = 0
            write_uint64(fid, track_length)
            
            # track 为空，不需要写入 track 数据

def write_points3D_txt(points, colors, output_file):
    """
    写入 COLMAP points3D.txt 文件（文本格式）
    参考: dense2colmap_points.py
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
        output_file: 输出文件路径
    """
    if points is None or len(points) == 0:
        print("Warning: 点云为空，创建空的 points3D.txt")
        with open(output_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...\n")
            f.write("# Number of points: 0\n")
        return
    
    print(f"正在写入 {len(points)} 个点到 {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...\n")
        f.write(f"# Number of points: {len(points)}\n")
        
        for i in range(len(points)):
            point_id = i + 1
            x, y, z = float(points[i][0]), float(points[i][1]), float(points[i][2])
            
            if colors is not None:
                r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            else:
                r, g, b = 255, 255, 255
            
            error = 0.0  # 稠密点云没有重投影误差
            
            # 写入点（没有 track 信息）
            line = f"{point_id} {x:.15f} {y:.15f} {z:.15f} {r} {g} {b} {error:.1f}"
            f.write(line + "\n")
    
    print(f"成功写入 {len(points)} 个点到 {output_file}")

def save_colmap_points3D(points, colors, output_dir):
    """
    保存点云为 COLMAP points3D 格式到指定目录（仅生成二进制格式）
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
        output_dir: 输出目录路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    points3d_bin = output_dir / "points3D.bin"
    
    # 生成二进制格式
    print(f"生成 points3D.bin...")
    write_points3D_bin(points, colors, str(points3d_bin))
    
    print(f"✅ 成功生成 COLMAP points3D 格式文件")
    print(f"   points3D.bin: {points3d_bin}")

def save_ply(points, colors, output_path):
    """
    保存点云为PLY文件
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (RGB, 0-255)
        output_path: 输出文件路径
    """
    # 确保颜色值在有效范围内
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    # 创建PLY数据
    vertices = np.empty(
        len(points),
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
        ]
    )
    
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 1]
    vertices['z'] = points[:, 2]
    vertices['red'] = colors[:, 0]
    vertices['green'] = colors[:, 1]
    vertices['blue'] = colors[:, 2]
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(str(output_path))
    print(f"✅ 保存合并后的点云到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="合并多个优化后的点云文件为一个PLY文件")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                       help="配置文件路径（默认: config.yaml）")
    parser.add_argument("--scene", type=str, default=None,
                       help="场景名称，覆盖配置文件")
    parser.add_argument("--colmap_dir", type=str, default=None,
                       help="colmap_STAGE数据集根目录，覆盖配置文件")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="输入点云目录，覆盖配置文件")
    parser.add_argument("--output", type=str, default=None,
                       help="输出PLY文件路径，覆盖配置文件")
    parser.add_argument("--camera_name", type=str, default=None,
                       help="点云所在的虚拟相机名称，覆盖配置文件")
    parser.add_argument("--no_transform", action="store_true", default=None,
                       help="不对点云应用坐标变换，覆盖配置文件")
    parser.add_argument("--voxel_size", type=float, default=None,
                       help="体素下采样大小（米），0表示不下采样，覆盖配置文件")
    parser.add_argument("--generate_colmap_points3d", action="store_true", default=None,
                       help="生成 COLMAP points3D 格式文件到 output 目录，覆盖配置文件")
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 从配置文件获取默认值，命令行参数优先
    paths_cfg = config.get("paths", {}) or {}
    processing_cfg = config.get("processing", {}) or {}
    
    # 设置参数值：命令行参数优先，否则使用配置文件，最后使用硬编码默认值
    scene = args.scene if args.scene is not None else paths_cfg.get("scene", "BridgeB")
    colmap_dir = args.colmap_dir if args.colmap_dir is not None else paths_cfg.get("colmap_dir", "/root/autodl-tmp/data/colmap_STAGE1_4x")
    input_dir = args.input_dir if args.input_dir is not None else paths_cfg.get("input_dir")
    output_path = args.output if args.output is not None else processing_cfg.get("output", "output/merged.ply")
    camera_name = args.camera_name if args.camera_name is not None else processing_cfg.get("camera_name", "pano_camera12")
    no_transform = args.no_transform if args.no_transform is not None else processing_cfg.get("no_transform", False)
    voxel_size = args.voxel_size if args.voxel_size is not None else processing_cfg.get("voxel_size", 0.05)
    generate_colmap_points3d = args.generate_colmap_points3d if args.generate_colmap_points3d is not None else processing_cfg.get("generate_colmap_points3d", False)
    
    # 构建路径
    colmap_dir = Path(colmap_dir)
    scene_colmap_dir = colmap_dir / scene
    colmap_sparse_dir = scene_colmap_dir / "sparse" / "0"
    
    # 输入点云目录
    if input_dir is None:
        print("❌ 未指定输入点云目录（使用 --input_dir 或配置文件中设置 paths.input_dir）")
        sys.exit(1)
    input_dir = Path(input_dir)
    
    # 输出文件路径
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查目录是否存在
    if not input_dir.exists():
        print(f"❌ 输入点云目录不存在: {input_dir}")
        sys.exit(1)
    
    if not colmap_sparse_dir.exists():
        print(f"❌ COLMAP模型目录不存在: {colmap_sparse_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("合并点云文件")
    print("=" * 60)
    print(f"场景: {scene}")
    print(f"COLMAP目录: {colmap_sparse_dir}")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")
    print(f"相机名称: {camera_name}")
    print(f"应用坐标变换: {not no_transform}")
    print(f"下采样体素大小: {voxel_size}m" + (" (不下采样)" if voxel_size <= 0 else ""))
    print(f"生成 COLMAP points3D: {generate_colmap_points3d}")
    if generate_colmap_points3d:
        print(f"points3D 输出目录: {output_path.parent}")
    print("=" * 60)
    
    # 加载COLMAP重建结果
    print(f"\n📖 读取COLMAP重建结果: {colmap_sparse_dir}")
    try:
        recon = pycolmap.Reconstruction(str(colmap_sparse_dir))
    except Exception as e:
        print(f"❌ 无法读取COLMAP重建结果: {e}")
        sys.exit(1)
    
    # 获取所有有pose的frames（rigs）
    frames_with_pose = {fid: f for fid, f in recon.frames.items() if f.has_pose()}
    
    if len(frames_with_pose) == 0:
        print("❌ 没有找到有pose的frames（rigs）")
        sys.exit(1)
    
    print(f"🔹 找到 {len(frames_with_pose)} 个有pose的rigs")
    
    # 建立pano_name到frame的映射
    print("🔗 建立全景图名称到frame的映射...")
    pano_to_frame = build_pano_to_frame_mapping(recon)
    print(f"   ✅ 找到 {len(pano_to_frame)} 个全景图")
    
    # 获取所有PLY文件
    ply_files = sorted(input_dir.glob("*.ply"))
    if len(ply_files) == 0:
        print(f"❌ 输入目录中没有找到*.ply文件: {input_dir}")
        sys.exit(1)
    
    print(f"\n📁 找到 {len(ply_files)} 个PLY文件")
    
    # DAP点云在camera12坐标系中的坐标轴修正
    # 点云的x轴 → camera12的-z轴
    # 点云的y轴 → camera12的x轴
    # 点云的z轴 → camera12的-y轴
    camera_coord_correction_matrix = np.array([
        [0,  1, 0],   # new_x = old_y
        [0,  0, -1],  # new_y = -old_z
        [-1, 0,  0]   # new_z = -old_x
    ], dtype=np.float32)
    
    # 存储所有合并的点云
    all_points = []
    all_colors = []
    
    # 处理每个PLY文件
    print("\n📦 处理点云文件...")
    processed_count = 0
    skipped_count = 0
    
    for ply_path in ply_files:
        try:
            # 从文件名提取pano_name（去掉".ply"后缀，并尝试去掉常见后缀）
            filename = ply_path.stem  # 去掉.ply
            
            # 尝试匹配pano_name：先尝试完整文件名，如果匹配不上，再尝试去掉常见后缀
            pano_name = filename
            if pano_name not in pano_to_frame:
                # 尝试去掉常见后缀
                common_suffixes = ["_corrected", "_optimized", "_refined", "_single_opt", "_median"]
                for suffix in common_suffixes:
                    if filename.endswith(suffix):
                        pano_name = filename[:-len(suffix)]
                        if pano_name in pano_to_frame:
                            break
                # 如果还是匹配不上，尝试去掉"optimized_"前缀（向后兼容）
                if pano_name not in pano_to_frame and filename.startswith("optimized_"):
                    pano_name = filename[len("optimized_"):]
            
            # 查找对应的frame
            if pano_name not in pano_to_frame:
                print(f"⚠️  跳过 {ply_path.name}: 在COLMAP中找不到对应的frame (尝试的pano_name: {pano_name}, 原始文件名: {filename})")
                skipped_count += 1
                continue
            
            frame_id = pano_to_frame[pano_name]
            if frame_id not in frames_with_pose:
                print(f"⚠️  跳过 {ply_path.name}: frame {frame_id} 没有pose")
                skipped_count += 1
                continue
            
            frame = frames_with_pose[frame_id]
            rig_from_world = frame.rig_from_world
            
            # 加载点云
            points_local, colors_ply = load_ply(ply_path)
            
            # 根据参数决定是否应用坐标变换
            if no_transform:
                # 假设点云已经在世界坐标系中
                points_world = points_local
            else:
                # 获取指定相机的cam_from_rig变换
                cam_from_rig = None
                camera_found = False
                
                for img_id, img in recon.images.items():
                    if img.frame_id == frame_id and camera_name in img.name:
                        # 获取相机的cam_from_world
                        cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                        # 计算cam_from_rig: cam_from_world = cam_from_rig @ rig_from_world
                        # 所以: cam_from_rig = cam_from_world @ world_from_rig
                        world_from_rig = rig_from_world.inverse()
                        cam_from_rig = cam_from_world * world_from_rig
                        camera_found = True
                        break
                
                if not camera_found:
                    print(f"⚠️  警告: 未找到{camera_name}，无法应用坐标变换，将跳过此点云: {ply_path.name}")
                    skipped_count += 1
                    continue
                
                # 应用坐标变换（从camera坐标系到世界坐标系）
                points_world = transform_points(
                    points_local,
                    rig_from_world,
                    cam_from_rig=cam_from_rig,
                    camera_coord_correction=camera_coord_correction_matrix
                )
            
            # 添加到合并列表
            all_points.append(points_world)
            all_colors.append(colors_ply)
            processed_count += 1
            
            print(f"   ✅ {pano_name}: {len(points_world):,} 点")
            
        except Exception as e:
            print(f"❌ 处理 {ply_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    if processed_count == 0:
        print("❌ 没有成功处理任何点云文件")
        sys.exit(1)
    
    print(f"\n📊 处理统计:")
    print(f"   成功处理: {processed_count} 个文件")
    print(f"   跳过: {skipped_count} 个文件")
    
    # 合并所有点云
    print("\n🔗 合并点云...")
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"   ✅ 合并后总点数: {len(merged_points):,}")
    print(f"   点云范围:")
    print(f"      X: [{merged_points[:, 0].min():.2f}, {merged_points[:, 0].max():.2f}]")
    print(f"      Y: [{merged_points[:, 1].min():.2f}, {merged_points[:, 1].max():.2f}]")
    print(f"      Z: [{merged_points[:, 2].min():.2f}, {merged_points[:, 2].max():.2f}]")
    
    # 下采样（如果启用）
    if voxel_size > 0:
        print(f"\n📉 下采样点云 (体素大小: {voxel_size}m)...")
        original_count = len(merged_points)
        merged_points, merged_colors = downsample_pointcloud(merged_points, merged_colors, voxel_size)
        print(f"   ✅ 下采样后点数: {len(merged_points):,} (从 {original_count:,} 减少到 {len(merged_points):,}, 减少 {100*(1-len(merged_points)/original_count):.1f}%)")
    
    # 保存合并后的点云
    print(f"\n💾 保存合并后的点云...")
    save_ply(merged_points, merged_colors, output_path)
    
    # 生成 COLMAP points3D 格式（如果启用）
    if generate_colmap_points3d:
        print(f"\n📦 生成 COLMAP points3D 格式...")
        try:
            # 输出到 output 目录（与 PLY 文件同一目录）
            output_dir = output_path.parent
            save_colmap_points3D(
                merged_points,
                merged_colors,
                output_dir
            )
        except Exception as e:
            print(f"⚠️  生成 COLMAP points3D 格式时出错: {e}")
            import traceback
            traceback.print_exc()
            print("   继续执行，PLY 文件已成功保存")
    
    print(f"\n✅ 合并完成!")
    print(f"   输出文件: {output_path}")
    print(f"   总点数: {len(merged_points):,}")

if __name__ == "__main__":
    main()
