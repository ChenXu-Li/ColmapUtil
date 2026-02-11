#!/usr/bin/env python3
"""
将 COLMAP 稠密重建的点云转换为 COLMAP points3D 格式
用于替换稀疏重建的 points3D，以便在 Gaussian Splatting 训练中使用稠密点云
"""

import os
import sys
import argparse
import struct
import shutil
from datetime import datetime
import numpy as np
from plyfile import PlyData

# 检测 pycolmap 版本并导入相应的 API
HAS_PYCOLMAP_0_0_1 = False
HAS_PYCOLMAP_3_X = False
PYCOLMAP_VERSION = None
SceneManager = None

try:
    from pycolmap import SceneManager
    HAS_PYCOLMAP_0_0_1 = True
    PYCOLMAP_VERSION = "0.0.1"
    print("检测到 pycolmap 0.0.1，将使用 SceneManager API")
except ImportError:
    # 检查是否是 pycolmap 3.x 版本
    try:
        import pycolmap
        if hasattr(pycolmap, '__version__'):
            version = pycolmap.__version__
            PYCOLMAP_VERSION = version
            if version.startswith('3.'):
                HAS_PYCOLMAP_3_X = True
                print(f"检测到 pycolmap {version}，将使用 Reconstruction API")
            else:
                print(f"Warning: 检测到 pycolmap {version}，但此脚本仅支持 0.0.1 或 3.x")
        else:
            # 尝试检查是否有 Reconstruction
            if hasattr(pycolmap, 'Reconstruction'):
                HAS_PYCOLMAP_3_X = True
                PYCOLMAP_VERSION = "3.x (unknown)"
                print("检测到 pycolmap 3.x（版本未知），将使用 Reconstruction API")
    except ImportError:
        pass

if not HAS_PYCOLMAP_0_0_1 and not HAS_PYCOLMAP_3_X:
    raise ImportError("需要安装 pycolmap 0.0.1 或 3.x 版本: pip install pycolmap==0.0.1 或 pip install pycolmap")


def _write_points3D_bin_direct(positions, colors, output_file):
    """
    直接写入 points3D.bin 文件（用于 pycolmap 3.x）
    参考: colmap/src/colmap/scene/reconstruction_io_binary.cc
    """
    # 使用小端序写入（COLMAP 使用小端序）
    def write_uint64(fid, value):
        fid.write(struct.pack('<Q', value))  # '<' 表示小端序
    
    def write_uint32(fid, value):
        fid.write(struct.pack('<I', value))
    
    def write_double(fid, value):
        fid.write(struct.pack('<d', value))
    
    def write_uint8(fid, value):
        fid.write(struct.pack('B', value))
    
    num_points = len(positions)
    
    with open(output_file, 'wb') as fid:
        # 写入点数
        write_uint64(fid, num_points)
        
        # 按 ID 顺序写入（从 1 开始）
        for i in range(num_points):
            point_id = i + 1
            xyz = positions[i].astype(np.float64)
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


def _save_points3D_bin_compat(manager, output_file):
    """
    兼容 Python 3 的 points3D.bin 保存函数（用于 pycolmap 0.0.1）
    修复 pycolmap 0.0.1 的 Python 2/3 兼容性问题
    """
    INVALID_POINT3D = SceneManager.INVALID_POINT3D
    
    # 计算有效点数（使用 Python 3 兼容的 values()）
    num_valid_points3D = sum(
        1 for point3D_idx in manager.point3D_id_to_point3D_idx.values()
        if point3D_idx != INVALID_POINT3D
    )
    
    # 使用 Python 3 兼容的 items()
    iter_point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx.items()
    
    with open(output_file, 'wb') as fid:
        # 写入点数（使用 'Q' 表示 unsigned long long）
        fid.write(struct.pack('Q', num_valid_points3D))
        
        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == INVALID_POINT3D:
                continue
            
            # 写入点 ID
            fid.write(struct.pack('Q', point3D_id))
            # 写入 3D 坐标 (3 * float64)
            fid.write(manager.points3D[point3D_idx].astype(np.float64).tobytes())
            # 写入颜色 (3 * uint8)
            fid.write(manager.point3D_colors[point3D_idx].astype(np.uint8).tobytes())
            # 写入误差 (float64)
            fid.write(struct.pack('d', manager.point3D_errors[point3D_idx]))
            # 写入 track 长度
            track_data = manager.point3D_id_to_images[point3D_id]
            # 确保 track_data 是 numpy 数组
            if not isinstance(track_data, np.ndarray):
                track_data = np.array(track_data, dtype=np.uint32)
            track_len = len(track_data)
            fid.write(struct.pack('Q', track_len))
            # 写入 track 数据（如果有）
            # track 格式：每个元素是 (IMAGE_ID, POINT2D_IDX) 对，需要展平为一维数组
            if track_len > 0:
                # 确保是 uint32 类型并展平
                track_flat = track_data.astype(np.uint32).flatten()
                fid.write(track_flat.tobytes())

def read_ply_file(ply_file):
    """从 PLY 文件读取点云和颜色"""
    try:
        ply = PlyData.read(ply_file)
        vertex = ply["vertex"]
        
        # 提取位置
        positions = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]],
            axis=1
        ).astype(np.float32)
        
        # 检查是否有颜色
        has_colors = any(p.name in ['red', 'green', 'blue'] for p in vertex.properties)
        
        if has_colors:
            try:
                colors = np.stack(
                    [vertex["red"], vertex["green"], vertex["blue"]],
                    axis=1
                ).astype(np.uint8)
            except (KeyError, ValueError) as e:
                print(f"Warning: 无法读取颜色: {e}")
                # 如果没有颜色，使用白色
                colors = np.ones((len(positions), 3), dtype=np.uint8) * 255
        else:
            # 默认使用白色
            colors = np.ones((len(positions), 3), dtype=np.uint8) * 255
        
        return positions, colors
    except Exception as e:
        print(f"Error: 无法读取 PLY 文件 {ply_file}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def write_points3d_txt(positions, colors, output_file):
    """
    写入 points3D.txt 文件（COLMAP 格式）
    
    格式: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX) ...
    注意: 稠密点云没有 track 信息，所以 TRACK 部分为空
    """
    if positions is None or len(positions) == 0:
        print("Warning: 点云为空，创建空的 points3D.txt")
        with open(output_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...\n")
            f.write("# Number of points: 0\n")
        return
    
    # 验证形状
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError(f"Invalid positions shape: {positions.shape}, expected (N, 3)")
    
    if colors is not None:
        if len(colors.shape) != 2 or colors.shape[1] != 3:
            raise ValueError(f"Invalid colors shape: {colors.shape}, expected (N, 3)")
        if len(colors) != len(positions):
            raise ValueError(f"Colors length ({len(colors)}) != positions length ({len(positions)})")
    
    print(f"正在写入 {len(positions)} 个点到 {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...\n")
        f.write(f"# Number of points: {len(positions)}\n")
        
        for i in range(len(positions)):
            point_id = i + 1
            x, y, z = float(positions[i][0]), float(positions[i][1]), float(positions[i][2])
            
            if colors is not None:
                r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            else:
                r, g, b = 255, 255, 255
            
            error = 0.0  # 稠密点云没有重投影误差
            
            # 写入点（没有 track 信息）
            # 使用精确的格式，确保没有尾随空格，且字段数正好是 8
            # 格式: POINT3D_ID X Y Z R G B ERROR (没有 TRACK 信息)
            line = f"{point_id} {x:.15f} {y:.15f} {z:.15f} {r} {g} {b} {error:.1f}"
            f.write(line + "\n")
    
    print(f"成功写入 {len(positions)} 个点到 {output_file}")


def convert_dense_to_colmap(dense_ply_file, colmap_sparse_dir, backup=True):
    """
    将稠密点云转换为 COLMAP points3D 格式并替换稀疏重建的点云
    
    Args:
        dense_ply_file: 稠密重建的 PLY 文件路径（如 dense/fused.ply）
        colmap_sparse_dir: COLMAP 稀疏重建目录（如 sparse/0 或 sparse）
        backup: 是否备份原始的 points3D 文件
    """
    # 检查输入文件
    if not os.path.exists(dense_ply_file):
        raise FileNotFoundError(f"稠密点云文件不存在: {dense_ply_file}")
    
    if not os.path.exists(colmap_sparse_dir):
        raise FileNotFoundError(f"COLMAP 稀疏重建目录不存在: {colmap_sparse_dir}")
    
    # 读取稠密点云
    print(f"正在读取稠密点云: {dense_ply_file}")
    positions, colors = read_ply_file(dense_ply_file)
    
    if positions is None:
        raise ValueError("无法读取点云数据")
    
    print(f"成功读取 {len(positions)} 个点")
    
    # 确定输出文件路径
    points3d_txt = os.path.join(colmap_sparse_dir, "points3D.txt")
    points3d_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    
    # 备份原始文件
    if backup:
        if os.path.exists(points3d_txt):
            backup_txt = points3d_txt + ".sparse_backup"
            # 如果备份文件已存在，使用带时间戳的备份文件名
            if os.path.exists(backup_txt):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_txt = f"{points3d_txt}.sparse_backup_{timestamp}"
                print(f"警告: 备份文件已存在，使用新的备份文件名: {backup_txt}")
            print(f"备份原始 points3D.txt 到 {backup_txt}")
            shutil.copy2(points3d_txt, backup_txt)
        
        if os.path.exists(points3d_bin):
            backup_bin = points3d_bin + ".sparse_backup"
            # 如果备份文件已存在，使用带时间戳的备份文件名
            if os.path.exists(backup_bin):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_bin = f"{points3d_bin}.sparse_backup_{timestamp}"
                print(f"警告: 备份文件已存在，使用新的备份文件名: {backup_bin}")
            print(f"备份原始 points3D.bin 到 {backup_bin}")
            shutil.copy2(points3d_bin, backup_bin)
    
    # 生成二进制格式 points3D.bin
    try:
        if HAS_PYCOLMAP_0_0_1:
            print("使用 pycolmap 0.0.1 (SceneManager) 生成二进制格式 points3D.bin...")
            
            # 读取现有的重建（保留 cameras 和 images 信息）
            manager = SceneManager(colmap_sparse_dir)
            manager.load_cameras()
            manager.load_images()
            
            # 清除现有的 points3D
            manager.points3D = []
            manager.point3D_ids = []
            manager.point3D_id_to_point3D_idx = {}
            manager.point3D_id_to_images = {}
            manager.point3D_colors = []
            manager.point3D_errors = []
            
            # 添加新的 points3D（没有 track 信息）
            for i in range(len(positions)):
                point_id = i + 1
                xyz = positions[i].astype(np.float64)
                color = colors[i].astype(np.uint8) if colors is not None else np.array([255, 255, 255], dtype=np.uint8)
                error = 0.0
                
                # 添加到 manager
                manager.point3D_ids.append(np.uint64(point_id))
                manager.point3D_id_to_point3D_idx[point_id] = len(manager.points3D)
                manager.points3D.append(xyz)
                manager.point3D_colors.append(color)
                manager.point3D_errors.append(np.float64(error))
                # 空的 track（没有图像对应关系）- 使用空数组
                manager.point3D_id_to_images[point_id] = np.array([], dtype=np.uint32).reshape(0, 2)
            
            # 转换为 numpy 数组
            manager.points3D = np.array(manager.points3D)
            manager.point3D_ids = np.array(manager.point3D_ids)
            manager.point3D_colors = np.array(manager.point3D_colors)
            manager.point3D_errors = np.array(manager.point3D_errors)
            
            # 使用兼容的保存函数直接保存 points3D.bin
            print("使用兼容方法生成 points3D.bin...")
            _save_points3D_bin_compat(manager, points3d_bin)
            
        elif HAS_PYCOLMAP_3_X:
            print(f"使用 pycolmap {PYCOLMAP_VERSION} 生成二进制格式 points3D.bin...")
            
            # 直接写入二进制文件，不通过 Reconstruction API（因为需要指定 point_id）
            # 不需要读取现有的重建，因为我们只替换 points3D
            print("直接写入 points3D.bin...")
            _write_points3D_bin_direct(positions, colors, points3d_bin)
        
        print(f"成功生成 points3D.bin 文件（{len(positions)} 个点）")
        
        # 删除 points3D.txt（如果存在），强制使用二进制格式
        if os.path.exists(points3d_txt):
            os.remove(points3d_txt)
            print("已删除 points3D.txt，将使用二进制格式")
        
        # 二进制格式成功生成
        print(f"\n转换完成！")
        print(f"新的 points3D.bin 已保存到: {points3d_bin}")
        print(f"\n注意: 稠密点云没有 track 信息（即点与图像的对应关系），")
        print(f"      这不会影响 Gaussian Splatting 的初始化，因为初始化只需要点的位置和颜色。")
        
    except Exception as e:
        print(f"Error: 无法生成二进制格式: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"生成 points3D.bin 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="将 COLMAP 稠密重建的点云转换为 COLMAP points3D 格式"
    )
    parser.add_argument(
        "--dense_ply",
        type=str,
        required=True,
        help="稠密重建的 PLY 文件路径（如 dense/fused.ply）"
    )
    parser.add_argument(
        "--colmap_sparse_dir",
        type=str,
        required=True,
        help="COLMAP 稀疏重建目录（如 sparse/0 或 sparse）"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不备份原始的 points3D 文件"
    )
    
    args = parser.parse_args()
    
    try:
        convert_dense_to_colmap(
            args.dense_ply,
            args.colmap_sparse_dir,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

