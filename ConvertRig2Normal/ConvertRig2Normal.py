#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP Rig格式转标准格式转换脚本

功能：
1. 将rig格式的图像目录（images/pano_camera0/, pano_camera1/, ...）扁平化为标准格式（images/）
2. 更新sparse重建中的图像路径
3. 移除rigs和frames信息，生成标准COLMAP格式

用法：
    python ConvertRig2Normal.py <输入路径> <输出路径> [--include-points3d] [--overwrite]
    
示例：
    python ConvertRig2Normal.py /root/autodl-tmp/data/colmap_STAGE1_4x/BridgeB /root/autodl-tmp/results/normalcolmap/colmap_STAGE1_4x/BridgeB
    python ConvertRig2Normal.py /path/to/rig_dataset /path/to/output_dataset
    python ConvertRig2Normal.py /path/to/rig_dataset /path/to/output_dataset --include-points3d
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional
import pycolmap


def print_info(msg: str):
    """打印信息"""
    print(f"[INFO] {msg}")


def print_success(msg: str):
    """打印成功信息"""
    print(f"[SUCCESS] {msg}")


def print_warning(msg: str):
    """打印警告信息"""
    print(f"[WARNING] {msg}")


def print_error(msg: str):
    """打印错误信息"""
    print(f"[ERROR] {msg}")


def copy_images_flat(input_images_dir: Path, output_images_dir: Path) -> Dict[str, str]:
    """
    将rig格式的图像目录扁平化
    
    Args:
        input_images_dir: 输入的images目录（包含pano_camera0/, pano_camera1/等子目录）
        output_images_dir: 输出的images目录（扁平化后的目录）
        
    Returns:
        path_mapping: 从旧路径到新路径的映射字典
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    path_mapping = {}
    
    # 统计信息
    total_images = 0
    camera_dirs = []
    
    # 遍历所有相机目录
    for camera_dir in sorted(input_images_dir.iterdir()):
        if not camera_dir.is_dir():
            continue
            
        camera_name = camera_dir.name
        if not camera_name.startswith('pano_camera'):
            print_warning(f"跳过非相机目录: {camera_name}")
            continue
            
        camera_dirs.append(camera_name)
        print_info(f"处理相机目录: {camera_name}")
        
        # 遍历该相机目录下的所有图像
        for image_file in sorted(camera_dir.iterdir()):
            if not image_file.is_file():
                continue
                
            # 检查是否为图像文件
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                continue
            
            # 旧路径（相对于images目录）
            old_rel_path = f"{camera_name}/{image_file.name}"
            
            # 新路径：为了避免不同相机目录下有同名文件，添加相机前缀
            # 格式：pano_camera0_point2_median.png
            new_name = f"{camera_name}_{image_file.name}"
            new_path = output_images_dir / new_name
            
            # 如果目标文件已存在，跳过（避免重复）
            if new_path.exists():
                print_warning(f"文件已存在，跳过: {new_name}")
                continue
            
            # 复制文件
            shutil.copy2(image_file, new_path)
            path_mapping[old_rel_path] = new_name
            total_images += 1
    
    print_success(f"共处理 {len(camera_dirs)} 个相机目录，复制 {total_images} 张图像")
    return path_mapping


def convert_sparse_reconstruction(
    input_sparse_dir: Path,
    output_sparse_dir: Path,
    path_mapping: Dict[str, str],
    include_points3d: bool = False
):
    """
    转换sparse重建，更新图像路径并移除rig信息
    
    Args:
        input_sparse_dir: 输入的sparse目录（包含0/子目录）
        output_sparse_dir: 输出的sparse目录（直接输出到此目录，不创建0/子目录）
        path_mapping: 图像路径映射字典
        include_points3d: 是否包含3D点信息，默认为False
    """
    input_model_dir = input_sparse_dir / "0"
    output_model_dir = output_sparse_dir
    
    if not input_model_dir.exists():
        print_error(f"输入模型目录不存在: {input_model_dir}")
        sys.exit(1)
    
    # 检查必要的文件
    required_files = ['cameras.bin', 'images.bin']
    if include_points3d:
        required_files.append('points3D.bin')
    for fname in required_files:
        if not (input_model_dir / fname).exists():
            print_error(f"缺少必要文件: {fname}")
            sys.exit(1)
    
    print_info("读取COLMAP重建...")
    try:
        reconstruction = pycolmap.Reconstruction(str(input_model_dir))
    except Exception as e:
        print_error(f"无法读取COLMAP重建: {e}")
        sys.exit(1)
    
    print_info(f"原始重建包含: {len(reconstruction.cameras)} 个相机, "
               f"{len(reconstruction.images)} 张图像, "
               f"{len(reconstruction.points3D)} 个3D点")
    
    # 创建新的重建对象（不包含rig信息）
    new_reconstruction = pycolmap.Reconstruction()
    
    # 复制相机（使用trivial rig，为标准格式准备）
    print_info("复制相机信息（创建trivial rig）...")
    for camera_id, camera in reconstruction.cameras.items():
        # 添加相机
        new_reconstruction.add_camera(camera)
        
        # 手动创建trivial rig（rig_id = camera_id）
        # 每个相机都有自己的trivial rig
        rig = pycolmap.Rig()
        rig.rig_id = camera_id
        # 添加ref_sensor（使用相机的sensor_id）
        rig.add_ref_sensor(camera.sensor_id)
        new_reconstruction.add_rig(rig)
    
    # 更新图像路径并复制图像
    print_info("更新图像路径...")
    updated_count = 0
    skipped_count = 0
    
    for image_id, image in reconstruction.images.items():
        old_path = image.name
        
        # 查找新路径
        if old_path in path_mapping:
            new_path = path_mapping[old_path]
        else:
            # 如果路径不在映射中，尝试直接使用文件名
            # 可能是已经扁平化的路径
            if '/' in old_path:
                # 提取文件名部分
                new_path = old_path.split('/')[-1]
                print_warning(f"路径未在映射中找到，使用文件名: {old_path} -> {new_path}")
            else:
                # 已经是扁平化的路径
                new_path = old_path
        
        # 获取图像的姿态（从rig格式转换为标准格式）
        # 在rig格式中，姿态存储在frame中，需要提取
        cam_from_world = image.cam_from_world()
        
        # 创建新的图像对象（复制原图像的所有属性）
        new_image = pycolmap.Image()
        new_image.image_id = image.image_id
        new_image.name = new_path
        new_image.camera_id = image.camera_id
        
        # 复制2D点信息（包含3D点的关联）
        if hasattr(image, 'points2D') and len(image.points2D) > 0:
            # 直接复制points2D列表
            new_image.points2D = list(image.points2D)
        
        # 创建trivial frame（frame_id = image_id）
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = image.camera_id
        # 添加data_id（使用图像的data_id）
        frame.add_data_id(new_image.data_id)
        # 设置姿态（rig_from_world = cam_from_world，因为trivial rig中相机是ref_sensor）
        frame.rig_from_world = cam_from_world
        new_reconstruction.add_frame(frame)
        new_reconstruction.register_frame(frame.frame_id)
        
        # 设置图像的frame_id
        new_image.frame_id = image.image_id
        
        # 添加图像
        new_reconstruction.add_image(new_image)
        updated_count += 1
    
    # 复制3D点（根据参数决定）
    if include_points3d:
        print_info("复制3D点信息...")
        # 由于add_point3D_with_id可能不可用，我们需要使用add_point3D
        # 但需要确保track中的image_id在新重建中存在
        point3D_count = 0
        skipped_count = 0
        
        for point3D_id, point3D in reconstruction.points3D.items():
            # 检查track中的所有图像是否都在新重建中
            track_valid = True
            for track_el in point3D.track.elements:
                if not new_reconstruction.exists_image(track_el.image_id):
                    track_valid = False
                    break
            
            if track_valid:
                try:
                    # 使用add_point3D添加点（会自动分配新ID）
                    # 注意：这会分配新的ID，但track和颜色等信息会保留
                    new_point3D_id = new_reconstruction.add_point3D(
                        point3D.xyz,
                        point3D.track,
                        point3D.color
                    )
                    # 更新error（如果存在）
                    new_point3D = new_reconstruction.point3D(new_point3D_id)
                    if point3D.error != -1:
                        new_point3D.error = point3D.error
                    point3D_count += 1
                except Exception as e:
                    print_warning(f"添加点 {point3D_id} 时出错: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
                if skipped_count <= 10:  # 只显示前10个警告
                    print_warning(f"跳过点 {point3D_id}，因为track中的图像不在新重建中")
        
        print_success(f"复制了 {point3D_count} 个3D点")
        if skipped_count > 0:
            print_warning(f"跳过了 {skipped_count} 个3D点")
    else:
        print_info("跳过3D点信息（未启用 --include-points3d）")
    
    print_success(f"更新了 {updated_count} 张图像的路径")
    
    # 保存新的重建
    output_model_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"保存重建到: {output_model_dir}")
    
    try:
        new_reconstruction.write(str(output_model_dir))
        print_success("重建保存成功")
    except Exception as e:
        print_error(f"保存重建失败: {e}")
        sys.exit(1)
    
    # 验证输出文件
    print_info("验证输出文件...")
    expected_files = ['cameras.bin', 'images.bin']
    if include_points3d:
        expected_files.append('points3D.bin')
    for fname in expected_files:
        output_file = output_model_dir / fname
        if output_file.exists():
            file_size = output_file.stat().st_size
            print_info(f"  {fname}: {file_size:,} bytes")
        else:
            print_error(f"  {fname}: 文件不存在")
    
    # 检查是否有rigs.bin和frames.bin（应该不存在）
    if (output_model_dir / "rigs.bin").exists():
        print_warning("检测到rigs.bin，将删除...")
        (output_model_dir / "rigs.bin").unlink()
    
    if (output_model_dir / "frames.bin").exists():
        print_warning("检测到frames.bin，将删除...")
        (output_model_dir / "frames.bin").unlink()
    
    print_success("转换完成！")


def main():
    parser = argparse.ArgumentParser(
        description="将COLMAP Rig格式转换为标准格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ConvertRig2Normal.py /path/to/rig_dataset /path/to/output_dataset
  
输入目录结构:
  input/
    ├── images/
    │   ├── pano_camera0/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── pano_camera1/
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.bin
            ├── images.bin
            ├── points3D.bin
            ├── rigs.bin
            └── frames.bin

输出目录结构:
  output/
    ├── images/
    │   ├── pano_camera0_image1.jpg
    │   └── ...
    └── sparse/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin (可选，使用--include-points3d时包含)
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='输入的rig格式COLMAP数据集路径'
    )
    
    parser.add_argument(
        'output_path',
        type=str,
        help='输出的标准格式COLMAP数据集路径'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='如果输出目录已存在，是否覆盖'
    )
    
    parser.add_argument(
        '--include-points3d',
        action='store_true',
        help='是否包含3D点信息（points3D.bin），默认不包含，只保留cameras.bin和images.bin'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # 检查输入路径
    if not input_path.exists():
        print_error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    input_images_dir = input_path / "images"
    input_sparse_dir = input_path / "sparse"
    
    if not input_images_dir.exists():
        print_error(f"输入images目录不存在: {input_images_dir}")
        sys.exit(1)
    
    if not input_sparse_dir.exists():
        print_error(f"输入sparse目录不存在: {input_sparse_dir}")
        sys.exit(1)
    
    # 检查输出路径
    if output_path.exists():
        if args.overwrite:
            print_warning(f"输出路径已存在，将覆盖: {output_path}")
            shutil.rmtree(output_path)
        else:
            print_error(f"输出路径已存在: {output_path}")
            print_error("使用 --overwrite 参数来覆盖现有目录")
            sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_path / "images"
    output_sparse_dir = output_path / "sparse"
    
    print_info("=" * 60)
    print_info("COLMAP Rig格式转标准格式转换工具")
    print_info("=" * 60)
    print_info(f"输入路径: {input_path}")
    print_info(f"输出路径: {output_path}")
    print_info("=" * 60)
    
    # 步骤1: 扁平化图像目录
    print_info("\n步骤1: 扁平化图像目录...")
    path_mapping = copy_images_flat(input_images_dir, output_images_dir)
    
    if not path_mapping:
        print_error("未找到任何图像文件")
        sys.exit(1)
    
    # 步骤2: 转换sparse重建
    print_info("\n步骤2: 转换sparse重建...")
    convert_sparse_reconstruction(input_sparse_dir, output_sparse_dir, path_mapping, args.include_points3d)
    
    print_info("\n" + "=" * 60)
    print_success("转换完成！")
    print_info("=" * 60)
    print_info(f"输出目录: {output_path}")
    print_info(f"  - images/: {len(list(output_images_dir.glob('*.*')))} 张图像")
    if args.include_points3d:
        print_info(f"  - sparse/: 包含 cameras.bin, images.bin, points3D.bin")
    else:
        print_info(f"  - sparse/: 包含 cameras.bin, images.bin")
    print_info("=" * 60)


if __name__ == "__main__":
    main()
