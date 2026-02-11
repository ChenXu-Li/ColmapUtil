#!/usr/bin/env python3
"""
可视化 COLMAP rig 相机组的位置和旋转，以及对应的点云
"""

import numpy as np
import viser
import viser.transforms as viser_tf
import pycolmap
import argparse
import socket
import sys
from pathlib import Path
from plyfile import PlyData, PlyElement

def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0

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

def add_coordinate_axes(server, name_prefix, position, rotation_matrix, axis_length=0.5, line_width=3.0):
    """
    添加三色坐标轴到场景中（使用spline线条绘制）
    Args:
        server: viser.ViserServer 对象
        name_prefix: 坐标轴名称前缀
        position: (3,) 位置向量
        rotation_matrix: (3, 3) 旋转矩阵（世界坐标系中的方向）
        axis_length: 坐标轴长度
        line_width: 线条宽度
    Returns:
        axes: dict, {'x': spline_x, 'y': spline_y, 'z': spline_z} 返回三个spline对象的引用
    """
    # 计算三个坐标轴的终点（在世界坐标系中）
    # X轴（红色）：rig坐标系中的 [1, 0, 0] 转换到世界坐标系
    x_axis_end = position + rotation_matrix @ np.array([axis_length, 0, 0])
    # Y轴（绿色）：rig坐标系中的 [0, 1, 0] 转换到世界坐标系
    y_axis_end = position + rotation_matrix @ np.array([0, axis_length, 0])
    # Z轴（蓝色）：rig坐标系中的 [0, 0, 1] 转换到世界坐标系
    z_axis_end = position + rotation_matrix @ np.array([0, 0, axis_length])
    
    # 使用spline绘制直线（只需要起点和终点，tension=0使其为直线）
    # X轴（红色）
    spline_x = server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_x",
        positions=np.array([position, x_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(255, 0, 0),  # 红色
    )
    
    # Y轴（绿色）
    spline_y = server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_y",
        positions=np.array([position, y_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(0, 255, 0),  # 绿色
    )
    
    # Z轴（蓝色）
    spline_z = server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_z",
        positions=np.array([position, z_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(0, 0, 255),  # 蓝色
    )
    
    return {'x': spline_x, 'y': spline_y, 'z': spline_z}

def main():
    parser = argparse.ArgumentParser(description="可视化 COLMAP rig 相机组的位置和旋转，以及对应的点云")
    parser.add_argument("--scene", type=str, default="BridgeB", 
                       help="场景名称（如 BridgeB, RoofTop, BridgeA 等）")
    parser.add_argument("--stage_dir", type=str, default="/root/autodl-tmp/data/STAGE1_4x",
                       help="STAGE数据集根目录")
    parser.add_argument("--colmap_dir", type=str, default="/root/autodl-tmp/data/colmap_STAGE1_4x",
                       help="colmap_STAGE数据集根目录")
    parser.add_argument("--port", type=int, default=8080,
                       help="Viser服务器端口（默认8080）")
    parser.add_argument("--axis_length", type=float, default=0.3,
                       help="坐标轴长度（默认0.3米）")
    parser.add_argument("--axis_width", type=float, default=3.0,
                       help="坐标轴线条宽度（默认3.0）")
    parser.add_argument("--hide_points", action="store_true",
                       help="隐藏COLMAP稀疏点云（默认显示）")
    parser.add_argument("--hide_cameras", action="store_true",
                       help="隐藏相机位置（默认显示）")
    parser.add_argument("--hide_ply", action="store_true",
                       help="隐藏点云文件（默认显示）")
    parser.add_argument("--camera_scale", type=float, default=0.05,
                       help="相机frustum的缩放比例（默认0.05）")
    parser.add_argument("--point_size", type=float, default=0.005,
                       help="点云点的大小（默认0.005）")
    parser.add_argument("--camera_name", type=str, default="pano_camera12",
                       help="点云所在的虚拟相机名称（默认：pano_camera12，用于计算cam_from_rig变换）")
    
    args = parser.parse_args()
    
    # DAP点云在camera12坐标系中的坐标轴修正
    # 点云的x轴 → camera12的-z轴
    # 点云的y轴 → camera12的x轴
    # 点云的z轴 → camera12的-y轴
    # 转换矩阵: [new_x, new_y, new_z]^T = [[0, 0, -1], [1, 0, 0], [0, -1, 0]] @ [old_x, old_y, old_z]^T
    camera_coord_correction_matrix = np.array([
        [0,  1, 0],  # new_x = -old_z
        [0,  0,  -1],  # new_y = old_x
        [-1, 0,  0]   # new_z = -old_y
    ], dtype=np.float32)
    print("ℹ️  应用camera坐标系修正：点云x→camera12的-z, y→x, z→-y")
    
    # 构建路径
    stage_dir = Path(args.stage_dir)
    colmap_dir = Path(args.colmap_dir)
    scene_stage_dir = stage_dir / args.scene
    scene_colmap_dir = colmap_dir / args.scene
    pointcloud_dir = scene_stage_dir / "pointclouds"
    colmap_sparse_dir = scene_colmap_dir / "sparse" / "0"
    
    # 检查目录是否存在
    if not pointcloud_dir.exists():
        print(f"❌ 点云目录不存在: {pointcloud_dir}")
        sys.exit(1)
    
    if not colmap_sparse_dir.exists():
        print(f"❌ COLMAP模型目录不存在: {colmap_sparse_dir}")
        sys.exit(1)
    
    print(f"📖 读取COLMAP重建结果: {colmap_sparse_dir}")
    try:
        recon = pycolmap.Reconstruction(str(colmap_sparse_dir))
    except Exception as e:
        print(f"❌ 无法读取COLMAP重建结果: {e}")
        sys.exit(1)
    
    # 检查端口并启动服务器
    port = args.port
    if check_port(port):
        print(f"⚠️  端口 {port} 已被占用，尝试使用 {port + 1}...")
        port = port + 1
    
    print(f"🚀 启动Viser服务器，端口: {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    
    # 加载COLMAP稀疏点云
    print("📊 加载COLMAP稀疏点云...")
    points = []
    colors = []
    for p in recon.points3D.values():
        xyz = np.array(p.xyz)
        if xyz.shape != (3,):
            xyz = xyz.flatten()[:3]
        points.append(xyz)
        
        color = np.array(p.color) / 255.0
        if color.shape != (3,):
            color = color.flatten()[:3]
        colors.append(color)
    
    if len(points) > 0:
        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        if len(points.shape) != 2 or points.shape[1] != 3:
            points = points.reshape(-1, 3)
        if len(colors.shape) != 2 or colors.shape[1] != 3:
            colors = colors.reshape(-1, 3)
        
        if not args.hide_points:
            server.scene.add_point_cloud(
                name="colmap_points",
                points=points,
                colors=colors,
                point_size=0.01,
            )
        print(f"   ✅ 加载了 {len(points)} 个点")
    else:
        print("   ⚠️  没有找到点云")
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float32)
    
    # 加载相机
    print("📷 加载相机位置...")
    camera_count = 0
    for image_id, image in recon.images.items():
        try:
            cam = recon.cameras[image.camera_id]
            
            # 世界坐标系下相机位姿
            cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            T_wc = np.eye(4)
            T_wc[:3, :3] = R.T
            T_wc[:3, 3] = -R.T @ t
            
            # Convert to viser SE3 format (3x4 matrix)
            T_wc_3x4 = T_wc[:3, :]
            T_world_camera = viser_tf.SE3.from_matrix(T_wc_3x4)
            
            if not args.hide_cameras:
                server.scene.add_camera_frustum(
                    name=f"cam_{image_id}",
                    fov=cam.params[0],
                    aspect=cam.width / cam.height,
                    scale=args.camera_scale,
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                )
            camera_count += 1
        except Exception as e:
            print(f"⚠️  处理相机 {image_id} 时出错: {e}")
            continue
    
    print(f"   ✅ 加载了 {camera_count} 个相机")
    
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
    
    # 获取所有点云文件
    ply_files = sorted(pointcloud_dir.glob("*.ply"))
    if len(ply_files) == 0:
        print(f"⚠️  点云目录中没有找到PLY文件: {pointcloud_dir}")
    else:
        print(f"📁 找到 {len(ply_files)} 个点云文件")
    
    # 存储点云数据和rig信息
    pointcloud_data = {}  # {pano_name: {'points': ..., 'colors': ..., 'frame_id': ...}}
    rig_origins = {}  # {pano_name: {'position': ..., 'rotation': ..., 'frame_id': ...}}
    camera_origins = {}  # {pano_name: {'position': ..., 'rotation': ..., 'frame_id': ..., 'camera_name': ...}}
    rig_axes_objects = {}  # {frame_id: {'x': spline_x, 'y': spline_y, 'z': spline_z}}
    camera_axes_objects = {}  # {f"{camera_name}_{frame_id}": {'x': spline_x, 'y': spline_y, 'z': spline_z}}
    rig_positions = []
    
    # 处理每个点云文件
    print("\n📦 加载点云和 rig 原点...")
    for ply_path in ply_files:
        try:
            pano_name = ply_path.stem
            
            # 查找对应的frame
            if pano_name not in pano_to_frame:
                print(f"⚠️  跳过 {ply_path.name}: 在COLMAP中找不到对应的frame")
                continue
            
            frame_id = pano_to_frame[pano_name]
            if frame_id not in frames_with_pose:
                print(f"⚠️  跳过 {ply_path.name}: frame {frame_id} 没有pose")
                continue
            
            frame = frames_with_pose[frame_id]
            rig_from_world = frame.rig_from_world
            
            # 计算 rig 在世界坐标系中的位置和旋转
            world_from_rig = rig_from_world.inverse()
            rig_position = world_from_rig.translation
            rig_rotation = world_from_rig.rotation.matrix()  # (3, 3)
            
            # 获取指定相机的cam_from_rig变换（DAP点云是在该相机坐标系中生成的）
            cam_from_rig = None
            camera_found = False
            camera_position = None
            camera_rotation = None
            camera_image_id = None
            
            for img_id, img in recon.images.items():
                if img.frame_id == frame_id and args.camera_name in img.name:
                    # 获取相机的cam_from_world
                    cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                    # 计算cam_from_rig: cam_from_world = cam_from_rig @ rig_from_world
                    # 所以: cam_from_rig = cam_from_world @ world_from_rig
                    cam_from_rig = cam_from_world * world_from_rig
                    
                    # 计算相机在世界坐标系中的位置和旋转
                    # cam_from_world 表示从世界坐标系到相机坐标系的变换
                    # 我们需要 world_from_cam 来获取相机在世界坐标系中的位姿
                    world_from_cam = cam_from_world.inverse()
                    camera_position = world_from_cam.translation
                    camera_rotation = world_from_cam.rotation.matrix()  # (3, 3)
                    camera_image_id = img_id
                    camera_found = True
                    print(f"   📷 找到{args.camera_name}，cam_from_rig变换已获取")
                    break
            
            if not camera_found:
                print(f"   ⚠️  警告: 未找到{args.camera_name}，假设点云在rig坐标系中")
            
            # 加载点云
            points_local, colors_ply = load_ply(ply_path)
            
            # 应用坐标变换（从局部坐标系到全局坐标系）
            # 1. 先在camera坐标系中应用坐标轴修正（点云x→camera12的-z, y→x, z→-y）
            # 2. 如果cam_from_rig不为None，将点云从camera坐标系转换到rig坐标系
            # 3. 最后转换到世界坐标系
            points_world = transform_points(
                points_local, 
                rig_from_world, 
                cam_from_rig=cam_from_rig,
                camera_coord_correction=camera_coord_correction_matrix
            )
            
            # 存储点云数据（同时保存局部坐标系点云，以便后续重新转换）
            pointcloud_data[pano_name] = {
                'points': points_world,
                'colors': colors_ply,
                'frame_id': frame_id,
                'points_local': points_local,  # 保存原始局部坐标系点云（camera12坐标系）
                'rig_from_world': rig_from_world,  # 保存变换，以便重新转换
                'cam_from_rig': cam_from_rig,  # 保存camera12相对于rig的变换
            }
            
            # 存储 rig 原点位置和旋转
            rig_origins[pano_name] = {
                'position': rig_position,
                'rotation': rig_rotation,
                'frame_id': frame_id,
            }
            
            # 存储相机原点位置和旋转（如果找到了相机）
            if camera_found and camera_position is not None:
                camera_origins[pano_name] = {
                    'position': camera_position,
                    'rotation': camera_rotation,
                    'frame_id': frame_id,
                    'camera_name': args.camera_name,
                    'image_id': camera_image_id,
                }
            
            rig_positions.append(rig_position)
            
            print(f"   ✅ {pano_name}: {len(points_world):,} 点, rig位置: [{rig_position[0]:.2f}, {rig_position[1]:.2f}, {rig_position[2]:.2f}]")
            
        except Exception as e:
            print(f"❌ 处理 {ply_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(pointcloud_data) == 0:
        print("⚠️  没有成功加载任何点云文件")
    else:
        print(f"\n✅ 成功加载 {len(pointcloud_data)} 个点云")
    
    # 添加 rig 坐标轴（在rig位置）
    print("\n📍 添加 rig 坐标轴...")
    for pano_name, rig_info in rig_origins.items():
        position = rig_info['position']
        rotation = rig_info['rotation']
        frame_id = rig_info['frame_id']
        
        axes = add_coordinate_axes(
            server=server,
            name_prefix=f"rig_{frame_id}",
            position=position,
            rotation_matrix=rotation,
            axis_length=args.axis_length,
            line_width=args.axis_width
        )
        rig_axes_objects[frame_id] = axes
    
    print(f"   ✅ 添加了 {len(rig_origins)} 个 rig 坐标轴")
    
    # 添加相机坐标轴（在camera位置）
    print("\n📍 添加相机坐标轴...")
    for pano_name, camera_info in camera_origins.items():
        position = camera_info['position']
        rotation = camera_info['rotation']
        camera_name = camera_info['camera_name']
        frame_id = camera_info['frame_id']
        
        axes = add_coordinate_axes(
            server=server,
            name_prefix=f"camera_{camera_name}_{frame_id}",
            position=position,
            rotation_matrix=rotation,
            axis_length=args.axis_length,
            line_width=args.axis_width
        )
        camera_axes_objects[f"{camera_name}_{frame_id}"] = axes
    
    print(f"   ✅ 添加了 {len(camera_origins)} 个相机坐标轴")
    
    # 添加点云（如果启用）
    if not args.hide_ply and len(pointcloud_data) > 0:
        print("\n☁️  添加点云...")
        for pano_name, pc_data in pointcloud_data.items():
            server.scene.add_point_cloud(
                name=f"/pointcloud_{pano_name}",
                points=pc_data['points'],
                colors=pc_data['colors'],
                point_size=args.point_size,
            )
        print(f"   ✅ 添加了 {len(pointcloud_data)} 个点云")
    
    if len(rig_positions) == 0:
        print("❌ 没有成功处理任何rig")
        sys.exit(1)
    
    rig_positions = np.array(rig_positions)
    
    # 计算rig位置范围
    pos_min = rig_positions.min(axis=0)
    pos_max = rig_positions.max(axis=0)
    pos_center = rig_positions.mean(axis=0)
    pos_range = pos_max - pos_min
    
    print(f"\n📊 Rig位置统计:")
    print(f"   数量: {len(rig_positions)}")
    print(f"   中心: [{pos_center[0]:.2f}, {pos_center[1]:.2f}, {pos_center[2]:.2f}]")
    print(f"   范围: X[{pos_min[0]:.2f}, {pos_max[0]:.2f}], "
          f"Y[{pos_min[1]:.2f}, {pos_max[1]:.2f}], "
          f"Z[{pos_min[2]:.2f}, {pos_max[2]:.2f}]")
    
    # 添加GUI控件
    with server.gui.add_folder("Rig Visualization Control"):
        axis_length_slider = server.gui.add_slider(
            "Axis Length",
            min=0.1,
            max=2.0,
            step=0.1,
            initial_value=args.axis_length
        )
        
        axis_width_slider = server.gui.add_slider(
            "Axis Width",
            min=1.0,
            max=10.0,
            step=0.5,
            initial_value=args.axis_width
        )
        
        show_points_toggle = server.gui.add_checkbox(
            "Show Points",
            initial_value=not args.hide_points
        )
        
        show_cameras_toggle = server.gui.add_checkbox(
            "Show Cameras",
            initial_value=not args.hide_cameras
        )
        
        show_ply_toggle = server.gui.add_checkbox(
            "Show Point Clouds",
            initial_value=not args.hide_ply
        )
        
        show_rig_axes_toggle = server.gui.add_checkbox(
            "Show Rig Axes",
            initial_value=True
        )
        
        show_camera_axes_toggle = server.gui.add_checkbox(
            f"Show {args.camera_name} Axes",
            initial_value=True
        )
        
        camera_scale_slider = server.gui.add_slider(
            "Camera Scale",
            min=0.01,
            max=0.2,
            step=0.01,
            initial_value=args.camera_scale
        )
        
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.02,
            step=0.001,
            initial_value=args.point_size
        )
        
        center_view_btn = server.gui.add_button("Center View")
        top_view_btn = server.gui.add_button("Top View")
        side_view_btn = server.gui.add_button("Side View")
        
        # 为每个点云创建checkbox
        ply_checkboxes = {}
        for pano_name in sorted(pointcloud_data.keys()):
            checkbox = server.gui.add_checkbox(
                f"Show: {pano_name}",
                initial_value=not args.hide_ply
            )
            ply_checkboxes[pano_name] = checkbox
            
            def make_checkbox_handler(pano_name_inner):
                def handler(_):
                    checkbox_inner = ply_checkboxes[pano_name_inner]
                    if checkbox_inner.value:
                        pc_data = pointcloud_data[pano_name_inner]
                        server.scene.add_point_cloud(
                            name=f"/pointcloud_{pano_name_inner}",
                            points=pc_data['points'],
                            colors=pc_data['colors'],
                            point_size=point_size_slider.value,
                        )
                    else:
                        try:
                            server.scene.remove(f"/pointcloud_{pano_name_inner}")
                        except:
                            pass
                return handler
            
            checkbox.on_update(make_checkbox_handler(pano_name))
        
        @axis_length_slider.on_update
        def update_axis_length(_):
            """更新所有坐标轴的长度"""
            new_length = axis_length_slider.value
            new_width = axis_width_slider.value
            
            # 更新rig坐标轴
            if show_rig_axes_toggle.value:
                for pano_name, rig_info in rig_origins.items():
                    try:
                        frame_id = rig_info['frame_id']
                        # 删除旧的坐标轴
                        for axis_name in ['_axis_x', '_axis_y', '_axis_z']:
                            try:
                                server.scene.remove(f"/rig_{frame_id}{axis_name}")
                            except:
                                pass
                        
                        # 添加新的坐标轴
                        add_coordinate_axes(
                            server=server,
                            name_prefix=f"rig_{frame_id}",
                            position=rig_info['position'],
                            rotation_matrix=rig_info['rotation'],
                            axis_length=new_length,
                            line_width=new_width
                        )
                    except:
                        continue
            
            # 更新相机坐标轴
            if show_camera_axes_toggle.value:
                for pano_name, camera_info in camera_origins.items():
                    try:
                        camera_name = camera_info['camera_name']
                        frame_id = camera_info['frame_id']
                        # 删除旧的坐标轴
                        for axis_name in ['_axis_x', '_axis_y', '_axis_z']:
                            try:
                                server.scene.remove(f"/camera_{camera_name}_{frame_id}{axis_name}")
                            except:
                                pass
                        
                        # 添加新的坐标轴
                        add_coordinate_axes(
                            server=server,
                            name_prefix=f"camera_{camera_name}_{frame_id}",
                            position=camera_info['position'],
                            rotation_matrix=camera_info['rotation'],
                            axis_length=new_length,
                            line_width=new_width
                        )
                    except:
                        continue
        
        @axis_width_slider.on_update
        def update_axis_width(_):
            """更新所有坐标轴的宽度"""
            new_length = axis_length_slider.value
            new_width = axis_width_slider.value
            
            # 更新rig坐标轴
            if show_rig_axes_toggle.value:
                for pano_name, rig_info in rig_origins.items():
                    try:
                        frame_id = rig_info['frame_id']
                        # 删除旧的坐标轴
                        for axis_name in ['_axis_x', '_axis_y', '_axis_z']:
                            try:
                                server.scene.remove(f"/rig_{frame_id}{axis_name}")
                            except:
                                pass
                        
                        # 添加新的坐标轴
                        add_coordinate_axes(
                            server=server,
                            name_prefix=f"rig_{frame_id}",
                            position=rig_info['position'],
                            rotation_matrix=rig_info['rotation'],
                            axis_length=new_length,
                            line_width=new_width
                        )
                    except:
                        continue
            
            # 更新相机坐标轴
            if show_camera_axes_toggle.value:
                for pano_name, camera_info in camera_origins.items():
                    try:
                        camera_name = camera_info['camera_name']
                        frame_id = camera_info['frame_id']
                        # 删除旧的坐标轴
                        for axis_name in ['_axis_x', '_axis_y', '_axis_z']:
                            try:
                                server.scene.remove(f"/camera_{camera_name}_{frame_id}{axis_name}")
                            except:
                                pass
                        
                        # 添加新的坐标轴
                        add_coordinate_axes(
                            server=server,
                            name_prefix=f"camera_{camera_name}_{frame_id}",
                            position=camera_info['position'],
                            rotation_matrix=camera_info['rotation'],
                            axis_length=new_length,
                            line_width=new_width
                        )
                    except:
                        continue
        
        @show_ply_toggle.on_update
        def toggle_ply(_):
            """切换所有点云显示"""
            if show_ply_toggle.value:
                for pano_name, pc_data in pointcloud_data.items():
                    server.scene.add_point_cloud(
                        name=f"/pointcloud_{pano_name}",
                        points=pc_data['points'],
                        colors=pc_data['colors'],
                        point_size=point_size_slider.value,
                    )
            else:
                for pano_name in pointcloud_data.keys():
                    try:
                        server.scene.remove(f"/pointcloud_{pano_name}")
                    except:
                        pass
        
        @point_size_slider.on_update
        def update_point_size(_):
            """更新点云点的大小"""
            if show_ply_toggle.value:
                for pano_name, pc_data in pointcloud_data.items():
                    try:
                        server.scene.remove(f"/pointcloud_{pano_name}")
                    except:
                        pass
                    server.scene.add_point_cloud(
                        name=f"/pointcloud_{pano_name}",
                        points=pc_data['points'],
                        colors=pc_data['colors'],
                        point_size=point_size_slider.value,
                    )
        
        @show_points_toggle.on_update
        def toggle_points(_):
            """切换点云显示"""
            if show_points_toggle.value and len(points) > 0:
                server.scene.add_point_cloud(
                    name="colmap_points",
                    points=points,
                    colors=colors,
                    point_size=0.01,
                )
            else:
                try:
                    server.scene.remove("colmap_points")
                except:
                    pass
        
        @show_rig_axes_toggle.on_update
        def toggle_rig_axes(_):
            """切换rig坐标轴显示"""
            if show_rig_axes_toggle.value:
                # 显示坐标轴：设置visible=True
                for frame_id, axes in rig_axes_objects.items():
                    try:
                        axes['x'].visible = True
                        axes['y'].visible = True
                        axes['z'].visible = True
                    except Exception as e:
                        print(f"[DEBUG] Failed to show rig axes {frame_id}: {e}")
            else:
                # 隐藏坐标轴：设置visible=False
                for frame_id, axes in rig_axes_objects.items():
                    try:
                        axes['x'].visible = False
                        axes['y'].visible = False
                        axes['z'].visible = False
                    except Exception as e:
                        print(f"[DEBUG] Failed to hide rig axes {frame_id}: {e}")
        
        @show_camera_axes_toggle.on_update
        def toggle_camera_axes(_):
            """切换相机坐标轴显示"""
            if show_camera_axes_toggle.value:
                # 显示坐标轴：设置visible=True
                for key, axes in camera_axes_objects.items():
                    try:
                        axes['x'].visible = True
                        axes['y'].visible = True
                        axes['z'].visible = True
                    except Exception as e:
                        print(f"[DEBUG] Failed to show camera axes {key}: {e}")
            else:
                # 隐藏坐标轴：设置visible=False
                for key, axes in camera_axes_objects.items():
                    try:
                        axes['x'].visible = False
                        axes['y'].visible = False
                        axes['z'].visible = False
                    except Exception as e:
                        print(f"[DEBUG] Failed to hide camera axes {key}: {e}")
        
        @show_cameras_toggle.on_update
        def toggle_cameras(_):
            """切换相机显示"""
            if show_cameras_toggle.value:
                for image_id, image in recon.images.items():
                    try:
                        cam = recon.cameras[image.camera_id]
                        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                        R = cam_from_world.rotation.matrix()
                        t = cam_from_world.translation
                        T_wc = np.eye(4)
                        T_wc[:3, :3] = R.T
                        T_wc[:3, 3] = -R.T @ t
                        T_wc_3x4 = T_wc[:3, :]
                        T_world_camera = viser_tf.SE3.from_matrix(T_wc_3x4)
                        
                        server.scene.add_camera_frustum(
                            name=f"cam_{image_id}",
                            fov=cam.params[0],
                            aspect=cam.width / cam.height,
                            scale=camera_scale_slider.value,
                            wxyz=T_world_camera.rotation().wxyz,
                            position=T_world_camera.translation(),
                        )
                    except:
                        continue
            else:
                for image_id in recon.images.keys():
                    try:
                        server.scene.remove(f"cam_{image_id}")
                    except:
                        pass
        
        @camera_scale_slider.on_update
        def update_camera_scale(_):
            """更新相机缩放"""
            if show_cameras_toggle.value:
                for image_id, image in recon.images.items():
                    try:
                        cam = recon.cameras[image.camera_id]
                        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                        R = cam_from_world.rotation.matrix()
                        t = cam_from_world.translation
                        T_wc = np.eye(4)
                        T_wc[:3, :3] = R.T
                        T_wc[:3, 3] = -R.T @ t
                        T_wc_3x4 = T_wc[:3, :]
                        T_world_camera = viser_tf.SE3.from_matrix(T_wc_3x4)
                        
                        try:
                            server.scene.remove(f"cam_{image_id}")
                        except:
                            pass
                        
                        server.scene.add_camera_frustum(
                            name=f"cam_{image_id}",
                            fov=cam.params[0],
                            aspect=cam.width / cam.height,
                            scale=camera_scale_slider.value,
                            wxyz=T_world_camera.rotation().wxyz,
                            position=T_world_camera.translation(),
                        )
                    except:
                        continue
        
        @center_view_btn.on_click
        def center_view(_):
            """居中视图"""
            if len(rig_positions) > 0:
                position = pos_center + np.array([0, 0, max(pos_range) * 0.5])
            elif len(points) > 0:
                position = points.mean(axis=0) + np.array([0, 0, 2.0])
            else:
                position = np.array([0.0, 0.0, 2.0])
            wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.wxyz = wxyz
        
        @top_view_btn.on_click
        def top_view(_):
            """俯视图"""
            if len(rig_positions) > 0:
                position = pos_center + np.array([0, 0, max(pos_range) * 1.2])
            elif len(points) > 0:
                position = points.mean(axis=0) + np.array([0, 0, 5.0])
            else:
                position = np.array([0.0, 0.0, 5.0])
            wxyz = np.array([0.707, 0.707, 0.0, 0.0])
            
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.wxyz = wxyz
        
        @side_view_btn.on_click
        def side_view(_):
            """侧视图"""
            if len(rig_positions) > 0:
                position = pos_center + np.array([max(pos_range) * 1.2, 0, max(pos_range) * 0.3])
            elif len(points) > 0:
                center = points.mean(axis=0)
                range_val = points.max(axis=0) - points.min(axis=0)
                position = center + np.array([max(range_val) * 1.2, 0, max(range_val) * 0.3])
            else:
                position = np.array([5.0, 0.0, 1.0])
            wxyz = np.array([0.707, 0.0, 0.707, 0.0])
            
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.wxyz = wxyz
    
    print(f"\n✅ Viser服务器运行中!")
    print(f"🌐 在浏览器中打开: http://<server-ip>:{port}")
    print(f"\n📋 说明:")
    print(f"   - 红色轴 = X轴，绿色轴 = Y轴，蓝色轴 = Z轴")
    print(f"   - Rig坐标轴：表示rig的位置和旋转方向（点云的rig坐标系原点）")
    print(f"   - {args.camera_name}坐标轴：表示{args.camera_name}的位置和旋转方向（点云的相机坐标系原点）")
    print(f"   - 点云已转换到全局坐标系显示")
    print(f"\n🎛️  GUI控件:")
    print(f"   - Axis Length/Width: 调整坐标轴长度和宽度")
    print(f"   - Show Rig Axes: 切换rig坐标轴显示")
    print(f"   - Show {args.camera_name} Axes: 切换{args.camera_name}坐标轴显示")
    print(f"   - Show Points: 切换COLMAP稀疏点云显示")
    print(f"   - Show Cameras: 切换相机显示")
    print(f"   - Show Point Clouds: 切换所有点云文件显示")
    print(f"   - Point Size: 调整点云点的大小")
    print(f"   - 每个点云都有独立的checkbox控制显示/隐藏")
    print(f"\n按 Ctrl+C 停止服务器")
    
    # 保持服务器运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == "__main__":
    main()

