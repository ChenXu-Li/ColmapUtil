import numpy as np
import viser
import viser.transforms as viser_tf
import pycolmap
import argparse
import socket
import sys
from pathlib import Path

def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0

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
    server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_x",
        positions=np.array([position, x_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(255, 0, 0),  # 红色
    )
    
    # Y轴（绿色）
    server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_y",
        positions=np.array([position, y_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(0, 255, 0),  # 绿色
    )
    
    # Z轴（蓝色）
    server.scene.add_spline_catmull_rom(
        name=f"/{name_prefix}_axis_z",
        positions=np.array([position, z_axis_end]),
        curve_type='chordal',
        tension=0.0,  # tension=0 使曲线变为直线
        line_width=line_width,
        color=(0, 0, 255),  # 蓝色
    )

def main():
    parser = argparse.ArgumentParser(description="可视化 COLMAP rig 相机组的位置和旋转")
    parser.add_argument("--scene", type=str, default="BridgeB", 
                       help="场景名称（如 BridgeB, RoofTop, BridgeA 等）")
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
    parser.add_argument("--camera_scale", type=float, default=0.05,
                       help="相机frustum的缩放比例（默认0.05）")
    
    args = parser.parse_args()
    
    # 构建COLMAP模型路径
    colmap_dir = Path(args.colmap_dir)
    colmap_model_path = colmap_dir / args.scene / "sparse" / "0"
    
    if not colmap_model_path.exists():
        print(f"❌ COLMAP模型目录不存在: {colmap_model_path}")
        sys.exit(1)
    
    print(f"📖 读取COLMAP重建结果: {colmap_model_path}")
    try:
        recon = pycolmap.Reconstruction(str(colmap_model_path))
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
            # cam_from_world gives camera pose in world coordinates (camera from world)
            # We need world from camera for visualization
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
    
    # 计算所有rig位置的中心（用于显示）
    rig_positions = []
    
    # 遍历所有rigs并添加坐标轴
    for frame_id, frame in frames_with_pose.items():
        try:
            # 获取 rig_from_world 变换
            rig_from_world = frame.rig_from_world
            
            # 计算 world_from_rig（rig在世界坐标系中的位姿）
            if hasattr(rig_from_world, 'inverse'):
                world_from_rig = rig_from_world.inverse()
                R_world_rig = world_from_rig.rotation.matrix()  # (3, 3)
                t_world_rig = world_from_rig.translation  # (3,)
            else:
                # 手动计算inverse
                R = rig_from_world.rotation.matrix()
                t = rig_from_world.translation
                R_world_rig = R.T
                t_world_rig = -R.T @ t
            
            # rig在世界坐标系中的位置
            rig_position = t_world_rig
            rig_positions.append(rig_position)
            
            # 添加坐标轴可视化
            add_coordinate_axes(
                server=server,
                name_prefix=f"rig_{frame_id}",
                position=rig_position,
                rotation_matrix=R_world_rig,
                axis_length=args.axis_length,
                line_width=args.axis_width
            )
            
        except Exception as e:
            print(f"⚠️  处理 rig {frame_id} 时出错: {e}")
            continue
    
    if len(rig_positions) == 0:
        print("❌ 没有成功处理任何rig")
        sys.exit(1)
    
    rig_positions = np.array(rig_positions)
    print(f"✅ 成功可视化 {len(rig_positions)} 个rigs")
    
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
        
        camera_scale_slider = server.gui.add_slider(
            "Camera Scale",
            min=0.01,
            max=0.2,
            step=0.01,
            initial_value=args.camera_scale
        )
        
        center_view_btn = server.gui.add_button("Center View")
        top_view_btn = server.gui.add_button("Top View")
        side_view_btn = server.gui.add_button("Side View")
        
        @axis_length_slider.on_update
        def update_axis_length(_):
            """更新所有坐标轴的长度"""
            new_length = axis_length_slider.value
            new_width = axis_width_slider.value
            for frame_id, frame in frames_with_pose.items():
                try:
                    rig_from_world = frame.rig_from_world
                    if hasattr(rig_from_world, 'inverse'):
                        world_from_rig = rig_from_world.inverse()
                        R_world_rig = world_from_rig.rotation.matrix()
                        t_world_rig = world_from_rig.translation
                    else:
                        R = rig_from_world.rotation.matrix()
                        t = rig_from_world.translation
                        R_world_rig = R.T
                        t_world_rig = -R.T @ t
                    
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
                        position=t_world_rig,
                        rotation_matrix=R_world_rig,
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
            for frame_id, frame in frames_with_pose.items():
                try:
                    rig_from_world = frame.rig_from_world
                    if hasattr(rig_from_world, 'inverse'):
                        world_from_rig = rig_from_world.inverse()
                        R_world_rig = world_from_rig.rotation.matrix()
                        t_world_rig = world_from_rig.translation
                    else:
                        R = rig_from_world.rotation.matrix()
                        t = rig_from_world.translation
                        R_world_rig = R.T
                        t_world_rig = -R.T @ t
                    
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
                        position=t_world_rig,
                        rotation_matrix=R_world_rig,
                        axis_length=new_length,
                        line_width=new_width
                    )
                except:
                    continue
        
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
        
        @show_cameras_toggle.on_update
        def toggle_cameras(_):
            """切换相机显示"""
            if show_cameras_toggle.value:
                # 重新添加所有相机
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
                # 移除所有相机
                for image_id in recon.images.keys():
                    try:
                        server.scene.remove(f"cam_{image_id}")
                    except:
                        pass
        
        @camera_scale_slider.on_update
        def update_camera_scale(_):
            """更新相机缩放"""
            if show_cameras_toggle.value:
                # 重新添加所有相机以应用新的缩放
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
                        
                        # 先删除旧的
                        try:
                            server.scene.remove(f"cam_{image_id}")
                        except:
                            pass
                        
                        # 添加新的
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
            wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # 默认朝向
            
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
            wxyz = np.array([0.707, 0.707, 0.0, 0.0])  # 向下看
            
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
            wxyz = np.array([0.707, 0.0, 0.707, 0.0])  # 从侧面看
            
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.wxyz = wxyz
    
    print(f"\n✅ Viser服务器运行中!")
    print(f"🌐 在浏览器中打开: http://<server-ip>:{port}")
    print(f"\n📋 说明:")
    print(f"   - 红色轴 = X轴，绿色轴 = Y轴，蓝色轴 = Z轴")
    print(f"   - 每个坐标轴的原点表示rig的位置")
    print(f"   - 坐标轴方向表示rig的旋转方向")
    print(f"   - 相机frustum显示每个相机的位置和朝向")
    print(f"   - 稀疏点云显示COLMAP重建的3D点")
    print(f"\n🎛️  GUI控件:")
    print(f"   - Axis Length: 调整rig坐标轴长度")
    print(f"   - Show Points: 切换稀疏点云显示")
    print(f"   - Show Cameras: 切换相机显示")
    print(f"   - Camera Scale: 调整相机frustum大小")
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

