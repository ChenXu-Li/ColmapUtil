#!/usr/bin/env python3
"""
交互式置信度可视化工具

功能：
1. 加载点云和置信度文件
2. 滑块控制置信度阈值
3. 低于阈值的点用红色、更大的点可视化（表示将被删除）
4. 点击保存按钮保存裁切后的 PLY 文件
"""

import argparse
import os
import sys
import time
import numpy as np
import open3d as o3d
import viser
from pathlib import Path


def get_project_root():
    """获取 filter 项目根目录"""
    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(scripts_dir)
    return project_root


def check_port(port):
    """检查端口是否可用"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0


def load_confidence_info(info_file):
    """加载置信度信息文件"""
    info = {}
    with open(info_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                info[key] = value
    return info


def main():
    parser = argparse.ArgumentParser(
        description="交互式置信度可视化工具"
    )
    parser.add_argument(
        "--confidence_file",
        type=str,
        default=None,
        help="置信度文件路径（.npy），如果不指定则从 outputs/geometry_confidence.npy 读取"
    )
    parser.add_argument(
        "--pointcloud",
        type=str,
        default=None,
        help="点云文件路径（.ply），如果不指定则从 confidence_info.txt 读取"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Viser服务器端口（默认8091）"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.005,
        help="正常点的显示大小（默认0.005）"
    )
    parser.add_argument(
        "--low_confidence_point_size",
        type=float,
        default=0.02,
        help="低置信度点的显示大小（默认0.02，红色）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，如果不指定则使用 outputs/"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="filtered_by_confidence.ply",
        help="输出文件名（默认: filtered_by_confidence.ply）"
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = get_project_root()
    
    # 解析置信度文件路径
    if args.confidence_file is None:
        # 尝试从 outputs 目录读取
        outputs_dir = os.path.join(project_root, "outputs")
        confidence_file = os.path.join(outputs_dir, "geometry_confidence.npy")
        info_file = os.path.join(outputs_dir, "confidence_info.txt")
    else:
        confidence_file = args.confidence_file
        if not os.path.isabs(confidence_file):
            confidence_file = os.path.join(project_root, confidence_file)
        info_file = os.path.join(os.path.dirname(confidence_file), "confidence_info.txt")
    
    if not os.path.exists(confidence_file):
        print(f"❌ 置信度文件不存在: {confidence_file}")
        print("   请先运行 geometry_lof_confidence.py 生成置信度文件")
        sys.exit(1)
    
    # 加载置信度
    print(f"📖 加载置信度文件: {confidence_file}")
    confidence = np.load(confidence_file)
    print(f"   ✅ 加载了 {len(confidence)} 个点的置信度")
    print(f"   置信度范围: [{confidence.min():.4f}, {confidence.max():.4f}], 均值: {confidence.mean():.4f}")
    
    # 解析点云路径
    if args.pointcloud is None:
        # 尝试从 info_file 读取
        if os.path.exists(info_file):
            info = load_confidence_info(info_file)
            pointcloud_path = info.get("pointcloud_path", "")
            if pointcloud_path and os.path.exists(pointcloud_path):
                pass  # 使用 info 文件中的路径
            else:
                print(f"⚠️  info 文件中的点云路径不存在，请使用 --pointcloud 参数指定")
                sys.exit(1)
        else:
            print(f"❌ 无法找到点云路径，请使用 --pointcloud 参数指定")
            sys.exit(1)
    else:
        pointcloud_path = args.pointcloud
        if not os.path.isabs(pointcloud_path):
            pointcloud_path = os.path.join(project_root, pointcloud_path)
    
    if not os.path.exists(pointcloud_path):
        print(f"❌ 点云文件不存在: {pointcloud_path}")
        sys.exit(1)
    
    # 加载点云
    print(f"📖 加载点云文件: {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pcd.points)
    
    if len(points) != len(confidence):
        print(f"⚠️  警告: 点云点数 ({len(points)}) 与置信度数量 ({len(confidence)}) 不匹配")
        print("   可能是点云经过了预处理（如下采样），请确保使用原始点云或匹配的置信度文件")
        # 取较小的长度
        min_len = min(len(points), len(confidence))
        points = points[:min_len]
        confidence = confidence[:min_len]
        print(f"   已截断到 {min_len} 个点")
    
    print(f"   ✅ 加载了 {len(points)} 个点")
    
    # 解析输出目录
    if args.output_dir is None:
        output_dir = os.path.join(project_root, "outputs")
    else:
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(project_root, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查端口并启动服务器
    port = args.port
    if check_port(port):
        print(f"⚠️  端口 {port} 已被占用，尝试使用 {port + 1}...")
        port = port + 1
    
    print(f"🚀 启动Viser服务器，端口: {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    
    # 获取点云颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        # 如果没有颜色，使用默认颜色（浅灰色）
        colors = np.ones((len(points), 3), dtype=np.float32) * 0.7
    
    # 初始阈值（使用置信度的中位数）
    initial_threshold = float(np.median(confidence))
    
    # 添加正常点云（高置信度点）
    def update_visualization(threshold):
        """根据阈值更新可视化"""
        # 计算哪些点会被保留（高置信度）
        keep_mask = confidence >= threshold
        low_conf_mask = ~keep_mask
        
        # 移除旧的点云
        try:
            server.scene.remove("/points_high_confidence")
        except:
            pass
        try:
            server.scene.remove("/points_low_confidence")
        except:
            pass
        
        # 添加高置信度点（正常显示）
        if keep_mask.sum() > 0:
            server.scene.add_point_cloud(
                name="/points_high_confidence",
                points=points[keep_mask].astype(np.float32),
                colors=colors[keep_mask].astype(np.float32),
                point_size=args.point_size,
            )
        
        # 添加低置信度点（红色，更大）
        if low_conf_mask.sum() > 0:
            low_conf_colors = np.ones((low_conf_mask.sum(), 3), dtype=np.float32)
            low_conf_colors[:, 0] = 1.0  # 红色
            low_conf_colors[:, 1] = 0.0
            low_conf_colors[:, 2] = 0.0
            
            server.scene.add_point_cloud(
                name="/points_low_confidence",
                points=points[low_conf_mask].astype(np.float32),
                colors=low_conf_colors,
                point_size=args.low_confidence_point_size,
            )
        
        return keep_mask.sum(), low_conf_mask.sum()
    
    # 初始可视化
    num_keep, num_remove = update_visualization(initial_threshold)
    
    # 添加GUI控件
    with server.gui.add_folder("Confidence Filter Control"):
        threshold_slider = server.gui.add_slider(
            "Confidence Threshold",
            min=float(confidence.min()),
            max=float(confidence.max()),
            step=0.01,
            initial_value=initial_threshold
        )
        
        initial_stats = (
            f"Points to keep: {num_keep} / {len(points)} ({100*num_keep/len(points):.1f}%)\n"
            f"Points to remove: {num_remove} / {len(points)} ({100*num_remove/len(points):.1f}%)"
        )
        stats_text = server.gui.add_text(
            "Statistics",
            initial_value=initial_stats
        )
        
        save_button = server.gui.add_button("Save Filtered Point Cloud")
    
    # 阈值滑块更新回调
    @threshold_slider.on_update
    def update_threshold(_):
        threshold = threshold_slider.value
        num_keep, num_remove = update_visualization(threshold)
        stats_text.value = (
            f"Points to keep: {num_keep} / {len(points)} ({100*num_keep/len(points):.1f}%)\n"
            f"Points to remove: {num_remove} / {len(points)} ({100*num_remove/len(points):.1f}%)"
        )
    
    # 保存按钮回调
    @save_button.on_click
    def save_filtered(_):
        threshold = threshold_slider.value
        keep_mask = confidence >= threshold
        
        # 创建过滤后的点云
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_mask])
        
        if pcd.has_colors():
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep_mask])
        
        # 保存文件（使用命令行参数指定的文件名，加上时间戳避免覆盖）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = args.output_filename
        if not base_filename.endswith(".ply"):
            base_filename += ".ply"
        # 在文件名中插入时间戳
        name_part, ext = os.path.splitext(base_filename)
        output_filename = f"{name_part}_{timestamp}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        o3d.io.write_point_cloud(output_path, filtered_pcd)
        
        print(f"✅ 已保存过滤后的点云到: {output_path}")
        print(f"   保留点数: {keep_mask.sum()} / {len(points)} ({100*keep_mask.sum()/len(points):.1f}%)")
        print(f"   置信度阈值: {threshold:.4f}")
        
        # 显示成功消息
        num_remove = (~keep_mask).sum()
        stats_text.value = (
            f"Points to keep: {keep_mask.sum()} / {len(points)} ({100*keep_mask.sum()/len(points):.1f}%)\n"
            f"Points to remove: {num_remove} / {len(points)} ({100*num_remove/len(points):.1f}%)\n"
            f"✅ Saved to: {output_path}"
        )
    
    print(f"\n✅ Viser服务器运行中!")
    print(f"🌐 在浏览器中打开: http://<server-ip>:{port}")
    print(f"\n📋 说明:")
    print(f"   - 绿色/正常颜色的点：置信度 >= 阈值（将被保留）")
    print(f"   - 红色的点：置信度 < 阈值（将被删除）")
    print(f"   - 使用滑块调整置信度阈值")
    print(f"   - 点击 'Save Filtered Point Cloud' 保存过滤后的点云")
    print(f"\n按 Ctrl+C 停止服务器")
    
    # 保持服务器运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
