#!/usr/bin/env python3
"""
可视化 pinhole 模型重建结果
包括稀疏点云、相机位置，以及可选的稠密点云
"""

import numpy as np
import viser
import viser.transforms as viser_tf
import pycolmap
import argparse
import atexit
import os
import shutil
import socket
import struct
import sys
import tempfile
from pathlib import Path
from plyfile import PlyData
from PIL import Image

from functools import lru_cache


def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0


def load_ply_xyzrgb(ply_path: Path):
    """
    读取 PLY 点云
    支持有颜色和无颜色的PLY文件
    """
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    # 获取实际数据数组
    vertex_data = vertex.data if hasattr(vertex, 'data') else vertex

    # 提取位置信息（必需）
    positions = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)

    # 检查是否有颜色信息
    has_colors = False
    colors = None

    # 获取字段名列表
    if hasattr(vertex_data, 'dtype') and hasattr(vertex_data.dtype, 'names'):
        field_names = vertex_data.dtype.names
    else:
        field_names = []

    # 检查是否有red, green, blue字段
    if field_names and all(field in field_names for field in ["red", "green", "blue"]):
        has_colors = True
        colors = np.stack([vertex_data["red"], vertex_data["green"], vertex_data["blue"]], axis=1).astype(np.uint8)
    elif field_names and all(field in field_names for field in ["r", "g", "b"]):
        has_colors = True
        colors = np.stack([vertex_data["r"], vertex_data["g"], vertex_data["b"]], axis=1).astype(np.uint8)
    else:
        # 尝试直接访问
        try:
            test_red = vertex_data["red"]
            test_green = vertex_data["green"]
            test_blue = vertex_data["blue"]
            has_colors = True
            colors = np.stack([test_red, test_green, test_blue], axis=1).astype(np.uint8)
        except (KeyError, ValueError, TypeError):
            try:
                test_r = vertex_data["r"]
                test_g = vertex_data["g"]
                test_b = vertex_data["b"]
                has_colors = True
                colors = np.stack([test_r, test_g, test_b], axis=1).astype(np.uint8)
            except (KeyError, ValueError, TypeError):
                has_colors = False

    # 如果没有颜色信息，根据位置生成颜色
    if not has_colors:
        # 归一化位置到[0, 1]范围
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range = np.where(pos_range > 1e-6, pos_range, 1.0)

        normalized_pos = (positions - pos_min) / pos_range

        # 使用z坐标生成颜色渐变
        z_norm = normalized_pos[:, 2]
        r = np.clip((z_norm - 0.5) * 2, 0, 1)
        g = np.clip(1 - abs(z_norm - 0.5) * 2, 0, 1)
        b = np.clip((0.5 - z_norm) * 2, 0, 1)

        colors = np.stack([r, g, b], axis=1)
        colors = (colors * 255).astype(np.uint8)

    return positions, colors


def prepare_sparse_model_path(sparse_path: Path) -> tuple[Path, bool]:
    """
    pycolmap 需要目录中存在 points3D.bin。
    若仅有小写 points3d.bin，在临时目录中链为 points3D.bin。
    若均不存在：写入空 points3D.bin 以便加载，并返回 skip_sparse=True。
    """
    p3d = sparse_path / "points3D.bin"
    p3d_lower = sparse_path / "points3d.bin"

    if p3d.is_file():
        return sparse_path, False

    def _tmpdir_with_symlinks() -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="viz_sparse_"))
        atexit.register(lambda p=tmp: shutil.rmtree(p, ignore_errors=True))
        for f in sparse_path.iterdir():
            if f.is_file() and f.name not in ("points3D.bin", "points3d.bin"):
                os.symlink(f.resolve(), tmp / f.name)
        return tmp

    if p3d_lower.is_file():
        tmp = _tmpdir_with_symlinks()
        os.symlink(p3d_lower.resolve(), tmp / "points3D.bin")
        return tmp, False

    if not os.environ.get("WATCH_PINHOLE_SCRIPT"):
        print("⚠️  未找到 points3D.bin，将不绘制稀疏点云。")
    tmp = _tmpdir_with_symlinks()
    (tmp / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    return tmp, True


def main():
    parser = argparse.ArgumentParser(
        description="可视化 pinhole 模型重建结果（稀疏点云、相机位置、稠密点云）"
    )
    parser.add_argument(
        "--workspace_path",
        type=Path,
        required=True,
        help="COLMAP工作目录路径（包含sparse/0和可选的fused.ply）"
    )
    parser.add_argument(
        "--sparse_path",
        type=Path,
        help="稀疏重建结果路径（默认: workspace_path/sparse/0）"
    )
    parser.add_argument(
        "--dense_ply",
        type=Path,
        help="稠密点云PLY文件路径（默认: workspace_path/fused.ply 或 workspace_path/dense/fused.ply）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser服务器端口（默认: 8080）"
    )
    parser.add_argument(
        "--sparse_point_size",
        type=float,
        default=0.01,
        help="稀疏点云点的大小（默认: 0.01）"
    )
    parser.add_argument(
        "--dense_point_size",
        type=float,
        default=0.005,
        help="稠密点云点的大小（默认: 0.005）"
    )
    parser.add_argument(
        "--camera_scale",
        type=float,
        default=0.05,
        help="相机frustum的缩放比例（默认: 0.05）"
    )
    parser.add_argument(
        "--hide_sparse_points",
        action="store_true",
        help="隐藏稀疏点云"
    )
    parser.add_argument(
        "--hide_cameras",
        action="store_true",
        help="隐藏相机位置"
    )
    parser.add_argument(
        "--show-camera-ids",
        action="store_true",
        help="在相机锥体旁显示相机编号（image_id）"
    )
    parser.add_argument(
        "--hide_dense_points",
        action="store_true",
        help="隐藏稠密点云"
    )

    parser.add_argument(
        "--frustum-images",
        action="store_true",
        help="在每个相机锥体上贴合对应图片（可能较慢、占用较多浏览器带宽）"
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="图片根目录（用于解析 images.bin 里的 image.name；默认 workspace_path/images）"
    )
    parser.add_argument(
        "--frustum-image-max-size",
        type=int,
        default=256,
        help="贴合到锥体上的图片最大边长（会自动等比缩放）"
    )
    parser.add_argument(
        "--frustum-image-every",
        type=int,
        default=1,
        help="每隔多少张图显示一次（用于减少浏览器压力）"
    )
    parser.add_argument(
        "--frustum-roll-deg",
        type=float,
        default=0.0,
        help="绕相机光轴(+Z，指向方向)的固定滚转角（单位: 度）；用于修正可视化与数据的roll约定差异"
    )

    args = parser.parse_args()

    workspace_path = Path(args.workspace_path)
    if not workspace_path.exists():
        print(f"❌ 工作目录不存在: {workspace_path}")
        sys.exit(1)

    # 设置默认路径
    if args.sparse_path:
        sparse_path = Path(args.sparse_path)
    else:
        sparse_path = workspace_path / "sparse" / "0"

    if args.dense_ply:
        dense_ply_path = Path(args.dense_ply)
    else:
        # 尝试多个可能的稠密点云位置
        possible_dense_paths = [
            workspace_path / "fused.ply",
            workspace_path / "dense" / "fused.ply",
            workspace_path / "stereo" / "fused.ply",
        ]
        dense_ply_path = None
        for path in possible_dense_paths:
            if path.exists():
                dense_ply_path = path
                break

    # 检查稀疏重建结果
    if not sparse_path.exists():
        print(f"❌ 稀疏重建结果不存在: {sparse_path}")
        sys.exit(1)

    for file in ["cameras.bin", "images.bin"]:
        if not (sparse_path / file).exists():
            print(f"❌ 稀疏重建文件不完整: 缺少 {file}")
            sys.exit(1)

    sparse_path, skip_sparse_no_points3d = prepare_sparse_model_path(sparse_path)
    hide_sparse = bool(args.hide_sparse_points or skip_sparse_no_points3d)

    print("=" * 60)
    print("Pinhole 模型重建结果可视化")
    print("=" * 60)
    print(f"工作目录: {workspace_path}")
    print(f"稀疏重建: {sparse_path}")
    if dense_ply_path and dense_ply_path.exists():
        print(f"稠密点云: {dense_ply_path}")
    else:
        print("稠密点云: 未找到（将仅显示稀疏重建结果）")
    print("=" * 60)

    # 加载COLMAP重建结果
    print("📖 读取COLMAP重建结果...")
    try:
        recon = pycolmap.Reconstruction(str(sparse_path))
        num_images = len(recon.images)
        num_points = len(recon.points3D)
        print(f"   ✅ 加载了 {num_images} 张图像, {num_points} 个3D点")
    except Exception as e:
        print(f"❌ 无法读取COLMAP重建结果: {e}")
        sys.exit(1)

    images_root = args.images_root if args.images_root is not None else (workspace_path / "images")

    @lru_cache(maxsize=128)
    def _load_image_for_frustum(image_path: str) -> np.ndarray | None:
        """
        读取并缩放图片，返回 (H, W, 3) uint8 RGB。
        返回 None 表示图片不存在或读取失败。
        """
        p = Path(image_path)
        if not p.is_file():
            return None

        try:
            with Image.open(str(p)) as im:
                im = im.convert("RGB")
                w, h = im.size
                max_side = max(w, h)
                if args.frustum_image_max_size > 0 and max_side > args.frustum_image_max_size:
                    scale = args.frustum_image_max_size / float(max_side)
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    im = im.resize((nw, nh), resample=Image.Resampling.LANCZOS)
                arr = np.asarray(im, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    return None
                return arr
        except Exception:
            return None

    # 检查端口并启动服务器
    port = args.port
    if check_port(port):
        print(f"⚠️  端口 {port} 已被占用，尝试使用 {port + 1}...")
        port = port + 1

    print(f"🚀 启动Viser服务器，端口: {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)

    # 加载稀疏点云
    if not hide_sparse:
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

            server.scene.add_point_cloud(
                name="sparse_points",
                points=points,
                colors=colors,
                point_size=args.sparse_point_size,
            )
            print(f"   ✅ 加载了 {len(points)} 个稀疏点")
        else:
            print("   ⚠️  没有找到稀疏点云")

    # 加载相机
    if not args.hide_cameras:
        print("📷 加载相机位置...")
        camera_count = 0
        image_counter = 0
        for image_id, image in recon.images.items():
            try:
                cam = recon.cameras[image.camera_id]

                # 世界坐标系下相机位姿
                cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                # COLMAP/pycolmap 中 cam_from_world 满足: p_cam = R @ p_world + t
                # 这里需要 world_from_cam (= T_wc) 以供 viser 使用。
                R_wc_original = R.T  # world_from_cam rotation (before roll correction)
                camera_center_w = -R_wc_original @ t

                # 可能存在 roll 约定差异（例如 +Y down / +Y up 的差异）。
                # 使用一个固定的绕相机本地 +Z 轴的旋转来修正。
                frustum_roll_rad = np.deg2rad(args.frustum_roll_deg)
                if abs(frustum_roll_rad) > 1e-12:
                    c = float(np.cos(frustum_roll_rad))
                    s = float(np.sin(frustum_roll_rad))
                    # Rodrigues for local rotation about +Z (right-handed)
                    Rz = np.array(
                        [
                            [c, -s, 0.0],
                            [s,  c, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    )
                    # world_from_cam_new = world_from_cam_original @ Rz
                    R_wc = R_wc_original @ Rz
                else:
                    R_wc = R_wc_original

                T_wc = np.eye(4)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = camera_center_w

                # Convert to viser SE3 format (3x4 matrix)
                T_wc_3x4 = T_wc[:3, :]
                T_world_camera = viser_tf.SE3.from_matrix(T_wc_3x4)

                # 计算FOV
                # 尝试从相机参数计算FOV，如果失败则使用默认值
                # viser 的 add_camera_frustum() 约定:
                #   fov: 垂直视场角 (radians)
                #   aspect: width / height
                fov = np.deg2rad(50.0)  # 默认FOV（弧度）
                try:
                    # 获取相机模型（CameraModelId枚举）
                    camera_model = cam.model

                    # 尝试从参数计算FOV
                    if len(cam.params) > 1:
                        # 对于 PINHOLE / OPENCV 这两类模型，cam.params 通常为 [fx, fy, cx, cy, ...]
                        fy = cam.params[1]
                        if fy > 0 and cam.height > 0:
                            # 垂直FOV: fov = 2 * arctan(height / (2 * fy))
                            fov = 2 * np.arctan(cam.height / (2 * fy))
                            # 限制FOV在合理范围内（弧度）
                            fov = np.clip(fov, np.deg2rad(10.0), np.deg2rad(170.0))
                except Exception:
                    # 如果计算失败，使用默认FOV
                    pass

                # 可选：贴图到相机锥体上
                frustum_image = None
                if args.frustum_images and args.frustum_image_every > 0:
                    if (image_counter % args.frustum_image_every) == 0:
                        # COLMAP 的 image.name 在这里通常就是 basename（由转换脚本决定）
                        img_path = images_root / image.name
                        frustum_image = _load_image_for_frustum(str(img_path))

                image_counter += 1

                server.scene.add_camera_frustum(
                    name=f"cam_{image_id}",
                    fov=fov,
                    aspect=cam.width / cam.height,
                    scale=args.camera_scale,
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    image=frustum_image,
                )
                if args.show_camera_ids:
                    # 在相机中心附近放置 2D label，方便在视锥体上直接读出对应编号
                    server.scene.add_label(
                        name=f"cam_{image_id}_id",
                        text=str(image_id),
                        wxyz=T_world_camera.rotation().wxyz,
                        position=T_world_camera.translation(),
                        anchor="bottom-center",
                        font_size_mode="screen",
                        depth_test=False,
                        visible=True,
                    )
                camera_count += 1
            except Exception as e:
                print(f"⚠️  处理相机 {image_id} 时出错: {e}")
                continue

        print(f"   ✅ 加载了 {camera_count} 个相机")

    # 加载稠密点云（如果存在）
    if dense_ply_path and dense_ply_path.exists() and not args.hide_dense_points:
        print("📦 加载稠密点云...")
        try:
            dense_positions, dense_colors = load_ply_xyzrgb(dense_ply_path)
            # 转换颜色格式（从uint8 [0-255] 到 float [0-1]）
            dense_colors_float = dense_colors.astype(np.float32) / 255.0

            server.scene.add_point_cloud(
                name="dense_points",
                points=dense_positions,
                colors=dense_colors_float,
                point_size=args.dense_point_size,
            )
            print(f"   ✅ 加载了 {len(dense_positions)} 个稠密点")
        except Exception as e:
            print(f"⚠️  加载稠密点云失败: {e}")
            import traceback
            traceback.print_exc()
    elif dense_ply_path and not dense_ply_path.exists():
        print("ℹ️  稠密点云文件不存在，跳过")

    print("=" * 60)
    print(f"✅ 可视化服务器已启动")
    print(f"   在浏览器中打开: http://localhost:{port}")
    print(f"   按 Ctrl+C 退出")
    print("=" * 60)

    # 保持服务器运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在关闭服务器...")


if __name__ == "__main__":
    main()

