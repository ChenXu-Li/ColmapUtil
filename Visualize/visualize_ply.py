import numpy as np
import viser
from plyfile import PlyData
import socket
import sys
import os
import argparse


def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result == 0


def load_ply_xyzrgb(ply_path: str):
    """
    读取 PLY 点云
    支持有颜色和无颜色的PLY文件
    如果没有颜色信息，将根据位置生成颜色
    """
    ply = PlyData.read(ply_path)
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
    # 检查是否有r, g, b字段（小写）
    elif field_names and all(field in field_names for field in ["r", "g", "b"]):
        has_colors = True
        colors = np.stack([vertex_data["r"], vertex_data["g"], vertex_data["b"]], axis=1).astype(np.uint8)
    else:
        # 尝试直接访问（可能字段存在但不在dtype中）
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
        pos_range = np.where(pos_range > 1e-6, pos_range, 1.0)  # 避免除零

        normalized_pos = (positions - pos_min) / pos_range

        # 使用简单的颜色映射：根据z坐标（高度）生成颜色
        # 从蓝色（低）到红色（高）
        z_norm = normalized_pos[:, 2]  # 使用z坐标

        # 创建颜色渐变：蓝色 -> 青色 -> 绿色 -> 黄色 -> 红色
        r = np.clip((z_norm - 0.5) * 2, 0, 1)  # 红色分量
        g = np.clip(1 - abs(z_norm - 0.5) * 2, 0, 1)  # 绿色分量
        b = np.clip((0.5 - z_norm) * 2, 0, 1)  # 蓝色分量

        # 增强对比度
        colors = np.stack([r, g, b], axis=1)
        colors = (colors * 255).astype(np.uint8)

    return positions, colors


def sanitize_viser_name(name: str) -> str:
    # viser path-like names often use "/" – keep it readable but safe
    return name.replace("\\", "/").replace(" ", "_")


def main():
    def log(msg: str):
        print(msg, flush=True)

    parser = argparse.ArgumentParser(description="Viser: load and show one or more PLY point clouds (XYZRGB).")
    parser.add_argument(
        "ply_paths",
        nargs="*",
        help="PLY file paths (absolute paths recommended). You can pass multiple.",
    )
    parser.add_argument(
        "--ply",
        dest="ply_paths_opt",
        action="append",
        default=[],
        help="PLY file path (repeatable). Example: --ply /abs/a.ply --ply /abs/b.ply",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--point_size", type=float, default=0.005, help="Point size (default: 0.005)")

    args = parser.parse_args()

    ply_paths = list(args.ply_paths_opt) + list(args.ply_paths)
    if len(ply_paths) == 0:
        parser.error("No PLY paths provided. Example: python viser_ply.py --ply /abs/a.ply --ply /abs/b.ply")

    # Normalize to absolute paths for nicer logs + uniqueness.
    ply_paths = [os.path.abspath(p) for p in ply_paths]

    # =========================
    # 1. 检查端口
    # =========================
    PORT = args.port
    if check_port(PORT):
        log(f"WARNING: Port {PORT} is already in use!")
        log("Trying port 8081 instead...")
        PORT = 8081

    # =========================
    # 2. 启动 viser 服务器
    # =========================
    log(f"Starting viser server on {args.host}:{PORT} ...")
    server = viser.ViserServer(host=args.host, port=PORT, verbose=True)

    # =========================
    # 3. 添加多个点云
    # =========================
    loaded = 0
    for idx, ply_path in enumerate(ply_paths):
        log(f"Reading PLY [{idx+1}/{len(ply_paths)}]: {ply_path}")
        try:
            positions, colors = load_ply_xyzrgb(ply_path)
        except Exception as e:
            log(f"❌ Error reading PLY file: {ply_path}\n   {e}")
            continue

        base = os.path.splitext(os.path.basename(ply_path))[0]
        name = sanitize_viser_name(f"/ply/{idx:02d}_{base}")
        server.scene.add_point_cloud(
            name=name,
            points=positions,
            colors=colors,
            point_size=args.point_size,
        )
        log(f"✅ Loaded {positions.shape[0]:,} points as {name}")
        loaded += 1

    if loaded == 0:
        log("❌ No point clouds were loaded successfully; exiting.")
        sys.exit(1)

    log("✅ Viser server running successfully!")
    log(f"🌐 Open in browser: http://<server-ip>:{PORT}")
    log("Press Ctrl+C to stop the server")

    # 保持服务器运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nServer stopped by user")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()

