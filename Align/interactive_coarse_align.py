#!/usr/bin/env python3
"""
Interactive coarse alignment tool for two PLY point clouds using Viser.

Step 1 (you):  使用本脚本在浏览器界面中调整 B 的旋转和平移，并点击保存按钮写出 4x4 变换矩阵。
Step 2 (align.py):  读取该矩阵作为初始位姿，执行精细 ICP 对齐。

为保证交互流畅，这里仅显示下采样后的 B（预览），RT 仍然对全分辨率点云有效。
"""

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import trimesh
import trimesh.creation
import viser
import yaml


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def pcd_to_numpy(pcd: o3d.geometry.PointCloud):
    """Convert Open3D point cloud to (points, colors) numpy arrays."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        raise RuntimeError("Point cloud has no points.")
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if colors.shape != pts.shape:
            colors = np.resize(colors, pts.shape)
    else:
        colors = np.ones_like(pts, dtype=np.float32)
    return pts, colors


def euler_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert XYZ Euler angles (deg) to rotation matrix."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)

    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in (w, x, y, z) order."""
    # Robust matrix->quat conversion
    m = R
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    # Normalize to be safe
    q /= np.linalg.norm(q) + 1e-8
    return q


def main():
    parser = argparse.ArgumentParser(
        description="Interactive coarse alignment for two PLY point clouds (A fixed, B adjustable)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Align config path (default: ./config.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Viser server port (default: 8090)",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.005,
        help="Point size for visualization (default: 0.005)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_cfg = cfg.get("input", {}) or {}
    init_cfg = cfg.get("init", {}) or {}

    path_A = Path(input_cfg.get("pointcloud_A"))
    path_B = Path(input_cfg.get("pointcloud_B"))
    tf_out = Path(init_cfg.get("transform_file", "output/init_transform.txt"))

    if not path_A.is_file():
        raise FileNotFoundError(f"pointcloud_A not found: {path_A}")
    if not path_B.is_file():
        raise FileNotFoundError(f"pointcloud_B not found: {path_B}")

    tf_out.parent.mkdir(parents=True, exist_ok=True)

    print("=======================================")
    print("Interactive coarse alignment (Viser)")
    print("=======================================")
    print(f"A (fixed) : {path_A}")
    print(f"B (moving): {path_B}")
    print(f"RT output : {tf_out}")
    print(f"Port      : {args.port}")
    print("Open your browser at: http://<server-ip>:{port}".format(port=args.port))
    print("Use the sliders to adjust B (preview is downsampled), then click 'Save transform' in the GUI.")
    print("=======================================")

    # Load full-resolution point clouds
    A_full = o3d.io.read_point_cloud(str(path_A))
    B_full = o3d.io.read_point_cloud(str(path_B))

    # Downsample for interactive preview（根据配置决定是否下采样）
    preview_downsample = cfg.get("preview_downsample", False)
    
    if preview_downsample:
        voxel = float(cfg.get("voxel_size", 0.03))
        preview_voxel = float(cfg.get("preview_voxel_size", voxel * 2.0))
        
        A_vis = A_full.voxel_down_sample(preview_voxel)
        B_vis = B_full.voxel_down_sample(preview_voxel)
        
        print(
            f"[CoarseAlign] Downsample for preview with voxel={preview_voxel:.3f} "
            f"(A: {len(A_full.points)} → {len(A_vis.points)}, "
            f"B: {len(B_full.points)} → {len(B_vis.points)})"
        )
    else:
        A_vis = A_full
        B_vis = B_full
        print(
            f"[CoarseAlign] Using full-resolution point clouds "
            f"(A: {len(A_full.points)} points, B: {len(B_full.points)} points)"
        )

    A_pts, A_colors = pcd_to_numpy(A_vis)
    B_pts, B_colors = pcd_to_numpy(B_vis)

    # Start viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port, verbose=True)

    # Add fixed A
    server.scene.add_point_cloud(
        name="/A_fixed",
        points=A_pts,
        colors=A_colors,
        point_size=args.point_size,
    )

    # Add a world XYZ axis using spline line segments (similar style as viser_rig.py).
    # Axis is anchored at the global origin (0,0,0), and its length adapts to scene scale.
    try:
        bbox = A_vis.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        axis_len = max(float(diag) * 0.3, 1.0)  # ≈ 30% of scene size, at least 1 m

        origin = np.zeros(3, dtype=np.float32)
        x_axis_end = origin + np.array([axis_len, 0.0, 0.0], dtype=np.float32)
        y_axis_end = origin + np.array([0.0, axis_len, 0.0], dtype=np.float32)
        z_axis_end = origin + np.array([0.0, 0.0, axis_len], dtype=np.float32)

        line_width = 3.0

        # X axis (red)
        server.scene.add_spline_catmull_rom(
            name="/world_axis_x",
            positions=np.stack([origin, x_axis_end]),
            curve_type="chordal",
            tension=0.0,
            line_width=line_width,
            color=(255, 0, 0),
        )
        # Y axis (green)
        server.scene.add_spline_catmull_rom(
            name="/world_axis_y",
            positions=np.stack([origin, y_axis_end]),
            curve_type="chordal",
            tension=0.0,
            line_width=line_width,
            color=(0, 255, 0),
        )
        # Z axis (blue)
        server.scene.add_spline_catmull_rom(
            name="/world_axis_z",
            positions=np.stack([origin, z_axis_end]),
            curve_type="chordal",
            tension=0.0,
            line_width=line_width,
            color=(0, 0, 255),
        )
    except Exception as e:
        print(f"[CoarseAlign] Failed to add world axes visualization: {e}")

    # Add adjustable B（初始姿态，后续只更新其位姿，不重新创建节点或发送整个点云）
    pc_B = server.scene.add_point_cloud(
        name="/B_moving",
        points=B_pts,
        colors=B_colors,
        point_size=args.point_size,
    )

    current_T = np.eye(4, dtype=float)

    # Containers for correspondence-based alignment
    picked_A: List[np.ndarray] = []
    picked_B: List[np.ndarray] = []
    picked_A_handles: List[viser.GlbHandle] = []
    picked_B_handles: List[viser.GlbHandle] = []

    # Correspondence picking controls（按钮触发一次选点，避免持续拦截鼠标）
    with server.gui.add_folder("Correspondence (A ↔ B)"):
        pick_A_button = server.gui.add_button("Pick one A (fixed)")
        pick_B_button = server.gui.add_button("Pick one B (moving)")
        solve_button = server.gui.add_button("Solve from correspondences & Save")
        clear_button = server.gui.add_button("Clear picked points")

    # View-based incremental adjustment controls（三视图 + 增量按钮）
    with server.gui.add_folder("View-based Fine Adjustment"):
        # 视距控制：用于“缩小 / 放大画面”
        view_dist_slider = server.gui.add_slider(
            "View distance",
            min=1.0,
            max=1000.0,
            step=1.0,
            initial_value=30.0,
        )
        front_view_cb = server.gui.add_checkbox("Front view (look -X)", initial_value=False)
        side_view_cb = server.gui.add_checkbox("Side view (look -Y)", initial_value=False)
        top_view_cb = server.gui.add_checkbox("Top view (look -Z)", initial_value=False)

        # 位置微调按钮
        move_u_pos = server.gui.add_button("Move +U")
        move_u_neg = server.gui.add_button("Move -U")
        move_v_pos = server.gui.add_button("Move +V")
        move_v_neg = server.gui.add_button("Move -V")
        # 绕视线方向旋转
        rot_pos = server.gui.add_button("Rotate +")
        rot_neg = server.gui.add_button("Rotate -")

    def apply_delta(delta_T: np.ndarray):
        """Apply an incremental SE(3) delta to current_T and update B's pose."""
        nonlocal current_T
        current_T = delta_T @ current_T

        R_total = current_T[:3, :3]
        t_total = current_T[:3, 3]
        q_wxyz = rotation_matrix_to_wxyz(R_total)
        pc_B.position = t_total.astype(np.float32)
        pc_B.wxyz = q_wxyz.astype(np.float32)

    # Mouse picking: build correspondences on A / B by点击一次（通过按钮显式进入选点模式）
    def handle_pointer_click(
        event: viser.ScenePointerEvent,
        which: str,  # "A" or "B"
    ) -> None:
        # 只响应真正的点击事件，避免拖动相机时误触
        if event.event_type != "click":
            return

        # 在 A_vis / B_vis 上做最近点搜索，得到 3D 坐标
        o = np.array(event.ray_origin, dtype=float)
        d = np.array(event.ray_direction, dtype=float)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-8:
            return
        d = d / d_norm

        def nearest_point_on_cloud(pts: np.ndarray) -> np.ndarray:
            # 对每个点 p，计算到射线的最近距离
            # t = (p - o)·d，q = o + t d，dist = ||p - q||
            v = pts - o[None, :]
            t = v @ d  # (N,)
            t = np.maximum(t, 0.0)
            q = o[None, :] + t[:, None] * d[None, :]
            dist2 = np.sum((pts - q) ** 2, axis=1)
            idx = np.argmin(dist2)
            return pts[idx]

        # 颜色方案：第 1 组（A1/B1）红色，第 2 组蓝色，第 3 组绿色，之后循环使用
        def color_for_index(idx: int) -> np.ndarray:
            palette = [
                np.array([1.0, 0.0, 0.0, 1.0]),  # red
                np.array([0.0, 0.0, 1.0, 1.0]),  # blue
                np.array([0.0, 1.0, 0.0, 1.0]),  # green
            ]
            return palette[(idx - 1) % len(palette)]

        # 按你的需求，将小球半径固定为 1m，便于在大场景中清晰可见
        marker_radius = 1.0

        if which == "A":
            pos = nearest_point_on_cloud(A_pts)
            picked_A.append(pos)
            print(f"[CoarseAlign] Picked A point #{len(picked_A)} at {pos}")

            color = color_for_index(len(picked_A))
            sphere = trimesh.creation.icosphere(radius=marker_radius)
            sphere.vertices += pos  # 平移到选中位置
            sphere.visual.vertex_colors = color  # type: ignore
            handle = server.scene.add_mesh_trimesh(
                name=f"/pick_A_{len(picked_A)}",
                mesh=sphere,
            )
            picked_A_handles.append(handle)
        elif which == "B":
            pos = nearest_point_on_cloud(B_pts)
            picked_B.append(pos)
            print(f"[CoarseAlign] Picked B point #{len(picked_B)} at {pos}")
            color = color_for_index(len(picked_B))
            sphere = trimesh.creation.icosphere(radius=marker_radius)
            sphere.vertices += pos
            sphere.visual.vertex_colors = color  # type: ignore
            handle = server.scene.add_mesh_trimesh(
                name=f"/pick_B_{len(picked_B)}",
                mesh=sphere,
            )
            picked_B_handles.append(handle)
        else:
            print(f"[CoarseAlign] Unknown target '{which}' for picking.")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # 每个客户端各自管理一次性选点回调 + 视图控制，避免影响相机拖动

        # ---- Correspondence picking ----
        @pick_A_button.on_click
        def _(_: viser.GuiEvent[None]):
            print("[CoarseAlign] Click in scene to pick ONE point on A (fixed)...")

            @client.scene.on_pointer_event(event_type="click")
            def _on_click(event: viser.ScenePointerEvent) -> None:
                handle_pointer_click(event, "A")
                # 选完一个点后立刻移除回调，恢复正常鼠标行为
                client.scene.remove_pointer_callback()

        @pick_B_button.on_click
        def _(_: viser.GuiEvent[None]):
            print("[CoarseAlign] Click in scene to pick ONE point on B (moving)...")

            @client.scene.on_pointer_event(event_type="click")
            def _on_click(event: viser.ScenePointerEvent) -> None:
                handle_pointer_click(event, "B")
                client.scene.remove_pointer_callback()

        # ---- Orthographic-like view toggles（这里只调整相机朝向和远近）----
        def set_view(direction: str):
            # 简单设置：相机放到对应轴的正方向，朝向原点
            cam = client.camera
            dist = float(view_dist_slider.value)  # 由滑块控制视距
            if direction == "front":  # look -X
                cam.position = np.array([dist, 0.0, 0.0], dtype=np.float32)
                cam.wxyz = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            elif direction == "side":  # look -Y
                cam.position = np.array([0.0, dist, 0.0], dtype=np.float32)
                # 先从 front 视图绕 Z 轴顺时针旋转 90°，避免 up 向量退化
                cam.wxyz = np.array([0.0, 0.0, -0.7071, 0.7071], dtype=np.float32)
            elif direction == "top":  # look -Z
                cam.position = np.array([0.0, 0.0, dist], dtype=np.float32)
                cam.wxyz = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)

        @front_view_cb.on_update
        def _(_: viser.GuiEvent[bool]):
            if front_view_cb.value:
                side_view_cb.value = False
                top_view_cb.value = False
                set_view("front")

        @side_view_cb.on_update
        def _(_: viser.GuiEvent[bool]):
            if side_view_cb.value:
                front_view_cb.value = False
                top_view_cb.value = False
                set_view("side")

        @top_view_cb.on_update
        def _(_: viser.GuiEvent[bool]):
            if top_view_cb.value:
                front_view_cb.value = False
                side_view_cb.value = False
                set_view("top")

        # ---- View-based incremental buttons ----
        def make_delta_for_view(view: str, du: float, dv: float, dtheta_deg: float) -> np.ndarray:
            """
            根据当前视图，构造在“屏幕 U/V 方向”和视线方向上的增量变换：
            - du, dv: 沿 U/V 方向的平移（米）
            - dtheta_deg: 沿视线方向的旋转（度）
            """
            if view == "front":  # look -X, 令 U=+Z, V=+Y, axis=X
                U = np.array([0.0, 0.0, 1.0])
                V = np.array([0.0, 1.0, 0.0])
                axis = "x"
            elif view == "side":  # look -Y, U=+Z, V=+X, axis=Y
                U = np.array([0.0, 0.0, 1.0])
                V = np.array([1.0, 0.0, 0.0])
                axis = "y"
            elif view == "top":  # look -Z, U=+Y, V=+X, axis=Z
                U = np.array([0.0, 1.0, 0.0])
                V = np.array([1.0, 0.0, 0.0])
                axis = "z"
            else:
                U = np.array([0.0, 0.0, 0.0])
                V = np.array([0.0, 0.0, 0.0])
                axis = "x"

            t = du * U + dv * V
            if axis == "x":
                R = euler_deg_to_matrix(dtheta_deg, 0.0, 0.0)
            elif axis == "y":
                R = euler_deg_to_matrix(0.0, dtheta_deg, 0.0)
            else:
                R = euler_deg_to_matrix(0.0, 0.0, dtheta_deg)

            delta = np.eye(4, dtype=float)
            delta[:3, :3] = R
            delta[:3, 3] = t
            return delta

        def current_view() -> str | None:
            if front_view_cb.value:
                return "front"
            if side_view_cb.value:
                return "side"
            if top_view_cb.value:
                return "top"
            return None

        move_step = 0.1   # 米
        rot_step = 1.0    # 度

        @move_u_pos.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Move +U.")
                return
            delta = make_delta_for_view(view, du=move_step, dv=0.0, dtheta_deg=0.0)
            apply_delta(delta)

        @move_u_neg.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Move -U.")
                return
            delta = make_delta_for_view(view, du=-move_step, dv=0.0, dtheta_deg=0.0)
            apply_delta(delta)

        @move_v_pos.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Move +V.")
                return
            delta = make_delta_for_view(view, du=0.0, dv=move_step, dtheta_deg=0.0)
            apply_delta(delta)

        @move_v_neg.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Move -V.")
                return
            delta = make_delta_for_view(view, du=0.0, dv=-move_step, dtheta_deg=0.0)
            apply_delta(delta)

        @rot_pos.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Rotate +.")
                return
            delta = make_delta_for_view(view, du=0.0, dv=0.0, dtheta_deg=rot_step)
            apply_delta(delta)

        @rot_neg.on_click
        def _(_: viser.GuiEvent[None]):
            view = current_view()
            if view is None:
                print("[CoarseAlign] No view selected for Rotate -.")
                return
            delta = make_delta_for_view(view, du=0.0, dv=0.0, dtheta_deg=-rot_step)
            apply_delta(delta)

    def solve_rigid_transform(B_corr: np.ndarray, A_corr: np.ndarray):
        """Solve rigid transform (R, t) such that R * B + t ≈ A."""
        assert B_corr.shape == A_corr.shape
        assert B_corr.shape[1] == 3

        centroid_B = B_corr.mean(axis=0)
        centroid_A = A_corr.mean(axis=0)

        B_centered = B_corr - centroid_B
        A_centered = A_corr - centroid_A

        H = B_centered.T @ A_centered  # 3x3
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_A - R @ centroid_B
        return R, t

    @solve_button.on_click
    def _(_: viser.GuiEvent[None]):
        if len(picked_A) != len(picked_B):
            print(
                f"[CoarseAlign] Cannot solve: |A|={len(picked_A)} != |B|={len(picked_B)}. "
                "Please pick the same number of points on A and B."
            )
            return
        if len(picked_A) < 3:
            print(
                f"[CoarseAlign] Need at least 3 correspondences to solve rigid transform, "
                f"but got {len(picked_A)}."
            )
            return

        A_corr = np.stack(picked_A, axis=0)
        B_corr = np.stack(picked_B, axis=0)

        R, t = solve_rigid_transform(B_corr, A_corr)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t

        np.savetxt(str(tf_out), T, fmt="%.8f")
        print(f"[CoarseAlign] Solved transform from correspondences and saved to {tf_out}:")
        print(T)

        # 同时更新当前可视化中的 B 位置，便于预览效果
        q_wxyz = rotation_matrix_to_wxyz(R)
        pc_B.position = t.astype(np.float32)
        pc_B.wxyz = q_wxyz.astype(np.float32)

        # 也更新 current_T，便于后续保存按钮保持一致
        nonlocal current_T
        current_T = T

    @clear_button.on_click
    def _(_: viser.GuiEvent[None]):
        picked_A.clear()
        picked_B.clear()
        # 同时清理场景中的参考小球
        for h in picked_A_handles:
            try:
                h.remove()
            except Exception:
                pass
        for h in picked_B_handles:
            try:
                h.remove()
            except Exception:
                pass
        picked_A_handles.clear()
        picked_B_handles.clear()
        print("[CoarseAlign] Cleared all picked correspondences (A, B) and marker spheres.")

    # Button to save transform
    def save_transform(_event: viser.GuiEvent[None]):
        np.savetxt(str(tf_out), current_T, fmt="%.8f")
        print(f"[CoarseAlign] Saved 4x4 transform to {tf_out}")

    server.gui.add_button("Save transform").on_click(save_transform)

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[CoarseAlign] Server stopped by user.")


if __name__ == "__main__":
    main()

