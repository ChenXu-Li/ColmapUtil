#!/usr/bin/env python3
"""
将稠密点云（世界坐标）重新投影到指定全景相机坐标系，生成等轴柱状（equirectangular）深度 / 颜色图。

典型用法（与当前工程默认目录匹配）：

    python fused_remap.py \
        --dataset_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB \
        --colmap_root /root/autodl-tmp/data/colmap_STAGE1_4x \
        --ply_path /root/autodl-tmp/data/colmap_STAGE1_4x/BridgeB/fused.ply \
        --camera_name pano_camera12 \
        --save_color

脚本会：
1. 读取 COLMAP 稀疏重建（用于获取 rig / camera 位姿）；
2. 读取给定的稠密点云（假定在 COLMAP 世界坐标系下，如 fused.ply / cut_dense_merge.ply 等）；
3. 对每一个全景帧，在指定的全景相机坐标系下，将所有 3D 点投影到等轴柱状（w×h）图像上：
   - 输出深度图：以相机为原点的欧式距离（单位与点云一致，通常是米）；
   - 可选输出颜色图：使用点云自带 RGB。

变换矩阵与 rig / camera 关系参考 `viser_rig_ply.py`。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
import pycolmap
from PIL import Image


def log(msg: str) -> None:
    print(msg, flush=True)


def load_ply_xyzrgb(ply_path: Path):
    """读取 PLY 点云（需要 vertex: x,y,z,red,green,blue），返回世界坐标点和颜色。"""
    log(f"📂 读取点云: {ply_path}")
    try:
        ply = PlyData.read(str(ply_path))
    except Exception as e:
        raise RuntimeError(f"无法读取 PLY 文件 {ply_path}: {e}")

    if "vertex" not in ply:
        raise RuntimeError(f"PLY 文件中缺少 'vertex' 元素: {ply_path}")

    vertex = ply["vertex"]

    if not all(k in vertex.data.dtype.names for k in ("x", "y", "z")):
        raise RuntimeError("PLY 须包含 x, y, z 顶点属性")

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )

    if all(k in vertex.data.dtype.names for k in ("red", "green", "blue")):
        colors = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=1
        ).astype(np.uint8)
    else:
        colors = None

    log(f"   ✅ 加载 {positions.shape[0]:,} 个点")
    return positions, colors


def build_pano_to_frame_mapping(recon: pycolmap.Reconstruction):
    """
    建立全景图名称到 frame_id 的映射。

    与 `viser_rig_ply.py` 中逻辑保持一致：
      - 图像名称格式：pano_camera{idx}/{pano_name}.png
      - 选择有 pose 的 frame 优先。
    """
    pano_to_frame: dict[str, int] = {}

    for img_id, img in recon.images.items():
        if img.frame_id not in recon.frames:
            continue

        img_name = img.name
        if "/" not in img_name:
            continue

        pano_name = img_name.split("/")[-1]
        pano_name = Path(pano_name).stem

        if pano_name not in pano_to_frame:
            pano_to_frame[pano_name] = img.frame_id
        else:
            current_frame = recon.frames[img.frame_id]
            existing_frame = recon.frames[pano_to_frame[pano_name]]
            if current_frame.has_pose() and not existing_frame.has_pose():
                pano_to_frame[pano_name] = img.frame_id

    return pano_to_frame


def get_cam_from_world_for_frame(
    recon: pycolmap.Reconstruction,
    frame_id: int,
    camera_name_substr: str,
) -> pycolmap.Rigid3d | None:
    """
    在给定 frame 中查找名称包含 camera_name_substr 的图像，并返回其 cam_from_world。
    """
    for img_id, img in recon.images.items():
        if img.frame_id != frame_id:
            continue
        if camera_name_substr not in img.name:
            continue
        cam_from_world = (
            img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        )
        return cam_from_world
    return None


def world_to_cam(points_world: np.ndarray, cam_from_world: pycolmap.Rigid3d):
    """
    使用 COLMAP 的 Rigid3d，将点从世界坐标系变换到相机坐标系。
    cam_point = R * world_point + t
    """
    R = cam_from_world.rotation.matrix()  # (3, 3)
    t = cam_from_world.translation  # (3,)
    pts_T = points_world.T  # (3, N)
    pts_cam_T = R @ pts_T + t[:, None]
    return pts_cam_T.T  # (N, 3)


def cam_points_to_equirect(
    points_cam: np.ndarray,
    colors: np.ndarray | None,
    width: int,
    height: int,
    max_depth: float | None = None,
):
    """
    将相机坐标系下的 3D 点投影到等轴柱状图（equirectangular）上。

    - points_cam: (N, 3)，单位为米（或任意统一单位）
    - colors: (N, 3) uint8 或 None
    - width, height: 输出 equirect 图像宽高，需满足 width = 2 * height
    - max_depth: 可选，超过该深度的点将被忽略

    返回：
      depth_map: (H, W) float32，单位与点云一致；无点处为 0
      color_map: (H, W, 3) uint8，如 colors 为 None，则为全 0
    """
    if width != 2 * height:
        raise ValueError("仅支持 360° 等轴柱状全景（width 应为 height 的 2 倍）")

    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError("points_cam 应为 (N, 3)")

    # 深度 = 到相机原点的欧式距离
    depths = np.linalg.norm(points_cam, axis=1).astype(np.float32)

    # 过滤：后方点、零深度点、可选最大距离
    z = points_cam[:, 2]
    valid = z > 0  # 只保留位于相机前方的点
    valid &= depths > 1e-6
    if max_depth is not None:
        valid &= depths <= float(max_depth)

    if not np.any(valid):
        log("   ⚠️ 当前帧在指定相机下没有有效点")
        return np.zeros((height, width), np.float32), np.zeros(
            (height, width, 3), np.uint8
        )

    pts = points_cam[valid]
    d = depths[valid]
    if colors is not None:
        cols = colors[valid]
    else:
        cols = None

    # 单位方向
    dirs = pts / d[:, None]  # (N, 3)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # 与 panorama.py 中 spherical_img_from_cam 一致的定义
    yaw = np.arctan2(x, z)  # [-pi, pi]
    pitch = -np.arctan2(y, np.sqrt(x * x + z * z))  # [-pi/2, pi/2]
    u = (1.0 + yaw / np.pi) * 0.5  # [0, 1]
    v = (1.0 - pitch * 2.0 / np.pi) * 0.5  # [0, 1]

    # 映射到像素坐标
    u_pix = np.clip(np.floor(u * width).astype(np.int64), 0, width - 1)
    v_pix = np.clip(np.floor(v * height).astype(np.int64), 0, height - 1)

    # 光栅化：对每个像素保留最近深度
    depth_map = np.full((height * width,), np.inf, dtype=np.float32)
    lin_idx = v_pix * width + u_pix
    np.minimum.at(depth_map, lin_idx, d)

    # 无效像素（仍为 inf）置为 0
    depth_map[~np.isfinite(depth_map)] = 0.0
    depth_map = depth_map.reshape((height, width))

    if cols is not None:
        # 对颜色同样按照最近深度选择
        color_map = np.zeros((height * width, 3), dtype=np.uint8)

        # 为了选择对应最小深度的颜色，先记录每个像素的当前最小深度索引
        # 简单做法：再次遍历，若该点深度等于像素中最小深度，则写入颜色。
        # （代价略高，但实现简单清晰，且只在 CPU 上一次性运行）
        depth_flat = depth_map.ravel()
        for i in range(d.shape[0]):
            idx = lin_idx[i]
            if depth_flat[idx] == 0.0:
                continue
            if abs(depth_flat[idx] - d[i]) < 1e-5 or d[i] <= depth_flat[idx] + 1e-5:
                color_map[idx] = cols[i]

        color_map = color_map.reshape((height, width, 3))
    else:
        color_map = np.zeros((height, width, 3), dtype=np.uint8)

    return depth_map, color_map


def process_dataset(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")

    scene_name = dataset_dir.name
    log(f"📌 数据集场景: {scene_name}")

    # COLMAP 重建目录
    colmap_root = Path(args.colmap_root).resolve()
    colmap_scene_dir = colmap_root / scene_name
    colmap_sparse_dir = colmap_scene_dir / "sparse" / "0"

    if not colmap_sparse_dir.exists():
        raise SystemExit(f"❌ COLMAP 稀疏模型目录不存在: {colmap_sparse_dir}")

    log(f"📖 读取 COLMAP 重建: {colmap_sparse_dir}")
    try:
        recon = pycolmap.Reconstruction(str(colmap_sparse_dir))
    except Exception as e:
        raise SystemExit(f"❌ 无法读取 COLMAP 重建结果: {e}")

    if len(recon.frames) == 0:
        raise SystemExit("❌ 重建中未找到任何 rig frame")

    # 读取点云（世界坐标）
    ply_path = Path(args.ply_path) if args.ply_path else dataset_dir / "cut_dense_merge.ply"
    if not ply_path.exists():
        raise SystemExit(f"❌ 找不到点云文件: {ply_path}")

    points_world, colors = load_ply_xyzrgb(ply_path)

    # 建立 pano_name -> frame_id 映射
    log("🔗 建立全景图名称到 frame 的映射...")
    pano_to_frame = build_pano_to_frame_mapping(recon)
    if not pano_to_frame:
        raise SystemExit("❌ 未能从 COLMAP 重建中解析任何全景帧（pano_camera*/*）")
    log(f"   ✅ 找到 {len(pano_to_frame)} 个全景 pano")

    # 背景全景图目录（用于确定输出分辨率）
    backgrounds_dir = dataset_dir / "backgrounds"
    if not backgrounds_dir.exists():
        log(f"⚠️ 背景目录不存在: {backgrounds_dir}，将使用 --width / --height 参数作为输出分辨率")

    # 输出目录
    out_depth_dir = dataset_dir / "dense_pano_depth"
    out_color_dir = dataset_dir / "dense_pano_color"
    out_depth_dir.mkdir(exist_ok=True, parents=True)
    out_color_dir.mkdir(exist_ok=True, parents=True)

    use_background_size = False

    for pano_name, frame_id in sorted(pano_to_frame.items()):
        log(f"\n📦 处理 pano: {pano_name} (frame {frame_id})")

        frame = recon.frames[frame_id]
        if not frame.has_pose():
            log("   ⚠️ 该 frame 没有有效 pose，跳过")
            continue

        # 获取该 frame 中指定 camera 的 cam_from_world
        cam_from_world = get_cam_from_world_for_frame(
            recon, frame_id, args.camera_name
        )
        if cam_from_world is None:
            log(
                f"   ⚠️ 在 frame {frame_id} 中未找到包含 '{args.camera_name}' 的图像，跳过"
            )
            continue

        # 确定输出全景分辨率
        if backgrounds_dir.exists():
            bg_img_path = backgrounds_dir / f"{pano_name}.png"
            if not bg_img_path.exists():
                bg_img_path = backgrounds_dir / f"{pano_name}.jpg"
            if bg_img_path.exists():
                with Image.open(bg_img_path) as im:
                    w_bg, h_bg = im.size
                if w_bg != 2 * h_bg:
                    log(
                        f"   ⚠️ 背景图尺寸非标准 360° 全景 ({w_bg}x{h_bg})，仍按该尺寸输出"
                    )
                width, height = w_bg, h_bg
                use_background_size = True
            else:
                width, height = args.width, args.height
        else:
            width, height = args.width, args.height

        log(f"   📐 输出分辨率: {width}x{height}")

        # 世界 -> 相机
        log("   🔄 世界坐标 → 相机坐标")
        points_cam = world_to_cam(points_world, cam_from_world)

        # 投影到 equirect
        log("   🌀 相机坐标 → 等轴柱状图 (equirect)")
        depth_map, color_map = cam_points_to_equirect(
            points_cam,
            colors,
            width=width,
            height=height,
            max_depth=args.max_depth,
        )

        # 保存
        depth_out_path = out_depth_dir / f"{pano_name}_dense_depth.npy"
        np.save(depth_out_path, depth_map.astype(np.float32))
        log(f"   💾 深度图保存: {depth_out_path}")

        if args.save_color and color_map is not None:
            color_out_path = out_color_dir / f"{pano_name}_dense_color.png"
            Image.fromarray(color_map).save(color_out_path)
            log(f"   💾 颜色图保存: {color_out_path}")

    log("\n✅ 全部处理完成")
    if not use_background_size:
        log(
            f"ℹ️ 未从背景图读取分辨率，全部 pano 使用命令行指定的 {args.width}x{args.height} 尺寸"
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将稠密点云投影到全景相机空间，生成等轴柱状深度 / 颜色图；"
            "变换矩阵与 rig / camera 关系参考 viser_rig_ply.py。"
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="STAGE 数据集中的单个场景目录，如 /root/autodl-tmp/data/STAGE1_4x/BridgeB",
    )
    parser.add_argument(
        "--colmap_root",
        type=Path,
        default=Path("/root/autodl-tmp/data/colmap_STAGE1_4x"),
        help="colmap_STAGE*_? 根目录（包含各场景子目录），默认与当前工程一致",
    )
    parser.add_argument(
        "--ply_path",
        type=Path,
        default=None,
        help="稠密点云 PLY 路径；默认使用 <dataset_dir>/cut_dense_merge.ply",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="pano_camera12",
        help="用于投影的全景相机名称子串，例如 'pano_camera12'（需与 COLMAP 图像名匹配）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4096,
        help="若无法从背景图推断尺寸时，使用的等轴柱状图宽度（必须是 height 的 2 倍）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2048,
        help="若无法从背景图推断尺寸时，使用的等轴柱状图高度（width 将被认为是其 2 倍）",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="可选，投影时忽略超过该深度的点（单位同点云），例如 150.0",
    )
    parser.add_argument(
        "--save_color",
        action="store_true",
        help="同时输出颜色 equirect 图（dense_pano_color/*.png）",
    )

    args = parser.parse_args(argv)
    if args.width != 2 * args.height:
        parser.error("width 必须等于 height 的 2 倍（360° 等轴柱状全景）")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        process_dataset(args)
    except KeyboardInterrupt:
        log("\n用户中断")
    except Exception as e:
        log(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

