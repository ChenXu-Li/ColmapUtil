"""
从 calib.json 写出经典 COLMAP 二进制（cameras.bin + images.bin，可选 points3D.bin，无 rigs/frames）。
新版 pycolmap 的 Reconstruction.write_binary 会附带 rigs.bin / frames.bin 等；
若下游只认旧版三件套或不想引入 rig，请用本仓库的 read_write_model（COLMAP 官方脚本）。

calib.json 约定：
  - cameras: 键 "001"… 与 int 相机 id 对应（int("001")==1）
  - camera_poses: 每相机 R(3x3), T(3,) 为世界坐标系到相机坐标系（与 COLMAP Image 一致）
图片与相机按排序后一一对应：相机 id 升序，图片路径升序。

gsplat simple_trainer（不修改 gsplat 源码时）：
  - 默认会额外写入「点数为零」的 points3D.bin，否则加载器会因缺少文件报错。
  - 随机初始化高斯请使用：--init-type random
  - 空点云无法在默认场景归一化里估计主方向，需同时：--normalize-world-space False
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from PIL import Image

import read_write_model as rw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SUFFIXES |= {s.upper() for s in IMAGE_SUFFIXES}


def _sorted_camera_keys(cameras: dict) -> list[str]:
    return sorted(cameras.keys(), key=lambda k: int(k))


def image_size_to_wh(c: dict, layout: str) -> tuple[int, int]:
    """返回 (width, height)。layout=wh 时 image_size 为 [宽, 高]；hw 时为 [高, 宽]。"""
    a, b = c["image_size"]
    if layout == "wh":
        return int(a), int(b)
    if layout == "hw":
        return int(b), int(a)
    raise ValueError(f"未知 image_size_layout: {layout}")


def calib_entry_to_camera(
    cam_key: str, c: dict, new_wh: tuple[int, int] | None, layout: str
) -> rw.Camera:
    w, h = image_size_to_wh(c, layout)
    K = c["K"]
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    d = c["dist"][0]
    k1, k2, p1, p2, k3 = d[:5]
    if new_wh is not None:
        nw, nh = new_wh
        sx = nw / float(w)
        sy = nh / float(h)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
        w, h = nw, nh
    # OPENCV（8 参）：gsplat 自带 pycolmap SceneManager 不支持 FULL_OPENCV
    params = np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64)
    cam_id = int(cam_key)
    return rw.Camera(
        id=cam_id,
        model="OPENCV",
        width=int(w),
        height=int(h),
        params=params,
    )


def calib_json_to_cameras(
    data: dict,
    image_sizes_by_cam: dict[int, tuple[int, int]] | None,
    layout: str,
) -> dict:
    cameras = {}
    for cam_key in data["cameras"]:
        c = data["cameras"][cam_key]
        cid = int(cam_key)
        new_wh = image_sizes_by_cam.get(cid) if image_sizes_by_cam else None
        cameras[cid] = calib_entry_to_camera(cam_key, c, new_wh, layout)
    return cameras


def collect_paths_from_dir(images_dir: Path) -> list[Path]:
    paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda p: p.name)


def load_paths_from_file(list_path: Path) -> list[Path]:
    lines = list_path.read_text(encoding="utf-8").strip().splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def build_images_and_scaled_cameras(
    data: dict,
    image_paths: list[Path],
    name_mode: str,
    images_root: Path | None,
    layout: str,
) -> tuple[dict, dict]:
    cam_keys = _sorted_camera_keys(data["cameras"])
    if len(image_paths) != len(cam_keys):
        raise ValueError(
            f"相机数 {len(cam_keys)} 与图片数 {len(image_paths)} 不一致，请检查顺序或路径。"
        )
    if "camera_poses" not in data:
        raise KeyError("calib.json 中缺少 camera_poses，无法写 images.bin")

    image_sizes_by_cam: dict[int, tuple[int, int]] = {}
    for cam_key, img_path in zip(cam_keys, image_paths):
        cid = int(cam_key)
        if not img_path.is_file():
            raise FileNotFoundError(f"图片不存在: {img_path}")
        image_sizes_by_cam[cid] = read_image_size(img_path)

    cameras = calib_json_to_cameras(data, image_sizes_by_cam, layout)
    images = {}
    for idx, (cam_key, img_path) in enumerate(zip(cam_keys, image_paths), start=1):
        pose = data["camera_poses"].get(cam_key)
        if pose is None:
            raise KeyError(f"camera_poses 中缺少相机 {cam_key}")
        R = np.array(pose["R"], dtype=np.float64)
        T = np.array(pose["T"], dtype=np.float64)
        qvec = rw.rotmat2qvec(R)
        cid = int(cam_key)
        if name_mode == "basename":
            name = img_path.name
        elif name_mode == "relative":
            if images_root is None:
                raise ValueError("name_mode=relative 时需要 --images_root")
            name = str(img_path.resolve().relative_to(images_root.resolve()))
        else:
            name = str(img_path)

        images[idx] = rw.Image(
            id=idx,
            qvec=qvec,
            tvec=T,
            camera_id=cid,
            name=name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros(0, dtype=np.int64),
        )
    return cameras, images


def main():
    parser = argparse.ArgumentParser(
        description="从 calib.json 写出 COLMAP cameras.bin / images.bin"
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path("/home/lcx/data/test_frame15/calib.json"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/lcx/data/test_frame15/sparse/0"),
        help="输出目录，将写入 cameras.bin 与 images.bin",
    )
    parser.add_argument(
        "--empty_images",
        action="store_true",
        help="仅写标定、不写图像位姿（images.bin 中图像数为 0）",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("/home/lcx/data/test_frame15/images"),
        help="含全部图像的目录（与相机一一对应，按文件名排序）",
    )
    parser.add_argument(
        "--images_list",
        type=Path,
        default=None,
        help="文本文件，每行一张图片的绝对或相对路径，顺序与相机 id 升序一致",
    )
    parser.add_argument(
        "--image_name_mode",
        choices=("basename", "relative", "absolute"),
        default="basename",
        help="写入 images.bin 的图像名字段：仅文件名 / 相对 images_root / 绝对路径",
    )
    parser.add_argument(
        "--images_root",
        type=Path,
        default=None,
        help="配合 image_name_mode=relative，用于计算相对路径",
    )
    parser.add_argument(
        "--image_size_layout",
        choices=("wh", "hw"),
        # 约定 calib.json 的 image_size 是否为 [width, height] 还是 [height, width]。
        # 对本仓库当前的 test_frame15 数据集，calib.json 写的是 [height, width]，
        # 因此默认取 hw；如你的 calib.json 使用 [width, height]，请显式传 --image_size_layout wh
        default="hw",
        help="calib 中 image_size 的含义：[宽,高]=wh；[高,宽]=hw（与实际 JPEG 宽高对不上时可试 hw）",
    )
    parser.add_argument(
        "--no_empty_points3d",
        action="store_true",
        help="不写入 points3D.bin（默认写入点数为零的 points3D.bin，供 gsplat 等检查文件存在）",
    )
    args = parser.parse_args()

    with open(args.calib, encoding="utf-8") as f:
        data = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.empty_images:
        cameras = calib_json_to_cameras(data, None, args.image_size_layout)
        images = {}
    else:
        if args.images_list is not None:
            image_paths = load_paths_from_file(args.images_list)
        elif args.images_dir is not None:
            image_paths = collect_paths_from_dir(args.images_dir)
        else:
            raise SystemExit(
                "未指定 --empty_images 时，必须提供 --images_dir 或 --images_list，"
                "以便写入位姿并与下采样后的真实分辨率对齐内参。"
            )
        cameras, images = build_images_and_scaled_cameras(
            data,
            image_paths,
            name_mode=args.image_name_mode,
            images_root=args.images_root,
            layout=args.image_size_layout,
        )

    rw.write_cameras_binary(cameras, str(args.out_dir / "cameras.bin"))
    rw.write_images_binary(images, str(args.out_dir / "images.bin"))
    print(
        f"Wrote {args.out_dir / 'cameras.bin'} ({len(cameras)} cams) and "
        f"{args.out_dir / 'images.bin'} ({len(images)} images)"
    )
    if not args.no_empty_points3d:
        p3d_path = args.out_dir / "points3D.bin"
        rw.write_points3D_binary({}, str(p3d_path))
        print(
            f"Wrote empty {p3d_path} (gsplat random init: also use "
            f"--init-type random --normalize-world-space False)"
        )


if __name__ == "__main__":
    main()
