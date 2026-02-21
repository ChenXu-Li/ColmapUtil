"""
An example for running incremental SfM on 360 spherical panorama images.
"""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
import logging


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


PANO_RENDER_OPTIONS: dict[str, PanoRenderOptions] = {
    "overlapping": PanoRenderOptions(
        num_steps_yaw=12,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    # Cubemap without top and bottom images.
    "non-overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}


def create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return pycolmap.Camera.create(
        0, "SIMPLE_PINHOLE", focal, image_width, image_height
    )


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(
    image_size: tuple[int, int], rays_in_cam: np.ndarray
) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama."""
    # Assuming that the panos are approximately upright.
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        render_options: PanoRenderOptions,
    ):
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)

        # We assign each pano pixel to the virtual camera
        # with the closest camera center.
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()

        # These are initialized on the first pano image
        # to avoid recomputing the rays for each pano image.
        self._camera = None
        self._pano_size = None
        self._rays_in_cam = None

    def process(self, pano_name: str) -> None:
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            return

        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        with self._lock:
            if self._camera is None:  # First image, precompute rays once.
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            else:  # Later images, verify consistent panoramas.
                if (pano_width, pano_height) != self._pano_size:
                    raise ValueError(
                        "Panoramas of different sizes are not supported."
                    )

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )

            image_name = (
                self.rig_config.cameras[cam_idx].image_prefix + pano_name
            )
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    render_options: PanoRenderOptions,
) -> pycolmap.RigConfig:
    processor = PanoProcessor(
        pano_image_dir, output_image_dir, mask_dir, render_options
    )

    num_panos = len(pano_image_names)
    max_workers = min(32, (os.cpu_count() or 2) - 1)

    with tqdm(total=num_panos) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    return processor.rig_config


def run(args: argparse.Namespace) -> None:
    pycolmap.set_random_seed(0)
    
    # 设置环境变量以避免某些段错误和OpenBLAS线程问题
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 关键：限制OpenBLAS线程数
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    
    logging.info("=" * 60)
    logging.info("Starting COLMAP panorama reconstruction")
    logging.info(f"Input images: {args.input_image_path}")
    logging.info(f"Output path: {args.output_path}")
    logging.info(f"Matcher: {args.matcher}")
    logging.info(f"Pano render type: {args.pano_render_type}")
    logging.info("Thread environment variables:")
    logging.info(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    logging.info(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    logging.info(f"  OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    logging.info("=" * 60)

    # Define the paths.
    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    # Search for input images.
    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")

    rig_config = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        PANO_RENDER_OPTIONS[args.pano_render_type],
    )

    # 启用GPU加速特征提取
    feature_opts = pycolmap.FeatureExtractionOptions()
    feature_opts.use_gpu = True
    feature_opts.gpu_index = "0"  # GPU索引需要是字符串类型
    feature_opts.num_threads = 4  # 限制CPU线程数，避免OpenBLAS问题
    
    # 创建ImageReaderOptions
    # 注意：camera_mode=PER_FOLDER 会自动确保同一文件夹下的所有图像使用同一个相机ID
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.mask_path = str(mask_dir)
    
    logging.info("Extracting features...")
    try:
        pycolmap.extract_features(
            str(database_path),
            str(image_dir),
            reader_options=reader_opts,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            extraction_options=feature_opts,  # 正确的参数名
        )
        logging.info("Feature extraction completed.")
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return

    logging.info("Applying rig configuration...")
    with pycolmap.Database.open(database_path) as db:
        # 在应用 rig 配置之前，检查并修复相机ID不一致的问题
        # 对于每个 rig camera 的前缀，确保所有图像使用同一个相机ID
        logging.info("Checking camera ID consistency for rig cameras...")
        images = db.read_all_images()
        prefix_to_camera_ids = {}
        prefix_to_images = {}
        
        for image in images:
            for rig_camera in rig_config.cameras:
                if image.name.startswith(rig_camera.image_prefix):
                    prefix = rig_camera.image_prefix
                    if prefix not in prefix_to_camera_ids:
                        prefix_to_camera_ids[prefix] = set()
                        prefix_to_images[prefix] = []
                    prefix_to_camera_ids[prefix].add(image.camera_id)
                    prefix_to_images[prefix].append(image)
                    break
        
        # 检查是否有不一致的相机ID
        inconsistent_prefixes = []
        for prefix, camera_ids in prefix_to_camera_ids.items():
            if len(camera_ids) > 1:
                inconsistent_prefixes.append((prefix, camera_ids))
                logging.warning(
                    f"Found inconsistent camera IDs for prefix '{prefix}': {camera_ids}. "
                    f"Will unify them to use the first camera ID."
                )
        
        # 统一相机ID：对于每个前缀，使用第一个遇到的相机ID
        if inconsistent_prefixes:
            for prefix, camera_ids in inconsistent_prefixes:
                target_camera_id = min(camera_ids)  # 使用最小的相机ID
                for image in prefix_to_images[prefix]:
                    if image.camera_id != target_camera_id:
                        logging.info(
                            f"Updating image '{image.name}' camera_id from {image.camera_id} to {target_camera_id}"
                        )
                        image.camera_id = target_camera_id
                        db.update_image(image)
        
        pycolmap.apply_rig_config([rig_config], db)
        # 诊断信息
        num_images = db.num_images()
        num_keypoints = db.num_keypoints()
        logging.info(f"Database contains {num_images} images and {num_keypoints} keypoints.")

    logging.info(f"Starting feature matching with {args.matcher} matcher...")
    matching_options = pycolmap.FeatureMatchingOptions()
    # 启用GPU加速特征匹配
    matching_options.use_gpu = True
    matching_options.gpu_index = "0"  # GPU索引需要是字符串类型
    # 限制匹配线程数，避免OpenBLAS内存问题
    if hasattr(matching_options, 'num_threads'):
        matching_options.num_threads = 4
    # We have perfect sensor_from_rig poses (except for potential stitching
    # artifacts by the spherical image provider), so we can perform geometric
    # verification using rig constraints.
    matching_options.rig_verification = True
    # The images within a frame do not have overlap due to the provided masks.
    matching_options.skip_image_pairs_in_same_frame = True
    # 获取本地vocab tree路径（如果存在）
    local_vocab_tree = Path.home() / ".cache" / "colmap" / "vocab_tree_faiss_flickr100K_words256K.bin"
    vocab_tree_path = str(local_vocab_tree) if local_vocab_tree.exists() else None
    if vocab_tree_path:
        logging.info(f"Using local vocab tree: {vocab_tree_path}")
    
    try:
        if args.matcher == "sequential":
            # 配置sequential pairing options（直接设置vocab_tree_path）
            seq_pairing_kwargs = {"loop_detection": True}
            if vocab_tree_path:
                seq_pairing_kwargs["vocab_tree_path"] = vocab_tree_path
            
            pycolmap.match_sequential(
                database_path,
                pairing_options=pycolmap.SequentialPairingOptions(**seq_pairing_kwargs),
                matching_options=matching_options,
            )
        elif args.matcher == "exhaustive":
            pycolmap.match_exhaustive(
                database_path, matching_options=matching_options
            )
        elif args.matcher == "vocabtree":
            pycolmap.match_vocabtree(
                database_path, matching_options=matching_options
            )
        elif args.matcher == "spatial":
            pycolmap.match_spatial(database_path, matching_options=matching_options)
        else:
            logging.fatal(f"Unknown matcher: {args.matcher}")
            return
        logging.info("Feature matching completed.")
    except Exception as e:
        logging.error(f"Feature matching failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 检查匹配结果
    try:
        with pycolmap.Database.open(database_path) as db:
            num_matches = db.num_matched_image_pairs()
            logging.info(f"Found {num_matches} matched image pairs.")
            if num_matches == 0:
                logging.warning("No image pairs were matched! This will cause reconstruction to fail.")
                logging.warning("Possible reasons:")
                logging.warning("  1. Images have insufficient overlap")
                logging.warning("  2. Feature extraction failed for some images")
                logging.warning("  3. Rig configuration issues")
    except Exception as e:
        logging.warning(f"Could not check match statistics: {e}")

    # 获取默认的mapper选项并放宽约束
    mapper_opts = pycolmap.IncrementalMapperOptions()
    # 放宽初始化约束，让更多frames能被注册
    mapper_opts.init_min_num_inliers = 30  # 进一步降低从50到30
    mapper_opts.init_max_error = 12.0  # 进一步增大从8.0到12.0
    mapper_opts.init_min_tri_angle = 4.0  # 进一步降低从8.0到4.0
    mapper_opts.init_max_reg_trials = 10  # 增加从5到10
    # 放宽绝对位姿估计约束
    mapper_opts.abs_pose_min_num_inliers = 10  # 进一步降低从15到10
    mapper_opts.abs_pose_min_inlier_ratio = 0.10  # 进一步降低从0.15到0.10
    mapper_opts.abs_pose_max_error = 20.0  # 进一步增大从16.0到20.0
    # 放宽三角化约束
    mapper_opts.filter_min_tri_angle = 0.5  # 进一步降低从1.0到0.5
    mapper_opts.filter_max_reproj_error = 8.0  # 进一步增大从6.0到8.0
    # 放宽其他约束（只设置存在的属性）
    if hasattr(mapper_opts, 'max_num_extra_reg_trials'):
        mapper_opts.max_num_extra_reg_trials = 5  # 增加额外注册尝试次数
    if hasattr(mapper_opts, 'min_num_matches'):
        mapper_opts.min_num_matches = 5  # 降低最小匹配数
    
    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
        mapper=mapper_opts,
        min_num_matches=5,  # 进一步降低从10到5
        min_model_size=3,  # 进一步降低从5到3
        init_num_trials=500,  # 增加从300到500
    )
    
    logging.info("Starting incremental mapping...")
    try:
        recs = pycolmap.incremental_mapping(
            database_path, image_dir, rec_path, opts
        )
        logging.info(f"Incremental mapping completed. Found {len(recs)} reconstruction(s).")
    except Exception as e:
        logging.error(f"Incremental mapping failed with exception: {e}")
        logging.error("This might be due to insufficient matches or data quality issues.")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    if not recs:
        logging.warning("No reconstructions were generated. This might indicate:")
        logging.warning("  1. Insufficient feature matches between images")
        logging.warning("  2. Poor image quality or insufficient overlap")
        logging.warning("  3. Issues with the rig configuration")
        return
    
    for idx, rec in recs.items():
        try:
            # 安全地获取重建摘要
            summary = rec.summary()
            logging.info(f"#{idx} {summary}")
        except Exception as e:
            # 如果summary()失败，手动输出关键信息
            logging.warning(f"#{idx} Reconstruction summary() failed: {e}")
            try:
                logging.info(f"#{idx} Reconstruction:")
                logging.info(f"        num_rigs = {len(rec.rigs)}")
                logging.info(f"        num_cameras = {len(rec.cameras)}")
                logging.info(f"        num_frames = {len(rec.frames)}")
                logging.info(f"        num_reg_frames = {sum(1 for f in rec.frames.values() if f.has_pose())}")
                logging.info(f"        num_images = {len(rec.images)}")
                logging.info(f"        num_points3D = {len(rec.points3D)}")
                if len(rec.points3D) > 0:
                    num_obs = sum(len(p.track.elements) for p in rec.points3D.values())
                    logging.info(f"        num_observations = {num_obs}")
                    if len(rec.images) > 0:
                        logging.info(f"        mean_observations_per_image = {num_obs / len(rec.images):.6f}")
            except Exception as e2:
                logging.error(f"Failed to get reconstruction details: {e2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=["sequential", "exhaustive", "vocabtree", "spatial"],
    )
    parser.add_argument(
        "--pano_render_type",
        default="overlapping",
        choices=list(PANO_RENDER_OPTIONS.keys()),
    )
    run(parser.parse_args())