import copy
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path


# ============================================================
# Utils
# ============================================================

def load_transform(path):
    if not Path(path).exists():
        print("Init transform not found, using identity.")
        return np.eye(4)
    T = np.loadtxt(path)
    print("Loaded init transform:\n", T)
    return T


def estimate_normals(pcd, voxel, factor):
    radius = voxel * factor
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=50,
        )
    )


def trimmed_icp_step(source, target, T_init,
                     max_corr,
                     max_iter,
                     trim_ratio,
                     method):
    """
    Trimmed ICP
    source -> target
    """

    source_temp = copy.deepcopy(source)
    source_temp.transform(T_init)

    kdtree = o3d.geometry.KDTreeFlann(target)
    src_pts = np.asarray(source_temp.points)

    correspondences = []
    residuals = []

    for i, pt in enumerate(src_pts):
        _, idx, dist = kdtree.search_knn_vector_3d(pt, 1)
        if dist[0] < max_corr ** 2:
            correspondences.append((i, idx[0]))
            residuals.append(dist[0])

    if len(residuals) < 10:
        print("Too few correspondences.")
        return T_init

    residuals = np.array(residuals)

    # 只保留最小的 trim_ratio 比例
    keep_num = int(len(residuals) * trim_ratio)
    if keep_num < 10:
        print("Too few trimmed correspondences.")
        return T_init

    sorted_idx = np.argsort(residuals)
    keep_idx = sorted_idx[:keep_num]

    trimmed_corr = [correspondences[i] for i in keep_idx]

    src_idx = [c[0] for c in trimmed_corr]
    tgt_idx = [c[1] for c in trimmed_corr]

    src_sel = source.select_by_index(src_idx)
    tgt_sel = target.select_by_index(tgt_idx)

    if method == "gicp":
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    elif method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        src_sel,
        tgt_sel,
        max_corr,
        T_init,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        ),
    )

    return result.transformation


# ============================================================
# Multi-scale registration
# ============================================================

def multi_scale_registration(A, B, config, T_init):

    voxel = config["voxel_size"]
    scales = [4.0, 2.0, 1.0, 0.5]

    T = T_init

    for scale in scales:
        voxel_s = voxel * scale

        print("\n=== Scale:", scale, "voxel:", voxel_s)

        A_ds = A.voxel_down_sample(voxel_s)
        B_ds = B.voxel_down_sample(voxel_s)

        estimate_normals(A_ds, voxel_s, config["normal_radius_factor"])
        estimate_normals(B_ds, voxel_s, config["normal_radius_factor"])

        max_corr = voxel_s * config["icp"]["max_correspondence_factor"]

        # ⚠️ 关键：A 做 source，小 → 大
        T = trimmed_icp_step(
            A_ds,
            B_ds,
            T,
            max_corr,
            config["icp"]["max_iteration"],
            config["robust"]["trim_ratio"],
            config["icp"]["method"]
        )

        print("Updated transform (A->B):\n", T)

    return T


# ============================================================
# Main
# ============================================================

def main():

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    A_path = config["input"]["pointcloud_A"]
    B_path = config["input"]["pointcloud_B"]
    out_path = config["output"]["aligned_B"]

    A = o3d.io.read_point_cloud(A_path)
    B = o3d.io.read_point_cloud(B_path)

    print("Loaded A:", len(A.points))
    print("Loaded B:", len(B.points))

    # ❌ 禁用 PCA
    # 只使用人工粗对齐

    if config["init"].get("use_init_transform", True):
        T_init = load_transform(config["init"]["transform_file"])
    else:
        print("Skipping init transform, using identity.")
        T_init = np.eye(4)

    # Multi-scale robust registration
    T_A_to_B = multi_scale_registration(A, B, config, T_init)

    # ========================================================
    # 关键：我们优化的是 A→B
    # 但要修改 B
    # 所以需要取逆
    # ========================================================

    T_B_to_A = np.linalg.inv(T_A_to_B)

    print("\nFinal transform (B->A):\n", T_B_to_A)

    # Apply to B
    B_aligned = copy.deepcopy(B)
    B_aligned.transform(T_B_to_A)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(out_path, B_aligned)

    np.savetxt(Path(out_path).with_suffix(".txt"), T_B_to_A, fmt="%.8f")

    print("\nRegistration finished.")
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
