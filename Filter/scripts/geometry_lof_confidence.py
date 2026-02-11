import os
import yaml
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from matplotlib import cm
import matplotlib


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(x):
    """归一化到 [0, 1]"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def get_project_root():
    """获取 filter 项目根目录（包含 config 和 scripts 目录的目录）"""
    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(scripts_dir)
    return project_root


def resolve_path(path, project_root, allow_absolute=True):
    """
    解析路径：
    - 如果是绝对路径且 allow_absolute=True，直接使用
    - 否则，相对于 project_root 解析
    """
    if os.path.isabs(path) and allow_absolute:
        return path
    return os.path.join(project_root, path)


def compute_pca_features(points, k_neighbors):
    """
    对每个点计算 PCA 特征（用于几何一致性）
    
    Args:
        points: (N, 3) 点云坐标
        k_neighbors: 邻域点数
    
    Returns:
        eigenvalues: (N, 3) 每个点的协方差矩阵特征值（降序）
        eigenvectors: (N, 3, 3) 每个点的特征向量
        centroids: (N, 3) 每个点邻域的质心
    """
    N = len(points)
    print(f"[INFO] Computing PCA features for {N} points with k={k_neighbors}...")
    
    # 构建 k-NN 搜索器
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')  # +1 因为包含自己
    nn.fit(points)
    
    eigenvalues = np.zeros((N, 3))
    eigenvectors = np.zeros((N, 3, 3))
    centroids = np.zeros((N, 3))
    
    # 对每个点计算 PCA
    for i in range(N):
        # 找 k 邻域（包括自己）
        distances, indices = nn.kneighbors([points[i]])
        indices = indices[0][1:]  # 排除自己
        neighbor_points = points[indices]
        
        # 计算质心
        centroid = neighbor_points.mean(axis=0)
        centroids[i] = centroid
        
        # 计算协方差矩阵
        centered = neighbor_points - centroid
        if len(centered) < 3:
            # 邻域点太少，使用单位矩阵
            eigenvalues[i] = [1.0, 1.0, 1.0]
            eigenvectors[i] = np.eye(3)
            continue
        
        cov = (centered.T @ centered) / len(centered)
        
        # 特征值分解
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # 按降序排列
            idx = np.argsort(eigenvals)[::-1]
            eigenvalues[i] = eigenvals[idx]
            eigenvectors[i] = eigenvecs[:, idx].T
        except:
            # 如果分解失败，使用单位矩阵
            eigenvalues[i] = [1.0, 1.0, 1.0]
            eigenvectors[i] = np.eye(3)
    
    print(f"[INFO] PCA features computed.")
    return eigenvalues, eigenvectors, centroids


def compute_geometry_confidence(points, eigenvalues, eigenvectors, centroids, cfg):
    """
    计算几何一致性置信度
    
    Args:
        points: (N, 3) 点云坐标
        eigenvalues: (N, 3) 特征值（降序）
        eigenvectors: (N, 3, 3) 特征向量
        centroids: (N, 3) 邻域质心
        cfg: 配置字典
    
    Returns:
        w_plane: (N,) 平面距离置信度
        w_structure: (N,) 结构指标置信度
    """
    N = len(points)
    w_plane = np.ones(N, dtype=np.float32)
    w_structure = np.ones(N, dtype=np.float32)
    
    geometry_cfg = cfg.get("geometry", {})
    
    # 1. 计算结构指标（Surface Variation）
    if geometry_cfg.get("structure", {}).get("enable", True):
        lambda_sum = eigenvalues.sum(axis=1)
        lambda_sum = np.maximum(lambda_sum, 1e-8)  # 避免除零
        S = eigenvalues[:, 2] / lambda_sum  # lambda_3 / (lambda_1 + lambda_2 + lambda_3)
        
        gamma = geometry_cfg.get("structure", {}).get("gamma", 10.0)
        w_structure = np.exp(-gamma * S)
        print(f"[INFO] Structure confidence: min={w_structure.min():.4f}, max={w_structure.max():.4f}, mean={w_structure.mean():.4f}")
    
    # 2. 计算点到面距离
    if geometry_cfg.get("plane_distance", {}).get("enable", True):
        # 法向量是最小特征值对应的特征向量（第3列，索引2）
        normals = eigenvectors[:, 2, :]  # (N, 3)
        
        # 计算点到平面的距离
        # d_i = |n^T (p_i - centroid)|
        point_to_centroid = points - centroids
        distances = np.abs(np.sum(normals * point_to_centroid, axis=1))
        
        beta = geometry_cfg.get("plane_distance", {}).get("beta", 20.0)
        w_plane = np.exp(-beta * distances)
        print(f"[INFO] Plane distance confidence: min={w_plane.min():.4f}, max={w_plane.max():.4f}, mean={w_plane.mean():.4f}")
    
    return w_plane, w_structure


def main(cfg_path):
    # 获取项目根目录
    project_root = get_project_root()
    
    # 解析配置文件路径
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(project_root, cfg_path)
    
    cfg = load_config(cfg_path)

    # 解析输入路径
    input_path = cfg["input"]["pointcloud_path"]
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)
    
    # 所有输出路径必须限制在项目根目录内
    out_dir = resolve_path(cfg["output"]["output_dir"], project_root, allow_absolute=False)
    log_dir = resolve_path(cfg["logs"]["log_dir"], project_root, allow_absolute=False)
    
    # 确保输出目录在项目根目录内
    out_dir = os.path.abspath(out_dir)
    log_dir = os.path.abspath(log_dir)
    project_root = os.path.abspath(project_root)
    
    def is_subpath(path, parent):
        """检查 path 是否是 parent 的子路径"""
        try:
            common = os.path.commonpath([os.path.abspath(parent), os.path.abspath(path)])
            return common == os.path.abspath(parent)
        except ValueError:
            return False
    
    if not is_subpath(out_dir, project_root):
        raise ValueError(f"输出目录必须在项目根目录内: {out_dir} (项目根目录: {project_root})")
    if not is_subpath(log_dir, project_root):
        raise ValueError(f"日志目录必须在项目根目录内: {log_dir} (项目根目录: {project_root})")
    
    ensure_dir(out_dir)
    ensure_dir(log_dir)
    
    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] 日志目录: {log_dir}")

    # 加载点云
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    print(f"[INFO] Loaded {len(points)} points")

    # 预处理
    if cfg["preprocess"]["voxel_downsample"]["enable"]:
        voxel = cfg["preprocess"]["voxel_downsample"]["voxel_size"]
        pcd = pcd.voxel_down_sample(voxel)
        points = np.asarray(pcd.points)
        print(f"[INFO] After voxel downsample: {len(points)} points")

    # ============================================================
    # Step 1: 计算 LOF 置信度
    # ============================================================
    print("\n[INFO] ===== Step 1: Computing LOF confidence =====")
    lof_cfg = cfg["lof"]
    lof = LocalOutlierFactor(
        n_neighbors=lof_cfg["n_neighbors"],
        metric=lof_cfg["metric"]
    )
    lof.fit(points)
    lof_scores = -lof.negative_outlier_factor_

    conf_cfg = cfg["confidence"]
    alpha = conf_cfg["exponential"]["alpha"]
    w_lof = np.exp(-alpha * np.maximum(0.0, lof_scores - 1.0))
    print(f"[INFO] LOF confidence: min={w_lof.min():.4f}, max={w_lof.max():.4f}, mean={w_lof.mean():.4f}")

    # ============================================================
    # Step 2: 计算几何一致性
    # ============================================================
    print("\n[INFO] ===== Step 2: Computing geometry confidence =====")
    geometry_cfg = cfg.get("geometry", {})
    k_neighbors = geometry_cfg.get("k_neighbors", 30)
    
    eigenvalues, eigenvectors, centroids = compute_pca_features(points, k_neighbors)
    w_plane, w_structure = compute_geometry_confidence(points, eigenvalues, eigenvectors, centroids, cfg)

    # ============================================================
    # Step 3: 融合置信度
    # ============================================================
    print("\n[INFO] ===== Step 3: Fusing confidence =====")
    fusion_cfg = cfg.get("fusion", {})
    alpha_lof = fusion_cfg.get("alpha_lof", 1.0)
    alpha_plane = fusion_cfg.get("alpha_plane", 1.0)
    alpha_structure = fusion_cfg.get("alpha_structure", 1.0)
    
    # 加权融合
    confidence = (w_lof ** alpha_lof) * (w_plane ** alpha_plane) * (w_structure ** alpha_structure)
    
    # 裁剪到有效范围
    min_weight = conf_cfg.get("min_weight", 0.05)
    confidence = np.clip(confidence, min_weight, 1.0)
    
    print(f"[INFO] Final confidence: min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}")

    # ============================================================
    # 保存结果
    # ============================================================
    # 始终保存置信度文件（用于后续可视化）
    confidence_file = os.path.join(out_dir, "geometry_confidence.npy")
    np.save(confidence_file, confidence)
    print(f"[INFO] Saved confidence to {confidence_file}")
    
    # 如果进行了下采样，保存下采样后的点云（用于可视化）
    voxel_downsample_enabled = cfg["preprocess"]["voxel_downsample"]["enable"]
    if voxel_downsample_enabled:
        downsampled_ply_path = os.path.join(out_dir, "pointcloud_downsampled.ply")
        o3d.io.write_point_cloud(downsampled_ply_path, pcd)
        print(f"[INFO] Saved downsampled point cloud to {downsampled_ply_path}")
        pointcloud_for_vis = downsampled_ply_path
    else:
        pointcloud_for_vis = input_path
    
    # 同时保存点云路径信息（用于可视化脚本）
    info_file = os.path.join(out_dir, "confidence_info.txt")
    with open(info_file, "w") as f:
        f.write(f"pointcloud_path={pointcloud_for_vis}\n")
        f.write(f"original_pointcloud_path={input_path}\n")
        f.write(f"confidence_file={confidence_file}\n")
        f.write(f"num_points={len(points)}\n")
        f.write(f"confidence_min={confidence.min():.6f}\n")
        f.write(f"confidence_max={confidence.max():.6f}\n")
        f.write(f"confidence_mean={confidence.mean():.6f}\n")
        f.write(f"voxel_downsample_enabled={voxel_downsample_enabled}\n")
        if voxel_downsample_enabled:
            f.write(f"voxel_size={cfg['preprocess']['voxel_downsample']['voxel_size']}\n")
    print(f"[INFO] Saved confidence info to {info_file}")

    # 可选过滤
    if cfg["filter"]["enable"]:
        keep = confidence > cfg["filter"]["min_confidence"]
        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points[keep])
        if pcd.has_colors():
            filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
        o3d.io.write_point_cloud(
            os.path.join(out_dir, cfg["output"]["filtered_ply"]),
            filtered
        )
        print(f"[INFO] Filtered points: {keep.sum()}/{len(points)}")

    # ============================================================
    # 可视化
    # ============================================================
    print("\n[INFO] ===== Generating visualizations =====")
    cmap_name = cfg["visualization"]["colormap"]
    # Matplotlib 3.7+: cm.get_cmap is deprecated; prefer matplotlib.colormaps
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        # Fallback for older Matplotlib
        cmap = cm.get_cmap(cmap_name)

    # LOF 着色
    lof_norm = normalize(lof_scores)
    lof_colors = cmap(lof_norm)[:, :3]
    pcd_lof = o3d.geometry.PointCloud()
    pcd_lof.points = o3d.utility.Vector3dVector(points)
    pcd_lof.colors = o3d.utility.Vector3dVector(lof_colors)
    o3d.io.write_point_cloud(os.path.join(log_dir, "lof_colored.ply"), pcd_lof)

    # 平面性指标着色
    lambda_sum = eigenvalues.sum(axis=1)
    lambda_sum = np.maximum(lambda_sum, 1e-8)
    S = eigenvalues[:, 2] / lambda_sum
    S_norm = normalize(S)
    planarity_colors = cmap(S_norm)[:, :3]
    pcd_planarity = o3d.geometry.PointCloud()
    pcd_planarity.points = o3d.utility.Vector3dVector(points)
    pcd_planarity.colors = o3d.utility.Vector3dVector(planarity_colors)
    o3d.io.write_point_cloud(os.path.join(log_dir, "planarity_colored.ply"), pcd_planarity)

    # 点到面距离着色
    normals = eigenvectors[:, 2, :]
    point_to_centroid = points - centroids
    distances = np.abs(np.sum(normals * point_to_centroid, axis=1))
    dist_norm = normalize(distances)
    distance_colors = cmap(dist_norm)[:, :3]
    pcd_distance = o3d.geometry.PointCloud()
    pcd_distance.points = o3d.utility.Vector3dVector(points)
    pcd_distance.colors = o3d.utility.Vector3dVector(distance_colors)
    o3d.io.write_point_cloud(os.path.join(log_dir, "distance_colored.ply"), pcd_distance)

    # 最终置信度着色
    conf_colors = cmap(confidence)[:, :3]
    pcd_conf = o3d.geometry.PointCloud()
    pcd_conf.points = o3d.utility.Vector3dVector(points)
    pcd_conf.colors = o3d.utility.Vector3dVector(conf_colors)
    o3d.io.write_point_cloud(os.path.join(log_dir, "confidence_colored.ply"), pcd_conf)

    # 直方图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(lof_scores, bins=100, alpha=0.7)
    plt.xlabel("LOF score")
    plt.ylabel("Count")
    plt.title("LOF Distribution")
    
    plt.subplot(1, 3, 2)
    plt.hist(S, bins=100, alpha=0.7, color='green')
    plt.xlabel("Surface Variation (S)")
    plt.ylabel("Count")
    plt.title("Planarity Distribution")
    
    plt.subplot(1, 3, 3)
    plt.hist(confidence, bins=100, alpha=0.7, color='red')
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Final Confidence Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confidence_histograms.png"), dpi=150)
    plt.close()

    print("[INFO] Done.")


if __name__ == "__main__":
    import sys
    
    # 如果没有提供参数，使用默认配置文件
    if len(sys.argv) == 1:
        project_root = get_project_root()
        default_config = os.path.join(project_root, "config", "geometry_lof.yaml")
        if os.path.exists(default_config):
            print(f"ℹ️  未提供配置文件，使用默认配置: {default_config}")
            cfg_path = default_config
        else:
            print("❌ 错误: 未提供配置文件参数")
            print(f"   用法: python {sys.argv[0]} config/geometry_lof.yaml")
            print(f"   或者将配置文件放在: {default_config}")
            sys.exit(1)
    elif len(sys.argv) == 2:
        cfg_path = sys.argv[1]
    else:
        print("❌ 错误: 参数过多")
        print(f"   用法: python {sys.argv[0]} [config/geometry_lof.yaml]")
        sys.exit(1)
    
    main(cfg_path)
