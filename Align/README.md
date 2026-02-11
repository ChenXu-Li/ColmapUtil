## Robust Point Cloud Registration (Partial Overlap)

本子模块实现 **部分重叠 / 尺度可能不同 / 初始位姿差异很大** 的两团点云的刚体（或相似变换）配准。

- **输入**：点云 **A（参考系 / 不动）**、**B（待对齐 / 运动）**
- **输出**：对齐后的点云 **B_aligned.ply**，其坐标系与 A 一致

整体思路：**粗到细（Global → Local）**，先通过 FPFH + RANSAC 获得一个鲁棒的全局初始位姿，再用 ICP 做局部精配准，可扩展到 Sim(3)。

---

## 一、整体流程概览

```text
A.ply (fixed)
   ↑
   │  Global registration (FPFH + RANSAC)
   │
B.ply (moving)
   ↓
Local refinement (Robust ICP)
   ↓
B_aligned.ply
```

---

## 二、目录结构

```text
Align/
├── README.md      # 当前文档
├── config.yaml    # 所有参数配置
├── align.py       # 主配准脚本
└── run.sh         # 一键运行脚本
```

---

## 三、配置文件 `config.yaml` 及各参数含义

当前默认配置如下：

```yaml
# ===== Input / Output =====
input:
  pointcloud_A: "data/A.ply"   # reference (fixed)
  pointcloud_B: "data/B.ply"   # moving

output:
  aligned_B: "output/B_aligned.ply"

# ===== Global registration =====
voxel_size: 0.05        # 下采样尺度（米）
fpfh_radius_factor: 5.0
normal_radius_factor: 2.0
ransac:
  max_iteration: 100000
  confidence: 0.999
  max_correspondence_factor: 1.5

# ===== ICP refinement =====
icp:
  method: "point_to_plane"   # point_to_point | point_to_plane
  max_correspondence_factor: 0.4
  max_iteration: 50

# ===== Robust / partial overlap =====
robust:
  enable: true
  trim_ratio: 0.7        # 仅保留 residual 最小的 70% 匹配点

# ===== Optional =====
scale_estimation: false   # true -> Sim(3)
visualize: true
```

下面逐项解释各参数的意义和调节建议。

### 1）输入 / 输出

- **`input.pointcloud_A`**  
  - 含义：参考点云 A 的路径（坐标系将作为最终对齐坐标系）。  
  - 建议：一般选为全局/尺度正确的一帧或合成点云；应尽量完整、干净。

- **`input.pointcloud_B`**  
  - 含义：待对齐点云 B 的路径，将被变换到 A 的坐标系中。  
  - 建议：可以是部分扫描、另一视角的重建结果等。

- **`output.aligned_B`**  
  - 含义：输出对齐后的 B 点云文件路径。  
  - 建议：可改为你自己的工程路径结构，例如 `output/scene1_B_aligned.ply`。

### 2）全局配准（Global registration）

- **`voxel_size`**  
  - 含义：体素下采样的体素边长（米）。用来减小点数，加速特征和 RANSAC。  
  - 影响：  
    - 数值 **越大**：点数更少，计算更快，但几何细节损失更多，可能降低精度。  
    - 数值 **越小**：点数更多，细节保留好，但计算更慢、RANSAC 容易过拟合噪声。  
  - 调参建议：  
    - 对尺度在 \([0.5\,\mathrm{m}, 5\,\mathrm{m}]\) 的室内场景，`0.03~0.05` 通常比较合适。  
    - 对更大尺度场景，按场景尺寸线性放大。

- **`fpfh_radius_factor`**  
  - 含义：计算 FPFH 特征时的半径 = `voxel_size * fpfh_radius_factor`。  
  - 影响：  
    - 半径太小：特征只看到很局部的噪声，不稳定。  
    - 半径太大：特征被过度平滑，区分度下降。  
  - 调参建议：保持在 `3.0 ~ 7.0` 区间，场景越大可适当增大。

- **`normal_radius_factor`**  
  - 含义：估计法向时的邻域半径 = `voxel_size * normal_radius_factor`。  
  - 影响：  
    - 半径太小：法向抖动大，影响 FPFH 和 ICP。  
    - 半径太大：细小结构会被平滑掉。  
  - 调参建议：一般比 `1.5` 大一些即可，默认 `2.0` 对大部分场景比较稳健。

- **`ransac.max_iteration`**  
  - 含义：RANSAC 最大迭代次数。  
  - 影响：  
    - 越大：更有机会找到正确的全局配准，但时间更长。  
    - 越小：运行快，但在高噪声或低重叠场景下可能失败。  
  - 调参建议：  
    - 调试时可先设为 `5e4` 降低时间。  
    - 正式对齐、重叠较小/噪声较大时保持 `1e5` 或更高。

- **`ransac.confidence`**  
  - 含义：RANSAC 所需的置信度（即找到“足够好”解的概率）。  
  - 范围：`(0, 1)`，越接近 1 迭代次数期望越大。  
  - 调参建议：  
    - 一般不改动，`0.99 ~ 0.999` 是常见取值。  
    - 对时间极其敏感时可轻微降低，例如 `0.995`。

- **`ransac.max_correspondence_factor`**  
  - 含义：RANSAC 中两点可视为匹配的最大距离 = `voxel_size * max_correspondence_factor`。  
  - 影响：  
    - 过小：很多真实匹配被丢弃，RANSAC 难以找到解。  
    - 过大：错误匹配太多，RANSAC 需要更多迭代才能排除。  
  - 调参建议：  
    - 对干净点云，`1.0 ~ 1.5` 通常够用。  
    - 对噪声大或有尺度偏差的情况，可尝试 `2.0` 以上。

### 3）ICP 精配准（Local refinement）

- **`icp.method`**  
  - 含义：ICP 的误差度量方式。  
  - 取值：  
    - `"point_to_point"`：点到点 ICP，对法向依赖小，适合噪声较大但表面不明显的点云。  
    - `"point_to_plane"`：点到面 ICP，收敛速度快、精度高，但依赖较好的法向估计。  
  - 调参建议：  
    - 若法向可靠（有规律的表面）推荐 `"point_to_plane"`（默认）。  
    - 若法向质量存疑，可先尝试 `"point_to_point"`。

- **`icp.max_correspondence_factor`**  
  - 含义：ICP 中允许的最大对应点距离 = `voxel_size * max_correspondence_factor`。  
  - 影响：  
    - 过大：错误对应较多，可能拖慢或拉偏收敛。  
    - 过小：只有很近的点才能配对，若初始位姿不够好可能无法收敛。  
  - 调参建议：  
    - 有较好 RANSAC 初始位姿时，`0.3 ~ 0.5` 较合适。  
    - 若初始化不太准，可以适当放宽到 `0.7` 左右。

- **`icp.max_iteration`**  
  - 含义：ICP 的最大迭代次数。  
  - 影响：  
    - 越大：允许更长的收敛过程，时间增加。  
    - 越小：可能在尚未收敛前就停止。  
  - 调参建议：  
    - 一般 `30~100` 即可，默认 `50` 是折中选择。  
    - 可通过打印残差或可视化结果判断是否需要增加。

### 4）鲁棒 / 部分重叠设置

- **`robust.enable`**  
  - 含义：是否启用“截断/裁剪”策略，仅保留低残差的一部分匹配点来减小 outlier 影响。  
  - 影响：  
    - `true`：对部分重叠/强 outlier 场景更鲁棒。  
    - `false`：使用普通 ICP，全量匹配参与优化。  
  - 调参建议：  
    - 对部分重叠场景建议始终开启。  
    - 对几乎全重叠且噪声不大的场景，可以关闭以获得略微更稳定的几何拟合。

- **`robust.trim_ratio`**  
  - 含义：在所有匹配对中，仅保留残差最小的 `trim_ratio` 比例参与优化。  
  - 取值范围：`(0, 1]`。  
  - 影响：  
    - 值越小：更“苛刻”，只相信少量误差很小的匹配，鲁棒性高，但有效点少。  
    - 值越大：接近普通 ICP，鲁棒性下降。  
  - 调参建议：  
    - 部分重叠且 outlier 较多时，可用 `0.6 ~ 0.8`。  
    - 几乎全重叠、少 outlier 时可以设为 `0.9` 以上甚至 `1.0`。

### 5）可选设置

- **`scale_estimation`**  
  - 含义：是否同时估计尺度（Sim(3)），即允许 B 发生各向同性缩放后再对齐。  
  - 当前实现：主流程仍按刚体（SE(3)）配准设计，该开关为后续扩展预留。  
  - 使用建议：当前版本保持 `false`，如需 Sim(3) 需额外实现对应逻辑。

- **`visualize`**  
  - 含义：配准完成后是否弹出可视化窗口。  
  - 影响：  
    - `true`：调用 Open3D 的 `draw_geometries`，展示 A 与 B_aligned。  
    - `false`：只写出结果文件，不弹窗。  
  - 使用建议：  
    - 调试阶段建议开启，快速肉眼检查对齐质量。  
    - 批处理/服务器环境建议关闭以避免 GUI 问题。

---

## 四、核心脚本 `align.py` 行为说明

`align.py` 的主要流程：

1. **读取配置**：从 `config.yaml` 中读取所有参数。  
2. **加载点云**：用 Open3D 读入 `pointcloud_A` 和 `pointcloud_B`。  
3. **预处理**：  
   - 以 `voxel_size` 做体素下采样；  
   - 以 `normal_radius_factor` 计算法向；  
   - 以 `fpfh_radius_factor` 计算 FPFH 特征。  
4. **全局配准**：使用 FPFH + RANSAC 获取初始位姿（受 `ransac.*` 参数控制）。  
5. **ICP 精配准**：以 RANSAC 结果为初值运行 ICP（受 `icp.*` 和 `robust.*` 参数影响）。  
6. **保存 & 可视化**：写出 `aligned_B`，根据 `visualize` 决定是否可视化。

如需改进或插入自己的算法，可以从 `global_register` 或 `robust_icp` 这两个函数入手。

---

## 五、运行方式

在 `Align` 目录下准备好：

- `config.yaml`（按需修改输入输出路径和参数）；  
- 对应的 `data/A.ply` 与 `data/B.ply`。

然后运行：

```bash
bash run.sh
```

`run.sh` 会：

1. 根据需要激活环境（可在脚本顶部自行修改）；  
2. 安装依赖包 `open3d / pyyaml / numpy`（若尚未安装）；  
3. 创建 `output/` 目录；  
4. 调用 `python align.py` 完成配准。

最终对齐结果将保存在 `output/B_aligned.ply` 中。

---

## 六、两步对齐工作流（交互 + 精细优化）

在当前实现中，更推荐使用 **“两步对齐”** 流程，而不是纯自动 RANSAC：

- **Step 1：交互式粗对齐（`interactive_coarse_align.py`）**  
  - 运行：
    ```bash
    cd Align
    python interactive_coarse_align.py --config config.yaml --port 8090
    ```
  - 浏览器访问 `http://<server-ip>:8090` 后，你会看到：
    - 固定的参考点云 A（`/A_fixed`）  
    - 可移动的点云 B（`/B_moving`，仅用于预览，下采样版）  
  - 交互方式有两种，可以任选其一或混合使用：
    - **方式 A：滑块调节 B 的 RT**  
      - 在 `Coarse Alignment (B → A)` 面板里，通过 `tx/ty/tz` 与 `rx/ry/rz` 大致把 B 拉到 A 上。  
      - 点击底部的 `Save transform` 按钮，会把当前 4×4 变换矩阵写到 `output/init_transform.txt`。
    - **方式 B：手动圈选对应点（推荐）**  
      - 在 `Correspondence (A ↔ B)` 面板中：  
        1. 点击 `Pick one A (fixed)`，然后在场景中单击 A 上一个明显特征点（会生成一个半径约 1m 的彩色小球，如 A1 红色）；  
        2. 点击 `Pick one B (moving)`，在 B 上点选对应位置（生成同色小球，如 B1 红色）；  
        3. 重复 2–3 组（A2/B2 蓝色、A3/B3 绿色 ...）；  
        4. 点 `Solve from correspondences & Save`：脚本会用这些对应点求解刚体变换 \(T\)，写入 `output/init_transform.txt`，并把 `/B_moving` 移到估计的初始位姿；  
        5. 如需重新选择，可点 `Clear picked points` 清除所有已选点及其小球标记。

- **Step 2：精细 ICP 对齐（`align.py`）**  
  - 保证 `config.yaml` 中的 `init.transform_file` 与上一步保存的路径一致，例如：
    ```yaml
    init:
      transform_file: "output/init_transform.txt"
    ```
  - 在 `Align` 目录下运行：
    ```bash
    bash run.sh
    ```
  - `align.py` 会：
    1. 读取 `init_T`（第一步写出的 4×4 矩阵）；  
    2. 对 A/B 做法向估计；  
    3. 以 `init_T` 为初始位姿运行 ICP（受 `icp.*` 与 `robust.*` 控制）；  
    4. 打印：
       - `init_T`（粗对齐）  
       - `ICP T`（精细对齐后的最终位姿）  
       - `delta_T = ICP_T * init_T^{-1}`（ICP 相对粗对齐又调整了多少）；  
    5. 将最终对齐结果写到 `output/B_aligned.ply`。

这种“两步式”工作流在 **初始姿态差很大 / A 是 B 的子集 / 部分重叠** 的场景下更加稳健：  
- 第一步由人提供可靠的几何先验（粗对齐或若干对应点）；  
- 第二步由 ICP 在局部做厘米级甚至毫米级的精细收敛。

---

**说明**：该 pipeline 适用于多视点重建、全景深度融合、3DGS 初始化等场景，可作为通用的点云对齐子模块直接复用或二次开发。
