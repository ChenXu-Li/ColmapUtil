# Combine Point Clouds

合并多个优化后的点云文件为一个PLY文件。

## 功能说明

本工具参考 `viser_rig_ply_optdepth.py` 的点云对齐方式，将 `elastic_refined` 目录中的多个PLY点云文件合并为一个PLY文件。

主要功能：
- 自动加载COLMAP重建结果
- 建立全景图名称到frame的映射关系
- 对每个点云应用坐标变换（从camera坐标系转换到世界坐标系）
- 合并所有点云并保存为单个PLY文件
- 支持体素下采样（默认0.05m）以减少点云大小
- 可选生成 COLMAP points3D 格式文件（points3D.bin 和 points3D.txt），用于替换稀疏重建的点云

## 文件说明

- `combine_plys.py`: 主程序脚本
- `config.yaml`: 配置文件
- `run.sh`: 运行脚本
- `README.md`: 本说明文件

## 使用方法

### 方法1: 使用配置文件（推荐）

1. 编辑 `config.yaml` 文件，设置相关路径和参数：

```yaml
paths:
  scene: "BridgeB"  # 场景名称
  colmap_dir: "/root/autodl-tmp/data/colmap_STAGE1_4x"
  input_dir: "/root/autodl-tmp/data/STAGE1_4x/BridgeB/elastic_refined"

processing:
  output: "output/merged.ply"
  camera_name: "pano_camera12"
  no_transform: false  # 如果点云已在世界坐标系中，设置为 true
  voxel_size: 0.05  # 体素下采样大小（米），0表示不下采样，默认0.05m
```

2. 运行脚本：

```bash
./run.sh
```

或直接运行Python脚本：

```bash
python3 combine_plys.py
```

### 方法2: 使用命令行参数

```bash
python3 combine_plys.py \
    --scene BridgeB \
    --colmap_dir /root/autodl-tmp/data/colmap_STAGE1_4x \
    --input_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/elastic_refined \
    --output output/merged.ply \
    --camera_name pano_camera12
```

或使用bash脚本：

```bash
./run.sh \
    --scene BridgeB \
    --input_dir /path/to/plys \
    --output merged.ply
```

### 方法3: 如果点云已在世界坐标系中

如果点云已经在世界坐标系中，不需要应用坐标变换：

```bash
python3 combine_plys.py --no_transform
```

### 方法4: 自定义下采样大小

可以指定下采样体素大小（米），0表示不下采样：

```bash
python3 combine_plys.py --voxel_size 0.1  # 使用0.1m体素大小
python3 combine_plys.py --voxel_size 0    # 不下采样
```

### 方法5: 生成 COLMAP points3D 格式

可以同时生成 COLMAP points3D 格式文件到 output 目录：

```bash
python3 combine_plys.py --generate_colmap_points3d
```

生成的文件会保存在 output 目录中：
- `output/points3D.bin`: COLMAP 二进制格式

## 参数说明

### 配置文件参数

- `paths.scene`: 场景名称（如 BridgeB, RoofTop, BridgeA 等）
- `paths.colmap_dir`: COLMAP数据集根目录
- `paths.input_dir`: 输入点云目录（包含要合并的PLY文件）
- `processing.output`: 输出合并后的PLY文件路径
- `processing.camera_name`: 点云所在的虚拟相机名称（用于计算cam_from_rig变换）
- `processing.no_transform`: 是否不对点云应用坐标变换（默认false，即应用变换）
- `processing.voxel_size`: 体素下采样大小（米），0表示不下采样（默认0.05m）
- `processing.generate_colmap_points3d`: 是否生成 COLMAP points3D 格式文件到 output 目录（默认false）

### 命令行参数

所有配置文件中的参数都可以通过命令行参数覆盖：

- `--config`: 配置文件路径（默认: config.yaml）
- `--scene`: 场景名称
- `--colmap_dir`: COLMAP数据集根目录
- `--input_dir`: 输入点云目录
- `--output`: 输出PLY文件路径
- `--camera_name`: 相机名称
- `--no_transform`: 不对点云应用坐标变换
- `--voxel_size`: 体素下采样大小（米），0表示不下采样
- `--generate_colmap_points3d`: 生成 COLMAP points3D 格式文件到 output 目录

## 输出

合并后的点云文件将保存到指定的输出路径（默认: `output/merged.ply`）。

如果启用了 `generate_colmap_points3d`，还会在 output 目录中生成：
- `points3D.bin`: COLMAP 二进制格式的点云文件

程序会输出：
- 处理的文件数量
- 合并后的总点数（下采样前）
- 下采样后的点数（如果启用下采样）
- 点云的空间范围（X, Y, Z）
- COLMAP points3D 文件生成状态（如果启用）

## 注意事项

1. 确保COLMAP重建结果存在且包含必要的pose信息
2. 输入目录中的PLY文件名需要能够匹配到COLMAP中的全景图名称
3. 如果点云文件名包含后缀（如 `_median`, `_refined` 等），程序会自动尝试匹配
4. 如果某个点云文件无法找到对应的frame或相机，会被跳过并输出警告信息
5. 生成的 points3D 文件保存在 output 目录中，每次运行会覆盖同名文件
6. 生成的 points3D 文件没有 track 信息（点与图像的对应关系），这不会影响 Gaussian Splatting 的初始化

## 示例

合并 BridgeB 场景的 elastic_refined 点云：

```bash
cd /root/autodl-tmp/code/ColmapUtil/CombinePLYS
./run.sh --scene BridgeB --input_dir /root/autodl-tmp/data/STAGE1_4x/BridgeB/elastic_refined
```

输出文件将保存在 `output/merged.ply`。
