# Viser Rig PLY Optimized Depth Visualization - 配置文件使用说明

## 配置文件

脚本 `viser_rig_ply_optdepth.py` 现在支持通过 YAML 配置文件指定路径和参数。

### 配置文件位置

默认配置文件：`viser_rig_ply_optdepth_config.yaml`

可以通过 `--config` 参数指定其他配置文件：

```bash
python viser_rig_ply_optdepth.py --config /path/to/your/config.yaml
```

### 配置文件结构

配置文件包含以下部分：

1. **paths**: 路径配置
   - `scene`: 场景名称
   - `colmap_dir`: COLMAP 数据集根目录
   - `stage_dir`: STAGE 数据集根目录
   - `optimized_dir`: 优化后的点云目录（可选，如果为 null，将使用默认路径）

2. **visualization**: 可视化设置
   - `load_original`: 是否加载原始点云
   - `hide_points`: 是否隐藏 COLMAP 稀疏点云
   - `hide_cameras`: 是否隐藏相机位置
   - `hide_ply`: 是否隐藏点云文件
   - `camera_name`: 虚拟相机名称
   - `no_transform`: 是否不对点云应用坐标变换

3. **server**: 服务器设置
   - `port`: Viser 服务器端口

4. **display**: 显示设置
   - `axis_length`: 坐标轴长度
   - `axis_width`: 坐标轴线条宽度
   - `camera_scale`: 相机 frustum 缩放比例
   - `point_size`: 点云点的大小

### 使用方式

#### 方式 1: 使用 Bash 脚本（最简单）

使用提供的 bash 脚本，它会自动检查依赖并运行：

```bash
# 使用默认配置文件
./run_viser_rig_ply_optdepth.sh

# 传递命令行参数覆盖配置
./run_viser_rig_ply_optdepth.sh --scene RoofTop --port 8082

# 指定不同的配置文件
./run_viser_rig_ply_optdepth.sh --config /path/to/other_config.yaml
```

#### 方式 2: 直接使用 Python 脚本（使用配置文件）

1. 编辑 `viser_rig_ply_optdepth_config.yaml`，设置你的路径和参数
2. 运行脚本：

```bash
python viser_rig_ply_optdepth.py
```

#### 方式 3: 命令行参数覆盖配置文件

命令行参数会覆盖配置文件中的对应设置：

```bash
# 使用配置文件，但覆盖场景名称
python viser_rig_ply_optdepth.py --scene RoofTop

# 使用配置文件，但覆盖端口
python viser_rig_ply_optdepth.py --port 8082

# 使用配置文件，但不加载原始点云
python viser_rig_ply_optdepth.py --no_load_original
```

#### 方式 4: 指定不同的配置文件

```bash
python viser_rig_ply_optdepth.py --config /path/to/other_config.yaml
```

### 参数优先级

1. **命令行参数**（最高优先级）
2. **配置文件**
3. **硬编码默认值**（最低优先级）

### 示例配置文件

```yaml
paths:
  scene: "BridgeB"
  colmap_dir: "/root/autodl-tmp/data/colmap_STAGE1_4x"
  stage_dir: "/root/autodl-tmp/data/STAGE1_4x"
  optimized_dir: "/root/autodl-tmp/data/STAGE1_4x/BridgeB/elastic_refined"

visualization:
  load_original: true
  hide_points: false
  hide_cameras: false
  hide_ply: false
  camera_name: "pano_camera12"
  no_transform: false

server:
  port: 8081

display:
  axis_length: 0.3
  axis_width: 3.0
  camera_scale: 0.05
  point_size: 0.005
```

## Bash 脚本使用说明

### 脚本功能

`run_viser_rig_ply_optdepth.sh` 提供了便捷的运行方式，它会：

1. **自动检查依赖**：检查并安装必要的 Python 包（numpy, viser, pycolmap, pyyaml, plyfile）
2. **检查配置文件**：验证配置文件是否存在
3. **传递参数**：将所有命令行参数传递给 Python 脚本

### 使用示例

```bash
# 基本使用（使用默认配置文件）
./run_viser_rig_ply_optdepth.sh

# 覆盖场景名称
./run_viser_rig_ply_optdepth.sh --scene RoofTop

# 覆盖多个参数
./run_viser_rig_ply_optdepth.sh --scene BridgeA --port 8082 --no_load_original

# 使用不同的配置文件
./run_viser_rig_ply_optdepth.sh --config /path/to/custom_config.yaml

# 组合使用
./run_viser_rig_ply_optdepth.sh --config custom_config.yaml --scene RoofTop --port 8083
```

### 注意事项

- 确保脚本有执行权限：`chmod +x run_viser_rig_ply_optdepth.sh`
- 如果需要在特定 conda/virtualenv 环境中运行，请在脚本中取消注释环境激活部分
- 脚本会自动切换到脚本所在目录，确保相对路径正确
