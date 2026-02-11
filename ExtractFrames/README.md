# ExtractFrames - 视频帧提取工具

从视频文件中均匀提取帧的便捷工具集，支持单个视频文件和视频文件夹批量处理。

## 📁 目录结构

```
ExtractFrames/
├── extract_frames.sh          # Bash 便捷脚本（推荐使用）
├── config.yaml                # 配置文件（可选）
├── scripts/
│   └── extract_frames.py      # Python 核心脚本
└── README.md                   # 本文档
```

## 🚀 快速开始

### 前置要求

- Python 3.6+
- OpenCV (`cv2`): `pip install opencv-python`
- PyYAML: `pip install pyyaml`（用于读取配置文件）
- Bash shell（用于使用 `.sh` 脚本）

### 安装依赖

```bash
pip install opencv-python pyyaml
```

## 📖 使用方法

### 方式一：使用 Bash 脚本（推荐）

```bash
./extract_frames.sh <input> <output> <num_frames> [选项]
```

#### 基本用法示例

```bash
# 从单个视频提取30帧
./extract_frames.sh video.mp4 output/ 30

# 从文件夹中的所有视频总共提取100帧（按时长比例分配）
./extract_frames.sh videos/ frames/ 100 --distribute

# 从文件夹中的所有视频各提取50帧
./extract_frames.sh videos/ frames/ 50

# 指定图像格式
./extract_frames.sh videos/ frames/ 30 --format png

# 指定帧范围
./extract_frames.sh videos/ frames/ 30 --start-frame 100 --end-frame 1000

# 使用序号前缀模式
./extract_frames.sh videos/ frames/ 30 --prefix-mode sequential

# 使用配置文件并指定最大分辨率
./extract_frames.sh videos/ frames/ 30 --config config.yaml

# 命令行覆盖配置文件的最大分辨率设置
./extract_frames.sh videos/ frames/ 30 --max-width 1920 --max-height 1080
```

### 方式二：直接使用 Python 脚本

```bash
python scripts/extract_frames.py -i <input> -o <output> -n <num_frames> [选项]
```

#### Python 脚本示例

```bash
# 从单个视频提取30帧
python scripts/extract_frames.py -i video.mp4 -o output/ -n 30

# 从文件夹中的所有视频各提取50帧
python scripts/extract_frames.py -i videos/ -o frames/ -n 50

# 指定图像格式和帧范围
python scripts/extract_frames.py -i videos/ -o frames/ -n 30 --format png --start-frame 100 --end-frame 1000

# 使用序号前缀
python scripts/extract_frames.py -i videos/ -o frames/ -n 30 --prefix-mode sequential

# 按视频时长比例分配总帧数
python scripts/extract_frames.py -i videos/ -o frames/ -n 100 --distribute-by-duration

# 使用配置文件
python scripts/extract_frames.py -i videos/ -o frames/ -n 30 -c config.yaml

# 命令行覆盖配置文件的最大分辨率设置
python scripts/extract_frames.py -i videos/ -o frames/ -n 30 --max-width 1920 --max-height 1080
```

## 📋 参数说明

### Bash 脚本参数

#### 必需参数

- `input`: 输入视频文件或文件夹路径
- `output`: 输出图像文件夹路径
- `num_frames`: 要提取的帧数
  - 单个视频：提取的帧数
  - 文件夹：总帧数（如果使用 `--distribute`）或每视频帧数（默认）

#### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format FORMAT` | 图像格式 (jpg\|png) | jpg |
| `--start-frame N` | 起始帧 | 0 |
| `--end-frame N` | 结束帧 | 到视频末尾 |
| `--prefix-mode MODE` | 前缀模式 (video_name\|sequential\|none) | video_name |
| `--distribute` | 按视频时长比例分配总帧数（文件夹模式） | 否 |
| `--equal-frames` | 每个视频提取相同数量的帧（默认，文件夹模式） | 是 |
| `-c, --config PATH` | 配置文件路径（默认: ExtractFrames/config.yaml） | - |
| `--max-width N` | 输出图像最大宽度（像素），覆盖配置文件设置 | - |
| `--max-height N` | 输出图像最大高度（像素），覆盖配置文件设置 | - |
| `--max-edge N` | 输出图像最大边长（像素），覆盖配置文件设置 | - |
| `--resize-mode MODE` | 缩放模式 (fit\|crop\|stretch)，覆盖配置文件设置 | - |
| `--jpeg-quality N` | JPEG 质量 (1-100)，覆盖配置文件设置 | - |
| `-h, --help` | 显示帮助信息 | - |

### Python 脚本参数

#### 必需参数

- `-i, --input`: 输入视频文件或文件夹路径
- `-o, --output`: 输出图像文件夹路径
- `-n, --num-frames`: 要提取的帧数

#### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format` | 输出图像格式 (jpg\|jpeg\|png) | jpg |
| `--start-frame` | 起始帧（每个视频） | 0 |
| `--end-frame` | 结束帧（每个视频，None表示到末尾） | None |
| `--prefix-mode` | 文件名前缀模式 (video_name\|sequential\|none) | video_name |
| `--video-extensions` | 支持的视频扩展名 | mp4 avi mov mkv flv wmv m4v |
| `--distribute-by-duration` | 按视频时长比例分配总帧数（仅文件夹模式） | 否 |
| `--equal-frames` | 每个视频提取相同数量的帧 | 是 |
| `-c, --config` | 配置文件路径 | None |
| `--max-width` | 输出图像最大宽度（像素），覆盖配置文件设置 | None |
| `--max-height` | 输出图像最大高度（像素），覆盖配置文件设置 | None |
| `--max-edge` | 输出图像最大边长（像素），覆盖配置文件设置 | None |
| `--resize-mode` | 缩放模式 (fit\|crop\|stretch)，覆盖配置文件设置 | None |
| `--jpeg-quality` | JPEG 质量 (1-100)，覆盖配置文件设置 | None |

## 🎯 功能特性

### 1. 均匀帧提取

从视频中均匀提取指定数量的帧，确保帧在整个视频时长范围内均匀分布。

### 2. 批量处理

支持处理文件夹中的多个视频文件，可以：
- **按时长比例分配**：根据每个视频的时长按比例分配总帧数
- **等量提取**：每个视频提取相同数量的帧

### 3. 灵活的前缀模式

- `video_name`: 使用视频文件名作为前缀（默认）
- `sequential`: 使用序号作为前缀 (video_001, video_002, ...)
- `none`: 不使用前缀

### 4. 帧范围控制

可以指定起始帧和结束帧，只提取视频的特定片段。

### 5. 多格式支持

支持多种视频格式：mp4, avi, mov, mkv, flv, wmv, m4v

支持输出图像格式：jpg, png

## ⚙️ 配置文件

工具支持通过 YAML 配置文件统一管理设置。配置文件位于 `ExtractFrames/config.yaml`。

### 配置文件示例

```yaml
# ExtractFrames 配置文件
# 视频帧提取工具配置

# =========================
# 输出图像配置
# =========================
output:
  # 最大分辨率设置
  # 如果提取的帧分辨率超过此值，将自动缩放
  # 设置为 null 或 0 表示不限制分辨率
  max_resolution:
    width: 1920   # 最大宽度（像素），null 表示不限制
    height: 1080  # 最大高度（像素），null 表示不限制
  
  # 或者使用单一的最大边长（宽度和高度中的较大值）
  # 如果设置了 max_edge，将优先使用 max_edge，忽略 width 和 height
  # max_edge: 1920  # 最大边长（像素），null 表示不限制
  
  # 缩放模式
  # "fit": 保持宽高比，缩放到适合最大分辨率（默认）
  # "crop": 保持宽高比，裁剪到最大分辨率
  # "stretch": 不保持宽高比，拉伸到最大分辨率
  resize_mode: "fit"
  
  # 图像格式
  format: "jpg"  # jpg, png
  
  # JPEG 质量（仅当 format 为 jpg 时有效）
  jpeg_quality: 95  # 1-100，值越大质量越好但文件越大

# =========================
# 提取配置
# =========================
extraction:
  # 前缀模式
  # "video_name": 使用视频文件名作为前缀（默认）
  # "sequential": 使用序号作为前缀
  # "none": 不使用前缀
  prefix_mode: "video_name"
  
  # 支持的视频扩展名
  video_extensions:
    - "mp4"
    - "avi"
    - "mov"
    - "mkv"
    - "flv"
    - "wmv"
    - "m4v"
  
  # 帧范围设置（每个视频）
  start_frame: 0        # 起始帧，0 表示从视频开头开始
  end_frame: null       # 结束帧，null 表示到视频末尾
  
  # 批量处理模式（仅对文件夹模式有效）
  # "distribute": 按视频时长比例分配总帧数
  # "equal": 每个视频提取相同数量的帧（默认）
  distribute_mode: "equal"  # distribute 或 equal

# =========================
# 日志配置
# =========================
logging:
  # 日志级别: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # 是否显示缩放信息
  show_resize_info: true
```

### 使用配置文件

```bash
# Bash 脚本
./extract_frames.sh videos/ frames/ 30 --config config.yaml

# Python 脚本
python scripts/extract_frames.py -i videos/ -o frames/ -n 30 -c config.yaml
```

### 配置文件优先级

1. **命令行参数**（最高优先级）
2. **配置文件**
3. **默认值**（最低优先级）

命令行参数会覆盖配置文件中的对应设置。

### 配置文件参数说明

#### output 部分

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_resolution.width` | 最大宽度（像素），null 表示不限制 | null |
| `max_resolution.height` | 最大高度（像素），null 表示不限制 | null |
| `max_edge` | 最大边长（像素），null 表示不限制 | null |
| `resize_mode` | 缩放模式 (fit\|crop\|stretch) | fit |
| `format` | 图像格式 (jpg\|png) | jpg |
| `jpeg_quality` | JPEG 质量 (1-100) | 95 |

#### extraction 部分

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `prefix_mode` | 前缀模式 (video_name\|sequential\|none) | video_name |
| `video_extensions` | 支持的视频扩展名列表 | ["mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v"] |
| `start_frame` | 起始帧（每个视频） | 0 |
| `end_frame` | 结束帧（每个视频），null 表示到末尾 | null |
| `distribute_mode` | 批量处理模式 (distribute\|equal) | equal |

#### logging 部分

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `level` | 日志级别 (DEBUG\|INFO\|WARNING\|ERROR) | INFO |
| `show_resize_info` | 是否显示缩放信息 | true |

## 📝 输出文件命名规则

### 使用 video_name 前缀模式（默认）

```
VID_20260208_080822_frame_000000.jpg
VID_20260208_080822_frame_000123.jpg
VID_20260208_080822_frame_000456.jpg
```

### 使用 sequential 前缀模式

```
video_001_frame_000000.jpg
video_001_frame_000123.jpg
video_002_frame_000000.jpg
```

### 使用 none 前缀模式

```
frame_000000.jpg
frame_000123.jpg
frame_000456.jpg
```

## 💡 使用场景

### 场景 1: COLMAP 重建准备

从视频中提取关键帧用于 COLMAP 稀疏重建：

```bash
./extract_frames.sh \
  /path/to/videos/ \
  /path/to/images/ \
  50 \
  --distribute \
  --prefix-mode none
```

### 场景 2: 单个视频处理

从单个视频中提取固定数量的帧：

```bash
./extract_frames.sh video.mp4 output/ 30
```

### 场景 3: 指定时间范围

只提取视频中间部分的帧：

```bash
./extract_frames.sh video.mp4 output/ 30 \
  --start-frame 1000 \
  --end-frame 5000
```

### 场景 4: 限制输出图像分辨率

使用配置文件限制输出图像最大分辨率为 1920x1080：

```bash
# 编辑 config.yaml 设置 max_resolution
./extract_frames.sh videos/ frames/ 30 --config config.yaml
```

或者通过命令行参数直接指定：

```bash
./extract_frames.sh videos/ frames/ 30 \
  --max-width 1920 \
  --max-height 1080 \
  --resize-mode fit
```

### 场景 5: 使用最大边长限制

限制图像的最大边长（适用于正方形或接近正方形的输出需求）：

```bash
./extract_frames.sh videos/ frames/ 30 --max-edge 1920
```

## 🔧 技术细节

### 帧提取算法

工具使用均匀采样算法：
1. 计算可用帧范围（考虑 start_frame 和 end_frame）
2. 如果请求的帧数 >= 可用帧数，提取所有帧
3. 否则，计算步长 `step = available_frames / num_frames`
4. 均匀分布提取帧索引

### 按时长比例分配算法

当使用 `--distribute` 选项时：
1. 计算每个视频的可用帧数（考虑帧范围限制）
2. 计算总可用帧数
3. 按比例分配：`frames_per_video = total_frames * video_duration / total_duration`
4. 最后一个视频分配剩余的所有帧，确保总数匹配

### 图像缩放算法

当设置了最大分辨率限制时：

1. **fit 模式**（默认）：
   - 计算宽度和高度的缩放比例
   - 选择较小的缩放比例，确保图像完全适合目标尺寸
   - 保持原始宽高比

2. **crop 模式**：
   - 计算宽度和高度的缩放比例
   - 选择较大的缩放比例，确保图像覆盖目标尺寸
   - 居中裁剪到目标尺寸
   - 保持原始宽高比

3. **stretch 模式**：
   - 直接拉伸到目标宽度和高度
   - 不保持宽高比

4. **最大边长模式**：
   - 如果设置了 `max_edge`，优先使用此模式
   - 计算宽度和高度中的较大值
   - 如果超过 `max_edge`，按比例缩放

## ⚠️ 注意事项

1. **内存使用**：处理大视频文件时，确保有足够的系统内存
2. **磁盘空间**：确保输出目录有足够的磁盘空间
3. **视频格式**：确保视频文件格式受支持，否则会被跳过
4. **帧范围**：如果指定的帧范围无效（start >= end 或超出视频范围），会报错
5. **Python 版本**：需要 Python 3.6 或更高版本

## 🐛 故障排除

### 问题：无法打开视频文件

**原因**：视频文件损坏或格式不支持

**解决**：检查视频文件是否完整，尝试使用其他视频格式

### 问题：提取的帧数少于预期

**原因**：视频可用帧数少于请求的帧数

**解决**：工具会自动调整，提取所有可用帧并显示警告信息

### 问题：Python 脚本找不到

**原因**：路径引用错误

**解决**：确保 `extract_frames.sh` 和 `scripts/extract_frames.py` 的相对路径正确

## 📚 相关文件

- `extract_frames.sh`: Bash 便捷脚本，提供简化的命令行接口
- `config.yaml`: 配置文件，用于统一管理工具设置
- `scripts/extract_frames.py`: Python 核心脚本，包含所有提取逻辑和缩放功能

## 📄 许可证

本工具是 ColmapUtil 项目的一部分。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题或建议，请通过项目 Issue 系统反馈。
