# 为什么需要安装支持 CUDA 的 COLMAP？

## 关键概念

### pycolmap ≠ COLMAP

- **pycolmap** 是 COLMAP 的 **Python 绑定**（wrapper），它只是一个接口层
- **COLMAP** 是底层的 **C++ 库**，包含实际的算法实现
- pycolmap 调用底层的 COLMAP C++ 库来执行计算

### CUDA 支持是在编译时决定的

```
┌─────────────────┐
│   Python 代码    │
│  (dense.py)     │
└────────┬────────┘
         │ 调用
         ▼
┌─────────────────┐
│   pycolmap      │  ← Python 绑定（只是接口）
│  (Python API)   │
└────────┬────────┘
         │ 调用
         ▼
┌─────────────────┐
│   COLMAP C++    │  ← 实际执行计算的库
│   Library       │     CUDA 支持在这里决定！
└────────┬────────┘
         │
         ▼
    ┌────────┐
    │  GPU   │  ← 需要 CUDA 支持才能使用
    └────────┘
```

## 为什么当前安装不支持 CUDA？

### 1. 预编译包通常不包含 CUDA 支持

通过 `pip install pycolmap` 或 `conda install pycolmap` 安装的包：
- 为了兼容性，通常编译时**不启用 CUDA**
- 或者只支持特定的 CUDA 版本
- 这样可以避免 CUDA 版本不匹配的问题

### 2. 错误信息说明了问题

```
[mvs.cc:40] PatchMatch requires CUDA but COLMAP was not compiled with it.
```

这个错误来自 COLMAP 的 C++ 代码（`mvs.cc`），说明：
- COLMAP 库在**编译时**没有启用 CUDA 支持
- 即使有 GPU，也无法使用 CUDA 功能

### 3. 检查当前安装

```bash
# 检查 pycolmap 版本
python -c "import pycolmap; print(pycolmap.__version__)"

# 尝试使用 CUDA 功能会报错
python -c "import pycolmap; pycolmap.patch_match_stereo(...)"
# 错误: PatchMatch requires CUDA but COLMAP was not compiled with it
```

## 解决方案

### 方案1：从源码编译（推荐）

```bash
# 1. 克隆 COLMAP 源码
git clone https://github.com/colmap/colmap.git
cd colmap

# 2. 编译时启用 CUDA
mkdir build && cd build
cmake .. \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native  # 自动检测 GPU 架构

# 3. 编译
make -j$(nproc)

# 4. 安装 Python 绑定
cd ../scripts/python
pip install -e .
```

这样编译的 pycolmap 会：
- ✅ 包含 CUDA 支持
- ✅ 匹配你的 CUDA 版本
- ✅ 针对你的 GPU 架构优化

### 方案2：使用预编译的 CUDA 版本（如果可用）

某些渠道可能提供预编译的 CUDA 版本，但需要：
- CUDA 版本匹配
- GPU 架构兼容

### 方案3：使用稀疏点云（当前可用）

即使没有 CUDA 支持，也可以：
- 使用稀疏重建的点云
- 使用 `visualizer.py` 可视化
- 导出为 PLY 格式在其他工具中使用

## 总结

| 组件 | 作用 | CUDA 支持位置 |
|------|------|--------------|
| pycolmap | Python 接口 | ❌ 不在这里 |
| COLMAP C++ 库 | 实际计算 | ✅ **在这里决定** |
| GPU | 硬件 | ✅ 需要 CUDA 驱动 |

**关键点**：即使安装了 pycolmap，如果底层 COLMAP 库没有 CUDA 支持，仍然无法使用 GPU 加速的稠密重建功能。

