# 数据查询实现

这个目录实现了 `数据查询要求.docx` 里的核心需求：

- 支持 3D 数据立方体的预处理，原始数据可按 `X*Y*T`、`Y*X*T`、`T*Y*X`、`T*X*Y` 轴顺序导入。
- 支持在空间平面上选择任意像素集合，计算该区域在时间维度上的均值曲线。
- 支持按时间范围查询，并在时间跨度很大时自动抽样显示，默认显示 1200 个点，不会低于 500 个点。
- 支持 3 个类别分别预处理，并用同一选区对多个类别独立绘制曲线。

## 文件说明

- `query_engine.py`
  - 预处理原始 `.npy` 立方体。
  - 生成标准化时间主序数据 `cube_tyx.npy`。
  - 生成空间块汇总 `block_sums.npy`，用于加速大区域查询。
  - 生成时间 LOD 金字塔 `block_sums_lod.npy`，用于加速大时间跨度聚合查询。
  - 提供查询引擎 `QueryEngine`。
- `prepare_data.py`
  - 预处理命令行入口。
- `data_query_app.py`
  - Tkinter + Matplotlib 桌面交互界面。
- `generate_demo_data.py`
  - 生成 3 类演示数据并自动完成预处理。
- `test_query_engine.py`
  - 基础正确性测试。

## 使用方式

### 1. 生成演示数据

```bash
python3 generate_demo_data.py
```

执行后会生成：

- `demo/raw/` 原始演示数据
- `demo/prepared/` 可直接查询的 prepared 数据
- 其中演示类别默认分别使用 `float32`、`int16`、`int8`

### 2. 预处理你自己的数据

如果你的原始数据是 `.npy` 文件，可以先运行：

```bash
python3 prepare_data.py --source "/path/to/category_a.npy" --prepared-root prepared --name category_a --layout xyt
python3 prepare_data.py --source "/path/to/category_b.npy" --prepared-root prepared --name category_b --layout xyt
python3 prepare_data.py --source "/path/to/category_c.npy" --prepared-root prepared --name category_c --layout xyt
```

参数说明：

- `--layout`
  - `xyt`: 原始数组是 `(X, Y, T)`
  - `yxt`: 原始数组是 `(Y, X, T)`
  - `tyx`: 原始数组是 `(T, Y, X)`
  - `txy`: 原始数组是 `(T, X, Y)`
- `--block-size`
  - 空间块大小，默认 `4096` 像素。块越大，预处理文件更小；块越小，大面积整块选区查询更快。
- `--dtype`
  - 默认 `auto`，自动保留源数据 dtype，适合 `float32`、`int16`、`int8` 混合类别一起预处理。
- `--workers`
  - 并行线程数，默认 `min(16, CPU 核心数)`。
- `--no-lod`
  - 禁用时间 LOD 金字塔预计算（默认启用）。

### 3. 启动查询界面

```bash
python3 data_query_app.py
```

界面支持：

- 点选像素开关
- 矩形批量添加/移除选区
- 导入 `.npy` 掩码或 `x,y` 两列坐标 CSV
- 单类别或多类别同时绘图
- 自定义时间起止范围

## 性能设计

为满足”首次加载后，时间范围和选区调整尽量秒级响应”的要求，这个实现用了两层数据：

1. `cube_tyx.npy`
   - 标准化后的原始数据，按时间主序保存，便于按时间范围切片。
2. `block_sums.npy`
   - 将空间像素按固定块做时间序列求和。
   - 当选区覆盖了完整空间块时，可以直接使用块汇总，而不必再读取每个像素。
3. 查询时的时间桶聚合
   - 当时间跨度超过显示点数时，不再生成完整时间序列，而是按显示桶直接做流式聚合。
   - 这样同时绘制多个类别时，内存占用主要取决于显示点数，而不是完整时间长度。

查询时会把选区拆成两部分：

- 完整块：直接从 `block_sums.npy` 读取
- 非完整块（partial ranges）：将零散像素按连续区间合并，使用切片索引从 `cube_tyx.npy` 读取，而非逐像素 fancy indexing

这样做的好处：

- 大面积连续选区时，速度明显快于逐像素扫描
- 小面积稀疏选区时，仍保持精确计算
- 非完整块的像素被合并为连续区间后，mmap 按连续内存读取，比散点索引快数量级
- 时间跨度很大时，显示层自动降采样，避免一次画过多点

### 时间 LOD 金字塔

预处理阶段会为 `block_sums` 额外构建时间维度的多级缓存（LOD，Level of Detail）：

```
Level 0: 原始 block_sums          (T 帧)
Level 1: 每 2 帧求和             (T/2 帧)
Level 2: 每 4 帧求和             (T/4 帧)
  ...
Level N: 直到帧数 ≤ 500
```

所有 LOD 级别合并存储在 `block_sums_lod.npy` 中，通过 mmap 按需读取。

查询时，如果时间跨度很大且需要聚合显示（如 50000 帧显示 500 点），引擎会自动选择最合适的 LOD 级别：

- 完整块的汇总值直接从 LOD 级别读取，只需读 ~500 行而非 50000 行
- LOD 帧边界上的首尾不对齐部分，从 Level 0 精确修正
- 非完整块的像素仍从 `cube_tyx.npy` 按 LOD 帧边界分组读取
- 当 LOD 无法提供足够精度时（如时间跨度本身就不大），自动回退到逐帧聚合

LOD 存储开销约为 `block_sums.npy` 的 50%（因为 T/2 + T/4 + … ≈ T），在最大规模下每类约 93 MiB，相比主数据体积可忽略。

LOD 默认启用，可通过预处理时传 `--no-lod` 或 `lod=False` 禁用。

### 多线程并行加速

在上述数据结构优化的基础上，系统还在四个层面利用多核 CPU 做并行处理：

1. **跨类别查询并行**
   - `QueryEngine.query_categories()` 使用 `ThreadPoolExecutor` 同时查询多个类别。
   - 界面点击 Query 后，3 个类别的查询并发执行，而不是依次串行。
2. **单类别查询内时间块并行**
   - `query_category()` 将时间范围拆分成多个独立块，使用线程池并发处理。
   - 聚合模式下按显示桶并行，非聚合模式下按时间块并行。
   - 每个线程读取 mmap 的不同切片，互不冲突。
3. **跨类别预处理并行**
   - `generate_demo_data.py` 使用 `ProcessPoolExecutor` 同时生成和预处理多个类别。
   - 每个类别写入各自独立的目录，互不干扰。
4. **单类别预处理内时间块并行**
   - `prepare_category()` 将时间维度拆分成多个块，使用线程池并发写入。
   - 各线程写入 memmap 的不重叠区域，保证并发安全。

并行度默认取 `min(16, CPU 核心数)`，可通过 `workers` 参数调整。跨类别查询时，线程会自动在”类别间”和”类别内时间块”之间分配。

选择 `ThreadPoolExecutor`（而非多进程）的原因：NumPy 的数组运算和 mmap 读取会释放 GIL，因此线程可以获得真正的并行加速，同时避免了跨进程序列化 mmap 句柄的开销。

## 最大规模资源估算

按需求上限 `X=5000`、`Y=400`、`T=50000` 计算：

- 单类像素数：`5000 * 400 = 2,000,000`
- 单类总采样点：`2,000,000 * 50,000 = 100,000,000,000`
- 默认最少显示点数：`500`
- 默认空间块大小：`4096`，对应 `489` 个空间块

单类原始/预处理主数据体积：

- `float32`：约 `400,000,000,000` B，约 `372.53 GiB`
- `int16`：约 `200,000,000,000` B，约 `186.26 GiB`
- `int8`：约 `100,000,000,000` B，约 `93.13 GiB`

单类附加 prepared 文件体积：

- `block_sums.npy`：约 `195,600,000` B，约 `186.54 MiB`
- `block_sums_lod.npy`：约 `97,800,000` B，约 `93.27 MiB`（LOD 金字塔，Level 1 到 N）
- `preview.npy`：约 `8,000,000` B，约 `7.63 MiB`

三类混合 `float32 + int16 + int8` 时：

- 原始数据总量约 `651.93 GiB`
- prepared 主数据总量约 `651.93 GiB`
- 三类 `block_sums.npy` 总量约 `559.62 MiB`
- 三类 `block_sums_lod.npy` 总量约 `279.81 MiB`
- 原始数据与 prepared 同时保留时，总硬盘占用约 `1.28 TiB`

内存方面：

- 查询阶段依赖内存映射和时间桶流式聚合，不会整段载入 `T=50000` 的完整时间序列
- 预处理阶段当前按时间块流式写入 `cube_tyx.npy` 和 `block_sums.npy`，目标块大小约 `128 MiB`
- 查询阶段当前按时间块流式聚合，目标块大小约 `64 MiB`
- 因此在 `64 GiB` 内存条件下，内存是足够的，瓶颈主要会转为磁盘吞吐和总处理时间

## 输入数据约定

当前实现假设原始数据文件是 NumPy `.npy` 格式。如果你的原始数据来自其他格式，可以先转换成 `.npy` 再预处理。

推荐把单类数据组织成一个 3D 数组，例如：

```python
cube.shape == (X, Y, T)  # 或其他受支持轴顺序
```

## 已知边界

- 在最大规模数据下，如果选区是高度离散且跨越大量部分块，精确查询仍然会受磁盘吞吐限制。
- 当前界面的交互方式是点选、矩形增删和外部掩码导入，适合演示和原型验证；如果需要更复杂的圈选逻辑，可以继续扩展为多边形或画刷工具。
- 当前实现使用多线程 CPU 并行和内存映射文件，没有引入 GPU 或分布式计算。

## 测试

```bash
python3 -m unittest test_query_engine.py
```
