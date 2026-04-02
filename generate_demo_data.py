from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from query_engine import DEFAULT_WORKERS, prepare_category


def build_demo_cube(seed: int, time_steps: int = 720, height: int = 96, width: int = 160) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 8.0 * np.pi, time_steps, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]

    base = 0.18 * x_axis + 0.24 * y_axis
    wave_a = np.sin(time_axis * (0.8 + 0.1 * seed), dtype=np.float32)[:, None, None]
    wave_b = np.cos(time_axis * (0.22 + 0.05 * seed), dtype=np.float32)[:, None, None]

    center_x = 0.55 * np.sin(time_axis * (0.14 + seed * 0.01), dtype=np.float32)
    center_y = 0.40 * np.cos(time_axis * (0.11 + seed * 0.02), dtype=np.float32)
    spread = 0.18 + seed * 0.02

    distance = (x_axis[None, :, :] - center_x[:, None, None]) ** 2 + (y_axis[None, :, :] - center_y[:, None, None]) ** 2
    hotspot = np.exp(-distance / spread, dtype=np.float32)

    stripes = np.sin((x_axis * (5 + seed) + y_axis * (3 + seed * 0.5)) * np.pi, dtype=np.float32)[None, :, :]
    noise = rng.normal(loc=0.0, scale=0.04 + seed * 0.005, size=(time_steps, height, width)).astype(np.float32)

    cube = 0.7 * wave_a * hotspot + 0.35 * wave_b * stripes + base[None, :, :] + noise
    return cube.astype(np.float32)


def cast_demo_cube(cube: np.ndarray, dtype: np.dtype, scale: float = 1.0) -> np.ndarray:
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.floating):
        return cube.astype(target_dtype)

    scaled = np.rint(cube.astype(np.float32) * scale)
    info = np.iinfo(target_dtype)
    return np.clip(scaled, info.min, info.max).astype(target_dtype)


def _generate_and_prepare(args: tuple) -> str:
    """Generate raw data and preprocess one category (runs in a worker process)."""
    name, seed, dtype, scale, raw_root, prepared_root, workers, lod = args
    raw_path = raw_root / f"{name}.npy"
    cube = cast_demo_cube(build_demo_cube(seed), dtype=dtype, scale=scale)
    np.save(raw_path, cube)
    prepare_category(raw_path, prepared_root=prepared_root, name=name, layout="tyx", block_size=1024, workers=workers, lod=lod)
    return f"生成演示数据: {name} ({np.dtype(dtype).name})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成演示数据并预处理。")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数，默认 {DEFAULT_WORKERS}。")
    parser.add_argument("--no-lod", action="store_true", help="禁用时间 LOD 预计算（默认启用）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd() / "demo"
    raw_root = root / "raw"
    prepared_root = root / "prepared"
    raw_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)

    categories = [
        ("class_a", 1, np.float32, 1.0),
        ("class_b", 2, np.int16, 4096.0),
        ("class_c", 3, np.int8, 96.0),
    ]

    use_lod = not args.no_lod
    tasks = [
        (name, seed, dtype, scale, raw_root, prepared_root, args.workers, use_lod)
        for name, seed, dtype, scale in categories
    ]

    with ProcessPoolExecutor(max_workers=len(categories)) as pool:
        for msg in pool.map(_generate_and_prepare, tasks):
            print(msg)

    print(f"演示数据已生成到 {root}")


if __name__ == "__main__":
    main()
