from __future__ import annotations

import argparse
import math
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from query_engine import (
    DEFAULT_DISPLAY_POINTS,
    CategoryMeta,
    QueryEngine,
    _chunk_block_sums,
    _default_time_chunk,
)

MAX_WIDTH = 5000
MAX_HEIGHT = 400
MAX_TIME_STEPS = 50000
DEFAULT_BLOCK_SIZE = 4096


def format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value:.2f} B"


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if np.dtype(np.intp).itemsize == 8 and Path("/System").exists():
        return int(value)
    return int(value) * 1024


def make_chunk_template(height: int, width: int, frames: int, dtype: np.dtype) -> np.ndarray:
    pixel_count = height * width
    spatial = (np.arange(pixel_count, dtype=np.int32) % 97).reshape(height, width)
    if np.issubdtype(dtype, np.floating):
        frame = (spatial.astype(np.float32) / 97.0).astype(dtype)
    else:
        frame = spatial.astype(dtype, copy=False)
    return np.broadcast_to(frame, (frames, height, width)).copy()


def run_prepare_stream_benchmark(
    dtype_name: str,
    width: int,
    height: int,
    time_steps: int,
    block_size: int,
) -> None:
    dtype = np.dtype(dtype_name)
    pixel_count = width * height
    block_count = math.ceil(pixel_count / block_size)
    time_chunk = _default_time_chunk(height, width, dtype)
    chunk_template = make_chunk_template(height, width, time_chunk, dtype)

    with tempfile.TemporaryDirectory(prefix="image_observer_bench_") as temp_dir:
        temp_root = Path(temp_dir)
        block_path = temp_root / "block_sums.npy"
        block_sums = np.lib.format.open_memmap(
            block_path,
            mode="w+",
            dtype=np.float64,
            shape=(time_steps, block_count),
        )

        started = time.perf_counter()
        checksum = 0.0
        iterations = 0
        for start in range(0, time_steps, time_chunk):
            stop = min(start + time_chunk, time_steps)
            frame_count = stop - start
            chunk = chunk_template[:frame_count]
            partial = _chunk_block_sums(
                chunk.reshape(frame_count, pixel_count),
                pixel_count=pixel_count,
                block_size=block_size,
                block_count=block_count,
            )
            block_sums[start:stop] = partial
            checksum += float(partial[:, 0].sum(dtype=np.float64))
            iterations += 1
        block_sums.flush()
        elapsed = time.perf_counter() - started

        logical_input_bytes = pixel_count * time_steps * dtype.itemsize
        block_bytes = time_steps * block_count * np.dtype(np.float64).itemsize
        print(f"mode=prepare_stream dtype={dtype.name}")
        print(f"shape=({time_steps}, {height}, {width}) block_size={block_size} block_count={block_count}")
        print(f"time_chunk={time_chunk} iterations={iterations}")
        print(f"logical_input={format_bytes(logical_input_bytes)} block_output={format_bytes(block_bytes)}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"peak_rss={format_bytes(max_rss_bytes())}")
        print(f"checksum={checksum:.3f}")


def build_block_sums_file(path: Path, time_steps: int, block_count: int) -> None:
    block_sums = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(time_steps, block_count),
    )
    rows_per_chunk = 4096
    base = np.arange(block_count, dtype=np.float64)[None, :]
    for start in range(0, time_steps, rows_per_chunk):
        stop = min(start + rows_per_chunk, time_steps)
        frame_ids = np.arange(start, stop, dtype=np.float64)[:, None]
        block_sums[start:stop] = frame_ids * 0.01 + base
    block_sums.flush()


def run_query_benchmark(
    width: int,
    height: int,
    time_steps: int,
    block_size: int,
    max_points: int,
) -> None:
    pixel_count = width * height
    block_count = math.ceil(pixel_count / block_size)

    with tempfile.TemporaryDirectory(prefix="image_observer_query_") as temp_dir:
        temp_root = Path(temp_dir)
        block_path = temp_root / "block_sums.npy"
        cube_path = temp_root / "cube_tyx.npy"
        preview_path = temp_root / "preview.npy"

        build_block_sums_file(block_path, time_steps=time_steps, block_count=block_count)
        np.save(preview_path, np.zeros((height, width), dtype=np.float32))

        meta = CategoryMeta(
            name="max_query",
            prepared_dir=temp_root,
            cube_path=cube_path,
            block_sums_path=block_path,
            preview_path=preview_path,
            source_path=temp_root / "synthetic.npy",
            layout="tyx",
            dtype="float32",
            time_steps=time_steps,
            height=height,
            width=width,
            pixel_count=pixel_count,
            block_size=block_size,
            block_count=block_count,
        )
        selected = np.arange(pixel_count, dtype=np.int64)
        engine = QueryEngine()

        started = time.perf_counter()
        result = engine.query_category(meta, selected, start=0, stop=time_steps, max_points=max_points)
        elapsed = time.perf_counter() - started

        print(f"mode=query_full_blocks max_points={max_points}")
        print(f"shape=({time_steps}, {height}, {width}) block_size={block_size} block_count={block_count}")
        print(f"selection_size={selected.size}")
        print(f"aggregated={result.aggregated} display_points={result.display_values.size}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"engine_elapsed_ms={result.elapsed_ms:.3f}")
        print(f"peak_rss={format_bytes(max_rss_bytes())}")
        print(f"first_display_value={float(result.display_values[0]):.6f}")
        print(f"last_display_value={float(result.display_values[-1]):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark max-sized synthetic workloads.")
    parser.add_argument("--mode", choices=("prepare_stream", "query"), required=True)
    parser.add_argument("--dtype", default="float32", help="Used by prepare_stream mode.")
    parser.add_argument("--width", type=int, default=MAX_WIDTH)
    parser.add_argument("--height", type=int, default=MAX_HEIGHT)
    parser.add_argument("--time-steps", type=int, default=MAX_TIME_STEPS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-points", type=int, default=DEFAULT_DISPLAY_POINTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "prepare_stream":
        run_prepare_stream_benchmark(
            dtype_name=args.dtype,
            width=args.width,
            height=args.height,
            time_steps=args.time_steps,
            block_size=args.block_size,
        )
    else:
        run_query_benchmark(
            width=args.width,
            height=args.height,
            time_steps=args.time_steps,
            block_size=args.block_size,
            max_points=args.max_points,
        )


if __name__ == "__main__":
    main()
