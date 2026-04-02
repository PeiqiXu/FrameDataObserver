from __future__ import annotations

import argparse

from query_engine import DEFAULT_WORKERS, LAYOUT_AXES, prepare_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将原始 3D .npy 数据准备成可快速查询的格式。")
    parser.add_argument("--source", required=True, help="原始 .npy 文件路径。")
    parser.add_argument("--prepared-root", default="prepared", help="prepared 输出目录。")
    parser.add_argument("--name", help="类别名称，默认使用文件名。")
    parser.add_argument("--layout", default="tyx", choices=sorted(LAYOUT_AXES), help="原始数组的轴顺序。")
    parser.add_argument("--dtype", default="auto", help="预处理后保存的数据类型，默认保留源数据 dtype。")
    parser.add_argument("--block-size", type=int, default=4096, help="空间块大小，单位是像素。")
    parser.add_argument("--preview-frames", type=int, default=64, help="生成空间预览时抽样的时间帧数。")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数，默认 {DEFAULT_WORKERS}。")
    parser.add_argument("--no-lod", action="store_true", help="禁用时间 LOD 预计算（默认启用）。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    meta = prepare_category(
        source_path=args.source,
        prepared_root=args.prepared_root,
        name=args.name,
        layout=args.layout,
        dtype=args.dtype,
        block_size=args.block_size,
        preview_frames=args.preview_frames,
        workers=args.workers,
        lod=not args.no_lod,
    )
    print(
        f"完成: {meta.name} -> {meta.prepared_dir} "
        f"(T={meta.time_steps}, Y={meta.height}, X={meta.width}, 块数={meta.block_count})"
    )


if __name__ == "__main__":
    main()
