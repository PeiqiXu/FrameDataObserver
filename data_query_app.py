from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

if sys.platform == "darwin" and os.environ.get("MPLBACKEND", "").lower() in {"", "tkagg", "tkcairo"}:
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RectangleSelector, TextBox

from query_engine import DEFAULT_DISPLAY_POINTS, DEFAULT_WORKERS, QueryEngine, list_prepared_categories, mask_to_flat_indices

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DataQueryApp:
    def __init__(self, prepared_root: Path, workers: int = DEFAULT_WORKERS) -> None:
        self.prepared_root = prepared_root
        self.engine = QueryEngine()
        self.workers = workers
        self.categories = {}
        self.category_names: list[str] = []
        self.category_states: dict[str, bool] = {}
        self.selection_mask: np.ndarray | None = None
        self.preview_meta = None
        self.mode = "point"

        self.figure = plt.figure(figsize=(15, 9))
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("数据查询演示")
        self.image_ax = self.figure.add_axes([0.06, 0.32, 0.56, 0.60])
        self.curve_ax = self.figure.add_axes([0.06, 0.08, 0.56, 0.18])
        self.mode_ax = self.figure.add_axes([0.70, 0.78, 0.24, 0.12])
        self.category_ax = self.figure.add_axes([0.70, 0.54, 0.24, 0.20])
        self.start_ax = self.figure.add_axes([0.70, 0.44, 0.10, 0.05])
        self.stop_ax = self.figure.add_axes([0.84, 0.44, 0.10, 0.05])
        self.points_ax = self.figure.add_axes([0.70, 0.34, 0.24, 0.05])
        self.query_ax = self.figure.add_axes([0.70, 0.24, 0.11, 0.06])
        self.clear_ax = self.figure.add_axes([0.83, 0.24, 0.11, 0.06])
        self.status_text = self.figure.text(0.06, 0.02, "", fontsize=10)
        self.help_text = self.figure.text(
            0.70,
            0.08,
            "操作说明\n"
            "1. 勾选类别\n"
            "2. 选 point / rect add / rect remove\n"
            "3. 在上图点选或拖框\n"
            "4. 设置时间范围后点 Query\n"
            "5. 关闭窗口即可退出",
            fontsize=10,
            va="bottom",
        )

        self.image_artist = None
        self.overlay_artist = None
        self.colorbar = None

        self.mode_selector = RadioButtons(self.mode_ax, ("point", "rect_add", "rect_remove"), active=0)
        self.mode_selector.on_clicked(self.on_mode_change)
        self.mode_ax.set_title("选区工具", fontsize=11)

        self.start_box = TextBox(self.start_ax, "start", initial="0")
        self.stop_box = TextBox(self.stop_ax, "stop", initial="")
        self.points_box = TextBox(self.points_ax, "points", initial=str(DEFAULT_DISPLAY_POINTS))

        self.query_button = Button(self.query_ax, "Query")
        self.query_button.on_clicked(self.run_query)
        self.clear_button = Button(self.clear_ax, "Clear")
        self.clear_button.on_clicked(self.clear_selection)

        self.check_buttons = None
        self.rect_selector = RectangleSelector(
            self.image_ax,
            self.on_rectangle_select,
            useblit=False,
            button=[1],
            interactive=False,
        )
        self.rect_selector.set_active(False)

        self.figure.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.load_categories()

    def load_categories(self) -> None:
        self.categories = {meta.name: meta for meta in list_prepared_categories(self.prepared_root)}
        self.category_names = list(self.categories)
        if not self.category_names:
            raise FileNotFoundError(f"在 {self.prepared_root} 下没有找到 prepared 类别。")

        if not self.category_states:
            self.category_states = {name: idx == 0 for idx, name in enumerate(self.category_names)}
        else:
            self.category_states = {name: self.category_states.get(name, False) for name in self.category_names}

        self._build_category_selector()
        self._sync_preview_meta()
        if self.preview_meta is not None and not self.stop_box.text:
            self.stop_box.set_val(str(self.preview_meta.time_steps))
        self.refresh_preview()
        self.set_status(
            f"已载入 {len(self.category_names)} 个类别，默认目录 {self.prepared_root}，当前预览 {self.preview_meta.name}。"
        )

    def _build_category_selector(self) -> None:
        self.category_ax.clear()
        states = [self.category_states[name] for name in self.category_names]
        self.check_buttons = CheckButtons(self.category_ax, self.category_names, states)
        self.check_buttons.on_clicked(self.on_category_toggle)
        self.category_ax.set_title("类别选择", fontsize=11)

    def _sync_preview_meta(self) -> None:
        active_names = self.selected_category_names()
        preview_name = active_names[0] if active_names else self.category_names[0]
        self.preview_meta = self.categories[preview_name]
        wanted_shape = (self.preview_meta.height, self.preview_meta.width)
        if self.selection_mask is None or self.selection_mask.shape != wanted_shape:
            self.selection_mask = np.zeros(wanted_shape, dtype=bool)

    def selected_category_names(self) -> list[str]:
        return [name for name in self.category_names if self.category_states.get(name, False)]

    def on_mode_change(self, label: str) -> None:
        self.mode = label
        self.rect_selector.set_active(self.mode in {"rect_add", "rect_remove"})
        self.set_status(f"当前工具已切换为 {self.mode}。")

    def on_category_toggle(self, label: str) -> None:
        self.category_states[label] = not self.category_states[label]
        self._sync_preview_meta()
        self.refresh_preview()
        selected_count = len(self.selected_category_names())
        self.set_status(f"类别选择已更新，当前勾选 {selected_count} 个类别，预览 {self.preview_meta.name}。")

    def refresh_preview(self) -> None:
        if self.preview_meta is None or self.selection_mask is None:
            return

        preview = np.asarray(self.engine.preview(self.preview_meta))
        overlay = np.ma.masked_where(~self.selection_mask, self.selection_mask.astype(float))

        if self.image_artist is None:
            self.image_artist = self.image_ax.imshow(preview, cmap="viridis", origin="upper", aspect="auto")
            self.overlay_artist = self.image_ax.imshow(
                overlay,
                cmap="autumn",
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                alpha=0.35,
            )
            self.colorbar = self.figure.colorbar(self.image_artist, ax=self.image_ax, fraction=0.046, pad=0.02)
        else:
            self.image_artist.set_data(preview)
            self.image_artist.set_clim(vmin=float(np.nanmin(preview)), vmax=float(np.nanmax(preview)))
            self.overlay_artist.set_data(overlay)

        self.image_ax.set_title(f"空间预览: {self.preview_meta.name} ({self.preview_meta.width} x {self.preview_meta.height})")
        self.image_ax.set_xlabel("X")
        self.image_ax.set_ylabel("Y")
        self.figure.canvas.draw_idle()

    def on_canvas_click(self, event) -> None:
        if self.preview_meta is None or self.selection_mask is None or self.mode != "point":
            return
        if event.inaxes != self.image_ax or event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < self.preview_meta.width and 0 <= y < self.preview_meta.height):
            return

        self.selection_mask[y, x] = ~self.selection_mask[y, x]
        self.refresh_preview()
        self.set_status(f"点选更新完成，当前已选像素 {int(self.selection_mask.sum())}。")

    def on_rectangle_select(self, eclick, erelease) -> None:
        if self.preview_meta is None or self.selection_mask is None:
            return
        if self.mode not in {"rect_add", "rect_remove"}:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return

        x0 = max(0, min(int(round(eclick.xdata)), int(round(erelease.xdata))))
        x1 = min(self.preview_meta.width - 1, max(int(round(eclick.xdata)), int(round(erelease.xdata))))
        y0 = max(0, min(int(round(eclick.ydata)), int(round(erelease.ydata))))
        y1 = min(self.preview_meta.height - 1, max(int(round(eclick.ydata)), int(round(erelease.ydata))))

        if self.mode == "rect_add":
            self.selection_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        else:
            self.selection_mask[y0 : y1 + 1, x0 : x1 + 1] = False

        self.refresh_preview()
        self.set_status(f"矩形操作完成，当前已选像素 {int(self.selection_mask.sum())}。")

    def clear_selection(self, _event=None) -> None:
        if self.selection_mask is None:
            return
        self.selection_mask[:] = False
        self.refresh_preview()
        self.set_status("选区已清空。")

    def run_query(self, _event=None) -> None:
        if self.preview_meta is None or self.selection_mask is None:
            self.set_status("没有可查询的数据。")
            return

        category_names = self.selected_category_names()
        if not category_names:
            self.set_status("请先勾选至少一个类别。")
            return

        selected_indices = mask_to_flat_indices(self.selection_mask)
        if selected_indices.size == 0:
            self.set_status("请先在空间预览图上选择至少一个像素。")
            return

        try:
            start = int(self.start_box.text or "0")
            stop = int(self.stop_box.text or str(self.preview_meta.time_steps))
            max_points = int(self.points_box.text or str(DEFAULT_DISPLAY_POINTS))
        except ValueError:
            self.set_status("时间范围和显示点数必须是整数。")
            return

        self.curve_ax.clear()
        total_ms = 0.0
        plotted = 0
        aggregated_count = 0
        skipped = []
        query_metas = []
        for name in category_names:
            meta = self.categories[name]
            if (meta.height, meta.width) != self.selection_mask.shape:
                skipped.append(name)
            else:
                query_metas.append(meta)

        if query_metas:
            try:
                results = self.engine.query_categories(
                    query_metas, selected_indices, start=start, stop=stop, max_points=max_points,
                    workers=self.workers,
                )
            except Exception as exc:
                self.set_status(f"查询失败: {exc}")
                self.figure.canvas.draw_idle()
                return
            for result in results:
                total_ms += result.elapsed_ms
                plotted += 1
                aggregated_count += int(result.aggregated)
                self.curve_ax.plot(result.display_time, result.display_values, linewidth=1.5, label=result.category_name)

        self.curve_ax.set_title("区域均值曲线")
        self.curve_ax.set_xlabel("时间")
        self.curve_ax.set_ylabel("均值")
        self.curve_ax.grid(True, alpha=0.3)
        if plotted:
            self.curve_ax.legend(loc="best")
        self.figure.canvas.draw_idle()

        skipped_text = f"，跳过 {','.join(skipped)}" if skipped else ""
        aggregate_text = f"，聚合显示 {aggregated_count} 个" if aggregated_count else ""
        self.set_status(
            f"查询完成：类别 {plotted} 个，像素 {selected_indices.size} 个，累计查询耗时 {total_ms:.1f} ms{aggregate_text}{skipped_text}。"
        )

    def set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="演示数据查询界面。")
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="prepared 数据目录。默认优先使用 demo/prepared，其次使用 ./prepared。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行线程数，默认 {DEFAULT_WORKERS}。",
    )
    return parser.parse_args()


def resolve_prepared_root(raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    demo_root = (Path.cwd() / "demo" / "prepared").resolve()
    if demo_root.exists():
        return demo_root
    return (Path.cwd() / "prepared").resolve()


def main() -> None:
    args = parse_args()
    prepared_root = resolve_prepared_root(args.prepared_root)
    app = DataQueryApp(prepared_root, workers=args.workers)
    app.show()


if __name__ == "__main__":
    main()
