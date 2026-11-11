"""从项目PDF提取展示图及矢量参考曲线；参考值只供比较。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz
import numpy as np


def prepare_assets(root: Path, output: Path):
    pdf = next(root.glob("*.pdf"))
    output.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        # 原图嵌入资源的真实边界；留3pt白边，不带caption或正文。
        for page_number, name in ((3, "fig01_rl_architecture.png"), (4, "fig02_frankfurt_bs_map.png")):
            page = doc[page_number]
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
            if len(blocks) != 1:
                raise RuntimeError("Unexpected PDF scene image layout")
            rect = fitz.Rect(blocks[0]["bbox"]) + (-3, -3, 3, 3)
            page.get_pixmap(clip=rect, dpi=300).save(output / name)
        reference = extract_reference(doc[5])
    (output / "paper_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def extract_reference(page):
    """从PDF矢量线条而非渲染像素读取；不外推图窗外的分位数。"""
    paths = [p for p in page.get_drawings() if len(p["items"]) > 1000 and p["rect"].x0 < 300]
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 CDF vector paths, found {len(paths)}")
    result = {"source": "published PDF page 6 vector polylines, legend/drawing order verified",
        "kind": "digitized plotted reference, not original numerical samples",
        "approximate_reading_resolution_db": 0.2, "visible_inr_window_db": [-20, 20],
        "fig4": [], "fig5": []}
    methods = ("No-coordination", "ASCENT-derived", "PPO")
    labels = (("density_6e-08", "density_1e-07", "density_3e-07"),
              ("T1_100_T2_0", "T1_50_T2_50", "T1_0_T2_100"))
    for figure in range(2):
        x0, x1 = 68.07772064208984, 289.06060791015625
        y0, y1 = ((51.20537567138672, 174.06027221679688) if figure == 0 else
                  (228.63534545898438, 351.490234375))
        for index, path in enumerate(paths[figure * 9:(figure + 1) * 9]):
            points = []
            for item in path["items"]:
                if item[0] == "l":
                    points.extend(((item[1].x, item[1].y), (item[2].x, item[2].y)))
            points = np.asarray(points)
            order = np.argsort(points[:, 0], kind="stable")
            x = -20 + 40 * (points[order, 0] - x0) / (x1 - x0)
            cdf = (y1 - points[order, 1]) / (y1 - y0)
            # 只导出图窗内坐标；每条原图曲线均完整保留参考的可见区域。
            grid = np.linspace(-20, 20, 161)
            visible_cdf = np.interp(grid, x, cdf)
            quantiles = {}
            for q in (0.25, 0.5, 0.75):
                quantiles[str(q)] = float(np.interp(q, cdf, x)) if cdf.min() <= q <= cdf.max() else None
            result[f"fig{figure+4}"].append({"scenario": labels[figure][index % 3],
                "method": methods[index // 3], "quantiles_db": quantiles,
                "inr_grid_db": grid.tolist(), "cdf": visible_cdf.tolist()})
    return result


def compare_reference(summary, arrays, reference, output):
    rows = []
    for figure, group in (("fig4", "density"), ("fig5", "mix")):
        for ref in reference[figure]:
            actual = arrays[group][ref["scenario"]][ref["method"]]["inr_db"]
            grid = np.asarray(ref["inr_grid_db"])
            empirical = np.searchsorted(np.sort(actual), grid, side="right") / len(actual)
            median = summary[group][ref["scenario"]][ref["method"]]["median_inr_db"]
            paper_median = ref["quantiles_db"]["0.5"]
            rows.append({"figure": figure, "scenario": ref["scenario"], "method": ref["method"],
                "paper_plotted_median_db": paper_median, "model_median_db": median,
                "median_error_db": median - paper_median if paper_median is not None else None,
                "max_abs_cdf_error_visible_window": float(np.max(np.abs(empirical - np.asarray(ref["cdf"]))))})
    with (output / "paper_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare_assets(root, root / "output")
