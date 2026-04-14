import csv
import html
import math
import os
from pathlib import Path


INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR / "analysis_results"
LIKELIHOOD_THRESHOLD = 0.90
MIN_TEMPLATE_FRAMES = 10


def median(values):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def quantile(values, q):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def read_dlc_csv(path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        scorer = next(reader)
        bodyparts_row = next(reader)
        coords_row = next(reader)

        bodyparts = []
        columns = []
        for col in range(1, len(scorer)):
            if coords_row[col] == "x":
                part = bodyparts_row[col]
                bodyparts.append(part)
                columns.append((part, col, col + 1, col + 2))

        frames = []
        for row in reader:
            if not row:
                continue
            frame_id = row[0]
            points = []
            likelihoods = []
            for _part, x_col, y_col, l_col in columns:
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                    likelihood = float(row[l_col])
                except (ValueError, IndexError):
                    x = y = likelihood = float("nan")
                points.append((x, y))
                likelihoods.append(likelihood)
            frames.append({"frame": frame_id, "points": points, "likelihoods": likelihoods})

    return bodyparts, frames


def centroid(points):
    return (sum(x for x, _y in points) / len(points), sum(y for _x, y in points) / len(points))


def center(points):
    cx, cy = centroid(points)
    return [(x - cx, y - cy) for x, y in points], (cx, cy)


def rotate(points, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def rms_radius(points):
    if not points:
        return 0.0
    return math.sqrt(sum(x * x + y * y for x, y in points) / len(points))


def kabsch_angle(points, template):
    # Closed-form 2D Kabsch rotation for centered row-vector points.
    a = 0.0
    b = 0.0
    for (px, py), (qx, qy) in zip(points, template):
        a += px * qx + py * qy
        b += px * qy - py * qx
    return math.atan2(b, a)


def align_to_template(points, template):
    centered, raw_centroid = center(points)
    angle = kabsch_angle(centered, template)
    aligned = rotate(centered, angle)
    residual = math.sqrt(
        sum((x - tx) ** 2 + (y - ty) ** 2 for (x, y), (tx, ty) in zip(aligned, template))
        / len(template)
    )
    return aligned, raw_centroid, angle, residual


def valid_frame(frame, threshold):
    return all(math.isfinite(v) and v >= threshold for v in frame["likelihoods"]) and all(
        math.isfinite(x) and math.isfinite(y) for x, y in frame["points"]
    )


def build_initial_template(frames):
    shapes = []
    for frame in frames:
        if valid_frame(frame, LIKELIHOOD_THRESHOLD):
            centered, _ = center(frame["points"])
            if rms_radius(centered) > 1e-9:
                shapes.append(centered)

    if len(shapes) < MIN_TEMPLATE_FRAMES:
        return None, len(shapes)

    template = shapes[0]
    for _ in range(4):
        aligned_shapes = []
        for shape in shapes:
            angle = kabsch_angle(shape, template)
            aligned_shapes.append(rotate(shape, angle))
        next_template = []
        for point_idx in range(len(template)):
            xs = [shape[point_idx][0] for shape in aligned_shapes]
            ys = [shape[point_idx][1] for shape in aligned_shapes]
            next_template.append((median(xs), median(ys)))
        centered_template, _ = center(next_template)
        template = centered_template
    return template, len(shapes)


def build_global_template(file_templates):
    base = file_templates[0]
    aligned = []
    for template in file_templates:
        angle = kabsch_angle(template, base)
        aligned.append(rotate(template, angle))
    global_template = []
    for idx in range(len(base)):
        global_template.append((median([t[idx][0] for t in aligned]), median([t[idx][1] for t in aligned])))
    global_template, _ = center(global_template)
    return global_template


def scale_points(points, width, height, pad=26):
    xs = [p[0] for p in points if math.isfinite(p[0]) and math.isfinite(p[1])]
    ys = [p[1] for p in points if math.isfinite(p[0]) and math.isfinite(p[1])]
    if not xs or not ys:
        return []
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min((width - pad * 2) / span_x, (height - pad * 2) / span_y)
    out = []
    for x, y in points:
        sx = pad + (x - min_x) * scale
        sy = height - pad - (y - min_y) * scale
        out.append((sx, sy))
    return out


def polyline(points, color, width=1.0, opacity=1.0):
    if len(points) < 2:
        return ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" />'
    )


def sample_points(points, max_points=2500):
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def plot_timeseries(values, width, height, color, qmax=None):
    clean = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if len(clean) < 2:
        return ""
    max_i = max(i for i, _ in clean)
    min_v = min(v for _, v in clean)
    max_v = qmax if qmax is not None else max(v for _, v in clean)
    if max_v <= min_v:
        max_v = min_v + 1.0
    pts = []
    for i, v in clean:
        clipped = min(v, max_v)
        x = 28 + i / max(max_i, 1) * (width - 56)
        y = height - 24 - (clipped - min_v) / (max_v - min_v) * (height - 48)
        pts.append((x, y))
    return polyline(pts, color, 1.1, 0.9)


def draw_scatter(points, color, radius=1.2, opacity=0.35):
    return "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}" />'
        for x, y in points
    )


def make_dataset_svg(result, bodyparts, output_path):
    width = 1180
    height = 820
    panel_w = 540
    panel_h = 300

    raw_points_by_part = [[] for _ in bodyparts]
    norm_points_by_part = [[] for _ in bodyparts]
    for frame in result["processed"]:
        if not frame["valid"]:
            continue
        for idx, point in enumerate(frame["raw"]):
            raw_points_by_part[idx].append(point)
        for idx, point in enumerate(frame["aligned"]):
            norm_points_by_part[idx].append(point)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="28" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(result["name"])}</text>')
    parts.append(
        f'<text x="28" y="60" font-family="Arial" font-size="13" fill="#444">'
        f'frames={result["frames"]}, valid={result["valid_frames"]} ({result["valid_ratio"]:.1%}), '
        f'median residual={result["median_residual"]:.2f}px, likelihood threshold={LIKELIHOOD_THRESHOLD}</text>'
    )

    def panel(x, y, title):
        parts.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#cccccc"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 22}" font-family="Arial" font-size="15" font-weight="700">{title}</text>')

    panel(28, 82, "Raw valid positions")
    panel(610, 82, "Normalized positions")
    panel(28, 430, "Residual over time")
    panel(610, 430, "Minimum likelihood over time")

    for idx, pts in enumerate(raw_points_by_part):
        scaled = scale_points(sample_points(pts), panel_w - 20, panel_h - 46)
        shifted = [(x + 38, y + 116) for x, y in scaled]
        parts.append(draw_scatter(shifted, colors[idx % len(colors)], 1.1, 0.32))

    for idx, pts in enumerate(norm_points_by_part):
        scaled = scale_points(sample_points(pts), panel_w - 20, panel_h - 46)
        shifted = [(x + 620, y + 116) for x, y in scaled]
        parts.append(draw_scatter(shifted, colors[idx % len(colors)], 1.1, 0.32))

    residual_path = plot_timeseries(result["residuals"], panel_w, panel_h - 44, "#111111", result["residual_q95"])
    parts.append(f'<g transform="translate(28,468)">{residual_path}</g>')
    parts.append(f'<text x="40" y="712" font-family="Arial" font-size="12" fill="#555">95% scale cap: {result["residual_q95"]:.2f}px</text>')

    min_like_path = plot_timeseries(result["min_likelihoods"], panel_w, panel_h - 44, "#0b6e4f", 1.0)
    parts.append(f'<g transform="translate(610,468)">{min_like_path}</g>')
    parts.append(f'<line x1="638" y1="494" x2="1122" y2="494" stroke="#c33" stroke-dasharray="4 4" stroke-width="1"/>')
    parts.append(f'<text x="622" y="712" font-family="Arial" font-size="12" fill="#555">red dashed line marks threshold {LIKELIHOOD_THRESHOLD}</text>')

    legend_x = 28
    legend_y = 776
    for idx, part in enumerate(bodyparts):
        x = legend_x + idx * 150
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{x + 10}" y="{legend_y + 4}" font-family="Arial" font-size="13">{html.escape(part)}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def process_file(path, bodyparts, frames, global_template):
    processed = []
    residuals = []
    min_likelihoods = []
    for frame in frames:
        min_likelihood = min(frame["likelihoods"])
        min_likelihoods.append(min_likelihood)
        is_valid = valid_frame(frame, LIKELIHOOD_THRESHOLD) and rms_radius(center(frame["points"])[0]) > 1e-9
        if is_valid:
            aligned, raw_centroid, angle, residual = align_to_template(frame["points"], global_template)
            residuals.append(residual)
        else:
            aligned = [(float("nan"), float("nan")) for _ in bodyparts]
            raw_centroid = (float("nan"), float("nan"))
            angle = float("nan")
            residual = float("nan")
        processed.append(
            {
                "frame": frame["frame"],
                "valid": is_valid,
                "raw": frame["points"],
                "aligned": aligned,
                "centroid": raw_centroid,
                "angle": angle,
                "residual": residual,
                "min_likelihood": min_likelihood,
            }
        )

    valid_count = sum(1 for row in processed if row["valid"])
    result = {
        "name": path.stem,
        "frames": len(frames),
        "valid_frames": valid_count,
        "valid_ratio": valid_count / len(frames) if frames else 0.0,
        "processed": processed,
        "residuals": residuals,
        "min_likelihoods": min_likelihoods,
        "median_residual": median(residuals),
        "mean_residual": mean(residuals),
        "residual_q95": quantile(residuals, 0.95),
    }
    return result


def write_normalized_csv(result, bodyparts, path):
    header = ["frame", "valid", "centroid_x", "centroid_y", "rotation_deg", "residual_px", "min_likelihood"]
    for part in bodyparts:
        header.extend([f"{part}_x_norm", f"{part}_y_norm"])

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in result["processed"]:
            angle_deg = math.degrees(row["angle"]) if math.isfinite(row["angle"]) else ""
            out = [
                row["frame"],
                int(row["valid"]),
                f'{row["centroid"][0]:.6f}' if math.isfinite(row["centroid"][0]) else "",
                f'{row["centroid"][1]:.6f}' if math.isfinite(row["centroid"][1]) else "",
                f"{angle_deg:.6f}" if angle_deg != "" else "",
                f'{row["residual"]:.6f}' if math.isfinite(row["residual"]) else "",
                f'{row["min_likelihood"]:.6f}' if math.isfinite(row["min_likelihood"]) else "",
            ]
            for x, y in row["aligned"]:
                out.extend([f"{x:.6f}" if math.isfinite(x) else "", f"{y:.6f}" if math.isfinite(y) else ""])
            writer.writerow(out)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit("No CSV files found.")

    datasets = []
    file_templates = []
    bodyparts = None

    for path in csv_paths:
        parts, frames = read_dlc_csv(path)
        if bodyparts is None:
            bodyparts = parts
        elif bodyparts != parts:
            raise SystemExit(f"Bodypart columns differ in {path.name}")
        template, template_frames = build_initial_template(frames)
        if template is None:
            raise SystemExit(f"Not enough high-confidence frames to build template for {path.name}: {template_frames}")
        datasets.append({"path": path, "frames": frames, "template_frames": template_frames})
        file_templates.append(template)

    global_template = build_global_template(file_templates)

    with (OUTPUT_DIR / "global_template.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bodypart", "x_template", "y_template"])
        for part, (x, y) in zip(bodyparts, global_template):
            writer.writerow([part, f"{x:.6f}", f"{y:.6f}"])

    summary_rows = []
    svg_links = []
    for dataset in datasets:
        result = process_file(dataset["path"], bodyparts, dataset["frames"], global_template)
        safe_stem = dataset["path"].stem
        normalized_path = OUTPUT_DIR / f"{safe_stem}_normalized.csv"
        svg_path = OUTPUT_DIR / f"{safe_stem}_overview.svg"
        write_normalized_csv(result, bodyparts, normalized_path)
        make_dataset_svg(result, bodyparts, svg_path)
        summary_rows.append(
            [
                dataset["path"].name,
                result["frames"],
                dataset["template_frames"],
                result["valid_frames"],
                f'{result["valid_ratio"]:.6f}',
                f'{result["median_residual"]:.6f}',
                f'{result["mean_residual"]:.6f}',
                f'{result["residual_q95"]:.6f}',
                normalized_path.name,
                svg_path.name,
            ]
        )
        svg_links.append((dataset["path"].name, svg_path.name))

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "file",
                "frames",
                "template_frames",
                "valid_frames",
                "valid_ratio",
                "median_residual_px",
                "mean_residual_px",
                "residual_q95_px",
                "normalized_csv",
                "overview_svg",
            ]
        )
        writer.writerows(summary_rows)

    index = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        "<title>Tmaze normalization results</title>",
        '<style>body{font-family:Arial,sans-serif;margin:24px} img{max-width:100%;border:1px solid #ccc;margin:8px 0 28px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px}</style>',
        "<h1>Tmaze normalization results</h1>",
        f"<p>Method: centroid translation removal + Kabsch rigid rotation to a global template. Likelihood threshold: {LIKELIHOOD_THRESHOLD}.</p>",
        '<p><a href="summary.csv">summary.csv</a> | <a href="global_template.csv">global_template.csv</a></p>',
    ]
    for name, svg_name in svg_links:
        index.append(f"<h2>{html.escape(name)}</h2>")
        index.append(f'<img src="{html.escape(svg_name)}" alt="{html.escape(name)} overview">')
    (OUTPUT_DIR / "index.html").write_text("\n".join(index), encoding="utf-8")

    print(f"Wrote results to {OUTPUT_DIR}")
    for row in summary_rows:
        print(row[0], "valid_ratio", row[4], "median_residual_px", row[5], "overview", row[9])


if __name__ == "__main__":
    main()
