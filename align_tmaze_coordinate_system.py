from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR / "tmaze_coordinate_aligned"

# Keep this reference stable so future datasets are mapped into the same
# coordinate system instead of redefining the target on every run.
REFERENCE_FILE_SUBSTRING = "20250608_160522529"
EXCLUDE_FILE_SUBSTRINGS = ("20250703_191708979",)

LIKELIHOOD_THRESHOLD = 0.90
MIN_VALID_PARTS_FOR_CENTER = 2
MIN_CLOUD_POINTS = 500
REPRESENTATIVE_GRID_BINS = 90
MAX_REPRESENTATIVE_POINTS = 8000
ANGLE_SEARCH_DEGREES = 15
ICP_TRIM_QUANTILE = 0.80
ICP_MAX_ITERATIONS = 35
MAZE_DISTANCE_THRESHOLD_PX = 35.0
PHYSICAL_STEM_LENGTH_CM = 89.0
STEM_BAND_HALF_WIDTH_PX = 25.0
STEM_POLYLINE_BINS = 80
CONNECTED_COMPONENT_RADIUS_PX = 45.0


def get_bodyparts(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(col[1] for col in df.columns[1:] if col[2] == "x"))


def read_dlc_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, header=[0, 1, 2])
    return df, get_bodyparts(df)


def part_series(df: pd.DataFrame, part: str, coord: str) -> pd.Series:
    return df.xs((part, coord), axis=1, level=[1, 2]).iloc[:, 0]


def extract_arrays(df: pd.DataFrame, bodyparts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    likelihoods = []
    for part in bodyparts:
        xs.append(part_series(df, part, "x").to_numpy(dtype=float))
        ys.append(part_series(df, part, "y").to_numpy(dtype=float))
        likelihoods.append(part_series(df, part, "likelihood").to_numpy(dtype=float))
    return np.column_stack(xs), np.column_stack(ys), np.column_stack(likelihoods)


def frame_centers(x: np.ndarray, y: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood)
    point_valid = finite & (likelihood >= LIKELIHOOD_THRESHOLD)
    enough_points = point_valid.sum(axis=1) >= MIN_VALID_PARTS_FOR_CENTER

    centers = np.full((len(x), 2), np.nan, dtype=float)
    masked_x = np.where(point_valid, x, np.nan)
    masked_y = np.where(point_valid, y, np.nan)
    centers[enough_points, 0] = np.nanmedian(masked_x[enough_points], axis=1)
    centers[enough_points, 1] = np.nanmedian(masked_y[enough_points], axis=1)
    return centers, enough_points


def finite_points(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def largest_component_mask(points: np.ndarray, radius_px: float = CONNECTED_COMPONENT_RADIUS_PX) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    finite_points_array = points[finite]
    mask = np.zeros(len(points), dtype=bool)
    if len(finite_points_array) == 0:
        return mask

    neighbors = cKDTree(finite_points_array).query_ball_point(finite_points_array, r=radius_px)
    visited = np.zeros(len(finite_points_array), dtype=bool)
    best_component: list[int] = []

    for start in range(len(finite_points_array)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best_component):
            best_component = component

    finite_indices = np.flatnonzero(finite)
    mask[finite_indices[best_component]] = True
    return mask


def representative_cloud(points: np.ndarray) -> np.ndarray:
    points = finite_points(points)
    if len(points) < MIN_CLOUD_POINTS:
        raise ValueError(f"Only {len(points)} valid trajectory points; need at least {MIN_CLOUD_POINTS}.")

    low = np.quantile(points, 0.005, axis=0)
    high = np.quantile(points, 0.995, axis=0)
    inlier = ((points >= low) & (points <= high)).all(axis=1)
    points = points[inlier]

    span = np.maximum(high - low, 1e-9)
    grid = np.floor((points - low) / span * REPRESENTATIVE_GRID_BINS).astype(int)
    grid = np.clip(grid, 0, REPRESENTATIVE_GRID_BINS - 1)
    cell_ids = grid[:, 0] * REPRESENTATIVE_GRID_BINS + grid[:, 1]

    order = np.argsort(cell_ids)
    ordered_ids = cell_ids[order]
    ordered_points = points[order]
    split_at = np.flatnonzero(np.diff(ordered_ids)) + 1
    groups = np.split(ordered_points, split_at)
    reps = np.array([np.median(group, axis=0) for group in groups])

    if len(reps) > MAX_REPRESENTATIVE_POINTS:
        keep = np.linspace(0, len(reps) - 1, MAX_REPRESENTATIVE_POINTS).round().astype(int)
        reps = reps[keep]
    return reps


def estimate_stem_polyline(points: np.ndarray, y_center: float | None = None) -> tuple[np.ndarray, float, float]:
    points = finite_points(points)
    if len(points) < MIN_CLOUD_POINTS:
        raise ValueError(f"Only {len(points)} valid points for stem calibration; need at least {MIN_CLOUD_POINTS}.")

    if y_center is None:
        hist, edges = np.histogram(points[:, 1], bins=90)
        densest = int(np.argmax(hist))
        y_center = float((edges[densest] + edges[densest + 1]) / 2.0)

    band_width = STEM_BAND_HALF_WIDTH_PX
    stem_points = points[np.abs(points[:, 1] - y_center) <= band_width]
    while len(stem_points) < 100 and band_width < STEM_BAND_HALF_WIDTH_PX * 4:
        band_width *= 1.5
        stem_points = points[np.abs(points[:, 1] - y_center) <= band_width]
    if len(stem_points) < 100:
        raise ValueError(f"Only {len(stem_points)} points found near the stem centerline y={y_center:.2f}.")

    center = np.median(stem_points, axis=0)
    centered = stem_points - center
    values, vectors = np.linalg.eigh(centered.T @ centered)
    axis = vectors[:, np.argmax(values)]
    if axis[0] < 0:
        axis = -axis
    projections = centered @ axis
    p_min = float(projections.min())
    p_max = float(projections.max())

    # Use the fitted stem centerline, not the noisy trajectory wiggle, for
    # calibration. This matches "length of stem polyline in pixels".
    polyline = center + np.outer(np.linspace(p_min, p_max, STEM_POLYLINE_BINS + 1), axis)
    stem_len_px = float(p_max - p_min)
    return polyline, stem_len_px, float(y_center)


def stem_axis_from_polyline(polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = polyline[0].astype(float)
    axis = (polyline[-1] - polyline[0]).astype(float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("Stem calibration failed because stem endpoints are identical.")
    return origin, axis / norm


def project_to_stem_cm(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, px_per_cm: float) -> np.ndarray:
    out = np.full_like(points, np.nan, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return out
    perpendicular = np.array([-axis[1], axis[0]], dtype=float)
    shifted = points[finite] - origin
    out[finite, 0] = shifted @ axis / px_per_cm
    out[finite, 1] = shifted @ perpendicular / px_per_cm
    return out


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def pca_angle(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, np.argmax(values)]
    return math.atan2(axis[1], axis[0])


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - source_mean @ rotation.T
    return rotation, translation


def trimmed_nn_score(source_aligned: np.ndarray, reference_tree: cKDTree) -> tuple[float, np.ndarray, np.ndarray]:
    distances, indices = reference_tree.query(source_aligned, k=1, workers=-1)
    cutoff = np.quantile(distances, ICP_TRIM_QUANTILE)
    keep = distances <= cutoff
    return float(distances[keep].mean()), distances, indices


def initial_transform(source: np.ndarray, reference: np.ndarray, reference_tree: cKDTree) -> tuple[np.ndarray, np.ndarray, float]:
    ref_center = np.median(reference, axis=0)
    src_center = np.median(source, axis=0)

    pca_delta = normalize_angle(pca_angle(reference) - pca_angle(source))
    base_angles = [0.0, pca_delta, normalize_angle(pca_delta + math.pi), normalize_angle(pca_delta - math.pi)]
    offsets = np.deg2rad(np.arange(-ANGLE_SEARCH_DEGREES, ANGLE_SEARCH_DEGREES + 1, 1))

    best_score = float("inf")
    best_rotation = np.eye(2)
    best_translation = ref_center - src_center

    for base in base_angles:
        for offset in offsets:
            angle = normalize_angle(base + offset)
            rotation = rotation_matrix(angle)
            rotated_center = src_center @ rotation.T
            base_translation = ref_center - rotated_center
            for dx in (-30.0, 0.0, 30.0):
                for dy in (-30.0, 0.0, 30.0):
                    translation = base_translation + np.array([dx, dy])
                    aligned = transform_points(source, rotation, translation)
                    score, _distances, _indices = trimmed_nn_score(aligned, reference_tree)
                    if score < best_score:
                        best_score = score
                        best_rotation = rotation
                        best_translation = translation

    return best_rotation, best_translation, best_score


def icp_align(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    reference_tree = cKDTree(reference)
    rotation, translation, best_score = initial_transform(source, reference, reference_tree)
    best_rotation = rotation.copy()
    best_translation = translation.copy()

    for _iteration in range(ICP_MAX_ITERATIONS):
        aligned = transform_points(source, rotation, translation)
        score, distances, indices = trimmed_nn_score(aligned, reference_tree)
        if score < best_score:
            best_score = score
            best_rotation = rotation.copy()
            best_translation = translation.copy()

        cutoff = np.quantile(distances, ICP_TRIM_QUANTILE)
        keep = distances <= cutoff
        if keep.sum() < MIN_CLOUD_POINTS:
            break

        matched_reference = reference[indices[keep]]
        next_rotation, next_translation = kabsch_transform(source[keep], matched_reference)
        delta = np.linalg.norm(next_translation - translation) + np.linalg.norm(next_rotation - rotation) * 100.0
        rotation, translation = next_rotation, next_translation
        if delta < 1e-4:
            break

    final_aligned = transform_points(source, best_rotation, best_translation)
    best_score, _distances, _indices = trimmed_nn_score(final_aligned, reference_tree)
    return best_rotation, best_translation, best_score


def rotation_degrees(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def write_aligned_csv(
    path: Path,
    df: pd.DataFrame,
    bodyparts: list[str],
    centers: np.ndarray,
    center_valid: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    reference_tree: cKDTree,
    calibration: dict,
) -> None:
    x, y, likelihood = extract_arrays(df, bodyparts)
    aligned_x = np.full_like(x, np.nan, dtype=float)
    aligned_y = np.full_like(y, np.nan, dtype=float)
    point_finite = np.isfinite(x) & np.isfinite(y)
    for idx in range(len(bodyparts)):
        points = np.column_stack((x[:, idx], y[:, idx]))
        aligned = transform_points(points[point_finite[:, idx]], rotation, translation)
        aligned_x[point_finite[:, idx], idx] = aligned[:, 0]
        aligned_y[point_finite[:, idx], idx] = aligned[:, 1]

    aligned_centers = np.full_like(centers, np.nan, dtype=float)
    center_finite = np.isfinite(centers).all(axis=1)
    aligned_centers[center_finite] = transform_points(centers[center_finite], rotation, translation)
    center_ref_distance = np.full(len(centers), np.nan, dtype=float)
    center_ref_distance[center_finite] = reference_tree.query(aligned_centers[center_finite], k=1, workers=-1)[0]
    center_in_maze = center_valid & (center_ref_distance <= MAZE_DISTANCE_THRESHOLD_PX)
    center_cm = project_to_stem_cm(
        aligned_centers,
        calibration["stem_origin_px"],
        calibration["stem_axis_unit"],
        calibration["px_per_cm"],
    )

    out = pd.DataFrame(
        {
            "frame": df.iloc[:, 0],
            "center_valid": center_valid.astype(int),
            "center_in_reference_maze": center_in_maze.astype(int),
            "center_ref_distance_px": center_ref_distance,
            "center_x_raw": centers[:, 0],
            "center_y_raw": centers[:, 1],
            "center_x_aligned": aligned_centers[:, 0],
            "center_y_aligned": aligned_centers[:, 1],
            "center_x_cm": center_cm[:, 0],
            "center_y_cm": center_cm[:, 1],
        }
    )
    for idx, part in enumerate(bodyparts):
        part_valid = np.isfinite(x[:, idx]) & np.isfinite(y[:, idx]) & (likelihood[:, idx] >= LIKELIHOOD_THRESHOLD)
        out[f"{part}_x_raw"] = x[:, idx]
        out[f"{part}_y_raw"] = y[:, idx]
        out[f"{part}_likelihood"] = likelihood[:, idx]
        out[f"{part}_valid"] = part_valid.astype(int)
        out[f"{part}_x_aligned"] = aligned_x[:, idx]
        out[f"{part}_y_aligned"] = aligned_y[:, idx]
        part_cm = project_to_stem_cm(
            np.column_stack((aligned_x[:, idx], aligned_y[:, idx])),
            calibration["stem_origin_px"],
            calibration["stem_axis_unit"],
            calibration["px_per_cm"],
        )
        out[f"{part}_x_cm"] = part_cm[:, 0]
        out[f"{part}_y_cm"] = part_cm[:, 1]

    out.to_csv(path, index=False)


def sample_for_plot(points: np.ndarray, max_points: int = 7000) -> np.ndarray:
    points = finite_points(points)
    if len(points) <= max_points:
        return points
    keep = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[keep]


def set_equal_axes(ax: plt.Axes, *clouds: np.ndarray) -> None:
    points = np.vstack([finite_points(cloud) for cloud in clouds if len(finite_points(cloud))])
    low = np.nanquantile(points, 0.005, axis=0)
    high = np.nanquantile(points, 0.995, axis=0)
    center = (low + high) / 2
    half_span = max(high - low) / 2 * 1.08
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] + half_span, center[1] - half_span)
    ax.set_aspect("equal", adjustable="box")


def plot_dataset(
    name: str,
    centers: np.ndarray,
    aligned_centers: np.ndarray,
    reference_cloud: np.ndarray,
    representative: np.ndarray,
    aligned_representative: np.ndarray,
    calibration: dict,
    score: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    raw_sample = sample_for_plot(centers)
    aligned_centers_cm = project_to_stem_cm(
        aligned_centers,
        calibration["stem_origin_px"],
        calibration["stem_axis_unit"],
        calibration["px_per_cm"],
    )
    reference_cloud_cm = project_to_stem_cm(
        reference_cloud,
        calibration["stem_origin_px"],
        calibration["stem_axis_unit"],
        calibration["px_per_cm"],
    )
    aligned_representative_cm = project_to_stem_cm(
        aligned_representative,
        calibration["stem_origin_px"],
        calibration["stem_axis_unit"],
        calibration["px_per_cm"],
    )
    aligned_sample_cm = sample_for_plot(aligned_centers_cm)
    ref_sample_cm = sample_for_plot(reference_cloud_cm)

    axes[0, 0].scatter(raw_sample[:, 0], raw_sample[:, 1], s=2, alpha=0.35, color="#1f77b4")
    axes[0, 0].set_title("Raw trajectory center (px)")
    axes[0, 0].set_xlabel("x (px)")
    axes[0, 0].set_ylabel("y (px)")
    set_equal_axes(axes[0, 0], raw_sample)

    axes[0, 1].scatter(ref_sample_cm[:, 0], ref_sample_cm[:, 1], s=3, alpha=0.25, color="#444444", label="reference")
    axes[0, 1].scatter(aligned_sample_cm[:, 0], aligned_sample_cm[:, 1], s=2, alpha=0.35, color="#d62728", label="aligned")
    axes[0, 1].set_title(f"Aligned to reference (cm), trimmed NN={score:.2f}px")
    axes[0, 1].set_xlabel("x along stem (cm)")
    axes[0, 1].set_ylabel("y across stem (cm)")
    axes[0, 1].legend(loc="best")
    set_equal_axes(axes[0, 1], ref_sample_cm, aligned_sample_cm)

    axes[1, 0].scatter(representative[:, 0], representative[:, 1], s=7, alpha=0.55, color="#1f77b4")
    axes[1, 0].set_title("Representative raw cloud (px)")
    axes[1, 0].set_xlabel("x (px)")
    axes[1, 0].set_ylabel("y (px)")
    set_equal_axes(axes[1, 0], representative)

    axes[1, 1].scatter(reference_cloud_cm[:, 0], reference_cloud_cm[:, 1], s=7, alpha=0.30, color="#444444", label="reference")
    axes[1, 1].scatter(aligned_representative_cm[:, 0], aligned_representative_cm[:, 1], s=7, alpha=0.55, color="#d62728", label="aligned")
    axes[1, 1].set_title("Representative cloud after transform (cm)")
    axes[1, 1].set_xlabel("x along stem (cm)")
    axes[1, 1].set_ylabel("y across stem (cm)")
    axes[1, 1].legend(loc="best")
    set_equal_axes(axes[1, 1], reference_cloud_cm, aligned_representative_cm)

    fig.suptitle(name)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_combined(clouds: list[dict], output_path: Path, aligned: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    plotted_clouds = []
    for idx, item in enumerate(clouds):
        points = item["aligned_centers_cm"] if aligned else item["centers"]
        sample = sample_for_plot(points, max_points=6000)
        plotted_clouds.append(sample)
        ax.scatter(sample[:, 0], sample[:, 1], s=2, alpha=0.30, color=colors[idx % len(colors)], label=item["label"])
    if aligned:
        ax.set_title("Aligned trajectory overlay (cm)")
        ax.set_xlabel("x along stem (cm)")
        ax.set_ylabel("y across stem (cm)")
    else:
        ax.set_title("Raw trajectory overlay (px)")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.legend(loc="best")
    set_equal_axes(ax, *plotted_clouds)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_paths = sorted(INPUT_DIR.glob("*.csv"))
    skipped = []
    paths = []
    for path in all_paths:
        matched = [token for token in EXCLUDE_FILE_SUBSTRINGS if token in path.name]
        if matched:
            skipped.append({"file": path.name, "reason": f"excluded by pattern: {matched[0]}"})
        else:
            paths.append(path)

    if not paths:
        raise SystemExit("No CSV files left after exclusions.")

    reference_matches = [path for path in paths if REFERENCE_FILE_SUBSTRING in path.name]
    if len(reference_matches) != 1:
        raise SystemExit(f"Expected exactly one reference file containing {REFERENCE_FILE_SUBSTRING!r}.")
    reference_path = reference_matches[0]

    data = {}
    reference_bodyparts = None
    for path in paths:
        df, bodyparts = read_dlc_csv(path)
        if reference_bodyparts is None:
            reference_bodyparts = bodyparts
        elif bodyparts != reference_bodyparts:
            raise SystemExit(f"Bodypart columns differ in {path.name}.")
        x, y, likelihood = extract_arrays(df, bodyparts)
        centers, center_valid = frame_centers(x, y, likelihood)
        representative = representative_cloud(centers)
        data[path] = {
            "df": df,
            "bodyparts": bodyparts,
            "centers": centers,
            "center_valid": center_valid,
            "representative": representative,
        }

    reference_raw = data[reference_path]["representative"]
    reference_component_mask = largest_component_mask(reference_raw)
    reference = reference_raw[reference_component_mask]
    reference_tree = cKDTree(reference)
    reference_stem_polyline, reference_stem_len_px, reference_stem_y = estimate_stem_polyline(reference)
    stem_origin_px, stem_axis_unit = stem_axis_from_polyline(reference_stem_polyline)
    pd.DataFrame(reference_raw, columns=["x_reference_raw", "y_reference_raw"]).to_csv(
        OUTPUT_DIR / "reference_cloud_raw.csv", index=False
    )
    pd.DataFrame(reference, columns=["x_reference", "y_reference"]).to_csv(OUTPUT_DIR / "reference_cloud.csv", index=False)
    pd.DataFrame(reference_stem_polyline, columns=["x_px", "y_px"]).to_csv(
        OUTPUT_DIR / "reference_stem_polyline.csv", index=False
    )

    summary = []
    plot_items = []
    for path in paths:
        item = data[path]
        if path == reference_path:
            rotation = np.eye(2)
            translation = np.array([0.0, 0.0])
            score = 0.0
            role = "reference"
        else:
            rotation, translation, score = icp_align(item["representative"], reference)
            role = "aligned"

        centers = item["centers"]
        center_finite = np.isfinite(centers).all(axis=1)
        aligned_centers = np.full_like(centers, np.nan, dtype=float)
        aligned_centers[center_finite] = transform_points(centers[center_finite], rotation, translation)
        aligned_representative = transform_points(item["representative"], rotation, translation)

        valid_count = int(item["center_valid"].sum())
        center_ref_distance = np.full(len(centers), np.nan, dtype=float)
        center_ref_distance[center_finite] = reference_tree.query(aligned_centers[center_finite], k=1, workers=-1)[0]
        in_maze = item["center_valid"] & (center_ref_distance <= MAZE_DISTANCE_THRESHOLD_PX)
        in_maze_count = int(in_maze.sum())
        valid_distances = center_ref_distance[item["center_valid"]]
        representative_distances = reference_tree.query(aligned_representative, k=1, workers=-1)[0]
        representative_in_maze = representative_distances <= MAZE_DISTANCE_THRESHOLD_PX
        plot_centers = centers[in_maze]
        plot_aligned_centers = aligned_centers[in_maze]
        plot_representative = item["representative"][representative_in_maze]
        plot_aligned_representative = aligned_representative[representative_in_maze]
        stem_source = aligned_centers[in_maze]
        if len(finite_points(stem_source)) < MIN_CLOUD_POINTS:
            stem_source = plot_aligned_representative
        stem_polyline, stem_len_px, stem_y_center = estimate_stem_polyline(stem_source, reference_stem_y)
        px_per_cm = stem_len_px / PHYSICAL_STEM_LENGTH_CM
        calibration = {
            "stem_origin_px": stem_origin_px,
            "stem_axis_unit": stem_axis_unit,
            "stem_len_px": stem_len_px,
            "px_per_cm": px_per_cm,
        }

        aligned_csv = OUTPUT_DIR / f"{path.stem}_coordinate_aligned.csv"
        overview_png = OUTPUT_DIR / f"{path.stem}_coordinate_alignment.png"
        write_aligned_csv(
            aligned_csv,
            item["df"],
            item["bodyparts"],
            centers,
            item["center_valid"],
            rotation,
            translation,
            reference_tree,
            calibration,
        )
        plot_dataset(
            path.name,
            plot_centers,
            plot_aligned_centers,
            reference,
            plot_representative,
            plot_aligned_representative,
            calibration,
            score,
            overview_png,
        )

        frames = len(item["df"])
        summary.append(
            {
                "file": path.name,
                "role": role,
                "frames": frames,
                "valid_center_frames": valid_count,
                "valid_center_ratio": valid_count / frames,
                "center_in_reference_maze_frames": in_maze_count,
                "center_in_reference_maze_ratio_of_valid": in_maze_count / valid_count if valid_count else np.nan,
                "center_ref_distance_q95_px": np.nanquantile(valid_distances, 0.95) if len(valid_distances) else np.nan,
                "representative_cloud_points": len(item["representative"]),
                "representative_cloud_points_plotted": int(representative_in_maze.sum()),
                "stem_len_px": stem_len_px,
                "px_per_cm": px_per_cm,
                "physical_stem_length_cm": PHYSICAL_STEM_LENGTH_CM,
                "stem_y_center_px": stem_y_center,
                "rotation_degrees_to_reference": rotation_degrees(rotation),
                "translation_x_to_reference": translation[0],
                "translation_y_to_reference": translation[1],
                "trimmed_nn_score_px": score,
                "aligned_csv": aligned_csv.name,
                "overview_png": overview_png.name,
            }
        )
        plot_items.append(
            {
                "label": path.name.split("DLC_")[0].replace("Basler_a2A1920-160ucBAS__40293797__", ""),
                "centers": plot_centers,
                "aligned_centers": plot_aligned_centers,
                "aligned_centers_cm": project_to_stem_cm(
                    plot_aligned_centers,
                    calibration["stem_origin_px"],
                    calibration["stem_axis_unit"],
                    calibration["px_per_cm"],
                ),
            }
        )

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "alignment_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUTPUT_DIR / "skipped_files.csv", index=False)

    plot_combined(plot_items, OUTPUT_DIR / "combined_raw_overlay.png", aligned=False)
    plot_combined(plot_items, OUTPUT_DIR / "combined_aligned_overlay.png", aligned=True)

    config = {
        "method": "session-level rigid registration of representative trajectory-center point clouds",
        "reference_file_substring": REFERENCE_FILE_SUBSTRING,
        "excluded_file_substrings": EXCLUDE_FILE_SUBSTRINGS,
        "likelihood_threshold": LIKELIHOOD_THRESHOLD,
        "min_valid_parts_for_center": MIN_VALID_PARTS_FOR_CENTER,
        "representative_grid_bins": REPRESENTATIVE_GRID_BINS,
        "icp_trim_quantile": ICP_TRIM_QUANTILE,
        "maze_distance_threshold_px": MAZE_DISTANCE_THRESHOLD_PX,
        "physical_stem_length_cm": PHYSICAL_STEM_LENGTH_CM,
        "stem_band_half_width_px": STEM_BAND_HALF_WIDTH_PX,
        "connected_component_radius_px": CONNECTED_COMPONENT_RADIUS_PX,
        "cm_coordinate_origin": "reference stem start",
        "cm_coordinate_x_axis": "reference stem direction",
    }
    (OUTPUT_DIR / "alignment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    index = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        "<title>T-maze coordinate alignment</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35} img{max-width:100%;border:1px solid #ccc;margin:8px 0 28px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px;text-align:left}</style>",
        "<h1>T-maze coordinate alignment</h1>",
        "<p>Session-level rigid registration. Raw panels are in pixels; aligned panels are calibrated to cm using the 89 cm stem. Plots show QC-filtered points inside the reference maze body.</p>",
        '<p><a href="alignment_summary.csv">alignment_summary.csv</a> | <a href="skipped_files.csv">skipped_files.csv</a> | <a href="alignment_config.json">alignment_config.json</a></p>',
        "<h2>Raw overlay</h2>",
        '<img src="combined_raw_overlay.png" alt="Raw overlay">',
        "<h2>Aligned overlay</h2>",
        '<img src="combined_aligned_overlay.png" alt="Aligned overlay">',
    ]
    for row in summary:
        index.append(f"<h2>{row['file']}</h2>")
        index.append(f"<p>role={row['role']}; score={row['trimmed_nn_score_px']:.2f}px; aligned CSV: {row['aligned_csv']}</p>")
        index.append(f'<img src="{row["overview_png"]}" alt="{row["file"]} alignment overview">')
    (OUTPUT_DIR / "index.html").write_text("\n".join(index), encoding="utf-8")

    print(f"Wrote results to {OUTPUT_DIR}")
    print(pd.DataFrame(summary).to_string(index=False))
    if skipped:
        print("Skipped files:")
        print(pd.DataFrame(skipped).to_string(index=False))


if __name__ == "__main__":
    main()
