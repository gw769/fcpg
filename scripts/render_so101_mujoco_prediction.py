#!/usr/bin/env python3
"""Render honest SO-101 state forecasts in the official MuJoCo model."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from fcpg.so101_kinematics import (
    SO101_MUJOCO_JOINT_NAMES,
    SO101MuJoCoKinematics,
    normalized_positions_to_mujoco_qpos,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.visualize_so101_lewm import (
    ACTUAL_COLOR,
    CANVAS_SIZE,
    PREDICTED_COLOR,
    RawFFmpegWriter,
    atomic_write_text,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    choose_episode,
    choose_one_episode_per_task,
    episode_task_key,
    load_caches,
    load_font,
    prediction_cycle_predictions,
    report_slug,
    write_json,
)


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "scene.xml"
)
PANEL_SIZE = (570, 480)


def fit_panel_image(image: np.ndarray, size: tuple[int, int] = PANEL_SIZE) -> np.ndarray:
    """Letterbox one RGB frame into a fixed panel without changing its aspect ratio."""

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"panel image must have shape [H, W, 3], got {values.shape}")
    if values.dtype != np.uint8:
        raise ValueError(f"panel image must be uint8, got {values.dtype}")
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("panel size must be positive")
    source = Image.fromarray(values, mode="RGB")
    scale = min(target_width / source.width, target_height / source.height)
    resized = source.resize(
        (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", size, "#171a1c")
    panel.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return np.asarray(panel, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render recorded and LeWM-predicted SO-101 joint states with the same "
            "official MuJoCo model and camera."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", metavar="SOURCE:ID")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--cache-base", type=Path)
    parser.add_argument(
        "--external-cache",
        type=Path,
        action="append",
        help=(
            "Select episodes from an external SO-101 cache while keeping the "
            "checkpoint and model frozen. Every external episode is treated as held out."
        ),
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cycle-horizon", type=int, default=8)
    parser.add_argument(
        "--layout",
        choices=("frozen-observation", "actual-vs-predicted"),
        default="frozen-observation",
        help=(
            "frozen-observation keeps the latest observed pose still while H1-H8 "
            "play on the right; actual-vs-predicted moves both panels to the target time."
        ),
    )
    parser.add_argument(
        "--origin-stride",
        type=int,
        default=1,
        help="Render every Nth forecast origin; all H1..H8 steps remain visible.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--media", choices=("mp4", "gif", "both"), default="both")
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=0.72)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.episode and args.all_tasks:
        parser.error("--episode and --all-tasks are mutually exclusive")
    if min(args.batch_size, args.threads, args.cycle_horizon, args.origin_stride) <= 0:
        parser.error("batch-size, threads, cycle-horizon and origin-stride must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.scene.is_file():
        parser.error(f"SO-101 MuJoCo scene not found: {args.scene}")
    return args


class SO101ReplayRenderer:
    def __init__(
        self,
        scene: Path,
        *,
        azimuth: float,
        elevation: float,
        distance: float,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=PANEL_SIZE[1], width=PANEL_SIZE[0]
        )
        joint_ids = []
        qpos_addresses = []
        for name in SO101_MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"MuJoCo scene is missing SO-101 joint {name!r}")
            joint_ids.append(joint_id)
            qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self.joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self.qpos_addresses = np.asarray(qpos_addresses, dtype=np.int32)
        self.joint_ranges = np.asarray(
            self.model.jnt_range[self.joint_ids], dtype=np.float64
        ).copy()
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        if self.site_id < 0:
            raise ValueError("MuJoCo scene is missing site 'gripperframe'")
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.18]
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation

    def render(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("SO-101 replay state must be one finite six-joint vector")
        clipped = bool(
            np.any(np.abs(values[:5]) > 100.0)
            or values[5] < 0.0
            or values[5] > 100.0
        )
        self.data.qpos[self.qpos_addresses] = normalized_positions_to_mujoco_qpos(
            values, self.joint_ranges
        )
        self.mujoco.mj_forward(self.model, self.data)
        xyz = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy(), xyz, clipped

    def close(self) -> None:
        self.renderer.close()


def selected_prediction_indices(display_inputs: np.ndarray, origin_stride: int) -> np.ndarray:
    values = np.asarray(display_inputs, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("prediction display inputs must be a non-empty vector")
    origins = []
    last = None
    for value in values:
        if last is None or int(value) != last:
            origins.append(int(value))
            last = int(value)
    selected = set(origins[::origin_stride])
    if origins[-1] not in selected:
        selected.add(origins[-1])
    return np.flatnonzero(np.isin(values, np.asarray(sorted(selected), dtype=np.int64)))


def external_validation_assignments(caches: list[Any]) -> dict[tuple[str, int], str]:
    assignments: dict[tuple[str, int], str] = {}
    for cache in caches:
        for raw_episode in np.unique(np.asarray(cache.data.episodes)):
            assignments[(cache.dataset_id, int(raw_episode))] = "val"
    if not assignments:
        raise ValueError("external caches contain no episodes")
    return assignments


def draw_xy_trace(
    draw: ImageDraw.ImageDraw,
    actual: np.ndarray,
    predicted: np.ndarray,
    current: int,
) -> None:
    left, top, width, height = 60, 655, 520, 120
    both = np.concatenate([actual[:, :2], predicted[:, :2]], axis=0) * 1000.0
    low = np.min(both, axis=0)
    high = np.max(both, axis=0)
    span = np.maximum(high - low, 1.0)
    margin = np.maximum(0.08 * span, 2.0)
    low -= margin
    high += margin

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        xy = values[:, :2] * 1000.0
        return [
            (
                left + width * float((point[0] - low[0]) / (high[0] - low[0])),
                top + height * (1.0 - float((point[1] - low[1]) / (high[1] - low[1]))),
            )
            for point in xy[: current + 1]
        ]

    draw.rectangle((left, top, left + width, top + height), fill="#f7f8f9", outline="#bcc5ca")
    actual_points = points(actual)
    predicted_points = points(predicted)
    if len(actual_points) > 1:
        draw.line(actual_points, fill=ACTUAL_COLOR, width=4)
        draw.line(predicted_points, fill=PREDICTED_COLOR, width=4)
    if actual_points:
        ax, ay = actual_points[-1]
        px, py = predicted_points[-1]
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=ACTUAL_COLOR)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=PREDICTED_COLOR)
    draw.text((left, top - 24), "Current H1-H8 base-frame XY path", fill="#20242a", font=load_font(18, True))


def compose_frame(
    left_image: np.ndarray,
    predicted_image: np.ndarray,
    *,
    source: str,
    task: str,
    episode: int,
    checkpoint_step: int,
    horizon: int,
    raw_ahead: int,
    actual_state: np.ndarray,
    predicted_state: np.ndarray,
    actual_xyz: np.ndarray,
    predicted_xyz: np.ndarray,
    cycle_actual_xyz: np.ndarray,
    cycle_predicted_xyz: np.ndarray,
    cycle_index: int,
    predicted_clipped: bool,
    frozen_observation: bool,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, "#eef1f2")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, True)
    label_font = load_font(20, True)
    text_font = load_font(16)
    draw.text((40, 22), "SO-ARM101 MuJoCo: recorded vs LeWM forecast", fill="#182024", font=title_font)
    draw.text(
        (40, 58),
        f"{source} | {task} | episode {episode} | checkpoint step {checkpoint_step}",
        fill="#4a5358",
        font=text_font,
    )
    draw.text(
        (40, 84),
        f"H{horizon}: predicted {raw_ahead} raw steps earlier from prior RGB/state and recorded actions",
        fill="#4a5358",
        font=text_font,
    )
    actual = Image.fromarray(fit_panel_image(left_image))
    predicted = Image.fromarray(fit_panel_image(predicted_image))
    canvas.paste(actual, (40, 130))
    canvas.paste(predicted, (670, 130))
    draw.rectangle((40, 130, 610, 610), outline=ACTUAL_COLOR, width=5)
    draw.rectangle((670, 130, 1240, 610), outline=PREDICTED_COLOR, width=5)
    left_label = (
        "REAL CAMERA OBSERVATION (FROZEN)"
        if frozen_observation
        else "REAL RECORDED FUTURE FRAME"
    )
    draw.text((56, 145), left_label, fill=ACTUAL_COLOR, font=label_font)
    draw.text((686, 145), "LEWM -> MUJOCO PREDICTION", fill=PREDICTED_COLOR, font=label_font)

    error_mm = 1000.0 * float(np.linalg.norm(predicted_xyz - actual_xyz))
    joint_rmse = float(np.sqrt(np.mean((predicted_state - actual_state) ** 2)))
    draw.text(
        (630, 666),
        f"EEF error {error_mm:.1f} mm\nJoint RMSE {joint_rmse:.2f}\n"
        + ("Prediction clipped for rendering" if predicted_clipped else "No joint clipping"),
        fill="#20242a",
        font=text_font,
    )
    draw_xy_trace(draw, cycle_actual_xyz, cycle_predicted_xyz, cycle_index)
    draw.text(
        (910, 678),
        "Teal path: recorded joints via FK (offline)\nRed path: predicted joints via FK\nRGB stays model input only",
        fill="#4a5358",
        font=text_font,
    )
    return np.asarray(canvas, dtype=np.uint8)


def build_html_page(metrics: dict[str, Any], media: list[str]) -> str:
    task = html.escape(str(metrics["task"]))
    source = html.escape(str(metrics["source"]))
    videos = []
    for name in media:
        escaped = html.escape(name)
        if name.endswith(".mp4"):
            videos.append(f'<video controls loop playsinline src="{escaped}"></video>')
        else:
            videos.append(f'<p><a href="{escaped}">Open GIF</a></p>')
    horizon_rows = "".join(
        f"<tr><td>H{row['horizon']}</td><td>{row['count']}</td>"
        f"<td>{row['eef_rmse_mm']:.2f}</td><td>{row['joint_rmse']:.3f}</td></tr>"
        for row in metrics["by_horizon"]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SO101 MuJoCo prediction replay</title>
<style>body{{margin:0;background:#eef1f2;color:#20242a;font:15px Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:24px}}section{{background:white;border:1px solid #c7ced2;margin:0 0 18px;padding:18px}}video{{width:100%;height:auto;background:#111}}.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.fact{{border-left:4px solid #087e8b;padding:8px 12px;background:#f5f7f8}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef1f2;padding:2px 4px}}</style></head><body><main>
<section><h1>SO-ARM101 MuJoCo prediction replay</h1><p>{source} | {task} | held-out episode {metrics['episode']}</p>
<p>The left panel is the held-out LeRobot camera frame available at the forecast origin. It remains frozen while the right panel plays H1-H8 predictions in the official SO101 MuJoCo model. No future RGB/state or train-neighbor image is used to create the prediction. Recorded future state is loaded only after prediction for offline error measurement.</p>
<p><b>Display boundary:</b> the public dataset does not provide camera calibration, so the MuJoCo view is a consistent diagnostic view rather than a pixel-aligned reconstruction of the real camera.</p>
<div class="facts"><div class="fact">EEF RMSE<br><b>{metrics['eef_rmse_mm']:.2f} mm</b></div><div class="fact">EEF p95<br><b>{metrics['eef_p95_mm']:.2f} mm</b></div><div class="fact">Joint RMSE<br><b>{metrics['joint_rmse']:.3f}</b></div></div></section>
<section>{''.join(videos)}</section><section><h2>Error by forecast horizon</h2><table><thead><tr><th>Horizon</th><th>Frames</th><th>EEF RMSE (mm)</th><th>Joint RMSE</th></tr></thead><tbody>{horizon_rows}</tbody></table></section>
<section><p><a href="metrics.json">metrics.json</a> | <a href="manifest.json">manifest.json</a></p></section></main></body></html>"""


def render_choice(
    args: argparse.Namespace,
    output: Path,
    choice: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_sha: str,
    frameskip: int,
    history: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    predictions = prediction_cycle_predictions(
        model,
        choice,
        frameskip,
        history,
        args.cycle_horizon,
        action_center,
        action_scale,
        args.batch_size,
        action_center,
        action_scale,
    )
    if predictions.predicted_states is None:
        raise ValueError("checkpoint has no direct six-state prediction head")
    keep = selected_prediction_indices(predictions.display_input_local, args.origin_stride)
    destination_rows = choice.rows[predictions.destination_local]
    actual_states = np.asarray(choice.cache.data.states[destination_rows], dtype=np.float64)
    predicted_states = np.asarray(predictions.predicted_states, dtype=np.float64)
    kinematics = SO101MuJoCoKinematics(args.scene)
    actual_xyz = np.full((len(predictions.predicted_states), 3), np.nan, dtype=np.float64)
    predicted_xyz = np.full_like(actual_xyz, np.nan)
    actual_xyz[keep] = kinematics.forward(actual_states[keep])
    predicted_xyz[keep] = kinematics.forward(predicted_states[keep])
    arm_excess = np.maximum(np.abs(predicted_states[:, :5]) - 100.0, 0.0)
    gripper_excess = np.maximum(
        np.maximum(-predicted_states[:, 5], 0.0),
        np.maximum(predicted_states[:, 5] - 100.0, 0.0),
    )
    max_excess = np.maximum(np.max(arm_excess, axis=1), gripper_excess)
    clipped = max_excess > 0.0

    renderer = SO101ReplayRenderer(
        args.scene,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )
    media_types = ["mp4", "gif"] if args.media == "both" else [args.media]
    writers = [RawFFmpegWriter(output / f"mujoco_prediction.{kind}", args.fps, kind) for kind in media_types]
    try:
        for raw_index in keep:
            index = int(raw_index)
            origin = int(predictions.display_input_local[index])
            left_local = (
                origin
                if args.layout == "frozen-observation"
                else int(predictions.destination_local[index])
            )
            left_row = int(choice.rows[left_local])
            left_image = np.asarray(choice.cache.data.frames[left_row], dtype=np.uint8)
            predicted_image, _, _ = renderer.render(predicted_states[index])
            cycle = np.flatnonzero(predictions.display_input_local == origin)
            cycle_kept = np.intersect1d(cycle, keep, assume_unique=True)
            cycle_position = int(np.flatnonzero(cycle_kept == index)[0])
            horizon = int((predictions.destination_local[index] - origin) // frameskip)
            frame = compose_frame(
                left_image,
                predicted_image,
                source=choice.cache.source,
                task=episode_task_key(choice),
                episode=choice.episode,
                checkpoint_step=int(checkpoint.get("step", -1)),
                horizon=horizon,
                raw_ahead=horizon * frameskip,
                actual_state=actual_states[index],
                predicted_state=predicted_states[index],
                actual_xyz=actual_xyz[index],
                predicted_xyz=predicted_xyz[index],
                cycle_actual_xyz=actual_xyz[cycle_kept],
                cycle_predicted_xyz=predicted_xyz[cycle_kept],
                cycle_index=cycle_position,
                predicted_clipped=bool(clipped[index]),
                frozen_observation=args.layout == "frozen-observation",
            )
            for writer in writers:
                writer.write(frame)
        for writer in writers:
            writer.close()
    except Exception:
        for writer in writers:
            writer.abort()
        raise
    finally:
        renderer.close()

    error = np.linalg.norm(predicted_xyz[keep] - actual_xyz[keep], axis=1)
    joint_error = predicted_states[keep] - actual_states[keep]
    by_horizon = []
    for horizon in range(1, args.cycle_horizon + 1):
        horizon_mask = (
            predictions.destination_local[keep] - predictions.display_input_local[keep]
        ) == horizon * frameskip
        by_horizon.append(
            {
                "horizon": horizon,
                "count": int(np.sum(horizon_mask)),
                "eef_rmse_mm": 1000.0 * float(np.sqrt(np.mean(error[horizon_mask] ** 2))),
                "joint_rmse": float(np.sqrt(np.mean(joint_error[horizon_mask] ** 2))),
            }
        )
    metrics = {
        "source": choice.cache.source,
        "task": episode_task_key(choice),
        "episode": choice.episode,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "prediction_records": len(keep),
        "eef_rmse_mm": 1000.0 * float(np.sqrt(np.mean(error**2))),
        "eef_mae_mm": 1000.0 * float(np.mean(error)),
        "eef_p95_mm": 1000.0 * float(np.quantile(error, 0.95)),
        "joint_rmse": float(np.sqrt(np.mean(joint_error**2))),
        "prediction_clipping_fraction": float(np.mean(clipped[keep])),
        "prediction_boundary_excess_p95": float(np.quantile(max_excess[keep], 0.95)),
        "prediction_boundary_excess_max": float(np.max(max_excess[keep])),
        "by_horizon": by_horizon,
    }
    manifest = {
        "format": "so101-mujoco-prediction-replay-v1",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "step": int(checkpoint.get("step", -1)),
        },
        "scene": str(args.scene.resolve()),
        "episode": {"source": choice.cache.source, "id": choice.episode, "task": episode_task_key(choice)},
        "prediction": {
            "kind": "checkpoint direct six-state residual head",
            "history": history,
            "frameskip": frameskip,
            "cycle_horizon": args.cycle_horizon,
            "origin_stride": args.origin_stride,
            "layout": args.layout,
            "future_information_used": False,
            "recorded_actions_used": True,
        },
        "render": {
            "camera": {"azimuth": args.azimuth, "elevation": args.elevation, "distance": args.distance},
            "left_panel": "held-out LeRobot RGB at forecast origin" if args.layout == "frozen-observation" else "held-out LeRobot RGB at forecast target",
            "right_panel": "official SO101 MuJoCo model driven by checkpoint-predicted six-joint state",
            "pixel_alignment_claimed": False,
            "gripper_mapping": "visualization-only linear [0,100] to official hinge range",
            "media": [writer.output.name for writer in writers],
        },
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "manifest.json", manifest)
    atomic_write_text(output / "index.html", build_html_page(metrics, manifest["render"]["media"]))
    return {**metrics, "path": str(output), "media": manifest["render"]["media"]}


def build_gallery(reports: list[dict[str, Any]]) -> str:
    cards = []
    for report in reports:
        relative = Path(report["path"]).name
        media = next(
            (name for name in report["media"] if name.endswith(".mp4")),
            report["media"][0],
        )
        media_path = html.escape(relative + "/" + media)
        media_markup = (
            f'<video controls loop playsinline src="{media_path}"></video>'
            if media.endswith(".mp4")
            else f'<img src="{media_path}" alt="SO101 MuJoCo prediction replay">'
        )
        cards.append(
            f'<article><h2>{html.escape(report["task"])}</h2>{media_markup}'
            f'<p>EEF RMSE <b>{report["eef_rmse_mm"]:.2f} mm</b> | <a href="{html.escape(relative + "/index.html")}">details</a></p></article>'
        )
    return """<!doctype html><html><head><meta charset="utf-8"><title>SO101 real-observation / MuJoCo-prediction gallery</title><style>body{font:15px Arial;background:#eef1f2;margin:0}main{max-width:1400px;margin:auto;padding:24px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}header{padding:24px 5%;background:#202b30;color:white}article{background:white;border:1px solid #c7ced2;padding:14px}video{width:100%}a{color:#087e8b}@media(max-width:850px){main{grid-template-columns:1fr}}</style></head><body><header><h1>SO-ARM101: real observation vs LeWM prediction</h1><p>Left: frozen held-out RGB. Right: direct six-state prediction rendered with the official robot model.</p></header><main>""" + "".join(cards) + "</main></body></html>"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint, embedded_args, cache_paths, frameskip, history, checkpoint_sha = checkpoint_contract(
        args.checkpoint.resolve(), args.cache_base
    )
    checkpoint_caches = load_caches(cache_paths)
    if args.external_cache:
        caches = load_caches(
            [path.expanduser().resolve() for path in args.external_cache]
        )
        assignments = external_validation_assignments(caches)
    else:
        caches = checkpoint_caches
        assignments = episode_splits(
            [cache.data for cache in checkpoint_caches],
            parse_exclusions(embedded_args.get("exclude_episode", [])),
            split_seed=int(embedded_args["split_seed"]),
            val_fraction=float(embedded_args["val_fraction"]),
        )
    selection_caches = [
        cache for cache in caches if not args.source or cache.source == args.source
    ]
    if not selection_caches:
        raise ValueError(f"unknown --source {args.source!r}")
    choices = (
        choose_one_episode_per_task(selection_caches, frameskip, history, assignments)
        if args.all_tasks
        else [choose_episode(selection_caches, frameskip, history, args.episode, assignments)]
    )
    action_center, action_scale, _ = checkpoint_action_normalizer(checkpoint, embedded_args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise ValueError("MuJoCo replay requires a state-residual LeWM checkpoint")
    reports = []
    for choice in choices:
        report_output = args.output / report_slug(choice) if args.all_tasks else args.output
        report = render_choice(
            args,
            report_output.resolve(),
            choice,
            model,
            checkpoint,
            checkpoint_sha,
            frameskip,
            history,
            action_center,
            action_scale,
        )
        reports.append(report)
        print(f"report: {report_output / 'index.html'}", flush=True)
    if args.all_tasks:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.output / "index.html", build_gallery(reports))
        (args.output / "gallery.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"gallery: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
