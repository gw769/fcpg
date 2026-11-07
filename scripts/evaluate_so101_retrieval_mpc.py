#!/usr/bin/env python3
"""Evaluate demonstration-retrieval MPC without commanding an SO-101 robot.

At each held-out observation the script retrieves phase-compatible action
prefixes from successful training episodes, rolls every prefix through LeWM,
and ranks the predicted H8 outcomes against train-only terminal goal latents.
The held-out future is materialized only for post-selection evaluation.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from fcpg.so101_retrieval_mpc import (
    candidate_start_rows,
    chunk_and_normalize_actions,
    prefix_rmse,
    raw_action_prefixes,
    rebase_action_prefixes,
    retrieve_diverse_candidates,
    select_safe_candidate,
    terminal_goal_cost,
    terminal_goal_rows,
    terminal_state_cost,
    validate_paired_camera_data,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.visualize_so101_lewm import (
    CacheView,
    EpisodeChoice,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    choose_episode,
    encode_rows,
    episode_rows,
    load_caches,
    normalized_action_chunk,
    normalized_state_rows,
    predict_state_residual_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline SO-101 demonstration retrieval and LeWM ranking"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--secondary-cache",
        type=Path,
        help=(
            "A row-aligned second-camera cache. When supplied, retrieval and "
            "H8 goal ranking use both views while sharing every action/state row."
        ),
    )
    parser.add_argument("--episode", help="Held-out SOURCE:EPISODE")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--planning-stride", type=int, default=5)
    parser.add_argument("--candidates-per-episode", type=int, default=2)
    parser.add_argument("--candidate-min-gap", type=int, default=20)
    parser.add_argument("--goal-tail-frames", type=int, default=15)
    parser.add_argument("--goal-final-steps", type=int, default=3)
    parser.add_argument("--state-retrieval-weight", type=float, default=1.0)
    parser.add_argument("--state-goal-weight", type=float, default=1.0)
    parser.add_argument("--secondary-visual-goal-weight", type=float, default=1.0)
    parser.add_argument("--max-first-command-jump", type=float, default=20.0)
    parser.add_argument(
        "--action-mode",
        choices=("raw", "rebase"),
        default="rebase",
        help="Use recorded absolute commands or translate them to the current measured state.",
    )
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _save_frame(path: Path, frame: np.ndarray) -> None:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    image.save(path, format="JPEG", quality=90, optimize=True)


def _training_episodes(
    cache: CacheView,
    assignments: dict[tuple[str, int], str],
) -> set[int]:
    return {
        int(episode)
        for episode in np.unique(np.asarray(cache.data.episodes, dtype=np.int64))
        if assignments.get((cache.dataset_id, int(episode))) == "train"
    }


def _rollout_candidates(
    model: StateResidualLeWM,
    context_latents: np.ndarray,
    context_states: np.ndarray,
    past_action_chunks: np.ndarray,
    candidate_chunks: np.ndarray,
    *,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    candidates, horizon, _ = candidate_chunks.shape
    if context_latents.shape[0] != 3 or context_states.shape != (3, 6):
        raise ValueError("SO-101 retrieval MPC requires three context observations")
    if past_action_chunks.shape[0] != 2:
        raise ValueError("three observations require two prior action chunks")

    latent_history = np.repeat(context_latents[None], candidates, axis=0)
    state_history = np.repeat(context_states[None], candidates, axis=0)
    action_history = np.repeat(past_action_chunks[None], candidates, axis=0)
    predicted_latents: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    for step in range(horizon):
        action_history = np.concatenate(
            (action_history, candidate_chunks[:, step : step + 1]), axis=1
        )
        next_latent, next_state = predict_state_residual_batch(
            model,
            latent_history[:, -3:],
            action_history[:, -3:],
            state_history[:, -3:],
            residual_scale=residual_scale,
        )
        latent_history = np.concatenate((latent_history, next_latent[:, None]), axis=1)
        state_history = np.concatenate((state_history, next_state[:, None]), axis=1)
        predicted_latents.append(next_latent)
        predicted_states.append(next_state)
    return (
        np.stack(predicted_latents, axis=1),
        np.stack(predicted_states, axis=1),
    )


def _display_neighbor(
    predicted_latent: np.ndarray,
    predicted_state: np.ndarray,
    train_latents: np.ndarray,
    train_states: np.ndarray,
    state_scale: np.ndarray,
) -> int:
    latent_distance = np.mean(
        np.square(train_latents - predicted_latent[None]), axis=1
    )
    state_distance = np.mean(
        np.square((train_states - predicted_state[None]) / state_scale[None]),
        axis=1,
    )
    return int(np.argmin(latent_distance + state_distance))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _build_html(report: dict[str, Any]) -> str:
    if report["planning"].get("dual_camera", False):
        return _build_dual_camera_html(report)
    decisions_json = json.dumps(report["decisions"], ensure_ascii=True).replace(
        "</", "<\\/"
    )
    summary = report["summary"]
    source = html.escape(str(report["episode"]["source"]))
    episode = int(report["episode"]["episode"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 LeWM 离线候选动作预览</title>
<style>
:root{{--bg:#f3f5f4;--ink:#18201d;--muted:#64706b;--line:#cbd3cf;--teal:#087e8b;--red:#c84b45;--gold:#a96f00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
header{{padding:18px 24px;border-bottom:1px solid var(--line);background:white}}h1{{margin:0 0 4px;font-size:22px}}
header p{{margin:3px 0;color:var(--muted)}}main{{max-width:1320px;margin:auto;padding:18px 22px 30px}}
.facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:16px}}
.fact{{background:white;padding:12px}}.fact b{{display:block;font-size:19px;color:var(--teal)}}
.controls{{display:flex;align-items:center;gap:12px;background:white;border:1px solid var(--line);padding:10px 12px}}
button{{border:1px solid #87938d;background:#fff;padding:7px 13px;cursor:pointer}}input[type=range]{{flex:1}}
.primary{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
.panel,.mini{{background:white;border:1px solid var(--line);padding:10px}}.panel h2,.mini h3{{font-size:14px;margin:0 0 8px}}
.panel img{{width:100%;aspect-ratio:1;object-fit:contain;background:#202322}}
.secondary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}}
.mini img{{width:100%;aspect-ratio:1;object-fit:contain;background:#202322}}
.metrics{{margin-top:12px;background:white;border:1px solid var(--line);padding:12px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.metric span{{display:block;color:var(--muted);font-size:12px}}.metric b{{font-size:16px}}
.warning{{color:#8b2f2a;font-weight:650}}.good{{color:#056b62}}code{{font-family:ui-monospace,monospace}}
@media(max-width:800px){{.facts,.metrics{{grid-template-columns:1fr 1fr}}.primary,.secondary{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>SO-101 LeWM 离线候选动作预览</h1>
<p>{source}，held-out episode {episode}。全程只读，没有向机械臂发送动作。</p>
<p class="warning">右侧不是生成 RGB，而是与 LeWM 预测 H8 latent/state 最接近的训练集帧；真实 H8 仅用于事后评价。</p></header>
<main><section class="facts">
<div class="fact">LeWM 首段动作 RMSE<b>{summary['selected_first_rmse']:.3f}</b></div>
<div class="fact">最近邻首段 RMSE<b>{summary['nearest_first_rmse']:.3f}</b></div>
<div class="fact">LeWM 优于最近邻比例<b>{100*summary['selected_better_fraction']:.1f}%</b></div>
<div class="fact">LeWM 改选候选比例<b>{100*summary['selection_changed_fraction']:.1f}%</b></div>
</section>
<section class="controls"><button id="play">播放</button><input id="slider" type="range" min="0" max="{len(report['decisions'])-1}" value="0"><b id="counter"></b></section>
<section class="primary">
<article class="panel"><h2>左：当前真实 wrist 观测</h2><img id="current"></article>
<article class="panel"><h2>右：LeWM H8 预测对应的训练帧</h2><img id="predicted"></article>
</section>
<section class="secondary">
<article class="mini"><h3>真实 H8（只作评价）</h3><img id="actualFuture"></article>
<article class="mini"><h3>被选示教：候选起点</h3><img id="candidateStart"></article>
<article class="mini"><h3>被选示教：40 帧后</h3><img id="candidateEnd"></article>
</section>
<section class="metrics">
<div class="metric"><span>held-out raw frame</span><b id="frame"></b></div>
<div class="metric"><span>selected / candidates</span><b id="choice"></b></div>
<div class="metric"><span>LeWM / nearest 首段 RMSE</span><b id="actionRmse"></b></div>
<div class="metric"><span>LeWM H8 latent 真值 MSE</span><b id="latentMse"></b></div>
<div class="metric"><span>selected source episode/frame</span><b id="sourceChoice"></b></div>
<div class="metric"><span>goal cost / nearest goal cost</span><b id="goalCost"></b></div>
<div class="metric"><span>首命令最大关节跳变</span><b id="jump"></b></div>
<div class="metric"><span>oracle candidate index</span><b id="oracle"></b></div>
</section></main>
<script>const rows={decisions_json};
const ids=['current','predicted','actualFuture','candidateStart','candidateEnd'];
const slider=document.getElementById('slider'),play=document.getElementById('play');let timer=null;
function draw(i){{const r=rows[i];ids.forEach(k=>document.getElementById(k).src=r.images[k]);
document.getElementById('counter').textContent=`${{i+1}} / ${{rows.length}}`;
document.getElementById('frame').textContent=r.frame;document.getElementById('choice').textContent=`${{r.selected_index}} / ${{r.candidate_count}}`;
document.getElementById('actionRmse').textContent=`${{r.selected_first_rmse.toFixed(3)}} / ${{r.nearest_first_rmse.toFixed(3)}}`;
document.getElementById('latentMse').textContent=r.selected_target_latent_mse.toFixed(5);
document.getElementById('sourceChoice').textContent=`${{r.selected_source_episode}} / ${{r.selected_source_frame}}`;
document.getElementById('goalCost').textContent=`${{r.selected_goal_cost.toFixed(5)}} / ${{r.nearest_goal_cost.toFixed(5)}}`;
document.getElementById('jump').textContent=r.selected_first_command_max_jump.toFixed(2)+' deg';
document.getElementById('oracle').textContent=r.oracle_index;slider.value=i;}}
slider.oninput=()=>draw(Number(slider.value));play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='播放';return;}}
play.textContent='暂停';timer=setInterval(()=>{{let i=(Number(slider.value)+1)%rows.length;draw(i)}},600);}};draw(0);
</script></body></html>"""


def _build_dual_camera_html(report: dict[str, Any]) -> str:
    decisions_json = json.dumps(report["decisions"], ensure_ascii=True).replace(
        "</", "<\\/"
    )
    summary = report["summary"]
    source = html.escape(str(report["episode"]["source"]))
    episode = int(report["episode"]["episode"])
    primary = html.escape(str(report["views"]["primary_camera"]))
    secondary = html.escape(str(report["views"]["secondary_camera"]))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 双相机 LeWM 离线候选动作预览</title>
<style>
:root{{--bg:#f3f5f4;--ink:#18201d;--muted:#64706b;--line:#cbd3cf;--teal:#087e8b;--red:#a23934}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
header{{padding:17px 22px;border-bottom:1px solid var(--line);background:#fff}}h1{{margin:0 0 4px;font-size:22px}}header p{{margin:3px 0;color:var(--muted)}}
main{{max-width:1460px;margin:auto;padding:16px 20px 28px}}.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}}
.fact{{background:#fff;padding:11px}}.fact b{{display:block;font-size:19px;color:var(--teal)}}.controls{{display:flex;gap:12px;align-items:center;background:#fff;border:1px solid var(--line);padding:10px 12px;margin-top:12px}}
button{{border:1px solid #87938d;background:#fff;padding:7px 13px;cursor:pointer}}input[type=range]{{flex:1}}.view{{margin-top:12px}}.view h2{{font-size:16px;margin:0;padding:9px 11px;background:#e7ece9;border:1px solid var(--line);border-bottom:0}}
.triplet{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}}.panel{{background:#fff;padding:9px}}.panel h3{{font-size:13px;margin:0 0 7px}}.panel img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#202322}}
.candidate{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-top:12px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;background:#fff;border:1px solid var(--line);padding:11px;margin-top:12px}}
.metric span{{display:block;color:var(--muted);font-size:12px}}.metric b{{font-size:16px}}.warning{{color:var(--red);font-weight:650}}
@media(max-width:850px){{.facts,.metrics,.candidate{{grid-template-columns:1fr 1fr}}.triplet{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>SO-101 双相机 LeWM 离线候选动作预览</h1>
<p>{source}，held-out episode {episode}；主视图 {primary}，辅助视图 {secondary}。</p>
<p class="warning">全程只读，没有向机械臂发送动作。两路“预测图”均是 LeWM H8 latent 对应的训练集近邻，不是生成 RGB。</p></header>
<main><section class="facts">
<div class="fact">LeWM 首段动作 RMSE<b>{summary['selected_first_rmse']:.3f}</b></div>
<div class="fact">最近邻首段 RMSE<b>{summary['nearest_first_rmse']:.3f}</b></div>
<div class="fact">LeWM 优于最近邻比例<b>{100*summary['selected_better_fraction']:.1f}%</b></div>
<div class="fact">改选候选比例<b>{100*summary['selection_changed_fraction']:.1f}%</b></div>
</section><section class="controls"><button id="play">播放</button><input id="slider" type="range" min="0" max="{len(report['decisions'])-1}" value="0"><b id="counter"></b></section>
<section class="view"><h2>Wrist：近场夹爪与物体</h2><div class="triplet">
<article class="panel"><h3>当前真实</h3><img id="primaryCurrent"></article><article class="panel"><h3>LeWM H8 预测近邻</h3><img id="primaryPredicted"></article><article class="panel"><h3>真实 H8（只作评价）</h3><img id="primaryActualFuture"></article>
</div></section>
<section class="view"><h2>Column：全局目标与机械臂上下文</h2><div class="triplet">
<article class="panel"><h3>当前真实</h3><img id="secondaryCurrent"></article><article class="panel"><h3>LeWM H8 预测近邻</h3><img id="secondaryPredicted"></article><article class="panel"><h3>真实 H8（只作评价）</h3><img id="secondaryActualFuture"></article>
</div></section>
<section class="candidate">
<article class="panel"><h3>Wrist 候选起点</h3><img id="primaryCandidateStart"></article><article class="panel"><h3>Wrist 候选终点</h3><img id="primaryCandidateEnd"></article><article class="panel"><h3>Column 候选起点</h3><img id="secondaryCandidateStart"></article><article class="panel"><h3>Column 候选终点</h3><img id="secondaryCandidateEnd"></article>
</section><section class="metrics">
<div class="metric"><span>held-out raw frame</span><b id="frame"></b></div><div class="metric"><span>selected / candidates</span><b id="choice"></b></div><div class="metric"><span>LeWM / nearest 首段 RMSE</span><b id="actionRmse"></b></div><div class="metric"><span>双视角 H8 latent 真值 MSE</span><b id="latentMse"></b></div>
<div class="metric"><span>selected source episode/frame</span><b id="sourceChoice"></b></div><div class="metric"><span>goal cost / nearest</span><b id="goalCost"></b></div><div class="metric"><span>首命令最大关节跳变</span><b id="jump"></b></div><div class="metric"><span>oracle candidate index</span><b id="oracle"></b></div>
</section></main><script>const rows={decisions_json};
const ids=['primaryCurrent','primaryPredicted','primaryActualFuture','secondaryCurrent','secondaryPredicted','secondaryActualFuture','primaryCandidateStart','primaryCandidateEnd','secondaryCandidateStart','secondaryCandidateEnd'];
const slider=document.getElementById('slider'),play=document.getElementById('play');let timer=null;
function draw(i){{const r=rows[i];ids.forEach(k=>document.getElementById(k).src=r.images[k]);document.getElementById('counter').textContent=`${{i+1}} / ${{rows.length}}`;document.getElementById('frame').textContent=r.frame;document.getElementById('choice').textContent=`${{r.selected_index}} / ${{r.candidate_count}}`;document.getElementById('actionRmse').textContent=`${{r.selected_first_rmse.toFixed(3)}} / ${{r.nearest_first_rmse.toFixed(3)}}`;document.getElementById('latentMse').textContent=r.selected_target_latent_mse.toFixed(5);document.getElementById('sourceChoice').textContent=`${{r.selected_source_episode}} / ${{r.selected_source_frame}}`;document.getElementById('goalCost').textContent=`${{r.selected_goal_cost.toFixed(5)}} / ${{r.nearest_goal_cost.toFixed(5)}}`;document.getElementById('jump').textContent=r.selected_first_command_max_jump.toFixed(2)+' deg';document.getElementById('oracle').textContent=r.oracle_index;slider.value=i;}}
slider.oninput=()=>draw(Number(slider.value));play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='播放';return;}}play.textContent='暂停';timer=setInterval(()=>draw((Number(slider.value)+1)%rows.length),700);}};draw(0);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda:1"):
        raise ValueError("CUDA1 is forbidden for this project")
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("use CPU or CUDA0")
    if args.horizon <= 0 or args.render_stride <= 0:
        raise ValueError("horizon and render stride must be positive")
    if args.secondary_visual_goal_weight < 0:
        raise ValueError("secondary visual goal weight must be non-negative")

    torch.set_num_threads(args.threads)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    (
        checkpoint,
        embedded_args,
        cache_paths,
        frameskip,
        history,
        checkpoint_sha256,
    ) = checkpoint_contract(checkpoint_path, None)
    if history != 3:
        raise ValueError("retrieval MPC currently requires LeWM history=3")
    caches = load_caches(cache_paths)
    if len(caches) != 1:
        raise ValueError("retrieval MPC requires exactly one primary checkpoint cache")
    cache = caches[0]
    secondary_cache: CacheView | None = None
    paired_contract = None
    if args.secondary_cache is not None:
        secondary_cache = load_caches(
            [args.secondary_cache.expanduser().resolve()]
        )[0]
        paired_contract = validate_paired_camera_data(
            cache.data,
            secondary_cache.data,
        )
    exclusions = parse_exclusions(embedded_args.get("exclude_episode", []))
    assignments = episode_splits(
        [cache.data],
        exclusions,
        split_seed=int(embedded_args["split_seed"]),
        val_fraction=float(embedded_args["val_fraction"]),
    )
    choice = choose_episode(
        caches, frameskip, history, args.episode, assignments
    )
    train_episodes = _training_episodes(cache, assignments)
    horizon_raw = frameskip * args.horizon
    bank_rows = candidate_start_rows(
        cache.data.episodes,
        cache.data.frame_indices,
        train_episodes,
        horizon_raw=horizon_raw,
        stride=frameskip,
    )
    goal_rows = terminal_goal_rows(
        cache.data.episodes,
        train_episodes,
        tail_frames=args.goal_tail_frames,
    )
    train_rows = np.concatenate(
        [episode_rows(cache, episode) for episode in sorted(train_episodes)]
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("retrieval MPC requires an SO-101 state-residual checkpoint")
    action_center, action_scale, action_normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )
    state_center = action_center
    state_scale = action_scale

    print("encoding train-only candidate and goal banks", flush=True)
    unique_train_rows = np.unique(np.concatenate((bank_rows, goal_rows, train_rows)))
    unique_train_latents = encode_rows(model, cache, unique_train_rows, args.batch_size)
    latent_by_row = {
        int(row): latent for row, latent in zip(unique_train_rows, unique_train_latents)
    }
    bank_latents = np.stack([latent_by_row[int(row)] for row in bank_rows])
    goal_latents = np.stack([latent_by_row[int(row)] for row in goal_rows])
    secondary_latent_by_row: dict[int, np.ndarray] | None = None
    secondary_bank_latents: np.ndarray | None = None
    secondary_goal_latents: np.ndarray | None = None
    secondary_train_latents: np.ndarray | None = None
    if secondary_cache is not None:
        secondary_unique_latents = encode_rows(
            model,
            secondary_cache,
            unique_train_rows,
            args.batch_size,
        )
        secondary_latent_by_row = {
            int(row): latent
            for row, latent in zip(unique_train_rows, secondary_unique_latents)
        }
        secondary_bank_latents = np.stack(
            [secondary_latent_by_row[int(row)] for row in bank_rows]
        )
        secondary_goal_latents = np.stack(
            [secondary_latent_by_row[int(row)] for row in goal_rows]
        )
        secondary_train_latents = np.stack(
            [secondary_latent_by_row[int(row)] for row in train_rows]
        )
    goal_states = normalized_state_rows(cache, goal_rows, state_center, state_scale)
    train_latents = np.stack([latent_by_row[int(row)] for row in train_rows])
    bank_states = np.asarray(cache.data.states[bank_rows], dtype=np.float32)
    bank_episodes = np.asarray(cache.data.episodes[bank_rows], dtype=np.int64)
    train_states = np.asarray(cache.data.states[train_rows], dtype=np.float32)

    first_origin = frameskip * (history - 1)
    final_origin = len(choice.rows) - horizon_raw - 1
    origins = np.arange(
        first_origin,
        final_origin + 1,
        args.planning_stride,
        dtype=np.int64,
    )
    if len(origins) == 0:
        raise ValueError("held-out episode is too short for retrieval MPC")
    context_locals = np.unique(
        np.concatenate([origin - np.arange(history - 1, -1, -1) * frameskip for origin in origins])
    )
    context_rows = choice.rows[context_locals]
    print("encoding held-out current/past observations", flush=True)
    context_encoded = encode_rows(model, cache, context_rows, args.batch_size)
    heldout_latent_by_local = {
        int(local): latent for local, latent in zip(context_locals, context_encoded)
    }
    secondary_heldout_latent_by_local: dict[int, np.ndarray] | None = None
    if secondary_cache is not None:
        secondary_context_encoded = encode_rows(
            model,
            secondary_cache,
            context_rows,
            args.batch_size,
        )
        secondary_heldout_latent_by_local = {
            int(local): latent
            for local, latent in zip(context_locals, secondary_context_encoded)
        }

    raw_decisions: list[dict[str, Any]] = []
    predicted_for_display: list[np.ndarray] = []
    secondary_predicted_for_display: list[np.ndarray] = []
    predicted_state_for_display: list[np.ndarray] = []
    for decision_index, raw_origin in enumerate(origins):
        origin = int(raw_origin)
        locals_ = origin - np.arange(history - 1, -1, -1) * frameskip
        context_latents = np.stack(
            [heldout_latent_by_local[int(local)] for local in locals_]
        )
        secondary_context_latents = None
        if secondary_heldout_latent_by_local is not None:
            secondary_context_latents = np.stack(
                [secondary_heldout_latent_by_local[int(local)] for local in locals_]
            )
        global_context_rows = choice.rows[locals_]
        context_states = normalized_state_rows(
            cache, global_context_rows, state_center, state_scale
        )
        past_actions = np.stack(
            [
                normalized_action_chunk(
                    cache,
                    int(choice.rows[int(local)]),
                    frameskip,
                    action_center,
                    action_scale,
                )
                for local in locals_[:-1]
            ]
        )
        retrieval = retrieve_diverse_candidates(
            (
                context_latents[-1]
                if secondary_context_latents is None
                else np.concatenate(
                    (context_latents[-1], secondary_context_latents[-1])
                )
            ),
            np.asarray(cache.data.states[choice.rows[origin]], dtype=np.float32),
            (
                bank_latents
                if secondary_bank_latents is None
                else np.concatenate((bank_latents, secondary_bank_latents), axis=1)
            ),
            bank_states,
            bank_rows,
            bank_episodes,
            state_scale=state_scale,
            per_episode=args.candidates_per_episode,
            minimum_row_gap=args.candidate_min_gap,
            state_weight=args.state_retrieval_weight,
        )
        raw_prefix = raw_action_prefixes(
            cache.data.actions,
            retrieval.rows,
            horizon_raw=horizon_raw,
        )
        current_state = np.asarray(
            cache.data.states[choice.rows[origin]], dtype=np.float32
        )
        if args.action_mode == "rebase":
            raw_prefix = rebase_action_prefixes(
                raw_prefix,
                np.asarray(cache.data.states[retrieval.rows], dtype=np.float32),
                current_state,
            )
        chunks = chunk_and_normalize_actions(
            raw_prefix,
            frameskip=frameskip,
            center=action_center,
            scale=action_scale,
        )
        predicted_latent, predicted_state = _rollout_candidates(
            model,
            context_latents,
            context_states,
            past_actions,
            chunks,
        )
        primary_visual_costs = terminal_goal_cost(
            predicted_latent,
            goal_latents,
            final_steps=args.goal_final_steps,
        )
        secondary_predicted_latent = None
        secondary_predicted_state = None
        secondary_visual_costs = np.zeros_like(primary_visual_costs)
        if secondary_context_latents is not None:
            secondary_predicted_latent, secondary_predicted_state = (
                _rollout_candidates(
                    model,
                    secondary_context_latents,
                    context_states,
                    past_actions,
                    chunks,
                )
            )
            if secondary_goal_latents is None:
                raise AssertionError("dual-camera goal bank is missing")
            secondary_visual_costs = terminal_goal_cost(
                secondary_predicted_latent,
                secondary_goal_latents,
                final_steps=args.goal_final_steps,
            )
        visual_costs = primary_visual_costs + (
            args.secondary_visual_goal_weight * secondary_visual_costs
        )
        state_costs = terminal_state_cost(
            predicted_state,
            goal_states,
            final_steps=args.goal_final_steps,
        )
        costs = visual_costs + args.state_goal_weight * state_costs
        actual_prefix = np.asarray(
            cache.data.actions[
                choice.rows[origin] : choice.rows[origin] + horizon_raw
            ],
            dtype=np.float32,
        )
        first_rmse = np.asarray(
            [prefix_rmse(candidate[:frameskip], actual_prefix[:frameskip]) for candidate in raw_prefix]
        )
        full_rmse = np.asarray(
            [prefix_rmse(candidate, actual_prefix) for candidate in raw_prefix]
        )
        oracle = int(np.argmin(first_rmse))
        selected, safe_mask, first_jumps = select_safe_candidate(
            costs,
            raw_prefix[:, 0],
            current_state,
            max_joint_jump=args.max_first_command_jump,
        )
        selected_row = int(retrieval.rows[selected])
        raw_decisions.append(
            {
                "decision_index": decision_index,
                "origin": origin,
                "frame": int(cache.data.frame_indices[choice.rows[origin]]),
                "target_local": origin + horizon_raw,
                "candidate_count": len(retrieval.rows),
                "candidate_rows": retrieval.rows,
                "candidate_episodes": retrieval.episodes,
                "candidate_goal_cost": costs,
                "candidate_visual_goal_cost": visual_costs,
                "candidate_primary_visual_goal_cost": primary_visual_costs,
                "candidate_secondary_visual_goal_cost": secondary_visual_costs,
                "candidate_state_goal_cost": state_costs,
                "candidate_first_command_jump": first_jumps,
                "candidate_safe": safe_mask,
                "candidate_first_rmse": first_rmse,
                "candidate_full_rmse": full_rmse,
                "selected_index": selected,
                "oracle_index": oracle,
                "selected_row": selected_row,
                "selected_source_episode": int(cache.data.episodes[selected_row]),
                "selected_source_frame": int(cache.data.frame_indices[selected_row]),
                "selected_first_rmse": float(first_rmse[selected]),
                "nearest_first_rmse": float(first_rmse[0]),
                "oracle_first_rmse": float(first_rmse[oracle]),
                "selected_full_rmse": float(full_rmse[selected]),
                "nearest_full_rmse": float(full_rmse[0]),
                "selected_goal_cost": float(costs[selected]),
                "nearest_goal_cost": float(costs[0]),
                "selected_first_command_max_jump": float(first_jumps[selected]),
                "safe_candidate_available": bool(np.any(safe_mask)),
                "selected_prediction": predicted_latent[selected],
                "nearest_prediction": predicted_latent[0],
                "selected_secondary_prediction": (
                    None
                    if secondary_predicted_latent is None
                    else secondary_predicted_latent[selected]
                ),
                "nearest_secondary_prediction": (
                    None
                    if secondary_predicted_latent is None
                    else secondary_predicted_latent[0]
                ),
                "selected_prediction_state": predicted_state[selected, -1],
            }
        )
        predicted_for_display.append(predicted_latent[selected, -1])
        if secondary_predicted_latent is not None:
            secondary_predicted_for_display.append(
                secondary_predicted_latent[selected, -1]
            )
        predicted_state_for_display.append(predicted_state[selected, -1])
        if (decision_index + 1) % 20 == 0 or decision_index + 1 == len(origins):
            print(f"ranked {decision_index + 1}/{len(origins)} decisions", flush=True)

    # Held-out future enters only after all candidate retrieval and ranking is complete.
    target_rows = choice.rows[origins + horizon_raw]
    print("encoding held-out futures for post-selection evaluation", flush=True)
    target_latents = encode_rows(model, cache, target_rows, args.batch_size)
    secondary_target_latents = None
    if secondary_cache is not None:
        secondary_target_latents = encode_rows(
            model,
            secondary_cache,
            target_rows,
            args.batch_size,
        )
    for decision_index, (row, target_latent, predicted_latent, predicted_state) in enumerate(zip(
        raw_decisions,
        target_latents,
        predicted_for_display,
        predicted_state_for_display,
    )):
        primary_target_mse = float(
            np.mean(np.square(np.asarray(row["selected_prediction"]) - target_latent))
        )
        primary_nearest_mse = float(
            np.mean(np.square(np.asarray(row["nearest_prediction"]) - target_latent))
        )
        row["selected_primary_target_latent_mse"] = primary_target_mse
        row["nearest_primary_target_latent_mse"] = primary_nearest_mse
        if secondary_target_latents is not None:
            secondary_target = secondary_target_latents[decision_index]
            secondary_prediction = np.asarray(
                row["selected_secondary_prediction"], dtype=np.float32
            )
            secondary_nearest = np.asarray(
                row["nearest_secondary_prediction"], dtype=np.float32
            )
            secondary_target_mse = float(
                np.mean(np.square(secondary_prediction - secondary_target))
            )
            secondary_nearest_mse = float(
                np.mean(np.square(secondary_nearest - secondary_target))
            )
            row["selected_secondary_target_latent_mse"] = secondary_target_mse
            row["nearest_secondary_target_latent_mse"] = secondary_nearest_mse
            row["selected_target_latent_mse"] = 0.5 * (
                primary_target_mse + secondary_target_mse
            )
            row["nearest_target_latent_mse"] = 0.5 * (
                primary_nearest_mse + secondary_nearest_mse
            )
        else:
            row["selected_target_latent_mse"] = primary_target_mse
            row["nearest_target_latent_mse"] = primary_nearest_mse
        display_index = _display_neighbor(
            predicted_latent,
            predicted_state * state_scale + state_center,
            train_latents,
            train_states,
            state_scale,
        )
        row["display_row"] = int(train_rows[display_index])
    secondary_display_rows: list[int] = []
    if secondary_cache is not None:
        if secondary_train_latents is None:
            raise AssertionError("dual-camera train latent bank is missing")
        for row, predicted_latent, predicted_state in zip(
            raw_decisions,
            secondary_predicted_for_display,
            predicted_state_for_display,
        ):
            display_index = _display_neighbor(
                predicted_latent,
                predicted_state * state_scale + state_center,
                secondary_train_latents,
                train_states,
                state_scale,
            )
            row["secondary_display_row"] = int(train_rows[display_index])
            secondary_display_rows.append(int(train_rows[display_index]))

    rendered = raw_decisions[:: args.render_stride]
    if rendered[-1] is not raw_decisions[-1]:
        rendered.append(raw_decisions[-1])
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "frames"
    assets.mkdir(exist_ok=True)
    public_rows: list[dict[str, Any]] = []
    for render_index, row in enumerate(rendered):
        origin = int(row["origin"])
        selected_row = int(row["selected_row"])
        if secondary_cache is None:
            image_rows = {
                "current": int(choice.rows[origin]),
                "predicted": int(row["display_row"]),
                "actualFuture": int(choice.rows[int(row["target_local"])]),
                "candidateStart": selected_row,
                "candidateEnd": selected_row + horizon_raw,
            }
        else:
            image_rows = {
                "primaryCurrent": int(choice.rows[origin]),
                "primaryPredicted": int(row["display_row"]),
                "primaryActualFuture": int(choice.rows[int(row["target_local"])]),
                "primaryCandidateStart": selected_row,
                "primaryCandidateEnd": selected_row + horizon_raw,
            }
        images: dict[str, str] = {}
        if secondary_cache is not None:
            image_rows.update(
                {
                    "secondaryCurrent": int(choice.rows[origin]),
                    "secondaryPredicted": int(row["secondary_display_row"]),
                    "secondaryActualFuture": int(choice.rows[int(row["target_local"])]),
                    "secondaryCandidateStart": selected_row,
                    "secondaryCandidateEnd": selected_row + horizon_raw,
                }
            )
        for name, cache_row in image_rows.items():
            filename = f"{render_index:04d}_{name}.jpg"
            source_cache = (
                secondary_cache
                if name.startswith("secondary")
                else cache
            )
            if source_cache is None:
                raise AssertionError("secondary image requested without secondary cache")
            _save_frame(assets / filename, source_cache.data.frames[cache_row])
            images[name] = f"frames/{filename}"
        public = {
            key: value
            for key, value in row.items()
            if key not in {
                "selected_prediction",
                "nearest_prediction",
                "selected_secondary_prediction",
                "nearest_secondary_prediction",
                "selected_prediction_state",
            }
        }
        public["images"] = images
        public_rows.append(_jsonable(public))

    summary = {
        "decisions": len(raw_decisions),
        "rendered_decisions": len(public_rows),
        "selected_first_rmse": _mean(raw_decisions, "selected_first_rmse"),
        "nearest_first_rmse": _mean(raw_decisions, "nearest_first_rmse"),
        "oracle_first_rmse": _mean(raw_decisions, "oracle_first_rmse"),
        "selected_full_rmse": _mean(raw_decisions, "selected_full_rmse"),
        "nearest_full_rmse": _mean(raw_decisions, "nearest_full_rmse"),
        "selected_target_latent_mse": _mean(
            raw_decisions, "selected_target_latent_mse"
        ),
        "nearest_target_latent_mse": _mean(
            raw_decisions, "nearest_target_latent_mse"
        ),
        "selected_better_fraction": float(
            np.mean(
                [
                    row["selected_first_rmse"] < row["nearest_first_rmse"]
                    for row in raw_decisions
                ]
            )
        ),
        "selection_changed_fraction": float(
            np.mean([row["selected_index"] != 0 for row in raw_decisions])
        ),
        "oracle_selection_accuracy": float(
            np.mean(
                [row["selected_index"] == row["oracle_index"] for row in raw_decisions]
            )
        ),
    }
    if secondary_cache is not None:
        summary.update(
            {
                "selected_primary_target_latent_mse": _mean(
                    raw_decisions, "selected_primary_target_latent_mse"
                ),
                "nearest_primary_target_latent_mse": _mean(
                    raw_decisions, "nearest_primary_target_latent_mse"
                ),
                "selected_secondary_target_latent_mse": _mean(
                    raw_decisions, "selected_secondary_target_latent_mse"
                ),
                "nearest_secondary_target_latent_mse": _mean(
                    raw_decisions, "nearest_secondary_target_latent_mse"
                ),
            }
        )
    report = {
        "format": (
            "so101-dual-camera-retrieval-mpc-offline-v1"
            if secondary_cache is not None
            else "so101-retrieval-mpc-offline-v1"
        ),
        "scope": "offline held-out demonstration evaluation; no robot commands",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "step": int(checkpoint.get("step", -1)),
        },
        "episode": {
            "source": choice.cache.source,
            "dataset_id": choice.cache.dataset_id,
            "episode": choice.episode,
            "frames": len(choice.rows),
        },
        "views": {
            "primary_cache": str(cache.root),
            "primary_camera": cache.data.camera,
            "secondary_cache": (
                None if secondary_cache is None else str(secondary_cache.root)
            ),
            "secondary_camera": (
                None if secondary_cache is None else secondary_cache.data.camera
            ),
            "paired_frames": (
                None if paired_contract is None else paired_contract.frames
            ),
            "physical_rows_identical": secondary_cache is not None,
        },
        "split": {
            "train_episodes": sorted(train_episodes),
            "heldout_episode": choice.episode,
            "split_seed": int(embedded_args["split_seed"]),
        },
        "planning": {
            "frameskip": frameskip,
            "history": history,
            "horizon": args.horizon,
            "horizon_raw": horizon_raw,
            "planning_stride": args.planning_stride,
            "candidates_per_episode": args.candidates_per_episode,
            "candidate_min_gap": args.candidate_min_gap,
            "goal_tail_frames": args.goal_tail_frames,
            "goal_final_steps": args.goal_final_steps,
            "state_goal_weight": args.state_goal_weight,
            "secondary_visual_goal_weight": args.secondary_visual_goal_weight,
            "dual_camera": secondary_cache is not None,
            "max_first_command_jump": args.max_first_command_jump,
            "action_normalization": action_normalization,
            "action_mode": args.action_mode,
            "future_heldout_input_used_for_ranking": False,
        },
        "summary": summary,
        "decisions": public_rows,
    }
    _write_json(output / "report.json", report)
    (output / "index.html").write_text(_build_html(report))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"html: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
