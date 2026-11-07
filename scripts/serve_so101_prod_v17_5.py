#!/usr/bin/env python3
"""Direct SO-101 service for the complete V17.5 FCPG/LeWM/Actor stack."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for source in (REPO, Path("/home/james/lerobot/src")):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fcpg.prod_v17_5_live import STEP_CAP_DEG, V175LivePolicy
from fcpg.so101_realtime_runtime import EndpointEnvelope, LiveObservation
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
import scripts.serve_so101_prod_v18_1 as live_base


DEFAULT_ROOT = Path(
    "/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/"
    "prod_v17_5_structured_kinematic"
)
DEFAULT_REAL_CALIBRATION = (
    REPO
    / "outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration"
    / "calibration_letterbox_v1.json"
)
DEFAULT_DINO_REPOSITORY = Path(
    "/home/james/.cache/torch/hub/facebookresearch_dinov2_main"
)
DEFAULT_DINO_WEIGHTS = Path(
    "/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
)
DEFAULT_ROBOT_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115543-if00"
)
DEFAULT_WRIST_DEVICE = (
    "/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0"
)
DEFAULT_COLUMN_DEVICE = (
    "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
)
DEFAULT_CONTROL_HZ = 10.0
DEFAULT_JOINT_LIMIT_GUARD = 3.0
DEFAULT_GRIPPER_CLOSED_POSITION = 1.3140604496002197
DEFAULT_GRIPPER_OPEN_POSITION = 31.603153228759766
DEFAULT_HOME_GRIPPER_POSITION = 13.600525856018066

V175_DEPLOYMENT_POSITION_LOWER = SO101_POSITION_LOWER.copy()
V175_DEPLOYMENT_POSITION_UPPER = SO101_POSITION_UPPER.copy()
V175_DEPLOYMENT_POSITION_LOWER[:5] += DEFAULT_JOINT_LIMIT_GUARD
V175_DEPLOYMENT_POSITION_UPPER[:5] -= DEFAULT_JOINT_LIMIT_GUARD
V175_DEPLOYMENT_POSITION_LOWER[5] = DEFAULT_GRIPPER_CLOSED_POSITION
V175_DEPLOYMENT_POSITION_UPPER[5] = DEFAULT_GRIPPER_OPEN_POSITION


class V175HardwareService(live_base.V181HardwareService):
    """Hardware transport with V17.5 planning and ordinary round counters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._round_counter = 0

    def _plan_from(
        self, observation: LiveObservation, *, replace_pending: bool
    ) -> dict[str, Any]:
        proposal = self.policy.propose(
            observation.frames,
            observation.state,
            replace_pending=replace_pending,
        )
        commands = np.asarray(proposal["commands_deg"], dtype=np.float32)
        audit = live_base._audit_chunk(commands, observation.state)
        self._round_counter += 1
        plan = {
            "round_id": f"v175-{self._round_counter:06d}",
            "created_monotonic_s": live_base.time.monotonic(),
            "source_timestamp_s": observation.timestamp_s,
            "source_state_deg": observation.state.copy(),
            "commands_deg": commands,
            "safe": bool(audit["safe"]),
            "reasons": audit["reasons"],
            "maximum_step_per_joint_deg": audit.get(
                "maximum_step_per_joint_deg"
            ),
            "inference_s": proposal["inference_s"],
            "memory_chunks": proposal["memory_chunks"],
        }
        with self.state_lock:
            self.last_observation = observation
            self.plan = plan
            self.last_error = None
        return plan

    def _fresh_plan_for_transport(self) -> tuple[LiveObservation, dict[str, Any]]:
        # Execute the exact first round rendered in the UI. The inherited
        # implementation replans after the operator presses Execute, making
        # the displayed and transported commands differ. Automatic follow-up
        # chunks have no pending preview and continue to plan from their fresh
        # endpoint below.
        if self.policy.runtime.has_pending_chunk:
            with self.state_lock:
                plan = None if self.plan is None else dict(self.plan)
            if plan is None:
                raise RuntimeError("V17.5 has a pending policy chunk without a plan")
            live = self.io.observe()
            source = np.asarray(plan["source_state_deg"], dtype=np.float32)
            state_drift = float(np.max(np.abs(live.state - source)))
            tolerance = float(self.io.settle_step_tolerance)
            if state_drift > tolerance + 1.0e-4:
                self._plan_from(live, replace_pending=True)
                raise RuntimeError(
                    "displayed V17.5 plan became stale because robot state moved; "
                    "a fresh plan is now displayed and was not executed "
                    f"(drift={state_drift:.3f}, tolerance={tolerance:.3f})"
                )
            audit = live_base._audit_chunk(
                np.asarray(plan["commands_deg"], dtype=np.float32), live.state
            )
            reasons = set(audit["reasons"])
            if "v18_1_per_joint_step_cap" in reasons:
                commands = np.asarray(plan["commands_deg"], dtype=np.float32)
                first_step = np.abs(commands[0] - live.state)
                later_steps = np.abs(np.diff(commands, axis=0))
                encoder_drift_only = bool(
                    np.all(first_step <= STEP_CAP_DEG + tolerance + 1.0e-4)
                    and np.all(later_steps <= STEP_CAP_DEG + 1.0e-4)
                )
                if encoder_drift_only:
                    reasons.remove("v18_1_per_joint_step_cap")
            if reasons:
                raise RuntimeError(
                    "displayed V17.5 command rejected against current state: "
                    + ", ".join(sorted(reasons))
                )
            return live, plan

        live = self.io.observe()
        plan = self._plan_from(live, replace_pending=True)
        if not bool(plan["safe"]):
            raise RuntimeError(
                "V17.5 command rejected: " + ", ".join(plan["reasons"])
            )
        return live, plan

    def next_chunk_plan_image(self, view: str) -> np.ndarray:
        image = super().next_chunk_plan_image(view)
        proposal = self.policy.last_proposal or {}
        positions = proposal.get("fm_positions_px")
        if positions is None:
            return image
        points = np.asarray(positions, dtype=np.float32)
        if points.shape != (2, 2, 2) or not np.all(np.isfinite(points)):
            return image
        view_index = 0 if view == "wrist" else 1
        scale = live_base.PLAN_IMAGE_SIZE[1] / 224.0
        colors = ((68, 211, 255), (255, 139, 72))
        labels = ("F", "M")
        for slot in range(2):
            x, y = np.rint(points[view_index, slot] * scale).astype(int)
            x = int(np.clip(x, 0, live_base.PLAN_IMAGE_SIZE[1] - 1))
            y = int(np.clip(y, 31, live_base.PLAN_IMAGE_SIZE[1] - 1))
            cv2.circle(image, (x, y), 7, colors[slot], 2, cv2.LINE_AA)
            cv2.putText(
                image,
                labels[slot],
                (min(x + 8, live_base.PLAN_IMAGE_SIZE[1] - 30), max(43, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colors[slot],
                1,
                cv2.LINE_AA,
            )
        return image

    def status(self) -> dict[str, Any]:
        payload = super().status()
        payload["version"] = "17.5"
        payload["point_coordinates_present"] = False
        payload["diagnostic_anchor_coordinates_present"] = True
        payload["phase_state_present"] = False
        payload.pop("latest_receipt", None)
        return payload


V175_HTML = (
    live_base.HTML.replace("V18.1", "V17.5")
    .replace("双视角无指尖闭环", "双视角 Grounded-Anchor 闭环")
    .replace("V17.5 / TIP-FREE", "V17.5 / ANCHOR-INNOVATION")
    .replace("不显示指尖点", "F/M/C 仅作结构化观测诊断，不进入控制")
    .replace("V17.5 当前闭环状态", "V17.5 Grounded-Anchor 闭环状态")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_ROOT / "fcpg-final.pt")
    parser.add_argument("--lewm-checkpoint", type=Path, default=DEFAULT_ROOT / "lewm-final.pt")
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ROOT / "actor-final.pt")
    parser.add_argument("--dino-repository", type=Path, default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument(
        "--real-calibration", type=Path, default=DEFAULT_REAL_CALIBRATION
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "service")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--settle-timeout-s", type=float, default=20.0)
    parser.add_argument("--settle-position-tolerance", type=float, default=3.0)
    parser.add_argument("--settle-step-tolerance", type=float, default=0.25)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument("--open-gripper-deg", type=float, default=DEFAULT_GRIPPER_OPEN_POSITION)
    parser.add_argument(
        "--home-state-report",
        type=Path,
        default=REPO / "configs/prod_v17_5_task_home.json",
    )
    parser.add_argument("--home-gripper-deg", type=float, default=DEFAULT_HOME_GRIPPER_POSITION)
    parser.add_argument("--max-auto-chunks", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.control_hz) != DEFAULT_CONTROL_HZ:
        raise ValueError("V17.5 command clock is fixed at 10 Hz")
    if not 1 <= int(args.max_auto_chunks) <= 16:
        raise ValueError("max-auto-chunks must lie in [1,16]")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = V175LivePolicy(
        fcpg_checkpoint=args.fcpg_checkpoint,
        lewm_checkpoint=args.lewm_checkpoint,
        actor_checkpoint=args.actor_checkpoint,
        dino_repository=args.dino_repository,
        dino_weights=args.dino_weights,
        real_calibration=args.real_calibration,
        device=args.device,
        action_lower_position=V175_DEPLOYMENT_POSITION_LOWER,
        action_upper_position=V175_DEPLOYMENT_POSITION_UPPER,
    )
    io_args = SimpleNamespace(
        wrist_device=args.wrist_device,
        column_device=args.column_device,
        output=output,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        max_command_jump=float(STEP_CAP_DEG.max() + 0.1),
        settle_timeout_s=args.settle_timeout_s,
        settle_position_tolerance=args.settle_position_tolerance,
        settle_step_tolerance=args.settle_step_tolerance,
        settle_samples=args.settle_samples,
        self_contact_calibration=False,
        i_accept_empty_jaw_self_contact=False,
        experiment_binding=None,
    )
    io = live_base.HardwareSO101IO(
        io_args,
        gripper_calibration_closed=DEFAULT_GRIPPER_CLOSED_POSITION,
        gripper_calibration_open=DEFAULT_GRIPPER_OPEN_POSITION,
        endpoint_envelope=EndpointEnvelope(
            max_joint_miss=float(STEP_CAP_DEG[:5].max()),
            max_tcp_miss_m=1.0,
            forward=None,
        ),
    )
    controller = V175HardwareService(
        policy,
        io,
        output=output,
        control_hz=args.control_hz,
        max_auto_chunks=args.max_auto_chunks,
        open_gripper_deg=args.open_gripper_deg,
        home_state=live_base._load_home_state(
            args.home_state_report,
            gripper_position=args.home_gripper_deg,
        ),
        motor_calibration=live_base._load_motor_calibration_contract(
            io.follower.calibration_fpath
        ),
    )
    server: ThreadingHTTPServer | None = None
    original_html = live_base.HTML
    try:
        controller.bootstrap()
        live_base.HTML = V175_HTML
        server = ThreadingHTTPServer(
            (args.host, args.port), live_base.handler_factory(controller)
        )
        server.daemon_threads = True
        print(
            f"SO-101 V17.5 direct hardware service: http://{args.host}:{args.port}",
            flush=True,
        )

        def stop_service(signum: int, _frame: Any) -> None:
            controller.stop_event.set()
            raise KeyboardInterrupt(signal.Signals(signum).name)

        signal.signal(signal.SIGINT, stop_service)
        signal.signal(signal.SIGTERM, stop_service)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        live_base.HTML = original_html
        if server is not None:
            server.server_close()
        controller.close()


if __name__ == "__main__":
    main()
