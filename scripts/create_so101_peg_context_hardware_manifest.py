#!/usr/bin/env python3
"""Create the frozen H0-H2 SO-101 peg-context hardware schedule.

This module only assigns experiment identities, strata, adversarial conditions,
and paired method order. It never opens a camera or robot port and never writes
an experimental result.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any


FORMAT = "so101-peg-context-hardware-schedule-v1"
PROTOCOL = "so101-peg-context-formal-hardware-h0-h2-v1"

MAX_CHUNKS_PER_ATTEMPT = 16
MAX_WALL_TIME_S_PER_ATTEMPT = 180
STRATA = ("crossing", "near", "clear")
METHODS = ("full", "no_context")
ADVERSARIES = ("peg_drift", "occlusion")

_TOP_KEYS = {"format", "protocol", "seed", "created_utc", "budgets", "stages"}
_BUDGET_KEYS = {"max_chunks_per_attempt", "max_wall_time_s_per_attempt"}
_STAGE_KEYS = {"h0", "h1", "h2"}
_H0_TRIAL_KEYS = {
    "ordinary": {"trial_id", "attempt_id", "round_kind", "stratum"},
    "adversarial": {"trial_id", "attempt_id", "round_kind", "adversary"},
}
_H1_TRIAL_KEYS = {"trial_id", "attempt_id", "root_id"}
_H2_ROOT_KEYS = {"root_id", "stratum", "order", "attempts"}
_H2_ATTEMPT_KEYS = {"trial_id", "attempt_id", "method", "order_index"}


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    if value < 0 or value > 2**63 - 1:
        raise ValueError("seed must lie in [0, 2**63 - 1]")
    return int(value)


def _validate_created_utc(value: object) -> str:
    text = _require_string(value, "created_utc")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("created_utc must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise ValueError("created_utc must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ")
    return text


def _rng(seed: int, stream: int) -> random.Random:
    """Return an independent deterministic stream for one schedule component."""

    return random.Random(seed ^ (0x5DEECE66D + stream * 0x9E3779B1))


def _build_h0(seed: int) -> dict[str, object]:
    ordinary = [stratum for stratum in STRATA for _ in range(4)]
    adversarial = [adversary for adversary in ADVERSARIES for _ in range(3)]
    _rng(seed, 0).shuffle(ordinary)
    _rng(seed, 1).shuffle(adversarial)

    trials: list[dict[str, object]] = []
    for index, stratum in enumerate(ordinary, start=1):
        trial_id = f"h0-trial-{index:03d}"
        trials.append(
            {
                "trial_id": trial_id,
                "attempt_id": f"{trial_id}-attempt-01",
                "round_kind": "ordinary",
                "stratum": stratum,
            }
        )
    for index, adversary in enumerate(adversarial, start=len(trials) + 1):
        trial_id = f"h0-trial-{index:03d}"
        trials.append(
            {
                "trial_id": trial_id,
                "attempt_id": f"{trial_id}-attempt-01",
                "round_kind": "adversarial",
                "adversary": adversary,
            }
        )
    return {"trials": trials}


def _build_h1() -> dict[str, object]:
    trials: list[dict[str, object]] = []
    for index in range(1, 5):
        root_id = f"h1-root-{index:03d}"
        trial_id = f"h1-trial-{index:03d}"
        trials.append(
            {
                "trial_id": trial_id,
                "attempt_id": f"{trial_id}-attempt-01",
                "root_id": root_id,
            }
        )
    return {"trials": trials}


def _build_h2(seed: int) -> dict[str, object]:
    strata = [stratum for stratum in STRATA for _ in range(8)]
    orders = [METHODS for _ in range(12)] + [METHODS[::-1] for _ in range(12)]
    _rng(seed, 2).shuffle(strata)
    _rng(seed, 3).shuffle(orders)

    roots: list[dict[str, object]] = []
    for index, (stratum, order_tuple) in enumerate(
        zip(strata, orders, strict=True), start=1
    ):
        root_id = f"h2-root-{index:03d}"
        order = list(order_tuple)
        attempts: list[dict[str, object]] = []
        for order_index, method in enumerate(order):
            trial_id = f"{root_id}-{method}"
            attempts.append(
                {
                    "trial_id": trial_id,
                    "attempt_id": f"{trial_id}-attempt-01",
                    "method": method,
                    "order_index": order_index,
                }
            )
        roots.append(
            {
                "root_id": root_id,
                "stratum": stratum,
                "order": order,
                "attempts": attempts,
            }
        )
    return {"roots": roots}


def _build_stages(seed: int) -> dict[str, object]:
    return {"h0": _build_h0(seed), "h1": _build_h1(), "h2": _build_h2(seed)}


def build_manifest(seed: int, created_utc: str) -> dict[str, object]:
    """Build and validate one deterministic schedule for ``seed``."""

    normalized_seed = _require_seed(seed)
    normalized_created = _validate_created_utc(created_utc)
    payload: dict[str, object] = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": normalized_seed,
        "created_utc": normalized_created,
        "budgets": {
            "max_chunks_per_attempt": MAX_CHUNKS_PER_ATTEMPT,
            "max_wall_time_s_per_attempt": MAX_WALL_TIME_S_PER_ATTEMPT,
        },
        "stages": _build_stages(normalized_seed),
    }
    return validate_manifest(payload)


def _validate_h0(stage: dict[str, object]) -> None:
    _require_exact_keys(stage, {"trials"}, "stages.h0")
    trials = _require_list(stage["trials"], "stages.h0.trials")
    if len(trials) != 18:
        raise ValueError("H0 requires exactly 18 trials")
    ordinary: list[dict[str, object]] = []
    adversarial: list[dict[str, object]] = []
    for index, raw in enumerate(trials):
        trial = _require_object(raw, f"stages.h0.trials[{index}]")
        kind = _require_string(trial.get("round_kind"), f"H0 trial {index} round_kind")
        if kind not in _H0_TRIAL_KEYS:
            raise ValueError(f"H0 trial {index} has unsupported round_kind {kind!r}")
        _require_exact_keys(trial, _H0_TRIAL_KEYS[kind], f"stages.h0.trials[{index}]")
        _require_string(trial["trial_id"], f"H0 trial {index} trial_id")
        _require_string(trial["attempt_id"], f"H0 trial {index} attempt_id")
        if kind == "ordinary":
            stratum = _require_string(trial["stratum"], f"H0 trial {index} stratum")
            if stratum not in STRATA:
                raise ValueError(f"H0 trial {index} has unsupported stratum {stratum!r}")
            ordinary.append(trial)
        else:
            adversary = _require_string(trial["adversary"], f"H0 trial {index} adversary")
            if adversary not in ADVERSARIES:
                raise ValueError(f"H0 trial {index} has unsupported adversary {adversary!r}")
            adversarial.append(trial)
    if len(ordinary) != 12 or Counter(row["stratum"] for row in ordinary) != Counter(
        {stratum: 4 for stratum in STRATA}
    ):
        raise ValueError("H0 ordinary trials require crossing/near/clear counts 4/4/4")
    if len(adversarial) != 6 or Counter(
        row["adversary"] for row in adversarial
    ) != Counter({adversary: 3 for adversary in ADVERSARIES}):
        raise ValueError("H0 adversarial trials require peg_drift/occlusion counts 3/3")
    if any(row["round_kind"] != "ordinary" for row in trials[:12]) or any(
        row["round_kind"] != "adversarial" for row in trials[12:]
    ):
        raise ValueError("H0 ordinary trials must precede adversarial trials")


def _validate_h1(stage: dict[str, object]) -> None:
    _require_exact_keys(stage, {"trials"}, "stages.h1")
    trials = _require_list(stage["trials"], "stages.h1.trials")
    if len(trials) != 4:
        raise ValueError("H1 requires exactly four pilot trials")
    for index, raw in enumerate(trials):
        trial = _require_object(raw, f"stages.h1.trials[{index}]")
        _require_exact_keys(trial, _H1_TRIAL_KEYS, f"stages.h1.trials[{index}]")
        for field in _H1_TRIAL_KEYS:
            _require_string(trial[field], f"H1 trial {index} {field}")


def _validate_h2(stage: dict[str, object]) -> None:
    _require_exact_keys(stage, {"roots"}, "stages.h2")
    roots = _require_list(stage["roots"], "stages.h2.roots")
    if len(roots) != 24:
        raise ValueError("H2 requires exactly 24 roots")
    first_methods: list[str] = []
    strata: list[str] = []
    for root_index, raw_root in enumerate(roots):
        root = _require_object(raw_root, f"stages.h2.roots[{root_index}]")
        _require_exact_keys(root, _H2_ROOT_KEYS, f"stages.h2.roots[{root_index}]")
        _require_string(root["root_id"], f"H2 root {root_index} root_id")
        stratum = _require_string(root["stratum"], f"H2 root {root_index} stratum")
        if stratum not in STRATA:
            raise ValueError(f"H2 root {root_index} has unsupported stratum {stratum!r}")
        strata.append(stratum)
        order = _require_list(root["order"], f"H2 root {root_index} order")
        if any(not isinstance(method, str) for method in order):
            raise ValueError(f"H2 root {root_index} methods must be strings")
        if len(order) != 2 or set(order) != set(METHODS):
            raise ValueError(f"H2 root {root_index} order must contain both methods once")
        first_methods.append(order[0])
        attempts = _require_list(root["attempts"], f"H2 root {root_index} attempts")
        if len(attempts) != 2:
            raise ValueError(f"H2 root {root_index} requires exactly two attempts")
        for order_index, raw_attempt in enumerate(attempts):
            attempt = _require_object(
                raw_attempt, f"stages.h2.roots[{root_index}].attempts[{order_index}]"
            )
            _require_exact_keys(
                attempt,
                _H2_ATTEMPT_KEYS,
                f"stages.h2.roots[{root_index}].attempts[{order_index}]",
            )
            _require_string(attempt["trial_id"], "H2 trial_id")
            _require_string(attempt["attempt_id"], "H2 attempt_id")
            if attempt["method"] != order[order_index]:
                raise ValueError(f"H2 root {root_index} attempt method differs from order")
            if (
                isinstance(attempt["order_index"], bool)
                or not isinstance(attempt["order_index"], int)
                or attempt["order_index"] != order_index
            ):
                raise ValueError(f"H2 root {root_index} attempt order_index is invalid")
    if Counter(strata) != Counter({stratum: 8 for stratum in STRATA}):
        raise ValueError("H2 roots require crossing/near/clear counts 8/8/8")
    if Counter(first_methods) != Counter({method: 12 for method in METHODS}):
        raise ValueError("H2 method order must be balanced 12/12")


def _validate_unique_ids(stages: dict[str, object]) -> None:
    trial_ids: list[str] = []
    attempt_ids: list[str] = []
    root_ids: list[str] = []
    for trial in stages["h0"]["trials"]:
        trial_ids.append(trial["trial_id"])
        attempt_ids.append(trial["attempt_id"])
    for trial in stages["h1"]["trials"]:
        trial_ids.append(trial["trial_id"])
        attempt_ids.append(trial["attempt_id"])
        root_ids.append(trial["root_id"])
    for root in stages["h2"]["roots"]:
        root_ids.append(root["root_id"])
        for attempt in root["attempts"]:
            trial_ids.append(attempt["trial_id"])
            attempt_ids.append(attempt["attempt_id"])
    for name, values in (
        ("trial_id", trial_ids),
        ("attempt_id", attempt_ids),
        ("root_id", root_ids),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"all {name} values must be unique")
    all_ids = trial_ids + attempt_ids + root_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("trial_id, attempt_id, and root_id namespaces must not overlap")


def validate_manifest(payload: dict[str, object]) -> dict[str, object]:
    """Validate and return a defensive copy of the canonical schedule payload."""

    manifest = _require_object(payload, "manifest")
    _require_exact_keys(manifest, _TOP_KEYS, "manifest")
    if manifest["format"] != FORMAT:
        raise ValueError(f"manifest format must be {FORMAT!r}")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError(f"manifest protocol must be {PROTOCOL!r}")
    seed = _require_seed(manifest["seed"])
    _validate_created_utc(manifest["created_utc"])

    budgets = _require_object(manifest["budgets"], "budgets")
    _require_exact_keys(budgets, _BUDGET_KEYS, "budgets")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in budgets.values()):
        raise ValueError("hardware attempt budgets must be integers")
    if budgets != {
        "max_chunks_per_attempt": MAX_CHUNKS_PER_ATTEMPT,
        "max_wall_time_s_per_attempt": MAX_WALL_TIME_S_PER_ATTEMPT,
    }:
        raise ValueError("hardware attempt budgets must remain frozen at 16 chunks/180 s")

    stages = _require_object(manifest["stages"], "stages")
    _require_exact_keys(stages, _STAGE_KEYS, "stages")
    h0 = _require_object(stages["h0"], "stages.h0")
    h1 = _require_object(stages["h1"], "stages.h1")
    h2 = _require_object(stages["h2"], "stages.h2")
    _validate_h0(h0)
    _validate_h1(h1)
    _validate_h2(h2)
    _validate_unique_ids(stages)

    expected_stages = _build_stages(seed)
    if stages != expected_stages:
        raise ValueError("stage schedule does not match the deterministic seed assignment")

    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must contain finite JSON values only") from exc
    return copy.deepcopy(manifest)


def iter_trials(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return H0/H1 trials and flattened H2 attempts in scheduled order."""

    manifest = validate_manifest(payload)
    stages = manifest["stages"]
    rows: list[dict[str, object]] = []
    for trial in stages["h0"]["trials"]:
        rows.append({"stage": "h0", **copy.deepcopy(trial)})
    for trial in stages["h1"]["trials"]:
        rows.append({"stage": "h1", **copy.deepcopy(trial)})
    for root in stages["h2"]["roots"]:
        for attempt in root["attempts"]:
            rows.append(
                {
                    "stage": "h2",
                    "root_id": root["root_id"],
                    "stratum": root["stratum"],
                    "order": copy.deepcopy(root["order"]),
                    **copy.deepcopy(attempt),
                }
            )
    return rows


def _current_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_manifest(payload: dict[str, object], output: Path, *, replace: bool) -> None:
    normalized = validate_manifest(payload)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with output.open(mode, encoding="ascii") as stream:
        json.dump(normalized, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--created-utc", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing schedule; default is fail closed.",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest(args.seed, args.created_utc or _current_utc())
    _write_manifest(manifest, args.output, replace=args.replace)
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
