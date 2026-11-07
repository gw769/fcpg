"""Public FCPG-H V4 query-initialized measurement model.

``FCPGHV4`` is the stable paper/deployment API.  Its frozen learned revision is
the v4.6 content-canonical PartGraph; the earlier experimental v4.0--v4.5
names remain immutable provenance and are not overwritten by this API name.

The module is deliberately smaller than Prod-V13.  It contains no V12 model,
LeWM, actor, SIGReg, camera metadata, calibration, optical flow, mask,
coordinate regressor, or reliability head.  Completed action/CAD history may
only rank an observed query-owned candidate bank.  A currently valid primary
query is protected, and every public pixel is gathered from the current image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fcpg.canonical_query_family_part_graph_v4_6 import (
    FCPGH_V4_6_CHECKPOINT_FORMAT,
    FCPGH_V4_6_METHOD_REVISION,
    FCPGH_V4_6_PROTOCOL_ID,
    canonical_query_family_six_part_decode,
)
from fcpg.fcpg_h_candidate_first import TIP_PART_INDICES
from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.permutation_exact_part_graph_v4_5 import (
    PermutationExactPartGraphTrackAssociation,
)
from fcpg.real_rgb_candidate_interface_v4_6 import (
    FROZEN_FCPGH_CHECKPOINT_SHA256,
)


FCPGH_V4_API_VERSION = "4.0"
FCPGH_V4_API_FORMAT = "so101-fcpg-h-v4-measurement-api-v1"
FCPGH_V4_LEARNED_REVISION = FCPGH_V4_6_METHOD_REVISION
FCPGH_V4_PARAMETER_COUNT = 100_224
FCPGH_V4_FROZEN_SEED = 12_690
FCPGH_V4_FROZEN_STEP = 3_000
FCPGH_V4_FROZEN_MODE = "canonical_bundle_canary"
FCPGH_V4_TRAINING_GIT_PREFIX = "3497215"


@dataclass(frozen=True)
class FCPGHV4Output:
    """Public two-tip measurement plus internal training/audit tensors."""

    logits: torch.Tensor
    tip_candidate_index: torch.Tensor
    protected_primary: torch.Tensor
    tip_visible: torch.Tensor
    tip_px: torch.Tensor
    center_px: torch.Tensor
    anchor_px: torch.Tensor
    anchor_valid: torch.Tensor
    canonical_candidate_order: torch.Tensor


class FCPGHV4(nn.Module):
    """Candidate-only FCPG-H V4 measurement boundary.

    The two tip identities are supplied by the immutable query-owner field of
    the candidate proposer.  The six-part score matrix remains internal; the
    public physical output is fixed tip, moving tip, and their strict midpoint.
    """

    def __init__(
        self,
        grounder: PermutationExactPartGraphTrackAssociation,
    ) -> None:
        super().__init__()
        if not isinstance(grounder, PermutationExactPartGraphTrackAssociation):
            raise TypeError("FCPGHV4 requires the permutation-exact PartGraph")
        if not bool(grounder.temporal_enabled):
            raise ValueError("FCPGHV4 requires causal temporal association")
        count = sum(parameter.numel() for parameter in grounder.parameters())
        if count != FCPGH_V4_PARAMETER_COUNT:
            raise ValueError("FCPGHV4 parameter inventory drifted")
        self.grounder = grounder

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def input_contract() -> dict[str, object]:
        return {
            "candidate_tracks_px": "[B,M,T,2] observed candidate pixels",
            "candidate_valid": "bool[B,M,T]",
            "candidate_source": "int64[B,M] global/local provenance",
            "candidate_query_owner": "int64[B,M] in {-1,0,1}",
            "primary_candidate_index": "int64[B,2] exact query tracks",
            "part_motion": "[B,T,6,44] completed state/action/CAD/FK/Jacobian",
            "camera_metadata": False,
            "calibration": False,
            "future_action": False,
            "lewm": False,
            "actor": False,
        }

    def forward(
        self,
        candidate_tracks_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_source: torch.Tensor,
        candidate_query_owner: torch.Tensor,
        primary_candidate_index: torch.Tensor,
        part_motion: torch.Tensor,
    ) -> FCPGHV4Output:
        association = self.grounder(
            candidate_tracks_px,
            candidate_valid,
            candidate_source,
            part_motion,
        )
        output = canonical_query_family_six_part_decode(
            association,
            candidate_tracks_px,
            candidate_valid,
            candidate_source,
            candidate_query_owner,
            primary_candidate_index,
        )
        if not torch.equal(output.center_px, output.tip_px.mean(dim=1)):
            raise AssertionError("FCPGHV4 center is not the exact tip midpoint")
        tip_rows = torch.as_tensor(
            TIP_PART_INDICES,
            dtype=torch.long,
            device=output.selected_candidate_index.device,
        )
        tip_candidate_index = output.selected_candidate_index.index_select(
            1, tip_rows
        )
        tip_visible = output.selected_visible.index_select(1, tip_rows)
        anchor_px = torch.cat((output.center_px[:, None], output.tip_px), dim=1)
        anchor_valid = torch.cat(
            (tip_visible.all(dim=1, keepdim=True), tip_visible), dim=1
        )
        return FCPGHV4Output(
            logits=output.logits,
            tip_candidate_index=tip_candidate_index,
            protected_primary=output.protected_primary,
            tip_visible=tip_visible,
            tip_px=output.tip_px,
            center_px=output.center_px,
            anchor_px=anchor_px,
            anchor_valid=anchor_valid,
            canonical_candidate_order=output.canonical_candidate_order,
        )


def load_frozen_fcpg_h_v4(
    path: str | Path,
    *,
    device: str | torch.device,
    trainable: bool = False,
) -> tuple[FCPGHV4, dict[str, Any]]:
    """Load the sole frozen V4 endpoint with complete provenance checks.

    ``trainable=True`` exists only for the registered Prod-V13 continuation;
    ordinary measurement inference must use the default frozen endpoint.
    """

    resolved = Path(path).expanduser().resolve()
    file_sha = sha256_file(resolved)
    if file_sha != FROZEN_FCPGH_CHECKPOINT_SHA256:
        raise ValueError("FCPGHV4 requires the frozen v4.6 checkpoint file")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != FCPGH_V4_6_CHECKPOINT_FORMAT
        or payload.get("protocol_id") != FCPGH_V4_6_PROTOCOL_ID
        or payload.get("method_revision") != FCPGH_V4_6_METHOD_REVISION
        or payload.get("arm") != "FCPGH"
        or payload.get("mode") != FCPGH_V4_FROZEN_MODE
        or int(payload.get("seed", -1)) != FCPGH_V4_FROZEN_SEED
        or int(payload.get("step", -1)) != FCPGH_V4_FROZEN_STEP
        or not str(payload.get("git_head", "")).startswith(
            FCPGH_V4_TRAINING_GIT_PREFIX
        )
    ):
        raise ValueError("FCPGHV4 checkpoint identity drifted")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("FCPGHV4 checkpoint has no model state")
    grounder = PermutationExactPartGraphTrackAssociation(
        temporal_enabled=True
    ).to(torch.device(device))
    grounder.load_state_dict(state, strict=True)
    model = FCPGHV4(grounder)
    model.eval()
    model.requires_grad_(bool(trainable))
    metadata = {
        "api_format": FCPGH_V4_API_FORMAT,
        "api_version": FCPGH_V4_API_VERSION,
        "learned_revision": FCPGH_V4_LEARNED_REVISION,
        "checkpoint_path": str(resolved),
        "checkpoint_sha256": file_sha,
        "checkpoint_step": FCPGH_V4_FROZEN_STEP,
        "checkpoint_seed": FCPGH_V4_FROZEN_SEED,
        "parameter_count": model.parameter_count,
        "trainable": bool(trainable),
        "input_contract": model.input_contract(),
    }
    return model, metadata


__all__ = [
    "FCPGHV4",
    "FCPGHV4Output",
    "FCPGH_V4_API_FORMAT",
    "FCPGH_V4_API_VERSION",
    "FCPGH_V4_FROZEN_MODE",
    "FCPGH_V4_FROZEN_SEED",
    "FCPGH_V4_FROZEN_STEP",
    "FCPGH_V4_LEARNED_REVISION",
    "FCPGH_V4_PARAMETER_COUNT",
    "load_frozen_fcpg_h_v4",
]
