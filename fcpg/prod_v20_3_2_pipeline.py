"""V20.3.2 training pipeline with a fresh dual-view observer contract."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch

from .prod_v20_3_pipeline import V203Pipeline
from .prod_v20_3_training import V203Schedule
from .prod_v20_3_2 import V2032Core, V2032Config
from .prod_v20_3_2_cache import RawObservationCache, build_observation_cache
from .prod_v20_3_2_sources import V2032Sources
from .prod_v20_3_2_bootstrap import import_cad_observer, validate_pre_refit_checkpoint
from .prod_v20_timing import FISH_COMMAND_PERIOD_S


@dataclass(frozen=True)
class V2032Schedule(V203Schedule):
    seed: int = 20000


class V2032Pipeline(V203Pipeline):
    core_class = V2032Core
    config_class = V2032Config
    schedule_class = V2032Schedule
    sources_class = V2032Sources
    cache_class = RawObservationCache
    cache_builder = staticmethod(build_observation_cache)

    def __init__(self, args):
        if getattr(args, "frozen_from", None) or getattr(args, "observer_from", None):
            raise ValueError("V20.3.2 requires a fresh run; old observer/checkpoint reuse is disabled")
        super().__init__(args)
        self.cad_from = Path(args.cad_from).expanduser().resolve() if getattr(args, "cad_from", None) else None
        if self.cad_from == self.output:
            raise ValueError("CAD reuse requires a separate rerun output")
        self._cad_payloads = None

    def _validated_cad_source(self):
        if self.cad_from is None:
            return None
        if self._cad_payloads is not None:
            return self._cad_payloads
        config = json.loads((self.cad_from / "config.json").read_text())
        expected = dict(version=self.version, contract=self.contract, model=asdict(self.core.config),
                        schedule=asdict(self.schedule), sources=asdict(self.sources.paths), smoke=False)
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"CAD source configuration {key} differs")
        current_fits = json.loads(Path(self.sources.paths.real_calibration).read_text())["fits"]
        if config.get("real_projection_fits") != current_fits:
            raise ValueError("CAD source data/camera contract differs")
        payloads = {}
        for stage in ("appearance", "fcpg"):
            payload = torch.load(self.cad_from / stage / "final.pt", map_location="cpu", weights_only=False)
            payloads[stage] = validate_pre_refit_checkpoint(payload, self.core, self.schedule, stage)
        self._cad_payloads = payloads
        return payloads

    def configuration_extra(self):
        self._validated_cad_source()
        return dict(cad_from=None if self.cad_from is None else str(self.cad_from),
                    refit_history_schedule="independent-view-suffix-v2",
                    runtime_feedback_contract="last-completed-feedback-v1",
                    policy_command_period_s=FISH_COMMAND_PERIOD_S)

    def reuse_cad_stage(self):
        payloads = self._validated_cad_source()
        if payloads is None:
            return False
        info = import_cad_observer(self.core, payloads["fcpg"], self.schedule)
        for stage, payload in payloads.items():
            self._record(stage, dict(complete=True, updates=payload["step"], total_updates=payload["step"],
                reused_from=str(self.cad_from / stage / "final.pt"), trained_in_this_run=False,
                evidence="training-development", **(info if stage == "fcpg" else {})))
        return True


__all__ = ["V2032Schedule", "V2032Pipeline"]
