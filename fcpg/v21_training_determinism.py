"""Opt-in, versioned numerical reproducibility for independent training runs.

No model architecture, precision, RNG draws or inference settings are changed.
Legacy runs keep their recorded policy; a resume may not switch this policy.
"""
import os

import torch

WORKSPACE = ':4096:8'


def configure_training_determinism(enabled):
    if not enabled:
        return
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace not in (None, WORKSPACE):
        raise ValueError('deterministic training requires CUDA workspace ' + WORKSPACE)
    if workspace is None and torch.cuda.is_initialized():
        raise ValueError('configure deterministic workspace before CUDA is initialized')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False


def reproducibility_record(enabled):
    if not enabled:
        return None
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != WORKSPACE):
        raise ValueError('strict training determinism is not configured')
    return dict(schema='v21-training-determinism-v1', torch=str(torch.__version__),
        torch_git_version=torch.version.git_version, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), cublas_workspace_config=WORKSPACE,
        deterministic_algorithms=True, warn_only=False, cudnn_benchmark=False,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        sdpa_backends=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
            efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled()))


def validate_determinism_resume(previous, enabled, *, current_runtime=None):
    if previous.get('schedule', {}).get('deterministic_training', False) != enabled:
        raise ValueError('training determinism changed; use a new experiment')
    if enabled and not previous.get('reproducibility'):
        raise ValueError('strict training checkpoint is missing its runtime record')
    if enabled and current_runtime is not None and previous['reproducibility'] != current_runtime:
        raise ValueError('deterministic training runtime changed; use a new experiment')
