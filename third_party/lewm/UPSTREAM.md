# LeWM Upstream Snapshot

- Repository: https://github.com/lucas-maes/le-wm
- Commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- License: MIT; see `LICENSE` in this directory.
- Snapshot scope: complete tracked upstream contents at the commit above, excluding
  only the upstream Git metadata.

The vendored snapshot includes the upstream `README.md`, `LICENSE`, training and
evaluation entry points (`train.py` and `eval.py`), Hydra configurations under
`config/`, media under `assets/`, and the implementation modules `module.py`,
`jepa.py`, and `utils.py`.

The files in this directory are retained as an unmodified provenance snapshot.
FCPG-specific adaptations live outside `third_party/lewm/`; project code does not
patch or directly modify the vendored sources.

