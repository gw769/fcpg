"""17.108: the full-patch LeWM's SIGReg weight is an option (default .02, the deployed LeWM's); runs saved before the
17.106 / 17.108 options still resume with the new code."""
from scripts.train_v21_horizon40 import canonical_protocol, parser

BASE = ['--parent', 'parent.pt', '--source-config', 'config.json', '--output', 'out']


def test_the_weight_defaults_to_the_deployed_lewms():
    assert parser().parse_args(BASE).lewm_sigreg_weight == .02
    assert parser().parse_args(BASE + ['--lewm-sigreg-weight', '0.09']).lewm_sigreg_weight == .09


def test_runs_saved_before_the_options_read_as_their_defaults():
    old = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False), sigreg_weight=.02)
    new = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False, lewm_width=384, lewm_layers=4,
                             window_readers=0, lewm_sigreg_weight=.02), sigreg_weight=.02)
    assert canonical_protocol(old) == canonical_protocol(new)
    stronger = dict(new, schedule=dict(new['schedule'], lewm_sigreg_weight=.09), sigreg_weight=.09)
    assert canonical_protocol(old) != canonical_protocol(stronger)
