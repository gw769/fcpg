"""No colour code in the system (10/31, James 「不要的废弃代码 这种颜色函数删掉」): the operator colour model, its chroma
helpers, the colour head target, the bright-background mask, the colour template tracker and the machinery that made them
raise at run time are gone from the code. This guards against their return: no file of the repository defines one of
these functions again, and the modules that held them no longer have them."""
import ast
import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOVED = ('chroma_direction_model', 'chroma_likelihood', 'locate_by_chroma', 'locate_visible_column_box',
           '_locate_dark_workspace_container', 'bright_background_mask', 'registered_head_candidate',
           'colour_model_for_pickup_view', 'colour_body_geometry', 'overwrite_pickup_with_colour_body',
           'reacquire_partial_colour_body', 'track_by_chroma', 'locate_by_operator_colour', 'forbid_colour',
           'restore_colour', '_chroma_deviation')
GONE_MODULES = ('fcpg/v21_head_target.py', 'fcpg/v21_head_grasp.py', 'fcpg/v21_head_contact.py')


def test_no_file_defines_a_colour_function():
    found = []
    for path in [*REPO.glob('fcpg/**/*.py'), *REPO.glob('scripts/**/*.py'), *REPO.glob('analysis/**/*.py')]:
        try:
            tree = ast.parse(path.read_text(errors='ignore'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in REMOVED:
                found.append(f'{path.relative_to(REPO)}:{node.lineno} {node.name}')
    assert not found, found


def test_the_modules_that_held_them_no_longer_have_them():
    assert not any((REPO / name).exists() for name in GONE_MODULES)
    targets = importlib.import_module('fcpg.so101_visual_targets')
    relative = importlib.import_module('fcpg.v21_object_relative')
    grounding = importlib.import_module('fcpg.so101_frozen_vit_grounding')
    finder = importlib.import_module('fcpg.v21_sam3_finder')
    for module in (targets, relative, grounding, finder):
        assert not [name for name in REMOVED if hasattr(module, name)], module.__name__
    for method in ('track', 'track_by_chroma', 'locate_by_operator_colour', '_score_map'):
        assert not hasattr(targets.VisualTargetTracker, method)
    assert not hasattr(relative, 'ObjectRelativeEncoder')
    assert sorted(relative.__all__) == sorted(('DIM', 'FIELDS', 'TRACKER_MINIMUM_SCORE', 'zeros', 'encode_vector',
                                                'encode_place_vector', 'assistance_cue'))
