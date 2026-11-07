"""Separate same-exposure Wrist FK references and Column RGB endpoints.

The service also uses these pairs to exclude jaws from target image support.
The Actor's Wrist-relative cue has its own FK consumer in the object finder's encoder;
this module does not generate actions or substitute Column coordinates for it.
"""
from pathlib import Path
import json

import numpy as np
import torch

from .prod_v20_3_2 import V2032Config, validate_v2032_metadata
from .prod_v20_3_2_observer import V2032PairObserver
from .prod_v20_3_geometry import V203FittedMechanicalProjection
from .robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec
from .so101_camera_calibration import ProjectionFit, predict_projection
from .v21_column_tip_visual import (SOURCE as COLUMN_TIP_SOURCE, locate_column_jaw_tips,
    CENTER_MAHALANOBIS_SQ_LIMIT, RELATIVE_MAHALANOBIS_SQ_LIMIT)
from .v21_column_pose_prior import ColumnPosePrior

DEFAULT_LEGACY_ROOT = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/prod_v20_3_2_dual_view_feedback_rerun')
WRIST_TIP_SOURCE = 'wrist-fk-projection'
TIP_SOURCE_BY_VIEW = dict(wrist=WRIST_TIP_SOURCE, column=COLUMN_TIP_SOURCE)
COLUMN_IDENTITY_MIN_ABS_ALIGNMENT = 0.80
COLUMN_OBSERVER_OFF = ('column jaw-tip observer not run: it reads DINOv2 and this service runs without DINOv2 '
                       '(V21_DINOV2=off, training overhaul doc section 17.57); display only, no control input')


def select_column_proposal_component(weights, visibility, proposals, *, temporal_prior,
                                     pose_prior, visibility_threshold):
    """Bounded temporal rerank of current DINO/RGB proposals, never new points."""
    weights = np.asarray(weights, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)
    proposals = np.asarray(proposals, dtype=np.float64)
    baseline = int(weights.argmax())
    diagnostic = dict(method='current-DINO-weight-plus-bounded-temporal-pair-bonus',
        baseline_component=baseline, selected_component=baseline, changed=False,
        current_frame_proposals_only=True, visibility_threshold=float(visibility_threshold),
        proposals_used_as_endpoints=False, candidates=[])
    if temporal_prior is None or pose_prior is None:
        diagnostic['reason'] = 'temporal or full-q prior unavailable; original component retained'
        return baseline, diagnostic
    scores = np.log(np.maximum(weights, 1e-12))
    for index, pair in enumerate(proposals):
        entry = dict(component=index, weight=float(weights[index]),
                     visibility=visibility[index].tolist(), temporal_bonus=0., eligible=False)
        diagnostic['candidates'].append(entry)
        if (not np.isfinite(pair).all() or not np.isfinite(visibility[index]).all()
                or (visibility[index] < visibility_threshold).any()
                or (pair < [0., 28.]).any() or (pair >= [224., 196.]).any()):
            entry['reason'] = 'current proposal visibility or image support insufficient'
            continue
        pose = pose_prior.score_pair(pair)
        if (pose['center_mahalanobis_sq'] > CENTER_MAHALANOBIS_SQ_LIMIT
                or pose['relative_mahalanobis_sq'] > RELATIVE_MAHALANOBIS_SQ_LIMIT):
            entry['reason'] = 'current proposal inconsistent with full-q geometry'
            continue
        match = temporal_prior.score_pair(pair)
        scores[index] += match['score_bonus']
        entry.update(eligible=True, temporal_bonus=match['score_bonus'], temporal_match=match)
    selected = int(scores.argmax())
    diagnostic.update(selected_component=selected, changed=selected != baseline,
                      reason=None, maximum_log_weight_bonus=temporal_prior.as_dict().get('maximum_score_bonus', .35))
    return selected, diagnostic


def tip_source_matches_view(view, record):
    """Accept only the independently defined tip source for one camera view."""
    return (view in TIP_SOURCE_BY_VIEW and isinstance(record, dict)
            and record.get('source') == TIP_SOURCE_BY_VIEW[view])


def column_pair_hits_sensor_edge(visual):
    """Honor image-support evidence without adding an exclusion band.

    Proximity alone is not clipping: a fully visible distal endpoint at 9.8
    letterbox pixels is still 28 source-camera pixels inside the frame.
    """
    if not isinstance(visual, dict) or visual.get('available') is not True:
        return False
    if visual.get('near_sensor_edge') is True:
        return True
    candidates = visual.get('candidates')
    indices = visual.get('selected_component_indices')
    if not isinstance(candidates, list) or not isinstance(indices, (list, tuple)):
        return False
    for index in indices:
        if type(index) is not int or not 0 <= index < len(candidates):
            continue
        row = candidates[index]
        if not isinstance(row, dict):
            continue
        if row.get('near_sensor_edge') is True:
            return True
    return False


def order_column_pair_from_q_geometry(pair, projected_fg_pair):
    """Order current RGB endpoints as F/G using only the Column FK axis.

    The projected points can be translated relative to the live image after a
    camera move, so their coordinates never replace either RGB endpoint.  Only
    the signed fixed-to-moving direction is used.  Weak or degenerate angular
    support leaves the pair explicitly unordered.
    """
    points = np.asarray(pair, dtype=np.float64)
    reference = np.asarray(projected_fg_pair, dtype=np.float64)
    diagnostic = dict(
        available=False,
        method='column-q-fk-axis',
        reason=None,
        swapped=False,
        alignment=None,
        minimum_absolute_alignment=float(COLUMN_IDENTITY_MIN_ABS_ALIGNMENT),
        endpoint_coordinates_source=COLUMN_TIP_SOURCE,
        projected_coordinates_used=False,
        physical_identity_verified=False,
    )
    if (points.shape != (2, 2) or reference.shape != (2, 2)
            or not np.isfinite(points).all() or not np.isfinite(reference).all()):
        diagnostic['reason'] = 'invalid RGB pair or Column geometry reference'
        return points, diagnostic
    observed_axis = points[1] - points[0]
    projected_axis = reference[1] - reference[0]
    observed_norm = float(np.linalg.norm(observed_axis))
    projected_norm = float(np.linalg.norm(projected_axis))
    diagnostic.update(
        observed_axis_px=observed_axis.tolist(),
        projected_fg_axis_px=projected_axis.tolist(),
        projected_fg_pair_px=reference.tolist(),
    )
    if observed_norm <= 1.0e-6 or projected_norm <= 4.0:
        diagnostic['reason'] = 'degenerate RGB or Column geometry direction'
        return points, diagnostic
    alignment = float(observed_axis @ projected_axis / (observed_norm * projected_norm))
    diagnostic['alignment'] = alignment
    if abs(alignment) < COLUMN_IDENTITY_MIN_ABS_ALIGNMENT:
        diagnostic['reason'] = 'Column geometry direction is too orthogonal to order F/G'
        return points, diagnostic
    swapped = alignment < 0.0
    ordered = points[::-1].copy() if swapped else points.copy()
    diagnostic.update(available=True, swapped=swapped, reason=None)
    return ordered, diagnostic


def tip_record(pair, source, image_time, q, *, reason=None, fk_m_px=None):
    """Display-only tip pair. ``m_px`` is mid(F,G) on the drawn segment only.

    ``fk_m_px`` is a disagreement diagnostic never promoted to M / available.
    Visual (or explicit tip-pair) evidence is required; FK is not ground truth.
    """
    result = dict(available=False, fm_px=None, m_px=None, fk_m_px=None,
                  source=source, display_only=True,
                  control_input=False, coordinate_space='letterbox224',
                  image_time_s=float(image_time), q_at_image=np.asarray(q).tolist(),
                  accuracy_verified=False, reason=reason,
                  fk_disagreement_px=None)
    if fk_m_px is not None:
        cand = np.asarray(fk_m_px, dtype=np.float64).reshape(2)
        if np.isfinite(cand).all() and not ((cand < [0, 28]).any() or (cand >= [224, 196]).any()):
            result['fk_m_px'] = cand.tolist()
    if pair is None:
        # Keep fk_m_px for diagnostics only — do not invent an on-image M.
        if result.get('reason') is None:
            result['reason'] = 'no tip pair; FK not used as M'
        return result
    pair = np.asarray(pair, dtype=np.float64)
    if (pair.shape != (2,2) or not np.isfinite(pair).all()
            or (pair < [0,28]).any() or (pair >= [224,196]).any()
            or np.linalg.norm(pair[0]-pair[1]) < 1e-6):
        result['reason'] = 'invalid or out-of-frame candidate; no clipping/FK-fallback'
        return result
    # M on the picture = midpoint of the F–G segment actually drawn.
    mid = pair.mean(axis=0)
    result.update(available=True, fm_px=pair.tolist(), m_px=mid.tolist())
    if result.get('fk_m_px') is not None:
        fk = np.asarray(result['fk_m_px'], dtype=np.float64)
        result['fk_disagreement_px'] = float(np.linalg.norm(fk - mid))
        # Large FK vs visual mid gap → mark unverified (column bias alarm).
        result['accuracy_verified'] = bool(result['fk_disagreement_px'] <= 20.0)
    return result


class TipDisplay:
    def __init__(self, calibration_path, wrist_kinematics, *, device='cpu', legacy_root=DEFAULT_LEGACY_ROOT):
        self.device = torch.device(device)
        self.wrist_kinematics = wrist_kinematics
        self.calibration_path = Path(calibration_path)
        self.legacy_root = Path(legacy_root)
        calibration = json.loads(self.calibration_path.read_text(encoding='utf-8'))
        if calibration.get('geometry_protocol') != 'letterbox_v1' or calibration.get('content_rect') != [0,28,224,196]:
            raise ValueError('tip display requires the actual letterbox calibration')
        self.raw_shape = (calibration['source_height'], calibration['source_width'])
        row = calibration['fits']['wrist']
        self.wrist_fit = ProjectionFit('wrist', tuple(row['feature_names']),
            np.asarray(row['coefficients'],np.float64),float(row['ridge']))
        # Column geometry orders the two current RGB endpoints; it never
        # supplies their coordinates.  Keep loading isolated so a missing or
        # invalid Column fit cannot break the independent Wrist projection.
        self.column_fit = None
        self.column_fit_error = None
        try:
            crow = calibration['fits']['column']
            self.column_fit = ProjectionFit('column', tuple(crow['feature_names']),
                np.asarray(crow['coefficients'],np.float64),float(crow['ridge']))
        except Exception as exc:
            self.column_fit_error = f'{type(exc).__name__}: {exc}'
        # Column's visual proposal dependencies are still loaded only in its branch.
        self.observer = self.geometry = None
        self.column_load_error = None

    def _fk_m_px(self, view, q):
        """Project FK centre M = forward_anchors[0] into letterbox224 (wrist only)."""
        if view != 'wrist':
            raise ValueError('FK tip projection is wrist-only; column uses visual tips')
        q = np.asarray(q, dtype=np.float64).reshape(6)
        world, local = self.wrist_kinematics.forward_anchor_frames(q[None])
        pixels = predict_projection(self.wrist_fit, world, local, np.asarray([q[-1]]))[0]
        return np.asarray(pixels[0], dtype=np.float64)

    def _column_identity_reference(self, q):
        """Return projected Column [F,G] for ordering, never pixel output."""
        fit = getattr(self, 'column_fit', None)
        if fit is None:
            return None, dict(available=False,
                reason=getattr(self, 'column_fit_error', None) or 'Column projection fit unavailable')
        q = np.asarray(q, dtype=np.float64).reshape(6)
        world, local = self.wrist_kinematics.forward_anchor_frames(q[None])
        projected = predict_projection(fit, world, local, np.asarray([q[-1]]))[0, 1:3]
        projected = np.asarray(projected, dtype=np.float64)
        if projected.shape != (2, 2) or not np.isfinite(projected).all():
            return None, dict(available=False, reason='invalid Column projected F/G direction')
        physical_span = float(np.linalg.norm(world[0, 2] - world[0, 1]))
        return projected, dict(
            available=True,
            source='column-q-fk-projection',
            q_conditioned=True,
            j5_position=float(q[4]),
            gripper_position=float(q[5]),
            physical_tip_separation_m=physical_span,
            absolute_projected_position_used=False,
        )

    def _column_pose_prior(self, q, proposal):
        """Project the complete exposure-time FK through uncertain Column views."""
        fit = getattr(self, 'column_fit', None)
        if fit is None:
            return None, dict(available=False, reason='Column camera basis unavailable')
        try:
            prior = ColumnPosePrior.from_kinematics(
                self.wrist_kinematics, q, fit, proposal_pair=proposal)
            record = prior.as_dict()
            record['available'] = True
            return prior, record
        except (TypeError, ValueError, AttributeError) as exc:
            return None, dict(available=False, reason=f'{type(exc).__name__}: {exc}')

    def _load_column(self):
        if self.observer is not None: return
        if self.column_load_error is not None: raise ValueError(self.column_load_error)
        try:
            data = json.loads((self.legacy_root/'config.json').read_text(encoding='utf-8'))
            calibration = json.loads(self.calibration_path.read_text(encoding='utf-8'))
            if data.get('smoke') is not False or data.get('real_projection_fits') != calibration.get('fits'):
                raise ValueError('RAW checkpoint and current calibration differ')
            config = V2032Config(**data['model'])
            payload = torch.load(self.legacy_root/'fish/final.pt', map_location='cpu', weights_only=False)
            validate_v2032_metadata(payload, config, stage='fish')
            if payload.get('step') != data['schedule']['fish_updates']:
                raise ValueError('RAW source checkpoint is not final')
            observer = V2032PairObserver(attention_dim=config.observer_attention_dim,
                attention_heads=config.observer_attention_heads, attention_layers=config.observer_attention_layers,
                evidence_threshold=config.visibility_evidence_threshold,
                max_column_residual_px=config.max_column_residual_px)
            values = {k.removeprefix('observer.'):v for k,v in payload['model'].items() if k.startswith('observer.')}
            if any(not torch.isfinite(v).all() for v in values.values()):
                raise ValueError('nonfinite RAW weights')
            observer.load_state_dict(values, strict=True)
            kin = DifferentiableRobotKinematics(so101_robot_spec()).to(self.device).eval().requires_grad_(False)
            geometry = V203FittedMechanicalProjection(self.calibration_path, device=self.device, kinematics=kin)
            self.observer = observer.to(self.device).eval().requires_grad_(False)
            self.geometry = geometry
        except Exception as exc:
            self.column_load_error = f'{type(exc).__name__}: {exc}'
            raise

    @torch.inference_mode()
    def observe(self, rgb, endpoint, raw_frames, history, *, column_temporal_prior=None, dino_tokens=None,
                column_observer=True):
        """dino_tokens: this exposure's DINOv2 tokens [2, 256, 384] when the model reads another encoder
        (fcpg/v21_live_encoder.py): the column observer below was trained on DINOv2. column_observer=False (a
        V21_DINOV2=off service, training overhaul doc section 17.57): the column pair is not looked for; its record says
        why. Column FK is still never drawn as F/G (_mark_column_fg_display)."""
        obs = endpoint.observation
        valid = obs.measurement_available.detach().clone()
        records = {}
        for index, view in enumerate(('wrist','column')):
            q = obs.q_at_image[0,index].detach().cpu().numpy()
            stamp = float(obs.image_times[0,index])
            source = TIP_SOURCE_BY_VIEW[view]
            try:
                if not bool(valid[0,index]):
                    raise ValueError('no current image; no old point copied forward')
                if view not in raw_frames or raw_frames[view].shape[:2] != self.raw_shape:
                    raise ValueError('actual image geometry differs from the tip reference')
                if view == 'wrist':
                    # The wrist camera is rigid on the arm: at the exposure q the
                    # fingertips sit at a kinematically determined pixel pair.
                    # Draw that FK+calibration pair as F/G (M = midpoint). Display
                    # reference; not visual ground truth. The Actor computes
                    # its Wrist-relative cue separately using its own consumer.
                    world, local = self.wrist_kinematics.forward_anchor_frames(q[None])
                    fk_pair = predict_projection(self.wrist_fit, world, local, np.asarray([q[-1]]))[0,1:3]
                    rec = tip_record(
                        fk_pair, source, stamp, q,
                        fk_m_px=self._fk_m_px('wrist', q))
                    fk = np.asarray(fk_pair, dtype=np.float64)
                    rec['fk_fm_px'] = fk.tolist()
                    rec['fk_based'] = True
                    rec['accuracy_verified'] = False
                    # An off-frame tip stays out of the measurement. The picture
                    # still shows F/G, pinned to the border when a point leaves it.
                    if (rec.get('fm_px') is None and fk.shape == (2, 2)
                            and np.isfinite(fk).all()):
                        rec['display_fm_px'] = fk.tolist()
                        rec['display_fg_source'] = 'wrist_fk'
                    records[view] = rec
                elif not column_observer:
                    # V21_DINOV2=off (section 17.57): the observer below reads DINOv2; no column pair is looked for
                    records[view] = tip_record(None, source, stamp, q, reason=COLUMN_OBSERVER_OFF)
                    records[view].update(unordered=True, ordered_identity_available=False,
                                         physical_identity_verified=False, fk_based=False, column_observer_run=False)
                else:
                    # Column G/F must come from the same exposure: DINO2 proposes
                    # the jaw region, then the RGB contour detector confirms the
                    # actual distal endpoints.  FK is allowed to rank/order and
                    # reject inconsistent observations, never to replace pixels.
                    self._load_column()
                    pose = obs.q_at_image[:, 1].to(self.device).clone()
                    prior = self.geometry.make_prior(pose)
                    pixels = torch.as_tensor(
                        np.array(rgb, copy=True), device=self.device
                    ).permute(0, 3, 1, 2)[None]
                    features = (obs.features if dino_tokens is None
                                else torch.as_tensor(np.asarray(dino_tokens), dtype=obs.features.dtype)[None])
                    output = self.observer(
                        pixels, features.to(self.device).clone(),
                        valid.to(self.device), prior, q_position=pose,
                        actual_history=history, mechanical_prior=prior)
                    candidate = output.candidate_belief
                    component = int(candidate.weights[0, 1].argmax())
                    visibility = candidate.visible_probability[0, 1, component].cpu().numpy()
                    proposal = output.raw_column_fm_px[0, component].cpu().numpy()
                    # Column F/G is this exposure's visual pair. The external
                    # camera FK is not a substitute for it.
                    search_proposal = np.asarray(proposal, np.float64)
                    pose_prior, pose_record = self._column_pose_prior(q, proposal)
                    component_selection = None
                    if column_temporal_prior is not None:
                        selected, component_selection = select_column_proposal_component(
                            candidate.weights[0, 1].cpu().numpy(),
                            candidate.visible_probability[0, 1].cpu().numpy(),
                            output.raw_column_fm_px[0].cpu().numpy(),
                            temporal_prior=column_temporal_prior,
                            pose_prior=pose_prior,
                            visibility_threshold=float(self.observer.evidence_threshold))
                        if selected != component:
                            component = selected
                            visibility = candidate.visible_probability[0, 1, component].cpu().numpy()
                            proposal = output.raw_column_fm_px[0, component].cpu().numpy()
                            search_proposal = np.asarray(proposal, np.float64)
                            pose_prior, pose_record = self._column_pose_prior(q, proposal)
                    pose_record['image_time_s'] = stamp
                    visual_kwargs = dict(image_space='raw')
                    if pose_prior is not None:
                        visual_kwargs['pose_prior'] = pose_prior
                    if column_temporal_prior is not None:
                        visual_kwargs['temporal_prior'] = column_temporal_prior
                    visual = locate_column_jaw_tips(
                        raw_frames[view], search_proposal, **visual_kwargs)
                    if column_pair_hits_sensor_edge(visual):
                        visual = dict(visual, available=False, fm_px=None,
                            reason='sensor-edge endpoint is not a jaw tip',
                            rejected_sensor_edge=True)
                    # A current DINO F/G pair is the jaw. The contour does not
                    # replace it.
                    dino_pair = np.asarray(proposal, np.float64)
                    dino_visible = False
                    try:
                        dino_visible = min(float(item) for item in np.asarray(visibility).reshape(-1)) >= 0.5
                    except (TypeError, ValueError):
                        dino_visible = False
                    if (dino_visible and dino_pair.shape == (2, 2) and np.isfinite(dino_pair).all()):
                        visual = dict(visual, available=True, fm_px=dino_pair.tolist(),
                                      reason=None, dino_fg_trusted=True,
                                      contour_fm_px=visual.get('fm_px'))
                    visual_pair = visual['fm_px'] if visual['available'] else None
                    ordered_pair = visual_pair
                    identity = dict(
                        available=False, method='column-q-fk-axis',
                        reason='no current RGB endpoint pair')
                    identity_reference = None
                    reference_status = dict(
                        available=False, reason='Column projection fit unavailable')
                    if visual_pair is not None and pose_prior is not None:
                        match = pose_prior.score_pair(visual_pair)
                        ordered_pair = match['ordered_pair']
                        identity = dict(
                            available=bool(match['identity_probability'] >= .95),
                            method='full-q-camera-distribution',
                            reason=(None if match['identity_probability'] >= .95
                                    else 'F/G identity remains ambiguous under camera uncertainty'),
                            swapped=bool(match['swapped']),
                            identity_probability=float(match['identity_probability']),
                            log_likelihood=float(match['log_likelihood']),
                            projected_coordinates_used=False,
                            endpoint_coordinates_source=COLUMN_TIP_SOURCE,
                            physical_identity_verified=False,
                        )
                    elif visual_pair is not None:
                        identity_reference, reference_status = self._column_identity_reference(q)
                        if identity_reference is None:
                            identity.update(reason=reference_status.get('reason'))
                        else:
                            ordered_pair, identity = order_column_pair_from_q_geometry(
                                visual_pair, identity_reference)
                            identity['reference_source'] = reference_status.get('source')
                            identity['q_pose_prior'] = reference_status
                    identity_available = bool(identity.get('available'))
                    records[view] = tip_record(
                        ordered_pair, source, stamp, q, reason=visual.get('reason'))
                    records[view].update(
                        visual_diagnostics=visual,
                        pose_prior=pose_record,
                        temporal_prior=(None if column_temporal_prior is None
                                        else column_temporal_prior.as_dict()),
                        legacy_proposal=dict(
                            pair_px=proposal.tolist(), component=component,
                            visibility=visibility.tolist(),
                            component_selection=component_selection,
                            checkpoint=str(self.legacy_root / 'fish' / 'final.pt'),
                            prior_used=True,
                            ignored_far_from_fk=False,
                            geometry_search_px=np.asarray(search_proposal, np.float64).tolist()),
                        identity_assignment=identity,
                        unordered=not identity_available,
                        ordered_identity_available=identity_available,
                        physical_identity_verified=False,
                        pixel_position_fk_based=False,
                        identity_order_fk_based=identity_available,
                        fk_based=False)
            except Exception as exc:
                records[view] = tip_record(None, source, stamp, q, reason=f'{type(exc).__name__}: {exc}')
                if view == 'column':
                    records[view].update(unordered=True, ordered_identity_available=False,
                        physical_identity_verified=False, fk_based=False)
            if view == 'column':
                records[view].update(observed=bool(
                        records[view]['available'] and records[view].get('source') == 'column-rgb-jaw-contour'),
                    suggested_fm_px=None, suggested_m_px=None, prediction_suggestion=None)
                if column_temporal_prior is not None and not records[view]['available']:
                    suggestion = column_temporal_prior.as_suggestion(reason=records[view].get('reason'))
                    suggestion['current_frame_available'] = bool(valid[0, index])
                    retained = tip_record(
                        suggestion['suggested_fm_px'], 'fcpg-retained-column-fg', stamp, q,
                        reason='current DINO target absent; FCPG retained F/G')
                    retained.update(observed=False, display_only=True, control_input=False,
                        learned_policy_input=False, actual_visual_memory_write=False,
                        fcpg_pixel_prediction_available=False, dino_target_absent=True,
                        current_frame_available=bool(valid[0, index]),
                        temporal_prior=column_temporal_prior.as_dict(),
                        suggested_fm_px=suggestion['suggested_fm_px'],
                        suggested_m_px=suggestion['suggested_m_px'],
                        prediction_suggestion=suggestion)
                    records[view] = retained
                self._mark_column_fg_display(records[view])
        return records

    @staticmethod
    def _mark_column_fg_display(rec):
        """Column F/G is the visual pair. External-camera FK is not drawn as F/G."""
        def as_pair(value):
            try:
                pair = np.asarray(value, np.float64)
            except (TypeError, ValueError):
                return None
            if pair.shape != (2, 2) or not np.isfinite(pair).all():
                return None
            return pair

        observed = as_pair(rec.get('fm_px') if rec.get('available') else None)
        proposal = as_pair((rec.get('legacy_proposal') or {}).get('pair_px'))
        if proposal is None:
            proposal = as_pair(rec.get('suggested_fm_px'))
        visibility = (rec.get('legacy_proposal') or {}).get('visibility') or []
        recognized = False
        try:
            recognized = len(visibility) > 0 and min(float(item) for item in visibility) >= .5
        except (TypeError, ValueError):
            recognized = False
        if recognized and proposal is not None:
            chosen, source = proposal, 'column_fg'
        elif observed is not None:
            chosen, source = observed, 'observed'
        elif proposal is not None:
            chosen, source = proposal, 'column_fg'
        else:
            return
        rec['display_fm_px'] = chosen.tolist()
        rec['display_fg_source'] = source

    @staticmethod
    def _retained_far_from_fk(retained_pair, fk_pair):
        """A leftover pair farther than the jaw itself is not this exposure's fingers."""
        retained = np.asarray(retained_pair, np.float64)
        fk = np.asarray(fk_pair, np.float64)
        if retained.shape != (2, 2) or fk.shape != (2, 2) or not np.isfinite(retained).all() or not np.isfinite(fk).all():
            return False
        span = float(np.linalg.norm(fk[1] - fk[0]))
        gap = float(np.linalg.norm(retained.mean(0) - fk.mean(0)))
        return gap > max(span, 1.)
