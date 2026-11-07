"""Column-only RGB jaw endpoints inside an explicitly supplied search region.

The old visual pair locates a broad search region and rejects an implausibly
collapsed pair; it is never an endpoint label. Two elongated cool-gray
components attached to pale gripper housings must be present in this image.
Their distal contour ends supply the pair.
An optional full-FK camera prior ranks image-supported pairs probabilistically;
it never supplies the endpoint coordinates. This module has no device IO.
"""
from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from .letterbox_geometry import letterbox_geometry


SOURCE = 'column-rgb-jaw-contour'
# 99% contour of a two-dimensional Gaussian model. This is a consistency
# check under explicit camera assumptions, not calibrated correctness.
CENTER_MAHALANOBIS_SQ_LIMIT = float(-2. * np.log(.01))
RELATIVE_MAHALANOBIS_SQ_LIMIT = CENTER_MAHALANOBIS_SQ_LIMIT


def _empty(reason, **metadata):
    return dict(available=False, fm_px=None, reason=reason, source=SOURCE,
                accuracy_verified=False, physical_identity_verified=False,
                fk_used=False, proposal_used_as_endpoint=False, unordered=True,
                ordered_identity_available=False, **metadata)


def _attached_root_support(points, axis, xs, ys, pale_regions, pale_labels, scale,
                           pixel_cache):
    """Return root-direction hypotheses from attached pale components.

    A bright tabletop can satisfy the plastic colour predicate alongside a
    blade, even while connected to the housing. Only the proximal portion of
    that component supplies root support. Support at both ends is retained as
    two explicitly ambiguous directions so a second, unambiguous blade can
    resolve the orientation during pair geometry checks.
    """
    projection = points @ axis
    normal = np.array([-axis[1], axis[0]])
    origin = points[0] - np.array([xs[0], ys[0]]) * scale
    support = []
    for label, _center in pale_regions:
        if label not in pixel_cache:
            pixel_cache[label] = np.nonzero(pale_labels == label)
        py, px = pixel_cache[label]
        pale_points = np.column_stack((px, py)) * scale + origin
        pale_projection = pale_points @ axis
        pale_normal = pale_points @ normal
        # A housing may share one connected pale component with a large
        # tabletop. Its global centroid says nothing about which blade end it
        # touches. Inspect the local cap at each end in blade coordinates.
        for side in (-1, 1):
            end = projection.min() if side < 0 else projection.max()
            cap_points = points[(projection - end) * side >= -3.]
            cap_normal = cap_points @ normal
            outward = (pale_projection - end) * side
            local_cap = ((outward >= -3.) & (outward <= 12.)
                         & (pale_normal >= cap_normal.min() - 2.)
                         & (pale_normal <= cap_normal.max() + 2.))
            # Side-only pale streaks do not establish a root. The local
            # housing must visibly extend beyond the blade end itself.
            if np.count_nonzero(local_cap & (outward >= 0.)) * scale**2 < 3.:
                continue
            cap = np.ones_like(pale_labels, dtype=np.uint8)
            cap[py[local_cap], px[local_cap]] = 0
            distance = cv2.distanceTransform(cap, cv2.DIST_L2, 5) * scale
            roots = points[distance[ys, xs] <= 3.]
            if len(roots) * scale**2 < 3:
                continue
            if float((points.mean(0) - roots.mean(0)) @ axis) * -side < 3:
                continue
            support.append((side, label, distance))
    if not support:
        return []
    sides = {side for side, _, _ in support}
    ambiguous = len(sides) > 1
    hypotheses = []
    for side in sorted(sides):
        entries = [entry for entry in support if entry[0] == side]
        distance = np.minimum.reduce([entry[2] for entry in entries])
        roots = points[distance[ys, xs] <= 3.]
        if len(roots) * scale**2 < 3:
            continue
        hypotheses.append(dict(
            roots=roots,
            distance=distance,
            labels=[entry[1] for entry in entries],
            side=side,
            ambiguous=ambiguous,
        ))
    return hypotheses


def _overexposed_full_frame_pair(image, proposal, *, image_space, scale, pad,
                                 content, pose_prior=None, temporal_prior=None):
    """Find a conservative pair when the legacy ROI misses an overexposed jaw.

    The live Column exposure can wash the pale housing into the tabletop while
    retaining cool, elongated blade ribs.  This fallback searches the complete
    raw exposure and uses the learned proposal only to rank geometrically
    supported pairs.  It is deliberately raw-only and diagnostic: no fixed
    pixel, FK point, or proposal coordinate can become an endpoint.
    """
    if image_space != 'raw' or image.shape[1] < 400 or image.shape[0] < 300:
        return None
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x0, y0 = 0, 0
    x1, y1 = rgb.shape[1], rgb.shape[0]
    content_pixels = gray[y0:y1, x0:x1]
    # A normal synthetic/legacy test frame does not have this clipped white
    # exposure signature.  Keeping the gate raw-only prevents the fallback from
    # turning arbitrary blue objects into a second detector in old tests.
    bright_fraction = float(np.mean(content_pixels >= 235))
    if bright_fraction < 0.015:
        return None

    channels = rgb.astype(np.float32)
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    delta = blue - red
    dark = gray <= 205
    finite_delta = delta[dark]
    if finite_delta.size == 0:
        return None
    # The threshold adapts to exposure but remains bounded away from warm wood.
    cool_delta = float(np.clip(np.percentile(finite_delta, 55.), 4., 12.))
    cool = ((delta >= cool_delta) & (green - red >= -10.) & dark).astype(np.uint8)
    cool = cv2.morphologyEx(cool, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, areas, _ = cv2.connectedComponentsWithStats(cool, 8)
    content_area = float(rgb.shape[0] * rgb.shape[1])
    pale = ((gray >= 170) & (delta >= -5.) & (green - red >= -10.)).astype(np.uint8)
    yy, xx = np.indices(gray.shape)
    candidates = []

    for label in range(1, count):
        x, y, width, height, area = labels_stats = areas[label, cv2.CC_STAT_LEFT], areas[label, cv2.CC_STAT_TOP], areas[label, cv2.CC_STAT_WIDTH], areas[label, cv2.CC_STAT_HEIGHT], areas[label, cv2.CC_STAT_AREA]
        area_lb = float(area) * scale ** 2
        width_lb, height_lb = float(width) * scale, float(height) * scale
        # Drop the connected warm/cool background and tiny rib fragments.
        if (area_lb < 35. or area_lb > .02 * content_area * scale ** 2
                or width_lb > 70. or height_lb > 130. or height_lb < 18.):
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float64)
        if len(points) < 30:
            continue
        variance, vectors = np.linalg.eigh(np.cov(points.T))
        elongation = float(variance[-1] / max(variance[0], 1e-6))
        axis = vectors[:, -1]
        length = float(np.ptp(points @ axis)) * scale
        if elongation < 4. or not 20. <= length <= 100.:
            continue
        projection = points @ axis
        band = max(3., .06 * float(np.ptp(projection)))
        endpoint_scores = []
        endpoint_points = []
        for side in (-1, 1):
            threshold = projection.min() + band if side < 0 else projection.max() - band
            endpoint = points[projection <= threshold] if side < 0 else points[projection >= threshold]
            center = endpoint.mean(0)
            radius = max(8., 5. / max(scale, 1e-6))
            disk = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
            pale_score = float(pale[disk].mean()) if disk.any() else 0.
            cool_pale = pale.astype(bool) & (delta >= 0.)
            cool_score = float(cool_pale[disk].mean()) if disk.any() else 0.
            endpoint_scores.append(.55 * pale_score + .45 * cool_score)
            endpoint_points.append((endpoint, center))
        root_index = int(np.argmax(endpoint_scores))
        distal_index = 1 - root_index
        root_score, distal_score = endpoint_scores[root_index], endpoint_scores[distal_index]
        if root_score < .35 or root_score - distal_score < .035:
            continue
        # Reorient from the supported pale housing toward the distal blade end.
        if root_index == 1:
            axis = -axis
        projection = points @ axis
        distal_band = max(2., .04 * float(np.ptp(projection)))
        distal = points[projection >= projection.max() - distal_band]
        root = points[projection <= projection.min() + band]
        tip = distal.mean(0)
        root_center = root.mean(0)
        tip_lb = tip * scale + pad
        root_lb = root_center * scale + pad
        edge_distance = min(tip_lb[0] - content[0], tip_lb[1] - content[1],
                            content[2] - scale - tip_lb[0],
                            content[3] - scale - tip_lb[1])
        if edge_distance < 1.:
            continue
        candidates.append(dict(label=int(label), tip_px=tip_lb.tolist(),
            root_px=root_lb.tolist(), distal_axis=(axis / np.linalg.norm(axis)).tolist(),
            length_px=length, area_px=area_lb, elongation=elongation,
            root_support_score=root_score, distal_support_score=distal_score,
            sensor_edge_distance_px=float(edge_distance), near_sensor_edge=bool(edge_distance < 1.5),
            raw_tip_px=tip.tolist(), raw_root_px=root_center.tolist()))

    if len(candidates) < 2:
        return None
    span = float(np.linalg.norm(proposal[0] - proposal[1]))
    center = np.asarray(proposal, dtype=np.float64).mean(0)
    pairs = []
    for first, second in combinations(range(len(candidates)), 2):
        a, b = candidates[first], candidates[second]
        aa, ab = np.asarray(a['distal_axis']), np.asarray(b['distal_axis'])
        alignment = abs(float(aa @ ab))
        if alignment < .85:
            continue
        pa, pb = np.asarray(a['tip_px']), np.asarray(b['tip_px'])
        separation = float(np.linalg.norm(pa - pb))
        ratio = a['length_px'] / max(b['length_px'], 1e-6)
        if not 15. <= separation <= 80. or not .45 <= ratio <= 2.2:
            continue
        direction = aa + ab if float(aa @ ab) >= 0. else aa - ab
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        along = abs(float((pa - pb) @ direction)) / max(min(a['length_px'], b['length_px']), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        transverse = abs(float((pa - pb) @ normal))
        if along > .45 or transverse < 8.:
            continue
        pair_center = (pa + pb) / 2.
        center_distance = float(np.linalg.norm(pair_center - center))
        if center_distance > max(48., 3. * max(span, 1.)):
            continue
        root_distance = float(np.linalg.norm(np.asarray(a['root_px']) - np.asarray(b['root_px'])))
        if root_distance < 18.:
            continue
        # The proposal is a same-frame visual association, never an endpoint.
        # In an overexposed frame a cable/edge component can form a nearly
        # parallel second pair.  Keep the geometric gate fail-closed, but
        # retain a bounded association distance so a genuinely near-tied pair
        # can be selected when it is materially closer to the current visual
        # proposal.  If the proposal cannot disambiguate, the ambiguity gate
        # below still rejects the frame.
        proposal_association = min(
            float(np.linalg.norm(np.asarray([a['tip_px'], b['tip_px']]) - proposal, axis=1).sum()),
            float(np.linalg.norm(np.asarray([b['tip_px'], a['tip_px']]) - proposal, axis=1).sum()),
        )
        score = (2.5 * alignment - 1.3 * along - abs(np.log(ratio))
                 - .45 * center_distance / max(48., 3. * max(span, 1.))
                 + .5 * (a['root_support_score'] + b['root_support_score']))
        pose_match = None
        if pose_prior is not None:
            pose_match = pose_prior.score_pair([pa, pb])
            if (pose_match['center_mahalanobis_sq'] > CENTER_MAHALANOBIS_SQ_LIMIT
                    or pose_match['relative_mahalanobis_sq'] > RELATIVE_MAHALANOBIS_SQ_LIMIT):
                continue
            score += pose_match['log_likelihood']
        temporal_match = None
        if temporal_prior is not None:
            temporal_match = temporal_prior.score_pair([pa, pb])
            score += temporal_match['score_bonus']
        pairs.append(dict(first=first, second=second, score=float(score),
                          alignment=alignment, separation_px=separation,
                          transverse_separation_px=transverse, along_axis_ratio=along,
                          center_distance_px=center_distance, root_distance_px=root_distance,
                          proposal_association_px=proposal_association,
                          pose_match=pose_match, temporal_match=temporal_match))
    if not pairs:
        return None
    pairs.sort(key=lambda row: row['score'], reverse=True)
    tie_break = None
    if len(pairs) > 1 and pairs[0]['score'] - pairs[1]['score'] < .15:
        # A current visual proposal may associate the intended pair without
        # supplying its coordinates.  Require a meaningful margin in that
        # association before accepting a geometric near-tie; otherwise keep
        # the detector unavailable rather than inventing an endpoint pair.
        association_gap = (pairs[1]['proposal_association_px']
                            - pairs[0]['proposal_association_px'])
        association_gate = max(4., .25 * span)
        if abs(association_gap) < association_gate:
            return None
        tie_break = dict(kind='proposal_association_only',
                         association_gap_px=float(abs(association_gap)),
                         minimum_gap_px=float(association_gate))
        if pairs[1]['proposal_association_px'] < pairs[0]['proposal_association_px']:
            pairs[0], pairs[1] = pairs[1], pairs[0]
    selected = pairs[0]
    ids = [selected['first'], selected['second']]
    pair = np.asarray([candidates[index]['tip_px'] for index in ids], dtype=np.float64)
    pose_match = selected['pose_match']
    swap = (pose_match['swapped'] if pose_match is not None else
            np.linalg.norm(pair[::-1] - proposal, axis=1).sum()
            < np.linalg.norm(pair - proposal, axis=1).sum())
    if swap:
        pair, ids = pair[::-1].copy(), ids[::-1]
    return dict(available=True, fm_px=pair.tolist(), reason=None, source=SOURCE,
        accuracy_verified=False, physical_identity_verified=False, fk_used=False,
        unordered=True, ordered_identity_available=False, proposal_used_as_endpoint=False,
        identity_source=('full-q camera likelihood association only' if pose_match is not None
                         else 'current overexposed RGB pair association only'),
        pose_identity_probability=(pose_match['identity_probability']
                                   if pose_match is not None else None),
        fallback_mode='overexposed-full-frame-cool-component-consensus',
        search_roi_raw=[0, 0, int(rgb.shape[1]), int(rgb.shape[0])],
        image_space=image_space, image_shape=list(rgb.shape), candidates=candidates,
        pair_candidates=pairs, selected_component_indices=ids,
        proposal_pair_px=np.asarray(proposal).tolist(), proposal_span_px=span,
        proposal_used_for_ranking_only=True, bright_fraction=bright_fraction,
        cool_delta=float(cool_delta), near_sensor_edge=any(candidates[index]['near_sensor_edge'] for index in ids),
        pose_prior_used=pose_prior is not None,
        temporal_prior_used=temporal_prior is not None,
        temporal_prior_source=None if temporal_prior is None else temporal_prior.as_dict()['source'],
        geometry_ranking_source='full-q-camera-distribution' if pose_prior is not None else None,
        pair_tie_break=tie_break)


def locate_column_jaw_tips(image, proposal_pair_px, *, image_space='raw', pose_prior=None,
                          temporal_prior=None):
    """Return image-supported endpoints in letterbox224 coordinates.

``proposal_pair_px`` is a current-frame visual proposal in letterbox224. The
RGB may be its original raw exposure or the explicit 224 canvas. Gray/blue
jaw color, pale root attachment, shape, and a distinct pair are all required;
missing evidence never falls back to the proposal or mechanical projection.
"""
    image = np.asarray(image)
    try:
        proposal = np.asarray(proposal_pair_px, dtype=np.float64)
    except (TypeError, ValueError):
        return _empty('missing finite current visual proposal')
    if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 32):
        return _empty('invalid RGB image')
    if proposal.shape != (2, 2) or not np.isfinite(proposal).all():
        return _empty('missing finite current visual proposal')
    if image_space == 'letterbox224':
        if image.shape != (224, 224, 3):
            return _empty('letterbox RGB shape differs')
        scale, pad = 1., np.array([0., 0.])
        content = np.array([0., 28., 224., 196.])
    elif image_space == 'raw':
        geometry = letterbox_geometry(image.shape[1], image.shape[0])
        if geometry.content_box() != (0, 28, 224, 196):
            return _empty('Column raw image is not the declared 4:3 exposure')
        scale, pad = geometry.scale, np.array(geometry.padding, dtype=np.float64)
        content = np.array(geometry.content_box(), dtype=np.float64)
    else:
        return _empty('unknown image coordinate space')
    span = float(np.linalg.norm(proposal[0]-proposal[1]))
    if (pose_prior is None and (not 4 <= span <= 80
            or (proposal < content[:2]).any() or (proposal >= content[2:]).any())):
        return _empty('visual proposal is collapsed or outside image support')
    center = proposal.mean(0)
    # A narrow opening still needs enough context to include the housing while
    # the proposal remains close to the distal tips required below.
    radius = max(32., 2.*span)
    lower = np.maximum(center-radius, content[:2])
    upper = np.minimum(center+radius, content[2:])
    if pose_prior is not None:
        # Camera uncertainty expands the region jointly for both tips. The
        # RGB proposal supplies only a weak common image location; its opening
        # is not a required span when full joint geometry is available.
        pose_record = pose_prior.as_dict()
        bounds = np.asarray(pose_record['fg_bounds_px'], np.float64)
        # Tip intervals exclude the proximal housing. Preserve image context
        # around them even when a foreshortened opening makes the tip interval
        # small; this expands only the search, never the accepted endpoints.
        housing_context = 32.
        lower = np.maximum(np.minimum(lower, bounds[:, 0].min(0) - housing_context), content[:2])
        upper = np.minimum(np.maximum(upper, bounds[:, 1].max(0) + housing_context), content[2:])
    x0, y0 = np.floor((lower-pad)/scale).astype(int)
    x1, y1 = np.ceil((upper-pad)/scale).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if min(crop.shape[:2]) < 8:
        return _empty('visual search region is too small')
    red, green, blue = crop[..., 0], crop[..., 1], crop[..., 2]
    # Current SO-101's gray blades carry a blue cast. Warm fish/wood/cardboard
    # and pale plastic cannot enter the blade support under this contract.
    # Require a little more blue chroma than the cool tabletop streaks beside
    # the blades. Filtering the binary silhouette instead erases thin tips.
    metal = ((blue-red >= 28) & (green-red >= 12) & (blue-green >= 6)
             & (red >= 20) & (crop.max(-1) <= 220)).astype(np.uint8)
    pale = ((red >= 140) & (green >= 190) & (blue >= 205)
            & (blue >= red-10)).astype(np.uint8)
    if not pale.any():
        fallback = _overexposed_full_frame_pair(
            image, proposal, image_space=image_space, scale=scale, pad=pad,
            content=content, pose_prior=pose_prior, temporal_prior=temporal_prior)
        if fallback is not None:
            return fallback
        return _empty('no pale gripper-root support', search_roi_raw=[x0, y0, x1, y1])
    # Blue-tinted pale plastic also satisfies the metal colour inequalities.
    # Exclude it before component labelling: a shared housing must not bridge
    # the two blades into one component with an invalid root-to-tip axis.
    metal[pale.astype(bool)] = 0
    pale_count, pale_labels, pale_areas, pale_centers = cv2.connectedComponentsWithStats(pale, 8)
    pale_regions = [(label, (pale_centers[label]+[x0, y0])*scale+pad)
                    for label in range(1, pale_count)
                    if pale_areas[label, cv2.CC_STAT_AREA]*scale**2 >= 3]
    root_pixel_cache = {}
    corners = cv2.cornerMinEigenVal(cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                   blockSize=3, ksize=3)
    yy, xx = np.indices(crop.shape[:2])
    coordinate = np.stack((xx+x0, yy+y0), -1)*scale+pad
    count, labels, areas, _ = cv2.connectedComponentsWithStats(metal, 8)
    candidates = []
    roi_cut_rejections = 0
    for label in range(1, count):
        if float(areas[label, cv2.CC_STAT_AREA])*scale**2 < 25:
            continue
        ys, xs = np.nonzero(labels == label)
        pixels = np.column_stack((xs, ys)).astype(np.float64)
        points = (pixels+np.array([x0, y0]))*scale+pad
        variance, vectors = np.linalg.eigh(np.cov(points.T))
        elongation = float(variance[-1]/max(variance[0], 1e-6))
        if elongation < 2.5:
            continue
        axis = vectors[:, -1]
        root_hypotheses = _attached_root_support(
            points, axis, xs, ys, pale_regions, pale_labels, scale,
            root_pixel_cache)
        if not root_hypotheses:
            continue
        for root_support in root_hypotheses:
            roots = root_support['roots']
            root_distance = root_support['distance']
            distal_axis = axis.copy()
            away = points.mean(0)-roots.mean(0)
            if abs(float(away@distal_axis)) < 3:
                continue
            distal_axis *= 1 if float(away@distal_axis) > 0 else -1
            projection = points@distal_axis
            length = float(np.ptp(projection))
            if not 12 <= length <= 90:
                continue
            terminal_projection_floor = projection.max()-.75
            terminal = projection >= terminal_projection_floor
            # Inspect the contour before corner refinement: a corner a few pixels
            # inside a truncated blade must not conceal its distal ROI cut.
            tx, ty = xs[terminal], ys[terminal]
            content_y0 = 28 if image_space == 'letterbox224' else 0
            content_y1 = 196 if image_space == 'letterbox224' else image.shape[0]
            terminal_cut = ((x0 > 0 and bool((tx == 0).any()))
                or (x1 < image.shape[1] and bool((tx == crop.shape[1]-1).any()))
                or (y0 > content_y0 and bool((ty == 0).any()))
                or (y1 < content_y1 and bool((ty == crop.shape[0]-1).any())))
            if terminal_cut:
                roi_cut_rejections += 1
                continue
            coarse_tip = points[terminal].mean(0)
            # Refine only in the distal contour band. Strong gripper ribs inside
            # the 2.5 px search disk must not pull the endpoint back toward its root.
            # The RGB corner can extend past the colored ridge; no offset is added.
            nearby = (np.linalg.norm(coordinate-coarse_tip, axis=-1) <= 2.5)
            nearby &= (coordinate@distal_axis >= terminal_projection_floor)
            nearby &= (root_distance >= 6.)
            scores = np.where(nearby, corners, -1.)
            cy, cx = np.unravel_index(scores.argmax(), scores.shape)
            corner_score = float(scores[cy, cx])
            if corner_score < 1e-5:
                continue
            tip = coordinate[cy, cx]
            raw_tip = (tip-pad)/scale
            cx, cy = np.rint(raw_tip-np.array([x0, y0])).astype(int)
            cx, cy = np.clip([cx, cy], [0, 0], [crop.shape[1]-1, crop.shape[0]-1])
            tip_root_distance = float(root_distance[cy, cx])
            if tip_root_distance < 6:
                continue
            # A search-ROI edge is an unknown continuation, not a visible tip.
            # Actual sensor borders are retained explicitly for review below.
            margin = 1./scale
            internal_edge = ((x0 > 0 and raw_tip[0]-x0 < margin)
                or (x1 < image.shape[1] and x1-1-raw_tip[0] < margin)
                or (y0 > (28 if image_space == 'letterbox224' else 0) and raw_tip[1]-y0 < margin)
                or (y1 < (196 if image_space == 'letterbox224' else image.shape[0]) and y1-1-raw_tip[1] < margin))
            if internal_edge:
                continue
            edge = min(tip[0]-content[0], tip[1]-content[1], content[2]-scale-tip[0], content[3]-scale-tip[1])
            candidates.append(dict(label=label, tip_px=tip.tolist(), root_px=roots.mean(0).tolist(),
                distal_axis=distal_axis.tolist(), length_px=length, area_px=float(len(points)*scale**2),
                elongation=elongation, root_support_pixels=int(len(roots)),
                root_pale_component_labels=root_support['labels'],
                root_support_ambiguous=bool(root_support['ambiguous']),
                root_hypothesis_side=int(root_support['side']),
                root_support_source='proximal pixels of attached pale components beyond one blade end',
                tip_root_distance_px=tip_root_distance, sensor_edge_distance_px=float(edge),
                coarse_tip_px=coarse_tip.tolist(), corner_score=corner_score,
                corner_refinement_delta_px=(tip-coarse_tip).tolist(),
                corner_distal_cap_margin_px=float(tip@distal_axis-terminal_projection_floor),
                near_sensor_edge=bool(edge < 1.5)))
    minimum_span = max(4., .5 * span) if pose_prior is None else 0.
    span_rejections = []
    metadata = dict(search_roi_raw=[int(x0), int(y0), int(x1), int(y1)],
        image_space=image_space, image_shape=list(image.shape), candidates=candidates,
        roi_cut_rejections=roi_cut_rejections,
        appearance_contract='cool gray blades attached to pale gripper housings',
        proposal_pair_px=proposal.tolist(), search_radius_px=radius,
        proposal_span_px=span, minimum_pair_span_px=minimum_span,
        proposal_span_used_for_rejection_only=pose_prior is None,
        pose_prior_used=pose_prior is not None,
        temporal_prior_used=temporal_prior is not None,
        temporal_prior_source=None if temporal_prior is None else temporal_prior.as_dict()['source'],
        geometry_ranking_source='full-q-camera-distribution' if pose_prior is not None else None,
        span_consistency_rejections=span_rejections,
        proposal_association_rejections=[],
        pose_center_consistency_rejections=[],
        pose_relative_consistency_rejections=[],
        overlap_unresolved_pairs=[],
        pose_overlap_affinity=(pose_record['overlap_affinity']
                               if pose_prior is not None else None),
        pose_center_mahalanobis_sq_limit=(CENTER_MAHALANOBIS_SQ_LIMIT
                                          if pose_prior is not None else None),
        pose_relative_mahalanobis_sq_limit=(RELATIVE_MAHALANOBIS_SQ_LIMIT
                                            if pose_prior is not None else None))
    pairs = []
    for first, second in combinations(range(len(candidates)), 2):
        a, b = candidates[first], candidates[second]
        if a['label'] == b['label']:
            continue
        ambiguous_count = (int(a['root_support_ambiguous'])
                           + int(b['root_support_ambiguous']))
        if ambiguous_count > 1:
            continue
        pa, pb = np.asarray(a['tip_px']), np.asarray(b['tip_px'])
        aa, ab = np.asarray(a['distal_axis']), np.asarray(b['distal_axis'])
        alignment = float(aa@ab)
        separation = float(np.linalg.norm(pa-pb))
        if separation < scale:
            metadata['overlap_unresolved_pairs'].append(dict(
                first=first, second=second, separation_px=separation))
            continue
        ratio = a['length_px']/b['length_px']
        if pose_prior is None and (
                alignment < .7 or not 4 <= separation <= 80 or not .4 <= ratio <= 2.5):
            continue
        direction = aa+ab
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        along = abs(float((pa-pb)@direction))/min(a['length_px'], b['length_px'])
        normal = np.array([-direction[1], direction[0]])
        transverse = abs(float((pa-pb)@normal))
        # Fragments from one blade can be separated along its axis while
        # still occupying the same jaw line.  Two jaws must be distinct in
        # the direction normal to their shared blade axis.
        if pose_prior is None and (along > .65 or transverse < 4.):
            continue
        # One blade may have pale support at both ends, but only a strongly
        # parallel, co-terminal unambiguous blade may resolve its direction.
        if ambiguous_count and (alignment < .9 or along > .3):
            continue
        # The legacy span includes the same exposure's opening prior, but is
        # not a calibrated width label. Use only a broad lower-bound veto:
        # two nearby fragments cannot represent a much wider proposed jaw.
        # Never stretch RGB endpoints or synthesize a pair from this prior.
        span_ratio = separation / span if span > 1e-12 else None
        if separation < minimum_span:
            span_rejections.append(dict(first=first, second=second,
                separation_px=separation, span_ratio_to_proposal=span_ratio))
            continue
        pair_center_distance = float(np.linalg.norm((pa+pb)/2-center))
        proposal_association = min(
            float(np.linalg.norm(np.asarray([pa, pb])-proposal, axis=1).sum()),
            float(np.linalg.norm(np.asarray([pb, pa])-proposal, axis=1).sum()),
        )
        center_gate = max(12., span)
        association_gate = max(32., 1.5*span)
        if (pair_center_distance > center_gate
                or (span >= 4. and proposal_association > association_gate)):
            metadata['proposal_association_rejections'].append(dict(
                first=first, second=second,
                midpoint_distance_px=pair_center_distance,
                maximum_midpoint_distance_px=center_gate,
                endpoint_association_px=proposal_association,
                maximum_endpoint_association_px=association_gate))
            continue
        score = (2.*alignment-abs(np.log(ratio))-along
                 -.25*np.linalg.norm((pa+pb)/2-center)/radius)
        pose_match = None
        if pose_prior is not None:
            pose_match = pose_prior.score_pair([pa, pb])
            if pose_match['center_mahalanobis_sq'] > CENTER_MAHALANOBIS_SQ_LIMIT:
                metadata['pose_center_consistency_rejections'].append(dict(
                    first=first, second=second,
                    center_mahalanobis_sq=pose_match['center_mahalanobis_sq'],
                    maximum_center_mahalanobis_sq=CENTER_MAHALANOBIS_SQ_LIMIT))
                continue
            if pose_match['relative_mahalanobis_sq'] > RELATIVE_MAHALANOBIS_SQ_LIMIT:
                metadata['pose_relative_consistency_rejections'].append(dict(
                    first=first, second=second,
                    relative_mahalanobis_sq=pose_match['relative_mahalanobis_sq'],
                    maximum_relative_mahalanobis_sq=RELATIVE_MAHALANOBIS_SQ_LIMIT))
                continue
            score += pose_match['log_likelihood']
        temporal_match = None
        if temporal_prior is not None:
            temporal_match = temporal_prior.score_pair([pa, pb])
            score += temporal_match['score_bonus']
        pairs.append(dict(first=first, second=second, score=float(score),
                          transverse_separation_px=transverse,
                          span_ratio_to_proposal=span_ratio,
                          root_ambiguous_count=ambiguous_count,
                          midpoint_distance_px=pair_center_distance,
                          proposal_association_px=proposal_association,
                          pose_match=pose_match, temporal_match=temporal_match))
    if not pairs:
        fallback = _overexposed_full_frame_pair(
            image, proposal, image_space=image_space, scale=scale, pad=pad,
            content=content, pose_prior=pose_prior, temporal_prior=temporal_prior)
        if fallback is not None:
            return fallback
        reason = ('image jaw pair span is less than half the same-frame proposal'
                  if span_rejections else 'two distinct supported jaw contours were not found')
        return _empty(reason, **metadata)
    pairs.sort(key=lambda item: item['score'], reverse=True)
    margin = None if len(pairs) == 1 else pairs[0]['score']-pairs[1]['score']
    metadata.update(pair_candidates=pairs, pair_score_margin=margin)
    if margin is not None and margin < .15:
        fallback = _overexposed_full_frame_pair(
            image, proposal, image_space=image_space, scale=scale, pad=pad,
            content=content, pose_prior=pose_prior, temporal_prior=temporal_prior)
        if fallback is not None:
            return fallback
        return _empty('ambiguous image-supported jaw pairing', **metadata)
    selected = pairs[0]
    ids = [selected['first'], selected['second']]
    pair = np.asarray([candidates[index]['tip_px'] for index in ids])
    # Preserve probabilistic FK identity association while keeping the
    # returned coordinates on the observed RGB contours. The caller can
    # re-score this ordering without a second reversal.
    pose_match = selected['pose_match']
    swap = (pose_match['swapped'] if pose_match is not None else
            np.linalg.norm(pair[::-1]-proposal, axis=1).sum()
            < np.linalg.norm(pair-proposal, axis=1).sum())
    if swap:
        pair, ids = pair[::-1].copy(), ids[::-1]
    return dict(available=True, fm_px=pair.tolist(), reason=None, source=SOURCE,
        accuracy_verified=False, physical_identity_verified=False, fk_used=False,
        unordered=True, ordered_identity_available=False,
        proposal_used_as_endpoint=False,
        identity_source=('full-q camera likelihood association only' if pose_match is not None
                         else 'current visual proposal association only'),
        pose_identity_probability=(pose_match['identity_probability']
                                   if pose_match is not None else None),
        near_sensor_edge=any(candidates[index]['near_sensor_edge'] for index in ids),
        selected_component_indices=ids, **metadata)
