import numpy as np
import pytest
import torch

import comfy_extras.nodes_mediapipe as mediapipe_nodes


def _edges(vertices):
    return frozenset((vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))


def _face_data():
    canonical = np.array([
        [-1.0, 0.0], [-0.7, -0.7], [0.0, -1.0], [0.7, -0.7],
        [1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [-0.7, 0.7],
        [-0.65, -0.25], [-0.45, -0.35], [-0.25, -0.25], [-0.45, -0.15],
        [0.25, -0.25], [0.45, -0.35], [0.65, -0.25], [0.45, -0.15],
        [-0.7, -0.55], [-0.45, -0.65], [-0.2, -0.55], [-0.45, -0.5],
        [0.2, -0.55], [0.45, -0.65], [0.7, -0.55], [0.45, -0.5],
        [-0.4, 0.45], [0.0, 0.35], [0.4, 0.45], [0.0, 0.65],
        [-0.2, -0.1], [0.0, -0.3], [0.2, -0.1], [0.0, 0.25],
    ], dtype=np.float32)
    canonical = np.column_stack([canonical, np.zeros(canonical.shape[0], dtype=np.float32)])
    connection_sets = {
        "face_oval": _edges(list(range(8))),
        "left_eye": _edges(list(range(8, 12))),
        "right_eye": _edges(list(range(12, 16))),
        "left_eyebrow": _edges(list(range(16, 20))),
        "right_eyebrow": _edges(list(range(20, 24))),
        "lips": _edges(list(range(24, 28))),
        "nose": _edges(list(range(28, 32))),
    }
    return canonical, connection_sets


def _face(canonical, center, scale):
    landmarks = canonical[:, :2] * scale + np.asarray(center, dtype=np.float32)
    minimum = landmarks.min(axis=0)
    maximum = landmarks.max(axis=0)
    return {
        "bbox_xyxy": np.array([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float32),
        "landmarks_xy": landmarks,
    }


def test_thin_plate_spline_reproduces_affine_mapping():
    control = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])
    values = control @ np.array([[1.2, 0.3], [-0.2, 0.8]]) + np.array([0.4, -0.7])
    weights, affine = mediapipe_nodes._solve_thin_plate_spline(control, values)
    points = np.array([[-0.5, 0.25], [0.2, -0.8], [0.75, 0.6]])

    actual = mediapipe_nodes._evaluate_thin_plate_spline(points, control, weights, affine)
    expected = points @ np.array([[1.2, 0.3], [-0.2, 0.8]]) + np.array([0.4, -0.7])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_transfer_anchors_use_feature_centers_only():
    canonical, connection_sets = _face_data()
    face = _face(canonical, (20, 20), 12)

    anchors = mediapipe_nodes._face_transfer_anchors(face, connection_sets)
    lip_vertices = mediapipe_nodes._edge_vertices(connection_sets["lips"])

    assert anchors.shape == (7, 2)
    np.testing.assert_allclose(anchors[4], face["landmarks_xy"][lip_vertices].mean(axis=0))


def test_transfer_mask_has_soft_target_boundary():
    canonical, connection_sets = _face_data()
    face = _face(canonical, (20, 20), 12)

    mask, _, _ = mediapipe_nodes._face_transfer_mask(48, 48, face, connection_sets)

    assert mask.max() == 1.0
    assert np.any((mask > 0.0) & (mask < 1.0))


def test_transfer_mask_excludes_forehead_color_edges():
    canonical, connection_sets = _face_data()
    face = _face(canonical, (20, 20), 12)
    image = torch.full((1, 48, 48, 3), 0.6)
    image[:, 8:16] = 0.0

    geometric, left, top = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0,
    )
    edge_aware, _, _ = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0, image=image,
    )

    assert edge_aware.sum() < geometric.sum()
    assert edge_aware[12 - top, 15 - left] < geometric[12 - top, 15 - left]


def test_transfer_mask_preserves_makeup_near_eye():
    canonical, connection_sets = _face_data()
    face = _face(canonical, (20, 20), 12)
    image = torch.full((1, 48, 48, 3), 0.6)
    image[:, 13:22, 7:18] = 0.0

    geometric, left, top = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0,
    )
    edge_aware, _, _ = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0, image=image,
    )

    assert edge_aware[17 - top, 9 - left] < geometric[17 - top, 9 - left]
    assert edge_aware[17 - top, 15 - left] == geometric[17 - top, 15 - left]


def test_transfer_mask_preserves_smooth_cheek_highlight():
    canonical, connection_sets = _face_data()
    face = _face(canonical, (20, 20), 12)
    _, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    highlight = 0.45 * torch.sigmoid((14.0 - xx) / 3.0)
    image = torch.full((1, 48, 48, 3), 0.4) + highlight[None, :, :, None]

    geometric, _, _ = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0,
    )
    edge_aware, _, _ = mediapipe_nodes._face_transfer_mask(
        48, 48, face, connection_sets, forehead_coverage=1.0, image=image,
    )

    np.testing.assert_allclose(edge_aware, geometric)


def test_face_matching_targets_large_similar_color_regions():
    source = torch.empty((32, 32, 3))
    source[:] = torch.tensor([0.70, 0.45, 0.40])
    target = torch.empty_like(source)
    target[:] = torch.tensor([0.76, 0.60, 0.46])
    source[12:16, 8:12] = 0.05
    target[12:16, 8:12] = 0.90
    mask = torch.ones((32, 32))

    output = mediapipe_nodes._harmonize_face(source, target, mask, 1.0, 1.0)

    assert torch.linalg.vector_norm(output[4, 4] - target[4, 4]) < torch.linalg.vector_norm(source[4, 4] - target[4, 4])
    torch.testing.assert_close(output[13, 9], source[13, 9], atol=1e-4, rtol=0)


def test_color_and_lighting_matching_are_independent():
    source = torch.empty((32, 32, 3))
    source[:] = torch.tensor([0.55, 0.35, 0.30])
    target = torch.empty_like(source)
    target[:] = torch.tensor([0.70, 0.50, 0.38])
    mask = torch.ones((32, 32))

    color = mediapipe_nodes._harmonize_face(source, target, mask, 1.0, 0.0)
    lighting = mediapipe_nodes._harmonize_face(source, target, mask, 0.0, 1.0)
    source_lab = mediapipe_nodes._rgb_to_lab(source.numpy())[0, 0]
    color_lab = mediapipe_nodes._rgb_to_lab(color.numpy())[0, 0]
    lighting_lab = mediapipe_nodes._rgb_to_lab(lighting.numpy())[0, 0]

    assert abs(color_lab[0] - source_lab[0]) < 1e-3
    assert np.linalg.norm(color_lab[1:] - source_lab[1:]) > 1.0
    assert abs(lighting_lab[0] - source_lab[0]) > 1.0
    np.testing.assert_allclose(lighting_lab[1:], source_lab[1:], atol=1e-3)


def test_large_regions_remove_small_components():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 2:12] = True
    mask[16:18, 16:18] = True

    regions = mediapipe_nodes._large_regions(mask, minimum_area=10)

    assert regions[5, 5]
    assert not regions[16, 16]


def test_largest_face_uses_landmark_area():
    faces = [
        {"bbox_xyxy": np.array([0, 0, 10, 20])},
        {"bbox_xyxy": np.array([0, 0, 30, 8])},
    ]

    assert mediapipe_nodes._largest_face(faces) is faces[1]
    assert mediapipe_nodes._largest_face([]) is None


def test_detection_selects_largest_and_retries_missing_with_full_range():
    small = {"bbox_xyxy": np.array([0, 0, 5, 5])}
    large = {"bbox_xyxy": np.array([0, 0, 10, 10])}
    fallback = {"bbox_xyxy": np.array([0, 0, 7, 8])}

    class Model:
        def __init__(self):
            self.calls = []

        def detect_batch(self, images, num_faces, score_thresh, variant):
            self.calls.append((len(images), num_faces, score_thresh, variant))
            if variant == "short":
                return [[small, large], []]
            return [[fallback]]

    model = Model()
    detected = mediapipe_nodes._detect_largest_faces(model, torch.zeros((2, 8, 8, 3)))

    assert detected == [large, fallback]
    assert model.calls == [(2, 0, 0.5, "short"), (1, 0, 0.5, "full")]


def test_transfer_preserves_target_outside_face_region():
    canonical, connection_sets = _face_data()
    source_face = _face(canonical, (20, 20), 12)
    target_face = _face(canonical, (43, 39), 15)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    source = torch.stack([
        ((xx + yy) % 7).float() / 6.0,
        xx.float() / 63.0,
        yy.float() / 63.0,
    ], dim=-1)[None]
    target = torch.full((1, 64, 64, 3), 0.2)

    output = mediapipe_nodes._transfer_face(source, target, source_face, target_face, connection_sets)

    assert output.shape == target.shape
    assert output.dtype == target.dtype
    torch.testing.assert_close(output[0, :10], target[0, :10], rtol=0, atol=0)
    torch.testing.assert_close(output[0, :, :10], target[0, :, :10], rtol=0, atol=0)
    torch.testing.assert_close(output[0, 24, 28], target[0, 24, 28], rtol=0, atol=0)
    assert not torch.equal(output[0, 30:49, 34:53], target[0, 30:49, 34:53])


def test_node_broadcasts_single_source(monkeypatch):
    face = {"bbox_xyxy": np.array([0, 0, 10, 10])}
    monkeypatch.setattr(mediapipe_nodes, "_detect_largest_faces", lambda _model, images: [face] * images.shape[0])
    monkeypatch.setattr(
        mediapipe_nodes,
        "_transfer_face",
        lambda source, target, *_args: target + source.mean(),
    )
    model = type("Model", (), {"connection_sets": {}, "canonical_data": {"canonical_vertices": np.empty((0, 3))}})()
    source = torch.ones((1, 8, 8, 3))
    target = torch.zeros((3, 8, 8, 3))

    output = mediapipe_nodes.MediaPipeFaceTransfer.execute(model, source, target)[0]

    torch.testing.assert_close(output, torch.ones_like(target))


def test_node_rejects_incompatible_batches():
    source = torch.zeros((2, 8, 8, 3))
    target = torch.zeros((3, 8, 8, 3))

    with pytest.raises(ValueError, match="one source image"):
        mediapipe_nodes.MediaPipeFaceTransfer.execute(None, source, target)


def test_node_reports_missing_source_face(monkeypatch):
    monkeypatch.setattr(mediapipe_nodes, "_detect_largest_faces", lambda _model, images: [None] * images.shape[0])
    source = torch.zeros((1, 8, 8, 3))
    target = torch.zeros((1, 8, 8, 3))

    with pytest.raises(ValueError, match="source face in image 0"):
        mediapipe_nodes.MediaPipeFaceTransfer.execute(None, source, target)
