"""ComfyUI nodes for the pure-PyTorch MediaPipe Face Landmarker port.

Custom IO types:
  FACE_LANDMARKER  — FaceLandmarkerModel wrapper (ModelPatcher inside)
  FACE_LANDMARKS   — {"frames": List[List[face_dict]], "image_size": (H, W),
                      "connection_sets": dict[str, frozenset[(int, int)]]}
                     face_dict: bbox_xyxy, blendshapes, landmarks_xy,
                                landmarks_3d, presence, score, transformation_matrix

MediaPipeFaceLandmarker also emits the core BOUNDING_BOX type — pair with DrawBBoxes.
"""


import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw
from tqdm.auto import tqdm
from typing_extensions import override

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from comfy_extras.mediapipe.face_landmarker import FaceLandmarker
from comfy_extras.mediapipe.face_geometry import transformation_matrix_from_detection


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceLandmarksType = io.Custom("FACE_LANDMARKS")

_CANONICAL_KEYS = ("canonical_vertices", "procrustes_indices", "procrustes_weights")
_CONTOUR_PARTS = ("face_oval", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips")


class FaceLandmarkerModel:
    """Loaded FaceLandmarker variants + ModelPatcher per variant.

    Safetensors layout: `detector_short.*` / `detector_full.*` plus shared
    `mesh.*`, `blendshapes.*`, `canonical_*`, and `topology.*`.
    PReLU forces plain-nn / fp32 (manual_cast strands buffers across devices).
    """

    def __init__(self, state_dict: dict):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32

        # FACEMESH_* connection sets, embedded as int32 (N, 2) under topology.*.
        base: dict[str, frozenset] = {}
        for k in [k for k in state_dict if k.startswith("topology.")]:
            base[k[len("topology."):]] = frozenset(map(tuple, state_dict.pop(k).tolist()))
        base["contours"] = frozenset().union(*(base[p] for p in _CONTOUR_PARTS))
        base["all"] = base["contours"] | base["irises"] | base["nose"]

        self.connection_sets: dict[str, frozenset] = base
        self.canonical_data: dict[str, np.ndarray] = {k: state_dict.pop(k).numpy() for k in _CANONICAL_KEYS}

        shared = {k: v for k, v in state_dict.items() if k.startswith(("mesh.", "blendshapes."))}

        self.models: dict[str, FaceLandmarker] = {}
        self.patchers: dict[str, comfy.model_patcher.ModelPatcher] = {}
        for variant in ("short", "full"):
            prefix = f"detector_{variant}."
            sub = dict(shared)
            sub.update({f"detector.{k[len(prefix):]}": v for k, v in state_dict.items() if k.startswith(prefix)})
            fl = FaceLandmarker(device=offload_device, dtype=self.dtype, operations=None, detector_variant=variant).eval()
            fl.load_state_dict(sub, strict=False)

            self.models[variant] = fl
            self.patchers[variant] = comfy.model_patcher.CoreModelPatcher(
                fl, load_device=self.load_device, offload_device=offload_device,
                size=comfy.model_management.module_size(fl),
            )

    def detect_batch(self, images, num_faces: int, score_thresh: float, variant: str):
        comfy.model_management.load_model_gpu(self.patchers[variant])
        return self.models[variant].detect_batch(images, num_faces=num_faces, score_thresh=score_thresh)


def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
    return image[..., :3].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def _parse_color(color: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(color)[:3]
    except ValueError:
        return (0, 255, 0)


def _copy_face(face: dict) -> dict:
    """Shallow copy of a face_dict with array-fields cloned so callers can mutate."""
    return {
        "bbox_xyxy":    face["bbox_xyxy"].copy(),
        "blendshapes":  dict(face["blendshapes"]),
        "landmarks_xy": face["landmarks_xy"].copy(),
        "landmarks_3d": face["landmarks_3d"].copy(),
        "presence":     face["presence"],
        "score":        face["score"],
    }


def _lerp_face(a: dict, b: dict, t: float) -> dict:
    return {
        "bbox_xyxy":    (1 - t) * a["bbox_xyxy"]    + t * b["bbox_xyxy"],
        "blendshapes":  {k: (1 - t) * a["blendshapes"][k] + t * b["blendshapes"][k] for k in a["blendshapes"]},
        "landmarks_xy": (1 - t) * a["landmarks_xy"] + t * b["landmarks_xy"],
        "landmarks_3d": (1 - t) * a["landmarks_3d"] + t * b["landmarks_3d"],
        "presence":     (1 - t) * a["presence"] + t * b["presence"],
        "score":        (1 - t) * a["score"]    + t * b["score"],
    }


def _match_faces(a: list[dict], b: list[dict]) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour pairing of faces between two frames by bbox
    centre distance. Unmatched (when counts differ) are dropped."""
    if not a or not b:
        return []
    centers_a = np.array([(0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                           0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3])) for f in a])
    centers_b = np.array([(0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                           0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3])) for f in b])
    dists = np.linalg.norm(centers_a[:, None] - centers_b[None], axis=-1)
    pairs: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    candidates = sorted((dists[ia, ib], ia, ib) for ia in range(len(a)) for ib in range(len(b)))
    for _, ia, ib in candidates:
        if ia in used_a or ib in used_b:
            continue
        pairs.append((ia, ib))
        used_a.add(ia)
        used_b.add(ib)
    return pairs


def _fill_missing_frames(frames: list[list[dict]], mode: str) -> None:
    """In-place fill empty frame slots from neighbouring detections. Multi-face
    aware: pairs faces across bracketing frames by greedy bbox-centre NN.
    When counts differ, unmatched faces are dropped from the synthesised frame."""
    if mode == "empty":
        return
    valid = [i for i, fr in enumerate(frames) if fr]
    if not valid:
        return  # nothing to fill from
    if mode == "previous":
        last: list[dict] = []
        for i, fr in enumerate(frames):
            if fr:
                last = fr
            elif last:
                frames[i] = [_copy_face(f) for f in last]
        return
    # interpolate: lerp between bracketing valid frames; clamp at ends.
    for i in range(len(frames)):
        if frames[i]:
            continue
        prev_i = max((v for v in valid if v < i), default=None)
        next_i = min((v for v in valid if v > i), default=None)
        if prev_i is None:
            frames[i] = [_copy_face(f) for f in frames[next_i]]
        elif next_i is None:
            frames[i] = [_copy_face(f) for f in frames[prev_i]]
        else:
            t = (i - prev_i) / (next_i - prev_i)
            pairs = _match_faces(frames[prev_i], frames[next_i])
            frames[i] = [_lerp_face(frames[prev_i][a], frames[next_i][b], t) for a, b in pairs]


def _ordered_rings(edges: frozenset[tuple[int, int]]) -> list[list[int]]:
    """Walk an unordered edge set into one or more closed-loop vertex rings
    (handles multi-loop sets like FACEMESH_LIPS: outer + inner)."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited: set[int] = set()
    rings: list[list[int]] = []
    for start in adj:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        prev, cur = -1, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev), None)
            if nxt is None or nxt == start:
                break
            ring.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        rings.append(ring)
    return rings


_TRANSFER_FEATURES = ("left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips", "nose")


def _edge_vertices(edges: frozenset[tuple[int, int]]) -> list[int]:
    return sorted({vertex for edge in edges for vertex in edge})


def _face_scale(face: dict) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = (float(v) for v in face["bbox_xyxy"])
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
    scale = max(x2 - x1, y2 - y1) * 0.5
    if scale < 1.0:
        raise ValueError("Face transfer requires non-degenerate face landmarks.")
    return center, scale


def _face_transfer_anchors(face: dict, connection_sets: dict[str, frozenset]) -> np.ndarray:
    landmarks = face["landmarks_xy"]
    anchors = [landmarks[_edge_vertices(connection_sets[feature])].mean(axis=0) for feature in _TRANSFER_FEATURES]
    oval_vertices = _edge_vertices(connection_sets["face_oval"])
    anchors.append(landmarks[oval_vertices].mean(axis=0))
    return np.asarray(anchors, dtype=np.float64)


def _solve_thin_plate_spline(control: np.ndarray, values: np.ndarray, smoothing: float = 1e-4):
    if control.shape != values.shape or control.ndim != 2 or control.shape[1] != 2 or control.shape[0] < 3:
        raise ValueError("Face transfer requires matching two-dimensional landmark anchors.")

    distances_sq = ((control[:, None] - control[None]) ** 2).sum(axis=2)
    kernel = distances_sq * np.log(distances_sq + 1e-12)
    affine = np.column_stack([np.ones(control.shape[0]), control])
    system = np.block([
        [kernel + np.eye(control.shape[0]) * smoothing, affine],
        [affine.T, np.zeros((3, 3))],
    ])
    targets = np.vstack([values, np.zeros((3, 2))])
    try:
        coefficients = np.linalg.solve(system, targets)
    except np.linalg.LinAlgError as error:
        raise ValueError("Face transfer could not solve the landmark deformation.") from error
    return coefficients[:control.shape[0]], coefficients[control.shape[0]:]


def _evaluate_thin_plate_spline(points: np.ndarray, control: np.ndarray, weights: np.ndarray, affine: np.ndarray) -> np.ndarray:
    output = np.empty_like(points, dtype=np.float64)
    chunk = 65536
    for start in range(0, points.shape[0], chunk):
        current = points[start:start + chunk]
        distances_sq = ((current[:, None] - control[None]) ** 2).sum(axis=2)
        kernel = distances_sq * np.log(distances_sq + 1e-12)
        output[start:start + chunk] = kernel @ weights + np.column_stack([np.ones(current.shape[0]), current]) @ affine
    return output


def _face_oval(face: dict, connection_sets: dict[str, frozenset]) -> np.ndarray:
    rings = _ordered_rings(connection_sets["face_oval"])
    if not rings:
        raise ValueError("Face transfer requires the MediaPipe face oval topology.")
    return face["landmarks_xy"][max(rings, key=len)]


def _polygon_mask(height: int, width: int, points: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(float(x) - offset_x, float(y) - offset_y) for x, y in points], fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _face_transfer_mask(height: int, width: int, face: dict, connection_sets: dict[str, frozenset],
                        edge_feather: float = 8.0, forehead_coverage: float = 0.5,
                        image: torch.Tensor | None = None) -> tuple[np.ndarray, int, int]:
    landmarks = face["landmarks_xy"]
    _, scale = _face_scale(face)
    blend_width = max(1.0, scale * edge_feather / 100.0)
    x1, y1, x2, y2 = (float(v) for v in face["bbox_xyxy"])
    left = max(0, int(np.floor(x1)) - 2)
    top_offset = max(0, int(np.floor(y1)) - 2)
    right = min(width, int(np.ceil(x2)) + 2)
    bottom = min(height, int(np.ceil(y2)) + 2)
    face_oval = _face_oval(face, connection_sets)
    oval = _polygon_mask(bottom - top_offset, right - left, face_oval, left, top_offset) > 0.5

    eyes = np.concatenate([
        landmarks[_edge_vertices(connection_sets["left_eye"])],
        landmarks[_edge_vertices(connection_sets["right_eye"])],
    ])
    eyebrows = np.concatenate([
        landmarks[_edge_vertices(connection_sets["left_eyebrow"])],
        landmarks[_edge_vertices(connection_sets["right_eyebrow"])],
    ])
    eye_center = eyes.mean(axis=0)
    mouth_center = landmarks[_edge_vertices(connection_sets["lips"])].mean(axis=0)
    vertical = mouth_center - eye_center
    vertical_length = float(np.linalg.norm(vertical))
    if vertical_length < 1.0:
        raise ValueError("Face transfer requires non-degenerate face landmarks.")
    vertical /= vertical_length
    eye_top = float(np.min((eyes - eye_center) @ vertical))
    oval_top = float(np.min((face_oval - eye_center) @ vertical))
    fade_center = eye_top + (oval_top - eye_top) * forehead_coverage
    fade_start = fade_center - vertical_length * 0.15
    fade_end = fade_center + vertical_length * 0.10

    yy, xx = np.mgrid[top_offset:bottom, left:right]
    relative_y = (xx - eye_center[0]) * vertical[0] + (yy - eye_center[1]) * vertical[1]
    top_weight = np.clip((relative_y - fade_start) / max(fade_end - fade_start, 1.0), 0.0, 1.0)
    top_weight = top_weight * top_weight * (3.0 - 2.0 * top_weight)

    edge_weight = np.clip(scipy.ndimage.distance_transform_edt(oval) / blend_width, 0.0, 1.0)
    edge_weight = edge_weight * edge_weight * (3.0 - 2.0 * edge_weight)
    mask = edge_weight * top_weight

    if image is not None:
        nose_center = landmarks[_edge_vertices(connection_sets["nose"])].mean(axis=0)
        reference_radius = max(3.0, scale * 0.07)
        reference = (xx - nose_center[0]) ** 2 + (yy - nose_center[1]) ** 2 <= reference_radius ** 2
        image_crop = image[0, top_offset:bottom, left:right, :3].detach().float().cpu().numpy()
        if int(reference.sum()) >= 16:
            skin_color = np.median(image_crop[reference], axis=0)
            color_distance = np.sqrt(((image_crop - skin_color) ** 2).mean(axis=2))
            reference_distance = color_distance[reference]
            low = max(0.08, float(np.percentile(reference_distance, 90)) * 1.5)
            high = max(low + 0.08, 0.22)
            color_edge = np.clip((color_distance - low) / (high - low), 0.0, 1.0)
            color_edge = color_edge * color_edge * (3.0 - 2.0 * color_edge)
            smooth_image = scipy.ndimage.gaussian_filter(image_crop, sigma=(1.0, 1.0, 0.0))
            gradient_x = np.stack([scipy.ndimage.sobel(smooth_image[..., channel], axis=1) for channel in range(3)], axis=2) * 0.25
            gradient_y = np.stack([scipy.ndimage.sobel(smooth_image[..., channel], axis=0) for channel in range(3)], axis=2) * 0.25
            strong_edge = np.sqrt(((gradient_x ** 2 + gradient_y ** 2).mean(axis=2))) > 0.08
            eyebrow_top = float(np.min((eyebrows - eye_center) @ vertical))
            forehead = np.clip(1.0 - (relative_y - eyebrow_top) / max(vertical_length * 0.10, 1.0), 0.0, 1.0)
            forehead = forehead * forehead * (3.0 - 2.0 * forehead)
            oval_distance = scipy.ndimage.distance_transform_edt(oval)
            outer = np.clip(1.0 - oval_distance / max(scale * 0.12, 1.0), 0.0, 1.0)
            occlusion_zone = np.maximum(forehead, outer)
            protected = np.zeros_like(oval)
            for eye_name in ("left_eye", "right_eye"):
                rings = _ordered_rings(connection_sets[eye_name])
                if rings:
                    eye = landmarks[max(rings, key=len)]
                    eye_mask = _polygon_mask(bottom - top_offset, right - left, eye, left, top_offset) > 0.5
                    protected |= scipy.ndimage.distance_transform_edt(~eye_mask) <= max(1.0, scale * 0.04)
            candidates = oval & ~protected & (color_edge > 0.25) & (occlusion_zone > 0.05)
            labels, count = scipy.ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
            if count > 0:
                boundary = oval & (oval_distance <= max(2.0, blend_width * 1.5))
                sizes = np.bincount(labels.ravel())
                touches_boundary = np.bincount(labels.ravel(), weights=boundary.ravel(), minlength=sizes.shape[0]) > 0
                edge_pixels = np.bincount(labels.ravel(), weights=strong_edge.ravel(), minlength=sizes.shape[0])
                keep = (sizes >= max(16, round(int(oval.sum()) * 0.005))) & touches_boundary & (edge_pixels >= np.maximum(2, sizes * 0.01))
                keep[0] = False
                mask *= 1.0 - color_edge * occlusion_zone * keep[labels]

    return mask.astype(np.float32), left, top_offset


def _sampling_grid(points: np.ndarray, height: int, width: int, left: int = 0, top: int = 0) -> np.ndarray:
    return np.column_stack([
        (2.0 * (points[:, 0] - left) + 1.0) / width - 1.0,
        (2.0 * (points[:, 1] - top) + 1.0) / height - 1.0,
    ])


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ matrix.T / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    threshold = (6.0 / 29.0) ** 3
    transformed = np.where(xyz > threshold, np.cbrt(xyz), xyz / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([
        116.0 * transformed[..., 1] - 16.0,
        500.0 * (transformed[..., 0] - transformed[..., 1]),
        200.0 * (transformed[..., 1] - transformed[..., 2]),
    ], axis=-1).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    threshold = 6.0 / 29.0
    xyz = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(xyz > threshold, xyz ** 3, 3.0 * threshold ** 2 * (xyz - 4.0 / 29.0))
    xyz *= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32)
    linear = xyz @ matrix.T
    positive = np.maximum(linear, 0.0)
    rgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * positive ** (1.0 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _large_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    labels, count = scipy.ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_area
    keep[0] = False
    return keep[labels]


def _harmonize_face(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    color_match: float = 0.5, lighting_match: float = 0.5) -> torch.Tensor:
    active = mask.detach().cpu().numpy() > 0.01
    if (color_match <= 0.0 and lighting_match <= 0.0) or int(active.sum()) < 16:
        return source

    source_rgb = source.detach().float().cpu().numpy()
    target_rgb = target.detach().float().cpu().numpy()
    source_lab = _rgb_to_lab(source_rgb)
    target_lab = _rgb_to_lab(target_rgb)
    sigma = max(1.0, max(source.shape[0], source.shape[1]) * 0.025)
    source_base = scipy.ndimage.gaussian_filter(source_lab, sigma=(sigma, sigma, 0))
    target_base = scipy.ndimage.gaussian_filter(target_lab, sigma=(sigma, sigma, 0))

    difference = source_base - target_base
    color_distance = np.sqrt((difference[..., 0] * 0.5) ** 2 + difference[..., 1] ** 2 + difference[..., 2] ** 2)
    similarity = np.clip((35.0 - color_distance) / 27.0, 0.0, 1.0)
    similarity = similarity * similarity * (3.0 - 2.0 * similarity)
    source_gradient = np.sqrt(sum(scipy.ndimage.sobel(source_base[..., channel], axis=0) ** 2 + scipy.ndimage.sobel(source_base[..., channel], axis=1) ** 2 for channel in range(3)))
    target_gradient = np.sqrt(sum(scipy.ndimage.sobel(target_base[..., channel], axis=0) ** 2 + scipy.ndimage.sobel(target_base[..., channel], axis=1) ** 2 for channel in range(3)))
    gradient = np.maximum(source_gradient, target_gradient)
    gradient_limit = float(np.percentile(gradient[active], 70))
    candidates = active & (similarity > 0.15) & (gradient <= gradient_limit)
    regions = _large_regions(candidates, max(32, round(int(active.sum()) * 0.01)))
    weight = similarity * regions
    channel_weight = np.array([lighting_match, color_match, color_match], dtype=np.float32)
    matched_lab = source_lab + (target_base - source_base) * weight[..., None] * channel_weight
    matched_rgb = torch.from_numpy(_lab_to_rgb(matched_lab)).to(device=source.device, dtype=source.dtype)
    return matched_rgb


def _largest_face(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    return max(faces, key=lambda face: max(0.0, float(face["bbox_xyxy"][2] - face["bbox_xyxy"][0])) * max(0.0, float(face["bbox_xyxy"][3] - face["bbox_xyxy"][1])))


def _detect_largest_faces(face_detection_model, images: torch.Tensor) -> list[dict | None]:
    images_np = list(_image_to_uint8(images))
    detected = face_detection_model.detect_batch(images_np, num_faces=0, score_thresh=0.5, variant="short")
    missing = [i for i, faces in enumerate(detected) if not faces]
    if missing:
        fallback = face_detection_model.detect_batch([images_np[i] for i in missing], num_faces=0, score_thresh=0.5, variant="full")
        for index, faces in zip(missing, fallback):
            detected[index] = faces
    return [_largest_face(faces) for faces in detected]


def _transfer_face(source: torch.Tensor, target: torch.Tensor, source_face: dict, target_face: dict,
                   connection_sets: dict[str, frozenset], source_scale_factor: float = 1.08,
                   edge_feather: float = 8.0, forehead_coverage: float = 0.5,
                   color_match: float = 0.5, lighting_match: float = 0.5) -> torch.Tensor:
    source_center, source_scale = _face_scale(source_face)
    target_center, target_scale = _face_scale(target_face)
    source_anchors = (_face_transfer_anchors(source_face, connection_sets) - source_center) / source_scale
    target_anchors = (_face_transfer_anchors(target_face, connection_sets) - target_center) / target_scale
    source_weights, source_affine = _solve_thin_plate_spline(target_anchors, source_anchors, smoothing=0.01)

    x1, y1, x2, y2 = (float(v) for v in target_face["bbox_xyxy"])
    padding = max(2, round(target_scale * 0.04))
    left = max(0, int(np.floor(x1)) - padding)
    top = max(0, int(np.floor(y1)) - padding)
    right = min(target.shape[2], int(np.ceil(x2)) + padding)
    bottom = min(target.shape[1], int(np.ceil(y2)) + padding)
    if right <= left or bottom <= top:
        raise ValueError("Face transfer produced an empty target region.")

    yy, xx = np.mgrid[top:bottom, left:right]
    target_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)
    target_normalized = (target_points - target_center) / target_scale
    source_normalized = _evaluate_thin_plate_spline(target_normalized, target_anchors, source_weights, source_affine)
    anchor_distance = np.sqrt(((target_normalized[:, None] - target_anchors[None]) ** 2).sum(axis=2)).min(axis=1)
    overscale_weight = np.clip(anchor_distance / 0.45, 0.0, 1.0)
    overscale_weight = overscale_weight * overscale_weight * (3.0 - 2.0 * overscale_weight)
    source_normalized *= (1.0 - (1.0 - 1.0 / source_scale_factor) * overscale_weight)[:, None]
    source_points = source_normalized * source_scale + source_center

    source_height, source_width = source.shape[1:3]
    source_grid = _sampling_grid(source_points, source_height, source_width).reshape(bottom - top, right - left, 2)
    target_height, target_width = target.shape[1:3]
    source_grid = torch.from_numpy(source_grid).to(device=source.device, dtype=source.dtype)[None]

    source_rgb = source[:, :, :, :3].movedim(-1, 1)
    warped_source = F.grid_sample(source_rgb, source_grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0].movedim(0, -1)

    source_mask_np, source_mask_left, source_mask_top = _face_transfer_mask(
        source_height, source_width, source_face, connection_sets, edge_feather, forehead_coverage, source,
    )
    source_mask = torch.from_numpy(source_mask_np).to(device=source.device, dtype=source.dtype)[None, None]
    source_mask_grid = _sampling_grid(source_points, source_mask_np.shape[0], source_mask_np.shape[1], source_mask_left, source_mask_top)
    source_mask_grid = torch.from_numpy(source_mask_grid.reshape(bottom - top, right - left, 2)).to(device=source.device, dtype=source.dtype)[None]
    warped_source_mask = F.grid_sample(source_mask, source_mask_grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]

    target_mask_np, target_mask_left, target_mask_top = _face_transfer_mask(
        target_height, target_width, target_face, connection_sets, edge_feather, forehead_coverage, target,
    )
    target_mask = torch.from_numpy(target_mask_np).to(device=source.device, dtype=source.dtype)[None, None]
    target_mask_grid = _sampling_grid(target_points, target_mask_np.shape[0], target_mask_np.shape[1], target_mask_left, target_mask_top)
    target_mask_grid = torch.from_numpy(target_mask_grid.reshape(bottom - top, right - left, 2)).to(device=source.device, dtype=source.dtype)[None]
    warped_target_mask = F.grid_sample(target_mask, target_mask_grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
    mask = torch.minimum(warped_source_mask, warped_target_mask).clamp(0.0, 1.0)

    original_target = target[0, top:bottom, left:right, :3].to(device=source.device, dtype=source.dtype)
    warped_source = _harmonize_face(warped_source, original_target, mask, color_match, lighting_match)
    composited = original_target * (1.0 - mask[..., None]) + warped_source * mask[..., None]

    output = target.clone()
    output[0, top:bottom, left:right, :3] = composited.to(device=target.device, dtype=target.dtype)
    return output


class LoadMediaPipeFaceLandmarker(io.ComfyNode):
    """Load MediaPipe Face Landmarker v2 weights. Contains both detector variants
    (short / full), shared mesh, blendshapes, and canonical geometry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadMediaPipeFaceLandmarker",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection"],
            display_name="Load Face Detection Model (MediaPipe)",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("detection"),
                               tooltip="Face detection model from models/detection/."),
            ],
            outputs=[FaceDetectionType.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("detection", model_name), safe_load=True)
        wrapper = FaceLandmarkerModel(sd)
        return io.NodeOutput(wrapper)


# Per-frame fallback modes for detection failures in a batch.
_FALLBACK_MODES = ("empty", "previous", "interpolate")


class MediaPipeFaceLandmarker(io.ComfyNode):
    """BlazeFace → FaceMesh v2 → ARKit-52 blendshapes, batched across the
    input. Also emits a BOUNDING_BOX list (landmark-extent bbox per face) —
    pair with DrawBBoxes for detector-only viz or MediaPipeFaceMeshVisualize
    for the mesh overlay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceLandmarker",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection"],
            display_name="Detect Face Landmarks (MediaPipe)",
            category="image/detection",
            description="Detects facial landmarks using MediaPipe model.",
            inputs=[
                FaceDetectionType.Input("face_detection_model"),
                io.Image.Input("image"),
                io.Combo.Input("detector_variant", options=["short", "full", "both"], default="short",
                               tooltip="Face detector range. 'short' is tuned for close-up faces "
                                       "(within ~2 m of the camera); 'full' covers farther / smaller "
                                       "faces (up to ~5 m) but is slower. 'both' runs both detectors and "
                                       "keeps whichever found more faces per frame (~2× detection cost)."),
                io.Int.Input("num_faces", default=1, min=0, max=16, step=1,
                             tooltip="Maximum faces to return per frame. 0 = no cap (return all detected)."),
                io.Float.Input("min_confidence", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="BlazeFace score threshold. Lower to catch small/occluded faces."),
                io.Combo.Input("missing_frame_fallback", options=list(_FALLBACK_MODES), default="empty", advanced=True,
                               tooltip="Per-frame behaviour when detection fails in a batch. "
                                       "'empty' leaves the frame faceless. 'previous' copies the most recent successful "
                                       "detection. 'interpolate' lerps landmarks/bbox/blendshapes between bracketing "
                                       "successful frames. Multi-face: pairs faces across frames by greedy bbox-centre NN."),
            ],
            outputs=[
                FaceLandmarksType.Output(display_name="face_landmarks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, face_detection_model, image, detector_variant, num_faces, min_confidence,
                missing_frame_fallback) -> io.NodeOutput:
        canonical = face_detection_model.canonical_data
        img_np = _image_to_uint8(image)
        B, H, W = img_np.shape[:3]
        chunk = 16
        is_both = detector_variant == "both"
        total_work = 2 * B if is_both else B
        pbar = comfy.utils.ProgressBar(total_work)

        def _run(variant: str) -> list[list[dict]]:
            res: list[list[dict]] = []
            with tqdm(total=B, desc=f"MediaPipe Face Landmarker ({variant})") as tq:
                for i in range(0, B, chunk):
                    end = min(i + chunk, B)
                    res.extend(face_detection_model.detect_batch(
                        [img_np[bi] for bi in range(i, end)],
                        num_faces=int(num_faces),
                        score_thresh=float(min_confidence),
                        variant=variant,
                    ))
                    pbar.update_absolute(min(pbar.current + (end - i), total_work))
                    tq.update(end - i)
            return res

        if is_both:
            short_res = _run("short")
            full_res = _run("full")
            # Per-frame keep whichever found more faces (tie → short).
            frames: list[list[dict]] = [
                short_res[bi] if len(short_res[bi]) >= len(full_res[bi]) else full_res[bi]
                for bi in range(B)
            ]
        else:
            frames = _run(detector_variant)
        _fill_missing_frames(frames, missing_frame_fallback)
        bboxes = []
        for per_frame in frames:
            per_bb = []
            for f in per_frame:
                f["transformation_matrix"] = transformation_matrix_from_detection(f, W, H, canonical)
                x1, y1, x2, y2 = (float(v) for v in f["bbox_xyxy"])
                per_bb.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": "face", "score": float(f["score"])})
            bboxes.append(per_bb)
        return io.NodeOutput({"frames": frames, "image_size": (H, W),
                              "connection_sets": face_detection_model.connection_sets}, bboxes)


class MediaPipeFaceTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceTransfer",
            search_aliases=["face", "facial", "mediapipe", "face transfer", "face swap"],
            display_name="Transfer Face (MediaPipe)",
            category="image/transform",
            description="Transfers the largest source face into the largest target face using MediaPipe landmarks.",
            inputs=[
                FaceDetectionType.Input("face_detection_model"),
                io.Image.Input("source_image"),
                io.Image.Input("target_image"),
                io.Float.Input("source_scale", default=1.08, min=0.9, max=1.3, step=0.01, advanced=True,
                               tooltip="Scales the source face coverage while keeping feature centers aligned."),
                io.Float.Input("edge_feather", default=8.0, min=1.0, max=25.0, step=0.5, advanced=True,
                               tooltip="Width of the source-to-target transition as a percentage of face size."),
                io.Float.Input("forehead_coverage", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="Extends valid forehead coverage from the eyes toward the face oval. Strong color edges such as bangs remain excluded."),
                io.Float.Input("color_match", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="Matches target color in large similar-color face regions."),
                io.Float.Input("lighting_match", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="Matches target lighting in large similar-color face regions."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, face_detection_model, source_image, target_image, source_scale=1.08,
                edge_feather=8.0, forehead_coverage=0.5, color_match=0.5, lighting_match=0.5) -> io.NodeOutput:
        source_batch = source_image.shape[0]
        target_batch = target_image.shape[0]
        if source_batch not in (1, target_batch):
            raise ValueError("Face transfer requires one source image or one source image per target image.")

        source_faces = _detect_largest_faces(face_detection_model, source_image)
        target_faces = _detect_largest_faces(face_detection_model, target_image)
        for i, face in enumerate(source_faces):
            if face is None:
                raise ValueError(f"Face transfer could not detect a source face in image {i}.")
        for i, face in enumerate(target_faces):
            if face is None:
                raise ValueError(f"Face transfer could not detect a target face in image {i}.")

        connection_sets = face_detection_model.connection_sets
        output = []
        for i in range(target_batch):
            source_index = 0 if source_batch == 1 else i
            output.append(_transfer_face(
                source_image[source_index:source_index + 1],
                target_image[i:i + 1],
                source_faces[source_index],
                target_faces[i],
                connection_sets,
                source_scale,
                edge_feather,
                forehead_coverage,
                color_match,
                lighting_match,
            ))
        return io.NodeOutput(torch.cat(output, dim=0))


# Topology keys unioned by the 'all' connections preset (contour parts + irises + nose).
_ALL_CONNECTION_PARTS: tuple[str, ...] = (*_CONTOUR_PARTS, "irises", "nose")
_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval",     True),
    ("lips",          True),
    ("left_eye",      True),
    ("right_eye",     True),
    ("left_eyebrow",  True),
    ("right_eyebrow", True),
    ("irises",        True),
    ("nose",          True),
    ("tesselation",   False),
)


class MediaPipeFaceMeshVisualize(io.ComfyNode):
    """Draw a FACEMESH_* subset over an image. Topology travels with the
    FACE_LANDMARKS payload (set at detection time)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceMeshVisualize",
            search_aliases=["face", "facial", "mediapipe", "face landmark", "face mesh", "blazeface", "face detection", "visualize"],
            display_name="Visualize Face Landmarks (MediaPipe)",
            category="image/detection",
            description="Draws face landmarks mesh on the input image.",
            inputs=[
                FaceLandmarksType.Input("face_landmarks"),
                io.Image.Input("image", optional=True, tooltip="If not connected, a black canvas will be used."),
                io.DynamicCombo.Input(
                    "connections",
                    tooltip="'all' = oval+eyes+brows+lips+irises+nose. 'fill' = solid face_oval polygon (silhouette mask). 'custom' = toggle each feature individually (including 'tesselation', the full 2547-edge wireframe).",
                    options=[
                        io.DynamicCombo.Option("all", []),
                        io.DynamicCombo.Option("fill", []),
                        io.DynamicCombo.Option("custom", [
                            io.Boolean.Input(feat, default=default,
                                             tooltip=f"Draw the '{feat}' connection set.")
                            for feat, default in _CUSTOM_FEATURES
                        ]),
                    ],
                ),
                io.Color.Input("color", default="#00ff00"),
                io.Int.Input("thickness", default=1, min=0, max=8, step=1,
                             tooltip="Edge line thickness in pixels. 0 disables edge drawing."),
                io.Int.Input("point_size", default=2, min=0, max=16, step=1,
                             tooltip="Landmark dot radius in pixels. 0 disables point drawing."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, face_landmarks, connections, color, thickness, point_size, image=None) -> io.NodeOutput:
        sets = face_landmarks["connection_sets"]
        sel = connections["connections"]
        fill_rings: list[list[int]] | None = None
        if sel == "fill":
            fill_rings = _ordered_rings(sets["face_oval"])
            edges = frozenset()
        elif sel == "custom":
            parts = [feat for feat, _ in _CUSTOM_FEATURES if connections.get(feat, False)]
            edges = frozenset().union(*(sets[p] for p in parts))
        else:  # "all"
            edges = frozenset().union(*(sets[p] for p in _ALL_CONNECTION_PARTS))
        rgb, thick, psize = _parse_color(color), int(thickness), int(point_size)
        frames = face_landmarks["frames"]
        if image is None:
            H, W = face_landmarks["image_size"]
            img_np = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
        else:
            img_np = _image_to_uint8(image)
        B = img_np.shape[0]
        n_frames = len(frames)
        pbar = comfy.utils.ProgressBar(B)
        out = np.empty_like(img_np)
        for bi in range(B):
            faces = frames[bi] if bi < n_frames else []
            out[bi] = _draw_mesh(img_np[bi], faces, edges, rgb, thick, psize, fill_rings)
            pbar.update_absolute(bi + 1)
        return io.NodeOutput(torch.from_numpy(out).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        ).div_(255.0))


def _draw_mesh(image_rgb: np.ndarray, faces: list, edges,
               rgb: tuple[int, int, int], thickness: int,
               point_size: int, fill_rings: list[list[int]] | None = None) -> np.ndarray:
    draw_edges = thickness > 0 and edges
    if not faces or (fill_rings is None and not draw_edges and point_size <= 0):
        return image_rgb.copy()
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    r = point_size * 0.5
    if fill_rings is not None:
        for f in faces:
            lmks = f["landmarks_xy"]
            for ring in fill_rings:
                draw.polygon([(float(lmks[i, 0]), float(lmks[i, 1])) for i in ring], fill=rgb)
        return np.asarray(pil)
    for f in faces:
        lmks = f["landmarks_xy"]
        n = lmks.shape[0]
        if draw_edges:
            for a, b in edges:
                if a < n and b < n:
                    draw.line([(float(lmks[a, 0]), float(lmks[a, 1])),
                               (float(lmks[b, 0]), float(lmks[b, 1]))], fill=rgb, width=thickness)
        if point_size == 1:
            draw.point(lmks.flatten().tolist(), fill=rgb)
        elif point_size > 1:
            for x, y in lmks:
                draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r), fill=rgb)
    return np.asarray(pil)


# Mask region presets — closed-loop topologies only.
_MASK_REGIONS: tuple[str, ...] = ("face_oval", "lips", "left_eye", "right_eye", "irises")
_MASK_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval",  True),
    ("lips",       False),
    ("left_eye",   False),
    ("right_eye",  False),
    ("irises",     False),
)


class MediaPipeFaceMask(io.ComfyNode):
    """Binary mask from face landmarks, filled polygon per face. One mask per
    frame in the batch; faces in the same frame composite (union)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeFaceMask",
            search_aliases=["face", "facial", "mediapipe", "face mask", "blazeface", "face detection", "visualize"],
            display_name="Draw Face Mask (MediaPipe)",
            category="image/detection",
            description="Draws a mask from face landmarks.",
            inputs=[
                FaceLandmarksType.Input("face_landmarks"),
                io.DynamicCombo.Input(
                    "regions",
                    tooltip="'all' = union of face_oval+lips+eyes+irises (which collapses to face_oval since it encloses the rest). 'custom' = toggle each region individually for combos like lips+eyes.",
                    options=[
                        io.DynamicCombo.Option("all", []),
                        io.DynamicCombo.Option("custom", [
                            io.Boolean.Input(reg, default=default,
                                             tooltip=f"Include the '{reg}' region in the mask.")
                            for reg, default in _MASK_CUSTOM_FEATURES
                        ]),
                    ],
                ),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, face_landmarks, regions) -> io.NodeOutput:
        sets = face_landmarks["connection_sets"]
        sel = regions["regions"]
        if sel == "custom":
            picked = [reg for reg, _ in _MASK_CUSTOM_FEATURES if regions.get(reg, False)]
        else:
            picked = list(_MASK_REGIONS)
        rings = [r for reg in picked for r in _ordered_rings(sets[reg])]
        frames = face_landmarks["frames"]
        H, W = face_landmarks["image_size"]
        masks = np.zeros((len(frames), H, W), dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(len(frames))
        for bi, per_frame in enumerate(frames):
            if per_frame:
                pil = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(pil)
                for f in per_frame:
                    lmks = f["landmarks_xy"]
                    for ring in rings:
                        draw.polygon([(float(lmks[i, 0]), float(lmks[i, 1])) for i in ring], fill=255)
                masks[bi] = np.asarray(pil)
            pbar.update_absolute(bi + 1)
        return io.NodeOutput(torch.from_numpy(masks).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        ).div_(255.0))


class MediaPipeFaceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadMediaPipeFaceLandmarker, MediaPipeFaceLandmarker, MediaPipeFaceTransfer, MediaPipeFaceMeshVisualize, MediaPipeFaceMask]


async def comfy_entrypoint() -> MediaPipeFaceExtension:
    return MediaPipeFaceExtension()
