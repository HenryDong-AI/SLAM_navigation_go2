# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Go2 Semantic Mapping contributors

import json
import math

import numpy as np
import pytest

from go2_semantic_mapping.core import (
    Detection,
    SemanticVoxelMap,
    as_transform,
    associate_detections,
    nearest_stamped_sample,
    pose_matrix,
    project_base_points,
    sample_image_rgb,
    save_snapshot_bundle_atomic,
    transform_points,
)


def test_projection_and_rgb_sampling():
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.2, -0.1, 2.0],
            [1.0, 0.0, -1.0],
            [10.0, 0.0, 1.0],
        ]
    )
    projection = project_base_points(
        points,
        np.eye(4),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=40.0,
        image_width=100,
        image_height=80,
        min_depth=0.1,
        max_depth=10.0,
    )
    assert projection.source_indices.tolist() == [0, 1]
    np.testing.assert_allclose(projection.uv, [[50.0, 40.0], [60.0, 35.0]])
    np.testing.assert_allclose(projection.depth, [2.0, 2.0])

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[40, 50] = [1, 2, 3]  # BGR
    image[35, 60] = [4, 5, 6]
    np.testing.assert_array_equal(sample_image_rgb(image, projection.uv), [[3, 2, 1], [6, 5, 4]])


def test_projection_rejects_invalid_intrinsics_and_transform():
    with pytest.raises(ValueError):
        project_base_points(np.zeros((1, 3)), np.eye(4), 0.0, 1.0, 0.0, 0.0, 10, 10)
    bad = np.eye(4)
    bad[3, 3] = 2.0
    with pytest.raises(ValueError):
        as_transform(bad.reshape(-1))


def test_detection_association_uses_mask_and_robust_depth_gate():
    uv = np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0], [13.0, 10.0], [30.0, 30.0]])
    depth = np.array([2.0, 2.05, 2.1, 8.0, 2.0])
    mask = np.zeros((40, 40), dtype=bool)
    mask[9:12, 9:15] = True
    detection = Detection(7, "chair", 0.8, (5.0, 5.0, 20.0, 20.0), mask)
    labels, confidences = associate_detections(
        uv,
        depth,
        [detection],
        min_points=3,
        absolute_depth_gate=0.2,
        mad_scale=3.0,
    )
    assert labels.tolist() == [7, 7, 7, 0, 0]
    np.testing.assert_allclose(confidences[:3], 0.8)
    np.testing.assert_allclose(confidences[3:], 0.0)


def test_high_confidence_detection_claims_overlap_first():
    uv = np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]])
    depth = np.array([1.0, 1.0, 1.0])
    low = Detection(1, "low", 0.4, (0.0, 0.0, 20.0, 20.0))
    high = Detection(2, "high", 0.9, (0.0, 0.0, 20.0, 20.0))
    labels, _ = associate_detections(uv, depth, [low, high], min_points=1)
    assert labels.tolist() == [2, 2, 2]


def test_pose_transform_and_nearest_sample():
    half = math.sqrt(0.5)
    transform = pose_matrix([1.0, 2.0, 3.0], [0.0, 0.0, half, half])
    transformed = transform_points(np.array([[1.0, 0.0, 0.0]]), transform)
    np.testing.assert_allclose(transformed, [[1.0, 3.0, 3.0]], atol=1e-7)

    payload, delta = nearest_stamped_sample([(1.0, "a"), (1.2, "b")], 1.18, 0.05)
    assert payload == "b"
    assert delta == pytest.approx(0.02)
    payload, delta = nearest_stamped_sample([(1.0, "a")], 2.0, 0.1)
    assert payload is None
    assert delta == pytest.approx(1.0)


def test_voxel_votes_are_per_frame_and_map_is_bounded():
    voxel_map = SemanticVoxelMap(voxel_size=1.0, max_voxels=2)
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    voxel_map.update(
        points,
        colors_rgb=np.array([[255, 0, 0], [255, 0, 0]]),
        color_valid=np.array([True, True]),
        labels=np.array([1, 1]),
        confidences=np.array([0.9, 0.8]),
        class_names={1: "chair"},
        observed_ns=1,
    )
    first = voxel_map.snapshot()
    assert len(first["points"]) == 1
    assert first["observations"].tolist() == [1]
    assert first["semantic_observations"].tolist() == [1]
    assert first["labels"].tolist() == [1]
    assert first["confidences"][0] == pytest.approx(0.9)

    voxel_map.update(
        np.array([[0.3, 0.3, 0.3]]),
        labels=np.array([2]),
        confidences=np.array([0.8]),
        class_names={2: "table"},
        observed_ns=2,
    )
    disputed = voxel_map.snapshot()
    assert disputed["labels"].tolist() == [1]
    assert 0.4 < float(disputed["confidences"][0]) < 0.6

    voxel_map.update(np.array([[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]), observed_ns=3)
    bounded = voxel_map.snapshot()
    assert len(bounded["points"]) == 2
    assert [0, 0, 0] not in bounded["voxel_keys"].tolist()


def test_atomic_bundle_contains_ply_and_vote_metadata(tmp_path):
    voxel_map = SemanticVoxelMap(voxel_size=0.25, max_voxels=10)
    voxel_map.update(
        np.array([[1.0, 2.0, 3.0]]),
        colors_rgb=np.array([[10, 20, 30]]),
        color_valid=np.array([True]),
        labels=np.array([4]),
        confidences=np.array([0.75]),
        class_names={4: "extinguisher"},
        observed_ns=123,
    )
    bundle = save_snapshot_bundle_atomic(
        voxel_map.snapshot(), str(tmp_path), "test_map", "odom", metadata={"source": "unit-test"}
    )
    assert bundle.is_dir()
    assert (bundle / "semantic_map.ply").is_file()
    assert (bundle / "semantic_map.json").is_file()
    assert not list(tmp_path.glob(".*-tmp-*"))

    document = json.loads((bundle / "semantic_map.json").read_text(encoding="utf-8"))
    assert document["frame_id"] == "odom"
    assert document["voxel_count"] == 1
    assert document["class_names"] == {"4": "extinguisher"}
    assert document["voxels"][0]["class_votes"] == {"4": pytest.approx(0.75)}
    ply = (bundle / "semantic_map.ply").read_text(encoding="ascii")
    assert "property uint label" in ply
    assert "property float confidence" in ply


def test_clear_reports_removed_voxels():
    voxel_map = SemanticVoxelMap(voxel_size=0.5, max_voxels=3)
    voxel_map.update(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert voxel_map.clear() == 2
    assert len(voxel_map) == 0
