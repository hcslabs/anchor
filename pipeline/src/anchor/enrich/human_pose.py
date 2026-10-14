"""Low-footprint human landmark extraction for Anchor proof runs.

This is a local screening/reproducibility path, not a replacement for the
full metric VideoMimic/SMPL-X reconstruction used for production training.
It proves that the retargeted trajectory is derived from the selected video
rather than from a released robot-motion dataset.  Frames are decoded through
ffmpeg and MediaPipe world landmarks are converted to a pelvis-centred,
body-aligned coordinate system.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import math
import subprocess
from typing import Any

import numpy as np


LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-6:
        raise ValueError(f"cannot construct body frame: degenerate {name}")
    return vector / norm


def canonical_landmarks(world: np.ndarray) -> dict[str, list[float]]:
    """Convert MediaPipe world landmarks to x-forward/y-left/z-up metres."""
    if world.shape != (33, 3):
        raise ValueError(f"expected 33x3 MediaPipe landmarks, got {world.shape}")
    left_hip = world[LANDMARK_INDEX["left_hip"]]
    right_hip = world[LANDMARK_INDEX["right_hip"]]
    left_shoulder = world[LANDMARK_INDEX["left_shoulder"]]
    right_shoulder = world[LANDMARK_INDEX["right_shoulder"]]
    pelvis = 0.5 * (left_hip + right_hip)
    torso = 0.5 * (left_shoulder + right_shoulder)

    left_axis = _unit(left_hip - right_hip, "hip lateral axis")
    raw_up = _unit(torso - pelvis, "torso axis")
    forward_axis = _unit(np.cross(left_axis, raw_up), "forward axis")
    up_axis = _unit(np.cross(forward_axis, left_axis), "orthogonal up axis")
    rotation = np.column_stack((forward_axis, left_axis, up_axis))

    converted: dict[str, list[float]] = {
        "pelvis": [0.0, 0.0, 0.0],
        "torso": (rotation.T @ (torso - pelvis)).tolist(),
    }
    for name, index in LANDMARK_INDEX.items():
        converted[name] = (rotation.T @ (world[index] - pelvis)).tolist()
    return converted


def interpolate_detection_timeline(
    detections: list[dict[str, Any] | None],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Fill internal pose gaps without changing the source sampling clock.

    Leading and trailing gaps have no two-sided support and are excluded.  An
    emitted item retains its original frame index so callers cannot compress
    time merely because the detector missed a frame.
    """
    detected_indices = [index for index, item in enumerate(detections) if item is not None]
    if not detected_indices:
        raise ValueError("detection timeline contains no usable frames")
    first_detected, last_detected = detected_indices[0], detected_indices[-1]
    next_detected: list[int | None] = [None] * len(detections)
    next_index: int | None = None
    for index in range(len(detections) - 1, -1, -1):
        if detections[index] is not None:
            next_index = index
        next_detected[index] = next_index

    filled: list[dict[str, Any]] = []
    previous_index: int | None = None
    for index in range(first_detected, last_detected + 1):
        detection = detections[index]
        if detection is not None:
            current = detection["landmarks"]
            visibility = float(detection["mean_visibility"])
            previous_index = index
            interpolated = False
        else:
            following_index = next_detected[index]
            if previous_index is None or following_index is None:
                raise RuntimeError("internal interpolation bounds are missing")
            before = detections[previous_index]
            after = detections[following_index]
            if before is None or after is None:
                raise RuntimeError("internal interpolation observation is missing")
            fraction = (index - previous_index) / (following_index - previous_index)
            current = {
                name: (1.0 - fraction) * before["landmarks"][name]
                + fraction * after["landmarks"][name]
                for name in before["landmarks"]
            }
            visibility = (
                (1.0 - fraction) * float(before["mean_visibility"])
                + fraction * float(after["mean_visibility"])
            )
            interpolated = True
        filled.append({
            "source_frame_index": index,
            "detected": not interpolated,
            "interpolated": interpolated,
            "mean_visibility": visibility,
            "landmarks": current,
        })
    return first_detected, last_detected, filled


def extract_human_motion(
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    sample_fps: float = 5.0,
    maximum_width: int = 640,
    minimum_visibility: float = 0.45,
    smoothing_alpha: float = 0.45,
) -> dict[str, Any]:
    if sample_fps <= 0 or maximum_width <= 0:
        raise ValueError("sample_fps and maximum_width must be positive")
    if not 0 < smoothing_alpha <= 1:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    video = Path(video_path).resolve()
    model = Path(model_path).resolve()
    output = Path(output_path).resolve()
    if not video.is_file() or not model.is_file():
        raise FileNotFoundError("video and pose model must exist")

    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("MediaPipe is required for extract-human-pose") from error

    source_width, source_height = _probe_video(video)
    scale = min(1.0, maximum_width / source_width)
    width = max(2, int(round(source_width * scale / 2.0)) * 2)
    height = max(2, int(round(source_height * scale / 2.0)) * 2)
    frame_bytes = width * height * 3
    command = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"scale={width}:{height},fps={sample_fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout is unavailable")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    detections: list[dict[str, Any] | None] = []
    frame_index = 0
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                process.kill()
                raise RuntimeError("ffmpeg returned a partial RGB frame")
            array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            timestamp_ms = int(round(frame_index * 1000.0 / sample_fps))
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=array), timestamp_ms
            )
            frame_index += 1
            if not result.pose_world_landmarks:
                detections.append(None)
                continue
            pose = result.pose_world_landmarks[0]
            important = [LANDMARK_INDEX[name] for name in LANDMARK_INDEX]
            visible = [float(pose[index].visibility or 0.0) for index in important]
            mean_visibility = float(np.mean(visible))
            if mean_visibility < minimum_visibility:
                detections.append(None)
                continue
            world = np.asarray([[point.x, point.y, point.z] for point in pose], dtype=float)
            current = {name: np.asarray(value) for name, value in canonical_landmarks(world).items()}
            detections.append({"mean_visibility": mean_visibility, "landmarks": current})

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed rc={return_code}: {stderr[-1000:]}")
    detected_indices = [index for index, item in enumerate(detections) if item is not None]
    if len(detected_indices) < 5:
        raise RuntimeError(f"only {len(detected_indices)} usable human-pose frames were detected")

    # Preserve the original sampling clock. Internal detection gaps are
    # linearly interpolated; unsupported leading/trailing spans are dropped.
    # This prevents a missing observation from silently speeding up motion.
    first_detected, last_detected, timeline = interpolate_detection_timeline(detections)

    frames: list[dict[str, Any]] = []
    smoothed: dict[str, np.ndarray] | None = None
    for sample in timeline:
        index = int(sample["source_frame_index"])
        current = sample["landmarks"]
        if smoothed is None:
            smoothed = {name: value.copy() for name, value in current.items()}
        else:
            smoothed = {
                name: smoothing_alpha * current[name] + (1.0 - smoothing_alpha) * smoothed[name]
                for name in current
            }
        frames.append({
            "source_frame_index": index,
            "time_s": index / sample_fps,
            "detected": sample["detected"],
            "interpolated": sample["interpolated"],
            "mean_visibility": sample["mean_visibility"],
            "landmarks": {name: value.tolist() for name, value in smoothed.items()},
        })

    video_hash = file_sha256(video)
    model_hash = file_sha256(model)
    document = {
        "schema_version": 1,
        "kind": "anchor_human_motion_v1",
        "coordinate_system": "x_forward_y_left_z_up_m",
        "fps": sample_fps,
        "source_video": {
            "path": str(video),
            "sha256": video_hash,
            "content_id": f"anc1-{video_hash}",
            "rights_status": "requires_manifest_attestation",
        },
        "extractor": {
            "name": "mediapipe_pose_landmarker_screening",
            "model_path": str(model),
            "model_sha256": model_hash,
            "production_reconstruction_required": "VideoMimic/SMPL-X or equivalent metric reconstruction",
            "sample_fps": sample_fps,
            "decoded_size": [width, height],
            "minimum_visibility": minimum_visibility,
            "smoothing_alpha": smoothing_alpha,
        },
        "quality": {
            "decoded_frame_count": frame_index,
            "detected_frame_count": len(detected_indices),
            "interpolated_frame_count": sum(item["interpolated"] for item in timeline),
            "emitted_frame_count": len(frames),
            "dropped_leading_frame_count": first_detected,
            "dropped_trailing_frame_count": frame_index - last_detected - 1,
            "detection_ratio": len(detected_indices) / max(frame_index, 1),
            "mean_visibility": float(np.mean([frame["mean_visibility"] for frame in frames])),
            "uniform_timeline_preserved": True,
            "screening_only": True,
        },
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document

