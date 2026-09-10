"""Geometry helpers for the first-level formation observation algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import numpy as np
import yaml


PathLike = Union[str, Path]


def _vector3(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _matrix3(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{label} must be a 3x3 matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


@dataclass(frozen=True)
class CameraExtrinsic:
    """One camera pose expressed in the robot body frame."""

    camera_id: int
    name: str
    position_mm: np.ndarray
    rotation_camera_to_robot: np.ndarray

    def transform_point(self, point_camera_mm: Iterable[float]) -> np.ndarray:
        point = _vector3(point_camera_mm, "point_camera_mm")
        return self.rotation_camera_to_robot @ point + self.position_mm


class CameraExtrinsics:
    """Load and serve the three fixed camera-to-robot transforms."""

    def __init__(self, robot_frame: str, cameras: Mapping[int, CameraExtrinsic]):
        self.robot_frame = str(robot_frame)
        self.cameras: Dict[int, CameraExtrinsic] = dict(cameras)

    @classmethod
    def load(cls, path: PathLike) -> "CameraExtrinsics":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        unit = str(config.get("unit", "mm")).lower()
        if unit != "mm":
            raise ValueError(
                f"Camera extrinsics must use mm, got unit={config.get('unit')!r}"
            )

        cameras_raw = config.get("cameras")
        if not isinstance(cameras_raw, Mapping):
            raise ValueError("camera extrinsics must contain a 'cameras' mapping")

        cameras = {}
        for camera_id_raw, camera_raw in cameras_raw.items():
            camera_id = int(camera_id_raw)
            if not isinstance(camera_raw, Mapping):
                raise ValueError(f"camera {camera_id} must be a mapping")
            cameras[camera_id] = CameraExtrinsic(
                camera_id=camera_id,
                name=str(camera_raw.get("name", f"camera_{camera_id}")),
                position_mm=_vector3(
                    camera_raw.get("position_mm"),
                    f"camera {camera_id} position_mm",
                ),
                rotation_camera_to_robot=_matrix3(
                    camera_raw.get("rotation_camera_to_robot"),
                    f"camera {camera_id} rotation_camera_to_robot",
                ),
            )

        missing = sorted(set((0, 1, 2)) - set(cameras))
        if missing:
            raise ValueError(
                "camera extrinsics must define camera ids 0, 1 and 2; "
                f"missing {missing}"
            )

        return cls(
            robot_frame=str(config.get("robot_frame", "robot_base")),
            cameras=cameras,
        )

    def get(self, camera_id: int) -> CameraExtrinsic:
        try:
            return self.cameras[int(camera_id)]
        except KeyError as exc:
            raise ValueError(f"unknown camera_id={camera_id}") from exc


class FormationTemplates:
    """Load formation slot templates and calculate local slot offsets."""

    def __init__(self, formations: Mapping[int, Mapping[int, np.ndarray]]):
        self.formations: Dict[int, Dict[int, np.ndarray]] = {
            int(formation_id): {
                int(slot_id): _vector3(point, f"formation {formation_id} slot {slot_id}")
                for slot_id, point in slots.items()
            }
            for formation_id, slots in formations.items()
        }

    @classmethod
    def load(cls, path: PathLike) -> "FormationTemplates":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        unit = str(config.get("unit", "mm")).lower()
        if unit != "mm":
            raise ValueError(
                f"Formation templates must use mm, got unit={config.get('unit')!r}"
            )

        entries = config.get("formation")
        if not isinstance(entries, list):
            raise ValueError("formation templates must contain a 'formation' list")

        formations = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("each formation entry must be a mapping")
            formation_id = int(entry["id"])
            slots_raw = entry.get("slots")
            if not isinstance(slots_raw, Mapping):
                raise ValueError(
                    f"formation {formation_id} must contain a 'slots' mapping"
                )
            slots = {
                int(slot_id): _vector3(
                    point,
                    f"formation {formation_id} slot {slot_id}",
                )
                for slot_id, point in slots_raw.items()
            }
            if not slots:
                raise ValueError(f"formation {formation_id} has no slots")
            if formation_id in formations:
                raise ValueError(f"duplicate formation id {formation_id}")
            formations[formation_id] = slots

        if not formations:
            raise ValueError("formation template list is empty")
        return cls(formations)

    def get_slots(self, formation_id: int) -> Dict[int, np.ndarray]:
        try:
            return self.formations[int(formation_id)]
        except KeyError as exc:
            known = sorted(self.formations)
            raise ValueError(
                f"unknown formation_id={formation_id}; available={known}"
            ) from exc

    def relative_slots(
        self,
        formation_id: int,
        self_slot_id: int,
    ) -> Dict[int, np.ndarray]:
        slots = self.get_slots(formation_id)
        if int(self_slot_id) not in slots:
            raise ValueError(
                f"self_slot_id={self_slot_id} is not present in "
                f"formation_id={formation_id}"
            )

        origin = slots[int(self_slot_id)]
        return {
            slot_id: np.asarray(point - origin, dtype=float)
            for slot_id, point in sorted(slots.items())
        }


def quaternion_from_rotation_matrix(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return an x/y/z/w quaternion for a 3x3 rotation matrix."""

    matrix = _matrix3(rotation, "rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("rotation matrix produced a zero quaternion")
    quaternion /= norm
    return tuple(float(value) for value in quaternion)
