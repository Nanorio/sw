"""3D PID + APF formation-control simulation for unicycle-style robots.

All tunable parameters live in JSON files under ./configs.

Run:
    python formation_pid_apf_sim.py

Examples:
    python formation_pid_apf_sim.py --config-dir configs
    python formation_pid_apf_sim.py --seed 7 --steps 1000
    python formation_pid_apf_sim.py --no-animate --save final.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import queue
import threading

import matplotlib.pyplot as plt
import numpy as np


import concurrent.futures
import os
import time

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = BASE_DIR / "configs"

CONFIG_FILE_NAMES = {
    "simulation": "simulation.json",
    "pid_linear": "pid_linear.json",
    "pid_angular": "pid_angular.json",
    "pid_vertical": "pid_vertical.json",
    "apf": "apf.json",
    "leader": "leader.json",
    "formation": "formation.json",
    "render": "render.json",
}

DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "simulation": {
        "steps": 2000,
        "dt": 0.1,
        "pool_size": 30.0,
        "z_min": 0.0,
        "z_max": 20.0,
        "spawn_range": 25.0,
        "spawn_z_min": 3.0,
        "spawn_z_max": 12.0,
        "seed": None,
        "draw_every": 1,
        "pause": 0.001,
        "animate": True,
        "save": None,
        "history_limit": 60,
        "log_positions": True,
    },
    "pid_linear": {
        "kp": 1.2,
        "ki": 0.05,
        "kd": 0.2,
        "quadratic_kp": 0.05,
        "max_out": 10.0,
        "integral_limit": 2.0,
    },
    "pid_angular": {
        "kp": 4.0,
        "ki": 0.0,
        "kd": 0.5,
        "max_out": 3.0,
        "integral_limit": 2.0,
    },
    "pid_vertical": {
        "kp": 1.0,
        "ki": 0.03,
        "kd": 0.18,
        "max_out": 1.8,
        "integral_limit": 2.0,
    },
    "apf": {
        "safe_dist": 3.0,
        "repulsion_gain": 8.0,
        "vertical_repulsion_scale": 1.0,
        "obstacle_min_dist": 0.1,
        "horizontal_target_reach_threshold": 0.1,
        "vertical_target_reach_threshold": 0.1,
    },
    "leader": {
        "boundary_margin": 3.0,
        "boundary_speed": 1.0,
        "boundary_turn_gain": 2.5,
        "boundary_turn_deadband": 0.1,
        "cruise_speed": 2.0,
        "cruise_turn_gain": 0.8,
        "cruise_turn_period": 25.0,
        "vertical_enabled": True,
        "vertical_center": 10.0,
        "vertical_amplitude": 5.0,
        "vertical_period": 80.0,
        "vertical_speed_gain": 0.8,
        "vertical_max_speed": 1.2,
    },
    "formation": {
        "follower_1_offset_x": -4.0,
        "follower_1_offset_y": 3.5,
        "follower_1_offset_z": -1.0,
        "follower_2_offset_x": -4.0,
        "follower_2_offset_y": -3.5,
        "follower_2_offset_z": -1.0,
        "follower_3_offset_x": -6.0,
        "follower_3_offset_y": 0.0,
        "follower_3_offset_z": -1.0,
    },
    "render": {
        "figure_width": 13.0,
        "figure_height": 11.0,
        "figure_dpi": 110.0,
        "view_padding": 5.0,
        "z_view_padding": 2.0,
        "title_prefix": "3D Ultimate: PID + APF Formation Control",
        "grid_linestyle": "--",
        "grid_alpha": 0.5,
        "arena_linewidth": 1.6,
        "arena_color": "black",
        "arena_alpha": 0.75,
        "trail_alpha": 0.45,
        "trail_linewidth": 2.0,
        "link_alpha": 0.4,
        "link_color": "black",
        "leader_marker_size": 80,
        "follower_marker_size": 58,
        "target_marker_size": 50,
        "safe_sphere_alpha": 0.12,
        "safe_sphere_resolution": 6,
        "safe_sphere_linewidth": 0.45,
        "arrow_length": 1.8,
        "arrow_linewidth": 1.8,
        "legend_loc": "upper right",
        "camera_elev": 24.0,
        "camera_azim": -58.0,
        "leader_color": "red",
        "follower1_color": "blue",
        "follower2_color": "green",
        "follower3_color": "tab:orange",
        "leader_trail_color": "red",
        "follower1_trail_color": "blue",
        "follower2_trail_color": "green",
        "follower3_trail_color": "tab:orange",
        "target1_color": "tab:blue",
        "target2_color": "tab:green",
        "target3_color": "tab:orange",
        "safe_sphere1_color": "blue",
        "safe_sphere2_color": "green",
        "safe_sphere3_color": "orange",
    },
}


@dataclass
class PIDController:
    """Independent PID controller with output and integral limits."""

    kp: float
    ki: float
    kd: float
    max_out: float
    quadratic_kp: float = 0.0
    integral_limit: float = 2.0
    integral: float = 0.0
    prev_error: float = 0.0

    def reset_integral(self) -> None:
        self.integral = 0.0

    def update(self, error: float, dt: float) -> float:
        p_out = self.kp * error + self.quadratic_kp * error * abs(error)

        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        i_out = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative
        self.prev_error = error

        output = p_out + i_out + d_out
        return float(np.clip(output, -self.max_out, self.max_out))


@dataclass
class UnicycleAgent:
    """Robot state with horizontal unicycle motion and independent vertical motion."""

    agent_id: str
    x: float
    y: float
    z: float
    theta: float
    history_limit: int = 60
    pid_v: PIDController | None = None
    pid_w: PIDController | None = None
    pid_z: PIDController | None = None
    v_xy: float = 0.0
    w: float = 0.0
    v_z: float = 0.0
    history_x: list[float] = field(init=False)
    history_y: list[float] = field(init=False)
    history_z: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.history_x = [self.x]
        self.history_y = [self.y]
        self.history_z = [self.z]

    def update_state(self, v_xy: float, w: float, v_z: float, dt: float) -> None:
        self.v_xy = v_xy
        self.w = w
        self.v_z = v_z
        self.x += v_xy * math.cos(self.theta) * dt
        self.y += v_xy * math.sin(self.theta) * dt
        self.z += v_z * dt
        self.theta += w * dt
        self.theta = wrap_angle(self.theta)

        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_z.append(self.z)
        if len(self.history_x) > self.history_limit:
            self.history_x.pop(0)
            self.history_y.pop(0)
            self.history_z.pop(0)

    def clip_position(self, pool_size: float, z_min: float, z_max: float) -> None:
        self.x = float(np.clip(self.x, -pool_size, pool_size))
        self.y = float(np.clip(self.y, -pool_size, pool_size))
        self.z = float(np.clip(self.z, z_min, z_max))


@dataclass(frozen=True, slots=True)
class Pose:
    """Lightweight snapshot of agent spatial state for obstacle calculations."""

    x: float
    y: float
    z: float
    theta: float


@dataclass(frozen=True, slots=True)
class APFParams:
    """Pre-extracted APF controller parameters."""

    safe_dist: float
    repulsion_gain: float
    vertical_repulsion_scale: float
    obstacle_min_dist: float
    horizontal_reach_threshold: float
    vertical_reach_threshold: float

    @staticmethod
    def from_config(apf_cfg: dict[str, Any]) -> APFParams:
        return APFParams(
            safe_dist=float(apf_cfg["safe_dist"]),
            repulsion_gain=float(apf_cfg["repulsion_gain"]),
            vertical_repulsion_scale=float(apf_cfg["vertical_repulsion_scale"]),
            obstacle_min_dist=float(apf_cfg["obstacle_min_dist"]),
            horizontal_reach_threshold=float(apf_cfg["horizontal_target_reach_threshold"]),
            vertical_reach_threshold=float(apf_cfg["vertical_target_reach_threshold"]),
        )


@dataclass(frozen=True, slots=True)
class LeaderMotionConfig:
    """Pre-extracted leader motion parameters."""

    pool_size: float
    z_min: float
    z_max: float
    boundary_margin: float
    boundary_speed: float
    boundary_turn_gain: float
    boundary_turn_deadband: float
    cruise_speed: float
    cruise_turn_gain: float
    cruise_turn_period: float
    vertical_enabled: bool
    vertical_center: float
    vertical_amplitude: float
    vertical_period: float
    vertical_speed_gain: float
    vertical_max_speed: float

    @staticmethod
    def from_configs(
        simulation_cfg: dict[str, Any], leader_cfg: dict[str, Any]
    ) -> LeaderMotionConfig:
        return LeaderMotionConfig(
            pool_size=float(simulation_cfg["pool_size"]),
            z_min=float(simulation_cfg["z_min"]),
            z_max=float(simulation_cfg["z_max"]),
            boundary_margin=float(leader_cfg["boundary_margin"]),
            boundary_speed=float(leader_cfg["boundary_speed"]),
            boundary_turn_gain=float(leader_cfg["boundary_turn_gain"]),
            boundary_turn_deadband=float(leader_cfg["boundary_turn_deadband"]),
            cruise_speed=float(leader_cfg["cruise_speed"]),
            cruise_turn_gain=float(leader_cfg["cruise_turn_gain"]),
            cruise_turn_period=float(leader_cfg["cruise_turn_period"]),
            vertical_enabled=bool(leader_cfg["vertical_enabled"]),
            vertical_center=float(leader_cfg["vertical_center"]),
            vertical_amplitude=float(leader_cfg["vertical_amplitude"]),
            vertical_period=float(leader_cfg["vertical_period"]),
            vertical_speed_gain=float(leader_cfg["vertical_speed_gain"]),
            vertical_max_speed=float(leader_cfg["vertical_max_speed"]),
        )


@dataclass(frozen=True, slots=True)
class RenderContext:
    """Pre-extracted render + simulation + APF scalars, built once before the loop."""

    pool_size: float
    z_min: float
    z_max: float
    safe_dist: float
    view_limit: float
    z_padding: float
    grid_linestyle: str
    grid_alpha: float
    title_prefix: str
    arena_color: str
    arena_linewidth: float
    arena_alpha: float
    trail_alpha: float
    trail_linewidth: float
    link_alpha: float
    link_color: str
    leader_marker_size: int
    follower_marker_size: int
    target_marker_size: float
    safe_sphere_alpha: float
    safe_sphere_resolution: int
    safe_sphere_linewidth: float
    arrow_length: float
    arrow_linewidth: float
    legend_loc: str
    camera_elev: float
    camera_azim: float
    leader_color: str
    follower_colors: tuple[str, str, str]
    trail_colors: tuple[str, str, str, str]
    target_colors: tuple[str, str, str]
    safe_sphere_colors: tuple[str, str, str]

    @staticmethod
    def from_configs(
        simulation_cfg: dict[str, Any],
        apf_cfg: dict[str, Any],
        render_cfg: dict[str, Any],
    ) -> RenderContext:
        pool_size = float(simulation_cfg["pool_size"])
        z_min = float(simulation_cfg["z_min"])
        z_max = float(simulation_cfg["z_max"])
        return RenderContext(
            pool_size=pool_size,
            z_min=z_min,
            z_max=z_max,
            safe_dist=float(apf_cfg["safe_dist"]),
            view_limit=pool_size + float(render_cfg["view_padding"]),
            z_padding=float(render_cfg["z_view_padding"]),
            grid_linestyle=str(render_cfg["grid_linestyle"]),
            grid_alpha=float(render_cfg["grid_alpha"]),
            title_prefix=str(render_cfg["title_prefix"]),
            arena_color=str(render_cfg["arena_color"]),
            arena_linewidth=float(render_cfg["arena_linewidth"]),
            arena_alpha=float(render_cfg["arena_alpha"]),
            trail_alpha=float(render_cfg["trail_alpha"]),
            trail_linewidth=float(render_cfg["trail_linewidth"]),
            link_alpha=float(render_cfg["link_alpha"]),
            link_color=str(render_cfg["link_color"]),
            leader_marker_size=int(render_cfg["leader_marker_size"]),
            follower_marker_size=int(render_cfg["follower_marker_size"]),
            target_marker_size=float(render_cfg["target_marker_size"]),
            safe_sphere_alpha=float(render_cfg["safe_sphere_alpha"]),
            safe_sphere_resolution=int(render_cfg["safe_sphere_resolution"]),
            safe_sphere_linewidth=float(render_cfg["safe_sphere_linewidth"]),
            arrow_length=float(render_cfg["arrow_length"]),
            arrow_linewidth=float(render_cfg["arrow_linewidth"]),
            legend_loc=str(render_cfg["legend_loc"]),
            camera_elev=float(render_cfg["camera_elev"]),
            camera_azim=float(render_cfg["camera_azim"]),
            leader_color=str(render_cfg["leader_color"]),
            follower_colors=(
                str(render_cfg["follower1_color"]),
                str(render_cfg["follower2_color"]),
                str(render_cfg["follower3_color"]),
            ),
            trail_colors=(
                str(render_cfg["leader_trail_color"]),
                str(render_cfg["follower1_trail_color"]),
                str(render_cfg["follower2_trail_color"]),
                str(render_cfg["follower3_trail_color"]),
            ),
            target_colors=(
                str(render_cfg["target1_color"]),
                str(render_cfg["target2_color"]),
                str(render_cfg["target3_color"]),
            ),
            safe_sphere_colors=(
                str(render_cfg["safe_sphere1_color"]),
                str(render_cfg["safe_sphere2_color"]),
                str(render_cfg["safe_sphere3_color"]),
            ),
        )


@dataclass
class FrameSnapshot:
    """Thread-safe snapshot of an agent for async rendering."""
    x: float
    y: float
    z: float
    theta: float
    hx: list[float]
    hy: list[float]
    hz: list[float]

    @staticmethod
    def from_agent(agent: UnicycleAgent) -> FrameSnapshot:
        return FrameSnapshot(
            x=agent.x, y=agent.y, z=agent.z, theta=agent.theta,
            hx=list(agent.history_x),
            hy=list(agent.history_y),
            hz=list(agent.history_z),
        )


@dataclass
class RenderScene:
    """Snapshot of the full scene for a single frame, sent to renderer thread."""
    leader: FrameSnapshot
    followers: list[FrameSnapshot]
    follower_targets: list[tuple[float, float, float]]
    sim_time: float
    step: int
    is_final: bool = False


_ARENA_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _build_sphere_template(
    radius: float, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * math.pi, resolution)
    v = np.linspace(0, math.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""

    return (angle + math.pi) % (2 * math.pi) - math.pi


def load_json_section(config_dir: Path, section: str) -> dict[str, Any]:
    """Load one JSON section file and merge it with defaults."""

    defaults = dict(DEFAULT_CONFIGS[section])
    path = config_dir / CONFIG_FILE_NAMES[section]
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object")
        defaults.update(loaded)
    return defaults


def load_all_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    config_dir = Path(config_dir)
    return {section: load_json_section(config_dir, section) for section in CONFIG_FILE_NAMES}


def ensure_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def ensure_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")


def validate_configs(configs: dict[str, dict[str, Any]]) -> None:
    simulation = configs["simulation"]
    apf = configs["apf"]
    leader = configs["leader"]
    render = configs["render"]
    pid_linear = configs["pid_linear"]

    ensure_positive("simulation.steps", float(simulation["steps"]))
    ensure_positive("simulation.dt", float(simulation["dt"]))
    ensure_positive("simulation.draw_every", float(simulation["draw_every"]))
    ensure_positive("simulation.pool_size", float(simulation["pool_size"]))
    ensure_positive("simulation.history_limit", float(simulation["history_limit"]))
    ensure_nonnegative("simulation.pause", float(simulation["pause"]))

    z_min = float(simulation["z_min"])
    z_max = float(simulation["z_max"])
    spawn_z_min = float(simulation["spawn_z_min"])
    spawn_z_max = float(simulation["spawn_z_max"])
    if z_max <= z_min:
        raise ValueError("simulation.z_max must be greater than simulation.z_min")
    if spawn_z_max <= spawn_z_min:
        raise ValueError("simulation.spawn_z_max must be greater than simulation.spawn_z_min")
    if spawn_z_min < z_min or spawn_z_max > z_max:
        raise ValueError("simulation spawn z range must be inside [z_min, z_max]")

    ensure_positive("apf.safe_dist", float(apf["safe_dist"]))
    ensure_positive("apf.obstacle_min_dist", float(apf["obstacle_min_dist"]))
    ensure_nonnegative("apf.vertical_repulsion_scale", float(apf["vertical_repulsion_scale"]))
    if float(apf["safe_dist"]) <= float(apf["obstacle_min_dist"]):
        raise ValueError("apf.safe_dist must be greater than apf.obstacle_min_dist")

    ensure_positive("leader.cruise_turn_period", float(leader["cruise_turn_period"]))
    ensure_positive("leader.vertical_period", float(leader["vertical_period"]))
    ensure_nonnegative("leader.vertical_amplitude", float(leader["vertical_amplitude"]))
    ensure_nonnegative("leader.vertical_max_speed", float(leader["vertical_max_speed"]))
    ensure_positive("render.figure_width", float(render["figure_width"]))
    ensure_positive("render.figure_height", float(render["figure_height"]))
    ensure_positive("render.figure_dpi", float(render["figure_dpi"]))
    ensure_positive("render.safe_sphere_resolution", float(render["safe_sphere_resolution"]))
    ensure_positive("pid_linear.max_out", float(pid_linear["max_out"]))
    ensure_nonnegative("pid_linear.quadratic_kp", float(pid_linear.get("quadratic_kp", 0.0)))


def build_pid_controller(config: dict[str, Any]) -> PIDController:
    return PIDController(
        kp=float(config["kp"]),
        ki=float(config["ki"]),
        kd=float(config["kd"]),
        quadratic_kp=float(config.get("quadratic_kp", 0.0)),
        max_out=float(config["max_out"]),
        integral_limit=float(config["integral_limit"]),
    )


def random_pose(
    rng: np.random.Generator,
    spawn_range: float,
    spawn_z_min: float,
    spawn_z_max: float,
) -> tuple[float, float, float, float]:
    return (
        float(rng.uniform(-spawn_range, spawn_range)),
        float(rng.uniform(-spawn_range, spawn_range)),
        float(rng.uniform(spawn_z_min, spawn_z_max)),
        float(rng.uniform(-math.pi, math.pi)),
    )


def create_leader(
    agent_id: str,
    rng: np.random.Generator,
    simulation_cfg: dict[str, Any],
) -> UnicycleAgent:
    x, y, z, theta = random_pose(
        rng,
        float(simulation_cfg["spawn_range"]),
        float(simulation_cfg["spawn_z_min"]),
        float(simulation_cfg["spawn_z_max"]),
    )
    return UnicycleAgent(
        agent_id=agent_id,
        x=x,
        y=y,
        z=z,
        theta=theta,
        history_limit=int(simulation_cfg["history_limit"]),
    )


def create_follower(
    agent_id: str,
    rng: np.random.Generator,
    simulation_cfg: dict[str, Any],
    pid_linear_cfg: dict[str, Any],
    pid_angular_cfg: dict[str, Any],
    pid_vertical_cfg: dict[str, Any],
) -> UnicycleAgent:
    x, y, z, theta = random_pose(
        rng,
        float(simulation_cfg["spawn_range"]),
        float(simulation_cfg["spawn_z_min"]),
        float(simulation_cfg["spawn_z_max"]),
    )
    return UnicycleAgent(
        agent_id=agent_id,
        x=x,
        y=y,
        z=z,
        theta=theta,
        history_limit=int(simulation_cfg["history_limit"]),
        pid_v=build_pid_controller(pid_linear_cfg),
        pid_w=build_pid_controller(pid_angular_cfg),
        pid_z=build_pid_controller(pid_vertical_cfg),
    )


def snapshot_pose(agent: UnicycleAgent) -> Pose:
    """Capture a pose-only copy for obstacle calculations."""

    return Pose(x=agent.x, y=agent.y, z=agent.z, theta=agent.theta)


def get_target_position(
    leader: UnicycleAgent | Pose, offset_x: float, offset_y: float, offset_z: float
) -> tuple[float, float, float]:
    """Convert a formation offset in leader coordinates to world coordinates."""

    target_x = leader.x + offset_x * math.cos(leader.theta) - offset_y * math.sin(
        leader.theta
    )
    target_y = leader.y + offset_x * math.sin(leader.theta) + offset_y * math.cos(
        leader.theta
    )
    target_z = leader.z + offset_z
    return target_x, target_y, target_z


def formation_controller(
    agent: UnicycleAgent,
    target_x: float,
    target_y: float,
    target_z: float,
    obstacles: Iterable[Pose],
    dt: float,
    apf: APFParams,
) -> tuple[float, float, float]:
    """Blend 3D target attraction and 3D APF repulsion, then track with PID."""

    if agent.pid_v is None or agent.pid_w is None or agent.pid_z is None:
        raise ValueError("Follower agents must have pid_v, pid_w, and pid_z controllers")

    att_x = target_x - agent.x
    att_y = target_y - agent.y
    att_z = target_z - agent.z

    # --- Vectorized repulsion (numpy releases GIL → threading parallelism) ---
    rep_x = rep_y = rep_z = 0.0
    obs_list = list(obstacles)
    if obs_list:
        ox = np.array([o.x for o in obs_list])
        oy = np.array([o.y for o in obs_list])
        oz = np.array([o.z for o in obs_list])
        
        dx = agent.x - ox
        dy = agent.y - oy
        dz = agent.z - oz
        dists = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        near = dists < apf.safe_dist
        if np.any(near):
            d_n = dists[near]
            dx_n = dx[near]
            dy_n = dy[near]
            dz_n = dz[near]
            
            eff_d = np.maximum(d_n, apf.obstacle_min_dist)
            strengths = apf.repulsion_gain * (1.0 / eff_d - 1.0 / apf.safe_dist)
            
            pos = strengths > 0
            if np.any(pos):
                s = strengths[pos]
                d_p = d_n[pos]
                dx_p = dx_n[pos]
                dy_p = dy_n[pos]
                dz_p = dz_n[pos]
                
                zero = d_p < 1e-9
                if np.any(zero):
                    ux = np.where(zero, math.cos(agent.theta), dx_p / d_p)
                    uy = np.where(zero, math.sin(agent.theta), dy_p / d_p)
                    uz = np.where(zero, 0.0, dz_p / d_p)
                else:
                    inv = 1.0 / d_p
                    ux = dx_p * inv
                    uy = dy_p * inv
                    uz = dz_p * inv
                
                rep_x = float(np.sum(s * ux))
                rep_y = float(np.sum(s * uy))
                rep_z = float(np.sum(s * uz * apf.vertical_repulsion_scale))

    final_x = att_x + rep_x
    final_y = att_y + rep_y
    final_z = att_z + rep_z

    horizontal_error = math.hypot(final_x, final_y)
    if horizontal_error < apf.horizontal_reach_threshold:
        agent.pid_v.reset_integral()
        v_xy = 0.0
        w = 0.0
    else:
        target_theta = math.atan2(final_y, final_x)
        theta_error = wrap_angle(target_theta - agent.theta)
        v_xy = agent.pid_v.update(horizontal_error, dt)
        w = agent.pid_w.update(theta_error, dt)

    if abs(final_z) < apf.vertical_reach_threshold:
        agent.pid_z.reset_integral()
        v_z = 0.0
    else:
        v_z = agent.pid_z.update(final_z, dt)

    return max(0.0, v_xy), w, v_z


def _compute_follower_controls_threaded(
    followers: list[UnicycleAgent],
    follower_targets: list[tuple[float, float, float]],
    leader_pose: Pose,
    follower_poses: list[Pose],
    dt: float,
    apf_params: APFParams,
    max_workers: int | None = None,
    ) -> list[tuple[float, float, float]]:
    """Compute follower formation controllers in parallel using threads.
    
    Each follower's control computation is independent and uses numpy/math
    operations that partially release the GIL.  Using a ThreadPoolExecutor
    allows the 3 followers to run concurrently within the same process,
    keeping shared mutable PID state accessible.
    """
    n_workers = max_workers or min(len(followers), (os.cpu_count() or 4))
    controls: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(followers)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_map: dict[concurrent.futures.Future, int] = {}
        for idx, follower in enumerate(followers):
            obstacles = [leader_pose] + [
                fp for fi, fp in enumerate(follower_poses) if fi != idx
            ]
            future = pool.submit(
                formation_controller,
                follower,
                *follower_targets[idx],
                obstacles=obstacles,
                dt=dt,
                apf=apf_params,
            )
            future_map[future] = idx
        
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            controls[idx] = future.result()
    
    return controls


def update_leader(
    leader: UnicycleAgent,
    t: int,
    dt: float,
    cfg: LeaderMotionConfig,
) -> tuple[float, float, float]:
    near_wall = (
        abs(leader.x) > (cfg.pool_size - cfg.boundary_margin)
        or abs(leader.y) > (cfg.pool_size - cfg.boundary_margin)
    )

    if near_wall:
        angle_to_center = math.atan2(-leader.y, -leader.x)
        diff = wrap_angle(angle_to_center - leader.theta)
        leader_w = (
            cfg.boundary_turn_gain * float(np.sign(diff))
            if abs(diff) > cfg.boundary_turn_deadband
            else 0.0
        )
        leader_v = cfg.boundary_speed
    else:
        leader_v = cfg.cruise_speed
        leader_w = cfg.cruise_turn_gain * math.sin(t / cfg.cruise_turn_period)

    leader_vz = 0.0
    if cfg.vertical_enabled:
        target_z = cfg.vertical_center + cfg.vertical_amplitude * math.sin(
            t / cfg.vertical_period)
        target_z = float(np.clip(target_z, cfg.z_min, cfg.z_max))
        leader_vz = cfg.vertical_speed_gain * (target_z - leader.z)
        leader_vz = float(np.clip(leader_vz, -cfg.vertical_max_speed, cfg.vertical_max_speed))

    leader.update_state(leader_v, leader_w, leader_vz, dt)
    leader.clip_position(cfg.pool_size, cfg.z_min, cfg.z_max)
    return leader_v, leader_w, leader_vz


def draw_arena_box(ax: plt.Axes, ctx: RenderContext) -> None:
    p, z0, z1 = ctx.pool_size, ctx.z_min, ctx.z_max
    corners = [
        (-p, -p, z0), (p, -p, z0), (p, p, z0), (-p, p, z0),
        (-p, -p, z1), (p, -p, z1), (p, p, z1), (-p, p, z1),
    ]
    for start, end in _ARENA_EDGES:
        c0, c1 = corners[start], corners[end]
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
                color=ctx.arena_color, linewidth=ctx.arena_linewidth,
                alpha=ctx.arena_alpha)


_sphere_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_sphere_template(
    resolution: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (resolution, radius)
    if key not in _sphere_cache:
        _sphere_cache[key] = _build_sphere_template(radius, resolution)
    return _sphere_cache[key]


def draw_safe_sphere(
    ax: plt.Axes,
    agent: UnicycleAgent,
    radius: float,
    color: str,
    ctx: RenderContext,
) -> None:
    sphere_x, sphere_y, sphere_z = _get_sphere_template(
        ctx.safe_sphere_resolution, radius)
    ax.plot_wireframe(
        agent.x + sphere_x, agent.y + sphere_y, agent.z + sphere_z,
        color=color, alpha=ctx.safe_sphere_alpha,
        linewidth=ctx.safe_sphere_linewidth,
    )


def draw_robot(
    ax: plt.Axes,
    agent: UnicycleAgent,
    color: str,
    label: str,
    size: int,
    ctx: RenderContext,
) -> None:
    ax.scatter([agent.x], [agent.y], [agent.z], color=color, s=size,
               label=label, depthshade=True)
    ax.quiver(
        [agent.x], [agent.y], [agent.z],
        [math.cos(agent.theta)], [math.sin(agent.theta)], [0.0],
        length=ctx.arrow_length, normalize=True, color=color,
        linewidth=ctx.arrow_linewidth,
    )


def set_3d_view(ax: plt.Axes, elev: float, azim: float, roll: float = 0.0) -> None:
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def format_agent_state(agent: UnicycleAgent) -> str:
    heading_deg = math.degrees(agent.theta)
    pitch_deg = math.degrees(math.atan2(agent.v_z, agent.v_xy)) if agent.v_xy > 1e-9 else 0.0
    roll_deg = math.degrees(math.atan2(agent.v_xy * agent.w, 9.81))
    return (
        f"{agent.agent_id}=({agent.x:.2f}, {agent.y:.2f}, {agent.z:.2f}) "
        f"Yaw={heading_deg:+.1f}deg Pit={pitch_deg:+.1f}deg Rol={roll_deg:+.1f}deg"
    )


def log_agent_positions(
    step: int,
    sim_time: float,
    leader: UnicycleAgent,
    followers: Sequence[UnicycleAgent],
) -> None:
    lines = [f"[step {step:04d} | t={sim_time:.1f}s]"]
    lines.append("  " + format_agent_state(leader))
    for follower in followers:
        lines.append("  " + format_agent_state(follower))
    print("\n".join(lines), flush=True)


def draw_scene(
    ax: plt.Axes,
    leader: UnicycleAgent,
    followers: Sequence[UnicycleAgent],
    targets: Sequence[tuple[float, float, float]],
    ctx: RenderContext,
    sim_time: float,
) -> None:
    """Incremental draw: static elements set up once, dynamic data updated in-place."""
    first = not getattr(ax, "_lf_has_drawn", False)

    if first:
        # ── Static setup (done once) ──
        ax.set_xlim(-ctx.view_limit, ctx.view_limit)
        ax.set_ylim(-ctx.view_limit, ctx.view_limit)
        ax.set_zlim(ctx.z_min - ctx.z_padding, ctx.z_max + ctx.z_padding)
        ax.set_box_aspect((ctx.view_limit * 2, ctx.view_limit * 2,
                           ctx.z_max - ctx.z_min + ctx.z_padding * 2))
        set_3d_view(ax, ctx.camera_elev, ctx.camera_azim, 0.0)
        ax.grid(True, linestyle=ctx.grid_linestyle, alpha=ctx.grid_alpha)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        draw_arena_box(ax, ctx)

        # Cached artists
        ax._lf_trails = [
            ax.plot([], [], [], color=c, alpha=ctx.trail_alpha,
                    linewidth=ctx.trail_linewidth)[0]
            for c in ctx.trail_colors
        ]
        ax._lf_links = [
            ax.plot([], [], [], color=ctx.link_color, linestyle="--",
                    alpha=ctx.link_alpha)[0]
            for _ in followers
        ]
        ax._lf_targets = [
            ax.scatter([], [], [], marker="x", color=c,
                       s=ctx.target_marker_size, alpha=0.8)
            for c in ctx.target_colors
        ]
        ax._lf_title = ax.set_title("")
        ax._lf_leader_scat = ax.scatter(
            [], [], [], color=ctx.leader_color, s=ctx.leader_marker_size,
            label="Leader", depthshade=True,
        )
        ax._lf_leader_arr = ax.quiver(
            [], [], [], [], [], [],
            color=ctx.leader_color, length=ctx.arrow_length,
            linewidth=ctx.arrow_linewidth,
        )
        ax._lf_foll_scats = []
        ax._lf_foll_arrs = []
        ax._lf_spheres = []
        for i in range(3):
            c = ctx.follower_colors[i]
            sc = ax.scatter([], [], [], color=c, s=ctx.follower_marker_size,
                            label=f"Follower {i+1}", depthshade=True)
            ar = ax.quiver([], [], [], [], [], [],
                           color=c, length=ctx.arrow_length,
                           linewidth=ctx.arrow_linewidth)
            sx, sy, sz = _get_sphere_template(ctx.safe_sphere_resolution, ctx.safe_dist)
            sw = ax.plot_wireframe(sx, sy, sz, color=ctx.safe_sphere_colors[i],
                                   alpha=ctx.safe_sphere_alpha,
                                   linewidth=ctx.safe_sphere_linewidth)
            ax._lf_foll_scats.append(sc)
            ax._lf_foll_arrs.append(ar)
            ax._lf_spheres.append(sw)
        ax.legend(loc=ctx.legend_loc)
        ax._lf_has_drawn = True
    else:
        elev = float(getattr(ax, "elev", ctx.camera_elev))
        azim = float(getattr(ax, "azim", ctx.camera_azim))
        roll = float(getattr(ax, "roll", 0.0))
        set_3d_view(ax, elev, azim, roll)

    # ── Update trails ──
    ax._lf_trails[0].set_data(leader.history_x, leader.history_y)
    ax._lf_trails[0].set_3d_properties(leader.history_z)
    for i, f in enumerate(followers):
        ax._lf_trails[i + 1].set_data(f.history_x, f.history_y)
        ax._lf_trails[i + 1].set_3d_properties(f.history_z)

    # ── Update leader ──
    ax._lf_leader_scat.set_offsets([[leader.x, leader.y]])
    ax._lf_leader_scat.set_3d_properties([leader.z], "z")
    ax._lf_leader_arr.remove()
    ax._lf_leader_arr = ax.quiver(
        [leader.x], [leader.y], [leader.z],
        [math.cos(leader.theta)], [math.sin(leader.theta)], [0.0],
        color=ctx.leader_color, length=ctx.arrow_length,
        linewidth=ctx.arrow_linewidth,
    )

    # ── Update followers ──
    for i, f in enumerate(followers):
        ax._lf_foll_scats[i].set_offsets([[f.x, f.y]])
        ax._lf_foll_scats[i].set_3d_properties([f.z], "z")
        ax._lf_foll_arrs[i].remove()
        ax._lf_foll_arrs[i] = ax.quiver(
            [f.x], [f.y], [f.z],
            [math.cos(f.theta)], [math.sin(f.theta)], [0.0],
            color=ctx.follower_colors[i], length=ctx.arrow_length,
            linewidth=ctx.arrow_linewidth,
        )
        ax._lf_spheres[i].remove()
        sx, sy, sz = _get_sphere_template(ctx.safe_sphere_resolution, ctx.safe_dist)
        ax._lf_spheres[i] = ax.plot_wireframe(
            f.x + sx, f.y + sy, f.z + sz,
            color=ctx.safe_sphere_colors[i], alpha=ctx.safe_sphere_alpha,
            linewidth=ctx.safe_sphere_linewidth,
        )

    # ── Update links ──
    for i, f in enumerate(followers):
        ax._lf_links[i].set_data([leader.x, f.x], [leader.y, f.y])
        ax._lf_links[i].set_3d_properties([leader.z, f.z])

    # ── Update targets ──
    for i, (tx, ty, tz) in enumerate(targets):
        ax._lf_targets[i].set_offsets([[tx, ty]])
        ax._lf_targets[i].set_3d_properties([tz], "z")

    # ── Title ──
    ax._lf_title.set_text(f"{ctx.title_prefix} (Time: {sim_time:.1f}s)")


def run_simulation(
    configs: dict[str, dict[str, Any]],
    use_threads: bool = False,
    max_thread_workers: int | None = None,
) -> None:
    simulation_cfg = configs["simulation"]
    pid_linear_cfg = configs["pid_linear"]
    pid_angular_cfg = configs["pid_angular"]
    pid_vertical_cfg = configs["pid_vertical"]
    apf_cfg = configs["apf"]
    leader_cfg = configs["leader"]
    formation_cfg = configs["formation"]
    render_cfg = configs["render"]

    seed = simulation_cfg["seed"]
    rng = np.random.default_rng(None if seed is None else int(seed))

    pool_size = float(simulation_cfg["pool_size"])
    z_min = float(simulation_cfg["z_min"])
    z_max = float(simulation_cfg["z_max"])
    dt = float(simulation_cfg["dt"])
    steps = int(simulation_cfg["steps"])
    draw_every = int(simulation_cfg["draw_every"])
    pause = float(simulation_cfg["pause"])
    animate = bool(simulation_cfg["animate"])
    save_path = simulation_cfg["save"]
    log_positions = bool(simulation_cfg["log_positions"])

    ctx = RenderContext.from_configs(simulation_cfg, apf_cfg, render_cfg)
    leader_motion = LeaderMotionConfig.from_configs(simulation_cfg, leader_cfg)
    apf_params = APFParams.from_config(apf_cfg)

    leader = create_leader("Leader", rng, simulation_cfg)
    followers = [
        create_follower(
            f"F{index}",
            rng,
            simulation_cfg,
            pid_linear_cfg,
            pid_angular_cfg,
            pid_vertical_cfg,
        )
        for index in range(1, 4)
    ]
    follower_offsets = [
        (
            float(formation_cfg[f"follower_{index}_offset_x"]),
            float(formation_cfg[f"follower_{index}_offset_y"]),
            float(formation_cfg[f"follower_{index}_offset_z"]),
        )
        for index in range(1, 4)
    ]

    fig = plt.figure(
        figsize=(float(render_cfg["figure_width"]), float(render_cfg["figure_height"])),
        dpi=float(render_cfg["figure_dpi"]),
    )
    ax = fig.add_subplot(111, projection="3d")
    last_targets = [get_target_position(leader, *offset) for offset in follower_offsets]

    # ── Phase 1: Record all frames (fast computation, no rendering) ──
    recorded_frames: list[RenderScene] = []
    need_frames = animate or bool(save_path)

    for t in range(steps):
        update_leader(leader, t, dt, leader_motion)
        leader_pose = snapshot_pose(leader)
        follower_poses = [snapshot_pose(follower) for follower in followers]
        follower_targets = [
            get_target_position(leader_pose, *offset) for offset in follower_offsets
        ]

        controls: list[tuple[float, float, float]]
        if use_threads:
            controls = _compute_follower_controls_threaded(
                followers, follower_targets, leader_pose,
                follower_poses, dt, apf_params, max_thread_workers,
            )
        else:
            controls = []
            for index, follower in enumerate(followers):
                obstacles = [leader_pose] + [
                    pose for pose_index, pose in enumerate(follower_poses)
                    if pose_index != index
                ]
                controls.append(
                    formation_controller(
                        follower,
                        *follower_targets[index],
                        obstacles=obstacles,
                        dt=dt,
                        apf=apf_params,
                    )
                )

        for follower, (v_xy, w, v_z) in zip(followers, controls):
            follower.update_state(v_xy, w, v_z, dt)
            follower.clip_position(pool_size, z_min, z_max)
        last_targets = follower_targets

        if need_frames and t % draw_every == 0:
            recorded_frames.append(RenderScene(
                leader=FrameSnapshot.from_agent(leader),
                followers=[FrameSnapshot.from_agent(f) for f in followers],
                follower_targets=list(follower_targets),
                sim_time=t * dt,
                step=t,
                is_final=False,
            ))
            if log_positions:
                log_agent_positions(t, t * dt, leader, followers)

    # Final frame
    recorded_frames.append(RenderScene(
        leader=FrameSnapshot.from_agent(leader),
        followers=[FrameSnapshot.from_agent(f) for f in followers],
        follower_targets=list(last_targets),
        sim_time=steps * dt,
                step=steps,
                 is_final=True,
    ))

    # ── Phase 2: Render animation or save ────────────────────────────
    def _restore(agent: UnicycleAgent, snap: FrameSnapshot) -> None:
        """Restore agent position and history from a recorded snapshot."""
        agent.x = snap.x
        agent.y = snap.y
        agent.z = snap.z
        agent.theta = snap.theta
        agent.history_x = list(snap.hx)
        agent.history_y = list(snap.hy)
        agent.history_z = list(snap.hz)

    n_recorded = len(recorded_frames)
    print(f"Recorded {n_recorded} frames. Rendering...", flush=True)

    if animate:
        plt.show(block=False)
        fig.canvas.draw()

        for idx, rec in enumerate(recorded_frames):
            _restore(leader, rec.leader)
            for f, snap in zip(followers, rec.followers):
                _restore(f, snap)
            draw_scene(ax, leader, followers, rec.follower_targets, ctx, rec.sim_time)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(max(dt * draw_every, 0.001))

        if save_path:
            fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
            print(f"Saved final frame to {save_path}")

        plt.show(block=True)

    else:
        # Non-animated: draw final frame only, then save or show
        rec = recorded_frames[-1]
        _restore(leader, rec.leader)
        for f, snap in zip(followers, rec.followers):
            _restore(f, snap)
        draw_scene(ax, leader, followers, rec.follower_targets, ctx, rec.sim_time)

        if save_path:
            fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
            print(f"Saved final frame to {save_path}")
            plt.close(fig)
        else:
            plt.show()


def _run_simulation_worker(configs: dict[str, dict[str, Any]]) -> None:
    """Worker entry point for multiprocessing batch runs.
    
    Runs a single simulation; called in child processes by run_batch_simulations.
    """
    run_simulation(configs)


def run_batch_simulations(
    configs: dict[str, dict[str, Any]],
    seeds: list[int],
    max_workers: int | None = None,
    save_dir: str | Path | None = None,
    use_threads: bool = False,
    max_thread_workers: int | None = None,
) -> float:
    """Run multiple simulations in parallel using process-based parallelism.
    
    Each simulation runs in its own child process (ProcessPoolExecutor),
    providing true multi-core utilisation.  Ideal for parameter sweeps,
    Monte Carlo runs, or exploring different random seeds.
    
    Args:
        configs:  Base configuration dictionary (seed is overridden per run)
        seeds:    Random seeds, one per simulation.
        max_workers: Max parallel processes (default: min(len(seeds), CPU count)).
        save_dir: Optional directory to save final frames.
        use_threads: Whether each child process also uses intra-step threading.
        max_thread_workers: Threads per child process.
    
    Returns:
        Total wall-clock time in seconds.
    """
    n_workers = max_workers or min(len(seeds), os.cpu_count() or 4)
    
    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Batch Mode  |  {len(seeds)} simulations  |  {n_workers} workers"
          f"{'  |  +threads' if use_threads else ''}")
    print(f"{'='*60}\n")
    
    start = time.perf_counter()
    
    # Prepare per-seed configs (deep-copied to avoid cross-process mutation)
    import copy
    batch_configs: list[dict[str, dict[str, Any]]] = []
    for seed in seeds:
        cfg = copy.deepcopy(configs)
        cfg["simulation"] = dict(cfg["simulation"])
        cfg["simulation"]["seed"] = seed
        cfg["simulation"]["animate"] = False
        cfg["simulation"]["log_positions"] = False
        out = str(save_path / f"sim_seed{seed}.png") if save_path else None
        cfg["simulation"]["save"] = out
        batch_configs.append(cfg)
    
    # Use non-interactive backend in worker processes (inherited via spawn)
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    
    completed = 0
    failed: list[int] = []
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures_map: dict[concurrent.futures.Future, int] = {}
            for cfg, seed in zip(batch_configs, seeds):
                future = executor.submit(_run_simulation_worker, cfg)
                futures_map[future] = seed
            
            for future in concurrent.futures.as_completed(futures_map):
                seed = futures_map[future]
                try:
                    future.result()
                    completed += 1
                    print(f"  \u2713 [{completed}/{len(seeds)}] Seed {seed} done")
                except Exception as e:
                    failed.append(seed)
                    import traceback
                    print(f"  \u2717 Seed {seed} failed: {e}")
                    traceback.print_exc()
    finally:
        # Restore original backend for the main process
        if old_backend is not None:
            os.environ["MPLBACKEND"] = old_backend
        else:
            os.environ.pop("MPLBACKEND", None)
    
    elapsed = time.perf_counter() - start
    print(f"\n  Done: {completed} completed, {len(failed)} failed")
    print(f"  Wall time: {elapsed:.1f}s  |  "
          f"Avg: {elapsed / max(len(seeds), 1):.1f}s/run")
    
    if failed:
        print(f"  Failed seeds: {failed}")
    
    return elapsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate 3D PID + APF formation control for a leader and three followers."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="directory containing the JSON config files",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override simulation.steps from JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override simulation.seed from JSON",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="override simulation.save from JSON",
    )

    animation_group = parser.add_mutually_exclusive_group()
    animation_group.add_argument(
        "--animate",
        dest="animate",
        action="store_true",
        help="override simulation.animate to true",
    )
    animation_group.add_argument(
        "--no-animate",
        dest="animate",
        action="store_false",
        help="override simulation.animate to false",
    )
    parser.set_defaults(animate=None)

    parser.add_argument(
        "--threads",
        action="store_true",
        help="enable intra-step threading for parallel follower control",
    )
    parser.add_argument(
        "--max-thread-workers",
        type=int,
        default=None,
        help="max threads for parallel follower control (default: n_followers)",
    )
    parser.add_argument(
        "--batch-runs",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help="run multiple simulations in parallel (space-separated seeds); "
             "enables --no-animate",
    )
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=None,
        help="max parallel processes for batch mode (default: min(len(seeds), CPU count))",
    )
    parser.add_argument(
        "--batch-save-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="save batch result frames to DIR",
    )

    return parser.parse_args(argv)


def build_runtime_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    configs = load_all_configs(Path(args.config_dir))

    simulation_cfg = dict(configs["simulation"])
    if args.steps is not None:
        simulation_cfg["steps"] = args.steps
    if args.seed is not None:
        simulation_cfg["seed"] = args.seed
    if args.save is not None:
        simulation_cfg["save"] = args.save
    if args.animate is not None:
        simulation_cfg["animate"] = args.animate
    configs["simulation"] = simulation_cfg

    validate_configs(configs)
    return configs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_runs is not None:
        seeds = args.batch_runs
        print(f"Starting batch: {len(seeds)} runs with seeds {seeds}")
        run_batch_simulations(
            build_runtime_configs(args),
            seeds=seeds,
            max_workers=args.batch_workers,
            save_dir=args.batch_save_dir,
            use_threads=args.threads or False,
            max_thread_workers=args.max_thread_workers,
        )
        return
    
    configs = build_runtime_configs(args)
    run_simulation(
        configs,
        use_threads=args.threads or False,
        max_thread_workers=args.max_thread_workers,
    )


if __name__ == "__main__":
    main()
