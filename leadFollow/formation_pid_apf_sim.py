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

import matplotlib.pyplot as plt
import numpy as np


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
        "max_out": 3.5,
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
    integral_limit: float = 2.0
    integral: float = 0.0
    prev_error: float = 0.0

    def reset_integral(self) -> None:
        self.integral = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        p_out = self.kp * error

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
    history_x: list[float] = field(init=False)
    history_y: list[float] = field(init=False)
    history_z: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.history_x = [self.x]
        self.history_y = [self.y]
        self.history_z = [self.z]

    def update_state(self, v_xy: float, w: float, v_z: float, dt: float) -> None:
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


def build_pid_controller(config: dict[str, Any]) -> PIDController:
    return PIDController(
        kp=float(config["kp"]),
        ki=float(config["ki"]),
        kd=float(config["kd"]),
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


def snapshot_agent(agent: UnicycleAgent) -> UnicycleAgent:
    """Capture a pose-only copy for obstacle calculations."""

    return UnicycleAgent(
        agent_id=agent.agent_id,
        x=agent.x,
        y=agent.y,
        z=agent.z,
        theta=agent.theta,
        history_limit=1,
    )


def get_target_position(
    leader: UnicycleAgent, offset_x: float, offset_y: float, offset_z: float
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
    obstacles: Iterable[UnicycleAgent],
    dt: float,
    apf_cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """Blend 3D target attraction and 3D APF repulsion, then track with PID."""

    if agent.pid_v is None or agent.pid_w is None or agent.pid_z is None:
        raise ValueError("Follower agents must have pid_v, pid_w, and pid_z controllers")

    safe_dist = float(apf_cfg["safe_dist"])
    repulsion_gain = float(apf_cfg["repulsion_gain"])
    vertical_repulsion_scale = float(apf_cfg["vertical_repulsion_scale"])
    obstacle_min_dist = float(apf_cfg["obstacle_min_dist"])
    horizontal_reach_threshold = float(apf_cfg["horizontal_target_reach_threshold"])
    vertical_reach_threshold = float(apf_cfg["vertical_target_reach_threshold"])

    att_x = target_x - agent.x
    att_y = target_y - agent.y
    att_z = target_z - agent.z

    rep_x = 0.0
    rep_y = 0.0
    rep_z = 0.0
    for obstacle in obstacles:
        dx = agent.x - obstacle.x
        dy = agent.y - obstacle.y
        dz = agent.z - obstacle.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < safe_dist:
            effective_dist = max(dist, obstacle_min_dist)
            rep_strength = repulsion_gain * (
                1.0 / effective_dist - 1.0 / safe_dist
            )
            if rep_strength > 0:
                if dist > 1e-9:
                    unit_x = dx / dist
                    unit_y = dy / dist
                    unit_z = dz / dist
                else:
                    unit_x = math.cos(agent.theta)
                    unit_y = math.sin(agent.theta)
                    unit_z = 0.0
                rep_x += rep_strength * unit_x
                rep_y += rep_strength * unit_y
                rep_z += rep_strength * unit_z * vertical_repulsion_scale

    final_x = att_x + rep_x
    final_y = att_y + rep_y
    final_z = att_z + rep_z

    horizontal_error = math.hypot(final_x, final_y)
    if horizontal_error < horizontal_reach_threshold:
        agent.pid_v.reset_integral()
        v_xy = 0.0
        w = 0.0
    else:
        target_theta = math.atan2(final_y, final_x)
        theta_error = wrap_angle(target_theta - agent.theta)
        v_xy = agent.pid_v.update(horizontal_error, dt)
        w = agent.pid_w.update(theta_error, dt)

    if abs(final_z) < vertical_reach_threshold:
        agent.pid_z.reset_integral()
        v_z = 0.0
    else:
        v_z = agent.pid_z.update(final_z, dt)

    return max(0.0, v_xy), w, v_z


def update_leader(
    leader: UnicycleAgent,
    t: int,
    dt: float,
    simulation_cfg: dict[str, Any],
    leader_cfg: dict[str, Any],
) -> tuple[float, float, float]:
    pool_size = float(simulation_cfg["pool_size"])
    z_min = float(simulation_cfg["z_min"])
    z_max = float(simulation_cfg["z_max"])

    margin = float(leader_cfg["boundary_margin"])
    boundary_speed = float(leader_cfg["boundary_speed"])
    boundary_turn_gain = float(leader_cfg["boundary_turn_gain"])
    boundary_turn_deadband = float(leader_cfg["boundary_turn_deadband"])
    cruise_speed = float(leader_cfg["cruise_speed"])
    cruise_turn_gain = float(leader_cfg["cruise_turn_gain"])
    cruise_turn_period = float(leader_cfg["cruise_turn_period"])

    near_wall = abs(leader.x) > (pool_size - margin) or abs(leader.y) > (
        pool_size - margin
    )

    if near_wall:
        angle_to_center = math.atan2(-leader.y, -leader.x)
        diff = wrap_angle(angle_to_center - leader.theta)
        leader_w = (
            boundary_turn_gain * float(np.sign(diff))
            if abs(diff) > boundary_turn_deadband
            else 0.0
        )
        leader_v = boundary_speed
    else:
        leader_v = cruise_speed
        leader_w = cruise_turn_gain * math.sin(t / cruise_turn_period)

    leader_vz = 0.0
    if bool(leader_cfg["vertical_enabled"]):
        center = float(leader_cfg["vertical_center"])
        amplitude = float(leader_cfg["vertical_amplitude"])
        period = float(leader_cfg["vertical_period"])
        speed_gain = float(leader_cfg["vertical_speed_gain"])
        max_speed = float(leader_cfg["vertical_max_speed"])

        target_z = center + amplitude * math.sin(t / period)
        target_z = float(np.clip(target_z, z_min, z_max))
        leader_vz = speed_gain * (target_z - leader.z)
        leader_vz = float(np.clip(leader_vz, -max_speed, max_speed))

    leader.update_state(leader_v, leader_w, leader_vz, dt)
    leader.clip_position(pool_size, z_min, z_max)
    return leader_v, leader_w, leader_vz


def draw_arena_box(
    ax: plt.Axes,
    pool_size: float,
    z_min: float,
    z_max: float,
    render_cfg: dict[str, Any],
) -> None:
    color = str(render_cfg["arena_color"])
    linewidth = float(render_cfg["arena_linewidth"])
    alpha = float(render_cfg["arena_alpha"])

    corners = [
        (-pool_size, -pool_size, z_min),
        (pool_size, -pool_size, z_min),
        (pool_size, pool_size, z_min),
        (-pool_size, pool_size, z_min),
        (-pool_size, -pool_size, z_max),
        (pool_size, -pool_size, z_max),
        (pool_size, pool_size, z_max),
        (-pool_size, pool_size, z_max),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        xs = [corners[start][0], corners[end][0]]
        ys = [corners[start][1], corners[end][1]]
        zs = [corners[start][2], corners[end][2]]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)


def draw_safe_sphere(
    ax: plt.Axes,
    agent: UnicycleAgent,
    radius: float,
    color: str,
    render_cfg: dict[str, Any],
) -> None:
    resolution = int(render_cfg["safe_sphere_resolution"])
    u = np.linspace(0, 2 * math.pi, resolution)
    v = np.linspace(0, math.pi, resolution)
    xs = agent.x + radius * np.outer(np.cos(u), np.sin(v))
    ys = agent.y + radius * np.outer(np.sin(u), np.sin(v))
    zs = agent.z + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(
        xs,
        ys,
        zs,
        color=color,
        alpha=float(render_cfg["safe_sphere_alpha"]),
        linewidth=float(render_cfg["safe_sphere_linewidth"]),
    )


def draw_robot(
    ax: plt.Axes,
    agent: UnicycleAgent,
    color: str,
    label: str,
    size: int,
    render_cfg: dict[str, Any],
) -> None:
    ax.scatter([agent.x], [agent.y], [agent.z], color=color, s=size, label=label, depthshade=True)

    arrow_length = float(render_cfg["arrow_length"])
    ax.quiver(
        [agent.x],
        [agent.y],
        [agent.z],
        [math.cos(agent.theta)],
        [math.sin(agent.theta)],
        [0.0],
        length=arrow_length,
        normalize=True,
        color=color,
        linewidth=float(render_cfg["arrow_linewidth"]),
    )


def set_3d_view(ax: plt.Axes, elev: float, azim: float, roll: float = 0.0) -> None:
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def format_agent_position(agent: UnicycleAgent) -> str:
    return f"{agent.agent_id}=({agent.x:.2f}, {agent.y:.2f}, {agent.z:.2f})"


def log_agent_positions(
    step: int,
    sim_time: float,
    leader: UnicycleAgent,
    followers: Sequence[UnicycleAgent],
) -> None:
    positions = [format_agent_position(leader)] + [
        format_agent_position(follower) for follower in followers
    ]
    print(f"[step {step:04d} | t={sim_time:.1f}s] " + " | ".join(positions), flush=True)


def draw_scene(
    ax: plt.Axes,
    leader: UnicycleAgent,
    followers: Sequence[UnicycleAgent],
    targets: Sequence[tuple[float, float, float]],
    simulation_cfg: dict[str, Any],
    apf_cfg: dict[str, Any],
    render_cfg: dict[str, Any],
    sim_time: float,
) -> None:
    pool_size = float(simulation_cfg["pool_size"])
    z_min = float(simulation_cfg["z_min"])
    z_max = float(simulation_cfg["z_max"])
    safe_dist = float(apf_cfg["safe_dist"])
    view_limit = pool_size + float(render_cfg["view_padding"])
    z_padding = float(render_cfg["z_view_padding"])
    has_drawn = getattr(ax, "_lf_has_drawn", False)

    if has_drawn:
        elev = float(getattr(ax, "elev", float(render_cfg["camera_elev"])))
        azim = float(getattr(ax, "azim", float(render_cfg["camera_azim"])))
        roll = float(getattr(ax, "roll", 0.0))
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        dist = getattr(ax, "dist", getattr(ax, "_dist", None))
        zoom = getattr(ax, "_zoom", None)
    else:
        elev = float(render_cfg["camera_elev"])
        azim = float(render_cfg["camera_azim"])
        roll = 0.0
        xlim = (-view_limit, view_limit)
        ylim = (-view_limit, view_limit)
        zlim = (z_min - z_padding, z_max + z_padding)
        dist = getattr(ax, "dist", getattr(ax, "_dist", None))
        zoom = getattr(ax, "_zoom", None)

    ax.cla()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((view_limit * 2, view_limit * 2, z_max - z_min + z_padding * 2))
    set_3d_view(ax, elev, azim, roll)
    if dist is not None and hasattr(ax, "dist"):
        ax.dist = dist
    if zoom is not None and hasattr(ax, "_zoom"):
        ax._zoom = zoom
    ax.grid(
        True,
        linestyle=str(render_cfg["grid_linestyle"]),
        alpha=float(render_cfg["grid_alpha"]),
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'{render_cfg["title_prefix"]} (Time: {sim_time:.1f}s)')

    draw_arena_box(ax, pool_size, z_min, z_max, render_cfg)

    ax.plot(
        leader.history_x,
        leader.history_y,
        leader.history_z,
        color=str(render_cfg["leader_trail_color"]),
        alpha=float(render_cfg["trail_alpha"]),
        linewidth=float(render_cfg["trail_linewidth"]),
    )
    for index, (follower, target) in enumerate(zip(followers, targets), start=1):
        ax.plot(
            follower.history_x,
            follower.history_y,
            follower.history_z,
            color=str(render_cfg[f"follower{index}_trail_color"]),
            alpha=float(render_cfg["trail_alpha"]),
            linewidth=float(render_cfg["trail_linewidth"]),
        )
        ax.plot(
            [leader.x, follower.x],
            [leader.y, follower.y],
            [leader.z, follower.z],
            color=str(render_cfg["link_color"]),
            linestyle="--",
            alpha=float(render_cfg["link_alpha"]),
        )
        ax.scatter(
            [target[0]],
            [target[1]],
            [target[2]],
            marker="x",
            color=str(render_cfg[f"target{index}_color"]),
            s=float(render_cfg["target_marker_size"]),
            alpha=0.8,
        )

    draw_robot(
        ax,
        leader,
        color=str(render_cfg["leader_color"]),
        label="Leader",
        size=int(render_cfg["leader_marker_size"]),
        render_cfg=render_cfg,
    )
    for index, follower in enumerate(followers, start=1):
        draw_robot(
            ax,
            follower,
            color=str(render_cfg[f"follower{index}_color"]),
            label=f"Follower {index}",
            size=int(render_cfg["follower_marker_size"]),
            render_cfg=render_cfg,
        )
        draw_safe_sphere(
            ax,
            follower,
            safe_dist,
            str(render_cfg[f"safe_sphere{index}_color"]),
            render_cfg,
        )

    ax.legend(loc=str(render_cfg["legend_loc"]))
    ax._lf_has_drawn = True


def run_simulation(configs: dict[str, dict[str, Any]]) -> None:
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

    if animate:
        plt.ion()

    fig = plt.figure(
        figsize=(float(render_cfg["figure_width"]), float(render_cfg["figure_height"])),
        dpi=float(render_cfg["figure_dpi"]),
    )
    ax = fig.add_subplot(111, projection="3d")
    last_targets = [get_target_position(leader, *offset) for offset in follower_offsets]

    for t in range(steps):
        update_leader(leader, t, dt, simulation_cfg, leader_cfg)
        leader_snapshot = snapshot_agent(leader)
        follower_snapshots = [snapshot_agent(follower) for follower in followers]
        follower_targets = [
            get_target_position(leader_snapshot, *offset) for offset in follower_offsets
        ]

        controls: list[tuple[float, float, float]] = []
        for index, follower in enumerate(followers):
            obstacles = [leader_snapshot] + [
                snapshot
                for snapshot_index, snapshot in enumerate(follower_snapshots)
                if snapshot_index != index
            ]
            controls.append(
                formation_controller(
                    follower,
                    *follower_targets[index],
                    obstacles=obstacles,
                    dt=dt,
                    apf_cfg=apf_cfg,
                )
            )

        for follower, (v_xy, w, v_z) in zip(followers, controls):
            follower.update_state(v_xy, w, v_z, dt)
            follower.clip_position(pool_size, z_min, z_max)
        last_targets = follower_targets

        if t % draw_every == 0:
            if log_positions:
                log_agent_positions(t, t * dt, leader, followers)
        if animate and t % draw_every == 0:
            draw_scene(
                ax,
                leader,
                followers,
                follower_targets,
                simulation_cfg,
                apf_cfg,
                render_cfg,
                t * dt,
            )
            plt.pause(pause)

    draw_scene(
        ax,
        leader,
        followers,
        last_targets,
        simulation_cfg,
        apf_cfg,
        render_cfg,
        steps * dt,
    )

    if save_path:
        fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"Saved final frame to {save_path}")

    if animate or not save_path:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


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
    configs = build_runtime_configs(args)
    run_simulation(configs)


if __name__ == "__main__":
    main()
