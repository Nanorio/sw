"""PID + APF formation-control simulation for unicycle robots.

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

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = BASE_DIR / "configs"

CONFIG_FILE_NAMES = {
    "simulation": "simulation.json",
    "pid_linear": "pid_linear.json",
    "pid_angular": "pid_angular.json",
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
        "spawn_range": 25.0,
        "seed": None,
        "draw_every": 2,
        "pause": 0.01,
        "animate": True,
        "save": None,
        "history_limit": 40,
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
    "apf": {
        "safe_dist": 3.0,
        "repulsion_gain": 8.0,
        "obstacle_min_dist": 0.1,
        "target_reach_threshold": 0.1,
    },
    "leader": {
        "boundary_margin": 3.0,
        "boundary_speed": 1.0,
        "boundary_turn_gain": 2.5,
        "boundary_turn_deadband": 0.1,
        "cruise_speed": 2.0,
        "cruise_turn_gain": 0.8,
        "cruise_turn_period": 25.0,
    },
    "formation": {
        "follower_1_offset_x": -4.0,
        "follower_1_offset_y": 3.5,
        "follower_2_offset_x": -4.0,
        "follower_2_offset_y": -3.5,
    },
    "render": {
        "figure_width": 9.0,
        "figure_height": 9.0,
        "view_padding": 5.0,
        "title_prefix": "Ultimate: PID + APF Formation Control",
        "grid_linestyle": "--",
        "grid_alpha": 0.5,
        "arena_linewidth": 2.0,
        "arena_color": "black",
        "arena_facecolor": "none",
        "trail_alpha": 0.4,
        "trail_linewidth": 2.0,
        "link_alpha": 0.4,
        "link_color": "black",
        "leader_marker_size": 12,
        "follower_marker_size": 10,
        "target_marker_size": 8,
        "safe_circle_alpha": 0.2,
        "arrow_length": 1.2,
        "arrow_head_width": 0.45,
        "arrow_head_length": 0.6,
        "legend_loc": "upper right",
        "leader_color": "red",
        "follower1_color": "blue",
        "follower2_color": "green",
        "leader_trail_color": "red",
        "follower1_trail_color": "blue",
        "follower2_trail_color": "green",
        "target1_color": "tab:blue",
        "target2_color": "tab:green",
        "safe_circle1_color": "blue",
        "safe_circle2_color": "green",
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
    """Unicycle robot state, history, and optional PID brains."""

    agent_id: str
    x: float
    y: float
    theta: float
    history_limit: int = 40
    pid_v: PIDController | None = None
    pid_w: PIDController | None = None
    history_x: list[float] = field(init=False)
    history_y: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.history_x = [self.x]
        self.history_y = [self.y]

    def update_state(self, v: float, w: float, dt: float) -> None:
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = wrap_angle(self.theta)

        self.history_x.append(self.x)
        self.history_y.append(self.y)
        if len(self.history_x) > self.history_limit:
            self.history_x.pop(0)
            self.history_y.pop(0)

    def clip_position(self, pool_size: float) -> None:
        self.x = float(np.clip(self.x, -pool_size, pool_size))
        self.y = float(np.clip(self.y, -pool_size, pool_size))


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


def validate_configs(configs: dict[str, dict[str, Any]]) -> None:
    simulation = configs["simulation"]
    apf = configs["apf"]
    render = configs["render"]

    ensure_positive("simulation.steps", float(simulation["steps"]))
    ensure_positive("simulation.dt", float(simulation["dt"]))
    ensure_positive("simulation.draw_every", float(simulation["draw_every"]))
    ensure_positive("simulation.pool_size", float(simulation["pool_size"]))
    ensure_positive("simulation.history_limit", float(simulation["history_limit"]))

    ensure_positive("apf.safe_dist", float(apf["safe_dist"]))
    ensure_positive("apf.obstacle_min_dist", float(apf["obstacle_min_dist"]))
    if float(apf["safe_dist"]) <= float(apf["obstacle_min_dist"]):
        raise ValueError("apf.safe_dist must be greater than apf.obstacle_min_dist")

    ensure_positive("leader.cruise_turn_period", float(configs["leader"]["cruise_turn_period"]))
    ensure_positive("render.figure_width", float(render["figure_width"]))
    ensure_positive("render.figure_height", float(render["figure_height"]))


def build_pid_controller(config: dict[str, Any]) -> PIDController:
    return PIDController(
        kp=float(config["kp"]),
        ki=float(config["ki"]),
        kd=float(config["kd"]),
        max_out=float(config["max_out"]),
        integral_limit=float(config["integral_limit"]),
    )


def random_pose(
    rng: np.random.Generator, spawn_range: float
) -> tuple[float, float, float]:
    return (
        float(rng.uniform(-spawn_range, spawn_range)),
        float(rng.uniform(-spawn_range, spawn_range)),
        float(rng.uniform(-math.pi, math.pi)),
    )


def create_leader(
    agent_id: str, rng: np.random.Generator, spawn_range: float, history_limit: int
) -> UnicycleAgent:
    x, y, theta = random_pose(rng, spawn_range)
    return UnicycleAgent(
        agent_id=agent_id,
        x=x,
        y=y,
        theta=theta,
        history_limit=history_limit,
    )


def create_follower(
    agent_id: str,
    rng: np.random.Generator,
    spawn_range: float,
    history_limit: int,
    pid_linear_cfg: dict[str, Any],
    pid_angular_cfg: dict[str, Any],
) -> UnicycleAgent:
    x, y, theta = random_pose(rng, spawn_range)
    return UnicycleAgent(
        agent_id=agent_id,
        x=x,
        y=y,
        theta=theta,
        history_limit=history_limit,
        pid_v=build_pid_controller(pid_linear_cfg),
        pid_w=build_pid_controller(pid_angular_cfg),
    )


def get_target_position(
    leader: UnicycleAgent, offset_x: float, offset_y: float
) -> tuple[float, float]:
    """Convert a formation offset in leader coordinates to world coordinates."""

    target_x = leader.x + offset_x * math.cos(leader.theta) - offset_y * math.sin(
        leader.theta
    )
    target_y = leader.y + offset_x * math.sin(leader.theta) + offset_y * math.cos(
        leader.theta
    )
    return target_x, target_y


def formation_controller(
    agent: UnicycleAgent,
    target_x: float,
    target_y: float,
    obstacles: Iterable[UnicycleAgent],
    dt: float,
    apf_cfg: dict[str, Any],
) -> tuple[float, float]:
    """Blend target attraction and APF repulsion, then track it with PID."""

    if agent.pid_v is None or agent.pid_w is None:
        raise ValueError("Follower agents must have pid_v and pid_w controllers")

    safe_dist = float(apf_cfg["safe_dist"])
    repulsion_gain = float(apf_cfg["repulsion_gain"])
    obstacle_min_dist = float(apf_cfg["obstacle_min_dist"])
    target_reach_threshold = float(apf_cfg["target_reach_threshold"])

    att_x = target_x - agent.x
    att_y = target_y - agent.y

    rep_x = 0.0
    rep_y = 0.0
    for obstacle in obstacles:
        dist = math.hypot(obstacle.x - agent.x, obstacle.y - agent.y)
        if dist < safe_dist:
            effective_dist = max(dist, obstacle_min_dist)
            rep_strength = repulsion_gain * (
                1.0 / effective_dist - 1.0 / safe_dist
            )
            if rep_strength > 0:
                away_angle = math.atan2(agent.y - obstacle.y, agent.x - obstacle.x)
                rep_x += rep_strength * math.cos(away_angle)
                rep_y += rep_strength * math.sin(away_angle)

    final_x = att_x + rep_x
    final_y = att_y + rep_y

    distance_error = math.hypot(final_x, final_y)
    target_theta = math.atan2(final_y, final_x)
    theta_error = wrap_angle(target_theta - agent.theta)

    if distance_error < target_reach_threshold:
        agent.pid_v.reset_integral()
        v = 0.0
    else:
        v = agent.pid_v.update(distance_error, dt)

    w = agent.pid_w.update(theta_error, dt)
    return max(0.0, v), w


def update_leader(
    leader: UnicycleAgent, t: int, dt: float, pool_size: float, leader_cfg: dict[str, Any]
) -> tuple[float, float]:
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
        leader_w = boundary_turn_gain * float(np.sign(diff)) if abs(diff) > boundary_turn_deadband else 0.0
        leader_v = boundary_speed
    else:
        leader_v = cruise_speed
        leader_w = cruise_turn_gain * math.sin(t / cruise_turn_period)

    leader.update_state(leader_v, leader_w, dt)
    leader.clip_position(pool_size)
    return leader_v, leader_w


def draw_robot(
    ax: plt.Axes,
    agent: UnicycleAgent,
    color: str,
    label: str,
    size: int,
    render_cfg: dict[str, Any],
) -> None:
    ax.plot(agent.x, agent.y, marker="o", color=color, markersize=size, label=label)
    arrow_length = float(render_cfg["arrow_length"])
    ax.arrow(
        agent.x,
        agent.y,
        math.cos(agent.theta) * arrow_length,
        math.sin(agent.theta) * arrow_length,
        head_width=float(render_cfg["arrow_head_width"]),
        head_length=float(render_cfg["arrow_head_length"]),
        fc=color,
        ec=color,
        alpha=0.8,
        length_includes_head=True,
        zorder=4,
    )


def draw_scene(
    ax: plt.Axes,
    leader: UnicycleAgent,
    follower_1: UnicycleAgent,
    follower_2: UnicycleAgent,
    f1_target: tuple[float, float],
    f2_target: tuple[float, float],
    simulation_cfg: dict[str, Any],
    apf_cfg: dict[str, Any],
    render_cfg: dict[str, Any],
    sim_time: float,
) -> None:
    pool_size = float(simulation_cfg["pool_size"])
    safe_dist = float(apf_cfg["safe_dist"])
    view_limit = pool_size + float(render_cfg["view_padding"])

    ax.cla()
    ax.set_xlim(-view_limit, view_limit)
    ax.set_ylim(-view_limit, view_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(
        True,
        linestyle=str(render_cfg["grid_linestyle"]),
        alpha=float(render_cfg["grid_alpha"]),
    )
    ax.set_title(f'{render_cfg["title_prefix"]} (Time: {sim_time:.1f}s)')

    arena = patches.Rectangle(
        (-pool_size, -pool_size),
        pool_size * 2,
        pool_size * 2,
        linewidth=float(render_cfg["arena_linewidth"]),
        edgecolor=str(render_cfg["arena_color"]),
        facecolor=str(render_cfg["arena_facecolor"]),
    )
    ax.add_patch(arena)

    ax.plot(
        leader.history_x,
        leader.history_y,
        color=str(render_cfg["leader_trail_color"]),
        alpha=float(render_cfg["trail_alpha"]),
        linewidth=float(render_cfg["trail_linewidth"]),
    )
    ax.plot(
        follower_1.history_x,
        follower_1.history_y,
        color=str(render_cfg["follower1_trail_color"]),
        alpha=float(render_cfg["trail_alpha"]),
        linewidth=float(render_cfg["trail_linewidth"]),
    )
    ax.plot(
        follower_2.history_x,
        follower_2.history_y,
        color=str(render_cfg["follower2_trail_color"]),
        alpha=float(render_cfg["trail_alpha"]),
        linewidth=float(render_cfg["trail_linewidth"]),
    )

    ax.plot(
        [leader.x, follower_1.x],
        [leader.y, follower_1.y],
        color=str(render_cfg["link_color"]),
        linestyle="--",
        alpha=float(render_cfg["link_alpha"]),
    )
    ax.plot(
        [leader.x, follower_2.x],
        [leader.y, follower_2.y],
        color=str(render_cfg["link_color"]),
        linestyle="--",
        alpha=float(render_cfg["link_alpha"]),
    )
    ax.plot(
        *f1_target,
        marker="x",
        color=str(render_cfg["target1_color"]),
        markersize=float(render_cfg["target_marker_size"]),
        alpha=0.7,
    )
    ax.plot(
        *f2_target,
        marker="x",
        color=str(render_cfg["target2_color"]),
        markersize=float(render_cfg["target_marker_size"]),
        alpha=0.7,
    )

    draw_robot(
        ax,
        leader,
        color=str(render_cfg["leader_color"]),
        label="Leader",
        size=int(render_cfg["leader_marker_size"]),
        render_cfg=render_cfg,
    )
    draw_robot(
        ax,
        follower_1,
        color=str(render_cfg["follower1_color"]),
        label="Follower 1",
        size=int(render_cfg["follower_marker_size"]),
        render_cfg=render_cfg,
    )
    draw_robot(
        ax,
        follower_2,
        color=str(render_cfg["follower2_color"]),
        label="Follower 2",
        size=int(render_cfg["follower_marker_size"]),
        render_cfg=render_cfg,
    )

    ax.add_patch(
        patches.Circle(
            (follower_1.x, follower_1.y),
            safe_dist,
            fill=False,
            edgecolor=str(render_cfg["safe_circle1_color"]),
            alpha=float(render_cfg["safe_circle_alpha"]),
        )
    )
    ax.add_patch(
        patches.Circle(
            (follower_2.x, follower_2.y),
            safe_dist,
            fill=False,
            edgecolor=str(render_cfg["safe_circle2_color"]),
            alpha=float(render_cfg["safe_circle_alpha"]),
        )
    )

    ax.legend(loc=str(render_cfg["legend_loc"]))


def run_simulation(configs: dict[str, dict[str, Any]]) -> None:
    simulation_cfg = configs["simulation"]
    pid_linear_cfg = configs["pid_linear"]
    pid_angular_cfg = configs["pid_angular"]
    apf_cfg = configs["apf"]
    leader_cfg = configs["leader"]
    formation_cfg = configs["formation"]
    render_cfg = configs["render"]

    seed = simulation_cfg["seed"]
    rng = np.random.default_rng(None if seed is None else int(seed))

    history_limit = int(simulation_cfg["history_limit"])
    spawn_range = float(simulation_cfg["spawn_range"])
    pool_size = float(simulation_cfg["pool_size"])
    dt = float(simulation_cfg["dt"])
    steps = int(simulation_cfg["steps"])
    draw_every = int(simulation_cfg["draw_every"])
    pause = float(simulation_cfg["pause"])
    animate = bool(simulation_cfg["animate"])
    save_path = simulation_cfg["save"]

    leader = create_leader("Leader", rng, spawn_range, history_limit)
    follower_1 = create_follower(
        "F1",
        rng,
        spawn_range,
        history_limit,
        pid_linear_cfg,
        pid_angular_cfg,
    )
    follower_2 = create_follower(
        "F2",
        rng,
        spawn_range,
        history_limit,
        pid_linear_cfg,
        pid_angular_cfg,
    )

    f1_offset = (
        float(formation_cfg["follower_1_offset_x"]),
        float(formation_cfg["follower_1_offset_y"]),
    )
    f2_offset = (
        float(formation_cfg["follower_2_offset_x"]),
        float(formation_cfg["follower_2_offset_y"]),
    )

    if animate:
        plt.ion()

    fig, ax = plt.subplots(
        figsize=(float(render_cfg["figure_width"]), float(render_cfg["figure_height"]))
    )
    last_targets = ((leader.x, leader.y), (leader.x, leader.y))

    for t in range(steps):
        update_leader(leader, t, dt, pool_size, leader_cfg)

        f1_target = get_target_position(leader, *f1_offset)
        f1_v, f1_w = formation_controller(
            follower_1,
            *f1_target,
            obstacles=[follower_2, leader],
            dt=dt,
            apf_cfg=apf_cfg,
        )
        follower_1.update_state(f1_v, f1_w, dt)
        follower_1.clip_position(pool_size)

        f2_target = get_target_position(leader, *f2_offset)
        f2_v, f2_w = formation_controller(
            follower_2,
            *f2_target,
            obstacles=[follower_1, leader],
            dt=dt,
            apf_cfg=apf_cfg,
        )
        follower_2.update_state(f2_v, f2_w, dt)
        follower_2.clip_position(pool_size)
        last_targets = (f1_target, f2_target)

        if animate and t % draw_every == 0:
            draw_scene(
                ax,
                leader,
                follower_1,
                follower_2,
                f1_target,
                f2_target,
                simulation_cfg,
                apf_cfg,
                render_cfg,
                t * dt,
            )
            plt.pause(pause)

    draw_scene(
        ax,
        leader,
        follower_1,
        follower_2,
        last_targets[0],
        last_targets[1],
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
        description="Simulate PID + APF formation control for three unicycle robots."
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
