"""3D formation switching simulation with smooth transitions.

Cycles through Triangle → Line → Diamond formations periodically.

Run:
    python formation_switching_sim.py
    python formation_switching_sim.py --no-animate --steps 1500 --save output.png
    python formation_switching_sim.py --warmup 200 --switch-interval 500 --transition 150
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

import concurrent.futures
import os
import time

from formation_pid_apf_sim import (
    UnicycleAgent, Pose, PIDController,
    RenderContext, APFParams, LeaderMotionConfig,
    wrap_angle, snapshot_pose, get_target_position,
    formation_controller, update_leader,
    draw_scene, format_agent_state,
    load_all_configs, validate_configs, run_batch_simulations,
    RenderScene, FrameSnapshot,
    _compute_follower_controls_threaded,
    create_leader, create_follower,
    BASE_DIR, DEFAULT_CONFIG_DIR,
)

# ── Formation definitions ────────────────────────────────────────────
# (name, [F1_offset, F2_offset, F3_offset]) — offsets in leader's local frame

Offset3 = tuple[float, float, float]
FormationDef = tuple[str, tuple[Offset3, Offset3, Offset3]]

FORMATION_CYCLE: list[FormationDef] = [
    (
        "Triangle",
        (
            (-5.0,  4.5,  0.0),   # F1: back-left
            (-5.0, -4.5,  0.0),   # F2: back-right
            (-9.0,  0.0, -1.0),   # F3: far back center (V tip)
        ),
    ),
    (
        "Line",
        (
            (-5.0,  5.5,  0.0),   # F1: left wing
            (-5.0,  0.0, -1.0),   # F2: center
            (-5.0, -5.5,  0.0),   # F3: right wing
        ),
    ),
    (
        "Diamond",
        (
            (-3.0,  3.0,  0.0),   # F1: back-left
            (-3.0, -3.0,  0.0),   # F2: back-right
            (-6.0,  0.0, -4.0),   # F3: deep center → 3D tetrahedron
        ),
    ),
]


def lerp_offsets(
    old: list[Offset3], new: list[Offset3], t: float
) -> list[Offset3]:
    """Linearly interpolate follower offsets between two formations."""
    return [
        (
            old[i][0] + (new[i][0] - old[i][0]) * t,
            old[i][1] + (new[i][1] - old[i][1]) * t,
            old[i][2] + (new[i][2] - old[i][2]) * t,
        )
        for i in range(3)
    ]


def _announce_switch(step: int, old_name: str, new_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step}:  [{old_name}]  →  [{new_name}]")
    print(f"{'=' * 60}\n", flush=True)


def _log_step(
    step: int,
    sim_time: float,
    formation_name: str,
    transitioning: bool,
    leader: UnicycleAgent,
    followers: Sequence[UnicycleAgent],
) -> None:
    marker = "~" if transitioning else "="
    lines = [f"[step {step:04d} | t={sim_time:.1f}s | {marker} {formation_name} {marker}]"]
    lines.append("  " + format_agent_state(leader))
    for f in followers:
        lines.append("  " + format_agent_state(f))
    print("\n".join(lines), flush=True)


def run_switching_simulation(
    configs: dict[str, dict[str, Any]],
    warmup_steps: int = 60,
    switch_interval: int = 80,
    transition_steps: int = 25,
    use_threads: bool = False,
    max_thread_workers: int | None = None,
) -> None:
    simulation_cfg = configs["simulation"]
    pid_linear_cfg = configs["pid_linear"]
    pid_angular_cfg = configs["pid_angular"]
    pid_vertical_cfg = configs["pid_vertical"]
    apf_cfg = configs["apf"]
    leader_cfg = configs["leader"]
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
            f"F{idx}", rng, simulation_cfg,
            pid_linear_cfg, pid_angular_cfg, pid_vertical_cfg,
        )
        for idx in range(1, 4)
    ]

    # ── Formation state machine ───────────────────────────────────────
    formation_idx = 0
    formation_name = FORMATION_CYCLE[0][0]
    current_offsets: list[Offset3] = list(FORMATION_CYCLE[0][1])
    target_offsets: list[Offset3] = current_offsets
    old_offsets: list[Offset3] = current_offsets

    transition_active = False
    transition_start = 0
    next_switch_step = warmup_steps

    print(f"Initial: [{formation_name}]  |  warmup={warmup_steps}  "
          f"switch_interval={switch_interval}  transition={transition_steps}")

    fig = plt.figure(
        figsize=(float(render_cfg["figure_width"]), float(render_cfg["figure_height"])),
        dpi=float(render_cfg["figure_dpi"]),
    )
    ax = fig.add_subplot(111, projection="3d")
    # ── Phase 1: Record all frames (fast computation, no rendering) ──
    recorded_frames: list[RenderScene] = []
    frame_titles: list[str] = []
    need_frames = animate or bool(save_path)

    for t in range(steps):
        # ── Formation switching ────────────────────────────────────
        if t == next_switch_step:
            old_name = formation_name
            old_offsets = current_offsets
            formation_idx = (formation_idx + 1) % len(FORMATION_CYCLE)
            formation_name, offsets_tuple = FORMATION_CYCLE[formation_idx]
            target_offsets = list(offsets_tuple)
            _announce_switch(t, old_name, formation_name)
            transition_active = True
            transition_start = t
            next_switch_step = t + switch_interval

        if transition_active:
            progress = (t - transition_start) / transition_steps
            if progress >= 1.0:
                progress = 1.0
                transition_active = False
                current_offsets = target_offsets
            else:
                current_offsets = lerp_offsets(old_offsets, target_offsets, progress)

        # ── Simulation step ────────────────────────────────────────
        update_leader(leader, t, dt, leader_motion)
        leader_pose = snapshot_pose(leader)
        follower_poses = [snapshot_pose(f) for f in followers]
        follower_targets = [
            get_target_position(leader_pose, *offset)
            for offset in current_offsets
        ]

        controls: list[tuple[float, float, float]]
        if use_threads:
            controls = _compute_follower_controls_threaded(
                followers, follower_targets, leader_pose,
                follower_poses, dt, apf_params, max_thread_workers,
            )
        else:
            controls = []
            for idx, follower in enumerate(followers):
                obstacles = [leader_pose] + [
                    pose for pi, pose in enumerate(follower_poses) if pi != idx
                ]
                controls.append(
                    formation_controller(
                        follower, *follower_targets[idx],
                        obstacles=obstacles, dt=dt, apf=apf_params,
                    )
                )

        for follower, (v_xy, w, v_z) in zip(followers, controls):
            follower.update_state(v_xy, w, v_z, dt)
            follower.clip_position(pool_size, z_min, z_max)

        # ── Logging & Recording ────────────────────────────────────
        if t % draw_every == 0:
            if log_positions:
                _log_step(t, t * dt, formation_name, transition_active,
                          leader, followers)
            if need_frames:
                recorded_frames.append(RenderScene(
                    leader=FrameSnapshot.from_agent(leader),
                    followers=[FrameSnapshot.from_agent(f) for f in followers],
                    follower_targets=list(follower_targets),
                    sim_time=t * dt,
                    step=t,
                    is_final=False,
                ))
                status = " ~transition~" if transition_active else ""
                frame_titles.append(
                    f"{ctx.title_prefix} [{formation_name}{status}]"
                )

    # Final frame
    final_targets = [
        get_target_position(leader, *offset) for offset in current_offsets
    ]
    recorded_frames.append(RenderScene(
        leader=FrameSnapshot.from_agent(leader),
        followers=[FrameSnapshot.from_agent(f) for f in followers],
        follower_targets=list(final_targets),
        sim_time=steps * dt,
        step=steps,
        is_final=True,
    ))
    frame_titles.append(f"{ctx.title_prefix} [{formation_name}]")

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
            ax.set_title(f"{frame_titles[idx]} (Time: {rec.sim_time:.1f}s)")
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
        ax.set_title(f"{frame_titles[-1]} (Time: {rec.sim_time:.1f}s)")

        if save_path:
            fig.savefig(str(save_path), dpi=160, bbox_inches="tight")
            print(f"Saved final frame to {save_path}")
            plt.close(fig)
        else:
            plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formation switching simulation — cycles through "
                    "Triangle → Line → Diamond with smooth transitions."
    )
    parser.add_argument("--config-dir", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=60,
                        help="steps before first switch (default: 60)")
    parser.add_argument("--switch-interval", type=int, default=80,
                        help="steps between switches (default: 80)")
    parser.add_argument("--transition", type=int, default=25,
                        help="steps for smooth morphing (default: 25)")

    anim = parser.add_mutually_exclusive_group()
    anim.add_argument("--animate", dest="animate", action="store_true")
    anim.add_argument("--no-animate", dest="animate", action="store_false")
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
    sim = dict(configs["simulation"])
    if args.steps is not None:
        sim["steps"] = args.steps
    if args.seed is not None:
        sim["seed"] = args.seed
    if args.save is not None:
        sim["save"] = args.save
    if args.animate is not None:
        sim["animate"] = args.animate
    configs["simulation"] = sim
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
    run_switching_simulation(
        configs,
        warmup_steps=args.warmup,
        switch_interval=args.switch_interval,
        transition_steps=args.transition,
        use_threads=args.threads or False,
        max_thread_workers=args.max_thread_workers,
    )


if __name__ == "__main__":
    main()
