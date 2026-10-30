"""Animate a personalised PageRank random walk with teleportation.

Requirements:
    python -m pip install matplotlib networkx numpy

Run:
    python personalised_pagerank_animation.py

The walker follows a randomly selected outgoing edge with probability ALPHA.
With probability 1 - ALPHA it teleports to the personalised target node. Each
visit adds a red glow to that node and grows it slightly; the glow fades back
to white. The readout compares the walk's visit frequencies with the exact
personalised PageRank vector.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Easy-to-change settings
# ---------------------------------------------------------------------------
NODE_COUNT = 50              # Number of nodes in the network (at least 3)
ALPHA = 0.80                  # Edge-following probability; teleport is 1 - ALPHA
WALKER_INTERVAL_MS = 80     # Time per walker step, in milliseconds
SUBSTEPS = 4                 # Animation frames per step (higher = smoother)
FADE_PER_STEP = 0.004        # Heat decay per walker step (0 to 1)
RANDOM_SEED = 7              # Set to None for a different network each run

# Saving
SAVE_GIF = True
SAVE_STEPS = 400             # Walker steps to render into the GIF
SAVE_DPI = 140                # Lower = smaller file

# Network/layout settings
EDGE_PROBABILITY = 0.03
LAYOUT_SEED = 17
NODE_SIZE = 50
EDGE_WIDTH = 1.15


def make_connected_graph(node_count: int, seed: int | None) -> nx.Graph:
    """Make a connected, undirected graph so every node is reachable."""
    if node_count < 3:
        raise ValueError("NODE_COUNT must be at least 3.")

    graph = nx.random_labeled_tree(node_count, seed=seed)
    rng = random.Random(seed)
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if not graph.has_edge(left, right) and rng.random() < EDGE_PROBABILITY:
                graph.add_edge(left, right)
    return graph


def build_animation() -> Tuple[FuncAnimation, plt.Figure]:
    if not 0.0 <= ALPHA <= 1.0:
        raise ValueError("ALPHA must be between 0 and 1, inclusive.")
    if not 0.0 <= FADE_PER_STEP <= 1.0:
        raise ValueError("FADE_PER_STEP must be between 0 and 1, inclusive.")
    if WALKER_INTERVAL_MS <= 0 or SUBSTEPS < 1:
        raise ValueError("WALKER_INTERVAL_MS and SUBSTEPS must be positive.")

    graph = make_connected_graph(NODE_COUNT, RANDOM_SEED)

    positions = nx.spring_layout(graph, seed=LAYOUT_SEED, iterations=100)
    target = max(graph.degree, key=lambda item: item[1])[0]
    origin = positions[target].copy()
    positions = {node: point - origin for node, point in positions.items()}
    max_radius = max(np.linalg.norm(point) for point in positions.values()) or 1.0
    positions = {node: point / max_radius for node, point in positions.items()}

    # Exact personalised PageRank, used as the "ground truth" for the readout.
    exact = nx.pagerank(graph, alpha=ALPHA, personalization={target: 1.0})
    exact_vec = np.array([exact[n] for n in range(NODE_COUNT)])

    neighbors: Dict[int, list[int]] = {
        node: sorted(graph.neighbors(node)) for node in graph.nodes
    }
    rng = random.Random(None if RANDOM_SEED is None else RANDOM_SEED + 1)

    # Walk state: the walker is travelling from `current` to `nxt`.
    state = {"current": target, "nxt": target, "kind": "start", "steps": 0}
    heat = np.zeros(NODE_COUNT, dtype=float)
    visits = np.zeros(NODE_COUNT, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#20242b")
    ax.set_facecolor("#20242b")
    ax.set_title(
        "Personalised PageRank: random walk with resets",
        color="white",
        fontsize=20,
        pad=16,
    )
    ax.axis("off")
    ax.set_aspect("equal")

    edge_segments = [(positions[a], positions[b]) for a, b in graph.edges]
    ax.add_collection(
        LineCollection(
            edge_segments,
            colors=(0.78, 0.82, 0.88, 0.55),
            linewidths=EDGE_WIDTH,
            zorder=1,
        )
    )

    # Highlight for the edge currently being traversed.
    (trail,) = ax.plot([], [], lw=3, alpha=0.9, zorder=1.5, solid_capstyle="round")

    xs = np.array([positions[node][0] for node in graph.nodes])
    ys = np.array([positions[node][1] for node in graph.nodes])
    node_artist = ax.scatter(
        xs,
        ys,
        s=NODE_SIZE,
        c=np.ones((NODE_COUNT, 4)),
        edgecolors="#333943",
        linewidths=1.2,
        zorder=2,
    )

    target_artist = ax.scatter(
        [positions[target][0]],
        [positions[target][1]],
        s=NODE_SIZE * 1.55,
        facecolors="none",
        edgecolors="#ffd166",
        linewidths=2.1,
        zorder=3,
    )
    walker_artist = ax.scatter(
        [positions[target][0]],
        [positions[target][1]],
        s=NODE_SIZE * 0.30,
        c="#2864ff",
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )

    info_artist = ax.text(
        0.02, 0.02, "", transform=ax.transAxes,
        color="white", fontsize=14, va="bottom", family="monospace",
    )

    margin = 1.08    # was 1.25; the layout spans -1 to 1, so this leaves a small border
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    red = np.array([0.95, 0.12, 0.10])
    trail_colors = {"edge": "#8ecbff", "RESET": "#ffd166", "start": "#8ecbff"}

    def update(frame: int):
        sub = frame % SUBSTEPS

        if sub == 0:
            # Arrive at the node we were travelling to, then pick the next move.
            current = state["nxt"]
            state["current"] = current
            state["steps"] += 1
            visits[current] += 1
            heat[current] = min(1.0, heat[current] + 0.32)

            if rng.random() < ALPHA:
                state["nxt"] = rng.choice(neighbors[current])
                state["kind"] = "edge"
            else:
                state["nxt"] = target
                state["kind"] = "RESET"

            # Node size grows with the share of visits so far.
            share = visits / visits.max()
            sizes = NODE_SIZE * (0.7 + 2.3 * np.sqrt(share))
            node_artist.set_sizes(sizes)
            target_artist.set_sizes([sizes[target] * 1.55])

        # Fade a little every frame so the decay is smooth.
        heat[:] = np.maximum(0.0, heat - FADE_PER_STEP / SUBSTEPS)
        colors = np.ones((NODE_COUNT, 4), dtype=float)
        colors[:, :3] = 1.0 - heat[:, None] * (1.0 - red[None, :])
        node_artist.set_facecolors(colors)

        # Glide from current to next with smoothstep easing.
        a = positions[state["current"]]
        b = positions[state["nxt"]]
        t = (sub + 1) / SUBSTEPS
        t = t * t * (3 - 2 * t)
        walker_artist.set_offsets([a + (b - a) * t])

        trail.set_data([a[0], b[0]], [a[1], b[1]])
        trail.set_color(trail_colors[state["kind"]])

        total = visits.sum()
        l1 = np.abs(visits / total - exact_vec).sum() if total else float("nan")
        info_artist.set_text(
            f"step: {state['steps']:>5}   from: {state['current']:>3}   "
            f"move: {state['kind']:<5}   α: {ALPHA:.2f}   target: {target}\n"
            f"L1 distance to exact PPR: {l1:.3f}"
        )
        return node_artist, walker_artist, info_artist, target_artist, trail

    animation = FuncAnimation(
        fig,
        update,
        interval=WALKER_INTERVAL_MS / SUBSTEPS,
        blit=True,
        cache_frame_data=False,
        save_count=SAVE_STEPS * SUBSTEPS,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
    return animation, fig


def main() -> None:
    if SAVE_GIF:
        Path("animations").mkdir(exist_ok=True)
        animation, figure = build_animation()
        animation.save(
            "animations/personalised_pagerank.gif",
            writer="pillow",
            fps=round(1000 * SUBSTEPS / WALKER_INTERVAL_MS),
            dpi=SAVE_DPI,
        )
        plt.close(figure)

    # Fresh copy for the live window so it starts from step 0.
    animation, _figure = build_animation()
    plt.show()


if __name__ == "__main__":
    main()