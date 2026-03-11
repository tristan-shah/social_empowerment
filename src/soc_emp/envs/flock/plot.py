import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from jax import Array

from soc_emp.envs.flock import Flock, make_reset, make_step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_np(state: Array, num_agents: int):
    """Decode state to numpy arrays (x, y, cx, cy) where cx/cy are unit heading vecs."""
    s = np.asarray(state).reshape(num_agents, 3)
    x, y, a = s[:, 0], s[:, 1], s[:, 2]
    return x, y, np.cos(a), np.sin(a)


def _setup_ax(ax: plt.Axes, grid_size: float) -> plt.Axes:
    """Shared axis configuration for all flock plots."""
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")
    border = patches.Rectangle(
        (-grid_size, -grid_size), 2 * grid_size, 2 * grid_size,
        linewidth=1, edgecolor="#cccccc", facecolor="none", linestyle="--",
    )
    ax.add_patch(border)
    return ax


def _draw_radii(ax: plt.Axes, x, y, radius: float) -> list:
    """Draw a faint interaction-radius circle around each agent."""
    circles = []
    for xi, yi in zip(x, y):
        c = patches.Circle(
            (xi, yi), radius,
            linewidth=0.4, edgecolor="#aaaaaa", facecolor="#0000ff08",
            linestyle="-", zorder=0,
        )
        ax.add_patch(c)
        circles.append(c)
    return circles


# def _make_colors(num_agents: int, leader: int | None) -> np.ndarray:
#     """Build an (N, 3) color array; leader is red, all others random."""
#     rng = np.random.default_rng(0)
#     colors = rng.uniform(0.0, 0.6, size=(num_agents, 3))
#     if leader is not None:
#         colors[leader] = [0.9, 0.1, 0.1]
#     return colors

# def _make_colors(num_agents: int, leader: int | None) -> np.ndarray:
#     """Build an (N, 3) color array; leader is red, all others blue."""
#     colors = np.full((num_agents, 3), [0.2, 0.4, 0.8])  # blue for all
#     if leader is not None:
#         colors[leader] = [0.9, 0.1, 0.1]  # red for leader
#     return colors

def _make_colors(num_agents: int, leader: int | None) -> np.ndarray:
    """Build an (N, 3) color array; leader is red, all others blue. If no leader, random colors."""
    if leader is None:
        rng = np.random.default_rng(0)
        return rng.uniform(0.0, 0.6, size=(num_agents, 3))
    colors = np.full((num_agents, 3), [0.2, 0.4, 0.8])
    colors[leader] = [0.9, 0.1, 0.1]
    return colors


# ---------------------------------------------------------------------------
# Single-frame render
# ---------------------------------------------------------------------------

def render_image(state: Array, flock: Flock, show_radius: bool = False, leader: int | None = None, close: bool = False) -> plt.Figure:
    """
    Render a single frame of the flock.

    Args:
        state:        Flat state vector, shape (num_agents * 3,).
        flock:        Flock config (used for num_agents, grid_size, speed).
        show_radius:  If True, draw the interaction radius around each agent.
        leader:       Optional index of the leader agent (rendered in red).

    Returns:
        fig: Matplotlib figure. Caller decides whether to show or save.
    """
    x, y, cx, cy = _decode_np(state, flock.num_agents)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    _setup_ax(ax, flock.grid_size)

    colors = _make_colors(flock.num_agents, leader)

    if show_radius:
        _draw_radii(ax, x, y, flock.neighbor_radius)

    ax.quiver(
        x, y, cx, cy,
        color = colors,
        scale = 50,
        width = 0.003,
        alpha = 0.85,
    )

    fig.tight_layout()#(pad=0)
    
    if close:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Video render
# ---------------------------------------------------------------------------

def render_video(
    states: Array,
    flock: Flock,
    fps: int = 50,
    path: str | None = None,
    show_radius: bool = False,
    dpi: int = 200,
    leader: int | None = None,
) -> animation.FuncAnimation:
    states_np = np.asarray(states)
    T, n = len(states_np), flock.num_agents

    # Vectorised decode — single reshape + two trig calls over the full (T, n) array
    s = states_np.reshape(T, n, 3)
    all_x  = s[:, :, 0]          # (T, n)
    all_y  = s[:, :, 1]
    all_cx = np.cos(s[:, :, 2])
    all_cy = np.sin(s[:, :, 2])

    # Pre-allocate offset buffer reused every frame
    offsets = np.empty((n, 2), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    _setup_ax(ax, flock.grid_size)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    colors = _make_colors(n, leader)
    circles = _draw_radii(ax, all_x[0], all_y[0], flock.neighbor_radius) if show_radius else []
    q = ax.quiver(all_x[0], all_y[0], all_cx[0], all_cy[0],
                  color=colors, scale=50, width=0.003, alpha=0.85)

    def update(frame):
        offsets[:, 0] = all_x[frame]
        offsets[:, 1] = all_y[frame]
        q.set_offsets(offsets)
        q.set_UVC(all_cx[frame], all_cy[frame])
        if circles:
            for c, xi, yi in zip(circles, all_x[frame], all_y[frame]):
                c.center = (xi, yi)
        return (q, *circles)

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=True,
    )

    if path is not None:
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=-1,
            extra_args=[
                "-vcodec", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-threads", "0",
            ],
        )
        with plt.rc_context({"path.simplify": True, "path.simplify_threshold": 1.0}):
            ani.save(path, writer=writer, dpi=dpi)
        plt.close(fig)

    return ani

# ---------------------------------------------------------------------------
# Order parameter plot
# ---------------------------------------------------------------------------

def plot_order_parameter(phi: Array) -> plt.Figure:
    """
    Plot the Vicsek order parameter phi over time.

    Args:
        phi: Array of order parameter values, shape (T,).

    Returns:
        fig: Matplotlib figure.
    """
    phi_np = np.asarray(phi)

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(phi_np, color="#3366cc", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Step", color="#444444")
    ax.set_ylabel("phi", color="#444444")
    ax.set_title("Order parameter", color="#222222")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    fig.tight_layout()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    import jax
    import jax.numpy as jnp

    key = jax.random.key(0)

    flock = Flock(
        num_agents = 500,
        grid_size = 5.0,
        speed = 1.0,
        neighbor_radius = 1.0,
        neighbor_falloff = 10.0,
        dt = 0.05,
    )

    reset = make_reset(flock)
    step = make_step(flock)

    T = 1000
    action = jnp.zeros(flock.num_agents)

    state0 = reset(key)

    def scan_fn(state, _):
        next_state = step(state, action)
        return next_state, next_state

    _, states = jax.lax.scan(scan_fn, state0, None, length=T)
    states = jnp.concatenate([state0[None], states], axis=0)

    fig = render_image(states[-1], flock, show_radius=True, leader=0)
    fig.savefig("flock_frame.png", dpi=300, bbox_inches="tight")
    print("Saved flock_frame.png")

    render_video(states, flock, fps=60, save_path="flock.mp4", show_radius=False, dpi=150, leader=0)
    print("Saved flock.mp4")

    def order_parameter(state):
        s = jnp.reshape(state, (flock.num_agents, 3))
        a = s[:, 2]
        return jnp.sqrt(jnp.mean(jnp.cos(a)) ** 2 + jnp.mean(jnp.sin(a)) ** 2)

    phi = jnp.stack([order_parameter(s) for s in states])
    fig2 = plot_order_parameter(phi)
    fig2.savefig("flock_order.png", dpi=300, bbox_inches="tight")
    print("Saved flock_order.png")

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import patches
# from jax import Array

# from soc_emp.envs.flock import Flock, make_reset, make_step

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _decode_np(state: Array, num_agents: int):
#     """Decode state to numpy arrays (x, y, cx, cy) where cx/cy are unit heading vecs."""
#     s = np.asarray(state).reshape(num_agents, 3)
#     x, y, a = s[:, 0], s[:, 1], s[:, 2]
#     return x, y, np.cos(a), np.sin(a)


# def _setup_ax(ax: plt.Axes, grid_size: float) -> plt.Axes:
#     """Shared axis configuration for all flock plots."""
#     ax.set_xlim(-grid_size, grid_size)
#     ax.set_ylim(-grid_size, grid_size)
#     ax.set_aspect("equal")
#     ax.set_axis_off()
#     ax.set_facecolor("white")
#     border = patches.Rectangle(
#         (-grid_size, -grid_size), 2 * grid_size, 2 * grid_size,
#         linewidth=1, edgecolor="#cccccc", facecolor="none", linestyle="--",
#     )
#     ax.add_patch(border)
#     return ax


# def _draw_radii(ax: plt.Axes, x, y, radius: float) -> list:
#     """Draw a faint interaction-radius circle around each agent."""
#     circles = []
#     for xi, yi in zip(x, y):
#         c = patches.Circle(
#             (xi, yi), radius,
#             linewidth=0.4, edgecolor="#aaaaaa", facecolor="#0000ff08",
#             linestyle="-", zorder=0,
#         )
#         ax.add_patch(c)
#         circles.append(c)
#     return circles


# # ---------------------------------------------------------------------------
# # Single-frame render
# # ---------------------------------------------------------------------------

# def render(state: Array, flock: Flock, show_radius: bool = False) -> plt.Figure:
#     """
#     Render a single frame of the flock.

#     Args:
#         state:        Flat state vector, shape (num_agents * 3,).
#         flock:        Flock config (used for num_agents, grid_size, speed).
#         show_radius:  If True, draw the interaction radius around each agent.

#     Returns:
#         fig: Matplotlib figure. Caller decides whether to show or save.
#     """
#     x, y, cx, cy = _decode_np(state, flock.num_agents)

#     fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
#     _setup_ax(ax, flock.grid_size)

#     rng = np.random.default_rng(0)
#     colors = rng.uniform(0.0, 0.6, size=(flock.num_agents, 3))

#     if show_radius:
#         _draw_radii(ax, x, y, flock.neighbor_radius)

#     ax.quiver(
#         x, y, cx, cy,
#         color=colors,
#         scale=50,
#         width=0.003,
#         alpha=0.85,
#     )

#     fig.tight_layout(pad=0)
#     plt.close(fig)
#     return fig


# # ---------------------------------------------------------------------------
# # Video render
# # ---------------------------------------------------------------------------

# def render_video(
#     states: Array,
#     flock: Flock,
#     fps: int = 50,
#     save_path: str | None = None,
#     show_radius: bool = False,
#     dpi: int = 150,
# ) -> animation.FuncAnimation:
#     """
#     Render a trajectory as an animation.

#     Args:
#         states:       State trajectory, shape (T, num_agents * 3).
#         flock:        Flock config.
#         fps:          Playback speed.
#         save_path:    Optional path to save as .mp4 (requires ffmpeg).
#         show_radius:  If True, draw the interaction radius around each agent.
#         dpi:          Resolution of the saved video. Has no effect on screen display.

#     Returns:
#         ani: FuncAnimation object.
#     """
#     states_np = np.asarray(states)

#     fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
#     _setup_ax(ax, flock.grid_size)

#     rng = np.random.default_rng(0)
#     colors = rng.uniform(0.0, 0.6, size=(flock.num_agents, 3))

#     x0, y0, cx0, cy0 = _decode_np(states_np[0], flock.num_agents)

#     circles = _draw_radii(ax, x0, y0, flock.neighbor_radius) if show_radius else []
#     q = ax.quiver(x0, y0, cx0, cy0, color=colors, scale=50, width=0.003, alpha=0.85)

#     fig.tight_layout(pad=0)

#     def update(frame):
#         x, y, cx, cy = _decode_np(states_np[frame], flock.num_agents)
#         q.set_offsets(np.stack([x, y], axis=1))
#         q.set_UVC(cx, cy)
#         for c, xi, yi in zip(circles, x, y):
#             c.center = (xi, yi)
#         return (q, *circles)

#     ani = animation.FuncAnimation(
#         fig, update, frames=len(states_np), interval=1000 // fps, blit=True
#     )

#     if save_path is not None:
#         writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
#         ani.save(save_path, writer=writer, dpi=dpi)

#     return ani


# # ---------------------------------------------------------------------------
# # Order parameter plot
# # ---------------------------------------------------------------------------

# def plot_order_parameter(phi: Array) -> plt.Figure:
#     """
#     Plot the Vicsek order parameter phi over time.

#     Args:
#         phi: Array of order parameter values, shape (T,).

#     Returns:
#         fig: Matplotlib figure.
#     """
#     phi_np = np.asarray(phi)

#     fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")
#     ax.set_facecolor("white")
#     ax.plot(phi_np, color="#3366cc", linewidth=1.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel("Step", color="#444444")
#     ax.set_ylabel("phi", color="#444444")
#     ax.set_title("Order parameter", color="#222222")
#     ax.tick_params(colors="#aaaaaa")
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#cccccc")
#     fig.tight_layout()
#     plt.close(fig)
#     return fig


# if __name__ == '__main__':
#     import jax
#     import jax.numpy as jnp

#     key = jax.random.key(0)

#     flock = Flock(
#         num_agents = 500,
#         grid_size = 5.0,
#         speed = 1.0,
#         neighbor_radius = 1.0,
#         neighbor_falloff = 10.0,
#         dt = 0.05,
#     )

#     reset = make_reset(flock)
#     step = make_step(flock)

#     # -- rollout via lax.scan --------------------------------------------------
#     T = 1000
#     action = jnp.zeros(flock.num_agents)

#     state0 = reset(key)

#     def scan_fn(state, _):
#         next_state = step(state, action)
#         return next_state, next_state

#     _, states = jax.lax.scan(scan_fn, state0, None, length=T)
#     states = jnp.concatenate([state0[None], states], axis=0)  # (T+1, num_agents * 3)

#     # -- single frame ----------------------------------------------------------
#     fig = render(states[-1], flock, show_radius=True)
#     fig.savefig("flock_frame.png", dpi=300, bbox_inches="tight")
#     print("Saved flock_frame.png")

#     # -- video -----------------------------------------------------------------
#     render_video(states, flock, fps=60, save_path="flock.mp4", show_radius=False, dpi=150)
#     print("Saved flock.mp4")

#     # -- order parameter -------------------------------------------------------
#     def order_parameter(state):
#         s = jnp.reshape(state, (flock.num_agents, 3))
#         a = s[:, 2]
#         return jnp.sqrt(jnp.mean(jnp.cos(a)) ** 2 + jnp.mean(jnp.sin(a)) ** 2)

#     phi = jnp.stack([order_parameter(s) for s in states])
#     fig2 = plot_order_parameter(phi)
#     fig2.savefig("flock_order.png", dpi=300, bbox_inches="tight")
#     print("Saved flock_order.png")