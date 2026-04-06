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
    dpi: int = 200,
    leader: int | None = None,
    trail_len: int = 60,          # was 40 — longer trails show structure better
    show_heads: bool = True,
    show_arrows: bool = False,
    arrow_scale: float = 70,
    point_size: float = 18,       # was 10 — more visible at print size
    trail_alpha: float = 0.9,     # was 0.7 — more opaque trails
    head_alpha: float = 1.0,
    linewidth: float = 0.9,       # was 0.45 — much more visible
    publication_style: bool = True,
) -> animation.FuncAnimation:
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    states_np = np.asarray(states)
    T, n = len(states_np), flock.num_agents

    s = states_np.reshape(T, n, 3)
    all_x = s[:, :, 0]
    all_y = s[:, :, 1]
    all_a = s[:, :, 2]
    all_cx = np.cos(all_a)
    all_cy = np.sin(all_a)

    # fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    fig_w_px = 2160   # or 1920, 2560, 3840, etc.
    fig_h_px = 2160
    render_dpi = dpi

    fig, ax = plt.subplots(
        figsize=(fig_w_px / render_dpi, fig_h_px / render_dpi),
        dpi=render_dpi,
        facecolor="white",
    )
    _setup_ax(ax, flock.grid_size)

    if publication_style:
        ax.set_facecolor("white")
        for artist in list(ax.patches):
            artist.remove()

        # Increase tick/label font sizes for print
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

    fig.tight_layout(pad=0)

    cmap = plt.get_cmap("hsv")
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    def angle_to_rgb(angles: np.ndarray) -> np.ndarray:
        return cmap(norm(angles))

    box_width = 2 * flock.grid_size

    def _segment_periodic_path(x: np.ndarray, y: np.ndarray):
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        jumps = (dx > 0.5 * box_width) | (dy > 0.5 * box_width)
        x_plot = x.astype(float).copy()
        y_plot = y.astype(float).copy()
        x_plot[1:][jumps] = np.nan
        y_plot[1:][jumps] = np.nan
        return x_plot, y_plot

    def _make_fading_segments(
        x: np.ndarray,
        y: np.ndarray,
        angles: np.ndarray,
        agent_idx: int,
    ):
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        valid = ~np.isnan(segments).any(axis=(1, 2))
        segments = segments[valid]
        seg_angles = angles[1:][valid]

        if len(segments) == 0:
            return segments, np.empty((0, 4))

        colors = angle_to_rgb(seg_angles)
        # Start at 0.15 instead of 0.02 — old trail segments now visible
        alphas = np.linspace(0.15, trail_alpha, len(segments))

        if leader is not None and agent_idx == leader:
            colors[:] = [0.85, 0.15, 0.15, 1.0]

        colors[:, 3] = alphas
        return segments, colors

    trail_collections = []
    for i in range(n):
        lc = LineCollection([], linewidths=linewidth, zorder=1, capstyle="round")
        ax.add_collection(lc)
        trail_collections.append(lc)

    initial_colors = angle_to_rgb(all_a[0])
    if leader is not None:
        initial_colors[leader] = np.array([0.85, 0.15, 0.15, 1.0])

    if show_heads:
        heads = ax.scatter(
            all_x[0], all_y[0],
            s=point_size,
            c=initial_colors,
            alpha=head_alpha,
            edgecolors="none",
            zorder=3,
        )
    else:
        heads = None

    if show_arrows:
        q = ax.quiver(
            all_x[0], all_y[0], all_cx[0], all_cy[0],
            color=initial_colors,
            scale=arrow_scale,
            width=0.0025,
            alpha=0.35,
            zorder=2,
        )
    else:
        q = None

    def update(frame):
        start = max(0, frame - trail_len)
        artists = []

        frame_colors = angle_to_rgb(all_a[frame])
        if leader is not None:
            frame_colors[leader] = np.array([0.85, 0.15, 0.15, 1.0])

        for i, lc in enumerate(trail_collections):
            x_traj = all_x[start:frame + 1, i]
            y_traj = all_y[start:frame + 1, i]
            a_traj = all_a[start:frame + 1, i]

            x_plot, y_plot = _segment_periodic_path(x_traj, y_traj)
            segments, colors = _make_fading_segments(x_plot, y_plot, a_traj, i)

            lc.set_segments(segments)
            lc.set_color(colors)
            artists.append(lc)

        if heads is not None:
            heads.set_offsets(np.column_stack([all_x[frame], all_y[frame]]))
            heads.set_facecolor(frame_colors)
            artists.append(heads)

        if q is not None:
            q.set_offsets(np.column_stack([all_x[frame], all_y[frame]]))
            q.set_UVC(all_cx[frame], all_cy[frame])
            q.set_color(frame_colors)
            artists.append(q)

        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 // fps,
        blit=False,
    )

    if path is not None:
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=-1,
            extra_args=[
                "-vcodec", "libx264",
                "-crf", "16",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-threads", "0",
            ],
        )
        
        ani.save(path, writer=writer, dpi=dpi)
        plt.close(fig)

    return ani

def render_video_hires(
    states: Array,
    flock: Flock,
    fps: int = 50,
    path: str | None = None,
    width: int = 3840,           # output pixel width  (e.g. 3840 = 4K, 7680 = 8K)
    height: int = 3840,          # output pixel height
    leader: int | None = None,
    trail_len: int = 60,
    show_heads: bool = True,
    show_arrows: bool = False,
    arrow_scale: float = 70,
    point_size: float = 18,
    trail_alpha: float = 0.9,
    head_alpha: float = 1.0,
    linewidth: float = 0.9,
    publication_style: bool = True,
) -> None:
    """
    Same as render_video but renders each frame via fig.savefig() (which
    correctly honours pixel dimensions) then stitches with ffmpeg directly.

    Resolution is controlled by `width` and `height` in pixels — not DPI.
    DPI is an internal rendering hint only; changing it does NOT affect output
    pixel count (that is the bug this function fixes).

    Requires `path` to be set (returns None, not an animation object).
    """
    import subprocess
    import tempfile
    import os
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if path is None:
        raise ValueError("render_video_hires requires a path to save the video.")

    states_np = np.asarray(states)
    T, n = len(states_np), flock.num_agents

    s = states_np.reshape(T, n, 3)
    all_x = s[:, :, 0]
    all_y = s[:, :, 1]
    all_a = s[:, :, 2]
    all_cx = np.cos(all_a)
    all_cy = np.sin(all_a)

    # Use a fixed internal DPI so that figsize * _dpi == desired pixel dims.
    # Changing _dpi here only trades memory/speed, never output resolution.
    _dpi = 100
    fig, ax = plt.subplots(
        figsize=(width / _dpi, height / _dpi),
        dpi=_dpi,
        facecolor="white",
    )
    _setup_ax(ax, flock.grid_size)

    if publication_style:
        ax.set_facecolor("white")
        for artist in list(ax.patches):
            artist.remove()
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

    fig.tight_layout(pad=0)

    cmap = plt.get_cmap("hsv")
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    def angle_to_rgb(angles: np.ndarray) -> np.ndarray:
        return cmap(norm(angles))

    box_width = 2 * flock.grid_size

    def _segment_periodic_path(x: np.ndarray, y: np.ndarray):
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        jumps = (dx > 0.5 * box_width) | (dy > 0.5 * box_width)
        x_plot = x.astype(float).copy()
        y_plot = y.astype(float).copy()
        x_plot[1:][jumps] = np.nan
        y_plot[1:][jumps] = np.nan
        return x_plot, y_plot

    def _make_fading_segments(x, y, angles, agent_idx):
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        valid = ~np.isnan(segments).any(axis=(1, 2))
        segments = segments[valid]
        seg_angles = angles[1:][valid]
        if len(segments) == 0:
            return segments, np.empty((0, 4))
        colors = angle_to_rgb(seg_angles)
        alphas = np.linspace(0.15, trail_alpha, len(segments))
        if leader is not None and agent_idx == leader:
            colors[:] = [0.85, 0.15, 0.15, 1.0]
        colors[:, 3] = alphas
        return segments, colors

    trail_collections = []
    for i in range(n):
        lc = LineCollection([], linewidths=linewidth, zorder=1, capstyle="round")
        ax.add_collection(lc)
        trail_collections.append(lc)

    initial_colors = angle_to_rgb(all_a[0])
    if leader is not None:
        initial_colors[leader] = np.array([0.85, 0.15, 0.15, 1.0])

    if show_heads:
        heads = ax.scatter(
            all_x[0], all_y[0],
            s=point_size,
            c=initial_colors,
            alpha=head_alpha,
            edgecolors="none",
            zorder=3,
        )
    else:
        heads = None

    if show_arrows:
        q = ax.quiver(
            all_x[0], all_y[0], all_cx[0], all_cy[0],
            color=initial_colors,
            scale=arrow_scale,
            width=0.0025,
            alpha=0.35,
            zorder=2,
        )
    else:
        q = None

    tmpdir = tempfile.mkdtemp(prefix="render_hires_")
    frame_digits = len(str(T - 1))
    frame_pattern = os.path.join(tmpdir, f"%0{frame_digits}d.png")

    try:
        for frame in range(T):
            start = max(0, frame - trail_len)

            frame_colors = angle_to_rgb(all_a[frame])
            if leader is not None:
                frame_colors[leader] = np.array([0.85, 0.15, 0.15, 1.0])

            for i, lc in enumerate(trail_collections):
                x_traj = all_x[start:frame + 1, i]
                y_traj = all_y[start:frame + 1, i]
                a_traj = all_a[start:frame + 1, i]
                x_plot, y_plot = _segment_periodic_path(x_traj, y_traj)
                segments, colors = _make_fading_segments(x_plot, y_plot, a_traj, i)
                lc.set_segments(segments)
                lc.set_color(colors)

            if heads is not None:
                heads.set_offsets(np.column_stack([all_x[frame], all_y[frame]]))
                heads.set_facecolor(frame_colors)

            if q is not None:
                q.set_offsets(np.column_stack([all_x[frame], all_y[frame]]))
                q.set_UVC(all_cx[frame], all_cy[frame])
                q.set_color(frame_colors)

            frame_path = os.path.join(tmpdir, f"{frame:0{frame_digits}d}.png")
            fig.savefig(frame_path, dpi=_dpi, facecolor=fig.get_facecolor())

        plt.close(fig)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-vcodec", "libx264",
                "-crf", "16",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-threads", "0",
                path,
            ],
            check=True,
        )
    finally:
        for fname in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, fname))
        os.rmdir(tmpdir)


def render_video_sharp(
    states: Array,
    flock: Flock,
    fps: int = 50,
    path: str = "vid_sharp.mp4",
    width: int = 3840,
    height: int = 3840,
    leader: int | None = None,
    trail_len: int = 60,
    show_heads: bool = True,
    show_arrows: bool = False,
    arrow_scale: float = 70,
    trail_alpha: float = 0.9,
    head_alpha: float = 1.0,
) -> None:
    """
    High-resolution, sharp-particle video render.

    Fixes vs render_video / render_video_hires:
    - Forces Agg backend (no macOS Retina DPI interference).
    - Uses dpi=300 for crisp rasterization of small particles/trails.
    - Fills the canvas fully via subplots_adjust — no tight_layout trimming.
    - Uses yuv444p in ffmpeg (no chroma subsampling → sharp colored dots).
    - point_size and linewidth scale with dpi so they look the same physical
      size regardless of output resolution.
    """
    import subprocess
    import tempfile
    import os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    states_np = np.asarray(states)
    T, n = len(states_np), flock.num_agents
    s = states_np.reshape(T, n, 3)
    all_x, all_y, all_a = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    all_cx, all_cy = np.cos(all_a), np.sin(all_a)

    _dpi = 300
    # scatter s= is in points², linewidth in points — scale both with dpi so
    # markers appear the same physical size at any output resolution
    _scale = _dpi / 100.0
    point_size = 6.0 * _scale ** 2
    linewidth = 0.3 * _scale

    fig, ax = plt.subplots(
        figsize=(width / _dpi, height / _dpi),
        dpi=_dpi,
        facecolor="white",
    )
    # Fill canvas exactly — avoids tight_layout / set_aspect trimming the frame
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _setup_ax(ax, flock.grid_size)
    ax.set_facecolor("white")
    for artist in list(ax.patches):  # remove border rect added by _setup_ax
        artist.remove()

    cmap = plt.get_cmap("hsv")
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    def angle_to_rgb(angles: np.ndarray) -> np.ndarray:
        return cmap(norm(angles))

    box_width = 2 * flock.grid_size

    def _break_periodic(x: np.ndarray, y: np.ndarray):
        jumps = (np.abs(np.diff(x)) > 0.5 * box_width) | (np.abs(np.diff(y)) > 0.5 * box_width)
        xp, yp = x.astype(float).copy(), y.astype(float).copy()
        xp[1:][jumps] = np.nan
        yp[1:][jumps] = np.nan
        return xp, yp

    def _fading_segments(x: np.ndarray, y: np.ndarray, angles: np.ndarray, agent_idx: int):
        pts = np.column_stack([x, y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        valid = ~np.isnan(segs).any(axis=(1, 2))
        segs = segs[valid]
        seg_a = angles[1:][valid]
        if len(segs) == 0:
            return segs, np.empty((0, 4))
        colors = angle_to_rgb(seg_a)
        colors[:, 3] = np.linspace(0.15, trail_alpha, len(segs))
        if leader is not None and agent_idx == leader:
            colors[:] = [0.85, 0.15, 0.15, 1.0]
        return segs, colors

    trail_cols = []
    for _ in range(n):
        lc = LineCollection([], linewidths=linewidth, zorder=1, capstyle="round")
        ax.add_collection(lc)
        trail_cols.append(lc)

    init_c = angle_to_rgb(all_a[0])
    if leader is not None:
        init_c[leader] = [0.85, 0.15, 0.15, 1.0]

    if show_heads:
        heads = ax.scatter(
            all_x[0], all_y[0],
            s=point_size, c=init_c,
            alpha=head_alpha, edgecolors="none", zorder=3,
        )
    else:
        heads = None

    tmpdir = tempfile.mkdtemp(prefix="render_sharp_")
    nd = len(str(T - 1))

    try:
        for frame in range(T):
            start = max(0, frame - trail_len)
            fc = angle_to_rgb(all_a[frame])
            if leader is not None:
                fc[leader] = [0.85, 0.15, 0.15, 1.0]

            for i, lc in enumerate(trail_cols):
                xp, yp = _break_periodic(all_x[start:frame + 1, i], all_y[start:frame + 1, i])
                segs, cols = _fading_segments(xp, yp, all_a[start:frame + 1, i], i)
                lc.set_segments(segs)
                lc.set_color(cols)

            if heads is not None:
                heads.set_offsets(np.c_[all_x[frame], all_y[frame]])
                heads.set_facecolor(fc)

            fig.savefig(os.path.join(tmpdir, f"{frame:0{nd}d}.png"), dpi=_dpi)

        plt.close(fig)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, f"%0{nd}d.png"),
                "-vcodec", "libx264",
                "-crf", "10",
                "-preset", "slow",
                "-pix_fmt", "yuv444p",
                "-threads", "0",
                str(path),
            ],
            check=True,
        )
    finally:
        for fname in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, fname))
        os.rmdir(tmpdir)


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