from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def plot_episode_stats(episode_lengths=None, total_rewards=None, out_dir='renders/plots', title=None, filename=None):
    """Plot episode statistics and save to PNG.

    Both `episode_lengths` and `total_rewards` are optional (may be None).
    Behavior:
      - If both are provided: draw two stacked subplots (total reward above, length below).
      - If only one is provided: draw a single plot for that series.
      - If neither provided: produce an empty figure with the title (if any).

    The first two parameters are intentionally positional so callers that
    pass only `episode_lengths` remain compatible.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if episode_lengths is not None or total_rewards is not None:
            name = 'episode_stats' if (episode_lengths is not None and total_rewards is not None) else (
                'total_rewards' if (episode_lengths is None and total_rewards is not None) else 'episode_lengths')
        else:
            name = 'episode_stats_empty'
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # determine episode count
    if episode_lengths is not None:
        episodes = list(range(1, len(episode_lengths) + 1))
    elif total_rewards is not None:
        episodes = list(range(1, len(total_rewards) + 1))
    else:
        episodes = []

    if episode_lengths is None and total_rewards is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        if title:
            ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    elif total_rewards is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        assert episode_lengths is not None
        ax.plot(episodes, episode_lengths, linestyle='-', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode length')
        if title:
            ax.set_title(title)
        ax.grid(True)
        # set reasonable x-ticks (up to 10, at least 1)
        if episodes:
            max_ticks = min(10, len(episodes))
            step = max(1, len(episodes) // max_ticks)
            xt = list(range(1, len(episodes) + 1, step))
            if xt[-1] != episodes[-1]:
                xt.append(episodes[-1])
            ax.set_xticks(xt)
    elif episode_lengths is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        assert total_rewards is not None
        ax.plot(episodes, total_rewards, linestyle='-', color='C1', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        if title:
            ax.set_title(title)
        ax.grid(True)
        if episodes:
            max_ticks = min(10, len(episodes))
            step = max(1, len(episodes) // max_ticks)
            xt = list(range(1, len(episodes) + 1, step))
            if xt[-1] != episodes[-1]:
                xt.append(episodes[-1])
            ax.set_xticks(xt)
    else:
        fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True,
                                             gridspec_kw={'height_ratios': [1, 1.2]})
        assert total_rewards is not None and episode_lengths is not None
        ax_top.plot(episodes, total_rewards, linestyle='-', color='C1', linewidth=1)
        ax_top.set_ylabel('Total reward')
        ax_top.grid(True)
        ax_top.tick_params(labelbottom=False)

        ax_bot.plot(episodes, episode_lengths, linestyle='-', color='C0', linewidth=1)
        ax_bot.set_xlabel('Episode')
        ax_bot.set_ylabel('Episode length')
        ax_bot.grid(True)

        # set x-ticks on bottom axis (up to 10, include last)
        if episodes:
            max_ticks = min(10, len(episodes))
            step = max(1, len(episodes) // max_ticks)
            xt = list(range(1, len(episodes) + 1, step))
            if xt[-1] != episodes[-1]:
                xt.append(episodes[-1])
            ax_bot.set_xticks(xt)
            ax_bot.tick_params(labelbottom=True)

        if title:
            fig.suptitle(title)

    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_loss(losses, out_dir='renders/plots', title=None, filename=None):
    """Plot loss vs iteration and save to PNG.

    Parameters:
      - losses: iterable of loss values. Iteration index will be 1..len(losses).
      - out_dir: output directory for PNG
      - title: optional plot title
      - filename: optional filename; auto-generated if None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    vals = list(losses)
    iters = list(range(1, len(vals) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, vals, marker='o', markersize=3, linewidth=1, color='C2')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    if title:
        ax.set_title(title)
    ax.grid(True)

    # set reasonable x-ticks (up to 10, include last)
    if iters:
        max_ticks = min(10, len(iters))
        step = max(1, len(iters) // max_ticks)
        xt = list(range(iters[0], iters[-1] + 1, step))
        if xt[-1] != iters[-1]:
            xt.append(iters[-1])
        ax.set_xticks(xt)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
