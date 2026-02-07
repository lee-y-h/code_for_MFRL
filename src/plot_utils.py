from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def plot_episode_lengths(episode_lengths, out_dir='renders/plots', title=None, filename=None):
    """Plot episode lengths vs episode index and save to PNG.

    Args:
        episode_lengths (list of int): lengths per episode.
        out_dir (str or Path): directory to save the plot into.
        title (str): optional plot title.
        filename (str): optional filename; if omitted a timestamped name is used.

    Returns:
        str: path to the saved image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"episode_lengths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    episodes = list(range(1, len(episode_lengths) + 1))
    ax.plot(episodes, episode_lengths, marker='o', linestyle='-')
    ax.set_xlabel('Episode index')
    ax.set_ylabel('Episode length')
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
