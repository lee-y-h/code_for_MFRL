import params

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class GridWorld:

    """
    grid world:
    +----------> x 
    |
    |
    |
    v y
    """

    ACTIONS = {
        1: (0, -1),  # up
        2: (0, 1),   # down
        3: (-1, 0),  # left
        4: (1, 0),   # right
        5: (0, 0),   # stay
    }

    def __init__(self, width, height, target, forbidden=None):
        self.width = width
        self.height = height
        self.target = tuple(target)
        self.forbidden = [tuple(f) for f in (forbidden or [])]

    def in_bounds(self, state):
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def is_forbidden(self, state):
        return state in self.forbidden

    def is_target(self, state):
        return state == self.target

    def get_next_state_and_reward(self, state, action):
        """
        Compute the next state and reward for taking `action` in `state`.

        This inlines the former `apply_action` and `get_reward` logic so callers
        get both pieces of information from one place.

        Returns: (next_state, reward)
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions: {list(self.ACTIONS.keys())}")

        x, y = tuple(state)
        dx, dy = self.ACTIONS[action]
        candidate = (x + dx, y + dy)

        if not self.in_bounds(candidate):
            next_state = (x, y)
            reward = params.REWARD_BOUNDARY
            return next_state, reward

        # Otherwise move into candidate
        next_state = candidate

        # Target / forbidden / step rewards
        if self.is_target(next_state):
            reward = params.REWARD_TARGET
        elif self.is_forbidden(next_state):
            reward = params.REWARD_FORBIDDEN
        else:
            reward = params.REWARD_STEP

        return next_state, reward
    
    def render(self, Values, Actions,
                folder_path: str = 'grid_world',
                title: str = 'GridWorld'):
        """
        Render the grid as an image and save it to a `renders/<folder_name>` folder

        - `Values` is a dict mapping (x,y) -> numeric value (may be None)
        - `Actions` is a dict mapping (x,y) -> action integer (1-5)

        The function writes a PNG image file for the current grid and returns the
        path to the saved image.
        """
        if plt is None or Rectangle is None:
            raise RuntimeError("matplotlib is required for render(); please install it (pip install matplotlib)")

        out_dir = Path(folder_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        w, h = self.width, self.height
        fig, ax = plt.subplots(figsize=(w * 1.2, h * 1.2))

        # Draw cells
        for x in range(w):
            for y in range(h):
                cell = (x, y)
                # default facecolor
                face = 'white'
                edge = 'black'
                if cell in self.forbidden:
                    face = 'orange'
                if cell == self.target:
                    face = '#a6cee3'  # light blue

                rect = Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edge)
                ax.add_patch(rect)

                # draw value if present
                if Values is not None:
                    v = Values.get(cell)
                    if v is not None:
                        ax.text(x + 0.5, y + 0.5, f"{v:.2f}", ha='center', va='center', fontsize=10)

                # draw action arrow if present
                if Actions is not None:
                    a = Actions.get(cell)
                    if a in self.ACTIONS and a != 5:
                        dx, dy = self.ACTIONS[a]
                        # scale arrow length
                        ax.arrow(x + 0.5 - 0.15 * dx, y + 0.5 - 0.15 * dy,
                                 0.3 * dx, 0.3 * dy,
                                 head_width=0.12, head_length=0.12, fc='k', ec='k')

        # Draw grid lines and styling
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xticks([i for i in range(w + 1)])
        ax.set_yticks([i for i in range(h + 1)])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.invert_yaxis()  # put (0,0) at top-left to match text/grid indexing

        # Title
        ax.set_title(title)

        # Save file
        fname = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        fig.savefig(str(fname), bbox_inches='tight')
        plt.close(fig)
        return str(fname)
