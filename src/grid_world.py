import random
from typing import Mapping, Sequence

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class GridWorld:

    """
    grid world:
    +----------> x 
    |
    |
    |
    v y
    """

    ACTIONS: dict[int, tuple[int, int]] = {
        1: (0, -1),  # up
        2: (0, 1),   # down
        3: (-1, 0),  # left
        4: (1, 0),   # right
        5: (0, 0),   # stay
    }

    def __init__(
        self,
        width: int,
        height: int,
        target: Sequence[int],
        forbidden: Sequence[Sequence[int]] | None = None,
        start: Sequence[int] | None = None,
        r_target: float | None = None,
        r_boundary: float | None = None,
        r_forbidden: float | None = None,
        r_step: float | None = None,
        r_stay: float | None = None,
    ):
        """
        GridWorld constructor.

        Reward and start parameters are expected to be passed directly.
        """
        self.width = width
        self.height = height
        self.target = (int(target[0]), int(target[1]))
        self.forbidden = [(int(f[0]), int(f[1])) for f in (forbidden or [])]

        self.states: list[tuple[int, int]] = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.actions = list(self.ACTIONS.keys())

        start_seq = (0, 0) if start is None else start

        self.start_state = (int(start_seq[0]), int(start_seq[1]))
        if not self.in_bounds(self.start_state):
            raise ValueError(f"Invalid start state {self.start_state}: out of grid bounds")

        self.current_state = self.start_state

        self.r_target = 1 if r_target is None else r_target
        self.r_boundary = -1 if r_boundary is None else r_boundary
        self.r_forbidden = -10 if r_forbidden is None else r_forbidden
        self.r_step = 0 if r_step is None else r_step
        self.r_stay = 0 if r_stay is None else r_stay

    def in_bounds(self, state: tuple[int, int]) -> bool:
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def is_forbidden(self, state: tuple[int, int]) -> bool:
        return state in self.forbidden

    def is_target(self, state: tuple[int, int]) -> bool:
        return state == self.target

    def sample_state_action_pair(self) -> tuple[tuple[int, int], int]:
        """
        Sample a random (state, action) pair from the grid world.

        Returns: (state, action)
        """
        
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        state = (x, y)

        action = random.choice(self.actions)
        return state, action

    def _sample_action_from_probs(
        self,
        state: tuple[int, int],
        policy_probs: Mapping[tuple[int, int], Mapping[int, float]],
    ) -> int:
        probs = policy_probs[state]
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        return random.choices(actions, weights=weights, k=1)[0]

    def reset(self, start_state: Sequence[int] | None = None) -> tuple[int, int]:
        state = self.start_state if start_state is None else (int(start_state[0]), int(start_state[1]))
        if not self.in_bounds(state):
            raise ValueError(f"Invalid reset state {state}: out of grid bounds")

        self.current_state = state
        return self.current_state

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        done = self.is_target(next_state)
        return next_state, reward, done

    def get_next_state_and_reward(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float]:
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
            reward = self.r_boundary
            return next_state, reward

        # Otherwise move into candidate
        next_state = candidate

        # Target / forbidden / step rewards
        if self.is_target(next_state):
            reward = self.r_target
        elif self.is_forbidden(next_state):
            reward = self.r_forbidden
        else:
            if action == 5:  # stay action
                reward = self.r_stay
            else:
                reward = self.r_step

        return next_state, reward
    
    def generate_stochastic_episode(self, start_state=None, policy_probs=None, max_length=100, action=None):
        """
        Generate an episode by following `policy_probs` starting from `start_state`.

        - `policy_probs` is a dict mapping state -> action probabilities (dict).
        - `max_length` is the maximum length of the episode (to prevent infinite loops).

        Returns: list of (state, action, reward, next_state, done) tuples.
        """
        if policy_probs is None:
            raise ValueError("policy_probs must be provided for stochastic episode generation")

        episode = []
        current_state = self.reset(start_state)
        if action is None:
            action = self._sample_action_from_probs(current_state, policy_probs)

        for _ in range(max_length):
            next_state, reward, done = self.step(action)
            episode.append((current_state, action, reward, next_state, done))
            if done:
                break
            current_state = next_state
            action = self._sample_action_from_probs(current_state, policy_probs)

        return episode

    def generate_deterministic_episode(self, start_state=None, deterministic_policy=None, max_length=100, action=None):
        """
        Generate an episode by following a deterministic policy.

        Returns: list of (state, action, reward, next_state, done) tuples.
        """
        if deterministic_policy is None:
            raise ValueError("deterministic_policy must be provided for deterministic episode generation")

        episode = []
        current_state = self.reset(start_state)
        if action is None:
            action = deterministic_policy[current_state]

        for _ in range(max_length):
            next_state, reward, done = self.step(action)
            episode.append((current_state, action, reward, next_state, done))
            if done:
                break
            current_state = next_state
            action = deterministic_policy[current_state]

        return episode


    def render(self, Values, Actions,
                folder_path: str = 'grid_world',
                title: str = 'GridWorld',
                file_name: str = ''):
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
                                 head_width=0.12, head_length=0.12, fc='green', ec='green')
                    elif a == 5:
                        # stay action: draw a small circle at the cell center
                        circ = Circle((x + 0.5, y + 0.5), 0.18, edgecolor='green', facecolor='none', linewidth=1.2)
                        ax.add_patch(circ)

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
        if file_name == '':
            fname = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        else:
            fname = out_dir / f"{file_name}.png"
        fig.savefig(str(fname), bbox_inches='tight')
        plt.close(fig)
        return str(fname)
