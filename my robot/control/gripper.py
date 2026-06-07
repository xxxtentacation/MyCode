import numpy as np


class RobotiqGripper:
    """Robotiq 2F-85 parallel gripper controller.

    The gripper has two fingers that move symmetrically.
    Args control the gripper position: positive = close, negative = open.
    """

    def __init__(self, open_width: float = 0.085, close_width: float = 0.0):
        self._open_width = open_width
        self._close_width = close_width
        self._current_width = open_width
        self._target_width = open_width
        self._max_force = 100.0  # N

    def open(self) -> float:
        """Fully open gripper. Returns target joint position."""
        self._target_width = self._open_width
        return self._target_width

    def close(self) -> float:
        """Fully close gripper. Returns target joint position."""
        self._target_width = self._close_width
        return self._target_width

    def set_width(self, width: float) -> float:
        """Set gripper to specific width in meters."""
        self._target_width = np.clip(width, self._close_width, self._open_width)
        return self._target_width

    @property
    def target_width(self) -> float:
        return self._target_width

    @property
    def is_open(self) -> bool:
        return self._target_width >= self._open_width * 0.9

    @property
    def is_closed(self) -> bool:
        return self._target_width <= self._close_width + 0.002

    @property
    def open_width(self) -> float:
        return self._open_width
