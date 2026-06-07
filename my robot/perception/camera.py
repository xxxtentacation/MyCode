import numpy as np


class SimCamera:
    """Simulation camera wrapper providing RGB, depth, and segmentation data.

    In MuJoCo, camera data is accessed via mjData.cam_xpos, mjData.cam_xmat,
    and rendered frames come from the renderer attached to the MuJoCo context.
    """

    def __init__(self, width: int = 640, height: int = 480, fov: float = 45.0):
        self.width = width
        self.height = height
        self.fov = fov
        self._focal_length = (height / 2) / np.tan(np.deg2rad(fov / 2))
        self._cx = width / 2.0
        self._cy = height / 2.0
        self._K = np.array([
            [self._focal_length, 0, self._cx],
            [0, self._focal_length, self._cy],
            [0, 0, 1],
        ])

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._K

    def project_3d_to_pixel(self, pt_3d: np.ndarray) -> np.ndarray:
        """Project a 3D point (camera frame) to pixel coordinates [u, v]."""
        if pt_3d[2] <= 0:
            return np.array([-1, -1])
        u = self._focal_length * pt_3d[0] / pt_3d[2] + self._cx
        v = self._focal_length * pt_3d[1] / pt_3d[2] + self._cy
        return np.array([u, v])

    def pixel_to_3d(self, u: float, v: float, depth: float) -> np.ndarray:
        """Back-project pixel + depth to 3D point (camera frame)."""
        x = (u - self._cx) * depth / self._focal_length
        y = (v - self._cy) * depth / self._focal_length
        return np.array([x, y, depth])
