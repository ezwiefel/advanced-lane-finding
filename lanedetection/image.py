from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

COLOR_PROFILES = {
    "gray-rgb": cv2.COLOR_RGB2GRAY,
    "gray": cv2.COLOR_BGR2GRAY,
    "hls-rgb": cv2.COLOR_RGB2HLS,
    "hls": cv2.COLOR_BGR2HLS
}


class LaneImage(object):
    def __init__(
            self,
            image_path: Union[str, Path],
            camera_settings: Optional['CameraSettings'] = None
    ):
        super().__init__()
        self.image_path = image_path
        self.img_array = self._read_image_to_array(image_path)
        self.camera_settings = camera_settings

    @property
    def size(self) -> Tuple[int, int]:
        return self.img_array.shape[:2]

    @staticmethod
    def _read_image_to_array(path) -> np.ndarray:
        return cv2.imread(str(path))

    def apply_colorspace(self, color_space: str = 'gray') -> np.ndarray:
        """Convert the image colorspace

        Args:
            color_space (str, optional): The color space to apply - see COLOR_PROFILES for available options
        Returns:
            np.ndarray: Grayscale array
        """
        img_array = self.img_array.copy()

        if color_space not in COLOR_PROFILES:
            raise ValueError("'{color_space}' not an available color space choice".format(
                color_space=color_space))

        return cv2.cvtColor(img_array, COLOR_PROFILES[color_space])
