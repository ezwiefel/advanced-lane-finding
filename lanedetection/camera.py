import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from lanedetection.image import LaneImage


class CameraSettings(object):

    def __init__(self,
                 cam_matrix: Optional[np.ndarray] = None,
                 distortion_coefs: Optional[np.ndarray] = None,
                 image_dims: Optional[np.ndarray] = None
                 ):
        super().__init__()

        self.cam_matrix = cam_matrix
        self.distortion_coefs = distortion_coefs
        self.image_dims = image_dims

    def calibrate(
            self,
            image_paths: Iterable[Path],
            chessboard_dims: Tuple[int, int],
            camera_image_dimensions: Optional[Tuple[int, int]] = None
    ) -> None:
        # Generate object_points and image_points
        obj_points, img_points, img_dims = self._generate_obj_and_image_points(
            image_paths=image_paths,
            chessboard_dims=chessboard_dims
        )

        if not camera_image_dimensions:
            camera_image_dimensions = img_dims

        if not self.image_dims:
            self.image_dims = camera_image_dimensions

        camera_matrix, distortion_coefs = self._calculate_camera_matrix(
            obj_points=obj_points,
            img_points=img_points
        )

        self.cam_matrix = camera_matrix
        self.distortion_coefs = distortion_coefs

    @staticmethod
    def _gen_obj_points_array(chessboard_dims: Tuple[int, int]) -> np.ndarray:
        """
        Generate the object points for the chessboard dimensions given

        Args:
            chessboard_dims (Tuple[int, int]): The dimensions in the chessboard images used for calibration that is
            expected by OpenCV - namely the number of inside corners for (x, y)

        Returns:
            np.ndarray with (x * y) entries
        """

        num_x, num_y = chessboard_dims

        obj_points = np.zeros((num_x * num_y, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

        return obj_points

    def _generate_obj_and_image_points(
            self,
            image_paths: Iterable[Path],
            chessboard_dims: Tuple[int, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
        """
        Generate the object points and image points lists from a directory of calibration images

        Args:
            image_paths: List of Path variable
            chessboard_dims (Tuple[int, int]): The dimensions in the chessboard images used for calibration that is
            expected by OpenCV - namely the number of inside corners for (x, y)

        Returns:
            object_points (List[np.ndarray]): The list of object points
            image_points (List[np.ndarray]): The list of detected chessboard coordinates
            image_dimensions (Tuple[int, int]): The dimensions of the last image
        """
        op_array = self._gen_obj_points_array(chessboard_dims=chessboard_dims)

        obj_points = []
        img_points = []

        for path in image_paths:
            image = LaneImage(image_path=path)

            are_corners_found, corner_coords = cv2.findChessboardCorners(
                image=image.apply_colorspace('gray'),
                patternSize=chessboard_dims
            )

            if are_corners_found:
                obj_points.append(op_array)
                img_points.append(corner_coords)

            image_dims = image.size

        return obj_points, img_points, image_dims

    def _calculate_camera_matrix(
            self,
            obj_points: List[np.ndarray],
            img_points: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        camera_matrix = cv2.initCameraMatrix2D(
            objectPoints=obj_points,
            imagePoints=img_points,
            imageSize=self.image_dims
        )

        ret_val, camera_matrix, distortion_coefs, _, _ = cv2.calibrateCamera(
            objectPoints=obj_points,
            imageSize=self.image_dims,
            imagePoints=img_points,
            cameraMatrix=camera_matrix,
            distCoeffs=np.zeros((3, 3))
        )

        return camera_matrix, distortion_coefs

    def save(self, save_path: Union[Path, str]) -> None:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        save_path.write_bytes(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, file_path: Union[Path, str]) -> 'CameraSettings':
        if isinstance(file_path, str):
            file_path = Path(file_path)

        obj_dict = pickle.loads(file_path.read_bytes())

        return cls(** obj_dict)
