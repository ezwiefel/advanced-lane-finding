from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import typer

from lanedetection.camera import CameraSettings
from lanedetection.image import LaneImage

app = typer.Typer()

process_app = typer.Typer()
app.add_typer(process_app, name='process')


# Calibrate Camera and Store Pickle Object
@app.command()
def calibrate(
        image_folder: Path = typer.Argument(..., file_okay=False, dir_okay=True, exists=True,
                                            help="The folder where the calibration images are stored"),
        chessboard_dims: Tuple[int, int] = typer.Option((9, 6),
                                                        help="The dimensions of the chessboard in the calibration "
                                                             "images (Columns Rows) Chessboard dimensions count "
                                                             "the inside points of the chessboard - so if a chessboard "
                                                             "has 8 columns and 6 rows, the option passed "
                                                             "would be '--chessboard-dims 7 5'"),
        image_name_template: str = typer.Option(default='**/calibration*.jpg',
                                                help='The template name to use for the calibration images. '
                                                     'Passed to glob'),
        calibration_data_path: Path = typer.Option('./data/calibration.pkl', exists=False, file_okay=True,
                                                   dir_okay=False,
                                                   help="The filename of where to write the camera calibration matrix")
):
    """Calibrate the camera given chessboard images and dimensions and store the calibration matrix
    in a serialized format

    python main.py calibrate --chessboard-dims 9 6 --calibration-data-path ./data/calibration/data.pkl ./data/calibration-images
    """

    calibration_images = image_folder.glob(image_name_template)
    camera_settings = CameraSettings()
    camera_settings.calibrate(calibration_images, chessboard_dims=chessboard_dims)
    camera_settings.save(calibration_data_path)


@app.command()
def load_calibration(
        calibration_data_path: Path = typer.Option(
            './data/calibration.pkl',
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="The filename of where the camera settings are stored")
):
    settings = CameraSettings.load(calibration_data_path)

    typer.echo(settings.distortion_coefs)


@process_app.command("images")
def process_images(
        image_paths: Path = typer.Argument(
            ...,
            file_okay=True,
            dir_okay=True,
            exists=True,
            help="The folder or file to process"),
        calibration_data_path: Path = typer.Option(
            './data/calibration.pkl',
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="The filename of where the camera settings are stored"),
        image_name_template: str = typer.Option(default='**/*.jpg',
                                                help='The template name to use for the images. Passed to glob'),
):
    """
    Process image(s) passed
    """
    settings = CameraSettings.load(calibration_data_path)

    if image_paths.is_dir():
        # Iterate through directory
        img_files = image_paths.glob(image_name_template)
    else:
        img_files = [image_paths]

    for file in img_files:
        image = LaneImage(file, camera_settings=settings)
        plt.imshow(image.apply_colorspace('gray'), cmap='gray')
        plt.show()


if __name__ == '__main__':
    app()
