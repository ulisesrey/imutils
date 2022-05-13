import pandas as pd
import glob
import os
import numpy as np


def generate_absolute_coordinates(bodypart_coordinates, image_center_coordinates, stage_coordinates, pixel_dimensions):
    """

    :param bodypart_coords:
    :param image_center_coordinates:
    :param stage_coordinates:
    :param pixel_dimensions:
    :return:
    """
    image_center_coordinates = np.asarray(image_center_coordinates)

    result = image_center_coordinates - np.asarray(bodypart_coordinates)

    result_mm = result * pixel_dimensions

    # somehow x coordinates needed to be substracted, whereas y coordinates added
    absolute_coordinates_df = pd.DataFrame()
    absolute_coordinates_df['x'] = stage_coordinates[:, 0] - result_mm[:, 0]
    absolute_coordinates_df['y'] = stage_coordinates[:, 1] + result_mm[:, 1]

    return absolute_coordinates_df


def generate_absolute_coordinates_wrapper(project_path, image_center_coordinates, pixel_dimensions):
    """

    :param project_path: str
    :param image_center_coordinates: tuple with the coordinates of the stage in the behaviour camera (not the center of the image necesarilly!)
    :param pixel_dimensions:
    :return:
    """
    stage_coordinates = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    stage_coordinates = stage_coordinates[['x', 'y']].values

    dlc_df = pd.read_hdf(glob.glob(os.path.join(project_path, '*behaviour*/*behaviour*.h5'))[0])
    bodypart_coordinates = dlc_df[dlc_df.columns.levels[0][0]]['head'][['x', 'y']][:].values

    absolute_coordinates_df = generate_absolute_coordinates(bodypart_coordinates, image_center_coordinates,
                                                            stage_coordinates, pixel_dimensions)

    absolute_coordinates_df.to_csv(os.path.join(project_path, 'nose_coords_mm.csv'))


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--project_path", required=True, help="path to project folder")
    ap.add_argument("-img_center_coords", "--image_center_coordinates", nargs='+', type=float, required=True, help="")
    ap.add_argument("-px_dim", "--pixel_dimensions", type=float, required=True, help="")
    args = vars(ap.parse_args())

    project_path = args['project_path']
    image_center_coordinates = tuple(args['image_center_coordinates'])
    print(image_center_coordinates)
    print(type(image_center_coordinates))
    pixel_dimensions = args['pixel_dimensions']

    generate_absolute_coordinates_wrapper(project_path, image_center_coordinates, pixel_dimensions)
