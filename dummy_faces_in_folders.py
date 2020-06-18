import argparse
import pathlib

import cv2
import numpy as np
import torch
import utils as utils
from compute_mean_std_dataset import compute_mean_and_std_from_folder
from generate_dataset.dummy_faces_generator.utils import from_group_ID_to_features, get_background_color, draw_face, TranslationType, get_translation_values, BackGroundColorType


def save_dataset(num_faces_each_transl, size_canvas, length_face, width_face, folder, translationX, translationY, background_color_type, groupID):

    is_smiling, eyes_type = from_group_ID_to_features(groupID)

    for idx, (x, y) in enumerate(zip(translationX.flatten(), translationY.flatten())):
        face_center = x, y
        for i in range(num_faces_each_transl):
            # face_center = face_center_array[i]
            canvas = np.zeros(size_canvas, np.uint8) + get_background_color(background_color_type)
            canvas = draw_face(length_face, width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder + '/t{}_{}_f{}.png'.format(x, y, i), canvas)


def generate_all(only_priming_and_testing, background_color_str, faces_for_training, faces_for_priming, faces_for_testing):
    def generate_group_dataset(groupID, num_faces, folder, translation_type: TranslationType, middle_empty=True):
        folder = folder + '/group{}'.format(groupID)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        trX, trY = get_translation_values(translation_type, length_face, size_canvas, width_face, grid_size, middle_empty)
        save_dataset(num_faces, size_canvas, length_face, width_face, folder, trX, trY, background_color_type, groupID)

    is_server = False
    if torch.cuda.is_available():
        print('Detected cuda - you are probably on the server')
        is_server = True

    # Fixed params
    size_canvas = (224, 224, 1)
    length_face = 60
    width_face = length_face * 0.8125
    grid_size = 10 if is_server else 70
    background_color_type = BackGroundColorType[str.upper(background_color_str)]

    if only_priming_and_testing:
        print('We will generate only validation')
    else:
        print('Training set and validation to be generated')

    folder_base = './data/dummy_faces_random/train_on_whole_canvas/'

    if not only_priming_and_testing:
        folder = folder_base + '/train/'
        utils.clean_folder(folder)
        groups = range(6)
        [generate_group_dataset(i, faces_for_training, folder, TranslationType.WHOLE) for i in groups]

    folder = folder_base + '/priming/left'
    utils.clean_folder(folder)
    groups = range(6)
    [generate_group_dataset(i, faces_for_priming, folder, TranslationType.LEFT) for i in groups]

    folder = folder_base + '/priming/right'
    utils.clean_folder(folder)
    groups = range(6)
    [generate_group_dataset(i, faces_for_priming, folder, TranslationType.RIGHT) for i in groups]

    folder = folder_base + '/priming/whole'
    utils.clean_folder(folder)
    groups = range(6)
    [generate_group_dataset(i, faces_for_priming, folder, TranslationType.WHOLE) for i in groups]

    folder = folder_base + '/testing/left'
    utils.clean_folder(folder)
    groups = range(6)
    [generate_group_dataset(i, faces_for_testing, folder, TranslationType.LEFT, middle_empty=False) for i in groups]

    folder = folder_base + '/testing/right'
    utils.clean_folder(folder)
    groups = range(6)
    [generate_group_dataset(i, faces_for_testing, folder, TranslationType.RIGHT, middle_empty=False) for i in groups]

    if not only_priming_and_testing:
        max_iter_stats = 50
        print('Computing mean and std for {} images in the training set'.format(max_iter_stats))
        compute_mean_and_std_from_folder(folder_base + '/train/', max_iter_stats)


parser = argparse.ArgumentParser()
args = parser.parse_args()

if __name__ == '__main__':
    parser.add_argument("-v", "--only_priming_and_testing",
                        type=lambda x: bool(int(x)),
                        default=0)
    parser.add_argument("-c", "--background_colour",
                        help='background of the canvas: [white] [gray] [random]',
                        type=str,
                        default='white')
    parser.add_argument("-ftr", "--num_face_training",
                        type=int,
                        default=30)
    parser.add_argument("-fp", "--num_face_priming",
                        type=int,
                        default=30)
    parser.add_argument("-fte", "--num_face_testing",
                        type=int,
                        default=10)

    generate_all(args.only_priming_and_testing, args.background_colour, args.num_face_training, args.num_face_priming, args.num_face_testing)
