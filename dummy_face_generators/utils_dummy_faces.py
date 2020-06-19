import cv2
import numpy as np
from enum import Enum

from generate_datasets.generators import random_colour, numpy_tuple_to_builtin_tuple


class EyesType(Enum):
    CIRCLE = 0
    SQUARE = 1
    CROSS = 2


def from_group_ID_to_features(groupID):
    if groupID == 0:
        is_smiling = True
        eyes_type = EyesType.CIRCLE
    if groupID == 1:
        is_smiling = False
        eyes_type = EyesType.CIRCLE
    if groupID == 2:
        is_smiling = True
        eyes_type = EyesType.SQUARE
    if groupID == 3:
        is_smiling = False
        eyes_type = EyesType.SQUARE
    if groupID == 4:
        is_smiling = True
        eyes_type = EyesType.CROSS
    if groupID == 5:
        is_smiling = False
        eyes_type = EyesType.CROSS
    return is_smiling, eyes_type

def draw_face(length_face, width_face, canvas, face_center, eyes_type: EyesType, is_smiling=True, scrambled=False, random_gray_face_color=True, random_face_jitter=True):
    length_face = length_face + (np.random.uniform(-int(length_face * 0.2), int(length_face * 0.2)) if random_face_jitter else 0)
    width_face = width_face + (np.random.uniform(-int(width_face * 0.2), int(width_face * 0.2)) if random_face_jitter else 0)

    cv2.ellipse(canvas, face_center, (int(length_face / 2),
                                      int(width_face / 2)), 90, 0, 360,
                color=numpy_tuple_to_builtin_tuple(tuple(random_colour(1, range_col=(20, 234))[0])) if random_gray_face_color else (150,) * 3,
                thickness=-1)

    eyes_dist = 1 / 6
    eyes_noise = lambda: np.random.uniform(-int(0.06 * length_face), int(0.06 * length_face)) if random_face_jitter else 0
    scrambled_factor = 0
    sf_left_x, sf_right_x, sf_left_y, sf_right_y = 0, 0, 0, 0
    if scrambled:
        sf_left_x = int(width_face * 1 / 6)
        sf_left_y = - int(length_face * 1 / 6)
        sf_right_x = -int(width_face * 1 / 6)
        sf_right_y = int(length_face * 1 / 2)

    if eyes_type == EyesType.CIRCLE:
        cv2.circle(canvas,
                   center=(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + sf_left_x,
                           face_center[1] - int(length_face * eyes_dist + eyes_noise()) + sf_left_y),
                   radius=int(length_face / 15),
                   color=numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]),
                   thickness=-1,
                   lineType=cv2.LINE_4)
        cv2.circle(canvas,
                   center=(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + sf_right_x,
                           face_center[1] - int(length_face * eyes_dist + eyes_noise()) + sf_right_y),
                   radius=int(length_face / 15),
                   color=numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]),
                   thickness=-1,
                   lineType=cv2.LINE_4)
    if eyes_type == EyesType.SQUARE:
        diag_square = int(length_face / 15)
        cv2.rectangle(canvas,
                      (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_left_y),
                      (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_left_y),
                      numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]),
                      thickness=-1)
        cv2.rectangle(canvas,
                      (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_right_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_right_y),
                      (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_right_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_right_y),
                      numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]),
                      thickness=-1)

    if eyes_type == EyesType.CROSS:
        length_line = int(length_face / 15)
        thickness = 2
        # left eye
        cv2.line(canvas,
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_y),
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_y),
                 numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]), thickness=thickness)
        cv2.line(canvas,
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_y),
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_y),
                 numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]), thickness=thickness)
        # right eye
        cv2.line(canvas,
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_y),
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_y),
                 numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]), thickness=thickness)
        cv2.line(canvas,
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_y),
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_y),
                 numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]), thickness=thickness)

    ## Draw nose
    sf_x, sf_y = 0, 0
    if scrambled:
        sf_x = 0  # + int(width_face * 1/5)
        sf_y = + int(length_face * 1/6)
    nose_noise = lambda: np.random.uniform(-int(0.07 * length_face), int(0.07 * length_face)) if random_face_jitter else 0
    pts = np.array([[int(face_center[0] + length_face / 10 + nose_noise()) + sf_x,
                     int(face_center[1] + length_face / 10 + nose_noise()) + sf_y],
                    [int(face_center[0] - length_face / 10 + nose_noise()) + sf_x,
                     int(face_center[1] + length_face / 10 + nose_noise()) + sf_y],
                    [int(face_center[0]) + sf_x,
                     int(face_center[1]) + sf_y]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]))

    ## Draw mouth
    mouth_noise = lambda: np.random.uniform(-int(0.05 * length_face), int(0.05 * length_face)) if random_face_jitter else 0
    if scrambled:
        scrambled_factor = - int(length_face * 1/3)
    if is_smiling:
        pts = np.array([[int(face_center[0] + length_face / 5 + mouth_noise()),
                         int(face_center[1] + length_face / 5 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0] - length_face / 5),
                         int(face_center[1] + length_face / 5 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0]),
                         int(face_center[1]) + length_face / 2.7 + mouth_noise() + scrambled_factor]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]))
    else:
        pts = np.array([[int(face_center[0] + length_face / 5 + mouth_noise()),
                         int(face_center[1] + length_face / 2.7 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0] - length_face / 5 + mouth_noise()),
                         int(face_center[1] + length_face / 2.7 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0]),
                         int(face_center[1]) + length_face / 5 + mouth_noise() + scrambled_factor]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_colour(1, range_col=(0, 10))[0]))

    return canvas


