import cv2
import numpy as np
from enum import Enum

class EyesType(Enum):
    CIRCLE = 0
    SQUARE = 1
    CROSS = 2


class TranslationType(Enum):
    LEFT = 0
    RIGHT = 1
    WHOLE = 2
    CUSTOM = 3
    SMALL_AREA_RIGHT = 4
    VERY_SMALL_AREA_RIGHT = 5
    ONE_PIXEL = 6
    MULTI = 7

class BackGroundColorType(Enum):
    WHITE = 0
    BLACK = 1
    GREY = 2
    RANDOM = 3

def get_background_color(background_type):
    if background_type == BackGroundColorType.WHITE:
        background_color = 254
    elif background_type == BackGroundColorType.BLACK:
        background_color = 0
    elif background_type == BackGroundColorType.GREY:
        background_color = 170
    elif background_type == BackGroundColorType.RANDOM:
        background_color = np.random.randint(0, 254)

    return background_color


def get_range_translation(translation_type, length_face, size_canvas, width_face, middle_empty):
    lfb = length_face / 10
    wfb = width_face / 10
    if translation_type == TranslationType.LEFT:
        minX = int(width_face / 2 + wfb)
        maxX = int(size_canvas[0] / 2 - ((width_face / 2) if middle_empty else 0))
        minY = int(length_face / 2 + lfb)
        maxY = int(size_canvas[1] - length_face / 2 - lfb)

    if translation_type == TranslationType.RIGHT:
        # we use that magic pixel to make the values exactly the same when right or whole canvas
        minX = int(size_canvas[0] / 2 + ((width_face / 2) if middle_empty else 0) - 1)
        maxX = int(size_canvas[0] - width_face / 2 - wfb)
        minY = int(length_face / 2 + lfb)
        maxY = int(size_canvas[1] - length_face / 2 - lfb)

    if translation_type == TranslationType.WHOLE:
        minX = int(width_face / 2 + wfb)
        maxX = int(size_canvas[0] - width_face / 2 - wfb)
        # np.sum(x_grid < np.array(size_canvas)[0] / 2) == np.sum(x_grid > np.array(size_canvas)[0] / 2)
        minY = int(length_face / 2 + lfb)
        maxY = int(size_canvas[1] - length_face / 2 - lfb)

    #
    if translation_type == TranslationType.SMALL_AREA_RIGHT:
        minX = int(size_canvas[1] / 2 + (size_canvas[1] / 2) * (1 / 3))
        maxX = int(size_canvas[1] / 2 + (size_canvas[1] / 2) * (2 / 3))
        minY = int(0 + (size_canvas[0] / 2) * (1 / 3))
        maxY = int(0 + (size_canvas[0] / 2) * (2 / 3))

    if translation_type == TranslationType.VERY_SMALL_AREA_RIGHT:
        minX = int(size_canvas[1] / 2 + (size_canvas[1] / 2) * (4 / 10))
        maxX = int(size_canvas[1] / 2 + (size_canvas[1] / 2) * (6 / 10))
        minY = int(0 + (size_canvas[0] / 2) * (4 / 10))
        maxY = int(0 + (size_canvas[0] / 2) * (6 / 10))

    if translation_type == TranslationType.ONE_PIXEL:
        minX = int(size_canvas[1] * 0.74)
        maxX = int(size_canvas[1] * 0.74) + 1
        minY = int(size_canvas[0] * 0.25)
        maxY = int(size_canvas[0] * 0.25) + 1
    return minX, maxX, minY, maxY


def get_translation_values(translation_type, length_face, size_canvas, width_face, grid_size, middle_empty):
    minX, maxX, minY, maxY = get_range_translation(translation_type, length_face, size_canvas, width_face, middle_empty)
    trX, trY = np.meshgrid(np.arange(minX, maxX, grid_size),
                           np.arange(minY, maxY,  grid_size))

    return trX, trY

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
                color=numpy_tuple_to_builtin_tuple(tuple(random_col(1, range_col=(20, 234))[0])) if random_gray_face_color else (150,) * 3,
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
                   color=numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]),
                   thickness=-1,
                   lineType=cv2.LINE_4)
        cv2.circle(canvas,
                   center=(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + sf_right_x,
                           face_center[1] - int(length_face * eyes_dist + eyes_noise()) + sf_right_y),
                   radius=int(length_face / 15),
                   color=numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]),
                   thickness=-1,
                   lineType=cv2.LINE_4)
    if eyes_type == EyesType.SQUARE:
        diag_square = int(length_face / 15)
        cv2.rectangle(canvas,
                      (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_left_y),
                      (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_left_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_left_y),
                      numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]),
                      thickness=-1)
        cv2.rectangle(canvas,
                      (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_right_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - diag_square) + sf_right_y),
                      (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_right_x,
                       int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + diag_square) + sf_right_y),
                      numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]),
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
                      numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]), thickness=thickness)
        cv2.line(canvas,
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_y),
                 (int(face_center[0] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_left_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_left_y),
                 numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]), thickness=thickness)
        # right eye
        cv2.line(canvas,
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_y),
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_y),
                 numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]), thickness=thickness)
        cv2.line(canvas,
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_y),
                 (int(face_center[0] + int(length_face * eyes_dist + eyes_noise()) - length_line) + sf_right_x,
                  int(face_center[1] - int(length_face * eyes_dist + eyes_noise()) + length_line) + sf_right_y),
                 numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]), thickness=thickness)

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
    cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]))

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
        cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]))
    else:
        pts = np.array([[int(face_center[0] + length_face / 5 + mouth_noise()),
                         int(face_center[1] + length_face / 2.7 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0] - length_face / 5 + mouth_noise()),
                         int(face_center[1] + length_face / 2.7 + mouth_noise()) + scrambled_factor],
                        [int(face_center[0]),
                         int(face_center[1]) + length_face / 5 + mouth_noise() + scrambled_factor]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], numpy_tuple_to_builtin_tuple(random_col(1, range_col=(0, 10))[0]))

    return canvas


def random_col(N, range_col=(0, 254)):
    def random_one(N):
        return np.random.choice(np.arange(range_col[0], range_col[1]), N, replace=False)

    col = np.array([random_one(N), random_one(N), random_one(N)]).reshape((-1, 3))
    return col


def numpy_tuple_to_builtin_tuple(nptuple):
    return tuple([i.item() for i in nptuple])
