from enum import Enum

import numpy as np


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


def get_range_translation(translation_type, size_object_y, size_canvas, size_object_x, middle_empty):
    if isinstance(translation_type, TranslationType):
        sy = size_object_y / 10
        sx = size_object_x / 10
        if translation_type == TranslationType.LEFT:
            minX = int(size_object_x / 2 + sx)
            maxX = int(size_canvas[0] / 2 - ((size_object_x / 2) if middle_empty else 0))
            minY = int(size_object_y / 2 + sy)
            maxY = int(size_canvas[1] - size_object_y / 2 - sy)

        if translation_type == TranslationType.RIGHT:
            # we use that magic pixel to make the values exactly the same when right or whole canvas
            minX = int(size_canvas[0] / 2 + ((size_object_x / 2) if middle_empty else 0) - 1)
            maxX = int(size_canvas[0] - size_object_x / 2 - sx)
            minY = int(size_object_y / 2 + sy)
            maxY = int(size_canvas[1] - size_object_y / 2 - sy)

        if translation_type == TranslationType.WHOLE:
            minX = int(size_object_x / 2 + sx)
            maxX = int(size_canvas[0] - size_object_x / 2 - sx)
            # np.sum(x_grid < np.array(size_canvas)[0] / 2) == np.sum(x_grid > np.array(size_canvas)[0] / 2)
            minY = int(size_object_y / 2 + sy)
            maxY = int(size_canvas[1] - size_object_y / 2 - sy)

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

        if translation_type == TranslationType.CENTER_ONE_PIXEL:
            minX = size_canvas[1] // 2
            maxX = size_canvas[1] // 2 + 1
            minY = size_canvas[0] // 2
            maxY = size_canvas[0] // 2 + 1

        if translation_type == TranslationType.CENTER_SMALL_AREA:
            minX = size_canvas[1] // 2 - 10
            maxX = size_canvas[1] // 2 + 10
            minY = size_canvas[0] // 2 - 10
            maxY = size_canvas[0] // 2 + 10


    elif isinstance(translation_type, tuple) and len(translation_type) == 2:
        minX, maxX, minY, maxY = translation_type[0], translation_type[0] + 1, translation_type[1], translation_type[1] + 1

    elif isinstance(translation_type, tuple) and len(translation_type) == 4:
        minX, maxX, minY, maxY = translation_type[0], translation_type[1], translation_type[2], translation_type[3]
    else:
        assert False, 'TranslationType type not understood'

    return minX, maxX, minY, maxY


def split_canvas_in_n_grids(num_classes, size_canvas, buffer=0):
    # buffer is the space left on the borders. Usually equal to the object size / 2
    num_grids_each_side = np.sqrt(num_classes)
    if num_grids_each_side % 1 != 0:
        assert False, 'You need a square number, but sqrt({}) is {}'.format(num_classes, num_grids_each_side)
    num_grids_each_side = int(num_grids_each_side)
    grid_size = size_canvas[0] // num_grids_each_side
    minX_side = np.linspace(buffer, size_canvas[0]-buffer-grid_size, num_grids_each_side)
    minY_side = np.linspace(buffer, size_canvas[1]-buffer-grid_size, num_grids_each_side)
    maxX_side = minX_side + grid_size
    maxY_side = minY_side + grid_size
    minX_m, minY_m = np.meshgrid(minX_side, minY_side)
    maxX_m, maxY_m = np.meshgrid(maxX_side, maxY_side)
    minX, minY = minX_m.flatten(), minY_m.flatten()
    maxX, maxY = maxX_m.flatten(), maxY_m.flatten()
    translation_type = {}
    for i in range(num_classes):
        translation_type[i] = int(minX[i]), int(maxX[i]), int(minY[i]), int(maxY[i])

    return translation_type

def get_translation_values(translation_type, length_face, size_canvas, width_face, grid_size, middle_empty):
    minX, maxX, minY, maxY = get_range_translation(translation_type, length_face, size_canvas, width_face, middle_empty)
    trX, trY = np.meshgrid(np.arange(minX, maxX, grid_size),
                           np.arange(minY, maxY,  grid_size))

    return trX, trY


def random_colour(N, range_col=(0, 254)):
    def random_one(N):
        return np.random.choice(np.arange(range_col[0], range_col[1]), N, replace=False)

    col = np.array([random_one(N), random_one(N), random_one(N)]).reshape((-1, 3))
    return col


def numpy_tuple_to_builtin_tuple(nptuple):
    return tuple([i.item() for i in nptuple])


class TranslationType(Enum):
    LEFT = 0
    RIGHT = 1
    WHOLE = 2
    CUSTOM = 3
    SMALL_AREA_RIGHT = 4
    VERY_SMALL_AREA_RIGHT = 5
    ONE_PIXEL = 6
    MULTI = 7
    CENTER_ONE_PIXEL = 8
    CENTER_SMALL_AREA = 9  # displaced by around 20 pixels

class BackGroundColorType(Enum):
    WHITE = 0
    BLACK = 1
    GREY = 2
    RANDOM = 3