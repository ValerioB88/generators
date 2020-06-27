from multiprocessing.dummy import freeze_support
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.dummy_face_generators.dummy_faces_random_generator import DummyFaceRandomGenerator
from generate_datasets.generators.translate_generator import TranslateGenerator
from generate_datasets.generators.leek_generator import LeekGenerator
import visualization.vis_utils as vis
from generate_datasets.generators.fixed_generator import FixedTranslationGenerator
from generate_datasets.generators.visual_acuity_drop_generator import VisualAcuityDropGenerator

class DummyFacesFixed(FixedTranslationGenerator, DummyFaceRandomGenerator):
    def __init__(self, grid_size, num_repetitions, translation_type, middle_empty, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), length_face=60):
        FixedTranslationGenerator.__init__(self, grid_size, num_repetitions)
        DummyFaceRandomGenerator.__init__(self, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, length_face=length_face)


class LeekImagesFixed(FixedTranslationGenerator, LeekGenerator):
    def __init__(self, folder, grid_size, num_repetitions, translation_type, middle_empty, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        FixedTranslationGenerator.__init__(self, grid_size, num_repetitions)
        LeekGenerator.__init__(self, folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)

class LeekVisualDrop(VisualAcuityDropGenerator, LeekGenerator):
    def __init__(self, blurring_coeff, folder, translation_type, middle_empty, background_color_type, name_generator = '', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        VisualAcuityDropGenerator.__init__(self, blurring_coeff)
        LeekGenerator.__init__(self, folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)


class LeekImagesFixedVisualDrop(FixedTranslationGenerator, VisualAcuityDropGenerator, LeekGenerator):
    def __init__(self, folder, grid_size, num_repetitions, blurred_coeff, translation_type, middle_empty, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        FixedTranslationGenerator.__init__(self, grid_size, num_repetitions)
        VisualAcuityDropGenerator.__init__(self, blurred_coeff)
        LeekGenerator.__init__(self, folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)

def do_stuff():
    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.RIGHT}
    leek_dataset = LeekImagesFixedVisualDrop(folder='./data/LeekImages_transparent', grid_size=10, num_repetitions=1, blurred_coeff=0.05, translation_type=translation_list, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.VERY_SMALL_AREA_RIGHT}

    dataset = DummyFacesFixed(grid_size=8, num_repetitions=1, translation_type=translation_list, background_color_type=BackGroundColorType.BLACK, middle_empty=True, grayscale=False, name_generator='prova')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)

    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.RIGHT}
    leek_dataset = LeekImagesFixed(folder='./data/LeekImages_transparent', grid_size=10, num_repetitions=1, translation_type=translation_list, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='',  grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()

# %%

