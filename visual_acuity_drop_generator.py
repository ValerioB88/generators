from multiprocessing.dummy import freeze_support
import PIL.Image as Image
from visualization import vis_utils as vis
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.dummy_face_generators.utils_dummy_faces import draw_face, from_group_ID_to_features
from generate_datasets.generators.utils_generator import get_background_color, TranslationType, BackGroundColorType
import cv2
from generate_datasets.generators.translate_generator import TranslateGenerator
from generate_datasets.generators.leek_generator import LeekGenerator
from generate_datasets.generators.folder_translation_generator import FolderTranslationGenerator
import PIL.Image as Image
from PIL import ImageFilter

class VisualAcuityDropGenerator(TranslateGenerator):
    def __init__(self, blurring_coeff):
        self.blurring_coeff = blurring_coeff

    def _finalize_get_item_(self, canvas: Image, label, random_center):
        distance_from_center = np.linalg.norm(np.array(random_center) - np.array(self.size_canvas) / 2)
        canvas = canvas.filter(ImageFilter.GaussianBlur(distance_from_center * self.blurring_coeff))
        return canvas, label, random_center


#
# def extend_generator_with_visual_drop(generator_class, blurring_coeff):
#     global VisualAcuityDropGenerator
#     class VisualAcuityDropGenerator(generator_class):
#         def finalize_get_item(self, canvas: Image, label, random_center):
#             distance_from_center = np.linalg.norm(np.array(random_center) - np.array(self.size_canvas) / 2)
#             canvas = canvas.filter(ImageFilter.GaussianBlur(distance_from_center * self.blurring_coeff))
#             return canvas, label, random_center
#
#     setattr(generator_class, 'blurring_coeff', blurring_coeff)
#     return VisualAcuityDropGenerator

class LeekVisualDrop(VisualAcuityDropGenerator, LeekGenerator):
    def __init__(self, blurring_coeff, folder, translation_type, middle_empty, background_color_type, name_generator = '', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        VisualAcuityDropGenerator.__init__(self, blurring_coeff)
        LeekGenerator.__init__(self, folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)


def do_stuff():
    leek_dataset = LeekVisualDrop(blurring_coeff=0.05, folder='./data/LeekImages_transparent', translation_type=TranslationType.WHOLE, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    # extended_class = extend_generator_with_visual_drop(FolderTranslationGenerator, blurring_coeff=0.25)
    # leek_dataset = extended_class('./data/LeekImages_transparent', TranslationType.WHOLE, background_color_type=BackGroundColorType.WHITE, middle_empty=False, grayscale=False, name_generator='dataLeek', size_object=np.array([50, 50]), size_canvas=(224, 224))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()
