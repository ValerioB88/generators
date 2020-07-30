import os
from multiprocessing.dummy import freeze_support
from visualization import vis_utils as vis
import glob
import PIL.Image as Image

from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType, get_background_color
from generate_datasets.generators.extension_generators import finite_extension
import utils
from generate_datasets.generators.translate_generator import TranslateGenerator
import copy
class VertLineGenerator(TranslateGenerator):
    """
    This generator is given a folder with images inside, and each image is treated as a different class.
    """
    def __init__(self, size_lines: list, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        # Label indicates the position of the longest one, that is: label = 0, the first one is the longest
        self.size_lines = size_lines
        self.trial_lines = copy.deepcopy(self.size_lines)
        # assert np.all([i < size_object[1] for i in size_lines]), "Size Lines must be smaller than size_object_y"
        # assert len(self.size_lines) == 2, "Use only two lines!"
        assert self.size_lines[0] < self.size_lines[1], "Size line should be [short] [long]"
        super().__init__(translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas, size_object)

    def _define_num_classes_(self):
        return len(self.size_lines)

    def _get_canvas_vert_lines_(self, label):
        canvas = np.zeros(self.size_object, np.uint8) + get_background_color(self.background_color_type)
        self.trial_lines = np.insert(np.random.permutation([i for idx,i in enumerate(self.size_lines) if i != np.max(self.size_lines)]), label, np.max(self.size_lines))
        for idx, l in enumerate(self.trial_lines):
            canvas[(self.size_object[1] - l) // 2:
                   (self.size_object[1] + l) // 2, (self.size_object[1] // (len(self.trial_lines) + 1)) * (idx + 1)] = 254

        canvas = Image.fromarray(canvas)

        return canvas

    def _transpose_selected_image(self, canvas, label, idx):
        image = self._resize_(canvas)
        random_center = self._get_translation_(label, None, None)
        image_in_canvas = utils.copy_img_in_canvas(image, np.array(self.size_canvas), random_center, color_canvas=get_background_color(self.background_color_type))
        return image_in_canvas, random_center


    def _get_my_item_(self, idx, label):
        canvas = self._get_canvas_vert_lines_(label)
        canvas, random_center = self._transpose_selected_image(canvas, label, idx)
        return canvas, label, {'center': random_center}


def do_stuff():
    test_generator = finite_extension(grid_step_size=10, num_repetitions=10, base_class=VertLineGenerator)

    vert_gen = test_generator((1, 2, 40, 41), TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='vert_line', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    dataloader = DataLoader(vert_gen, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, vert_gen.stats['mean'], vert_gen.stats['std'], title_lab=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()

