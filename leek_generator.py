import os
import glob
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.folder_translation_generator import FolderTranslationGenerator
from torch.utils.data import DataLoader
import visualization.vis_utils as vis
from multiprocessing.dummy import freeze_support


class LeekGenerator(FolderTranslationGenerator):
    """
    This generator takes a leek folder, and split them in two according to their name (ending in SD or SS)
    """
    def __init__(self, folder, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        super().__init__(folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)

    def _define_num_classes_(self):
        self.name_classes = ['A', 'B']
        num_classes = len(self.name_classes)
        self.group = {}
        self.group['A'] = [os.path.basename(i) for i in glob.glob(self.folder + '/**_SD.png')]
        self.group['B'] = [os.path.basename(i) for i in glob.glob(self.folder + '/**_SS.png')]
        return num_classes

    def _get_label_(self, item):
        return np.random.randint(len(self.name_classes))

    def _get_my_item_(self, idx, label):
        name_class_selected = self.name_classes[label]
        choice_image = np.random.choice(self.group[name_class_selected])
        canvas, random_center = self._transpose_selected_image(choice_image, label, idx)
        return canvas, label, random_center

import re
class LeekGeneratorDoubleSide(LeekGenerator):
    """
    For the first half of leek objects, they will have a horizontal translation opposite to the indicated one.
    """
    def __init__(self, folder, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50), type_double=0):
        self.num_images_in_folder = None
        # establish which half goes to the which side. Use 0 for training, 1 for testing. If type_double==0, the first half is swapped side. If type_double==1, the second half.
        self.type_double = type_double
        super(LeekGeneratorDoubleSide, self).__init__(folder, translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas=size_canvas, size_object=size_object)

    def _finalize_init_(self):
        self.num_images_in_folder = len(glob.glob(self.folder + '/**.png'))
        super(LeekGeneratorDoubleSide, self)._finalize_init_()

    def _random_translation_(self, groupID, image_name=None):
        minX, maxX, minY, maxY = self.translations_range[groupID]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        num = int(re.sub(r"\D", "", image_name))
        if (self.type_double and num > self.num_images_in_folder // 4) or \
           (not self.type_double and num <= self.num_images_in_folder // 4):  # it's 4 and not 2 because of the way the images are numbered
                x = self.size_canvas[0] // 2 - (x - self.size_canvas[0] // 2)
        return x, y

def do_stuff():
    leek_dataset = LeekGeneratorDoubleSide('./data/LeekImages_transparent', (198, 112), middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_object=(50, 50), type_double=1)
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    leek_dataset = LeekGenerator('./data/LeekImages_transparent', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    leek_dataset = LeekGenerator('./data/LeekImages_transparent', translation_type=(50, 150), middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    leek_dataset = LeekGenerator('./data/LeekImages_transparent', translation_type=(50, 150, 223, 224), middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()
