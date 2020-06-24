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
    This generator takes the two leek folder, and treat each folder as a separate class
    """
    def __init__(self, folder, translation_type, background_color_type: BackGroundColorType, middle_empty, grayscale, name_generator, size_canvas=(224, 224), size_object=(50, 50)):
        super(LeekGenerator, self).__init__(folder, translation_type, background_color_type, middle_empty, grayscale, name_generator, size_canvas=size_canvas, size_object=size_object)

    def define_num_classes(self):
        self.name_classes = ['A', 'B']
        num_classes = len(self.name_classes)
        self.group = {}
        self.group['A'] = [os.path.basename(i) for i in glob.glob(self.folder + '/**_SD.png')]
        self.group['B'] = [os.path.basename(i) for i in glob.glob(self.folder + '/**_SS.png')]
        return num_classes

    def __getitem__(self, idx):
        label = np.random.randint(len(self.name_classes))
        name_class_selected = self.name_classes[label]
        choice_image = np.random.choice(self.group[name_class_selected])
        canvas, random_center = self._transpose_selected_image(choice_image, label)
        return canvas, label, random_center


def do_stuff():
    leek_dataset = LeekGenerator('./data/LeekImages', TranslationType.LEFT, BackGroundColorType.BLACK, middle_empty=False, grayscale=False, name_generator='dataLeek', size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    leek_dataset = LeekGenerator('./data/LeekImages', translation_type=(50, 150), background_color_type=BackGroundColorType.BLACK, middle_empty=False, grayscale=False, name_generator='dataLeek', size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

    leek_dataset = LeekGenerator('./data/LeekImages', translation_type=(50, 150, 223, 224), background_color_type=BackGroundColorType.BLACK, middle_empty=False, grayscale=False, name_generator='dataLeek', size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()
