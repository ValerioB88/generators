import os
from multiprocessing.dummy import freeze_support
from visualization import vis_utils as vis
import glob
import PIL.Image as Image

from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from generate_datasets.generators import TranslationType, BackGroundColorType, get_background_color

import utils
from translate_generator import TranslateGenerator

class FolderTranslationGenerator(TranslateGenerator):
    '''
    This generator is given a folder with images inside, and each image is treated as a different class.
    '''
    def __init__(self, folder, translation_type: TranslationType, background_color_type: BackGroundColorType, middle_empty, grayscale, name_generator, size_canvas=(224, 224), size_object=None):
        self.folder = folder
        self.folder_basename = os.path.basename(os.path.normpath(folder))
        self.name_classes = [os.path.basename(i) for i in glob.glob(self.folder + '/**')]

        super(FolderTranslationGenerator, self).__init__(translation_type, middle_empty, background_color_type, name_generator, size_canvas, size_object, grayscale)

    def define_num_classes(self):
        return len(self.name_classes)

    def random_translation(self, groupID):
        minX, maxX, minY, maxY = self.translations_range[groupID]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

    def set_range_translation(self, minX, maxX, minY, maxY):
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

    def __len__(self):
        return 300000  # np.iinfo(np.int64).max

    def _transpose_selected_image(self, image_name, label):
        image = Image.open(self.folder + '/' + image_name)
        if self.size_object is not None:
            image = image.resize(self.size_object)

        random_center = self.random_translation(label)
        image_in_canvas = utils.copy_img_in_canvas(image, np.array(self.size_canvas), random_center, color_canvas=get_background_color(self.background_color_type))

        if self.transform is not None:
            canvas = self.transform(image_in_canvas)
        else:
            canvas = np.array(image_in_canvas)

        return canvas, random_center

    def __getitem__(self, idx):
        label = np.random.randint(len(self.name_classes))
        name_class_selected = self.name_classes[label]
        canvas, random_center = self._transpose_selected_image(name_class_selected, label)
        return canvas, label, random_center

def do_stuff():
    leek_dataset = FolderTranslationGenerator('./data/LeekImages', TranslationType.LEFT, background_color_type=BackGroundColorType.RANDOM, middle_empty=False, grayscale=False, name_generator='dataLeek', size_object=np.array([50, 50]), size_canvas=(400, 400))
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()