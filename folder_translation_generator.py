import os
from multiprocessing.dummy import freeze_support
from visualization import vis_utils as vis
import glob
import PIL.Image as Image

from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType, get_background_color

import utils
from generate_datasets.generators.translate_generator import TranslateGenerator

class FolderTranslationGenerator(TranslateGenerator):
    '''
    This generator is given a folder with images inside, and each image is treated as a different class.
    '''
    def __init__(self, folder, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        self.folder = folder
        if not os.path.exists(self.folder):
            assert False, 'Folder {} does not exist'.format(self.folder)
        self.name_classes = [os.path.basename(i) for i in np.sort(glob.glob(self.folder + '/**'))]

        super().__init__(translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas, size_object)

    def _define_num_classes_(self):
        return len(self.name_classes)

    def _transpose_selected_image(self, image_name, label, idx):
        image = Image.open(self.folder + '/' + image_name)
        if self.size_object is not None:
            image = image.resize(self.size_object)

        random_center = self._get_translation_(label, image_name, idx)
        image_in_canvas = utils.copy_img_in_canvas(image, np.array(self.size_canvas), random_center, color_canvas=get_background_color(self.background_color_type))

        return image_in_canvas, random_center

    def _get_label_(self, idx):
        return np.random.randint(len(self.name_classes))

    def _get_my_item_(self, idx, label):
        name_class_selected = self.name_classes[label]
        canvas, random_center = self._transpose_selected_image(name_class_selected, label, idx)
        return canvas, label, {'center': random_center}

class MultiFoldersTranslationGenerator(FolderTranslationGenerator):
    """
    This generator is given directory which contains many folder, and each folder is a class.
    """
    def _get_my_item_(self, idx, label):
        name_class_selected = self.name_classes[label]
        image_name = name_class_selected + '/' + os.path.basename(np.random.choice(glob.glob(self.folder + '/' + name_class_selected + '/**.png')))
        canvas, random_center = self._transpose_selected_image(image_name, label, idx)
        return canvas, label, {'center': random_center}


def do_stuff():
    multi_folder_mnist = MultiFoldersTranslationGenerator('./data/MNIST/png/training/', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    dataloader = DataLoader(multi_folder_mnist, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, multi_folder_mnist.stats['mean'], multi_folder_mnist.stats['std'], title_lab=lab)

    leek_dataset = FolderTranslationGenerator('./data/LeekImages_transparent', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.RANDOM, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title_lab=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()