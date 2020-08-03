import os
from multiprocessing.dummy import freeze_support

import framework_utils
from visualization import vis_utils as vis
import glob
import PIL.Image as Image

from torch.utils.data import DataLoader
from torchvision import utils
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType, get_background_color

import utils
from generate_datasets.generators.translate_generator import TranslateGenerator

class FolderGen(TranslateGenerator):
    """
    This generator is given a folder with images inside, and each image is treated as a different class.
    If the folder contains images, then multi_folder is set to False, each image is a class. Otherwise we traverse the folder
    structure and arrive to the last folder before each set of images, and that folder path is a class.
    """
    def __init__(self, folder, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        self.folder = folder
        if not os.path.exists(self.folder):
            assert False, 'Folder {} does not exist'.format(self.folder)
        if np.all([os.path.splitext(i)[1] == '.png' for i in glob.glob(self.folder + '/**')]):
            self.multi_folder = False
            self.name_classes = [os.path.basename(i) for i in np.sort(glob.glob(self.folder + '/**'))]
            self.samples = {k: [str.split(i, self.folder)[-1].strip('\\')] for k, i in enumerate(np.sort(glob.glob(self.folder + '/**')))}
        elif np.all([os.path.splitext(i)[1] == '' for i in glob.glob(self.folder + '/**')]):
            self.multi_folder = True
        else:
            assert False, "Either provide a folder with only images or a folder with only folder (classes)"

        self.name_classes = []
        index_class = 0
        if self.multi_folder:
            self.samples = {}
            for dirpath, dirname, filenames in os.walk(self.folder):
                if filenames != [] and '.png' in filenames[0]:
                    name_class = str.split(dirpath, self.folder)[-1].strip('\\')
                    self.name_classes.append(name_class)
                    self.samples[index_class] = []
                    [self.samples[index_class].append(name_class + '/' + i) for i in filenames]
                    index_class += 1


        super().__init__(translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas, size_object)

    def _finalize_init_(self):
        super()._finalize_init_()
        print('Created Dataset from folder: {}, {}'.format(self.folder, 'multifolder' if self.multi_folder else 'single folder'))

    def _define_num_classes_(self):
        return len(self.samples)

    def _transpose_selected_image(self, image_name, label, idx):
        image = Image.open(self.folder + '/' + image_name)
        image = self._resize_(image)
        random_center = self._get_translation_(label, image_name, idx)
        image_in_canvas = framework_utils.copy_img_in_canvas(image, np.array(self.size_canvas), random_center, color_canvas=get_background_color(self.background_color_type))

        return image_in_canvas, random_center

    def _get_my_item_(self, idx, label):
        # Notice that we ignore idx and just take a random image
        image_name = np.random.choice(self.samples[label])

        canvas, random_center = self._transpose_selected_image(image_name, label, idx)
        return canvas, label, {'center': random_center}

def do_stuff():
    multi_folder_mnist = FolderGen('./data/Omniglot/transparent_white/images_background', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50))
    dataloader = DataLoader(multi_folder_mnist, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, multi_folder_mnist.stats['mean'], multi_folder_mnist.stats['std'], title_lab=lab)

    leek_dataset = FolderGen('./data/LeekImages/transparent', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.RANDOM, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title_lab=lab)

    leek_dataset = FolderGen('./data/LeekImages/grouped', TranslationType.LEFT, middle_empty=False, background_color_type=BackGroundColorType.RANDOM, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]))
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title_lab=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()