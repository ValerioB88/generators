from multiprocessing.dummy import freeze_support
import framework_utils
import glob
import PIL.Image as Image
from torch.utils.data import DataLoader
import numpy as np

from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.translate_generator import TranslateGenerator, InputImagesGenerator
import pathlib
import os


def get_folder_generator(base_class):
    class FolderGen(base_class):

        """
        This generator is given a folder with images inside, and each image is treated as a different class.
        If the folder contains images, then multi_folder is set to False, each image is a class. Otherwise we traverse the folder
        structure and arrive to the last folder before each set of images, and that folder path is a class.
        """
        def __init__(self, folder, **kwargs):
            self.folder = folder
            self.name_classes = []
            self.samples = {}
            if not os.path.exists(self.folder):
                assert False, 'Folder {} does not exist'.format(self.folder)
            if np.all([os.path.splitext(i)[1] == '.png' for i in glob.glob(self.folder + '/**')]):
                self.multi_folder = False
                for i in np.sort(glob.glob(self.folder + '/**')):
                    if 'translation_type' in kwargs and isinstance(kwargs['translation_type'], dict) and os.path.splitext(os.path.basename(i))[0] not in kwargs['translation_type']:
                        continue
                    name_class = os.path.splitext(os.path.basename(i))[0]
                    self.name_classes.append(name_class)
                    self.samples[name_class] = [os.path.basename(i)]
            elif np.all([os.path.isdir(i) for i in glob.glob(self.folder + '/**')]):
                self.multi_folder = True
            else:
                assert False, f"Either provide a folder with only images or a folder with only folders (classes)\nFolder {folder}"

            folder_path = pathlib.Path(self.folder)
            if self.multi_folder:
                self.samples = {}
                from natsort import natsorted

                for dirpath, dirname, filenames in natsorted(os.walk(self.folder)):
                    dir_path = pathlib.Path(dirpath)
                    if filenames != [] and '.png' in filenames[0]:
                        name_class = (set(dir_path.parts) - (set(folder_path.parts) & set(dir_path.parts))).pop()
                        if 'translation_type' in kwargs and isinstance(kwargs['translation_type'], dict) and name_class not in kwargs['translation_type']:
                            continue
                        self.name_classes.append(name_class)
                        self.samples[name_class] = []
                        [self.samples[name_class].append(name_class + '/' + i) for i in filenames]
            super().__init__(**kwargs)
            self._finalize_init()

        def _finalize_init(self):
            super()._finalize_init()
            print(f"Created Dataset from folder: {self.folder}, {'multifolder' if self.multi_folder else 'single folder'}") if self.verbose else None

        def _define_num_classes_(self):
            return len(self.samples)


        def _get_image(self, idx, class_name):
            # Notice that we ignore idx and just take a random image
            num = np.random.choice(len(self.samples[class_name]))
            image_name = self.samples[class_name][num]
            # image_name = np.random.choice(self.samples[label])
            image = Image.open(self.folder + '/' + image_name)
            image = image.convert('RGB')
            return image, image_name
    return FolderGen


FolderGenWithTranslation = get_folder_generator(TranslateGenerator)
FolderGen = get_folder_generator(InputImagesGenerator)

def do_stuff():
    translation = {'S1_SD': TranslationType.LEFT, 'S10_SD': TranslationType.WHOLE}
    omn = './data/Omniglot/transparent_white/images_background'
    multi_folder_mnist = FolderGenWithTranslation('./data/LeekImages/transparent10/groupA', translation_type=translation, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50), max_iteration_mean_std=10)
    dataloader = DataLoader(multi_folder_mnist, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, multi_folder_mnist.stats, labels=lab)

    leek_dataset = FolderGenWithTranslation('./data/LeekImages/transparent', translation_type=TranslationType.LEFT, background_color_type=BackGroundColorType.RANDOM, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]), max_iteration_mean_std=10)
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, leek_dataset.stats, labels=lab)

    leek_dataset = FolderGenWithTranslation('./data/LeekImages/grouped', translation_type=TranslationType.LEFT, background_color_type=BackGroundColorType.RANDOM, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=np.array([50, 50]), max_iteration_mean_std=10)
    dataloader = DataLoader(leek_dataset, batch_size=4, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, leek_dataset.stats, labels=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()
