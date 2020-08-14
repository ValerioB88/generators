from multiprocessing.dummy import freeze_support
from torch.utils.data import DataLoader
import numpy as np
import torch
import framework_utils
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
# from generate_datasets.generators.folder_translation_generator import FolderGen
import PIL.Image as Image
from PIL import ImageFilter
from external.Image_Foveation_Python.retina_transform import foveat_img
import cv2
from generate_datasets.generators.custom_transforms import SaltPepperNoiseFixation
from torchvision.transforms import Normalize


def add_salt_pepper_fixation(base_class, type_noise='pepper', strength=0.5):
    class AddSaltPepperNoiseFixation(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            idx_norm = [idx for idx, i in enumerate(self.transform.transforms) if isinstance(i, Normalize)][0]
            self.transform.transforms.insert(idx_norm, SaltPepperNoiseFixation(type_noise=type_noise,
                                                                               strength=strength,
                                                                               size_canvas=self.size_canvas))
    return AddSaltPepperNoiseFixation


def ffoveate_extension(base_class, blurring_coeff=0.248):
    class ImageFoveationGeneratorMixin(base_class):
        def __init__(self, *args, **kwargs):
            self.blurring_coeff = blurring_coeff
            super().__init__(*args, **kwargs)

        def _finalize_get_item_(self, canvas: Image, label, more):
            canvas, label, more = super()._finalize_get_item_(canvas, label, more)
            # convert PIL to opencv
            open_cv_image = np.array(canvas)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1]
            img_f = foveat_img(open_cv_image, sigma=self.blurring_coeff)
            img = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            canvas = Image.fromarray(img)
            return canvas, label, more

    return ImageFoveationGeneratorMixin


def visual_drop_extension(generator_class, blurring_coeff):
    class VisualAcuityDropGenerator(generator_class):
        def __init__(self,  *args, **kwargs):
            self.blurring_coeff = blurring_coeff
            super().__init__(*args, **kwargs)

        def _finalize_get_item_(self, canvas: Image, label, more):
            canvas, label, more =  super()._finalize_get_item_(canvas, label, more)
            distance_from_center = np.linalg.norm(np.array(more['center']) - np.array(self.size_canvas) / 2)
            canvas = canvas.filter(ImageFilter.GaussianBlur(distance_from_center * self.blurring_coeff))
            return canvas, label, more

    return VisualAcuityDropGenerator


def finite_extension(base_class, grid_step_size, num_repetitions=1):
    class FiniteTranslationGeneratorMixin(base_class):
        """
            This mixin will disable the random generation, and it will generate samples at fixed locations of a canvas.
            If shuffle is disabled, and num_repetitions is 1, it will pass through each translation, for each class at the time.
            If num_repetitions = n, it will resample that class n times for each translation (useful when each object within a class is
            itself randomly sampled).
            This means that if a classes contains multiple and different elements (as it should), num_repetition should be set AT LEAST to that number of different elements. You won't have the guarantee that every element of that class is gonna be present for every translation. (I don't think it matters)
        """

        def __init__(self, *args, **kwargs):
            self.grid_step_size = grid_step_size
            self.num_repetitions = num_repetitions
            self.mesh_trX, self.mesh_trY, self.mesh_class, self.mesh_rep = [], [], [], []
            self.current_index = 0
            self.get_translation_values = self.symmetric_steps
            super().__init__(*args, **kwargs)

        def min_to_max_steps(self, min, max):
            return np.arange(min, max, self.grid_step_size)

        def symmetric_steps(self, min, max):
            return np.hstack((np.arange(self.size_canvas[0] // 2, min, -self.grid_step_size)[1::][::-1], np.arange(self.size_canvas[0] // 2, max, self.grid_step_size)))

        def _finalize_init_(self):
            for groupID in range(self.num_classes):
                minX, maxX, minY, maxY = self.translations_range[groupID]
                # Recall that maxX should NOT be included (it's [minX, maxX)
                x, y, c, r = np.meshgrid(self.get_translation_values(minX, maxX),
                                         self.get_translation_values(minY, maxY),
                                         groupID,
                                         np.arange(self.num_repetitions))
                self.mesh_trX.extend(x.flatten())
                self.mesh_trY.extend(y.flatten())
                self.mesh_class.extend(c.flatten().astype(np.int64))
                self.mesh_rep.extend(r.flatten())
            print('Created Finite Dataset [{}] with {} elements'.format(self.name_generator, self.__len__()))
            print('Elements: X {}, Y {}'.format(self.get_translation_values(minX, maxX), self.get_translation_values(minY, maxY)))

            self.save_stats()

        def __len__(self):
            return len(self.mesh_trX)

        def _get_translation_(self, label=None, image_name=None, idx=None):
            return self.mesh_trX[idx], self.mesh_trY[idx]

        def _get_label_(self, item):
            return self.mesh_class[item]

    return FiniteTranslationGeneratorMixin


def random_resize_extension(base_class, low_val=1.0, high_val=1.0):
    """
    This works only for FolderTranslationGenerators, not for the dummyfaces!
    @param low_val: object smallest size is size_object * low_val
    @param high_val: object biggest size is size_object * high_val
    @return:
    """

    class RandomResize(base_class):
        def __init__(self, *args, **kwargs):
            self.low_val = low_val
            self.high_val = high_val
            assert self.low_val <= self.high_val
            super().__init__(*args, **kwargs)

        def _resize_(self, image: Image):
            if self.size_object is not None:
                resize_factor = np.random.uniform(self.low_val, self.high_val)
                image = image.resize((int(self.size_object[0] * resize_factor),
                                     int(self.size_object[1] * resize_factor)))
            return image
    return RandomResize


def do_stuff():
    # extended_class = finite_extension(DummyFaceRandomGenerator, grid_size=2, num_repetitions=2)
    #
    # mnist_dataset = extended_class(translation_type=TranslationType.WHOLE, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, length_face=50)
    # dataloader = DataLoader(mnist_dataset, batch_size=16, shuffle=False, num_workers=0)
    #
    # for i, data in enumerate(dataloader):
    #     img, lab, _ = data
    #     vis.imshow_batch(img, mnist_dataset.stats['mean'], mnist_dataset.stats['std'], title_lab=lab)
    #

    # translation_list = {0: TranslationType.CENTER_ONE_PIXEL,
    #                     1: TranslationType.RIGHT}
    #
    # extension = finite_extension(grid_size=8, num_repetitions=1, base_class=DummyFaceRandomGenerator)
    # dataset = extension(translation_type=translation_list, background_color_type=BackGroundColorType.BLACK, middle_empty=True, grayscale=False, name_generator='prova')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    #
    # for i, data in enumerate(dataloader):
    #     img, lab, _ = data
    #     vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)


    # extended_class = random_resize_extension(low_val=0.5, high_val=1.5,
    #                                          base_class=foveate_extension(blurring_coeff=1.5,
    #                                          base_class=finite_extension(FolderGen, grid_size=50, num_repetitions=2)))

    # extended_class = finite_extension(FolderGen, grid_step_size=50, num_repetitions=2)
    # mnist_dataset = extended_class(folder='./data/LeekImages/transparent', translation_type=TranslationType.WHOLE, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, size_object=(50, 50))
    # dataloader = DataLoader(mnist_dataset, batch_size=16, shuffle=False, num_workers=0)

    # for i, data in enumerate(dataloader):
    #     img, lab, _ = data
    #     vis.imshow_batch(img, mnist_dataset.stats['mean'], mnist_dataset.stats['std'], title_lab=lab)

    extended_class = finite_extension(FolderGen, grid_step_size=50, num_repetitions=2)
    mnist_dataset = extended_class(folder='./data/LeekImages/transparenc', translation_type=TranslationType.WHOLE, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(mnist_dataset, batch_size=16, shuffle=False, num_workers=0)

    extended_class = random_resize_extension(low_val=0.5, high_val=1.5,
                                             base_class=foveate_extension(blurring_coeff=1.5,
                                                                          base_class=finite_extension(grid_step_size=50, num_repetitions=1,
                                                                                                      base_class=FolderGen)))

    mnist_dataset = extended_class(folder='./data/MNIST/png/training', translation_type=TranslationType.HLINE, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, size_object=(50, 50))

    dataloader = DataLoader(mnist_dataset, batch_size=16, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        framework_utils.imshow_batch(img, mnist_dataset.stats['mean'], mnist_dataset.stats['std'], title_lab=lab)

    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.RIGHT}
    extension = finite_extension(grid_step_size=10, num_repetitions=1, base_class=LeekGenerator)

    leek_dataset = extension(folder='./data/LeekImages/transparent', translation_type=translation_list, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, size_object=(50, 50))
    dataloader = DataLoader(leek_dataset, batch_size=16, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        framework_utils.imshow_batch(img, leek_dataset.stats['mean'], leek_dataset.stats['std'], title_lab=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()

# %%

