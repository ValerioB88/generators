from torchvision import transforms
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from generate_datasets.generators.utils_generator import get_range_translation, TranslationType, BackGroundColorType
import pathlib
import os
import numpy as np
import torchvision

class InputImagesGenerator(ABC, Dataset):
    def __init__(self, background_color_type=BackGroundColorType.BLACK, name_generator='', grayscale=False, convert_to_grayscale=None, size_canvas=(224, 224), num_image_calculate_mean_std=50, stats=None, verbose=True, additional_transform=None):
        self.additional_transform = additional_transform
        if self.additional_transform is None:
            self.additional_transform = []
        self.convert_to_grayscale = convert_to_grayscale
        self.verbose = verbose
        self.num_image_calculate_mean_std = num_image_calculate_mean_std
        self.transform = torchvision.transforms.Compose([*self.additional_transform, torchvision.transforms.ToTensor()])

        self.name_generator = name_generator
        self.stats = {}
        self.grayscale = grayscale
        self.size_canvas = size_canvas  # x and y
        self.background_color_type = background_color_type
        self.num_classes = self._define_num_classes_()
        if not self.num_classes:
            assert False, 'Dataset has no classes!'
        if not hasattr(self, 'name_classes'):
            self.name_classes = [str(i) for i in range(self.num_classes)]

        if self.convert_to_grayscale is None:
            self.convert_to_grayscale = self.grayscale

        self.stats = stats
        print(f"\n**Creating Generator {self.name_generator}**")

    def _finalize_init(self):
        self.class_to_idx = {i: idx for idx, i in enumerate(self.name_classes)}
        self.idx_to_class = {idx: i for idx, i in enumerate(self.name_classes)}
        print(f'Map class_name -> labels: \n {self.class_to_idx}') if self.verbose else None
        self.save_stats()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename), max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

    def save_stats(self):
        """
        We deleted everything concerning the comparison of dataloader - instead, we always compute the stats for now. It's time consuming but don't have time to come up with a better way.
        """
        pathlib.Path('./data/generators/').mkdir(parents=True, exist_ok=True)
        filename = 'none'
        if self.stats is None:
            self.stats = self.call_compute_stat(filename)
        else:
            print("Stats passed as an argument - they won't be computed")
        # else:
        #     self.stats = cloudpickle.load(open('./data/generators/stats_{}'.format(filename), 'rb'))
        if self.grayscale:
            mean = [self.stats['mean'][0]]
            std = [self.stats['std'][0]]
        else:
            mean = self.stats['mean']
            std = self.stats['std']

        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        if self.convert_to_grayscale:
            self.transform = transforms.Compose([*self.additional_transform, transforms.Grayscale(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([*self.additional_transform, transforms.ToTensor(),  normalize])

    @abstractmethod
    def _define_num_classes_(self):
        raise NotImplementedError

    @abstractmethod
    def _get_image(self, idx, class_name):  #-> Tuple[Image, str]:
        raise NotImplementedError

    def __len__(self):
        return 300000  # np.iinfo(np.int64).max

    # @abstractmethod
    # def _get_my_item(self, item, label):
    #     raise NotImplementedError

    def _finalize_get_item(self, canvas, class_name, more, idx=0):
        return canvas, class_name, more

    def _get_label(self, idx):
        return np.random.choice(self.name_classes)

    def _prepare_get_item(self):
        pass

    def _resize(self, image):
        if self.size_canvas is not None:
            image = image.resize(self.size_canvas)
        return image, (self.size_canvas if self.size_canvas is not None else -1)

    def _rotate(self, image):
        return image, 0

    def _transpose(self, image, image_name, class_name, idx):
        return image, (-1, -1)

    def __getitem__(self, idx, class_name=None):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        self._prepare_get_item()
        if class_name is None:
            class_name = self._get_label(idx)
        more = 1

        # _get_image returns a PIL image
        image, image_name = self._get_image(idx, class_name)
        image, new_size = self._resize(image)
        image, rotate = self._rotate(image)

        canvas, random_center = self._transpose(image, image_name, class_name, idx)
        # more = {'center': random_center, 'size': new_size, 'rotation': rotate}
        # get my item must return a PIL image
        canvas, class_name, more = self._finalize_get_item(canvas, class_name, more, idx)

        if self.transform is not None:
            canvas = self.transform(canvas)
        else:
            canvas = np.array(canvas)
        return canvas, self.class_to_idx[class_name], more


# class InputImagesGenerator3D(InputImagesGenerator):
#     def __getitem__(self, idx):
#         self._prepare_get_item(idx)
#         class_name = self._get_label(idx)
#
#         # _get_image returns a PIL image
#         image, image_name = self._get_image(idx, class_name)
#         image, new_size = self._resize(image)
#         image, rotate = self._rotate(image)
#
#         canvas, random_center = self._transpose(image, image_name, class_name, idx)
#         more = {}
#         # more = {'center': random_center, 'size': new_size, 'rotation': rotate}
#         # canvas, class_name, more = self._get_my_item(idx, class_name)
#         # get my item must return a PIL image
#         canvas, class_name, more = self._finalize_get_item(canvas, class_name, more, idx)
#
#         if self.transform is not None:
#             canvas = self.transform(canvas)
#         else:
#             canvas = np.array(canvas)
#
#         return canvas, self.map_name_to_num[class_name], more