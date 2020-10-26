from torchvision import transforms
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from generate_datasets.generators.utils_generator import get_range_translation, TranslationType
import pathlib
import os
import numpy as np



class InputImagesGenerator(ABC, Dataset):
    def __init__(self, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), max_iteration_mean_std=50, verbose=True):
        self.verbose = verbose
        self.max_iteration_mean_std = max_iteration_mean_std
        self.transform = None
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

        self.stats = None

    def _finalize_init(self):
        self.map_name_to_num = {i: idx for idx, i in enumerate(self.name_classes)}
        self.map_num_to_name = {idx: i for idx, i in enumerate(self.name_classes)}
        print(f'\nMap class_name -> labels: \n {self.map_name_to_num}') if self.verbose else None
        self.save_stats()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename), max_iteration=self.max_iteration_mean_std, verbose=self.verbose)

    def save_stats(self):
        """
        We deleted everything concerning the comparison of dataloader - instead, we always compute the stats for now. It's time consuming but don't have time to come up with a better way.
        """
        pathlib.Path('./data/generators/').mkdir(parents=True, exist_ok=True)
        filename = 'none'
        self.stats = self.call_compute_stat(filename)
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
        if self.grayscale:
            self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])

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

    def _finalize_get_item(self, canvas, class_name, more):
        return canvas, class_name, more

    def _get_label(self, idx):
        return np.random.choice(self.name_classes)

    def _prepare_get_item(self):
        pass

    def _resize(self, image):
        if self.size_object is not None:
            image = image.resize(self.size_object)
        return image, (self.size_object if self.size_object is not None else -1)

    def _rotate(self, image):
        return image, 0

    def _transpose(self, image, image_name, class_name, idx):
        return image, (-1, -1)

    def __getitem__(self, idx):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        self._prepare_get_item()
        class_name = self._get_label(idx)

        # _get_image returns a PIL image
        image, image_name = self._get_image(idx, class_name)
        image, new_size = self._resize(image)
        image, rotate = self._rotate(image)

        canvas, random_center = self._transpose(image, image_name, class_name, idx)
        more = {'center': random_center, 'size': new_size, 'rotation': rotate}
        # canvas, class_name, more = self._get_my_item(idx, class_name)
        # get my item must return a PIL image
        canvas, class_name, more = self._finalize_get_item(canvas, class_name, more)

        if self.transform is not None:
            canvas = self.transform(canvas)
        else:
            canvas = np.array(canvas)

        return canvas, self.map_name_to_num[class_name], more
