from torchvision import transforms
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from generate_datasets.generators.utils_generator import get_range_translation, TranslationType
import pathlib
import os
import numpy as np
import cloudpickle


class InputImagesGenerator(ABC, Dataset):
    def __init__(self, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50), jitter=0, max_iteration_mean_std=50):
        self.max_iteration_mean_std = max_iteration_mean_std
        self.transform = None
        self.name_generator = name_generator
        self.stats = {}
        self.grayscale = grayscale
        self.size_canvas = size_canvas
        self.size_object = size_object  # x and y
        self.background_color_type = background_color_type
        self.num_classes = self._define_num_classes_()
        if not self.num_classes:
            assert False, 'Dataset has no classes!'
        if not hasattr(self, 'num_classes'):
            self.name_classes = [str(i) for i in range(self.num_classes)]
        self.size_object = self.size_object if self.size_object is not None else (0, 0)
        self.stats = None

    def _finalize_init(self):
        self.map_name_to_num = {i: idx for idx, i in enumerate(self.name_classes)}
        self.map_num_to_name = {idx: i for idx, i in enumerate(self.name_classes)}

        print(f'Map class_name -> labels: \n {self.map_name_to_num}')
        self.save_stats()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename), max_iteration=self.max_iteration_mean_std)

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

    def __len__(self):
        return 300000  # np.iinfo(np.int64).max

    @abstractmethod
    def _get_my_item(self, item, label):
        raise NotImplementedError

    def _finalize_get_item(self, canvas, class_name, more):
        return canvas, class_name, more

    def _get_label(self, idx):
        return np.random.choice(self.name_classes)

    def _prepare_get_item(self):
        pass

    def _resize(self, image):
        if self.size_object is not None:
            image = image.resize(self.size_object)
        return image

    def __getitem__(self, idx):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        self._prepare_get_item()
        class_name = self._get_label(idx)
        # _get_my_item should return a PIL image unless finalize_get_item is implemented and it does return a PIL image
        canvas, class_name, more = self._get_my_item(idx, class_name)
        # get my item must return a PIL image
        canvas, class_name, more = self._finalize_get_item(canvas, class_name, more)

        if self.transform is not None:
            canvas = self.transform(canvas)
        else:
            canvas = np.array(canvas)

        return canvas, self.map_name_to_num[class_name], more
