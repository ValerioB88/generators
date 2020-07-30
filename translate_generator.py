from torchvision import transforms
from compute_mean_std_dataset import compute_mean_and_std_from_dataset
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from generate_datasets.generators.utils_generator import get_range_translation, TranslationType
import pathlib
import os
import filecmp
import numpy as np
import cloudpickle

class TranslateGenerator(ABC, Dataset):
    def __init__(self, translation_type, middle_empty, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        """
        @param translation_type: could either be one TypeTranslation, or a dict of TypeTranslation (one for each class), or a tuple of two elements (x and y for the translated location) or a tuple of 4 elements (minX, maxX, minY, maxY)
        """
        self.transform = None
        self.translation_type = translation_type
        self.middle_empty = middle_empty
        self.name_generator = name_generator
        self.background_color_type = background_color_type
        self.stats = {}
        self.grayscale = grayscale
        self.size_canvas = size_canvas
        self.size_object = size_object  # x and y
        self.num_classes = self._define_num_classes_()

        self.translations_range = {}
        def translation_type_to_str(translation_type):
            return str.lower(str.split(str(translation_type), '.')[1]) if isinstance(translation_type, TranslationType) else "_".join(str(translation_type).split(", "))

        size_object = self.size_object if self.size_object is not None else (0, 0)
        if isinstance(self.translation_type, dict):
            self.translation_type_str = "".join(["".join([str(k), translation_type_to_str(v)]) for k, v in self.translation_type.items()])
            if len(self.translation_type_str) > 250:
                self.translation_type_str = 'multi_long'
            assert len(self.translation_type) == self.num_classes
            for idx, transl in self.translation_type.items():
                self.translations_range[idx] = get_range_translation(transl, size_object[1], self.size_canvas, size_object[0], self.middle_empty)

        # Same translation type for all classes
        # can be TranslationType, tuple (X, Y) or tuple of (minX, maxX, minY, maxY)
        if not isinstance(self.translation_type, dict):
            self.translation_type_str = translation_type_to_str(self.translation_type)
            for idx in range(self.num_classes):
                self.translations_range[idx] = get_range_translation(self.translation_type, size_object[1], self.size_canvas, size_object[0], self.middle_empty)

        self._finalize_init_()

    def _finalize_init_(self):
        self.save_stats()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename))

    def save_stats(self):
        pathlib.Path('./data/generators/').mkdir(parents=True, exist_ok=True)
        filename = '{}_tr_[{}]_bg_{}_md_{}_gs{}_sk.pickle'.format(type(self).__name__, self.translation_type_str, self.background_color_type.value, int(self.middle_empty), int(self.grayscale))

        if os.path.exists('./data/generators/{}'.format(filename)) and os.path.exists('./data/generators/stats_{}'.format(filename)):
            if cloudpickle.load(open('./data/generators/{}'.format(filename), 'rb')).__dict__ == self.__dict__:
                compute_mean_std = False
            else:
                compute_mean_std = True
        else:
            compute_mean_std = True

        if compute_mean_std:
            cloudpickle.dump(self, open('./data/generators/{}'.format(filename), 'wb'))
            self.stats = self.call_compute_stat(filename)
        else:
            self.stats = cloudpickle.load(open('./data/generators/stats_{}'.format(filename), 'rb'))
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

    def _get_translation_(self, label, image_name=None, idx=None):
        return self._random_translation_(label, image_name)

    def _random_translation_(self, groupID, image_name=None):
        minX, maxX, minY, maxY = self.translations_range[groupID]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

    @abstractmethod
    def _define_num_classes_(self):
        raise NotImplementedError

    def __len__(self):
        return 300000  # np.iinfo(np.int64).max

    @abstractmethod
    def _get_my_item_(self, item, label):
        raise NotImplementedError

    def _finalize_get_item_(self, canvas, label, more):
        return canvas, label, more

    def _get_label_(self, idx):
        return np.random.randint(self.num_classes)

    def _prepare_get_item_(self):
        pass

    def _resize_(self, image):
        if self.size_object is not None:
            image = image.resize(self.size_object)
        return image

    def __getitem__(self, idx):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        self._prepare_get_item_()
        label = self._get_label_(idx)
        # _get_my_item should return a PIL image unless finalize_get_item is implemented and it does return a PIL image
        canvas, label, more = self._get_my_item_(idx, label)
        # get my item must return a PIL image
        canvas, label, more = self._finalize_get_item_(canvas, label, more)

        if self.transform is not None:
            canvas = self.transform(canvas)
        else:
            canvas = np.array(canvas)

        return canvas, label, more
