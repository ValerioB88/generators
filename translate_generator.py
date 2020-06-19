from torchvision import transforms
from compute_mean_std_dataset import compute_mean_and_std_from_dataset
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from generate_datasets.generators import get_range_translation
import pathlib
import pickle
import os
import filecmp
import numpy as np

class TranslateGenerator(ABC, Dataset):
    def __init__(self, translation_type, middle_empty, background_color_type, name_generator='', size_canvas=(224, 224), size_object=(50, 50), grayscale=False):
        self.transform = None
        self.translation_type = translation_type
        self.size_object = 60
        self.middle_empty = middle_empty
        self.name_generator = name_generator
        self.background_color_type = background_color_type
        self.stats = {}
        self.grayscale = grayscale
        self.size_canvas = size_canvas
        self.size_object = size_object  # x and y
        self.num_classes = self.define_num_classes()

        self.multi_translation_mode = 1 if isinstance(self.translation_type, dict) else 0
        self.translations_range = {}
        assert len(self.translation_type) == self.num_classes if self.multi_translation_mode else True

        if not self.multi_translation_mode:
            translation_type_tmp = dict([(i, self.translation_type) for i in range(self.num_classes)])
            self.translation_type_str = str.lower(str.split(str(self.translation_type), '.')[1])
        else:
            translation_type_tmp = self.translation_type
            self.translation_type_str = "".join(["".join([str(k), str.lower(str.split(str(v), '.')[1])]) for k, v in self.translation_type.items()])

        for idx, transl in translation_type_tmp.items():
            self.translations_range[idx] = get_range_translation(transl, self.size_object[1], self.size_canvas, self.size_object[0], self.middle_empty)

        self.finalize()

    def finalize(self):
        self.save_stats()

    def save_stats(self):
        pathlib.Path('./data/generators/').mkdir(parents=True, exist_ok=True)
        filename = '{}_tr_[{}]_bg_{}_md_{}_gs{}_sk.pickle'.format(type(self).__name__, self.translation_type_str, self.background_color_type.value, int(self.middle_empty), int(self.grayscale))
        if os.path.exists('./data/generators/{}'.format(filename)) and os.path.exists('./data/generators/stats_{}'.format(filename)):
            with open('./data/generators/tmp_{}'.format(filename), 'wb') as f:
                pickle.dump(self, f)
            # check if the files are the same
            if filecmp.cmp('./data/generators/{}'.format(filename), './data/generators/{}'.format(filename)):
                compute_mean_std = False
            else:
                compute_mean_std = True
            os.remove('./data/generators/tmp_{}'.format(filename))

        else:
            with open('./data/generators/{}'.format(filename), 'wb') as f:
                pickle.dump(self, f)
            compute_mean_std = True

        if compute_mean_std:
            self.stats = compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename))
        else:
            self.stats = pickle.load(open('./data/generators/stats_{}'.format(filename), 'rb'))
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

    def random_translation(self, groupID):
        minX, maxX, minY, maxY = self.translations_range[groupID]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

    @abstractmethod
    def define_num_classes(self):
        return 1, 2

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass
