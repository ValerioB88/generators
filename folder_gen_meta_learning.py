from multiprocessing.dummy import freeze_support
import warnings
import framework_utils
from visualization import vis_utils as vis
from generate_datasets.generators.utils_generator import get_range_translation
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from torch.utils.data import Sampler
from generate_datasets.generators.folder_translation_generator import FolderGen
import warnings
import PIL.Image as Image


class FolderGenMetaLearning(FolderGen):
    def __init__(self, folder, translation_type_training, translation_type_test, sampler, **kwargs):
        self.translation_type_test = translation_type_test
        self.sampler = sampler(dataset=self)
        super().__init__(folder, translation_type=translation_type_training, **kwargs)

    def _finalize_init(self):
        if self.num_classes < self.sampler.k:
            warnings.warn(f'FolderGen: Number of classes in the folder < k. K will be changed to {self.num_classes}')
            self.sampler.k = self.num_classes

        self.translation_range_test = get_range_translation(self.translation_type_test, self.size_object[1], self.size_canvas, self.size_object[0], jitter=self.jitter)
        super()._finalize_init()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename),
                                                 data_loader=DataLoader(self, batch_sampler=self.sampler, num_workers=1),
                                                 max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

    def _get_label(self, idx):
        return idx[0]

    def _get_translation(self, label, image_name=None, idx=None):
        if idx[2] == '0':
            return self._random_translation(label, image_name)
        else:
            return self._random_translation_test(label, image_name)

    def _random_translation_test(self, groupID, image_name=None):
        minX, maxX, minY, maxY = self.translation_range_test
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

class NShotTaskSampler(Sampler):
    warning_done = False

    def __init__(self, dataset: FolderGen, n, k, q, num_tasks=1, episodes_per_epoch=999999, disjoint_train_and_test=True):
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.dataset = dataset
        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []  # batch is [label, index, train/test (0/1)]

            for task in range(self.num_tasks):
                # Get random classes
                episode_classes = np.random.choice(self.dataset.name_classes, size=self.k, replace=False)

                for k in episode_classes:
                    batch.extend([[k, i, 0] for i in np.random.choice(len(self.dataset.samples[k]), size=self.n)])

                for k in episode_classes:
                    exclude = [i[1] for i in batch if i[0] == k]
                    good_to_take = [i for i in range(len(self.dataset.samples[k])) if i not in exclude]
                    if not good_to_take:
                        if not self.warning_done:
                            warnings.warn(f'No elements left for testing in this class n. {k}.'
                                          f'We will sample from the whole dataset.'
                                          f'This is usually intentional if using different translations!'
                                          f'This warning won''t be repeated. ')
                            self.warning_done = True
                        good_to_take = len(self.dataset.samples[k])
                    batch.extend([[k, i, 1] for i in np.random.choice(good_to_take, size=self.q)])

            yield np.stack(batch)


def do_stuff():
    sampler = partial(NShotTaskSampler, n=2, k=5, q=1)
    multi_folder_omniglot = FolderGenMetaLearning('./data/Omniglot/transparent_white/images_background', translation_type_training=TranslationType.LEFTMOST, translation_type_test=TranslationType.WHOLE, sampler=sampler, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50))
    dataloader = DataLoader(multi_folder_omniglot, batch_sampler=sampler(dataset=multi_folder_omniglot), num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, multi_folder_omniglot.stats, title_lab=lab)

    sampler = partial(NShotTaskSampler, n=3, k=2, q=2)
    multi_folder_omniglot = FolderGenMetaLearning('./data/MNIST/png/training', translation_type_training=TranslationType.LEFTMOST, translation_type_test=TranslationType.WHOLE, sampler=sampler, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50))
    dataloader = DataLoader(multi_folder_omniglot, batch_sampler=sampler(dataset=multi_folder_omniglot), num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, multi_folder_omniglot.stats, title_lab=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()

