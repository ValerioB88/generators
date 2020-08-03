from multiprocessing.dummy import freeze_support
from visualization import vis_utils as vis
from generate_datasets.generators.utils_generator import get_range_translation
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from torch.utils.data import Sampler
from generate_datasets.generators.folder_translation_generator import FolderGen

class FolderGenMetaLearning(FolderGen):
    def __init__(self, folder, translation_type_training, translation_type_test, sampler, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        self.translation_type_test = translation_type_test
        self.sampler = sampler(dataset=self)
        super().__init__(folder, translation_type_training, middle_empty, background_color_type, name_generator, grayscale, size_canvas, size_object)

    def _finalize_init_(self):
        self.translation_range_test = get_range_translation(self.translation_type_test, self.size_object[1], self.size_canvas, self.size_object[0], self.middle_empty)
        super()._finalize_init_()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename), data_loader=DataLoader(self, batch_sampler=self.sampler, num_workers=1))

    def _get_label_(self, idx):
        return idx[0]

    def _get_translation_(self, label, image_name=None, idx=None):
        if idx[2] == 0:
            return self._random_translation_(label, image_name)
        else:
            return self._random_translation_test(label, image_name)

    def _random_translation_test(self, groupID, image_name=None):
        minX, maxX, minY, maxY = self.translation_range_test
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

    def _get_my_item_(self, idx, label):
        image_name = self.samples[label][idx[1]]

        canvas, random_center = self._transpose_selected_image(image_name, label, idx)
        # the label returned are the class labels (not the one used for meta-learning, those are done in the training step)
        return canvas, label, {'center': random_center}

class NShotTaskSampler(Sampler):
    def __init__(self, dataset: FolderGen, n, k, q, num_tasks=1, episodes_per_epoch=999999):
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
                episode_classes = np.random.choice(self.dataset.num_classes, size=self.k, replace=False)

                for k in episode_classes:
                    batch.extend([[k, i, 0] for i in np.random.choice(len(self.dataset.samples[k]), size=self.n)])

                for k in episode_classes:
                    exclude = [i[1] for i in batch if i[0] == k]
                    good_to_take = [i for i in range(len(self.dataset.samples[k])) if i not in exclude]
                    batch.extend([[k, i, 1] for i in np.random.choice(good_to_take, size=self.q)])

            yield np.stack(batch)

def do_stuff():
    sampler = partial(NShotTaskSampler, n=2, k=5, q=1)
    multi_folder_omniglot = FolderGenMetaLearning('./data/Omniglot/transparent_white/images_background', translation_type_training=TranslationType.LEFTMOST, translation_type_test=TranslationType.WHOLE, sampler=sampler, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50))
    dataloader = DataLoader(multi_folder_omniglot, batch_sampler=sampler(dataset=multi_folder_omniglot), num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, multi_folder_omniglot.stats['mean'], multi_folder_omniglot.stats['std'], title_lab=lab)

    sampler = partial(NShotTaskSampler, n=3, k=2, q=2)
    multi_folder_omniglot = FolderGenMetaLearning('./data/MNIST/png/training', translation_type_training=TranslationType.LEFTMOST, translation_type_test=TranslationType.WHOLE, sampler=sampler, middle_empty=False, background_color_type=BackGroundColorType.BLACK, name_generator='dataLeek', grayscale=False, size_canvas=(224, 224), size_object=(50, 50))
    dataloader = DataLoader(multi_folder_omniglot, batch_sampler=sampler(dataset=multi_folder_omniglot), num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, multi_folder_omniglot.stats['mean'], multi_folder_omniglot.stats['std'], title_lab=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()

