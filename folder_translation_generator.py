from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
from multiprocessing.dummy import freeze_support
from torch.utils.data import Sampler
import torch
from torch.utils.data import SequentialSampler, RandomSampler
import framework_utils
import glob
import PIL.Image as Image
import copy
from natsort import natsorted
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.translate_generator import TranslateGenerator, InputImagesGenerator
import pathlib
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
import torchvision
from torch.utils.data import Dataset
import os
from torchvision.datasets import ImageFolder
import re

class ImageFolderSelective(ImageFolder):
    def __init__(self, name_classes=None, num_viewpoints=None, **kwargs):
        self.name_classes = name_classes
        super().__init__(**kwargs)
        from progress.bar import Bar
        if num_viewpoints is not None:
            get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
            selected_vp_samples = []
            index_next_object = 0
            bar = Bar(f"Selecting {num_viewpoints} viewpoints", max=len(self.samples))
            while True:
                # find next index:
                obj_num = get_obj_num(self.samples[index_next_object][0])
                j = index_next_object
                while j < len(self.samples) and get_obj_num(self.samples[j][0]) == obj_num:
                    j += 1
                itmp = np.random.choice(j-index_next_object, np.min((num_viewpoints, j-index_next_object)))[0]
                selected_vp_samples.extend([self.samples[index_next_object:j][itmp]])
                bar.next(n=j-index_next_object)
                index_next_object = j

                if j >= len(self.samples):
                    break

            all_objects = np.unique([get_obj_num(i[0]) for i in self.samples])
            print(f"Num Objects: {len(all_objects)}, num tot samples: {len(self.samples)}, num selected vp samples: {len(selected_vp_samples)}")
            self.samples = selected_vp_samples

    def _find_classes(self, dir: str):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes if self.name_classes is not None else True)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SubclassImageFolder(ImageFolderSelective):
    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.subclasses = {}
        self.subclass_to_idx = {}
        self.subclasses_names = []
        self.subclasses_to_classes = {}
        subclass_idx = 0
        for idx, i in enumerate(self.samples):
            subclass_name = self.get_subclass_from_sample(i)

            if subclass_name in self.subclasses:
                self.subclasses[subclass_name].append(idx)
            else:
                self.subclasses_names.append(subclass_name)
                self.subclasses[subclass_name] = [idx]
                self.subclasses_to_classes[subclass_name] = i[1]

                subclass_idx += 1

        self.subclass_to_idx = {k: idx for idx, k in enumerate(self.subclasses_names)}
        self.samples_sb = [(a[0], a[1], self.subclass_to_idx[self.get_subclass_from_sample(a)]) for a in self.samples]
        self.sampler = sampler(subclasses=self.subclasses, subclasses_names=self.subclasses_names, dataset=self)
        print(f"Subclasses folder dataset. Num classes: {len(self.classes)}, num subclasses: {len(self)}")  #name: {self.subclasses_names}") # nope! Too many!

    def __len__(self):
        return len(self.subclasses_names)

    def __getitem__(self, index):
        path, class_idx, object_idx = self.samples_sb[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, class_idx, object_idx

    def get_subclass_from_sample(self, sample):
        get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
        name_class = self.idx_to_class[sample[1]]
        return name_class + '_' + str(get_obj_num(sample[0]))
        # name_class = self.idx_to_classes[sample[1]]
        # return name_class + '_' + os.path.split(os.path.split(sample[0])[0])[1]

from torch.utils.data.sampler import WeightedRandomSampler
def get_weighted_sampler(dataset):
    weights_class = 1 / np.array([np.sum([True for k, v in dataset.subclasses_to_classes.items() if v == c]) for c in range(len(dataset.classes))])
    weights_dataset = [weights_class[dataset.subclasses_to_classes[i]] for i in dataset.subclasses_names]
    return WeightedRandomSampler(weights=weights_dataset, num_samples=len(dataset.subclasses_names), replacement=True)


class KBatchSampler(Sampler):
    def __init__(self, batch_size, dataset, subclasses, subclasses_names, prob_same=0.5, rebalance_classes=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subclasses = subclasses
        self.subclasses_names = subclasses_names
        self.prob_same = prob_same
        # self.subclasses_list = copy.deepcopy(self.subclasses_names)
        if rebalance_classes:
            self.sampler_training = get_weighted_sampler(dataset)
            self.sampler_candidate = get_weighted_sampler(dataset)
        else:
            self.sampler_training = RandomSampler(self.subclasses_names)
            self.sampler_candidate = RandomSampler(self.subclasses_names)
        # else:
        #     self.sampler = SequentialSampler(self.subclasses_names)

    def __iter__(self):
        training_idx = []
        candidate_idx = []
        i = iter(self.sampler_candidate)
        for idx in self.sampler_training:
            sel_obj = self.subclasses_names[idx]
            training_idx.append(np.random.choice(self.subclasses[sel_obj], 1)[0])
            if np.random.rand() < self.prob_same:
                candidate_idx.append(np.random.choice(self.subclasses[sel_obj], 1)[0])
            else:
                candidate_idx.append(np.random.choice(self.subclasses[self.subclasses_names[next(i)]], 1)[0])
                # candidate_idx.append(np.random.choice(self.subclasses[np.random.choice(list(set(self.subclasses_names) - set([sel_obj])), 1)[0]], 1)[0])
            if len(candidate_idx) == self.batch_size:
                yield np.hstack((candidate_idx, training_idx))
                candidate_idx = []
                training_idx = []


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        ## This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_generator, additional_transform=None, num_image_calculate_mean_std=70, grayscale=False, **kwargs):
            print(f"\n**Creating Dataset {name_generator}**")
            super().__init__(**kwargs)

            if additional_transform is None:
                additional_transform = []
            if grayscale and torchvision.transforms.Grayscale() not in additional_transform:
                print("Warning! Compute stats is in grayscale mode but no grayscale transform is used. This may be ok with some datasets (such as unity)")
            self.transform = torchvision.transforms.Compose([*additional_transform, torchvision.transforms.ToTensor()])

            self.name_generator = name_generator
            self.additional_transform = additional_transform
            self.grayscale = grayscale
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)
            self.name_classes = self.classes

            self.stats = self.call_compute_stats()
            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])
            # self.stats = {}
            # self.stats['mean'] = [0.491, 0.482, 0.44]
            # self.stats['std'] = [0.247, 0.243, 0.262]
            # normalize = torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            self.transform.transforms += [normalize]
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            print(f'Map class_name -> labels: \n {self.class_to_idx}\n{len(self)} samples.')

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, grayscale=self.grayscale)

        def __getitem__(self, idx, class_name=None):
            image, *rest = super().__getitem__(idx)
            label = rest[0]
            return image, label, rest[1] if len(rest) > 1 else 1

    return ComputeStatsUpdateTransform


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