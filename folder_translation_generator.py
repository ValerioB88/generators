from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
from multiprocessing.dummy import freeze_support
from torch.utils.data import Sampler
import torch
from sty import fg, bg, ef, rs
from torch.utils.data import SequentialSampler, RandomSampler
import framework_utils
import glob
import PIL.Image as Image
import copy
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.translate_generator import TranslateGenerator, InputImagesGenerator
import pathlib
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
import torchvision
import os
from torchvision.datasets import ImageFolder
import re
import pickle
from pathlib import Path


class MyImageFolder(ImageFolder):
    def finalize_getitem(self, path, sample, labels, info=None):
        if info is None:
            info = {}
        return sample, labels, info

    def __init__(self, name_classes=None, *args, **kwargs):
        self.name_classes = name_classes
        super().__init__(*args, **kwargs)

    def _find_classes(self, dir: str):
        if self.name_classes is None:
            return super()._find_classes(dir)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes)]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = self.finalize_getitem(path=path, sample=sample, labels=target)
        return output


class SubsetImageFolder(MyImageFolder):
    def __init__(self, num_images_per_class=None, save_load_samples_filename=None, **kwargs):
        super().__init__(**kwargs)
        loaded = False
        if num_images_per_class:
            if save_load_samples_filename is not None:
                if os.path.isfile(save_load_samples_filename):
                    print(fg.red + f"LOADING SAMPLES FROM /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                    self.samples = pickle.load(open(save_load_samples_filename, 'rb'))
                    loaded = True
                else:
                    print(fg.yellow + f"Path {save_load_samples_filename} not found, will compute samples" + rs.fg)

            if not loaded:
                subset_samples = []
                for c in range(len(self.classes)):
                    objs_class = [i for i in self.samples if i[1] == c]
                    subset_samples.extend([objs_class[i] for i in np.random.choice(len(objs_class), num_images_per_class, replace=False)])
                self.samples = subset_samples

            if save_load_samples_filename is not None and loaded is False:
                print(fg.yellow + f"SAVING SAMPLES IN /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                pathlib.Path(Path(save_load_samples_filename).parent).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.samples, open(save_load_samples_filename, 'wb'))


class SelectObjects(MyImageFolder):
    """
    This is used for the objects in ShapeNet
    """
    def __init__(self, name_classes=None, num_objects_per_class=None, selected_objects=None, num_viewpoints_per_object=None, take_specific_azi_incl=True, save_load_samples_filename=None, **kwargs):
        self.name_classes = name_classes
        # take_specific_azi_incl = False  take random viewpoints
        #          a                 True  take one specific viewpoint: 75, 36
        #                           (x, y) take one specific viewpoint: (x,y)
        # only work if num_viewpoints_per_object is 1, otherwise changed to False.
        if num_viewpoints_per_object == 1 and take_specific_azi_incl is True:
            take_specific_azi_incl = (75, 36)
        super().__init__(name_classes, **kwargs)
        self.selected_objects = selected_objects
        get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
        original_sample_size = len(self.samples)
        loaded = False
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if save_load_samples_filename is not None:
            if os.path.isfile(save_load_samples_filename):
                print(fg.red + f"LOADING SAMPLES FROM /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                if num_objects_per_class is not None or selected_objects is not None or num_viewpoints_per_object is not None:
                    print("Max num objects, Num Viewpoints and specific object to select will be ignored")
                self.samples = pickle.load(open(save_load_samples_filename, 'rb'))
                self.selected_objects = np.hstack([np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == c]) for c in range(len(self.classes))])
                loaded = True
            else:
                print(fg.yellow + f"Path {save_load_samples_filename} not found, will compute samples" + rs.fg)

        if not loaded:
            from progress.bar import Bar
            if num_objects_per_class is not None or self.selected_objects is not None:
                print("SELECTING OBJECTS")

                all_objs_count = ([np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == c]) for c in range(len(self.classes))])
                if self.selected_objects is None:
                    self.selected_objects = np.hstack([np.random.choice(o, np.min((num_objects_per_class, len(o))), replace=False) for o in all_objs_count])
                self.samples = [i for i in self.samples if get_subclass_from_sample(self.idx_to_class, i) in self.selected_objects]

            if num_viewpoints_per_object is not None:
                print("SELECTING VIEWPOINTS")
                selected_vp_samples = []
                index_next_object = 0
                bar = Bar(f"Selecting {num_viewpoints_per_object} viewpoints", max=len(self.samples))

                if take_specific_azi_incl and num_viewpoints_per_object > 1:
                    print(fg.yellow + "Num_viewpoints_per_object > 1 and take_specific_azi_incl is True. Take_specific_azi_incl changed to false (random vp)" + rs.fg)

                while True:
                    # find next index:
                    obj_num = get_obj_num(self.samples[index_next_object][0])
                    j = index_next_object
                    while j < len(self.samples) and get_obj_num(self.samples[j][0]) == obj_num:
                        j += 1

                    if num_viewpoints_per_object == 1 and take_specific_azi_incl:
                        selected_sample = [i for i in self.samples[index_next_object:j] if re.search(rf'O\d+_I{take_specific_azi_incl[0]}_A{take_specific_azi_incl[1]}', i[0])]
                        assert len(selected_sample) <= 1
                        if not selected_sample:
                            #if not found
                            incl_azi = [re.search(rf'O\d+_I(\d+)_A(\d+)', i[0]) for i in self.samples[index_next_object:j]]
                            incl_azi = [(int(i.groups()[0]), int(i.groups()[1])) for i in incl_azi]
                            selected_index = np.argmin(np.mean(np.abs(np.array(incl_azi)-take_specific_azi_incl), 1))
                            selected_incl_azi = incl_azi[selected_index]
                            print(fg.red + f"Viewpoint {take_specific_azi_incl} not found, using {selected_incl_azi}" + rs.fg)
                            selected_sample = self.samples[index_next_object:j][selected_index]
                        else:
                            selected_sample = selected_sample[0]

                        selected_vp_samples.append(selected_sample)

                    else:
                        itmp = np.random.choice(j - index_next_object, np.min((num_viewpoints_per_object, j - index_next_object)), replace=False)
                        selected_vp_samples.extend([self.samples[index_next_object:j][i] for i in itmp])

                    bar.next(n=j-index_next_object)
                    index_next_object = j

                    if j >= len(self.samples):
                        break

                self.samples = selected_vp_samples

        # all_objects = np.unique([get_obj_num(i[0]) for i in self.samples])
        print(f"\nNum ALL samples: {original_sample_size}, Num Objects: {len(self.selected_objects) if self.selected_objects is not None else 'all'},  Num selected vp: {num_viewpoints_per_object}, \nFinal Num Samples = num objs x num vp: {len(self.samples)}")
        all_objs_count = {self.idx_to_class[j]: len(np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == j])) for j in range(len(self.classes))}
        [print('{:25}: {:4}'.format(k, i)) for k, i in all_objs_count.items()]
        tmplst = [np.min((i, len(self.samples)-1)) for i in [0, 1, 3, 5, 100, 2000, -1]]
        print(fg.cyan + f"Samples in indexes {tmplst}:\t" + rs.fg, end="")
        [print(rf'{i[1]}: {get_subclass_with_azi_incl(self.idx_to_class, i[0])}', end="\t") for i in [(self.samples[s], s) for s in tmplst]]
        if self.selected_objects is not None:
            tmplst = [np.min((i, len(self.selected_objects)-1)) for i in [0, 1, 3, 5, 100, 2000, -1]]
            print(fg.cyan + f"\nObjects in indexes {tmplst}:\t" + rs.fg, end="")
            [print(f'{i[1]}: {i[0]}', end="\t") for i in [(self.selected_objects[s], s) for s in tmplst]]

        print()

        if save_load_samples_filename is not None and loaded is False:
            print(fg.yellow + f"SAVING SAMPLES IN /samples/{Path(save_load_samples_filename).name}" + rs.fg)
            pathlib.Path(Path(save_load_samples_filename).parent).mkdir(parents=True, exist_ok=True)
            pickle.dump(self.samples, open(save_load_samples_filename, 'wb'))

    def _find_classes(self, dir: str):
        #re.findall(r'[a-zA-Z]+_?[a-zA-Z]+.n.\d+', self.name_classes) if self.name_classes is not None else None
        classes_to_take = list(self.name_classes.keys()) if self.name_classes is not None else None
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in classes_to_take if self.name_classes is not None else True)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SubclassImageFolder(SelectObjects):
    def __init__(self, sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.subclasses = {}
        self.subclass_to_idx = {}
        self.classes_to_subclasses = {}
        self.subclasses_names = []
        self.subclasses_to_classes = {}
        subclass_idx = 0
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for idx, i in enumerate(self.samples):
            subclass_name = get_subclass_from_sample(self.idx_to_class, i)

            if subclass_name in self.subclasses:
                self.subclasses[subclass_name].append(idx)
            else:
                self.subclasses_names.append(subclass_name)
                self.subclasses[subclass_name] = [idx]
                self.subclasses_to_classes[subclass_name] = i[1]
                if self.idx_to_class[i[1]] not in self.classes_to_subclasses:
                    self.classes_to_subclasses[self.idx_to_class[i[1]]] = [subclass_name]
                else:
                    self.classes_to_subclasses[self.idx_to_class[i[1]]].append(subclass_name)

                subclass_idx += 1
        print('Length classes:')
        # [print('{:20}: {:4}'.format(k, len(i))) for k, i in self.classes_to_subclasses.items()]
        self.subclass_to_idx = {k: idx for idx, k in enumerate(self.subclasses_names)}
        self.idx_to_subclass = {v: k for k, v in self.subclass_to_idx.items()}
        self.samples_sb = [(a[0], a[1], self.subclass_to_idx[get_subclass_from_sample(self.idx_to_class, a)]) for a in self.samples]

        # if sampler is not None:
        #     # Same-Different sampler
        #     self.sampler = sampler(subclasses=self.subclasses, subclasses_names=self.subclasses_names, dataset=self)
        print(f"Subclasses folder dataset. Num classes: {len(self.classes)}, num subclasses: {len(self)}")  #name: {self.subclasses_names}") # nope! Too many!

    # def __len__(self):
    #     return len(self.subclasses_names)

    def __getitem__(self, index):
        path, class_idx, object_idx = self.samples_sb[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample, class_idx, info = self.finalize_getitem(path=path, sample=sample, labels=class_idx, info={'label_object': object_idx})
        return sample, class_idx, info

def get_subclass_from_sample(idx_to_class, sample):
    get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
    name_class = idx_to_class[sample[1]]
    return name_class + '_' + str(get_obj_num(sample[0]))
    # name_class = self.idx_to_classes[sample[1]]
    # return name_class + '_' + os.path.split(os.path.split(sample[0])[0])[1]

def get_subclass_with_azi_incl(idx_to_class, sample):
    get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
    name_class = idx_to_class[sample[1]]
    incl_azi = re.search('O\d+_I(\d+)_A(\d+).png', sample[0]).groups()
    return name_class + '_' + str(get_obj_num(sample[0])) + f'_I{incl_azi[0]}_A{incl_azi[1]}'

from torch.utils.data.sampler import WeightedRandomSampler



class SameDifferentSampler(Sampler):
    def get_subclasses_weighted_sampler(self, dataset):
        # print("QUI: ")
        # print(np.array([np.sum([True for k, v in dataset.subclasses_to_classes.items() if v == c]) for c in range(len(dataset.classes))]))
        weights_class = 1 / np.array([np.sum([True for k, v in dataset.subclasses_to_classes.items() if v == c]) for c in range(len(dataset.classes))])
        weights_dataset = [weights_class[dataset.subclasses_to_classes[i]] for i in dataset.subclasses_names]
        return WeightedRandomSampler(weights=weights_dataset, num_samples=len(dataset.subclasses_names), replacement=True)

    def __init__(self, batch_size, dataset, subclasses, subclasses_names, prob_same=0.5, rebalance_classes=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subclasses = subclasses
        self.subclasses_names = subclasses_names
        self.prob_same = prob_same
        # self.subclasses_list = copy.deepcopy(self.subclasses_names)
        if rebalance_classes:
            self.sampler_training = self.get_subclasses_weighted_sampler(dataset)
            self.sampler_candidate = self.get_subclasses_weighted_sampler(dataset)
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
        yield np.hstack((candidate_idx, training_idx))

def add_return_path(class_obj):
    class ImageFolderWithPath(class_obj):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            path = self.samples[index][0]
            return img, target, {'path': path}
    return ImageFolderWithPath


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        ## This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_generator, add_PIL_transforms=None, add_tensor_transforms=None, num_image_calculate_mean_std=70, stats=None, save_stats_file=None, **kwargs):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u], a path to a pickle file, or None
            @param save_stats_file:
            @param kwargs:
            """
            print(fg.yellow + f"\n**Creating Dataset [" + fg.cyan + f"{name_generator}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_generator = name_generator
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)
            self.name_classes = self.classes

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + f"Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, 'rb'))
                    print(fg.red + f"Using stats from file [{Path(stats).name}]: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, 'wb'))

            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])
            # self.stats = {}
            # self.stats['mean'] = [0.491, 0.482, 0.44]
            # self.stats['std'] = [0.247, 0.243, 0.262]
            # normalize = torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            self.transform.transforms += [normalize]

            print(f'Map class_name -> labels: \n {self.class_to_idx}\n{len(self)} samples.')
            self.finalize_init()

        def finalize_init(self):
            pass

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std)

        def __getitem__(self, idx, class_name=None):
            image, *rest = super().__getitem__(idx)
            label = rest[0]
            return image, label, rest[1] if len(rest) > 1 else {}

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
