
from mlagents_envs.environment import UnityEnvironment, BehaviorName, DecisionSteps
import mlagents_envs.logging_util as log

from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Sampler
import warnings
import PIL.Image as Image
from generate_datasets.generators.input_image_generator import InputImagesGenerator
from typing import Tuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import os
import atexit

class UnityGenMetaLearning(InputImagesGenerator):
    def __init__(self, name_dataset_unity=None, sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler(name_dataset_unity=name_dataset_unity, dataset=self)
        super()._finalize_init()
        # atexit.register(self.sampler.env.close())

    def _finalize_init(self):
        if self.num_classes < self.sampler.k:
            warnings.warn(f'FolderGen: Number of classes in the folder < k. K will be changed to {self.num_classes}')
            self.sampler.k = self.num_classes

        super()._finalize_init()

    def call_compute_stat(self, filename):
        return compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename),
                                                 data_loader=DataLoader(self, batch_sampler=self.sampler, num_workers=0),
                                                 max_iteration=self.max_iteration_mean_std, verbose=self.verbose)

    def _get_label(self, idx):
        return str(idx[1])

    def _define_num_classes_(self):
        return 24  # ToDo: make this something we would get from a unity channel

    def _resize(self, image):
        if self.size_canvas is not None:
            image = image.resize(self.size_canvas)
        return image, (self.size_canvas if self.size_canvas is not None else -1)

    def _get_image(self, idx, label):
        # we use idx here because it's taken into account by the sampler
        image_name = ""
        # image_name = np.random.choice(self.samples[label])
        image = idx[0]
        return Image.fromarray((image * 255).astype(np.uint8)), image_name


class UnitySampler(Sampler):
    """
    This sampler takes the samples from the UniyScene
    """
    warning_done = False

    def __init__(self, dataset, name_dataset_unity, n, k, q, num_tasks=1, episodes_per_epoch=999999, channel=0, disjoint_train_and_test=True):
        # super(NShotTaskSampler, self).__init__(unity_env)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # may be useful later on

        # TODO: implement multi tasks
        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        build_win = 0
        if os.name == 'nt':
            machine_name = 'win'
            ext = 'exe'
        else:
            machine_name = 'linux'
            ext = 'x86_64'
        self.scene_path = f'./Unity-ML-Agents-Computer-Vision/Builds/Builds_{machine_name}/MetaLearning/'
        self.scene_folder = f'N{n}_K{k}_Q{q}_sc{dataset.size_canvas[0]}_d{name_dataset_unity}'
        self.scene = f'{self.scene_path}/{self.scene_folder}/scene.{ext}'
        if not os.path.exists(self.scene):
            assert False, f"Unity scene {self.scene} generator does not exist, create it in windows!"
        log.set_log_level('INFO')

        side_channel = EnvironmentParametersChannel()

        self.env = UnityEnvironment(file_name=self.scene, side_channels=[side_channel], no_graphics=False, worker_id=channel)
        # channel.set_float_parameter("N_shot", self.n)

        self.env.reset()
        self.behaviour_names = list(self.env.behavior_specs.keys())
        # self.behaviour_specs = self.env.behavior_specs[self.behaviour_names[0]]

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []  # batch is [image, label, train/test (0/1)]
            # ToDo: add each batch in a list and then return a stack of it
            for task in range(self.num_tasks):
                # Get random classes
                self.env.step()
                DS, TS = self.env.get_steps(self.behaviour_names[0])
                # when agent receives an action, it setups a new batch
                self.env.set_actions(self.behaviour_names[0], np.array([[1]]))  # just give a random thing as an action, it doesn't matter here
                labels = DS.obs[-1].astype(int)
                images = DS.obs[:-1]
                train_test = [0 for i in range(self.n * self.k)]
                train_test.extend([1 for i in range(self.q * self.k)])
                batch = list((im[0], lab, tr) for im, lab, tr in zip(images, labels[0], train_test))
            yield batch

