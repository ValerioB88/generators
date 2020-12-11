
from mlagents_envs.environment import UnityEnvironment, BehaviorName, DecisionSteps
import mlagents_envs.logging_util as log
from mlagents_envs.exception import UnityWorkerInUseException
import framework_utils
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from torch.utils.data import DataLoader

from torch.utils.data import Sampler
import warnings
import PIL.Image as Image
from generate_datasets.generators.input_image_generator import InputImagesGenerator
from typing import Tuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import os

from mlagents_envs.environment import UnityEnvironment

import numpy as np
from enum import Enum
from experiments.novel_objects_recognition.SequenceMetaLearning.unity_channels import StringDebugLogChannel, StringLogChannel, StringEnvParamsChannel


class UnityGenMetaLearning(InputImagesGenerator):
    def __init__(self, name_dataset_unity, unity_env_params=None, sampler=None, **kwargs):
        self.sampler = sampler(dataset=self, unity_env_params=unity_env_params, name_dataset_unity=name_dataset_unity)
        super().__init__(**kwargs)
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

    # def _finalize_get_item(self, canvas, class_name, more, idx):
    #     # more['camera_positions'] = idx[2]
    #     # more['labels'] = idx[1]
    #     return canvas, class_name, more

    # def _get_label(self, idx):
    #     # return str(idx[1])
    #     return '0'

    def _define_num_classes_(self):
        return self.sampler.num_objects  # ToDo: make this something we would get from a unity channel

    def _resize(self, image):
        if self.size_canvas is not None:
            image = image.resize(self.size_canvas)
        return image, (self.size_canvas if self.size_canvas is not None else -1)

    def _get_image(self, idx, label):
        # we use idx here because it's taken into account by the sampler
        # image_name = ""
        # image_name = np.random.choice(self.samples[label])
        image = idx
        return Image.fromarray((image * 255).astype(np.uint8)), None

    def __getitem__(self, idx):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        # self._prepare_get_item()
        # class_name = self._get_label(idx)

        # _get_image returns a PIL image
        image, image_name = self._get_image(idx, None)
        image, new_size = self._resize(image)
        # image, rotate = self._rotate(image)

        # canvas, random_center = self._transpose(image, image_name, class_name, idx)
        # more = {'center': random_center, 'size': new_size, 'rotation': rotate}
        # get my item must return a PIL image
        # canvas, class_name, more = self._finalize_get_item(canvas=image, class_name=None, more=None, idx=idx)

        if self.transform is not None:
            canvas = self.transform(image)
        else:
            canvas = np.array(image)

        return canvas, 1, 1 #, self.map_name_to_num[class_name], more

class TestType(Enum):
    INTERPOLATE = 0
    EXTRAPOLATE = 1
    ORTHOGONAL = 3

class UnitySamplerSequenceLearning(Sampler):
    """
    This sampler takes the samples from the UniyScene
    """
    warning_done = False

    def __init__(self, dataset, name_dataset_unity, unity_env_params, nSc, nSt, nFc, nFt, k, size_canvas, episodes_per_epoch=999999, channel=0, test_mode: TestType =None):
        """
        @param k: number of objects used .the number of matching pairs will depends on the test flag. If it's in testing mode,
         we do one matching pair for each k (only implemented for k=1 for now). Which means that total matching pair = k, total number of labels passed is k * 2.
        @param test_mode: For now @test_mode is a special flag that does several stuff useful for testing:
         1) the camera positions are passed by python, not generated in unity.
         2) k = number of matching comparisons, e.g. with k=1 we spawn two objects (may be the same) and compare between them.
            2) implies a different setup for passing the labels. k * 2 labels are passed.
        TODO: In the future we should test if this can be the default setup in terms of speed of computation, as it is much cleared.
        If test = 0 we make k * k comparison: each object in the scene is compared with every other.
        """
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # may be useful later on
        self.test_mode = test_mode  # None, or TestType
        self.k = k

        # TODO: Raise errors if initialise badly
        if self.test_mode and self.k != 1:
            print(f"TestMode only supported for k=1. K was = {self.k}, it will be changed to 1")
            self.k = 1
        self.nSc = nSc
        self.nSt = nSt
        self.nFc = nFc
        self.nFt = nFt
        if name_dataset_unity is None:
            self.scene = None
            channel = 0
        else:
            if os.name == 'nt':
                machine_name = 'win'
                ext = 'exe'
            else:
                machine_name = 'linux'
                ext = 'x86_64'
            self.scene_path = f'./Unity-ML-Agents-Computer-Vision/Builds/Builds_{machine_name}/SequenceLearning{"_testMode" if test_mode else ""}/'
            self.scene_folder = f'k{self.k}_nSt{self.nSt}_nSc{self.nSc}_nFt{self.nFt}_nFc{self.nFc}_sc{size_canvas[0]}'
            self.scene = f'{self.scene_path}/{self.scene_folder}/scene.{ext}'
            if not os.path.exists(self.scene):
                assert False, f"Unity scene {self.scene} generator does not exist, create it in windows!"

        log.set_log_level('INFO')

        self.observation_channel = EnvironmentParametersChannel()
        while True:
            try:
                env_params_channel = StringEnvParamsChannel("621f0a70-4f87-11ea-a6bf-784f4387d1f7")
                debug_channel = StringDebugLogChannel("8e8d2cbd-ea04-444d-9180-56ed79a2b94e")
                self.env = UnityEnvironment(file_name=self.scene, side_channels=[self.observation_channel, env_params_channel, debug_channel], no_graphics=False, worker_id=channel, additional_args=['-name_dataset', name_dataset_unity])
                break
            except UnityWorkerInUseException as e:
                channel += 1

        self.env_params = unity_env_params(self.observation_channel)
        self.env.reset()
        self.observation_channel.set_float_parameter("newLevel", float(0))

        self.behaviour_names = list(self.env.behavior_specs.keys())
        self.num_objects = env_params_channel.num_objects
        self.labels = None
        self.camera_positions = None
        # _, self.ax = framework_utils.create_sphere()
        if self.test_mode:
            self.ranges = []
            self.matrix_values = []
            self.organize_test_values()

        self.tot_num_frames_each_iter = self.k * (self.nSt * self.nFt + self.nSc * self.nFc)
        self.tot_num_matching = self.k
        self.num_labels_passed_by_unity = self.k * 2
        self.tot_num_frames_each_comparison =  int(self.tot_num_frames_each_iter / self.tot_num_matching)

        self.max_length_elements = np.max((self.tot_num_matching, self.tot_num_frames_each_iter))
        self.dummy_labels = np.empty(self.tot_num_frames_each_iter)

    def organize_test_values(self):
        self.range = []
        self.num_repetitions_each = 20;
        self.observation_channel.set_float_parameter("test_type", float(self.test_mode.value))

        if self.test_mode == TestType.INTERPOLATE:
            self.ranges = np.arange(0, -90 - 15, -15)

        if self.test_mode == TestType.EXTRAPOLATE or self.test_mode == TestType.ORTHOGONAL:
            self.ranges = np.arange(0, 90 + 15, 15)

        obj1, obj2, deg, rot = np.meshgrid(np.arange(self.num_objects), np.arange(self.num_objects), self.ranges, np.arange(0, 360, 15))
        self.matrix_values = np.vstack((obj1.flatten(), obj2.flatten(), deg.flatten(), rot.flatten()))
        # np.random.shuffle(self.matrix_values.T)
        self.episodes_per_epoch = self.matrix_values.shape[1]


    def __len__(self):
        return self.episodes_per_epoch

    def send_test_info(self, i):
        self.observation_channel.set_float_parameter("objT", float(self.matrix_values[0, i]))
        self.observation_channel.set_float_parameter("objC", float(self.matrix_values[1, i]))
        self.observation_channel.set_float_parameter("degree", float(self.matrix_values[2, i]))
        self.observation_channel.set_float_parameter("rotation", float(self.matrix_values[3, i]))


    def __iter__(self):
        vh1, vh2, vh3 = [], [], []
        for idx in range(self.episodes_per_epoch):
            if self.test_mode is not None:
                self.send_test_info(idx)
            # remember that the images are passed in alphabetical order (which is C0, C1, T0, T1 ..), whereas the camera positions are passed in a more convenient format:
            # organizes in number of matching (k), and inside that all the Cs, then all the Ts
            # labels is a list of size k with [C, T]
            self.env.step()
            self.observation_channel.set_float_parameter("newLevel", float(0))

            DS, TS = self.env.get_steps(self.behaviour_names[0])

            # when agent receives an action, it setups a new batch
            self.env.set_actions(self.behaviour_names[0], np.array([[1]]))  # just give a random thing as an action, it doesn't matter here

            labels = DS.obs[-1][0][:self.num_labels_passed_by_unity].astype(int).reshape((-1, 2))
            camera_positions = DS.obs[-1][0][self.num_labels_passed_by_unity:].reshape((self.tot_num_matching, self.tot_num_frames_each_comparison, 3))
            images = [i[0] for i in DS.obs[:-1]]   # .reshape((self.tot_num_frames_each_iter, self.tot_num_frames_each_comparison, 64, 64, 3))

#################################~~~~~~DEBUG~~~~~~###############################################
            # plt.show()
            # import copy
            # def unity2python(v):
            #     v = copy.deepcopy(v)
            #     v.T[[1, 2]] = v.T[[2, 1]]
            #     return v
            # for idx, c in enumerate(camera_positions):
            #     if vh1:
            #         [i.remove() for i in vh1]
            #         [i.remove() for i in vh2]
            #         vh1 = []
            #         vh2 = []
            #     vh1.append(framework_utils.add_norm_vector(unity2python(c[0]), 'k', ax=self.ax))
            #     for i in range(len(camera_positions[0]) - 1):
            #         vh2.append(framework_utils.add_norm_vector(unity2python(c[i + 1]), 'r', ax=self.ax))
                    ## ali = framework_utils.align_vectors(c, t)
                    ## vh3 = framework_utils.add_norm_vector(ali, 'r', ax=self.ax )
#################################################################################################

            self.labels = labels
            self.camera_positions = camera_positions
            batch = images[:]
            # batch = list((im, lb, cp) for im, lb, cp in zip(images, labels, camera_positions))
            yield batch

# # ##
# #
# vh1.remove()
# vh2.remove()
# # vh3.remove()

#batch # # ##
#

##

