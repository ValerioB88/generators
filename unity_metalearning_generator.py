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
from copy import deepcopy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class UnityGenMetaLearning(InputImagesGenerator):
    def __init__(self, name_dataset_unity, unity_env_params=None, sampler=None, **kwargs):
        self.sampler = sampler(dataset=self, unity_env_params=unity_env_params, name_dataset_unity=name_dataset_unity, grayscale=kwargs['grayscale'])
        super().__init__(**kwargs, convert_to_grayscale=False)
        super()._finalize_init()
        # atexit.register(self.sampler.env.close())

    def _finalize_init(self):
        if self.num_classes < self.sampler.k:
            warnings.warn(f'FolderGen: Number of classes in the folder < k. K will be changed to {self.num_classes}')
            self.sampler.k = self.num_classes

        super()._finalize_init()

    def call_compute_stat(self, filename):
        if self.sampler.matrix_values != []:
            m_copy = deepcopy(self.sampler.matrix_values)
            np.random.shuffle(self.sampler.matrix_values.T)

        stats = compute_mean_and_std_from_dataset(self, './data/generators/stats_{}'.format(filename),
                                                  data_loader=DataLoader(self, batch_sampler=self.sampler, num_workers=0),
                                                  grayscale=self.grayscale,
                                                  max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)
        if self.sampler.matrix_values != []:
            self.sampler.matrix_values = m_copy
        return stats

    def _define_num_classes_(self):
        return self.sampler.num_objects

    def _resize(self, image):
        if self.size_canvas is not None:
            image = image.resize(self.size_canvas)
        return image, (self.size_canvas if self.size_canvas is not None else -1)

    def _get_image(self, idx, label):
        image = idx.squeeze() if self.grayscale else idx
        return Image.fromarray((image * 255).astype(np.uint8)), None

    def __getitem__(self, idx):
        """
        @param idx: this is only used with 'Fixed' generators. Otherwise it doesn't mean anything to use an index for an infinite generator.
        @return:
        """
        image, image_name = self._get_image(idx, None)
        image, new_size = self._resize(image)

        if self.transform is not None:
            canvas = self.transform(image)
        else:
            canvas = np.array(image)

        return canvas, 1, 1 #, self.map_name_to_num[class_name], more

    def __len__(self):
        return len(self.sampler)

class TrialType(Enum):
    RND_TRIAL = 0
    DET_TRIAL_EDELMAN_SAME_AXIS_HOR = 1
    DET_TRIAL_EDELMAN_ORTHO_HOR = 2
    DET_TRIAL_EDELMAN_SAME_AXIS_VER = 3
    DET_TRIAL_EDELMAN_ORTHO_VER = 4
    DET_TRIAL_STATIC = 5

class TrainComparisonType(Enum):
    ALL = 0
    GROUP = 1

class UnitySamplerSequenceLearning(ABC, Sampler):
    """
    This sampler takes the samples from the UniyScene
    """
    warning_done = False
    def __init__(self, dataset, name_dataset_unity, unity_env_params, nSc, nSt, nFc, nFt, k, size_canvas, grayscale, episodes_per_epoch=999999, channel=0):
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # may be useful later on
        self.k = k

        if self.trial_type != TrialType.RND_TRIAL and self.k != 1:
            print(f"Deterministic Mode only supported for k=1. K was = {self.k}, it will be changed to 1")
            self.k = 1
        self.nSc = nSc
        self.nSt = nSt
        self.nFc = nFc
        self.grayscale = grayscale
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
            self.scene_path = f'./Unity-ML-Agents-Computer-Vision/Builds/Builds_{machine_name}/SequenceLearning/'
            self.scene_folder = f'k{self.k}_nSt{self.nSt}_nSc{self.nSc}_nFt{self.nFt}_nFc{self.nFc}_sc{size_canvas[0]}_g{int(self.grayscale)}'
            self.scene = f'{self.scene_path}/{self.scene_folder}/scene.{ext}'
            if not os.path.exists(self.scene):
                assert False, f"Unity scene {self.scene} generator does not exist, create it in windows!"

        log.set_log_level('INFO')

        self.observation_channel = EnvironmentParametersChannel()
        while True:
            try:
                env_params_channel = StringEnvParamsChannel("621f0a70-4f87-11ea-a6bf-784f4387d1f7")
                debug_channel = StringDebugLogChannel("8e8d2cbd-ea04-444d-9180-56ed79a2b94e")
                print(f"\n*** Trying to open Unity scene with dataset: {name_dataset_unity}")
                self.env = UnityEnvironment(file_name=self.scene, side_channels=[self.observation_channel, env_params_channel, debug_channel], no_graphics=False, worker_id=channel, additional_args=['-name_dataset', name_dataset_unity])
                break
            except UnityWorkerInUseException as e:
                channel += 1

        self.env_params = unity_env_params(self.observation_channel)
        self.observation_channel.set_float_parameter("trialType", float(self.trial_type.value))
        # this is set as default, it may be changed in "send additional exp info" next.
        self.observation_channel.set_float_parameter("trainComparisonType", float(TrainComparisonType.ALL.value))

        self.send_additional_experiment_info()
        self.env.reset()
        self.observation_channel.set_float_parameter("newLevel", float(0))

        self.behaviour_names = list(self.env.behavior_specs.keys())
        self.num_objects = env_params_channel.num_objects
        self.labels = None
        self.camera_positions = None
        self.matrix_values = []

        self.tot_num_frames_each_iter = self.k * (self.nSt * self.nFt + self.nSc * self.nFc)
        self.tot_num_matching = self.k
        self.num_labels_passed_by_unity = self.k * 2
        self.tot_num_frames_each_comparison = int(self.tot_num_frames_each_iter / self.tot_num_matching)

        self.max_length_elements = np.max((self.tot_num_matching, self.tot_num_frames_each_iter))
        self.dummy_labels = np.empty(self.tot_num_frames_each_iter)
        self.images = []

    def __len__(self):
        return self.episodes_per_epoch

    def send_additional_experiment_info(self):
        pass

    def send_episode_info(self, idx):
        pass

    def __iter__(self):
        # vh1, vh2, vh3 = [], [], []
        for idx in range(self.episodes_per_epoch):
            self.send_episode_info(idx)
            # remember that the images are passed in alphabetical order (which is C0, C1, T0, T1 ..), whereas the camera positions are passed in a more convenient format:
            # organizes in number of matching (k), and inside that all the Cs, then all the Ts
            # labels is a list of size k with [[C, T] *k]
            self.env.step()
            self.observation_channel.set_float_parameter("newLevel", float(0))

            DS, TS = self.env.get_steps(self.behaviour_names[0])

            # when agent receives an action, it setups a new batch
            self.env.set_actions(self.behaviour_names[0], np.array([[1]]))  # just give a random thing as an action, it doesn't matter here

            labels = DS.obs[-1][0][:self.num_labels_passed_by_unity].astype(int).reshape((-1, 2))
            camera_positions = DS.obs[-1][0][self.num_labels_passed_by_unity:].reshape((self.tot_num_matching, self.tot_num_frames_each_comparison, 3))
            self.images = [i[0] for i in DS.obs[:-1]]   # .reshape((self.tot_num_frames_each_iter, self.tot_num_frames_each_comparison, 64, 64, 3))

#################################~~~~~~DEBUG~~~~~~###############################################
            # _, self.ax = framework_utils.create_sphere()
            # vh1, vh2 = [], []
            # import matplotlib.pyplot as plt
            # plt.show()
            # import copy
            # def unity2python(v):
            #     v = copy.deepcopy(v)
            #     v.T[[1, 2]] = v.T[[2, 1]]
            #     return v
            # for idx, c in enumerate(camera_positions):
            #     if vh1:
            #         # [i.remove() for i in vh1]
            #         # [i.remove() for i in vh2]
            #         vh1 = []
            #         vh2 = []
            #     for i in range(len(camera_positions[0]) - 1):
            #         vh2.append(framework_utils.add_norm_vector(unity2python(c[i + 1]), 'r', ax=self.ax))
            #         vh1.append(framework_utils.add_norm_vector(unity2python(c[0]), 'k', ax=self.ax))

                    ## ali = framework_utils.align_vectors(c, t)
                    ## vh3 = framework_utils.add_norm_vector(ali, 'r', ax=self.ax )
#################################################################################################

            self.labels = labels
            self.camera_positions = camera_positions
            batch = self.images[:]
            yield batch


class UnitySamplerSequenceLearningRandom(UnitySamplerSequenceLearning):
    def __init__(self, group_classes: TrainComparisonType = TrainComparisonType.ALL, **kwargs):
        self.trial_type = TrialType.RND_TRIAL
        self.group_classes = group_classes
        super().__init__(**kwargs)

    def send_episode_info(self, idx):
        pass

    def send_additional_experiment_info(self):
        self.observation_channel.set_float_parameter("trainComparisonType",  float(self.group_classes.value))


class UnitySamplerSequenceLearningFixed(UnitySamplerSequenceLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.trial_type != TrialType.RND_TRIAL
        self.ranges = []
        self.organize_test_values()

    def organize_test_values(self):
        raise NotImplemented


class UnitySamplerSequenceLearningEdelmanFixed(UnitySamplerSequenceLearningFixed):
    def __init__(self, trial_type, **kwargs):
        self.trial_type = trial_type
        assert (self.trial_type == TrialType.DET_TRIAL_EDELMAN_ORTHO_HOR or
                self.trial_type == TrialType.DET_TRIAL_EDELMAN_ORTHO_VER or
                self.trial_type == TrialType.DET_TRIAL_EDELMAN_SAME_AXIS_HOR or
                self.trial_type == TrialType.DET_TRIAL_EDELMAN_SAME_AXIS_VER)
        super().__init__(**kwargs)

    def organize_test_values(self):
        obj1, obj2, deg, rot = np.meshgrid(np.arange(self.num_objects), np.arange(self.num_objects),  np.arange(0, 360 - 5, 10), np.arange(0, 360, 45))

        self.matrix_values = np.vstack((obj1.flatten(), obj2.flatten(), deg.flatten(), rot.flatten()))
        self.num_test = 50000  # this can be overwritten by the overall "-num_testing_iteration"
        chosen_trials = np.random.choice(range(self.matrix_values.shape[1]), np.min((self.num_test, self.matrix_values.shape[1])), replace=False)
        self.matrix_values = self.matrix_values[:, chosen_trials]
        self.episodes_per_epoch = self.matrix_values.shape[1]
        print(f"Testing [{self.episodes_per_epoch}] values")

    def send_episode_info(self, i):
        self.observation_channel.set_float_parameter("objT", float(self.matrix_values[0, i]))
        self.observation_channel.set_float_parameter("objC", float(self.matrix_values[1, i]))
        self.observation_channel.set_float_parameter("degree", float(self.matrix_values[2, i]))
        self.observation_channel.set_float_parameter("rotation", float(self.matrix_values[3, i]))  # of all the cameras around the object


class UnitySamplerStaticFixed(UnitySamplerSequenceLearningFixed):
    # Implement this when nSc, nSt, nFc, nFt = 1
    def __init__(self, **kwargs):
        self.trial_type = TrialType.DET_TRIAL_STATIC
        super().__init__(**kwargs)
        assert self.nSc == 1 and self.nSt == 1 and self.nFt == 1 and self.nFc == 1

    def send_episode_info(self, i):
        self.observation_channel.set_float_parameter("objT", float(self.matrix_values[0, i]))
        self.observation_channel.set_float_parameter("objC", float(self.matrix_values[1, i]))
        self.observation_channel.set_float_parameter("azimuthT", float(self.matrix_values[2, i]))
        self.observation_channel.set_float_parameter("azimuthC", float(self.matrix_values[3, i]))
        self.observation_channel.set_float_parameter("inclinationT", float(self.matrix_values[4, i]))
        self.observation_channel.set_float_parameter("inclinationC", float(self.matrix_values[5, i]))


class UnitySamplerStaticFixedGoker(UnitySamplerStaticFixed):
    def organize_test_values(self):
        ## All comparisons in groups: base group with all variants (A-> B C D E; F -> G H I J ..)
        objs = np.vstack([np.vstack((np.repeat(i*9, 8), np.arange(i*9 + 1, i*9 + 9))).T for i in np.arange(10)])

        ## add all comparison of different base objects
        x = np.meshgrid(np.arange(0, 9 * 9 + 9, 9), np.arange(0, 9 * 9 + 9, 9))
        objs = np.vstack((objs, np.vstack((x[0].flatten(), x[1].flatten())).T))

        objsStr, aziT, aziC, inclT, inclC = np.meshgrid([str(i) for i in objs], np.arange(0, 360, 45), np.arange(0, 360, 45), 45, 45, indexing='ij')
        objs = np.array([[int(i) for i in a[1:-1].split(" ") if i.isdigit()] for a in objsStr.flatten()]).T

        self.matrix_values = np.vstack((*objs, aziT.flatten(), aziC.flatten(), inclT.flatten(), inclC.flatten()))
        self.num_test = 100000  # this can be overwritten by the overall "-num_testing_iteration"
        # sort at the beginning will make it unshuffled
        chosen_trials = np.sort(np.random.choice(range(self.matrix_values.shape[1]), np.min((self.num_test, self.matrix_values.shape[1])), replace=False))
        self.matrix_values = self.matrix_values[:, chosen_trials]
        self.episodes_per_epoch = self.matrix_values.shape[1]
        print(f"Testing [{self.episodes_per_epoch}] values")



class UnitySamplerStaticFixedLeek(UnitySamplerStaticFixed):
    def organize_test_values(self):

        aziTC = np.arange(0, 360, 45)
        inclTC = np.arange(45, 135+45, 45)
        azi, incl = np.meshgrid(aziTC, inclTC)
        pos = np.vstack((azi.flatten(), incl.flatten())).T

        ## add all comparison of different base objects
        objT, objC, tmp = np.meshgrid(range(self.num_objects), range(self.num_objects), [str(i) for i in pos], indexing='ij')
        posTmp = np.array([[int(i) for i in a[1:-1].split(" ") if i.isdigit()] for a in tmp.flatten()]).T


        self.matrix_values = np.vstack((objT.flatten(), objC.flatten(), posTmp[0], posTmp[0],  posTmp[1], posTmp[1]))
        self.num_test = 100000
        # sort at the beginning will make it unshuffled
        chosen_trials = np.sort(np.random.choice(range(self.matrix_values.shape[1]), np.min((self.num_test, self.matrix_values.shape[1])), replace=False))
        self.matrix_values = self.matrix_values[:, chosen_trials]
        self.episodes_per_epoch = self.matrix_values.shape[1]
        print(f"Testing [{self.episodes_per_epoch}] values")




