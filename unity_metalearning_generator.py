
from mlagents_envs.environment import UnityEnvironment, BehaviorName, DecisionSteps
import mlagents_envs.logging_util as log
from mlagents_envs.exception import UnityWorkerInUseException
import framework_utils
from generate_datasets.dataset_utils.compute_mean_std_dataset import compute_mean_and_std_from_dataset
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation
import copy
from torch.utils.data import Sampler
import warnings
import PIL.Image as Image
from generate_datasets.generators.input_image_generator import InputImagesGenerator
from typing import Tuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import os
import atexit
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid

# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self, uid) -> None:
        super().__init__(uuid.UUID(uid))
        self.num_objects = None


    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

class StringEnvParamsChannel(StringLogChannel):
    def on_message_received(self, msg: IncomingMessage) -> None:
        self.num_objects = int(str.split(msg.read_string(), '_')[0])


class StringDebugLogChannel(StringLogChannel):
    def on_message_received(self, msg: IncomingMessage) -> None:
        print(msg.read_string())


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

    def _finalize_get_item(self, canvas, class_name, more, idx):
        more['camera_positions'] = idx[2]
        return canvas, class_name, more

    def _get_label(self, idx):
        return str(idx[1])

    def _define_num_classes_(self):
        return self.sampler.num_objects  # ToDo: make this something we would get from a unity channel

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

class UnitySamplerMetaLearning(Sampler):
    """
    This sampler takes the samples from the UniyScene
    """
    warning_done = False

    def __init__(self, dataset, name_dataset_unity, unity_env_params, n, k, q, num_tasks=1, episodes_per_epoch=999999, channel=0, disjoint_train_and_test=True):
        # super(NShotTaskSampler, self).__init__(unity_env)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # may be useful later on

        # TODO: implement multi tasks
        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
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
            self.scene_path = f'./Unity-ML-Agents-Computer-Vision/Builds/Builds_{machine_name}/MetaLearning/'
            self.scene_folder = f'N{n}_K{k}_Q{q}_sc{dataset.size_canvas[0]}'
            self.scene = f'{self.scene_path}/{self.scene_folder}/scene.{ext}'
            if not os.path.exists(self.scene):
                assert False, f"Unity scene {self.scene} generator does not exist, create it in windows!"

        log.set_log_level('INFO')

        side_channel = EnvironmentParametersChannel()
        while True:
            try:
                self.env = UnityEnvironment(file_name=self.scene, side_channels=[side_channel], no_graphics=False, worker_id=channel, additional_args=['-name_dataset', name_dataset_unity])
                break
            except UnityWorkerInUseException as e:
                channel += 1

        self.env_params = unity_env_params(side_channel)
        self.env.reset()
        self.behaviour_names = list(self.env.behavior_specs.keys())
        self.tot_num_frame_each_iter = self.n * self.k + self.k * self.q

        # self.ax = framework_utils.create_sphere()

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):

            batch = []  # batch is [image, labels (real labels), support0/query1, positioncamera X, Y, Z]
            # ToDo: add each batch in a list and then return a stack of it
            for task in range(self.num_tasks):
                # Get random classes
                self.env.step()
                DS, TS = self.env.get_steps(self.behaviour_names[0])

                # when agent receives an action, it setups a new batch
                self.env.set_actions(self.behaviour_names[0], np.array([[1]]))  # just give a random thing as an action, it doesn't matter here
                labels = DS.obs[-1][0][:self.tot_num_frame_each_iter].astype(int)
                camera_positions = DS.obs[-1][0][self.tot_num_frame_each_iter:]
                images = DS.obs[:-1]
                #
                # s = camera_positions[0:3]
                # q = camera_positions[3 * self.n * self.k:(3 * self.n * self.k) + 3 ]
                #
                # rot = Rotation.align_vectors([[1, 0, 0]], [q])
                # distance = rot[0].apply(s)
                # # vh1 = framework_utils.add_norm_vector(s, 'r', ax=self.ax)
                # # vh2 = framework_utils.add_norm_vector(q, 'b', ax=self.ax )
                # vh2 = framework_utils.add_norm_vector(distance, 'k', ax=self.ax )

                # plt.show()
                batch = list((im[0], lab, cp) for im, lab, cp in zip(images, labels, camera_positions.reshape((-1, 3))))
            yield batch


class UnitySamplerSequenceLearning(Sampler):
    """
    This sampler takes the samples from the UniyScene
    """
    warning_done = False

    def __init__(self, dataset, name_dataset_unity, unity_env_params, nSc, nSt, nFc, nFt, k, size_canvas, episodes_per_epoch=999999, channel=0, disjoint_train_and_test=True):
        # super(NShotTaskSampler, self).__init__(unity_env)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # may be useful later on

        # TODO: Raise errors if initialise badly
        self.k = k
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
            self.scene_path = f'./Unity-ML-Agents-Computer-Vision/Builds/Builds_{machine_name}/SequenceLearning/'
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
        self.tot_num_frame_each_iter = self.nSt * self.nFt + self.k * (self.nSc * self.nFc)

        # _, self.ax = framework_utils.create_sphere()

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):

        vh1, vh2, vh3 = [], [], []
        batch = []  # batch is [image, labels (real labels), support0/query1, positioncamera X, Y, Z]
        self.num_tasks = 10
        for _ in range(self.episodes_per_epoch):
            images_list = []
            labels_list = []
            camera_positions_list = []
            for _ in range(self.num_tasks):
                # self.env_params.send_channel_info()
                # ToDo: add each batch in a list and then return a stack of it
                # Get random classes
                labels = np.empty((self.tot_num_frame_each_iter), np.int)
                labels[:] = 0
                camera_positions = np.empty((self.tot_num_frame_each_iter))
                camera_positions[:] = 0

                self.env.step()
                self.observation_channel.set_float_parameter("newLevel", float(0))

                DS, TS = self.env.get_steps(self.behaviour_names[0])

                # when agent receives an action, it setups a new batch
                self.env.set_actions(self.behaviour_names[0], np.array([[1]]))  # just give a random thing as an action, it doesn't matter here

                labels[:self.k + 1] = DS.obs[-1][0][:self.k + 1].astype(int) # this includes the correct choice (first value, [0, k]) and the "real" labels of the objects

                camera_positions = DS.obs[-1][0][1 + self.k:].reshape((-1, 3))
                images = DS.obs[:-1]


                labels_list.extend(labels)
                camera_positions_list.extend(camera_positions)
                images_list.extend(images)
    ##
                # if vh1 != []:
                #     [i.remove() for i in vh1]
                #     [i.remove() for i in vh2]
                #     vh1 = []
                #     vh2 = []
                #
                # def unity2python(v):
                #     v = copy.deepcopy(v)
                #     v.T[[1, 2]] = v.T[[2, 1]]
                #     return v
                #
                # camera_positions_candidates = camera_positions[:self.k * self.nFc * self.nSc]
                # camera_positions_trainings = camera_positions[self.k * self.nFc * self.nSc:]
                # #
                # tr = unity2python(camera_positions_trainings[0])
                # vh1.append(framework_utils.add_norm_vector(tr, ax=self.ax, col='g'))
                # for i in camera_positions_candidates:
                #     cand = unity2python(i)
                #     vh2.append(framework_utils.add_norm_vector(cand, ax=self.ax, col='k'))
                #     ali = framework_utils.align_vectors(cand, tr)
                #     vh3 = framework_utils.add_norm_vector(ali, ax=self.ax, col='r')

    ############################
            batch = list((im[0], lb, cp) for im, lb, cp in zip(images_list, labels_list, camera_positions_list))
            yield batch

# # ##
# #
# vh1.remove()
# vh2.remove()
# # vh3.remove()

# # # ##
#

##

