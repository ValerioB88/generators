import os
from multiprocessing.dummy import freeze_support
import PIL.Image as Image
from visualization import vis_utils as vis
import pickle
import pathlib
import filecmp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from generate_dataset.dummy_faces_generator.utils import TranslationType, draw_face, get_range_translation, from_group_ID_to_features, BackGroundColorType, get_background_color
from torchvision import transforms
from compute_mean_std_dataset import compute_mean_and_std_from_dataset
import cv2


class DummyFaceRandomGenerator(Dataset):
    def __init__(self, translation_type, background_color_type: BackGroundColorType, middle_empty, grayscale, name='', random_gray_face=True, random_face_jitter=True):
        """
        @param random_face_jitter:
        @param translation_type: this can be either a TranslationType or a dictionary with the name of the classes and the respective translation type, e.g.
            translation_type = {0: TranslationType.WHOLE, 1: TranslationType.ONE_PIXEL}
        @param background_color_type:
        @param middle_empty:
        @param grayscale:
        @param name:
        """
        self.random_face_jitter = random_face_jitter
        self.random_gray_face = random_gray_face
        self.transform = None
        self.grayscale = grayscale
        self.translation_type = translation_type
        self.size_canvas = (224, 224, 1)
        self.length_face = 60
        self.width_face = self.length_face * 0.8125
        self.background_color_type = background_color_type
        self.middle_empty = middle_empty
        self.num_classes = 6
        self.name = name
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
            self.translations_range[idx] = get_range_translation(transl, self.length_face, self.size_canvas, self.width_face, self.middle_empty)

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

    def _get_translation(self, groupID):
        return self.random_translation(groupID)

    def __len__(self):
        return 300000  # np.iinfo(np.int64).max

    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label, random_face_jitter=True):
        return draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=False, random_face_jitter=random_face_jitter)

    def get_img_id(self, label):
        canvas = np.zeros(self.size_canvas, np.uint8) + get_background_color(self.background_color_type)
        is_smiling, eyes_type = from_group_ID_to_features(label)

        face_center = self._get_translation(label)

        canvas = self._call_draw_face(canvas, face_center, is_smiling, eyes_type, label, random_face_jitter=self.random_face_jitter)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas = Image.fromarray(canvas)
        if self.transform is not None:
            canvas = self.transform(canvas)
        else:
            canvas = np.array(canvas)

        return canvas, face_center

    def _get_id(self):
        return np.random.randint(self.num_classes)

    def __getitem__(self, idx):
        label = self._get_id()
        canvas, face_center = self.get_img_id(label)
        return canvas, label, face_center


class ScrambledDummyFaceRandomGenerator(DummyFaceRandomGenerator):
    def __init__(self,scrambling_list, translation_type: dict, background_color_type: TranslationType, middle_empty: BackGroundColorType, grayscale, name='', random_gray_face=True, random_face_jitter=True):
        self.scrambling_group = scrambling_list
        super(ScrambledDummyFaceRandomGenerator, self).__init__(translation_type, background_color_type, middle_empty, grayscale, name=name, random_gray_face=random_gray_face, random_face_jitter=random_face_jitter)


    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label, random_face_jitter=True):
        return draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=self.scrambling_group[label], random_gray_face_color=self.random_gray_face, random_face_jitter=random_face_jitter)


def do_stuff():
    # translation_list = {0: TranslationType.LEFT,
    #                     1: TranslationType.VERY_SMALL_AREA_RIGHT,
    #                     2: TranslationType.RIGHT,
    #                     3: TranslationType.WHOLE,
    #                     4: TranslationType.ONE_PIXEL,
    #                     5: TranslationType.LEFT}
    # # neptune.init('valeriobiscione/valerioERC')
    # # neptune.create_experiment(name='Test Generator', tags=['test'])
    # dataset = DummyFaceGenerator(translation_list, BackGroundColorType.RANDOM, middle_empty=True, grayscale=True, name='prova')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    # # utils.neptune_log_dataset_info(dataloader, logTxt='training')
    #
    # dataset = DummyFaceGenerator(TranslationType.LEFT, BackGroundColorType.RANDOM, middle_empty=True, grayscale=True, name='prova')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    #
    # iterator = iter(dataloader)
    # img, lab, _ = next(iterator)
    # vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)
    scrambling_list = {0: False,
                       1: False,
                       2: False,
                       3: True,
                       4: True,
                       5: True}

    dataset = ScrambledDummyFaceRandomGenerator(scrambling_list, TranslationType.LEFT, BackGroundColorType.RANDOM, grayscale=True, name='prova', middle_empty=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()