from multiprocessing.dummy import freeze_support
import PIL.Image as Image

import framework_utils
from visualization import vis_utils as vis
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.dummy_face_generators.utils_dummy_faces import draw_face, from_group_ID_to_features
from generate_datasets.generators.utils_generator import get_background_color, TranslationType, BackGroundColorType
import cv2
from generate_datasets.generators.translate_generator import TranslateGenerator

class DummyFaceRandomGenerator(TranslateGenerator):
    def __init__(self, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=False, random_gray_face=True, random_face_jitter=True, size_canvas=(224, 224), length_face=60):
        """

        @param translation_type: when a dict, translation type key refers to the face_id. You don't need to use all implemented
        type of faces.For example, you can only use the id 0 and 5. The attribute self.num_classes will be equal to 2.
        The returned labels will be contiguous: 0, 1, regardless of the face id used.
        """
        self.random_face_jitter = random_face_jitter
        self.random_gray_face = random_gray_face
        self.length_face = length_face
        self.width_face = int(self.length_face * 0.8125)
        self.size_object = (self.width_face, self.length_face)

        super().__init__(translation_type, middle_empty, background_color_type, name_generator, grayscale, size_canvas, self.size_object)

    def _define_num_classes_(self):
        return 6 if isinstance(self.translation_type, TranslationType) else len(self.translation_type)

    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label):
        return draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=False, random_face_jitter=self.random_face_jitter)

    def get_img_id(self, face_id, idx):
        canvas = np.zeros(self.size_canvas, np.uint8) + get_background_color(self.background_color_type)
        is_smiling, eyes_type = from_group_ID_to_features(face_id)
        face_center = self._get_translation_(face_id, canvas, idx)
        canvas = self._call_draw_face(canvas, face_center, is_smiling, eyes_type, face_id)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas = Image.fromarray(canvas)

        return canvas, face_center

    def _get_label_(self, item):
        return int(np.random.choice(list(self.translations_range.keys())))

    def _get_my_item_(self, idx, label):
        canvas, face_center = self.get_img_id(label, idx)
        label = list(self.translations_range.keys()).index(label)
        return canvas, label, {'center': face_center}


class ScrambledDummyFaceRandomGenerator(DummyFaceRandomGenerator):
    def __init__(self, scrambling_list, translation_type, middle_empty, background_color_type, name_generator='', grayscale=False, size_canvas=(224, 224), size_object=(50, 50)):
        self.scrambling_group = scrambling_list
        super(ScrambledDummyFaceRandomGenerator, self).__init__(translation_type, middle_empty, background_color_type, name_generator=name_generator, grayscale=grayscale)
        assert self.scrambling_group.keys() == self.translations_range.keys(), 'Scrambling list and translation list should contain the same groups'

    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label):
        return draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=self.scrambling_group[label], random_gray_face_color=self.random_gray_face, random_face_jitter=self.random_face_jitter)


def do_stuff():
    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.VERY_SMALL_AREA_RIGHT,
                        2: TranslationType.RIGHT,
                        3: TranslationType.WHOLE,
                        4: TranslationType.ONE_PIXEL,
                        5: TranslationType.LEFT}
    # neptune.init('valeriobiscione/valerioERC')
    # neptune.create_experiment(name='Test Generator', tags=['test'])
    dataset = DummyFaceRandomGenerator(translation_list, middle_empty=True, background_color_type=BackGroundColorType.RANDOM, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    # utils.neptune_log_dataset_info(dataloader, logTxt='training')

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)


    dataset = DummyFaceRandomGenerator(TranslationType.LEFT, middle_empty=True, background_color_type=BackGroundColorType.RANDOM, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

    translation_list = {0: TranslationType.LEFT,
                        4: TranslationType.ONE_PIXEL,
                        5: TranslationType.LEFT}
    dataset = DummyFaceRandomGenerator(translation_list, middle_empty=True, background_color_type=BackGroundColorType.BLACK, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

    dataset = DummyFaceRandomGenerator((10, 51), middle_empty=True, background_color_type=BackGroundColorType.BLACK, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

    scrambling_list = {0: False,
                       5: True}
    translation_list = {0: TranslationType.LEFT,
                        5: TranslationType.LEFT}

    dataset = ScrambledDummyFaceRandomGenerator(scrambling_list, translation_type=translation_list, middle_empty=False, background_color_type=BackGroundColorType.RANDOM, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

    translation_list = {0: (10, 20),
                        4: (100, 20),
                        5: (150, 150)}
    dataset = DummyFaceRandomGenerator(translation_list, middle_empty=True, background_color_type=BackGroundColorType.BLACK, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

    translation_list = {0: (10, 20, 11, 21),
                        4: (0, 120, 200, 201),
                        5: (150, 150)}
    dataset = DummyFaceRandomGenerator(translation_list, middle_empty=True, background_color_type=BackGroundColorType.BLACK, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    framework_utils.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title_lab=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()