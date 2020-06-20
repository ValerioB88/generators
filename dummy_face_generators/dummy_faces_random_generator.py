from multiprocessing.dummy import freeze_support
import PIL.Image as Image
from visualization import vis_utils as vis
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.dummy_face_generators.utils_dummy_faces import draw_face, from_group_ID_to_features
from generate_datasets.generators.utils_generator import get_background_color, TranslationType, BackGroundColorType
import cv2
from generate_datasets.generators.translate_generator import TranslateGenerator
print('UPDATED SUBMODULE!')
class DummyFaceRandomGenerator(TranslateGenerator):
    def __init__(self, translation_type, middle_empty, background_color_type: BackGroundColorType, name_generator='', grayscale=True, random_gray_face=True, random_face_jitter=True, size_canvas=(224, 224), length_face=60):
        self.random_face_jitter = random_face_jitter
        self.random_gray_face = random_gray_face
        self.length_face = length_face
        self.width_face = self.length_face * 0.8125
        self.size_object = (self.width_face, self.length_face)

        super(DummyFaceRandomGenerator, self).__init__(translation_type, middle_empty, background_color_type, name_generator, size_canvas, self.size_object, grayscale)

    def define_num_classes(self):
        return 6

    def _get_translation(self, class_ID):
        return self.random_translation(class_ID)

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
    def __init__(self, scrambling_list, translation_type, middle_empty, background_color_type, name_generator='', grayscale = True, random_gray_face=True, random_face_jitter=True):
        self.scrambling_group = scrambling_list
        super(ScrambledDummyFaceRandomGenerator, self).__init__(translation_type, middle_empty, background_color_type, name_generator=name_generator, grayscale=grayscale, random_gray_face=random_gray_face, random_face_jitter=random_face_jitter)

    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label, random_face_jitter=True):
        return draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=self.scrambling_group[label], random_gray_face_color=self.random_gray_face, random_face_jitter=random_face_jitter)


def do_stuff():
    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.VERY_SMALL_AREA_RIGHT,
                        2: TranslationType.RIGHT,
                        3: TranslationType.WHOLE,
                        4: TranslationType.ONE_PIXEL,
                        5: TranslationType.LEFT}
    # neptune.init('valeriobiscione/valerioERC')
    # neptune.create_experiment(name='Test Generator', tags=['test'])
    dataset = DummyFaceRandomGenerator(translation_list, background_color_type=BackGroundColorType.RANDOM, middle_empty=True, grayscale=True, name_generator='prova')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
    # utils.neptune_log_dataset_info(dataloader, logTxt='training')

    dataset = DummyFaceRandomGenerator(TranslationType.LEFT,background_color_type=BackGroundColorType.RANDOM, middle_empty=True, grayscale=True, name_generator='prova')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)
    scrambling_list = {0: False,
                       1: False,
                       2: False,
                       3: True,
                       4: True,
                       5: True}

    dataset = ScrambledDummyFaceRandomGenerator(scrambling_list, background_color_type=BackGroundColorType.RANDOM, translation_type=TranslationType.LEFT, middle_empty=False, name_generator='prova', grayscale=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterator = iter(dataloader)
    img, lab, _ = next(iterator)
    vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)


if __name__ == '__main__':
    freeze_support()
    do_stuff()