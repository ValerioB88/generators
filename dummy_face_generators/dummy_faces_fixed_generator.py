from multiprocessing.dummy import freeze_support
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.dummy_face_generators.utils_dummy_faces import draw_face
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.dummy_face_generators.dummy_faces_random_generator import DummyFaceRandomGenerator

class DummyFaceFixedGenerator(DummyFaceRandomGenerator):
    def __init__(self, grid_size, num_repetitions, translation_type, background_color_type: BackGroundColorType, middle_empty, grayscale, name_generator='', random_gray_face=True, random_face_jitter=True):
        self.grid_size = grid_size
        self.num_repetitions = num_repetitions
        self.mesh_trX, self.mesh_trY, self.mesh_class, self.mesh_rep = [], [], [], []
        self.current_index = 0
        super(DummyFaceFixedGenerator, self).__init__(translation_type, background_color_type, middle_empty, grayscale, name_generator=name_generator, random_gray_face=random_gray_face, random_face_jitter=random_face_jitter)

    def finalize(self):
        for groupID in range(self.num_classes):
            minX, maxX, minY, maxY = self.translations_range[groupID]
            x, y, c, r = np.meshgrid(np.arange(minX, maxX, self.grid_size),
                                     np.arange(minY, maxY, self.grid_size),
                                     groupID,
                                     np.arange(self.num_repetitions))
            self.mesh_trX.extend(x.flatten())
            self.mesh_trY.extend(y.flatten())
            self.mesh_class.extend(c.flatten())
            self.mesh_rep.extend(r.flatten())

        self.save_stats()
        self.current_index = 0

    def __len__(self):
        return len(self.mesh_trX)

    def _get_translation(self, class_ID):
        return self.mesh_trX[self.current_index], self.mesh_trY[self.current_index]

    def _get_id(self):
        return self.mesh_class[self.current_index]

    def _call_draw_face(self, canvas, face_center, is_smiling, eyes_type, label, random_face_jitter=True):
        img = draw_face(self.length_face, self.width_face, canvas, face_center, eyes_type=eyes_type, is_smiling=is_smiling, scrambled=False, random_gray_face_color=self.random_gray_face, random_face_jitter=random_face_jitter)
        self.current_index += 1
        return img

def do_stuff():
    translation_list = {0: TranslationType.LEFT,
                        1: TranslationType.VERY_SMALL_AREA_RIGHT,
                        2: TranslationType.RIGHT,
                        3: TranslationType.WHOLE,
                        4: TranslationType.ONE_PIXEL,
                        5: TranslationType.LEFT}

    dataset = DummyFaceFixedGenerator(grid_size=30, num_repetitions=1, translation_type=translation_list, background_color_type=BackGroundColorType.BLACK, middle_empty=True, grayscale=True, name_generator='prova')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        img, lab, _ = data
        # vis.imshow_batch(img, dataset.stats['mean'], dataset.stats['std'], title=lab)

if __name__ == '__main__':
    freeze_support()
    do_stuff()

# %%

